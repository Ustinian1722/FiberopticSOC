from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from run_representation_conditioning_diagnostic import PairTCN
from run_sequence_representation_benchmark import (
    PROFILES,
    WindowDataset,
    count_params,
    load_sources,
    metric_dict,
    normalize_sources,
    predict_model,
    seed_everything,
    train_model,
    train_normalizer,
)

MODEL_ORDER = ("IUW-TCN", "MC-TCN", "LR-MC-TCN")


class CausalDepthwiseConv1d(nn.Module):
    def __init__(self, channels: int, kernel_size: int):
        super().__init__()
        self.left = kernel_size - 1
        self.conv = nn.Conv1d(channels, channels, kernel_size, groups=channels, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(F.pad(x, (self.left, 0)))


class ModernCausalBlock(nn.Module):
    """ModernTCN-inspired separation of temporal, feature and variate mixing."""

    def __init__(self, n_vars: int, dim: int, kernel_size: int, expansion: int = 2):
        super().__init__()
        c = n_vars * dim
        self.n_vars = n_vars
        self.dim = dim

        self.temporal = CausalDepthwiseConv1d(c, kernel_size)
        self.temporal_norm = nn.BatchNorm1d(c)

        self.feature_ffn = nn.Sequential(
            nn.Conv1d(c, c * expansion, 1, groups=n_vars),
            nn.GELU(),
            nn.Conv1d(c * expansion, c, 1, groups=n_vars),
        )
        self.feature_norm = nn.BatchNorm1d(c)

        # After permutation channels are ordered as [feature, variable], so groups=dim
        # mixes physical variables independently for every latent feature.
        cv = dim * n_vars
        self.variate_ffn = nn.Sequential(
            nn.Conv1d(cv, cv * expansion, 1, groups=dim),
            nn.GELU(),
            nn.Conv1d(cv * expansion, cv, 1, groups=dim),
        )
        self.variate_norm = nn.BatchNorm1d(cv)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        # z: [B,M,D,T]
        b, m, d, t = z.shape
        flat = z.reshape(b, m * d, t)
        flat = flat + F.gelu(self.temporal_norm(self.temporal(flat)))
        flat = flat + self.feature_norm(self.feature_ffn(flat))
        z = flat.reshape(b, m, d, t)

        v = z.permute(0, 2, 1, 3).contiguous().reshape(b, d * m, t)
        v = v + self.variate_norm(self.variate_ffn(v))
        return v.reshape(b, d, m, t).permute(0, 2, 1, 3).contiguous()


class ModernCausalEncoder(nn.Module):
    def __init__(self, n_vars: int = 4, dim: int = 12, kernels: tuple[int, ...] = (9, 17, 33)):
        super().__init__()
        self.n_vars = n_vars
        self.dim = dim
        self.embed = nn.Conv1d(n_vars, n_vars * dim, 1, groups=n_vars)
        self.blocks = nn.ModuleList([ModernCausalBlock(n_vars, dim, k) for k in kernels])
        self.out_norm = nn.LayerNorm(n_vars * dim)

    @property
    def out_dim(self) -> int:
        return self.n_vars * self.dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x [B,T,M]
        b, t, m = x.shape
        if m != self.n_vars:
            raise ValueError((m, self.n_vars))
        z = self.embed(x.transpose(1, 2)).reshape(b, self.n_vars, self.dim, t)
        for block in self.blocks:
            z = block(z)
        h = z[:, :, :, -1].reshape(b, -1)
        return self.out_norm(h)


class ModernCausalTCN(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = ModernCausalEncoder()
        h = self.encoder.out_dim
        self.head = nn.Sequential(nn.Linear(h, 32), nn.GELU(), nn.Linear(32, 1))

    def forward(self, x: torch.Tensor):
        z = x[:, :, (0, 1, 2, 3)]
        h = self.encoder(z)
        return self.head(h).squeeze(-1), None


class LevelResidualModernCausalTCN(nn.Module):
    """Local-stationary dynamic path + preserved level/trend path."""

    def __init__(self):
        super().__init__()
        self.dynamic = ModernCausalEncoder()
        self.level = nn.Sequential(nn.Linear(16, 32), nn.GELU(), nn.Linear(32, 24), nn.GELU())
        self.head = nn.Sequential(
            nn.Linear(self.dynamic.out_dim + 24, 40), nn.GELU(), nn.Linear(40, 1)
        )

    def forward(self, x: torch.Tensor):
        z = x[:, :, (0, 1, 2, 3)]
        mean = z.mean(dim=1, keepdim=True)
        std = z.std(dim=1, keepdim=True, correction=0).clamp_min(1e-4)
        residual = (z - mean) / std
        h_dyn = self.dynamic(residual)

        last = z[:, -1, :]
        first = z[:, 0, :]
        level = torch.cat([last, mean.squeeze(1), std.squeeze(1), last - first], dim=-1)
        h_level = self.level(level)
        pred = self.head(torch.cat([h_dyn, h_level], dim=-1)).squeeze(-1)
        return pred, None


def build_model(name: str) -> nn.Module:
    if name == "IUW-TCN":
        return PairTCN((2, 3), None)
    if name == "MC-TCN":
        return ModernCausalTCN()
    if name == "LR-MC-TCN":
        return LevelResidualModernCausalTCN()
    raise ValueError(name)


def run_fold(args, sources: list[dict], device: torch.device) -> pd.DataFrame:
    held_out = args.held_out_profile
    train_raw = [s for s in sources if s["rate"] == "1C" and s["profile"] != held_out]
    val_raw = [s for s in sources if s["rate"] == "1C" and s["profile"] == held_out]
    if len(train_raw) != 5 or len(val_raw) != 1:
        raise RuntimeError(f"Bad source split: train={len(train_raw)} val={len(val_raw)}")
    if any(s["rate"] != "1C" for s in train_raw + val_raw):
        raise RuntimeError("Non-1C source entered source-only architecture screen")

    mean, std = train_normalizer(train_raw)
    train = normalize_sources(train_raw, mean, std)
    val = normalize_sources(val_raw, mean, std)
    train_ds = WindowDataset(train, args.window, args.train_stride)
    val_ds = WindowDataset(val, args.window, args.val_stride)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size * 2, shuffle=False, num_workers=0)

    rows = []
    for name in MODEL_ORDER:
        seed_everything(args.seed)
        model = build_model(name)
        params = count_params(model)
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
        print(f"\n=== topconf source held_out={held_out} model={name} params={params} ===")
        train_model(model, train_loader, device, args.epochs, args.lr)
        y, pred, _, aux = predict_model(model, val_loader, device)
        if aux is not None:
            raise RuntimeError(f"Unexpected aux output: {name}")
        rows.append(
            {
                "protocol": "topconf_extension_source_only_1C_LOPO",
                "held_out_profile": held_out,
                "rate": "1C",
                "model": name,
                "seed": args.seed,
                "epochs": args.epochs,
                "window": args.window,
                "train_stride": args.train_stride,
                "val_stride": args.val_stride,
                "params": params,
                "n_train": len(train_ds),
                "n_val": len(y),
                "source_only": True,
                "two_c_metrics_used": False,
                **metric_dict(y, pred),
            }
        )
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return pd.DataFrame(rows)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=Path, default=Path("data/extracted/SiC-18"))
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--held-out-profile", choices=PROFILES, required=True)
    p.add_argument("--window", type=int, default=64)
    p.add_argument("--train-stride", type=int, default=4)
    p.add_argument("--val-stride", type=int, default=1)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sources = load_sources(args.data)
    result = run_fold(args, sources, device)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.out_dir / f"topconf_source_{args.held_out_profile}.csv", index=False)
    print(result.sort_values("MAE").to_string(index=False))


if __name__ == "__main__":
    main()
