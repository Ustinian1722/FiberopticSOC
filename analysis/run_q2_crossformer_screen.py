from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from run_representation_conditioning_diagnostic import PairTCN
from run_sequence_representation_benchmark import (
    PROFILES,
    CausalConv1d,
    TCNBlock,
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

MODEL_ORDER = (
    "IUW-TCN",
    "CGA-Matched",
    "VIW-Transformer",
    "DualTCN-Transformer",
    "EO-CrossFormer",
    "EO-CrossFormer-TF",
)


class SinusoidalPosition(nn.Module):
    def __init__(self, d_model: int, max_len: int = 512):
        super().__init__()
        pos = torch.arange(max_len, dtype=torch.float32)[:, None]
        div = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model, dtype=torch.float32)
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[: x.shape[1]][None, :, :].to(dtype=x.dtype, device=x.device)


def causal_mask(length: int, device: torch.device) -> torch.Tensor:
    return torch.triu(torch.ones(length, length, dtype=torch.bool, device=device), diagonal=1)


class CausalTCNSequenceEncoder(nn.Module):
    def __init__(self, in_dim: int, hidden: int = 24):
        super().__init__()
        self.proj = nn.Conv1d(in_dim, hidden, kernel_size=1)
        self.blocks = nn.Sequential(
            TCNBlock(hidden, 1),
            TCNBlock(hidden, 2),
            TCNBlock(hidden, 4),
        )
        self.norm = nn.LayerNorm(hidden)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.proj(x.transpose(1, 2))
        z = self.blocks(z).transpose(1, 2)
        return self.norm(z)


class CausalTransformer(nn.Module):
    def __init__(self, d_model: int = 48, nhead: int = 4, depth: int = 2, ff: int = 96):
        super().__init__()
        self.pos = SinusoidalPosition(d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=ff,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=depth, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pos(x)
        mask = causal_mask(x.shape[1], x.device)
        return self.norm(self.encoder(x, mask=mask))


class CGAMatched(nn.Module):
    """CNN-GRU-temporal-attention comparator under the repository's strict protocol.

    This is intentionally described as a matched CGA-style baseline rather than an exact
    layer-for-layer reproduction of the companion paper, whose complete implementation is
    not public in this repository.
    """

    def __init__(self, hidden: int = 48):
        super().__init__()
        self.conv1 = CausalConv1d(4, 32, kernel_size=5)
        self.conv2 = CausalConv1d(32, 32, kernel_size=3)
        self.norm1 = nn.GroupNorm(1, 32)
        self.norm2 = nn.GroupNorm(1, 32)
        self.gru = nn.GRU(input_size=32, hidden_size=hidden, num_layers=1, batch_first=True)
        self.attn = nn.Sequential(nn.Linear(hidden, hidden // 2), nn.Tanh(), nn.Linear(hidden // 2, 1))
        self.head = nn.Sequential(nn.Linear(hidden, hidden), nn.GELU(), nn.Linear(hidden, 1))

    def forward(self, x: torch.Tensor):
        z = x[:, :, (0, 1, 2, 3)].transpose(1, 2)
        z = F.gelu(self.norm1(self.conv1(z)))
        z = F.gelu(self.norm2(self.conv2(z))).transpose(1, 2)
        h, _ = self.gru(z)
        a = torch.softmax(self.attn(h).squeeze(-1), dim=1)
        context = torch.sum(h * a[:, :, None], dim=1)
        return self.head(context).squeeze(-1), None


class VIWTransformer(nn.Module):
    def __init__(self, d_model: int = 48):
        super().__init__()
        self.input = nn.Sequential(nn.Linear(4, d_model), nn.GELU(), nn.LayerNorm(d_model))
        self.backbone = CausalTransformer(d_model=d_model, nhead=4, depth=2, ff=96)
        self.head = nn.Sequential(nn.Linear(d_model, d_model), nn.GELU(), nn.Linear(d_model, 1))

    def forward(self, x: torch.Tensor):
        z = self.input(x[:, :, (0, 1, 2, 3)])
        h = self.backbone(z)[:, -1]
        return self.head(h).squeeze(-1), None


class DualTCNTransformer(nn.Module):
    def __init__(self, optical_pair: tuple[int, int] = (2, 3), hidden: int = 24, d_model: int = 48):
        super().__init__()
        self.optical_pair = optical_pair
        self.electrical = CausalTCNSequenceEncoder(2, hidden)
        self.optical = CausalTCNSequenceEncoder(2, hidden)
        self.fusion = nn.Sequential(nn.Linear(2 * hidden, d_model), nn.GELU(), nn.LayerNorm(d_model))
        self.backbone = CausalTransformer(d_model=d_model, nhead=4, depth=2, ff=96)
        self.head = nn.Sequential(nn.Linear(d_model, d_model), nn.GELU(), nn.Linear(d_model, 1))

    def forward(self, x: torch.Tensor):
        e = self.electrical(x[:, :, (0, 1)])
        o = self.optical(x[:, :, self.optical_pair])
        z = self.fusion(torch.cat([e, o], dim=-1))
        h = self.backbone(z)[:, -1]
        return self.head(h).squeeze(-1), None


class CausalCrossModalBlock(nn.Module):
    def __init__(self, hidden: int = 24, nhead: int = 4):
        super().__init__()
        self.e_from_o = nn.MultiheadAttention(hidden, nhead, dropout=0.0, batch_first=True)
        self.o_from_e = nn.MultiheadAttention(hidden, nhead, dropout=0.0, batch_first=True)
        self.enorm1 = nn.LayerNorm(hidden)
        self.onorm1 = nn.LayerNorm(hidden)
        self.enorm2 = nn.LayerNorm(hidden)
        self.onorm2 = nn.LayerNorm(hidden)
        self.eff = nn.Sequential(nn.Linear(hidden, 2 * hidden), nn.GELU(), nn.Linear(2 * hidden, hidden))
        self.off = nn.Sequential(nn.Linear(hidden, 2 * hidden), nn.GELU(), nn.Linear(2 * hidden, hidden))

    def forward(self, e: torch.Tensor, o: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mask = causal_mask(e.shape[1], e.device)
        de, _ = self.e_from_o(self.enorm1(e), self.onorm1(o), self.onorm1(o), attn_mask=mask, need_weights=False)
        do, _ = self.o_from_e(self.onorm1(o), self.enorm1(e), self.enorm1(e), attn_mask=mask, need_weights=False)
        e = e + de
        o = o + do
        e = e + self.eff(self.enorm2(e))
        o = o + self.off(self.onorm2(o))
        return e, o


class EOCrossFormer(nn.Module):
    def __init__(self, optical_pair: tuple[int, int] = (2, 3), hidden: int = 24, d_model: int = 48):
        super().__init__()
        self.optical_pair = optical_pair
        self.electrical = CausalTCNSequenceEncoder(2, hidden)
        self.optical = CausalTCNSequenceEncoder(2, hidden)
        self.cross = CausalCrossModalBlock(hidden=hidden, nhead=4)
        # Explicit interaction terms make fusion richer than a scalar reliability gate.
        self.fusion = nn.Sequential(
            nn.Linear(4 * hidden, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model),
        )
        self.backbone = CausalTransformer(d_model=d_model, nhead=4, depth=2, ff=96)
        self.head = nn.Sequential(nn.Linear(d_model, d_model), nn.GELU(), nn.Linear(d_model, 1))

    def forward(self, x: torch.Tensor):
        e = self.electrical(x[:, :, (0, 1)])
        o = self.optical(x[:, :, self.optical_pair])
        e, o = self.cross(e, o)
        inter = torch.cat([e, o, e * o, torch.abs(e - o)], dim=-1)
        z = self.fusion(inter)
        h = self.backbone(z)[:, -1]
        return self.head(h).squeeze(-1), None


def build_model(name: str) -> nn.Module:
    if name == "IUW-TCN":
        return PairTCN((2, 3), None)
    if name == "CGA-Matched":
        return CGAMatched()
    if name == "VIW-Transformer":
        return VIWTransformer()
    if name == "DualTCN-Transformer":
        return DualTCNTransformer((2, 3))
    if name == "EO-CrossFormer":
        return EOCrossFormer((2, 3))
    if name == "EO-CrossFormer-TF":
        return EOCrossFormer((4, 5))
    raise ValueError(name)


def representation(name: str) -> str:
    if name == "EO-CrossFormer-TF":
        return "physics_decoupled_TF"
    if name == "CGA-Matched":
        return "raw_W_CGA_style"
    return "raw_W"


def run_profile(args, all_sources: list[dict], device: torch.device, held_out: str) -> pd.DataFrame:
    train_raw = [s for s in all_sources if s["rate"] == "1C" and s["profile"] != held_out]
    test_raw = [s for s in all_sources if s["rate"] == "2C" and s["profile"] == held_out]
    if len(train_raw) != 5 or len(test_raw) != 1:
        raise RuntimeError(f"Bad development split for {held_out}")

    mean, std = train_normalizer(train_raw)
    train = normalize_sources(train_raw, mean, std)
    test = normalize_sources(test_raw, mean, std)
    train_ds = WindowDataset(train, args.window, args.train_stride)
    test_ds = WindowDataset(test, args.window, args.test_stride)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size * 2, shuffle=False, num_workers=0)

    rows = []
    for name in MODEL_ORDER:
        seed_everything(args.seed)
        model = build_model(name)
        params = count_params(model)
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
        print(f"\n=== held_out={held_out} model={name} params={params} train={len(train_ds)} test={len(test_ds)} ===")
        train_model(model, train_loader, device, args.epochs, args.lr)
        y, pred, _, aux = predict_model(model, test_loader, device)
        if aux is not None:
            raise RuntimeError(f"Unexpected auxiliary output from {name}")
        rows.append({
            "protocol": "final_development_1C_to_2C_unseen_profile",
            "train_rate": "1C",
            "test_rate": "2C",
            "held_out_profile": held_out,
            "model": name,
            "representation": representation(name),
            "seed": args.seed,
            "epochs": args.epochs,
            "window": args.window,
            "train_stride": args.train_stride,
            "test_stride": args.test_stride,
            "params": params,
            "n_train": len(train_ds),
            "n_test": len(y),
            **metric_dict(y, pred),
        })
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return pd.DataFrame(rows)


def main() -> None:
    p = argparse.ArgumentParser(description="Final literature-aligned architecture-family development screen")
    p.add_argument("--data", type=Path, default=Path("data/extracted/SiC-18"))
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--held-out-profile", choices=PROFILES, required=True)
    p.add_argument("--window", type=int, default=64)
    p.add_argument("--train-stride", type=int, default=4)
    p.add_argument("--test-stride", type=int, default=1)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}")
    sources = load_sources(args.data)
    result = run_profile(args, sources, device, args.held_out_profile)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    path = args.out_dir / f"profile_metrics_{args.held_out_profile}.csv"
    result.to_csv(path, index=False)
    print("\n=== profile result ===")
    print(result.sort_values("MAE").to_string(index=False))


if __name__ == "__main__":
    main()
