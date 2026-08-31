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
    CausalConv1d,
    PROFILES,
    TCNEncoder,
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

MODEL_ORDER = ("IUW-TCN", "MD-ResTCN")


class FlexibleTCNBlock(nn.Module):
    def __init__(self, channels: int, kernel_size: int, dilation: int):
        super().__init__()
        self.conv1 = CausalConv1d(channels, channels, kernel_size, dilation=dilation)
        self.conv2 = CausalConv1d(channels, channels, kernel_size, dilation=dilation)
        self.norm1 = nn.GroupNorm(1, channels)
        self.norm2 = nn.GroupNorm(1, channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = F.gelu(self.norm1(self.conv1(x)))
        z = self.norm2(self.conv2(z))
        return F.gelu(x + z)


class FlexibleTCNEncoder(nn.Module):
    def __init__(self, in_dim: int, hidden: int, kernel_size: int, dilations: tuple[int, ...]):
        super().__init__()
        self.proj = nn.Conv1d(in_dim, hidden, 1)
        self.blocks = nn.Sequential(
            *[FlexibleTCNBlock(hidden, kernel_size, d) for d in dilations]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.proj(x.transpose(1, 2))
        z = self.blocks(z)
        return z[:, :, -1]


class ModalityDelayResidualTCN(nn.Module):
    """Strong joint TCN + zero-initialized modality-specific delay correction.

    Base path: exact raw V/I/W1/W2 TCN family used by IUW-TCN.
    Electrical correction path: short RF (k=3,d=1,2 -> RF~13).
    Optical correction path: long RF (k=5,d=1,2,4 -> RF~57).
    """

    def __init__(self, base_hidden: int = 24, branch_hidden: int = 12):
        super().__init__()
        self.base_encoder = TCNEncoder(4, base_hidden)
        self.base_head = nn.Sequential(
            nn.Linear(base_hidden, base_hidden), nn.GELU(), nn.Linear(base_hidden, 1)
        )

        self.electrical = FlexibleTCNEncoder(2, branch_hidden, 3, (1, 2))
        self.optical = FlexibleTCNEncoder(2, branch_hidden, 5, (1, 2, 4))
        fusion_dim = branch_hidden * 4
        self.correction = nn.Sequential(
            nn.Linear(fusion_dim, 24),
            nn.GELU(),
            nn.Linear(24, 1),
        )
        # Candidate starts as the strong base path and must learn evidence-supported corrections.
        nn.init.zeros_(self.correction[-1].weight)
        nn.init.zeros_(self.correction[-1].bias)

    def forward(self, x: torch.Tensor):
        electrical = x[:, :, (0, 1)]
        optical = x[:, :, (2, 3)]
        joint = torch.cat([electrical, optical], dim=-1)

        h_base = self.base_encoder(joint)
        base = self.base_head(h_base).squeeze(-1)

        e = self.electrical(electrical)
        o = self.optical(optical)
        interaction = torch.cat([e, o, e * o, torch.abs(e - o)], dim=-1)
        correction = self.correction(interaction).squeeze(-1)
        return base + correction, None


def build_model(name: str) -> nn.Module:
    if name == "IUW-TCN":
        return PairTCN((2, 3), None)
    if name == "MD-ResTCN":
        return ModalityDelayResidualTCN()
    raise ValueError(name)


def run_fold(args, sources: list[dict], device: torch.device) -> pd.DataFrame:
    held_out = args.held_out_profile
    train_raw = [s for s in sources if s["rate"] == "1C" and s["profile"] != held_out]
    val_raw = [s for s in sources if s["rate"] == "1C" and s["profile"] == held_out]
    if len(train_raw) != 5 or len(val_raw) != 1:
        raise RuntimeError(f"Bad source split for {held_out}: train={len(train_raw)} val={len(val_raw)}")
    if any(s["rate"] != "1C" for s in train_raw + val_raw):
        raise RuntimeError("Non-1C data entered source-only screen")

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
        print(f"\n=== modality-delay source held_out={held_out} model={name} params={params} ===")
        train_model(model, train_loader, device, args.epochs, args.lr)
        y, pred, _, aux = predict_model(model, val_loader, device)
        if aux is not None:
            raise RuntimeError(f"Unexpected aux output from {name}")
        rows.append(
            {
                "protocol": "modality_delay_source_only_1C_LOPO",
                "rate": "1C",
                "held_out_profile": held_out,
                "model": name,
                "seed": args.seed,
                "epochs": args.epochs,
                "window": args.window,
                "train_stride": args.train_stride,
                "val_stride": args.val_stride,
                "params": params,
                "n_train": len(train_ds),
                "n_val": len(y),
                "electrical_receptive_field": 13 if name == "MD-ResTCN" else 29,
                "optical_receptive_field": 57 if name == "MD-ResTCN" else 29,
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
    result.to_csv(args.out_dir / f"modality_delay_{args.held_out_profile}.csv", index=False)
    print(result.sort_values("MAE").to_string(index=False))


if __name__ == "__main__":
    main()
