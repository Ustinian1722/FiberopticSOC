from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from run_q2_modality_delay_tcn_source_screen import FlexibleTCNEncoder
from run_representation_conditioning_diagnostic import PairTCN
from run_sequence_representation_benchmark import (
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

MODEL_ORDER = ("IUW-TCN", "DMD-ResTCN")


class DynamicMultiDelayResidualTCN(nn.Module):
    """Strong joint TCN + sample-adaptive short/mid/long optical delay correction."""

    def __init__(self, base_hidden: int = 24, branch_hidden: int = 12):
        super().__init__()
        # Exact strong joint path.
        self.base_encoder = TCNEncoder(4, base_hidden)
        self.base_head = nn.Sequential(
            nn.Linear(base_hidden, base_hidden), nn.GELU(), nn.Linear(base_hidden, 1)
        )

        # Fast electrical context.
        self.electrical = FlexibleTCNEncoder(2, branch_hidden, 3, (1, 2))  # RF ~13

        # Three optical temporal scales, fixed before screening.
        self.opt_short = FlexibleTCNEncoder(2, branch_hidden, 3, (1, 2))       # RF ~13
        self.opt_medium = FlexibleTCNEncoder(2, branch_hidden, 3, (1, 2, 4))   # RF ~29
        self.opt_long = FlexibleTCNEncoder(2, branch_hidden, 5, (1, 2, 4))     # RF ~57

        selector_in = branch_hidden * 4
        self.selector = nn.Sequential(
            nn.Linear(selector_in, 24),
            nn.GELU(),
            nn.Linear(24, 3),
        )

        interaction_dim = branch_hidden * 4
        self.correction = nn.Sequential(
            nn.Linear(interaction_dim, 24),
            nn.GELU(),
            nn.Linear(24, 1),
        )
        # Start exactly as a base-only predictor; delay module must earn its correction.
        nn.init.zeros_(self.correction[-1].weight)
        nn.init.zeros_(self.correction[-1].bias)

    def forward(self, x: torch.Tensor):
        electrical = x[:, :, (0, 1)]
        optical = x[:, :, (2, 3)]
        joint = torch.cat([electrical, optical], dim=-1)

        h_base = self.base_encoder(joint)
        base = self.base_head(h_base).squeeze(-1)

        e = self.electrical(electrical)
        o_short = self.opt_short(optical)
        o_medium = self.opt_medium(optical)
        o_long = self.opt_long(optical)

        selector_input = torch.cat([e, o_short, o_medium, o_long], dim=-1)
        weights = torch.softmax(self.selector(selector_input), dim=-1)
        stacked = torch.stack([o_short, o_medium, o_long], dim=1)  # [B,3,H]
        o = torch.sum(stacked * weights.unsqueeze(-1), dim=1)

        interaction = torch.cat([e, o, e * o, torch.abs(e - o)], dim=-1)
        correction = self.correction(interaction).squeeze(-1)
        return base + correction, weights


def build_model(name: str) -> nn.Module:
    if name == "IUW-TCN":
        return PairTCN((2, 3), None)
    if name == "DMD-ResTCN":
        return DynamicMultiDelayResidualTCN()
    raise ValueError(name)


def run_fold(args, sources: list[dict], device: torch.device) -> tuple[pd.DataFrame, pd.DataFrame]:
    held_out = args.held_out_profile
    train_raw = [s for s in sources if s["rate"] == "1C" and s["profile"] != held_out]
    val_raw = [s for s in sources if s["rate"] == "1C" and s["profile"] == held_out]
    if len(train_raw) != 5 or len(val_raw) != 1:
        raise RuntimeError(f"Bad source split for {held_out}: train={len(train_raw)} val={len(val_raw)}")
    if any(s["rate"] != "1C" for s in train_raw + val_raw):
        raise RuntimeError("Non-1C data entered source-only DMD screen")

    mean, std = train_normalizer(train_raw)
    train = normalize_sources(train_raw, mean, std)
    val = normalize_sources(val_raw, mean, std)
    train_ds = WindowDataset(train, args.window, args.train_stride)
    val_ds = WindowDataset(val, args.window, args.val_stride)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size * 2, shuffle=False, num_workers=0)

    rows = []
    gate_rows = []
    for name in MODEL_ORDER:
        seed_everything(args.seed)
        model = build_model(name)
        params = count_params(model)
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
        print(f"\n=== dynamic multi-delay source held_out={held_out} model={name} params={params} ===")
        train_model(model, train_loader, device, args.epochs, args.lr)
        y, pred, _, aux = predict_model(model, val_loader, device)
        rows.append(
            {
                "protocol": "dynamic_multi_delay_source_only_1C_LOPO",
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
                "source_only": True,
                "two_c_metrics_used": False,
                **metric_dict(y, pred),
            }
        )
        if name == "DMD-ResTCN":
            if aux is None or aux.ndim != 2 or aux.shape[1] != 3:
                raise RuntimeError(f"Expected [N,3] selector weights, got {None if aux is None else aux.shape}")
            if not np.allclose(aux.sum(axis=1), 1.0, atol=1e-5):
                raise RuntimeError("Selector weights do not sum to one")
            labels = ("short_RF13", "medium_RF29", "long_RF57")
            for k, label in enumerate(labels):
                gate_rows.append(
                    {
                        "held_out_profile": held_out,
                        "scale": label,
                        "weight_mean": float(aux[:, k].mean()),
                        "weight_std": float(aux[:, k].std()),
                        "weight_q10": float(np.quantile(aux[:, k], 0.10)),
                        "weight_q50": float(np.quantile(aux[:, k], 0.50)),
                        "weight_q90": float(np.quantile(aux[:, k], 0.90)),
                    }
                )
        elif aux is not None:
            raise RuntimeError("IUW-TCN unexpectedly returned selector weights")
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return pd.DataFrame(rows), pd.DataFrame(gate_rows)


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
    metrics, gates = run_fold(args, sources, device)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(args.out_dir / f"dynamic_multi_delay_{args.held_out_profile}.csv", index=False)
    gates.to_csv(args.out_dir / f"dynamic_multi_delay_weights_{args.held_out_profile}.csv", index=False)
    print(metrics.sort_values("MAE").to_string(index=False))
    print(gates.to_string(index=False))


if __name__ == "__main__":
    main()
