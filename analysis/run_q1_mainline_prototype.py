from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from run_sequence_representation_benchmark import (
    PROFILES,
    CausalConv1d,
    TCNBlock,
    TCNEncoder,
    SingleViewTCN,
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


class MultiScaleElectricalEncoder(nn.Module):
    """Causal electrical encoder with parallel local receptive fields."""

    def __init__(self, hidden: int = 24):
        super().__init__()
        self.branches = nn.ModuleList(
            [CausalConv1d(2, hidden, kernel_size=k, dilation=1) for k in (3, 5, 7)]
        )
        self.fuse = nn.Conv1d(hidden * 3, hidden, kernel_size=1)
        self.norm = nn.GroupNorm(1, hidden)
        self.blocks = nn.Sequential(
            TCNBlock(hidden, 1),
            TCNBlock(hidden, 2),
            TCNBlock(hidden, 4),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B,T,2]
        z = x.transpose(1, 2)
        z = torch.cat([F.gelu(branch(z)) for branch in self.branches], dim=1)
        z = F.gelu(self.norm(self.fuse(z)))
        z = self.blocks(z)
        return z[:, :, -1]


class SlowStateGRU(nn.Module):
    """Low-complexity encoder for thermo-mechanical state evolution."""

    def __init__(self, hidden: int = 24):
        super().__init__()
        self.gru = nn.GRU(input_size=2, hidden_size=hidden, num_layers=1, batch_first=True)
        self.norm = nn.LayerNorm(hidden)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, h = self.gru(x)
        return self.norm(h[-1])


class ElectricalOnlyMS(nn.Module):
    def __init__(self, hidden: int = 24):
        super().__init__()
        self.electrical = MultiScaleElectricalEncoder(hidden)
        self.head = nn.Sequential(nn.Linear(hidden, hidden), nn.GELU(), nn.Linear(hidden, 1))

    def forward(self, x: torch.Tensor):
        e = self.electrical(x[:, :, (0, 1)])
        return self.head(e).squeeze(-1), None


class HeterogeneousDirectFusion(nn.Module):
    """Electrical multi-scale TCN + thermo-mechanical GRU with direct latent fusion."""

    def __init__(self, aux_idx: tuple[int, int] = (4, 5), hidden: int = 24):
        super().__init__()
        self.aux_idx = aux_idx
        self.electrical = MultiScaleElectricalEncoder(hidden)
        self.auxiliary = SlowStateGRU(hidden)
        self.head = nn.Sequential(
            nn.Linear(hidden * 2, hidden * 2),
            nn.GELU(),
            nn.Linear(hidden * 2, 1),
        )

    def forward(self, x: torch.Tensor):
        e = self.electrical(x[:, :, (0, 1)])
        m = self.auxiliary(x[:, :, self.aux_idx])
        pred = self.head(torch.cat([e, m], dim=-1)).squeeze(-1)
        return pred, None


class BoundedResidualFusion(nn.Module):
    """Electrical base prediction plus gated, bounded optical/mechanical correction.

    The auxiliary branch cannot replace the electrical estimator globally. Its SOC
    correction is bounded by max_residual and scaled by a learned gate in [0,1].
    """

    def __init__(
        self,
        aux_idx: tuple[int, int] = (4, 5),
        hidden: int = 24,
        max_residual: float = 0.15,
    ):
        super().__init__()
        self.aux_idx = aux_idx
        self.max_residual = float(max_residual)
        self.electrical = MultiScaleElectricalEncoder(hidden)
        self.auxiliary = SlowStateGRU(hidden)
        self.base_head = nn.Sequential(
            nn.Linear(hidden, hidden), nn.GELU(), nn.Linear(hidden, 1)
        )
        self.residual_head = nn.Sequential(
            nn.Linear(hidden * 2, hidden), nn.GELU(), nn.Linear(hidden, 1)
        )
        self.gate_head = nn.Sequential(
            nn.Linear(hidden * 2, hidden), nn.GELU(), nn.Linear(hidden, 1)
        )

    def forward(self, x: torch.Tensor):
        e = self.electrical(x[:, :, (0, 1)])
        m = self.auxiliary(x[:, :, self.aux_idx])
        joint = torch.cat([e, m], dim=-1)
        base = self.base_head(e).squeeze(-1)
        residual = self.max_residual * torch.tanh(self.residual_head(joint).squeeze(-1))
        gate = torch.sigmoid(self.gate_head(joint).squeeze(-1))
        pred = base + gate * residual
        return pred, gate


def build_specs(max_residual: float):
    return (
        ("VI-MS-TCN", lambda: ElectricalOnlyMS()),
        ("IUTF-StackedTCN", lambda: SingleViewTCN((0, 1, 4, 5))),
        ("DualBranch-Direct-TF", lambda: HeterogeneousDirectFusion((4, 5))),
        ("BoundedResidual-TF", lambda: BoundedResidualFusion((4, 5), max_residual=max_residual)),
        ("BoundedResidual-W", lambda: BoundedResidualFusion((2, 3), max_residual=max_residual)),
    )


def direction_pairs(direction: str):
    if direction == "1C_to_2C":
        return (("1C", "2C"),)
    if direction == "2C_to_1C":
        return (("2C", "1C"),)
    if direction == "both":
        return (("1C", "2C"), ("2C", "1C"))
    raise ValueError(direction)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("data/extracted/SiC-18"))
    parser.add_argument("--out-dir", type=Path, default=Path("results/q1_mainline_prototype"))
    parser.add_argument("--direction", choices=("1C_to_2C", "2C_to_1C", "both"), default="1C_to_2C")
    parser.add_argument("--window", type=int, default=64)
    parser.add_argument("--train-stride", type=int, default=4)
    parser.add_argument("--test-stride", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-residual", type=float, default=0.15)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}")
    sources = load_sources(args.data)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    gate_rows: list[dict] = []

    for train_rate, test_rate in direction_pairs(args.direction):
        direction = f"{train_rate}_to_{test_rate}"
        for held_out in PROFILES:
            train_raw = [
                s for s in sources
                if s["rate"] == train_rate and s["profile"] != held_out
            ]
            test_raw = [
                s for s in sources
                if s["rate"] == test_rate and s["profile"] == held_out
            ]
            if len(train_raw) != 5 or len(test_raw) != 1:
                raise RuntimeError(f"Bad split {direction}/{held_out}")

            mean, std = train_normalizer(train_raw)
            train_sources = normalize_sources(train_raw, mean, std)
            test_sources = normalize_sources(test_raw, mean, std)
            train_ds = WindowDataset(train_sources, args.window, args.train_stride)
            test_ds = WindowDataset(test_sources, args.window, args.test_stride)
            test_loader = DataLoader(
                test_ds, batch_size=args.batch_size * 2, shuffle=False, num_workers=0
            )

            print(
                f"\n=== Q1 prototype {direction}/{held_out}: "
                f"train={len(train_ds)} test={len(test_ds)} ==="
            )
            for model_name, factory in build_specs(args.max_residual):
                # Reset before construction and rebuild shuffled loader for a fair RNG start.
                seed_everything(args.seed)
                model = factory()
                params = count_params(model)
                train_loader = DataLoader(
                    train_ds,
                    batch_size=args.batch_size,
                    shuffle=True,
                    num_workers=0,
                )
                print(f"--- {model_name} params={params} ---")
                train_model(model, train_loader, device, args.epochs, args.lr)
                y, pred, _, gate = predict_model(model, test_loader, device)
                rows.append(
                    {
                        "protocol": "q1_prototype_cross_rate_unseen_profile",
                        "direction": direction,
                        "held_out_profile": held_out,
                        "model": model_name,
                        "seed": args.seed,
                        "epochs": args.epochs,
                        "window": args.window,
                        "params": params,
                        "n_test": len(y),
                        **metric_dict(y, pred),
                    }
                )
                if gate is not None:
                    gate_rows.append(
                        {
                            "direction": direction,
                            "held_out_profile": held_out,
                            "model": model_name,
                            "gate_mean": float(np.mean(gate)),
                            "gate_std": float(np.std(gate)),
                            "gate_q10": float(np.quantile(gate, 0.10)),
                            "gate_q50": float(np.quantile(gate, 0.50)),
                            "gate_q90": float(np.quantile(gate, 0.90)),
                        }
                    )
                del model
                if device.type == "cuda":
                    torch.cuda.empty_cache()

    split_df = pd.DataFrame(rows)
    gate_df = pd.DataFrame(gate_rows)
    aggregate = (
        split_df.groupby(["direction", "model"], as_index=False)
        .agg(
            n_splits=("MAE", "size"),
            MAE_mean=("MAE", "mean"),
            MAE_std=("MAE", "std"),
            RMSE_mean=("RMSE", "mean"),
            R2_mean=("R2", "mean"),
            Q95_AE_mean=("Q95_AE", "mean"),
            MaxAE_mean=("MaxAE", "mean"),
            params=("params", "first"),
        )
    )

    split_df.to_csv(args.out_dir / "split_metrics.csv", index=False)
    gate_df.to_csv(args.out_dir / "gate_summary.csv", index=False)
    aggregate.to_csv(args.out_dir / "aggregate_summary.csv", index=False)

    print("\n=== Q1 M1 prototype summary ===")
    print(aggregate.sort_values(["direction", "MAE_mean"]).to_string(index=False))
    if not gate_df.empty:
        print("\n=== Learned bounded-residual gate ===")
        print(gate_df.to_string(index=False))
    print(
        "\nNOTE: this is a fixed-epoch development screen only. "
        "Any surviving candidate must be rerun with source-only epoch selection and multiple seeds."
    )


if __name__ == "__main__":
    main()
