from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from run_sequence_representation_benchmark import (
    PROFILES,
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
from run_q1_mainline_prototype import (
    ElectricalOnlyMS,
    HeterogeneousDirectFusion,
    MultiScaleElectricalEncoder,
    SlowStateGRU,
)


class ETMFNet(nn.Module):
    """Compact electrical-thermomechanical fusion network.

    The electrical branch captures fast V/I dynamics with a multi-scale causal TCN.
    The thermo-mechanical branch captures slower T/F evolution with a GRU. Each branch
    first gates information projected from the other branch, after which a scalar
    adaptive mixing coefficient combines the refined latent states. The scalar mixing
    weight is exported for interpretation but is not supervised by test labels.
    """

    def __init__(self, hidden: int = 24):
        super().__init__()
        self.electrical = MultiScaleElectricalEncoder(hidden)
        self.thermomechanical = SlowStateGRU(hidden)

        self.m_to_e = nn.Sequential(nn.Linear(hidden, hidden), nn.Tanh())
        self.e_to_m = nn.Sequential(nn.Linear(hidden, hidden), nn.Tanh())
        self.e_cross_gate = nn.Sequential(nn.Linear(hidden, hidden), nn.Sigmoid())
        self.m_cross_gate = nn.Sequential(nn.Linear(hidden, hidden), nn.Sigmoid())
        self.e_norm = nn.LayerNorm(hidden)
        self.m_norm = nn.LayerNorm(hidden)

        self.mix_gate = nn.Sequential(
            nn.Linear(hidden * 3, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
            nn.Sigmoid(),
        )
        self.head = nn.Sequential(
            nn.Linear(hidden * 3, hidden * 2),
            nn.GELU(),
            nn.Linear(hidden * 2, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x: torch.Tensor):
        e = self.electrical(x[:, :, (0, 1)])
        m = self.thermomechanical(x[:, :, (4, 5)])

        e_refined = self.e_norm(e + self.e_cross_gate(m) * self.m_to_e(m))
        m_refined = self.m_norm(m + self.m_cross_gate(e) * self.e_to_m(e))
        interaction = torch.abs(e_refined - m_refined)

        alpha = self.mix_gate(torch.cat([e_refined, m_refined, interaction], dim=-1))
        fused = alpha * e_refined + (1.0 - alpha) * m_refined
        pred = self.head(torch.cat([fused, e_refined * m_refined, interaction], dim=-1)).squeeze(-1)
        return pred, alpha.squeeze(-1)


def specs():
    return (
        ("VI-MS-TCN", lambda: ElectricalOnlyMS()),
        ("IUTF-StackedTCN", lambda: SingleViewTCN((0, 1, 4, 5))),
        ("DualBranch-Direct-TF", lambda: HeterogeneousDirectFusion((4, 5))),
        ("ETMF-TF", lambda: ETMFNet()),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("data/extracted/SiC-18"))
    parser.add_argument("--out-dir", type=Path, default=Path("results/q2_etmf_prototype"))
    parser.add_argument("--window", type=int, default=64)
    parser.add_argument("--train-stride", type=int, default=4)
    parser.add_argument("--test-stride", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}")
    sources = load_sources(args.data)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    gate_rows: list[dict] = []

    # Development screen is deliberately only the hardest low->high-rate direction.
    train_rate, test_rate = "1C", "2C"
    direction = "1C_to_2C"
    for held_out in PROFILES:
        train_raw = [s for s in sources if s["rate"] == train_rate and s["profile"] != held_out]
        test_raw = [s for s in sources if s["rate"] == test_rate and s["profile"] == held_out]
        if len(train_raw) != 5 or len(test_raw) != 1:
            raise RuntimeError(f"Bad split {direction}/{held_out}")

        mean, std = train_normalizer(train_raw)
        train_sources = normalize_sources(train_raw, mean, std)
        test_sources = normalize_sources(test_raw, mean, std)
        train_ds = WindowDataset(train_sources, args.window, args.train_stride)
        test_ds = WindowDataset(test_sources, args.window, args.test_stride)
        test_loader = DataLoader(test_ds, batch_size=args.batch_size * 2, shuffle=False, num_workers=0)

        print(f"\n=== Q2-P1 {direction}/{held_out}: train={len(train_ds)} test={len(test_ds)} ===")
        for model_name, factory in specs():
            # Reset before construction so all parameter-matched comparisons have a reproducible RNG start.
            seed_everything(args.seed)
            model = factory()
            params = count_params(model)
            train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
            print(f"--- {model_name} params={params} ---")
            train_model(model, train_loader, device, args.epochs, args.lr)
            y, pred, _, gate = predict_model(model, test_loader, device)
            rows.append(
                {
                    "protocol": "q2_p1_cross_rate_unseen_profile",
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
                        "alpha_mean": float(np.mean(gate)),
                        "alpha_std": float(np.std(gate)),
                        "alpha_q10": float(np.quantile(gate, 0.10)),
                        "alpha_q50": float(np.quantile(gate, 0.50)),
                        "alpha_q90": float(np.quantile(gate, 0.90)),
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
    winner = (
        split_df.loc[split_df.groupby(["direction", "held_out_profile"])["MAE"].idxmin(),
                     ["direction", "held_out_profile", "model", "MAE"]]
        .rename(columns={"model": "winner_model", "MAE": "winner_MAE"})
        .sort_values(["direction", "held_out_profile"])
    )

    split_df.to_csv(args.out_dir / "split_metrics.csv", index=False)
    gate_df.to_csv(args.out_dir / "fusion_alpha.csv", index=False)
    aggregate.to_csv(args.out_dir / "aggregate_summary.csv", index=False)
    winner.to_csv(args.out_dir / "split_winners.csv", index=False)

    print("\n=== Q2-P1 ETMF summary ===")
    print(aggregate.sort_values(["direction", "MAE_mean"]).to_string(index=False))
    print("\n=== Split winners ===")
    print(winner.to_string(index=False))
    if not gate_df.empty:
        print("\n=== ETMF electrical mixing alpha (1=electrical, 0=thermomechanical) ===")
        print(gate_df.to_string(index=False))
    print("\nDevelopment-only screen: no publication claim from seed42/fixed epochs.")


if __name__ == "__main__":
    main()
