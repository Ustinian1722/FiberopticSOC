from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from run_q2_etmf_prototype import ETMFNet
from run_q1_mainline_prototype import MultiScaleElectricalEncoder, SlowStateGRU
from run_representation_conditioning_diagnostic import PairTCN, covariance_condition, whitening_matrix
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


class ETMFOptical(nn.Module):
    """ETMF with an explicitly selected optical coordinate and optional train-only whitening."""

    def __init__(self, pair: tuple[int, int], whiten: np.ndarray | None = None, hidden: int = 24):
        super().__init__()
        self.pair = pair
        if whiten is None:
            self.register_buffer("whiten", torch.empty(0), persistent=False)
        else:
            self.register_buffer("whiten", torch.tensor(whiten, dtype=torch.float32), persistent=True)

        self.electrical = MultiScaleElectricalEncoder(hidden)
        self.optical = SlowStateGRU(hidden)
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
        optical = x[:, :, self.pair]
        if self.whiten.numel():
            optical = torch.matmul(optical, self.whiten.T)
        m = self.optical(optical)

        e_refined = self.e_norm(e + self.e_cross_gate(m) * self.m_to_e(m))
        m_refined = self.m_norm(m + self.m_cross_gate(e) * self.e_to_m(e))
        interaction = torch.abs(e_refined - m_refined)
        alpha = self.mix_gate(torch.cat([e_refined, m_refined, interaction], dim=-1))
        fused = alpha * e_refined + (1.0 - alpha) * m_refined
        pred = self.head(torch.cat([fused, e_refined * m_refined, interaction], dim=-1)).squeeze(-1)
        return pred, alpha.squeeze(-1)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=Path, default=Path("data/extracted/SiC-18"))
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--window", type=int, default=64)
    p.add_argument("--train-stride", type=int, default=4)
    p.add_argument("--test-stride", type=int, default=1)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sources = load_sources(args.data)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    metric_rows: list[dict] = []
    condition_rows: list[dict] = []
    gate_rows: list[dict] = []

    for held_out in PROFILES:
        train_raw = [s for s in sources if s["rate"] == "1C" and s["profile"] != held_out]
        test_raw = [s for s in sources if s["rate"] == "2C" and s["profile"] == held_out]
        if len(train_raw) != 5 or len(test_raw) != 1:
            raise RuntimeError(f"Bad representation screen split {held_out}")

        mean, std = train_normalizer(train_raw)
        train = normalize_sources(train_raw, mean, std)
        test = normalize_sources(test_raw, mean, std)
        train_all = np.concatenate([s["x"] for s in train], axis=0)

        transforms: dict[str, np.ndarray] = {}
        for label, pair in (("W", (2, 3)), ("TF", (4, 5))):
            z = train_all[:, pair]
            white = whitening_matrix(z)
            transforms[label] = white
            condition_rows.append(
                {
                    "held_out_profile": held_out,
                    "representation": label,
                    "condition_before": covariance_condition(z),
                    "condition_after": covariance_condition(z @ white.T),
                    "corr_before": float(np.corrcoef(z.T)[0, 1]),
                }
            )

        train_ds = WindowDataset(train, args.window, args.train_stride)
        test_ds = WindowDataset(test, args.window, args.test_stride)
        test_loader = DataLoader(test_ds, batch_size=args.batch_size * 2, shuffle=False, num_workers=0)

        specs = (
            ("IUW-TCN", lambda: PairTCN((2, 3), None)),
            ("IUTF-TCN", lambda: PairTCN((4, 5), None)),
            ("IUWwhite-TCN", lambda: PairTCN((2, 3), transforms["W"])),
            ("IUTFwhite-TCN", lambda: PairTCN((4, 5), transforms["TF"])),
            ("ETMF-W", lambda: ETMFOptical((2, 3), None)),
            ("ETMF-TF", lambda: ETMFNet()),
            ("ETMF-Wwhite", lambda: ETMFOptical((2, 3), transforms["W"])),
            ("ETMF-TFwhite", lambda: ETMFOptical((4, 5), transforms["TF"])),
        )

        print(f"\n=== representation screen 1C->2C/{held_out} train={len(train_ds)} test={len(test_ds)} ===")
        for model_name, factory in specs:
            seed_everything(args.seed)
            model = factory()
            train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
            params = count_params(model)
            print(f"--- {model_name} params={params} epochs={args.epochs} ---")
            train_model(model, train_loader, device, args.epochs, args.lr)
            y, pred, _, gate = predict_model(model, test_loader, device)
            metric_rows.append(
                {
                    "protocol": "q2_representation_equal_budget_1C_to_2C_unseen_profile",
                    "held_out_profile": held_out,
                    "model": model_name,
                    "seed": args.seed,
                    "epochs": args.epochs,
                    "params": params,
                    "n_test": len(y),
                    **metric_dict(y, pred),
                }
            )
            if gate is not None:
                gate_rows.append(
                    {
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

    metrics = pd.DataFrame(metric_rows)
    aggregate = (
        metrics.groupby("model", as_index=False)
        .agg(
            n_profiles=("MAE", "size"),
            MAE_mean=("MAE", "mean"),
            MAE_std=("MAE", "std"),
            RMSE_mean=("RMSE", "mean"),
            R2_mean=("R2", "mean"),
            Q95_AE_mean=("Q95_AE", "mean"),
            MaxAE_mean=("MaxAE", "mean"),
            params=("params", "first"),
        )
        .sort_values("MAE_mean")
    )
    winners = metrics.loc[metrics.groupby("held_out_profile")["MAE"].idxmin(), ["held_out_profile", "model", "MAE"]]
    winners = winners.rename(columns={"model": "winner_model", "MAE": "winner_MAE"}).sort_values("held_out_profile")

    ref = metrics[metrics["model"] == "IUW-TCN"][["held_out_profile", "MAE"]].rename(columns={"MAE": "IUW_MAE"})
    paired = metrics.merge(ref, on="held_out_profile", how="left")
    paired["gain_vs_IUW"] = paired["IUW_MAE"] - paired["MAE"]
    paired["relative_gain_vs_IUW"] = paired["gain_vs_IUW"] / paired["IUW_MAE"]

    metrics.to_csv(args.out_dir / "split_metrics.csv", index=False)
    aggregate.to_csv(args.out_dir / "aggregate_summary.csv", index=False)
    winners.to_csv(args.out_dir / "split_winners.csv", index=False)
    paired.to_csv(args.out_dir / "paired_vs_IUW.csv", index=False)
    pd.DataFrame(condition_rows).to_csv(args.out_dir / "conditioning.csv", index=False)
    pd.DataFrame(gate_rows).to_csv(args.out_dir / "fusion_alpha.csv", index=False)

    print("\n=== equal-budget representation summary ===")
    print(aggregate.to_string(index=False))
    print("\n=== split winners ===")
    print(winners.to_string(index=False))
    print("\nDecision rule: do not retain TF/ETMF merely for physical narrative if raw-W/simple models dominate.")


if __name__ == "__main__":
    main()
