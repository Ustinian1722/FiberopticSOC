from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from run_q1_mainline_prototype import SlowStateGRU
from run_q2_etmf_prototype import ETMFNet
from run_sequence_representation_benchmark import (
    PROFILES,
    RATES,
    CausalConv1d,
    TCNBlock,
    WindowDataset,
    count_params,
    discover_workbooks,
    metric_dict,
    normalize_sources,
    predict_model,
    seed_everything,
    train_model,
    train_normalizer,
)

BASE_COLUMNS = (
    "Voltage_V",
    "Current_A",
    "Wavelength_1",
    "Wavelength_2",
    "temperature_℃",
    "force_N",
)
DT_INDEX = 6


def load_sources_with_dt(root: Path) -> list[dict]:
    sources = []
    for path in discover_workbooks(root):
        profile, rate = path.stem.rsplit("_", 1)
        if profile not in PROFILES or rate not in RATES:
            continue
        df = pd.read_excel(path)
        t = df["Time_s"].to_numpy(dtype=np.float64)
        dt = np.diff(t, prepend=t[0])
        # delta-t is a causal elapsed-time feature, not absolute trajectory time.
        # Duplicate timestamps are retained as dt=0. Negative time would be invalid.
        if np.any(dt < 0):
            raise ValueError(f"Negative timestamp step in {path.name}")
        log_dt = np.log1p(dt).astype(np.float32)
        x = np.column_stack(
            [df[list(BASE_COLUMNS)].to_numpy(dtype=np.float32), log_dt]
        ).astype(np.float32)
        y = df["SOC"].to_numpy(dtype=np.float32)
        sources.append(
            {
                "name": path.name,
                "profile": profile,
                "rate": rate,
                "x": x,
                "y": y,
            }
        )
    if len(sources) != 12:
        raise RuntimeError(f"Expected 12 sources, got {len(sources)}")
    return sources


class MultiScaleElectricalDTEncoder(nn.Module):
    """Same electrical encoder capacity as the Q2 prototype, but with causal log(1+dt)."""

    def __init__(self, hidden: int = 24):
        super().__init__()
        self.branches = nn.ModuleList(
            [CausalConv1d(3, hidden, kernel_size=k, dilation=1) for k in (3, 5, 7)]
        )
        self.fuse = nn.Conv1d(hidden * 3, hidden, kernel_size=1)
        self.norm = nn.GroupNorm(1, hidden)
        self.blocks = nn.Sequential(
            TCNBlock(hidden, 1), TCNBlock(hidden, 2), TCNBlock(hidden, 4)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = x.transpose(1, 2)
        z = torch.cat([F.gelu(branch(z)) for branch in self.branches], dim=1)
        z = F.gelu(self.norm(self.fuse(z)))
        z = self.blocks(z)
        return z[:, :, -1]


class ETMFNetDT(nn.Module):
    """ETMF variant that observes causal elapsed-time increments in the electrical branch."""

    def __init__(self, hidden: int = 24):
        super().__init__()
        self.electrical = MultiScaleElectricalDTEncoder(hidden)
        self.thermomechanical = SlowStateGRU(hidden)
        self.m_to_e = nn.Sequential(nn.Linear(hidden, hidden), nn.Tanh())
        self.e_to_m = nn.Sequential(nn.Linear(hidden, hidden), nn.Tanh())
        self.e_cross_gate = nn.Sequential(nn.Linear(hidden, hidden), nn.Sigmoid())
        self.m_cross_gate = nn.Sequential(nn.Linear(hidden, hidden), nn.Sigmoid())
        self.e_norm = nn.LayerNorm(hidden)
        self.m_norm = nn.LayerNorm(hidden)
        self.mix_gate = nn.Sequential(
            nn.Linear(hidden * 3, hidden), nn.GELU(), nn.Linear(hidden, 1), nn.Sigmoid()
        )
        self.head = nn.Sequential(
            nn.Linear(hidden * 3, hidden * 2),
            nn.GELU(),
            nn.Linear(hidden * 2, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x: torch.Tensor):
        e = self.electrical(x[:, :, (0, 1, DT_INDEX)])
        m = self.thermomechanical(x[:, :, (4, 5)])
        e_refined = self.e_norm(e + self.e_cross_gate(m) * self.m_to_e(m))
        m_refined = self.m_norm(m + self.m_cross_gate(e) * self.e_to_m(e))
        interaction = torch.abs(e_refined - m_refined)
        alpha = self.mix_gate(torch.cat([e_refined, m_refined, interaction], dim=-1))
        fused = alpha * e_refined + (1.0 - alpha) * m_refined
        pred = self.head(
            torch.cat([fused, e_refined * m_refined, interaction], dim=-1)
        ).squeeze(-1)
        return pred, alpha.squeeze(-1)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=Path, default=Path("data/extracted/SiC-18"))
    p.add_argument("--out-dir", type=Path, default=Path("results/q2_timegap_ablation"))
    p.add_argument("--epochs", type=int, default=12)
    p.add_argument("--window", type=int, default=64)
    p.add_argument("--train-stride", type=int, default=4)
    p.add_argument("--test-stride", type=int, default=1)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    sources = load_sources_with_dt(args.data)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows = []

    for held_out in PROFILES:
        train_raw = [s for s in sources if s["rate"] == "1C" and s["profile"] != held_out]
        test_raw = [s for s in sources if s["rate"] == "2C" and s["profile"] == held_out]
        mean, std = train_normalizer(train_raw)
        train = normalize_sources(train_raw, mean, std)
        test = normalize_sources(test_raw, mean, std)
        train_ds = WindowDataset(train, args.window, args.train_stride)
        test_ds = WindowDataset(test, args.window, args.test_stride)
        test_loader = DataLoader(test_ds, batch_size=args.batch_size * 2, shuffle=False)

        for name, factory in (
            ("ETMF-no-dt", ETMFNet),
            ("ETMF-causal-dt", ETMFNetDT),
        ):
            seed_everything(args.seed)
            model = factory()
            train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
            train_model(model, train_loader, device, args.epochs, args.lr)
            y, pred, _, gate = predict_model(model, test_loader, device)
            rows.append(
                {
                    "direction": "1C_to_2C",
                    "held_out_profile": held_out,
                    "model": name,
                    "seed": args.seed,
                    "epochs": args.epochs,
                    "params": count_params(model),
                    "alpha_mean": float(np.mean(gate)) if gate is not None else np.nan,
                    **metric_dict(y, pred),
                }
            )

    df = pd.DataFrame(rows)
    summary = df.groupby("model", as_index=False).agg(
        MAE_mean=("MAE", "mean"),
        MAE_std=("MAE", "std"),
        RMSE_mean=("RMSE", "mean"),
        R2_mean=("R2", "mean"),
        Q95_AE_mean=("Q95_AE", "mean"),
    )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out_dir / "split_metrics.csv", index=False)
    summary.to_csv(args.out_dir / "aggregate_summary.csv", index=False)
    print(summary.sort_values("MAE_mean").to_string(index=False))
    print("Development ablation only; delta-t is a data-integrity feature, not a novelty claim.")


if __name__ == "__main__":
    main()
