from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from run_representation_conditioning_diagnostic import PairTCN
from run_sequence_representation_benchmark import (
    PROFILES,
    RATES,
    TCNEncoder,
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
    """Load raw sensing channels plus strictly causal log1p(delta-t).

    Absolute Time_s is never exposed to the model. The first sample in every trajectory
    receives delta-t=0. All statistics, including delta-t normalization, are fitted only
    on the source-training profiles by train_normalizer().
    """
    sources: list[dict] = []
    for path in discover_workbooks(root):
        profile, rate = path.stem.rsplit("_", 1)
        if profile not in PROFILES or rate not in RATES:
            continue
        df = pd.read_excel(path)
        required = set(BASE_COLUMNS) | {"Time_s", "SOC"}
        missing = required.difference(df.columns)
        if missing:
            raise ValueError(f"{path.name} missing columns: {sorted(missing)}")

        t = df["Time_s"].to_numpy(dtype=np.float64)
        dt = np.diff(t, prepend=t[0])
        if not np.isfinite(dt).all() or np.any(dt < 0):
            raise ValueError(f"Invalid timestamp increments in {path.name}")
        log_dt = np.log1p(dt).astype(np.float32)

        x = np.column_stack(
            [df[list(BASE_COLUMNS)].to_numpy(dtype=np.float32), log_dt]
        ).astype(np.float32)
        y = df["SOC"].to_numpy(dtype=np.float32)
        if not np.isfinite(x).all() or not np.isfinite(y).all():
            raise ValueError(f"Non-finite values in {path.name}")
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


class RawWPairTCNDT(nn.Module):
    """Frozen raw-W PairTCN body with one additional causal delta-t input channel."""

    def __init__(self, hidden: int = 24):
        super().__init__()
        # I, U, W1, W2, log1p(delta-t). Body depth/width/head match frozen PairTCN.
        self.encoder = TCNEncoder(5, hidden)
        self.head = nn.Sequential(nn.Linear(hidden, hidden), nn.GELU(), nn.Linear(hidden, 1))

    def forward(self, x: torch.Tensor):
        z = x[:, :, (0, 1, 2, 3, DT_INDEX)]
        h = self.encoder(z)
        return self.head(h).squeeze(-1), None


def main() -> None:
    p = argparse.ArgumentParser(
        description=(
            "Matched source-only validation of causal delta-t for the frozen IUW-TCN. "
            "Uses one same-rate source profile for validation and never constructs an opposite-rate target."
        )
    )
    p.add_argument("--data", type=Path, default=Path("data/extracted/SiC-18"))
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--train-rate", choices=RATES, required=True)
    p.add_argument("--val-profile", choices=PROFILES, required=True)
    p.add_argument("--epochs", type=int, required=True)
    p.add_argument("--window", type=int, default=64)
    p.add_argument("--train-stride", type=int, default=4)
    p.add_argument("--val-stride", type=int, default=1)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    if args.epochs < 1:
        raise ValueError("epochs must be positive")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sources = load_sources_with_dt(args.data)

    # Same-rate source-only LOPO. No object from the opposite rate is ever selected below.
    train_raw = [
        s for s in sources if s["rate"] == args.train_rate and s["profile"] != args.val_profile
    ]
    val_raw = [
        s for s in sources if s["rate"] == args.train_rate and s["profile"] == args.val_profile
    ]
    if len(train_raw) != 5 or len(val_raw) != 1:
        raise RuntimeError("Expected five same-rate source-train profiles and one validation profile")

    mean, std = train_normalizer(train_raw)
    train = normalize_sources(train_raw, mean, std)
    val = normalize_sources(val_raw, mean, std)
    train_ds = WindowDataset(train, args.window, args.train_stride)
    val_ds = WindowDataset(val, args.window, args.val_stride)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size * 2, shuffle=False, num_workers=0)

    rows: list[dict] = []
    for name, factory in (
        ("IUW-TCN-no-dt", lambda: PairTCN((2, 3), None)),
        ("IUW-TCN-causal-dt", RawWPairTCNDT),
    ):
        # Reset RNG before each matched arm. The only designed model difference is the
        # additional delta-t input coefficient in the first 1x1 projection.
        seed_everything(args.seed)
        model = factory()
        train_loader = DataLoader(
            train_ds,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=0,
        )
        train_model(model, train_loader, device, args.epochs, args.lr)
        y, pred, _, _ = predict_model(model, val_loader, device)
        rows.append(
            {
                "train_rate": args.train_rate,
                "val_profile": args.val_profile,
                "model": name,
                "seed": args.seed,
                "epochs": args.epochs,
                "params": count_params(model),
                "n_train_windows": len(train_ds),
                "n_val_windows": len(val_ds),
                "source_only": True,
                "opposite_rate_target_constructed": False,
                **metric_dict(y, pred),
            }
        )
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    df = pd.DataFrame(rows)
    no_dt = df.loc[df.model == "IUW-TCN-no-dt"].iloc[0]
    dt = df.loc[df.model == "IUW-TCN-causal-dt"].iloc[0]
    paired = pd.DataFrame(
        [
            {
                "train_rate": args.train_rate,
                "val_profile": args.val_profile,
                "seed": args.seed,
                "epochs": args.epochs,
                "delta_MAE_dt_minus_no_dt": float(dt.MAE - no_dt.MAE),
                "delta_RMSE_dt_minus_no_dt": float(dt.RMSE - no_dt.RMSE),
                "delta_Q95_AE_dt_minus_no_dt": float(dt.Q95_AE - no_dt.Q95_AE),
                "dt_wins_MAE": bool(dt.MAE < no_dt.MAE),
                "source_only": True,
            }
        ]
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out_dir / "arm_metrics.csv", index=False)
    paired.to_csv(args.out_dir / "paired_delta.csv", index=False)
    print(df.to_string(index=False))
    print(paired.to_string(index=False))
    print("No opposite-rate target was constructed or evaluated.")


if __name__ == "__main__":
    main()
