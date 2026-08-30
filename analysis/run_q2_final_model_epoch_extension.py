from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from run_representation_conditioning_diagnostic import PairTCN
from run_sequence_representation_benchmark import (
    PROFILES,
    WindowDataset,
    load_sources,
    metric_dict,
    normalize_sources,
    predict_model,
    seed_everything,
    train_normalizer,
)

FROZEN_MODEL = "IUW-TCN"


def train_fold_earlystop(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    *,
    max_epochs: int,
    min_epochs: int,
    patience: int,
    min_delta: float,
    lr: float,
) -> tuple[list[dict], int, int, bool]:
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    loss_fn = nn.MSELoss()

    rows: list[dict] = []
    best_mae = float("inf")
    best_epoch = 1
    patience_anchor = float("inf")
    stale = 0

    for epoch in range(1, max_epochs + 1):
        model.train()
        total = 0.0
        n = 0
        for x, y, _ in train_loader:
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad(set_to_none=True)
            pred, _ = model(x)
            loss = loss_fn(pred, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()
            total += float(loss.detach()) * len(y)
            n += len(y)

        yv, pv, _, _ = predict_model(model, val_loader, device)
        metrics = metric_dict(yv, pv)
        mae = float(metrics["MAE"])
        rows.append(
            {
                "epoch": epoch,
                "train_mse": total / max(n, 1),
                **{f"val_{k}": v for k, v in metrics.items()},
            }
        )

        if mae < best_mae:
            best_mae = mae
            best_epoch = epoch

        if mae < patience_anchor - min_delta:
            patience_anchor = mae
            stale = 0
        else:
            stale += 1

        if epoch >= min_epochs and stale >= patience:
            return rows, best_epoch, epoch, False

    return rows, best_epoch, max_epochs, True


def main() -> None:
    p = argparse.ArgumentParser(
        description=(
            "Extend exactly one ceiling-bound source-only inner fold for the frozen raw-W IUW-TCN. "
            "The opposite-rate target trajectory is never constructed or evaluated."
        )
    )
    p.add_argument("--data", type=Path, default=Path("data/extracted/SiC-18"))
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--train-rate", choices=("1C", "2C"), required=True)
    p.add_argument("--held-out-profile", choices=PROFILES, required=True)
    p.add_argument("--inner-val-profile", choices=PROFILES, required=True)
    p.add_argument("--window", type=int, default=64)
    p.add_argument("--train-stride", type=int, default=4)
    p.add_argument("--val-stride", type=int, default=1)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--max-epochs", type=int, default=100)
    p.add_argument("--min-epochs", type=int, default=12)
    p.add_argument("--patience", type=int, default=8)
    p.add_argument("--min-delta", type=float, default=5e-5)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    if args.inner_val_profile == args.held_out_profile:
        raise ValueError("inner-val-profile cannot equal held-out-profile")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    all_sources = load_sources(args.data)

    # Source pool only. The opposite-rate target is intentionally absent from this program.
    source_pool = [
        s
        for s in all_sources
        if s["rate"] == args.train_rate and s["profile"] != args.held_out_profile
    ]
    if len(source_pool) != 5:
        raise RuntimeError(
            f"Expected five source profiles after excluding {args.held_out_profile}, got {len(source_pool)}"
        )
    if args.inner_val_profile not in {s["profile"] for s in source_pool}:
        raise RuntimeError(f"Inner validation profile {args.inner_val_profile} is not in the source pool")

    inner_train_raw = [s for s in source_pool if s["profile"] != args.inner_val_profile]
    inner_val_raw = [s for s in source_pool if s["profile"] == args.inner_val_profile]
    if len(inner_train_raw) != 4 or len(inner_val_raw) != 1:
        raise RuntimeError("Expected four inner-train profiles and one inner-validation profile")

    mean, std = train_normalizer(inner_train_raw)
    inner_train = normalize_sources(inner_train_raw, mean, std)
    inner_val = normalize_sources(inner_val_raw, mean, std)

    train_ds = WindowDataset(inner_train, args.window, args.train_stride)
    val_ds = WindowDataset(inner_val, args.window, args.val_stride)

    seed_everything(args.seed)
    model = PairTCN((2, 3), None)  # frozen IUW-TCN: electrical I/U + raw W1/W2
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size * 2, shuffle=False, num_workers=0)

    curve, best_epoch, stopped_epoch, hit_max = train_fold_earlystop(
        model,
        train_loader,
        val_loader,
        device,
        max_epochs=args.max_epochs,
        min_epochs=args.min_epochs,
        patience=args.patience,
        min_delta=args.min_delta,
        lr=args.lr,
    )
    best_row = min(curve, key=lambda r: r["val_MAE"])
    direction = f"{args.train_rate}_to_{'2C' if args.train_rate == '1C' else '1C'}"

    fold = pd.DataFrame(
        [
            {
                "direction": direction,
                "main_held_out_profile": args.held_out_profile,
                "model": "VI+W",
                "frozen_model": FROZEN_MODEL,
                "inner_val_profile": args.inner_val_profile,
                "seed": args.seed,
                "best_epoch": best_epoch,
                "best_val_MAE": float(best_row["val_MAE"]),
                "stopped_epoch": stopped_epoch,
                "hit_max_epoch": hit_max,
                "max_epochs": args.max_epochs,
                "source_only": True,
                "target_metrics_computed": False,
            }
        ]
    )
    curves = pd.DataFrame(
        [
            {
                **row,
                "direction": direction,
                "main_held_out_profile": args.held_out_profile,
                "model": "VI+W",
                "frozen_model": FROZEN_MODEL,
                "inner_val_profile": args.inner_val_profile,
                "seed": args.seed,
            }
            for row in curve
        ]
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    curves.to_csv(args.out_dir / "inner_validation_curve.csv", index=False)
    fold.to_csv(args.out_dir / "fold_extension.csv", index=False)

    print("=== Frozen IUW-TCN source-only ceiling extension ===")
    print(fold.to_string(index=False))
    print("No opposite-rate target metrics were computed.")


if __name__ == "__main__":
    main()
