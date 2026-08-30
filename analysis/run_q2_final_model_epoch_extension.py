from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from run_q2_etmf_prototype import ETMFNet
from run_q2_representation_aware_screen import ETMFOptical
from run_representation_conditioning_diagnostic import PairTCN, whitening_matrix
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

# Exact candidate identities from run_q2_representation_aware_screen.py.
CANDIDATES = (
    "IUW-TCN",
    "IUTF-TCN",
    "IUWwhite-TCN",
    "IUTFwhite-TCN",
    "ETMF-W",
    "ETMF-TF",
    "ETMF-Wwhite",
    "ETMF-TFwhite",
)


def build_model(name: str, normalized_train_sources: list[dict]) -> nn.Module:
    """Build the frozen candidate using transforms fitted on source-train only."""
    train_all = np.concatenate([s["x"] for s in normalized_train_sources], axis=0)

    if name == "IUW-TCN":
        return PairTCN((2, 3), None)
    if name == "IUTF-TCN":
        return PairTCN((4, 5), None)
    if name == "IUWwhite-TCN":
        white = whitening_matrix(train_all[:, (2, 3)])
        return PairTCN((2, 3), white)
    if name == "IUTFwhite-TCN":
        white = whitening_matrix(train_all[:, (4, 5)])
        return PairTCN((4, 5), white)
    if name == "ETMF-W":
        return ETMFOptical((2, 3), None)
    if name == "ETMF-TF":
        return ETMFNet()
    if name == "ETMF-Wwhite":
        white = whitening_matrix(train_all[:, (2, 3)])
        return ETMFOptical((2, 3), white)
    if name == "ETMF-TFwhite":
        white = whitening_matrix(train_all[:, (4, 5)])
        return ETMFOptical((4, 5), white)
    raise ValueError(name)


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
            "Source-only epoch extension for exactly one frozen Q2 representation/model. "
            "This program never evaluates the opposite-rate target trajectory."
        )
    )
    p.add_argument("--data", type=Path, default=Path("data/extracted/SiC-18"))
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--model", choices=CANDIDATES, required=True)
    p.add_argument("--train-rate", choices=("1C", "2C"), required=True)
    p.add_argument("--held-out-profile", choices=PROFILES, required=True)
    p.add_argument("--window", type=int, default=64)
    p.add_argument("--train-stride", type=int, default=4)
    p.add_argument("--val-stride", type=int, default=1)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--max-epochs", type=int, default=100)
    p.add_argument("--min-epochs", type=int, default=12)
    p.add_argument("--patience", type=int, default=10)
    p.add_argument("--min-delta", type=float, default=5e-5)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    all_sources = load_sources(args.data)

    # The opposite-rate target is intentionally never constructed or evaluated.
    source_pool = [
        s
        for s in all_sources
        if s["rate"] == args.train_rate and s["profile"] != args.held_out_profile
    ]
    if len(source_pool) != 5:
        raise RuntimeError(
            f"Expected five source profiles after excluding {args.held_out_profile}, "
            f"got {len(source_pool)}"
        )

    curve_rows: list[dict] = []
    fold_rows: list[dict] = []
    best_epochs: list[int] = []
    best_maes: list[float] = []

    for fold_idx, val_profile in enumerate(sorted(s["profile"] for s in source_pool)):
        inner_train_raw = [s for s in source_pool if s["profile"] != val_profile]
        inner_val_raw = [s for s in source_pool if s["profile"] == val_profile]
        if len(inner_train_raw) != 4 or len(inner_val_raw) != 1:
            raise RuntimeError(f"Bad inner split for {val_profile}")

        mean, std = train_normalizer(inner_train_raw)
        inner_train = normalize_sources(inner_train_raw, mean, std)
        inner_val = normalize_sources(inner_val_raw, mean, std)

        train_ds = WindowDataset(inner_train, args.window, args.train_stride)
        val_ds = WindowDataset(inner_val, args.window, args.val_stride)

        seed_everything(args.seed)
        model = build_model(args.model, inner_train)
        train_loader = DataLoader(
            train_ds,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=0,
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=args.batch_size * 2,
            shuffle=False,
            num_workers=0,
        )

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
        best_epochs.append(best_epoch)
        best_maes.append(float(best_row["val_MAE"]))

        fold_rows.append(
            {
                "train_rate": args.train_rate,
                "held_out_profile": args.held_out_profile,
                "model": args.model,
                "inner_val_profile": val_profile,
                "fold": fold_idx,
                "seed": args.seed,
                "best_epoch": best_epoch,
                "best_val_MAE": float(best_row["val_MAE"]),
                "stopped_epoch": stopped_epoch,
                "hit_max_epoch": hit_max,
            }
        )
        for row in curve:
            curve_rows.append(
                {
                    **row,
                    "train_rate": args.train_rate,
                    "held_out_profile": args.held_out_profile,
                    "model": args.model,
                    "inner_val_profile": val_profile,
                    "fold": fold_idx,
                    "seed": args.seed,
                }
            )
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    selected_epoch = int(np.median(np.asarray(best_epochs, dtype=int)))
    selected = pd.DataFrame(
        [
            {
                "direction": f"{args.train_rate}_to_{'2C' if args.train_rate == '1C' else '1C'}",
                "train_rate": args.train_rate,
                "held_out_profile": args.held_out_profile,
                "model": args.model,
                "seed": args.seed,
                "selected_epoch": selected_epoch,
                "fold_best_epoch_min": int(np.min(best_epochs)),
                "fold_best_epoch_median": float(np.median(best_epochs)),
                "fold_best_epoch_max": int(np.max(best_epochs)),
                "fold_best_MAE_mean": float(np.mean(best_maes)),
                "fold_best_MAE_std": float(np.std(best_maes, ddof=1)),
                "folds_hit_max": int(sum(bool(r["hit_max_epoch"]) for r in fold_rows)),
                "max_epochs": args.max_epochs,
                "source_only": True,
                "target_metrics_computed": False,
            }
        ]
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(curve_rows).to_csv(args.out_dir / "inner_validation_curves.csv", index=False)
    pd.DataFrame(fold_rows).to_csv(args.out_dir / "fold_best_epochs.csv", index=False)
    selected.to_csv(args.out_dir / "selected_epoch.csv", index=False)

    print("=== Source-only final-model epoch extension ===")
    print(selected.to_string(index=False))
    print("No opposite-rate target metrics were computed.")


if __name__ == "__main__":
    main()
