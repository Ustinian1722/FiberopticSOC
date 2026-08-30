from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from run_nested_epoch_selection import model_factory, train_fixed_epochs
from run_sequence_representation_benchmark import (
    PROFILES,
    WindowDataset,
    count_params,
    load_sources,
    metric_dict,
    normalize_sources,
    predict_model,
    seed_everything,
    train_normalizer,
)


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
        total, n = 0.0, 0
        for x, y, _ in train_loader:
            x, y = x.to(device), y.to(device)
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

        # Patience is based on meaningful progress, while best_epoch still records
        # the exact minimum among all observed epochs.
        if mae < patience_anchor - min_delta:
            patience_anchor = mae
            stale = 0
        else:
            stale += 1

        if epoch >= min_epochs and stale >= patience:
            return rows, best_epoch, epoch, False

    return rows, best_epoch, max_epochs, True


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=Path, default=Path("data/extracted"))
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--train-rate", choices=["1C", "2C"], required=True)
    p.add_argument("--held-out-profile", choices=list(PROFILES), required=True)
    p.add_argument("--window", type=int, default=64)
    p.add_argument("--train-stride", type=int, default=4)
    p.add_argument("--val-stride", type=int, default=1)
    p.add_argument("--test-stride", type=int, default=1)
    p.add_argument("--max-epochs", type=int, default=60)
    p.add_argument("--min-epochs", type=int, default=12)
    p.add_argument("--patience", type=int, default=8)
    p.add_argument("--min-delta", type=float, default=5e-5)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    test_rate = "2C" if args.train_rate == "1C" else "1C"
    direction = f"{args.train_rate}_to_{test_rate}"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sources = load_sources(args.data)
    main_train_raw = [
        s for s in sources
        if s["rate"] == args.train_rate and s["profile"] != args.held_out_profile
    ]
    final_test_raw = [
        s for s in sources
        if s["rate"] == test_rate and s["profile"] == args.held_out_profile
    ]
    if len(main_train_raw) != 5 or len(final_test_raw) != 1:
        raise RuntimeError("Unexpected main split size")

    curve_rows: list[dict] = []
    fold_rows: list[dict] = []
    selected_rows: list[dict] = []
    test_rows: list[dict] = []

    for model_name in ("VI", "VI+W", "VI+W-white"):
        best_epochs: list[int] = []
        fold_best_maes: list[float] = []
        for fold_idx, val_profile in enumerate(sorted(s["profile"] for s in main_train_raw)):
            inner_train_raw = [s for s in main_train_raw if s["profile"] != val_profile]
            inner_val_raw = [s for s in main_train_raw if s["profile"] == val_profile]
            mean, std = train_normalizer(inner_train_raw)
            inner_train = normalize_sources(inner_train_raw, mean, std)
            inner_val = normalize_sources(inner_val_raw, mean, std)
            train_ds = WindowDataset(inner_train, args.window, args.train_stride)
            val_ds = WindowDataset(inner_val, args.window, args.val_stride)

            seed_everything(args.seed)
            model = model_factory(model_name, inner_train)
            train_loader = DataLoader(
                train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0
            )
            val_loader = DataLoader(
                val_ds, batch_size=args.batch_size * 2, shuffle=False, num_workers=0
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
            fold_best_maes.append(float(best_row["val_MAE"]))
            fold_rows.append(
                {
                    "direction": direction,
                    "main_held_out_profile": args.held_out_profile,
                    "model": model_name,
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
                row.update(
                    {
                        "direction": direction,
                        "main_held_out_profile": args.held_out_profile,
                        "model": model_name,
                        "inner_val_profile": val_profile,
                        "fold": fold_idx,
                        "seed": args.seed,
                    }
                )
                curve_rows.append(row)
            del model

        # Five folds -> median is an observed integer epoch and is resistant to one
        # unusually slow/fast profile. This epoch is frozen before final test use.
        selected_epoch = int(np.median(np.asarray(best_epochs, dtype=int)))
        selected_rows.append(
            {
                "direction": direction,
                "held_out_profile": args.held_out_profile,
                "model": model_name,
                "seed": args.seed,
                "selected_epoch": selected_epoch,
                "fold_best_epoch_min": int(np.min(best_epochs)),
                "fold_best_epoch_median": float(np.median(best_epochs)),
                "fold_best_epoch_max": int(np.max(best_epochs)),
                "fold_best_MAE_mean": float(np.mean(fold_best_maes)),
                "fold_best_MAE_std": float(np.std(fold_best_maes, ddof=1)),
                "folds_hit_max": int(
                    sum(
                        1
                        for r in fold_rows
                        if r["direction"] == direction
                        and r["main_held_out_profile"] == args.held_out_profile
                        and r["model"] == model_name
                        and r["hit_max_epoch"]
                    )
                ),
            }
        )

        mean, std = train_normalizer(main_train_raw)
        final_train = normalize_sources(main_train_raw, mean, std)
        final_test = normalize_sources(final_test_raw, mean, std)
        train_ds = WindowDataset(final_train, args.window, args.train_stride)
        test_ds = WindowDataset(final_test, args.window, args.test_stride)
        seed_everything(args.seed)
        model = model_factory(model_name, final_train)
        train_loader = DataLoader(
            train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0
        )
        test_loader = DataLoader(
            test_ds, batch_size=args.batch_size * 2, shuffle=False, num_workers=0
        )
        params = count_params(model)
        train_fixed_epochs(model, train_loader, device, selected_epoch, args.lr)
        y, pred, _, _ = predict_model(model, test_loader, device)
        test_rows.append(
            {
                "protocol": "nested_profile_cv_median_earlystop_epoch_selection",
                "direction": direction,
                "train_rate": args.train_rate,
                "test_rate": test_rate,
                "held_out_profile": args.held_out_profile,
                "model": model_name,
                "params": params,
                "development_seed": args.seed,
                "selected_epoch": selected_epoch,
                "n_test": len(y),
                **metric_dict(y, pred),
            }
        )
        del model

    args.out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(curve_rows).to_csv(args.out_dir / "inner_validation_curves.csv", index=False)
    pd.DataFrame(fold_rows).to_csv(args.out_dir / "fold_best_epochs.csv", index=False)
    pd.DataFrame(selected_rows).to_csv(args.out_dir / "selected_epochs.csv", index=False)
    pd.DataFrame(test_rows).to_csv(args.out_dir / "development_test_audit.csv", index=False)

    print("=== Frozen epoch candidates from train-only CV ===")
    print(pd.DataFrame(selected_rows).to_string(index=False))
    print("\n=== Development-seed test audit (not final statistics) ===")
    print(pd.DataFrame(test_rows).sort_values("MAE").to_string(index=False))


if __name__ == "__main__":
    main()
