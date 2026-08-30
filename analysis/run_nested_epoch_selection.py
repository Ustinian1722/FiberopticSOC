from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from run_representation_conditioning_diagnostic import PairTCN, whitening_matrix
from run_same_rate_lopo_selective_optical import ParameterMatchedVITCN
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


def train_curve(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    max_epochs: int,
    lr: float,
) -> list[dict]:
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    loss_fn = nn.MSELoss()
    rows = []
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
        rows.append(
            {
                "epoch": epoch,
                "train_mse": total / max(n, 1),
                **{f"val_{k}": v for k, v in metric_dict(yv, pv).items()},
            }
        )
    return rows


def train_fixed_epochs(model, loader, device, epochs, lr):
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    loss_fn = nn.MSELoss()
    for _ in range(epochs):
        model.train()
        for x, y, _ in loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            pred, _ = model(x)
            loss = loss_fn(pred, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()


def model_factory(name: str, train_sources: list[dict]):
    if name == "VI":
        return ParameterMatchedVITCN()
    if name == "VI+W":
        return PairTCN((2, 3))
    if name == "VI+W-white":
        train_all = np.concatenate([s["x"] for s in train_sources], axis=0)
        return PairTCN((2, 3), whitening_matrix(train_all[:, (2, 3)]))
    raise ValueError(name)


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
    p.add_argument("--max-epochs", type=int, default=15)
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

    models = ("VI", "VI+W", "VI+W-white")
    curve_rows = []
    selected_rows = []
    test_rows = []

    for model_name in models:
        fold_curves = []
        for fold_idx, val_profile in enumerate(sorted(s["profile"] for s in main_train_raw)):
            inner_train_raw = [s for s in main_train_raw if s["profile"] != val_profile]
            inner_val_raw = [s for s in main_train_raw if s["profile"] == val_profile]
            mean, std = train_normalizer(inner_train_raw)
            inner_train = normalize_sources(inner_train_raw, mean, std)
            inner_val = normalize_sources(inner_val_raw, mean, std)
            train_ds = WindowDataset(inner_train, args.window, args.train_stride)
            val_ds = WindowDataset(inner_val, args.window, args.val_stride)
            train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
            val_loader = DataLoader(val_ds, batch_size=args.batch_size * 2, shuffle=False, num_workers=0)

            seed_everything(args.seed)
            model = model_factory(model_name, inner_train)
            curve = train_curve(model, train_loader, val_loader, device, args.max_epochs, args.lr)
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
            fold_curves.append(pd.DataFrame(curve))
            del model

        merged = pd.concat(
            [c.assign(fold=i) for i, c in enumerate(fold_curves)], ignore_index=True
        )
        epoch_summary = merged.groupby("epoch", as_index=False).agg(
            val_MAE_mean=("val_MAE", "mean"),
            val_MAE_std=("val_MAE", "std"),
            val_RMSE_mean=("val_RMSE", "mean"),
        )
        best = epoch_summary.sort_values(["val_MAE_mean", "epoch"], kind="stable").iloc[0]
        selected_epoch = int(best["epoch"])
        selected_rows.append(
            {
                "direction": direction,
                "held_out_profile": args.held_out_profile,
                "model": model_name,
                "seed": args.seed,
                "selected_epoch": selected_epoch,
                "inner_val_MAE_mean": float(best["val_MAE_mean"]),
                "inner_val_MAE_std": float(best["val_MAE_std"]),
                "hit_max_epoch": bool(selected_epoch == args.max_epochs),
            }
        )

        mean, std = train_normalizer(main_train_raw)
        final_train = normalize_sources(main_train_raw, mean, std)
        final_test = normalize_sources(final_test_raw, mean, std)
        train_ds = WindowDataset(final_train, args.window, args.train_stride)
        test_ds = WindowDataset(final_test, args.window, args.test_stride)
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
        test_loader = DataLoader(test_ds, batch_size=args.batch_size * 2, shuffle=False, num_workers=0)
        seed_everything(args.seed)
        model = model_factory(model_name, final_train)
        params = count_params(model)
        train_fixed_epochs(model, train_loader, device, selected_epoch, args.lr)
        y, pred, _, _ = predict_model(model, test_loader, device)
        test_rows.append(
            {
                "protocol": "nested_profile_validation_epoch_selection",
                "direction": direction,
                "train_rate": args.train_rate,
                "test_rate": test_rate,
                "held_out_profile": args.held_out_profile,
                "model": model_name,
                "params": params,
                "seed": args.seed,
                "selected_epoch": selected_epoch,
                "n_test": len(y),
                **metric_dict(y, pred),
            }
        )
        del model

    args.out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(curve_rows).to_csv(args.out_dir / "inner_validation_curves.csv", index=False)
    pd.DataFrame(selected_rows).to_csv(args.out_dir / "selected_epochs.csv", index=False)
    pd.DataFrame(test_rows).to_csv(args.out_dir / "test_metrics.csv", index=False)
    print("=== Selected epochs ===")
    print(pd.DataFrame(selected_rows).to_string(index=False))
    print("\n=== Final held-out test ===")
    print(pd.DataFrame(test_rows).sort_values("MAE").to_string(index=False))


if __name__ == "__main__":
    main()
