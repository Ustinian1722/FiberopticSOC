from __future__ import annotations

import argparse
import copy
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from q2_protocols import blocked_mixed_condition_split
from run_q2_nested_epoch_selection import MODEL_NAMES, model_factory
from run_sequence_representation_benchmark import (
    WindowDataset,
    count_params,
    load_sources,
    metric_dict,
    normalize_sources,
    predict_model,
    seed_everything,
    train_normalizer,
)


def train_earlystop_restore_best(
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
) -> tuple[list[dict], int]:
    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    loss_fn = nn.MSELoss()
    curve = []
    best_mae = float("inf")
    best_epoch = 1
    best_state = copy.deepcopy(model.state_dict())
    patience_anchor = float("inf")
    stale = 0

    for epoch in range(1, max_epochs + 1):
        model.train()
        total, n = 0.0, 0
        for x, y, _ in train_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad(set_to_none=True)
            pred, _ = model(x)
            loss = loss_fn(pred, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            opt.step()
            total += float(loss.detach()) * len(y)
            n += len(y)

        yv, pv, _, _ = predict_model(model, val_loader, device)
        vm = metric_dict(yv, pv)
        mae = float(vm["MAE"])
        curve.append({"epoch": epoch, "train_mse": total / max(n, 1), **{f"val_{k}": v for k, v in vm.items()}})
        if mae < best_mae:
            best_mae = mae
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())

        if mae < patience_anchor - min_delta:
            patience_anchor = mae
            stale = 0
        else:
            stale += 1
        if epoch >= min_epochs and stale >= patience:
            break

    model.load_state_dict(best_state)
    return curve, best_epoch


def prediction_frame(raw_sources: list[dict], ds: WindowDataset, y: np.ndarray, pred: np.ndarray, gate: np.ndarray | None) -> pd.DataFrame:
    rows = []
    for source_id, end in ds.index:
        src = raw_sources[source_id]
        rows.append(
            {
                "source_name": src["name"],
                "parent_name": src.get("parent_name", src["name"]),
                "profile": src["profile"],
                "rate": src["rate"],
                "segment_start": int(src.get("segment_start", 0)),
                "row_end_in_segment": int(end),
                "current_A": float(src["x"][end, 1]),
                "voltage_V": float(src["x"][end, 0]),
            }
        )
    out = pd.DataFrame(rows)
    out["y_true"] = y
    out["y_pred"] = pred
    out["abs_error"] = np.abs(y - pred)
    out["alpha_fusion"] = gate if gate is not None else np.nan
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=Path, default=Path("data/extracted/SiC-18"))
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--models", default=",".join(MODEL_NAMES))
    p.add_argument("--window", type=int, default=64)
    p.add_argument("--train-stride", type=int, default=2)
    p.add_argument("--eval-stride", type=int, default=1)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--max-epochs", type=int, default=50)
    p.add_argument("--min-epochs", type=int, default=10)
    p.add_argument("--patience", type=int, default=7)
    p.add_argument("--min-delta", type=float, default=5e-5)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--seed", type=int, required=True)
    args = p.parse_args()

    models = tuple(x.strip() for x in args.models.split(",") if x.strip())
    bad = [m for m in models if m not in MODEL_NAMES]
    if bad:
        raise ValueError(f"Unknown models: {bad}")

    raw = load_sources(args.data)
    split = blocked_mixed_condition_split(raw, window=args.window)
    mean, std = train_normalizer(split.train)
    train = normalize_sources(split.train, mean, std)
    val = normalize_sources(split.validation, mean, std)
    cal = normalize_sources(split.calibration, mean, std)
    test = normalize_sources(split.test, mean, std)

    train_ds = WindowDataset(train, args.window, args.train_stride)
    val_ds = WindowDataset(val, args.window, args.eval_stride)
    cal_ds = WindowDataset(cal, args.window, args.eval_stride)
    test_ds = WindowDataset(test, args.window, args.eval_stride)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size * 2, shuffle=False)
    cal_loader = DataLoader(cal_ds, batch_size=args.batch_size * 2, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size * 2, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    metric_rows = []
    curve_rows = []
    cal_frames = []
    test_frames = []

    for model_name in models:
        seed_everything(args.seed)
        model = model_factory(model_name)
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
        curve, best_epoch = train_earlystop_restore_best(
            model, train_loader, val_loader, device,
            max_epochs=args.max_epochs, min_epochs=args.min_epochs,
            patience=args.patience, min_delta=args.min_delta, lr=args.lr,
        )
        for row in curve:
            curve_rows.append({"seed": args.seed, "model": model_name, **row})

        yc, pc, _, gc = predict_model(model, cal_loader, device)
        yt, pt, _, gt = predict_model(model, test_loader, device)
        metric_rows.append(
            {
                "protocol": split.name,
                "split_id": split.split_id,
                "model": model_name,
                "seed": args.seed,
                "best_epoch": best_epoch,
                "params": count_params(model),
                "n_train_windows": len(train_ds),
                "n_val_windows": len(val_ds),
                "n_cal_windows": len(cal_ds),
                "n_test_windows": len(test_ds),
                **metric_dict(yt, pt),
            }
        )
        cf = prediction_frame(split.calibration, cal_ds, yc, pc, gc)
        cf["model"] = model_name; cf["seed"] = args.seed; cf["set"] = "calibration"
        tf = prediction_frame(split.test, test_ds, yt, pt, gt)
        tf["model"] = model_name; tf["seed"] = args.seed; tf["set"] = "test"
        cal_frames.append(cf); test_frames.append(tf)
        del model

    metrics = pd.DataFrame(metric_rows)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(args.out_dir / "point_metrics.csv", index=False)
    pd.DataFrame(curve_rows).to_csv(args.out_dir / "validation_curves.csv", index=False)
    pd.concat(cal_frames, ignore_index=True).to_csv(args.out_dir / "calibration_predictions.csv", index=False)
    pd.concat(test_frames, ignore_index=True).to_csv(args.out_dir / "test_predictions.csv", index=False)
    print(metrics.sort_values("MAE").to_string(index=False))
    print("T1 calibration blocks were not used for model fitting or early stopping and are reserved for UQ.")


if __name__ == "__main__":
    main()
