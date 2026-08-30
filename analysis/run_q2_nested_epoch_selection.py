from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from run_nested_epoch_earlystop import train_fold_earlystop
from run_nested_epoch_selection import train_fixed_epochs
from run_q1_mainline_prototype import ElectricalOnlyMS, HeterogeneousDirectFusion
from run_q2_etmf_prototype import ETMFNet
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
    train_normalizer,
)

MODEL_NAMES = (
    "VI-MS-TCN",
    "IUTF-StackedTCN",
    "DualBranch-Direct-TF",
    "ETMF-TF",
)


def model_factory(name: str):
    if name == "VI-MS-TCN":
        return ElectricalOnlyMS()
    if name == "IUTF-StackedTCN":
        return SingleViewTCN((0, 1, 4, 5))
    if name == "DualBranch-Direct-TF":
        return HeterogeneousDirectFusion((4, 5))
    if name == "ETMF-TF":
        return ETMFNet()
    raise ValueError(name)


def parse_models(text: str) -> tuple[str, ...]:
    names = tuple(x.strip() for x in text.split(",") if x.strip())
    bad = [x for x in names if x not in MODEL_NAMES]
    if bad:
        raise ValueError(f"Unknown model names: {bad}; allowed={MODEL_NAMES}")
    if not names:
        raise ValueError("At least one model is required")
    return names


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=Path, default=Path("data/extracted/SiC-18"))
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--train-rate", choices=["1C", "2C"], default="1C")
    p.add_argument("--held-out-profile", choices=list(PROFILES), required=True)
    p.add_argument("--models", type=str, default="IUTF-StackedTCN,ETMF-TF")
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

    models = parse_models(args.models)
    test_rate = "2C" if args.train_rate == "1C" else "1C"
    direction = f"{args.train_rate}_to_{test_rate}"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sources = load_sources(args.data)

    # Strict T4 main split. The held-out profile is absent from *all* source-side
    # training and epoch-selection data, even at the source rate.
    main_train_raw = [
        s for s in sources
        if s["rate"] == args.train_rate and s["profile"] != args.held_out_profile
    ]
    final_test_raw = [
        s for s in sources
        if s["rate"] == test_rate and s["profile"] == args.held_out_profile
    ]
    if len(main_train_raw) != 5 or len(final_test_raw) != 1:
        raise RuntimeError("Unexpected strict T4 split size")

    curve_rows: list[dict] = []
    fold_rows: list[dict] = []
    selected_rows: list[dict] = []
    audit_rows: list[dict] = []

    source_profiles = sorted(s["profile"] for s in main_train_raw)
    for model_name in models:
        best_epochs: list[int] = []
        best_maes: list[float] = []

        for fold_idx, val_profile in enumerate(source_profiles):
            inner_train_raw = [s for s in main_train_raw if s["profile"] != val_profile]
            inner_val_raw = [s for s in main_train_raw if s["profile"] == val_profile]

            mean, std = train_normalizer(inner_train_raw)
            inner_train = normalize_sources(inner_train_raw, mean, std)
            inner_val = normalize_sources(inner_val_raw, mean, std)
            train_ds = WindowDataset(inner_train, args.window, args.train_stride)
            val_ds = WindowDataset(inner_val, args.window, args.val_stride)

            seed_everything(args.seed)
            model = model_factory(model_name)
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
            best_epochs.append(int(best_epoch))
            best_maes.append(float(best_row["val_MAE"]))
            fold_rows.append(
                {
                    "direction": direction,
                    "main_held_out_profile": args.held_out_profile,
                    "model": model_name,
                    "inner_val_profile": val_profile,
                    "fold": fold_idx,
                    "development_seed": args.seed,
                    "best_epoch": int(best_epoch),
                    "best_val_MAE": float(best_row["val_MAE"]),
                    "stopped_epoch": int(stopped_epoch),
                    "hit_max_epoch": bool(hit_max),
                }
            )
            for row in curve:
                curve_rows.append(
                    {
                        **row,
                        "direction": direction,
                        "main_held_out_profile": args.held_out_profile,
                        "model": model_name,
                        "inner_val_profile": val_profile,
                        "fold": fold_idx,
                        "development_seed": args.seed,
                    }
                )
            del model

        # Five inner folds => median is robust to one unusually slow/fast profile.
        selected_epoch = int(np.median(np.asarray(best_epochs, dtype=int)))
        selected_rows.append(
            {
                "direction": direction,
                "held_out_profile": args.held_out_profile,
                "model": model_name,
                "development_seed": args.seed,
                "selected_epoch": selected_epoch,
                "fold_best_epoch_min": int(np.min(best_epochs)),
                "fold_best_epoch_median": float(np.median(best_epochs)),
                "fold_best_epoch_max": int(np.max(best_epochs)),
                "fold_best_MAE_mean": float(np.mean(best_maes)),
                "fold_best_MAE_std": float(np.std(best_maes, ddof=1)),
                "folds_hit_max": int(
                    sum(
                        int(r["hit_max_epoch"])
                        for r in fold_rows
                        if r["direction"] == direction
                        and r["main_held_out_profile"] == args.held_out_profile
                        and r["model"] == model_name
                    )
                ),
            }
        )

        # Development-seed final-test pass is audit-only and cannot change the frozen epoch.
        mean, std = train_normalizer(main_train_raw)
        final_train = normalize_sources(main_train_raw, mean, std)
        final_test = normalize_sources(final_test_raw, mean, std)
        train_ds = WindowDataset(final_train, args.window, args.train_stride)
        test_ds = WindowDataset(final_test, args.window, args.test_stride)
        seed_everything(args.seed)
        model = model_factory(model_name)
        train_loader = DataLoader(
            train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0
        )
        test_loader = DataLoader(
            test_ds, batch_size=args.batch_size * 2, shuffle=False, num_workers=0
        )
        params = count_params(model)
        train_fixed_epochs(model, train_loader, device, selected_epoch, args.lr)
        y, pred, _, gate = predict_model(model, test_loader, device)
        row = {
            "protocol": "q2_nested_source_profile_cv_epoch_selection",
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
        if gate is not None:
            row.update(
                {
                    "alpha_mean": float(np.mean(gate)),
                    "alpha_q10": float(np.quantile(gate, 0.10)),
                    "alpha_q50": float(np.quantile(gate, 0.50)),
                    "alpha_q90": float(np.quantile(gate, 0.90)),
                }
            )
        audit_rows.append(row)
        del model

    args.out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(curve_rows).to_csv(args.out_dir / "inner_validation_curves.csv", index=False)
    pd.DataFrame(fold_rows).to_csv(args.out_dir / "fold_best_epochs.csv", index=False)
    pd.DataFrame(selected_rows).to_csv(args.out_dir / "selected_epochs.csv", index=False)
    pd.DataFrame(audit_rows).to_csv(args.out_dir / "development_test_audit.csv", index=False)

    print("=== Q2 source-only frozen epoch candidates ===")
    print(pd.DataFrame(selected_rows).to_string(index=False))
    print("\n=== seed42 final-test audit only (must not alter the epoch plan) ===")
    print(pd.DataFrame(audit_rows).sort_values("MAE").to_string(index=False))


if __name__ == "__main__":
    main()
