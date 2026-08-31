from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from run_external_e1_cross_cell import (
    CELLS,
    FEATURE_INDEX,
    MODELS,
    ExternalTCN,
    WindowDataset,
    apply_normalizer,
    fit_normalizer,
    load_segments,
    metric_dict,
    predict,
    seed_everything,
    train,
)

SOURCE_RATES = (0.2, 0.5)
TARGET_RATE = 1.0


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", type=Path, required=True)
    p.add_argument("--metadata", type=Path, required=True)
    p.add_argument("--cell", choices=CELLS, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--window", type=int, default=64)
    p.add_argument("--train-stride", type=int, default=4)
    p.add_argument("--test-stride", type=int, default=1)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    all_segments = load_segments(args.dataset, args.metadata)

    cell_segments = [s for s in all_segments if s["cell"] == args.cell]
    train_raw = [s for s in cell_segments if any(abs(s["rate_C"] - r) < 1e-9 for r in SOURCE_RATES)]
    test_raw = [s for s in cell_segments if abs(s["rate_C"] - TARGET_RATE) < 1e-9]
    other = [s for s in cell_segments if s not in train_raw and s not in test_raw]
    if other:
        raise RuntimeError(f"unexpected rates for {args.cell}: {[s['rate_C'] for s in other]}")
    if not train_raw or not test_raw:
        raise RuntimeError(f"missing E2 train/test segments for {args.cell}")
    if any(abs(s["rate_C"] - TARGET_RATE) < 1e-9 for s in train_raw):
        raise RuntimeError("1C target leakage into E2 training")

    mean, std = fit_normalizer(train_raw)
    train_segments = apply_normalizer(train_raw, mean, std)
    test_segments = apply_normalizer(test_raw, mean, std)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    np.savez(args.out_dir / "source_rate_normalizer.npz", mean=mean, std=std)

    fold_rows = []
    segment_rows = []
    prediction_rows = []

    for model_name in MODELS:
        feature_idx = FEATURE_INDEX[model_name]
        train_ds = WindowDataset(train_segments, args.window, args.train_stride, feature_idx)
        test_ds = WindowDataset(test_segments, args.window, args.test_stride, feature_idx)
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
        test_loader = DataLoader(test_ds, batch_size=args.batch_size * 2, shuffle=False, num_workers=0)

        seed_everything(args.seed)
        model = ExternalTCN(len(feature_idx), hidden=24)
        params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(
            f"\n=== E2 cell={args.cell} model={model_name} params={params} "
            f"train_segments={len(train_segments)} test_segments={len(test_segments)} "
            f"train_windows={len(train_ds)} test_windows={len(test_ds)} ==="
        )
        train(model, train_loader, device, args.epochs, args.lr)
        y, pred, seg_ids = predict(model, test_loader, device)

        fold_rows.append({
            "protocol": "external_E2_same_cell_0p2_0p5_to_1C",
            "cell": args.cell,
            "model": model_name,
            "source_rates": "0.2C+0.5C",
            "target_rate_C": 1.0,
            "seed": args.seed,
            "epochs": args.epochs,
            "window": args.window,
            "train_stride": args.train_stride,
            "test_stride": args.test_stride,
            "params": params,
            "n_train_segments": len(train_segments),
            "n_test_segments": len(test_segments),
            "n_train_windows": len(train_ds),
            "n_test_windows": len(test_ds),
            "source_rate_only_normalization": True,
            "target_1C_used_for_training": False,
            "target_1C_used_for_selection": False,
            **metric_dict(y, pred),
        })

        for local_sid, seg in enumerate(test_segments):
            mask = seg_ids == local_sid
            if not mask.any():
                continue
            m = metric_dict(y[mask], pred[mask])
            segment_rows.append({
                "cell": args.cell,
                "model": model_name,
                "segment_key": seg["key"],
                "rate_C": seg["rate_C"],
                "n_windows": int(mask.sum()),
                **m,
            })
            for yy, pp in zip(y[mask], pred[mask]):
                prediction_rows.append({
                    "cell": args.cell,
                    "model": model_name,
                    "segment_key": seg["key"],
                    "rate_C": seg["rate_C"],
                    "y_true": float(yy),
                    "y_pred": float(pp),
                    "abs_error": float(abs(yy - pp)),
                })

        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    pd.DataFrame(fold_rows).to_csv(args.out_dir / "e2_fold_metrics.csv", index=False)
    pd.DataFrame(segment_rows).to_csv(args.out_dir / "e2_segment_metrics.csv", index=False)
    pd.DataFrame(prediction_rows).to_csv(args.out_dir / "e2_predictions.csv", index=False)

    print("\n=== E2 fold metrics ===")
    print(pd.DataFrame(fold_rows).to_string(index=False))


if __name__ == "__main__":
    main()
