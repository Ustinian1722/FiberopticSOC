from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

CELLS = ("A1", "A2", "P1", "P2")
MODELS = ("VI-TCN", "VI-S5rel-TCN")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    args = p.parse_args()

    fold_files = sorted(args.root.rglob("e2_fold_metrics.csv"))
    seg_files = sorted(args.root.rglob("e2_segment_metrics.csv"))
    pred_files = sorted(args.root.rglob("e2_predictions.csv"))
    if len(fold_files) != 4:
        raise RuntimeError(f"expected four E2 fold files, found {len(fold_files)}")

    folds = pd.concat([pd.read_csv(f) for f in fold_files], ignore_index=True)
    segs = pd.concat([pd.read_csv(f) for f in seg_files], ignore_index=True)
    preds = pd.concat([pd.read_csv(f) for f in pred_files], ignore_index=True)

    if len(folds) != 8:
        raise RuntimeError(f"expected 8 fold/model rows, found {len(folds)}")
    if set(folds.cell.astype(str)) != set(CELLS) or set(folds.model.astype(str)) != set(MODELS):
        raise RuntimeError("E2 cell/model set mismatch")
    if folds.duplicated(["cell", "model"]).any():
        raise RuntimeError("duplicate E2 cell/model row")
    if not folds.source_rate_only_normalization.astype(str).str.lower().eq("true").all():
        raise RuntimeError("non-source-rate normalization detected")
    if folds.target_1C_used_for_training.astype(str).str.lower().eq("true").any():
        raise RuntimeError("1C target used for training")
    if folds.target_1C_used_for_selection.astype(str).str.lower().eq("true").any():
        raise RuntimeError("1C target used for selection")

    summary = folds.groupby("model", as_index=False).agg(
        n_cells=("MAE", "size"),
        MAE_cell_mean=("MAE", "mean"),
        MAE_cell_std=("MAE", "std"),
        RMSE_cell_mean=("RMSE", "mean"),
        RMSE_cell_std=("RMSE", "std"),
        R2_cell_mean=("R2", "mean"),
        R2_cell_std=("R2", "std"),
        Q95_AE_cell_mean=("Q95_AE", "mean"),
        Q95_AE_cell_std=("Q95_AE", "std"),
        MaxAE_cell_mean=("MaxAE", "mean"),
        MaxAE_cell_std=("MaxAE", "std"),
    )

    wide = folds.pivot(index="cell", columns="model", values=["MAE", "RMSE", "R2", "Q95_AE", "MaxAE"])
    paired_rows = []
    for cell in CELLS:
        base_mae = float(wide.loc[cell, ("MAE", "VI-TCN")])
        opt_mae = float(wide.loc[cell, ("MAE", "VI-S5rel-TCN")])
        paired_rows.append({
            "cell": cell,
            "VI_MAE": base_mae,
            "VI_S5rel_MAE": opt_mae,
            "delta_MAE_optical_minus_base": opt_mae - base_mae,
            "relative_MAE_change": (opt_mae - base_mae) / base_mae,
            "optical_MAE_win": opt_mae < base_mae,
            "VI_RMSE": float(wide.loc[cell, ("RMSE", "VI-TCN")]),
            "VI_S5rel_RMSE": float(wide.loc[cell, ("RMSE", "VI-S5rel-TCN")]),
            "VI_Q95_AE": float(wide.loc[cell, ("Q95_AE", "VI-TCN")]),
            "VI_S5rel_Q95_AE": float(wide.loc[cell, ("Q95_AE", "VI-S5rel-TCN")]),
            "VI_R2": float(wide.loc[cell, ("R2", "VI-TCN")]),
            "VI_S5rel_R2": float(wide.loc[cell, ("R2", "VI-S5rel-TCN")]),
        })
    paired = pd.DataFrame(paired_rows)

    base = summary[summary.model == "VI-TCN"].iloc[0]
    opt = summary[summary.model == "VI-S5rel-TCN"].iloc[0]
    wins = int(paired.optical_MAE_win.sum())
    worst_rel = float(paired.relative_MAE_change.max())
    criteria = {
        "criterion_mean_cell_MAE_improves": bool(opt.MAE_cell_mean < base.MAE_cell_mean),
        "criterion_wins_at_least_3_of_4_cells": bool(wins >= 3),
        "criterion_mean_cell_RMSE_nonworse": bool(opt.RMSE_cell_mean <= base.RMSE_cell_mean),
        "criterion_mean_cell_Q95_nonworse": bool(opt.Q95_AE_cell_mean <= base.Q95_AE_cell_mean),
        "criterion_worst_cell_MAE_increase_le_10pct": bool(worst_rel <= 0.10),
    }
    keep = all(criteria.values())
    decision_name = (
        "SUPPORTS_ROBUST_SAME_CELL_CROSS_RATE_FBG_BENEFIT"
        if keep
        else "DOES_NOT_SUPPORT_ROBUST_SAME_CELL_CROSS_RATE_FBG_BENEFIT"
    )
    decision = {
        "decision": decision_name,
        "mean_delta_MAE_optical_minus_base": float(opt.MAE_cell_mean - base.MAE_cell_mean),
        "mean_delta_RMSE_optical_minus_base": float(opt.RMSE_cell_mean - base.RMSE_cell_mean),
        "mean_delta_Q95_optical_minus_base": float(opt.Q95_AE_cell_mean - base.Q95_AE_cell_mean),
        "MAE_wins": wins,
        "worst_relative_MAE_change": worst_rel,
        **criteria,
        "E1_conclusion_reopened": False,
        "SiC_model_selection_reopened": False,
        "target_1C_used_for_tuning": False,
    }

    pooled = folds.groupby("model", as_index=False).agg(
        n_test_windows=("n_test_windows", "sum"),
    )
    pooled_metrics = []
    for model in MODELS:
        pdat = preds[preds.model == model]
        ae = (pdat.y_true - pdat.y_pred).abs()
        pooled_metrics.append({
            "model": model,
            "n_windows": len(pdat),
            "MAE": float(ae.mean()),
            "RMSE": float(((pdat.y_true - pdat.y_pred) ** 2).mean() ** 0.5),
            "Q95_AE": float(ae.quantile(0.95)),
            "MaxAE": float(ae.max()),
        })
    pooled = pd.DataFrame(pooled_metrics)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    folds.to_csv(args.out_dir / "all_e2_fold_metrics.csv", index=False)
    segs.to_csv(args.out_dir / "all_e2_segment_metrics.csv", index=False)
    summary.to_csv(args.out_dir / "e2_cell_mean_summary.csv", index=False)
    paired.to_csv(args.out_dir / "e2_paired_cell_comparison.csv", index=False)
    pooled.to_csv(args.out_dir / "e2_pooled_window_summary.csv", index=False)
    pd.DataFrame([decision]).to_csv(args.out_dir / "e2_external_evidence_decision.csv", index=False)
    (args.out_dir / "e2_external_evidence_decision.json").write_text(json.dumps(decision, indent=2), encoding="utf-8")

    print("=== E2 equal-cell summary ===")
    print(summary.to_string(index=False))
    print("=== E2 paired cell comparison ===")
    print(paired.to_string(index=False))
    print("=== E2 preregistered decision ===")
    print(json.dumps(decision, indent=2))


if __name__ == "__main__":
    main()
