from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

CELLS = ("A1", "A2", "P1", "P2")
BASE = "VI-TCN"
OPTICAL = "VI-S5rel-TCN"


def load_all(root: Path, filename: str) -> pd.DataFrame:
    files = sorted(root.rglob(filename))
    if not files:
        raise FileNotFoundError(f"no {filename} under {root}")
    return pd.concat([pd.read_csv(p) for p in files], ignore_index=True)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    args = p.parse_args()

    fold = load_all(args.root, "e1_fold_metrics.csv")
    rate = load_all(args.root, "e1_rate_metrics.csv")
    seg = load_all(args.root, "e1_segment_metrics.csv")
    pred = load_all(args.root, "e1_predictions.csv")

    expected = {(c, m) for c in CELLS for m in (BASE, OPTICAL)}
    observed = set(zip(fold.held_out_cell, fold.model))
    if observed != expected or len(fold) != 8:
        raise RuntimeError(f"expected 8 exact cell/model rows, got {len(fold)} and keys={sorted(observed)}")
    if fold.held_out_cell_used_for_training.astype(bool).any() or fold.held_out_cell_used_for_selection.astype(bool).any():
        raise RuntimeError("external target leakage flag detected")

    metrics = ["MAE", "RMSE", "R2", "MaxAE", "Q95_AE"]
    summary_rows = []
    for model, g in fold.groupby("model"):
        row = {"model": model, "n_cells": int(len(g))}
        for metric in metrics:
            row[f"{metric}_cell_mean"] = float(g[metric].mean())
            row[f"{metric}_cell_std"] = float(g[metric].std(ddof=1))
        summary_rows.append(row)
    summary = pd.DataFrame(summary_rows)

    paired = fold.pivot(index="held_out_cell", columns="model", values=metrics)
    paired_rows = []
    for c in CELLS:
        row = {"held_out_cell": c}
        for metric in metrics:
            b = float(paired.loc[c, (metric, BASE)])
            o = float(paired.loc[c, (metric, OPTICAL)])
            row[f"{BASE}_{metric}"] = b
            row[f"{OPTICAL}_{metric}"] = o
            row[f"delta_{metric}_optical_minus_base"] = o - b
            if metric in ("MAE", "RMSE", "Q95_AE", "MaxAE"):
                row[f"relative_{metric}_change"] = (o - b) / b if b != 0 else np.nan
        paired_rows.append(row)
    paired_df = pd.DataFrame(paired_rows)

    mean_delta_mae = float(paired_df.delta_MAE_optical_minus_base.mean())
    mean_delta_rmse = float(paired_df.delta_RMSE_optical_minus_base.mean())
    mean_delta_q95 = float(paired_df.delta_Q95_AE_optical_minus_base.mean())
    wins = int((paired_df.delta_MAE_optical_minus_base < 0).sum())
    worst_rel_mae = float(paired_df.relative_MAE_change.max())
    criteria = {
        "mean_cell_MAE_improves": mean_delta_mae < 0,
        "wins_at_least_3_of_4_cells": wins >= 3,
        "mean_cell_RMSE_nonworse": mean_delta_rmse <= 0,
        "mean_cell_Q95_nonworse": mean_delta_q95 <= 0,
        "worst_cell_MAE_increase_le_10pct": worst_rel_mae <= 0.10,
    }
    decision = "SUPPORTS_ROBUST_CROSS_CELL_FBG_BENEFIT" if all(criteria.values()) else "DOES_NOT_SUPPORT_ROBUST_CROSS_CELL_FBG_BENEFIT"
    decision_row = {
        "decision": decision,
        "mean_delta_MAE_optical_minus_base": mean_delta_mae,
        "mean_delta_RMSE_optical_minus_base": mean_delta_rmse,
        "mean_delta_Q95_optical_minus_base": mean_delta_q95,
        "MAE_wins": wins,
        "worst_relative_MAE_change": worst_rel_mae,
        **{f"criterion_{k}": v for k, v in criteria.items()},
        "external_target_used_for_SiC_model_selection": False,
        "external_target_used_for_E1_architecture_search": False,
    }
    decision_df = pd.DataFrame([decision_row])

    rate_summary = rate.groupby(["model", "rate_C"], as_index=False).agg(
        n_cells=("held_out_cell", "nunique"),
        MAE_cell_mean=("MAE", "mean"),
        RMSE_cell_mean=("RMSE", "mean"),
        R2_cell_mean=("R2", "mean"),
        Q95_AE_cell_mean=("Q95_AE", "mean"),
        MaxAE_cell_mean=("MaxAE", "mean"),
    )

    pooled_rows = []
    for model, g in pred.groupby("model"):
        y = g.y_true.to_numpy(dtype=float)
        p_ = g.y_pred.to_numpy(dtype=float)
        ae = np.abs(y - p_)
        sse = np.sum((y - p_) ** 2)
        sst = np.sum((y - y.mean()) ** 2)
        pooled_rows.append({
            "model": model,
            "n_windows": int(len(g)),
            "MAE": float(ae.mean()),
            "RMSE": float(np.sqrt(np.mean((y - p_) ** 2))),
            "R2": float(1.0 - sse / sst),
            "Q95_AE": float(np.quantile(ae, 0.95)),
            "MaxAE": float(ae.max()),
        })
    pooled = pd.DataFrame(pooled_rows)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    fold.to_csv(args.out_dir / "all_e1_fold_metrics.csv", index=False)
    rate.to_csv(args.out_dir / "all_e1_rate_metrics.csv", index=False)
    seg.to_csv(args.out_dir / "all_e1_segment_metrics.csv", index=False)
    summary.to_csv(args.out_dir / "e1_cell_mean_summary.csv", index=False)
    paired_df.to_csv(args.out_dir / "e1_paired_cell_comparison.csv", index=False)
    rate_summary.to_csv(args.out_dir / "e1_rate_summary.csv", index=False)
    pooled.to_csv(args.out_dir / "e1_pooled_window_summary.csv", index=False)
    decision_df.to_csv(args.out_dir / "e1_external_evidence_decision.csv", index=False)
    (args.out_dir / "e1_external_evidence_decision.json").write_text(json.dumps(decision_row, indent=2), encoding="utf-8")

    print("=== cell mean summary ===")
    print(summary.to_string(index=False))
    print("=== paired cell comparison ===")
    print(paired_df.to_string(index=False))
    print("=== rate summary ===")
    print(rate_summary.to_string(index=False))
    print("=== decision ===")
    print(decision_df.to_string(index=False))


if __name__ == "__main__":
    main()
