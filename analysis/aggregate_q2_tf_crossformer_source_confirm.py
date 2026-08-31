from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

EXPECTED_PROFILES = ("HWFET", "LA92", "NEDC", "NYCC", "US06", "WLTC")
EXPECTED_MODELS = ("IUW-TCN", "EO-CrossFormer-TF")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--input-dir", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    args = p.parse_args()

    files = sorted(args.input_dir.rglob("source_confirm_*.csv"))
    if len(files) != 6:
        raise RuntimeError(f"Expected 6 source-confirm files, found {len(files)}: {files}")
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    if len(df) != 12:
        raise RuntimeError(f"Expected 12 rows, found {len(df)}")
    if set(df["held_out_profile"].astype(str)) != set(EXPECTED_PROFILES):
        raise RuntimeError("Profile set mismatch")
    if set(df["model"].astype(str)) != set(EXPECTED_MODELS):
        raise RuntimeError("Model set mismatch")
    if df.duplicated(["held_out_profile", "model"]).any():
        raise RuntimeError("Duplicate fold/model rows")
    if not df["source_only"].astype(str).str.lower().eq("true").all():
        raise RuntimeError("Non-source-only row detected")
    if df["two_c_loaded_for_decision"].astype(str).str.lower().eq("true").any():
        raise RuntimeError("2C decision use detected")

    wide = df.pivot(index="held_out_profile", columns="model", values=["MAE", "RMSE", "Q95_AE", "MaxAE", "R2"])
    paired_rows = []
    for profile in EXPECTED_PROFILES:
        paired_rows.append(
            {
                "held_out_profile": profile,
                "IUW_MAE": float(wide.loc[profile, ("MAE", "IUW-TCN")]),
                "TF_MAE": float(wide.loc[profile, ("MAE", "EO-CrossFormer-TF")]),
                "delta_MAE_TF_minus_IUW": float(wide.loc[profile, ("MAE", "EO-CrossFormer-TF")] - wide.loc[profile, ("MAE", "IUW-TCN")]),
                "TF_MAE_win": bool(wide.loc[profile, ("MAE", "EO-CrossFormer-TF")] < wide.loc[profile, ("MAE", "IUW-TCN")]),
                "IUW_RMSE": float(wide.loc[profile, ("RMSE", "IUW-TCN")]),
                "TF_RMSE": float(wide.loc[profile, ("RMSE", "EO-CrossFormer-TF")]),
                "IUW_Q95_AE": float(wide.loc[profile, ("Q95_AE", "IUW-TCN")]),
                "TF_Q95_AE": float(wide.loc[profile, ("Q95_AE", "EO-CrossFormer-TF")]),
            }
        )
    paired = pd.DataFrame(paired_rows)

    summary = (
        df.groupby("model", as_index=False)
        .agg(
            n_profiles=("MAE", "size"),
            params=("params", "first"),
            MAE_mean=("MAE", "mean"),
            MAE_std=("MAE", "std"),
            RMSE_mean=("RMSE", "mean"),
            RMSE_std=("RMSE", "std"),
            R2_mean=("R2", "mean"),
            Q95_AE_mean=("Q95_AE", "mean"),
            MaxAE_mean=("MaxAE", "mean"),
        )
    )

    mean_delta = float(paired["delta_MAE_TF_minus_IUW"].mean())
    median_delta = float(paired["delta_MAE_TF_minus_IUW"].median())
    wins = int(paired["TF_MAE_win"].sum())
    iuw = summary[summary["model"] == "IUW-TCN"].iloc[0]
    tf = summary[summary["model"] == "EO-CrossFormer-TF"].iloc[0]

    criteria = {
        "criterion_mean_delta_MAE_lt_0": bool(mean_delta < 0),
        "criterion_median_delta_MAE_lt_0": bool(median_delta < 0),
        "criterion_TF_wins_ge_4_of_6": bool(wins >= 4),
        "criterion_TF_mean_RMSE_lower": bool(tf.RMSE_mean < iuw.RMSE_mean),
        "criterion_TF_mean_Q95_no_worse": bool(tf.Q95_AE_mean <= iuw.Q95_AE_mean),
    }
    keep = all(criteria.values())
    decision = pd.DataFrame([
        {
            "candidate": "EO-CrossFormer-TF",
            "baseline": "IUW-TCN",
            "mean_delta_MAE_TF_minus_IUW": mean_delta,
            "median_delta_MAE_TF_minus_IUW": median_delta,
            "TF_MAE_wins": wins,
            "IUW_MAE_mean": float(iuw.MAE_mean),
            "TF_MAE_mean": float(tf.MAE_mean),
            "IUW_RMSE_mean": float(iuw.RMSE_mean),
            "TF_RMSE_mean": float(tf.RMSE_mean),
            "IUW_Q95_AE_mean": float(iuw.Q95_AE_mean),
            "TF_Q95_AE_mean": float(tf.Q95_AE_mean),
            **criteria,
            "source_only": True,
            "two_c_metrics_used": False,
            "decision": "KEEP" if keep else "DROP",
        }
    ])

    args.out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out_dir / "all_source_confirm_metrics.csv", index=False)
    paired.to_csv(args.out_dir / "paired_source_confirm.csv", index=False)
    summary.to_csv(args.out_dir / "source_confirm_summary.csv", index=False)
    decision.to_csv(args.out_dir / "tf_crossformer_source_decision.csv", index=False)

    print("\n=== source-side confirmation summary ===")
    print(summary.to_string(index=False))
    print("\n=== paired folds ===")
    print(paired.to_string(index=False))
    print("\n=== preregistered decision ===")
    print(decision.to_string(index=False))


if __name__ == "__main__":
    main()
