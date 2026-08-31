from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

PROFILES = ("HWFET", "LA92", "NEDC", "NYCC", "US06", "WLTC")
MODELS = ("IUW-TCN", "MD-ResTCN")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--input-dir", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    args = p.parse_args()

    files = sorted(args.input_dir.rglob("modality_delay_*.csv"))
    if len(files) != 6:
        raise RuntimeError(f"Expected 6 fold files, got {len(files)}: {files}")
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    if len(df) != 12:
        raise RuntimeError(f"Expected 12 rows, got {len(df)}")
    if set(df.held_out_profile.astype(str)) != set(PROFILES):
        raise RuntimeError("Profile set mismatch")
    if set(df.model.astype(str)) != set(MODELS):
        raise RuntimeError("Model set mismatch")
    if df.duplicated(["held_out_profile", "model"]).any():
        raise RuntimeError("Duplicate fold/model rows")
    if not df.source_only.astype(str).str.lower().eq("true").all():
        raise RuntimeError("Non-source-only row detected")
    if df.two_c_metrics_used.astype(str).str.lower().eq("true").any():
        raise RuntimeError("2C metric use detected")

    summary = df.groupby("model", as_index=False).agg(
        n_profiles=("MAE", "size"), params=("params", "first"),
        MAE_mean=("MAE", "mean"), MAE_std=("MAE", "std"),
        RMSE_mean=("RMSE", "mean"), RMSE_std=("RMSE", "std"),
        R2_mean=("R2", "mean"), Q95_AE_mean=("Q95_AE", "mean"),
        MaxAE_mean=("MaxAE", "mean"),
    )

    paired = []
    deltas = []
    wins = 0
    for profile in PROFILES:
        b = df[(df.held_out_profile == profile) & (df.model == "IUW-TCN")].iloc[0]
        c = df[(df.held_out_profile == profile) & (df.model == "MD-ResTCN")].iloc[0]
        d = float(c.MAE - b.MAE)
        w = bool(c.MAE < b.MAE)
        deltas.append(d)
        wins += int(w)
        paired.append({
            "held_out_profile": profile,
            "IUW_MAE": float(b.MAE), "MD_MAE": float(c.MAE),
            "delta_MAE_MD_minus_IUW": d, "MD_MAE_win": w,
            "IUW_RMSE": float(b.RMSE), "MD_RMSE": float(c.RMSE),
            "IUW_Q95_AE": float(b.Q95_AE), "MD_Q95_AE": float(c.Q95_AE),
        })

    d = pd.Series(deltas)
    base = summary[summary.model == "IUW-TCN"].iloc[0]
    cand = summary[summary.model == "MD-ResTCN"].iloc[0]
    criteria = {
        "criterion_mean_delta_MAE_lt_0": bool(d.mean() < 0),
        "criterion_median_delta_MAE_lt_0": bool(d.median() < 0),
        "criterion_wins_ge_4_of_6": bool(wins >= 4),
        "criterion_mean_RMSE_lower": bool(cand.RMSE_mean < base.RMSE_mean),
        "criterion_mean_Q95_no_worse": bool(cand.Q95_AE_mean <= base.Q95_AE_mean),
    }
    keep = all(criteria.values())
    decision = pd.DataFrame([{
        "candidate": "MD-ResTCN", "baseline": "IUW-TCN",
        "mean_delta_MAE": float(d.mean()), "median_delta_MAE": float(d.median()),
        "MAE_wins": wins,
        "baseline_MAE_mean": float(base.MAE_mean), "candidate_MAE_mean": float(cand.MAE_mean),
        "baseline_RMSE_mean": float(base.RMSE_mean), "candidate_RMSE_mean": float(cand.RMSE_mean),
        "baseline_Q95_AE_mean": float(base.Q95_AE_mean), "candidate_Q95_AE_mean": float(cand.Q95_AE_mean),
        **criteria,
        "source_only": True, "two_c_metrics_used": False,
        "decision": "KEEP" if keep else "DROP",
    }])

    args.out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out_dir / "all_modality_delay_metrics.csv", index=False)
    summary.to_csv(args.out_dir / "modality_delay_summary.csv", index=False)
    pd.DataFrame(paired).to_csv(args.out_dir / "paired_vs_iuw.csv", index=False)
    decision.to_csv(args.out_dir / "modality_delay_decision.csv", index=False)
    print(summary.to_string(index=False))
    print(pd.DataFrame(paired).to_string(index=False))
    print(decision.to_string(index=False))


if __name__ == "__main__":
    main()
