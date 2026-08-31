from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

EXPECTED_PROFILES = ("HWFET", "LA92", "NEDC", "NYCC", "US06", "WLTC")
MODELS = ("IUW-TCN", "MC-TCN", "LR-MC-TCN")
CANDIDATES = ("MC-TCN", "LR-MC-TCN")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--input-dir", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    args = p.parse_args()

    files = sorted(args.input_dir.rglob("topconf_source_*.csv"))
    if len(files) != 6:
        raise RuntimeError(f"Expected six fold files, got {len(files)}")
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    if len(df) != 18:
        raise RuntimeError(f"Expected 18 rows, got {len(df)}")
    if set(df.held_out_profile.astype(str)) != set(EXPECTED_PROFILES):
        raise RuntimeError("Profile mismatch")
    if set(df.model.astype(str)) != set(MODELS):
        raise RuntimeError("Model mismatch")
    if df.duplicated(["held_out_profile", "model"]).any():
        raise RuntimeError("Duplicate fold/model rows")
    if not df.source_only.astype(str).str.lower().eq("true").all():
        raise RuntimeError("Non-source-only row detected")
    if df.two_c_metrics_used.astype(str).str.lower().eq("true").any():
        raise RuntimeError("2C metric leakage detected")

    summary = df.groupby("model", as_index=False).agg(
        n_profiles=("MAE", "size"),
        params=("params", "first"),
        MAE_mean=("MAE", "mean"), MAE_std=("MAE", "std"),
        RMSE_mean=("RMSE", "mean"), RMSE_std=("RMSE", "std"),
        R2_mean=("R2", "mean"),
        Q95_AE_mean=("Q95_AE", "mean"), MaxAE_mean=("MaxAE", "mean"),
    )
    base = summary[summary.model == "IUW-TCN"].iloc[0]

    paired_rows = []
    decision_rows = []
    for candidate in CANDIDATES:
        deltas = []
        wins = 0
        for profile in EXPECTED_PROFILES:
            b = df[(df.model == "IUW-TCN") & (df.held_out_profile == profile)].iloc[0]
            c = df[(df.model == candidate) & (df.held_out_profile == profile)].iloc[0]
            delta = float(c.MAE - b.MAE)
            win = bool(c.MAE < b.MAE)
            wins += int(win)
            deltas.append(delta)
            paired_rows.append({
                "candidate": candidate,
                "held_out_profile": profile,
                "IUW_MAE": float(b.MAE),
                "candidate_MAE": float(c.MAE),
                "delta_MAE_candidate_minus_IUW": delta,
                "candidate_MAE_win": win,
                "IUW_RMSE": float(b.RMSE),
                "candidate_RMSE": float(c.RMSE),
                "IUW_Q95_AE": float(b.Q95_AE),
                "candidate_Q95_AE": float(c.Q95_AE),
            })
        csum = summary[summary.model == candidate].iloc[0]
        s = pd.Series(deltas)
        criteria = {
            "criterion_mean_delta_MAE_lt_0": bool(s.mean() < 0),
            "criterion_median_delta_MAE_lt_0": bool(s.median() < 0),
            "criterion_wins_ge_4_of_6": bool(wins >= 4),
            "criterion_mean_RMSE_lower": bool(csum.RMSE_mean < base.RMSE_mean),
            "criterion_mean_Q95_no_worse": bool(csum.Q95_AE_mean <= base.Q95_AE_mean),
        }
        decision_rows.append({
            "candidate": candidate,
            "baseline": "IUW-TCN",
            "mean_delta_MAE": float(s.mean()),
            "median_delta_MAE": float(s.median()),
            "MAE_wins": wins,
            "baseline_MAE_mean": float(base.MAE_mean),
            "candidate_MAE_mean": float(csum.MAE_mean),
            "baseline_RMSE_mean": float(base.RMSE_mean),
            "candidate_RMSE_mean": float(csum.RMSE_mean),
            "baseline_Q95_AE_mean": float(base.Q95_AE_mean),
            "candidate_Q95_AE_mean": float(csum.Q95_AE_mean),
            **criteria,
            "decision": "KEEP" if all(criteria.values()) else "DROP",
        })

    decisions = pd.DataFrame(decision_rows)
    kept = decisions[decisions.decision == "KEEP"].copy()
    if len(kept):
        winner = kept.sort_values(["candidate_MAE_mean", "candidate_Q95_AE_mean"]).iloc[0].candidate
        final = pd.DataFrame([{"selected_model": winner, "architecture_extension_decision": "ADVANCE"}])
    else:
        final = pd.DataFrame([{"selected_model": "IUW-TCN", "architecture_extension_decision": "CLOSE"}])

    args.out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out_dir / "all_topconf_source_metrics.csv", index=False)
    summary.to_csv(args.out_dir / "topconf_source_summary.csv", index=False)
    pd.DataFrame(paired_rows).to_csv(args.out_dir / "paired_vs_iuw.csv", index=False)
    decisions.to_csv(args.out_dir / "candidate_decisions.csv", index=False)
    final.to_csv(args.out_dir / "topconf_extension_decision.csv", index=False)

    print("=== summary ===")
    print(summary.sort_values("MAE_mean").to_string(index=False))
    print("=== candidate decisions ===")
    print(decisions.to_string(index=False))
    print("=== final ===")
    print(final.to_string(index=False))


if __name__ == "__main__":
    main()
