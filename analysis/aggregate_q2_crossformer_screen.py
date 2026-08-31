from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

EXPECTED_MODELS = (
    "IUW-TCN",
    "CGA-Matched",
    "VIW-Transformer",
    "DualTCN-Transformer",
    "EO-CrossFormer",
    "EO-CrossFormer-TF",
)
EXPECTED_PROFILES = ("HWFET", "LA92", "NEDC", "NYCC", "US06", "WLTC")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--input-dir", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    args = p.parse_args()

    files = sorted(args.input_dir.rglob("profile_metrics_*.csv"))
    if len(files) != 6:
        raise RuntimeError(f"Expected 6 profile files, found {len(files)}: {files}")
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    if len(df) != 36:
        raise RuntimeError(f"Expected 36 model/profile rows, found {len(df)}")
    if set(df["model"].astype(str)) != set(EXPECTED_MODELS):
        raise RuntimeError("Model set mismatch")
    if set(df["held_out_profile"].astype(str)) != set(EXPECTED_PROFILES):
        raise RuntimeError("Profile set mismatch")
    if df.duplicated(["held_out_profile", "model"]).any():
        raise RuntimeError("Duplicate profile/model rows")

    rank = (
        df.groupby("model", as_index=False)
        .agg(
            n_profiles=("MAE", "size"),
            params=("params", "first"),
            MAE_mean=("MAE", "mean"),
            MAE_std=("MAE", "std"),
            RMSE_mean=("RMSE", "mean"),
            R2_mean=("R2", "mean"),
            Q95_AE_mean=("Q95_AE", "mean"),
            MaxAE_mean=("MaxAE", "mean"),
        )
    )
    base = df[df["model"] == "IUW-TCN"][["held_out_profile", "MAE", "RMSE", "Q95_AE"]].rename(
        columns={"MAE": "base_MAE", "RMSE": "base_RMSE", "Q95_AE": "base_Q95_AE"}
    )
    paired = df.merge(base, on="held_out_profile", how="left")
    paired["MAE_win_vs_IUW"] = paired["MAE"] < paired["base_MAE"]
    paired["delta_MAE_vs_IUW"] = paired["MAE"] - paired["base_MAE"]
    wins = paired.groupby("model", as_index=False).agg(
        wins_vs_IUW_TCN=("MAE_win_vs_IUW", "sum"),
        mean_delta_MAE_vs_IUW_TCN=("delta_MAE_vs_IUW", "mean"),
    )
    rank = rank.merge(wins, on="model", how="left").sort_values("MAE_mean")

    def row(name: str) -> pd.Series:
        r = rank[rank["model"] == name]
        if len(r) != 1:
            raise RuntimeError(name)
        return r.iloc[0]

    b = row("IUW-TCN")
    c = row("EO-CrossFormer")
    decision = pd.DataFrame([
        {
            "candidate": "EO-CrossFormer",
            "baseline": "IUW-TCN",
            "criterion_lower_MAE": bool(c.MAE_mean < b.MAE_mean),
            "criterion_lower_RMSE": bool(c.RMSE_mean < b.RMSE_mean),
            "criterion_MAE_wins_ge_4_of_6": bool(int(c.wins_vs_IUW_TCN) >= 4),
            "criterion_Q95_no_worse": bool(c.Q95_AE_mean <= b.Q95_AE_mean),
            "wins_vs_IUW_TCN": int(c.wins_vs_IUW_TCN),
            "candidate_MAE": float(c.MAE_mean),
            "baseline_MAE": float(b.MAE_mean),
            "candidate_RMSE": float(c.RMSE_mean),
            "baseline_RMSE": float(b.RMSE_mean),
            "candidate_Q95_AE": float(c.Q95_AE_mean),
            "baseline_Q95_AE": float(b.Q95_AE_mean),
            "decision": "KEEP" if (
                c.MAE_mean < b.MAE_mean
                and c.RMSE_mean < b.RMSE_mean
                and int(c.wins_vs_IUW_TCN) >= 4
                and c.Q95_AE_mean <= b.Q95_AE_mean
            ) else "DROP",
        }
    ])

    raw = row("EO-CrossFormer")
    tf = row("EO-CrossFormer-TF")
    rep = pd.DataFrame([
        {
            "architecture": "EO-CrossFormer",
            "raw_W_MAE": float(raw.MAE_mean),
            "TF_MAE": float(tf.MAE_mean),
            "raw_W_RMSE": float(raw.RMSE_mean),
            "TF_RMSE": float(tf.RMSE_mean),
            "preferred_representation": "raw_W" if raw.MAE_mean <= tf.MAE_mean else "physics_decoupled_TF",
        }
    ])

    args.out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out_dir / "all_profile_metrics.csv", index=False)
    rank.to_csv(args.out_dir / "screen_ranking.csv", index=False)
    paired.to_csv(args.out_dir / "paired_vs_iuw_tcn.csv", index=False)
    decision.to_csv(args.out_dir / "crossformer_decision.csv", index=False)
    rep.to_csv(args.out_dir / "representation_decision.csv", index=False)

    print("\n=== Final architecture screen ranking ===")
    print(rank.to_string(index=False))
    print("\n=== Pre-registered CrossFormer decision ===")
    print(decision.to_string(index=False))
    print("\n=== Matched raw-W vs T/F ===")
    print(rep.to_string(index=False))


if __name__ == "__main__":
    main()
