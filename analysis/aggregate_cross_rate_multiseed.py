from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(args.input_root.rglob("split_metrics.csv"))
    if not files:
        raise FileNotFoundError(f"No split_metrics.csv found below {args.input_root}")

    frames = []
    for path in files:
        df = pd.read_csv(path)
        if "seed" not in df.columns:
            raise ValueError(f"Missing seed column: {path}")
        frames.append(df)
    all_df = pd.concat(frames, ignore_index=True)
    all_df.to_csv(args.out_dir / "all_seed_split_metrics.csv", index=False)

    per_seed = (
        all_df.groupby(["seed", "direction", "model"], as_index=False)
        .agg(
            n_profiles=("held_out_profile", "nunique"),
            MAE_mean=("MAE", "mean"),
            RMSE_mean=("RMSE", "mean"),
            R2_mean=("R2", "mean"),
            Q95_AE_mean=("Q95_AE", "mean"),
        )
    )
    per_seed.to_csv(args.out_dir / "per_seed_summary.csv", index=False)

    across = (
        per_seed.groupby(["direction", "model"], as_index=False)
        .agg(
            n_seeds=("seed", "nunique"),
            MAE_mean=("MAE_mean", "mean"),
            MAE_std_across_seeds=("MAE_mean", lambda x: float(np.std(x, ddof=1)) if len(x) > 1 else np.nan),
            RMSE_mean=("RMSE_mean", "mean"),
            R2_mean=("R2_mean", "mean"),
            Q95_AE_mean=("Q95_AE_mean", "mean"),
        )
    )
    across.to_csv(args.out_dir / "across_seed_summary.csv", index=False)

    win_rows = []
    for (direction, held_out, model), g in all_df.groupby(
        ["direction", "held_out_profile", "model"], observed=True
    ):
        win_rows.append(
            {
                "direction": direction,
                "held_out_profile": held_out,
                "model": model,
                "n_seeds": int(g["seed"].nunique()),
                "MAE_mean": float(g["MAE"].mean()),
                "MAE_std": float(g["MAE"].std(ddof=1)) if len(g) > 1 else np.nan,
            }
        )
    profile_summary = pd.DataFrame(win_rows)
    profile_summary.to_csv(args.out_dir / "profile_across_seed_summary.csv", index=False)

    winners = []
    for (seed, direction, held_out), g in all_df.groupby(
        ["seed", "direction", "held_out_profile"], observed=True
    ):
        best = g.sort_values("MAE", kind="stable").iloc[0]
        winners.append(
            {
                "seed": seed,
                "direction": direction,
                "held_out_profile": held_out,
                "winner": best["model"],
                "winner_MAE": best["MAE"],
            }
        )
    winners_df = pd.DataFrame(winners)
    winners_df.to_csv(args.out_dir / "winner_by_seed_profile.csv", index=False)

    counts = (
        winners_df.groupby(["direction", "winner"], as_index=False)
        .size()
        .rename(columns={"size": "wins"})
    )
    counts.to_csv(args.out_dir / "winner_counts.csv", index=False)

    print("=== Across-seed summary ===")
    print(across.sort_values(["direction", "MAE_mean"]).to_string(index=False))
    print("\n=== Winner counts ===")
    print(counts.sort_values(["direction", "wins"], ascending=[True, False]).to_string(index=False))


if __name__ == "__main__":
    main()
