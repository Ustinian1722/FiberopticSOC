from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


MAIN_SELECTOR = "NestedCal99-VI-or-W-white"
BASELINE = "VI"


def seed_cluster_bootstrap_gain(
    paired: pd.DataFrame,
    *,
    baseline_col: str,
    method_col: str,
    n_boot: int,
    seed: int,
) -> tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    seeds = np.array(sorted(paired["seed"].unique()))
    by_seed = {
        s: paired.loc[paired["seed"] == s, baseline_col].to_numpy()
        - paired.loc[paired["seed"] == s, method_col].to_numpy()
        for s in seeds
    }
    observed = float(
        np.mean(np.concatenate([by_seed[s] for s in seeds]))
    )
    boot = np.empty(n_boot, dtype=np.float64)
    for i in range(n_boot):
        sampled = rng.choice(seeds, size=len(seeds), replace=True)
        boot[i] = np.mean(np.concatenate([by_seed[s] for s in sampled]))
    lo, hi = np.quantile(boot, [0.025, 0.975])
    return observed, float(lo), float(hi)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--input-root", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--bootstrap-reps", type=int, default=20000)
    p.add_argument("--bootstrap-seed", type=int, default=20260831)
    args = p.parse_args()

    paths = sorted(args.input_root.rglob("split_metrics.csv"))
    if not paths:
        raise RuntimeError(f"No split_metrics.csv under {args.input_root}")
    df = pd.concat([pd.read_csv(path) for path in paths], ignore_index=True)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out_dir / "all_seed_split_metrics.csv", index=False)

    across = (
        df.groupby(["direction", "model"], as_index=False)
        .agg(
            n=("MAE", "size"),
            n_seeds=("seed", "nunique"),
            MAE_mean=("MAE", "mean"),
            MAE_std=("MAE", "std"),
            RMSE_mean=("RMSE", "mean"),
            RMSE_std=("RMSE", "std"),
            R2_mean=("R2", "mean"),
            Q95_AE_mean=("Q95_AE", "mean"),
            MaxAE_mean=("MaxAE", "mean"),
        )
    )
    across.to_csv(args.out_dir / "across_seed_summary.csv", index=False)

    profile = (
        df.groupby(["direction", "held_out_profile", "model"], as_index=False)
        .agg(
            n_seeds=("seed", "nunique"),
            MAE_mean=("MAE", "mean"),
            MAE_std=("MAE", "std"),
            RMSE_mean=("RMSE", "mean"),
            R2_mean=("R2", "mean"),
        )
    )
    profile.to_csv(args.out_dir / "profile_across_seed_summary.csv", index=False)

    key = ["direction", "seed", "held_out_profile"]
    wide = df.pivot_table(index=key, columns="model", values="MAE", aggfunc="first").reset_index()
    if BASELINE not in wide.columns or MAIN_SELECTOR not in wide.columns:
        raise RuntimeError(f"Need {BASELINE} and {MAIN_SELECTOR} in final metrics")
    wide["gain_VI_minus_selector"] = wide[BASELINE] - wide[MAIN_SELECTOR]
    wide["selector_beats_VI"] = wide["gain_VI_minus_selector"] > 0
    wide.to_csv(args.out_dir / "paired_seed_profile.csv", index=False)

    stat_rows = []
    seed_rows = []
    winner_rows = []
    for direction, g in wide.groupby("direction"):
        diff = g["gain_VI_minus_selector"].dropna().to_numpy(dtype=np.float64)
        t = stats.ttest_1samp(diff, 0.0, alternative="greater")
        try:
            w = stats.wilcoxon(diff, alternative="greater", zero_method="wilcox")
            wilcox_stat, wilcox_p = float(w.statistic), float(w.pvalue)
        except ValueError:
            wilcox_stat, wilcox_p = np.nan, np.nan
        obs, lo, hi = seed_cluster_bootstrap_gain(
            g,
            baseline_col=BASELINE,
            method_col=MAIN_SELECTOR,
            n_boot=args.bootstrap_reps,
            seed=args.bootstrap_seed,
        )
        stat_rows.append(
            {
                "direction": direction,
                "baseline": BASELINE,
                "method": MAIN_SELECTOR,
                "n_seed_profile_pairs": len(diff),
                "n_independent_seeds": g["seed"].nunique(),
                "mean_gain": float(np.mean(diff)),
                "median_gain": float(np.median(diff)),
                "wins": int(np.sum(diff > 0)),
                "ties": int(np.sum(diff == 0)),
                "losses": int(np.sum(diff < 0)),
                "paired_t_stat_one_sided": float(t.statistic),
                "paired_t_p_one_sided": float(t.pvalue),
                "wilcoxon_stat_one_sided": wilcox_stat,
                "wilcoxon_p_one_sided": wilcox_p,
                "seed_cluster_bootstrap_mean_gain": obs,
                "seed_cluster_bootstrap_ci95_low": lo,
                "seed_cluster_bootstrap_ci95_high": hi,
            }
        )
        for seed_value, gs in g.groupby("seed"):
            d = gs["gain_VI_minus_selector"].to_numpy(dtype=np.float64)
            seed_rows.append(
                {
                    "direction": direction,
                    "seed": seed_value,
                    "mean_gain": float(np.mean(d)),
                    "median_gain": float(np.median(d)),
                    "wins": int(np.sum(d > 0)),
                    "losses": int(np.sum(d < 0)),
                }
            )

    pd.DataFrame(stat_rows).to_csv(args.out_dir / "paired_statistics.csv", index=False)
    pd.DataFrame(seed_rows).to_csv(args.out_dir / "per_seed_gain.csv", index=False)

    for direction, gd in df.groupby("direction"):
        for (seed_value, profile_value), gsp in gd.groupby(["seed", "held_out_profile"]):
            best = gsp.sort_values(["MAE", "model"], kind="stable").iloc[0]
            winner_rows.append(
                {
                    "direction": direction,
                    "seed": seed_value,
                    "held_out_profile": profile_value,
                    "winner": best["model"],
                    "winner_MAE": float(best["MAE"]),
                }
            )
    winner = pd.DataFrame(winner_rows)
    winner.to_csv(args.out_dir / "winner_by_seed_profile.csv", index=False)
    winner_counts = (
        winner.groupby(["direction", "winner"], as_index=False)
        .size()
        .rename(columns={"size": "wins"})
    )
    winner_counts.to_csv(args.out_dir / "winner_counts.csv", index=False)

    print("=== Across-seed summary ===")
    print(across.sort_values(["direction", "MAE_mean"]).to_string(index=False))
    print("\n=== Paired selector-vs-VI statistics ===")
    print(pd.DataFrame(stat_rows).to_string(index=False))
    print("\n=== Winner counts ===")
    print(winner_counts.sort_values(["direction", "wins"], ascending=[True, False]).to_string(index=False))


if __name__ == "__main__":
    main()
