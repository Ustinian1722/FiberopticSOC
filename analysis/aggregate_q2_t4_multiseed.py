from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

BASELINES = ("VI-MS-TCN", "IUTF-StackedTCN", "DualBranch-Direct-TF")
TARGET = "ETMF-TF"


def one_sided_t_greater(x: np.ndarray) -> float:
    if len(x) < 2:
        return float("nan")
    res = stats.ttest_1samp(x, popmean=0.0, alternative="greater")
    return float(res.pvalue)


def wilcoxon_greater(x: np.ndarray) -> float:
    x = x[np.isfinite(x)]
    x = x[np.abs(x) > 1e-15]
    if len(x) == 0:
        return 1.0
    try:
        return float(stats.wilcoxon(x, alternative="greater", zero_method="wilcox").pvalue)
    except ValueError:
        return float("nan")


def seed_cluster_bootstrap(
    paired: pd.DataFrame,
    gain_col: str,
    *,
    n_boot: int = 20000,
    seed: int = 20260831,
) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    seeds = np.array(sorted(paired["seed"].unique()))
    if len(seeds) == 0:
        return float("nan"), float("nan")
    values = []
    by_seed = {s: paired.loc[paired["seed"] == s, gain_col].to_numpy(float) for s in seeds}
    for _ in range(n_boot):
        sampled = rng.choice(seeds, size=len(seeds), replace=True)
        pooled = np.concatenate([by_seed[s] for s in sampled])
        values.append(float(np.mean(pooled)))
    return float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))


def paired_stats(metrics: pd.DataFrame) -> pd.DataFrame:
    rows = []
    scopes = [("all", metrics)] + [
        (direction, metrics[metrics["direction"] == direction])
        for direction in sorted(metrics["direction"].unique())
    ]
    for scope, sdf in scopes:
        target = sdf[sdf["model"] == TARGET][
            ["direction", "held_out_profile", "seed", "MAE"]
        ].rename(columns={"MAE": "target_MAE"})
        for baseline in BASELINES:
            base = sdf[sdf["model"] == baseline][
                ["direction", "held_out_profile", "seed", "MAE"]
            ].rename(columns={"MAE": "baseline_MAE"})
            p = target.merge(base, on=["direction", "held_out_profile", "seed"], how="inner")
            p["gain"] = p["baseline_MAE"] - p["target_MAE"]
            gains = p["gain"].to_numpy(float)
            lo, hi = seed_cluster_bootstrap(p, "gain")
            rows.append(
                {
                    "scope": scope,
                    "target": TARGET,
                    "baseline": baseline,
                    "n_pairs": len(p),
                    "n_seeds": int(p["seed"].nunique()),
                    "mean_MAE_gain": float(np.mean(gains)),
                    "median_MAE_gain": float(np.median(gains)),
                    "relative_gain_vs_baseline": float(
                        np.mean(gains) / np.mean(p["baseline_MAE"].to_numpy(float))
                    ),
                    "wins": int(np.sum(gains > 1e-15)),
                    "ties": int(np.sum(np.abs(gains) <= 1e-15)),
                    "losses": int(np.sum(gains < -1e-15)),
                    "one_sided_paired_t_p": one_sided_t_greater(gains),
                    "one_sided_wilcoxon_p": wilcoxon_greater(gains),
                    "seed_cluster_bootstrap95_lo": lo,
                    "seed_cluster_bootstrap95_hi": hi,
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--input-root", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    args = p.parse_args()

    files = sorted(args.input_root.rglob("split_metrics.csv"))
    if not files:
        raise FileNotFoundError(f"No split_metrics.csv under {args.input_root}")
    frames = [pd.read_csv(f) for f in files]
    metrics = pd.concat(frames, ignore_index=True)
    required_models = set(BASELINES) | {TARGET}
    if set(metrics["model"].unique()) != required_models:
        raise ValueError(
            f"Unexpected model set: {sorted(metrics['model'].unique())}; expected={sorted(required_models)}"
        )

    key = ["direction", "held_out_profile", "model", "seed"]
    if metrics.duplicated(key).any():
        raise ValueError("Duplicate final T4 metric rows")

    across_seed = (
        metrics.groupby(["direction", "model"], as_index=False)
        .agg(
            n_rows=("MAE", "size"),
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
    profile_summary = (
        metrics.groupby(["direction", "held_out_profile", "model"], as_index=False)
        .agg(
            n_seeds=("seed", "nunique"),
            MAE_mean=("MAE", "mean"),
            MAE_std=("MAE", "std"),
            RMSE_mean=("RMSE", "mean"),
        )
    )
    stats_df = paired_stats(metrics)

    winners = metrics.loc[
        metrics.groupby(["direction", "held_out_profile", "seed"])["MAE"].idxmin(),
        ["direction", "held_out_profile", "seed", "model", "MAE"],
    ].rename(columns={"model": "winner_model", "MAE": "winner_MAE"})
    winner_counts = (
        winners.groupby(["direction", "winner_model"], as_index=False)
        .size()
        .rename(columns={"size": "wins"})
    )

    per_seed_rows = []
    for seed, sdf in metrics.groupby("seed"):
        for direction in sorted(sdf["direction"].unique()):
            d = sdf[sdf["direction"] == direction]
            target = float(d[d["model"] == TARGET]["MAE"].mean())
            row = {"seed": int(seed), "direction": direction, "ETMF_MAE": target}
            for baseline in BASELINES:
                b = float(d[d["model"] == baseline]["MAE"].mean())
                row[f"{baseline}_MAE"] = b
                row[f"gain_vs_{baseline}"] = b - target
            per_seed_rows.append(row)
    per_seed = pd.DataFrame(per_seed_rows)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(args.out_dir / "all_seed_split_metrics.csv", index=False)
    across_seed.to_csv(args.out_dir / "across_seed_summary.csv", index=False)
    profile_summary.to_csv(args.out_dir / "profile_across_seed_summary.csv", index=False)
    stats_df.to_csv(args.out_dir / "paired_statistics.csv", index=False)
    winners.to_csv(args.out_dir / "winner_by_seed_profile.csv", index=False)
    winner_counts.to_csv(args.out_dir / "winner_counts.csv", index=False)
    per_seed.to_csv(args.out_dir / "per_seed_gain.csv", index=False)

    print("=== across seed ===")
    print(across_seed.sort_values(["direction", "MAE_mean"]).to_string(index=False))
    print("\n=== paired ETMF statistics ===")
    print(stats_df.to_string(index=False))
    print("\n=== winners ===")
    print(winner_counts.to_string(index=False))


if __name__ == "__main__":
    main()
