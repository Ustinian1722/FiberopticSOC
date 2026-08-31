from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def equal_split_summary(df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    return (
        df.groupby(group_cols, as_index=False)
        .agg(
            n=("MAE", "size"),
            MAE_mean=("MAE", "mean"),
            MAE_std=("MAE", "std"),
            RMSE_mean=("RMSE", "mean"),
            R2_mean=("R2", "mean"),
            Q95_AE_mean=("Q95_AE", "mean"),
            MaxAE_mean=("MaxAE", "mean"),
        )
        .sort_values(group_cols[:-1] + ["MAE_mean"] if len(group_cols) > 1 else ["MAE_mean"])
    )


def seed_cluster_ci(paired: pd.DataFrame, diff_col: str, reps: int, seed: int) -> tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    seeds = np.array(sorted(paired.seed.unique()))
    by_seed = {s: paired.loc[paired.seed == s, diff_col].to_numpy(float) for s in seeds}
    obs = float(np.mean(np.concatenate([by_seed[s] for s in seeds])))
    boot = np.empty(reps, dtype=float)
    for i in range(reps):
        sampled = rng.choice(seeds, size=len(seeds), replace=True)
        boot[i] = float(np.mean(np.concatenate([by_seed[s] for s in sampled])))
    lo, hi = np.quantile(boot, [0.025, 0.975])
    return obs, float(lo), float(hi)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--benchmark-root", type=Path, required=True)
    p.add_argument("--ablation-root", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--bootstrap", type=int, default=20000)
    p.add_argument("--seed", type=int, default=20260831)
    args = p.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    bp = sorted(args.benchmark_root.rglob("split_metrics.csv"))
    ap = sorted(args.ablation_root.rglob("split_metrics.csv"))
    if not bp or not ap:
        raise RuntimeError(f"Missing metrics: benchmark={len(bp)} ablation={len(ap)}")

    b = pd.concat([pd.read_csv(x) for x in bp], ignore_index=True)
    a = pd.concat([pd.read_csv(x) for x in ap], ignore_index=True)
    b.to_csv(args.out_dir / "benchmark_all_split_metrics.csv", index=False)
    a.to_csv(args.out_dir / "ablation_all_seed_split_metrics.csv", index=False)

    bdir = equal_split_summary(b, ["direction", "model"])
    ball = equal_split_summary(b.assign(direction="ALL"), ["direction", "model"])
    pd.concat([bdir, ball], ignore_index=True).to_csv(args.out_dir / "benchmark_summary.csv", index=False)

    adir = equal_split_summary(a, ["direction", "model"])
    aall = equal_split_summary(a.assign(direction="ALL"), ["direction", "model"])
    pd.concat([adir, aall], ignore_index=True).to_csv(args.out_dir / "ablation_summary.csv", index=False)

    key = ["direction", "seed", "held_out_profile"]
    wide = a.pivot_table(index=key, columns="model", values="MAE", aggfunc="first").reset_index()
    required = {"VI", "VI+TF", "VI+W"}
    if not required.issubset(wide.columns):
        raise RuntimeError(f"Ablation missing models: {sorted(required.difference(wide.columns))}")
    wide["gain_W_vs_VI"] = wide["VI"] - wide["VI+W"]
    wide["gain_W_vs_TF"] = wide["VI+TF"] - wide["VI+W"]
    wide.to_csv(args.out_dir / "ablation_paired_seed_profile.csv", index=False)

    paired_rows = []
    for direction in list(sorted(wide.direction.unique())) + ["ALL"]:
        g = wide if direction == "ALL" else wide[wide.direction == direction]
        for label, col in (("VI+W_vs_VI", "gain_W_vs_VI"), ("VI+W_vs_VI+TF", "gain_W_vs_TF")):
            vals = g[col].to_numpy(float)
            obs, lo, hi = seed_cluster_ci(g, col, args.bootstrap, args.seed)
            paired_rows.append(
                {
                    "direction": direction,
                    "comparison": label,
                    "n_pairs": len(vals),
                    "n_seeds": g.seed.nunique(),
                    "mean_MAE_gain": float(vals.mean()),
                    "median_MAE_gain": float(np.median(vals)),
                    "wins": int((vals > 0).sum()),
                    "ties": int((vals == 0).sum()),
                    "losses": int((vals < 0).sum()),
                    "seed_cluster_bootstrap_gain_ci95_low": lo,
                    "seed_cluster_bootstrap_gain_ci95_high": hi,
                    "bootstrap_observed_gain": obs,
                }
            )
    paired = pd.DataFrame(paired_rows)
    paired.to_csv(args.out_dir / "ablation_paired_summary.csv", index=False)

    # Compact percentage tables intended for direct manuscript drafting.
    bt = pd.concat([bdir, ball], ignore_index=True).copy()
    at = pd.concat([adir, aall], ignore_index=True).copy()
    for frame in (bt, at):
        for c in ("MAE_mean", "MAE_std", "RMSE_mean", "Q95_AE_mean", "MaxAE_mean"):
            frame[c + "_pct"] = 100.0 * frame[c]
    bt.to_csv(args.out_dir / "manuscript_backbone_table.csv", index=False)
    at.to_csv(args.out_dir / "manuscript_ablation_table.csv", index=False)

    print("=== Backbone benchmark ===")
    print(bt[["direction","model","MAE_mean_pct","RMSE_mean_pct","R2_mean","Q95_AE_mean_pct"]].to_string(index=False))
    print("\n=== Five-seed input ablation ===")
    print(at[["direction","model","MAE_mean_pct","RMSE_mean_pct","R2_mean","Q95_AE_mean_pct"]].to_string(index=False))
    print("\n=== Paired ablation evidence ===")
    print(paired.to_string(index=False))


if __name__ == "__main__":
    main()
