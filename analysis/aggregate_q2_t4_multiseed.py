from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

FROZEN_MODEL = "IUW-TCN"
EXPECTED_DIRECTIONS = ("1C_to_2C", "2C_to_1C")
EXPECTED_PROFILES = ("HWFET", "LA92", "NEDC", "NYCC", "US06", "WLTC")
EXPECTED_SEEDS = (0, 1, 2, 3, 4)
EXPECTED_SIGMA_PM = (0.5, 1.0, 2.0)
EXPECTED_NOISE_DRAWS = 5


def seed_cluster_bootstrap_mean(
    df: pd.DataFrame,
    value_col: str,
    *,
    n_boot: int = 20000,
    seed: int = 20260831,
) -> tuple[float, float]:
    """Bootstrap seeds as clusters, preserving all profile/direction rows within a seed."""
    rng = np.random.default_rng(seed)
    seeds = np.asarray(sorted(df["seed"].unique()), dtype=int)
    if len(seeds) == 0:
        return float("nan"), float("nan")
    by_seed = {
        int(s): df.loc[df["seed"] == s, value_col].to_numpy(dtype=float)
        for s in seeds
    }
    vals = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        sampled = rng.choice(seeds, size=len(seeds), replace=True)
        pooled = np.concatenate([by_seed[int(s)] for s in sampled])
        vals[i] = float(np.mean(pooled))
    return float(np.quantile(vals, 0.025)), float(np.quantile(vals, 0.975))


def validate_clean_metrics(metrics: pd.DataFrame) -> None:
    required = {
        "direction", "held_out_profile", "model", "feature_mode", "seed",
        "MAE", "RMSE", "R2", "Q95_AE", "MaxAE", "selected_epoch",
    }
    missing = required.difference(metrics.columns)
    if missing:
        raise ValueError(f"Missing clean metric columns: {sorted(missing)}")
    if set(metrics["model"].astype(str)) != {FROZEN_MODEL}:
        raise ValueError(f"Formal T4 model set must be exactly {{{FROZEN_MODEL}}}")
    if set(metrics["feature_mode"].astype(str)) != {"raw_w"}:
        raise ValueError("Formal T4 feature_mode must be exactly raw_w")
    if set(metrics["direction"].astype(str)) != set(EXPECTED_DIRECTIONS):
        raise ValueError(f"Unexpected T4 directions: {sorted(metrics['direction'].unique())}")
    if set(metrics["held_out_profile"].astype(str)) != set(EXPECTED_PROFILES):
        raise ValueError(f"Unexpected T4 profiles: {sorted(metrics['held_out_profile'].unique())}")
    if set(metrics["seed"].astype(int)) != set(EXPECTED_SEEDS):
        raise ValueError(f"Unexpected publication seeds: {sorted(metrics['seed'].unique())}")

    expected_rows = len(EXPECTED_DIRECTIONS) * len(EXPECTED_PROFILES) * len(EXPECTED_SEEDS)
    if len(metrics) != expected_rows:
        raise ValueError(f"Expected {expected_rows} clean T4 rows, found {len(metrics)}")
    key = ["direction", "held_out_profile", "seed"]
    if metrics.duplicated(key).any():
        raise ValueError("Duplicate formal T4 direction/profile/seed rows")

    expected_keys = {
        (d, p, s)
        for d in EXPECTED_DIRECTIONS
        for p in EXPECTED_PROFILES
        for s in EXPECTED_SEEDS
    }
    seen = set(
        zip(
            metrics["direction"].astype(str),
            metrics["held_out_profile"].astype(str),
            metrics["seed"].astype(int),
        )
    )
    if seen != expected_keys:
        raise ValueError(
            f"Formal T4 key set mismatch; missing={sorted(expected_keys-seen)}, extra={sorted(seen-expected_keys)}"
        )


def validate_noise_metrics(noise: pd.DataFrame) -> None:
    required = {
        "direction", "held_out_profile", "model", "feature_mode", "seed",
        "sigma_pm_each_wavelength", "noise_draw", "MAE", "MAE_increase",
        "RMSE", "Q95_AE", "clean_MAE",
    }
    missing = required.difference(noise.columns)
    if missing:
        raise ValueError(f"Missing noise metric columns: {sorted(missing)}")
    if set(noise["model"].astype(str)) != {FROZEN_MODEL}:
        raise ValueError("Noise artifact contains a non-frozen model")
    if set(noise["feature_mode"].astype(str)) != {"raw_w"}:
        raise ValueError("Noise artifact contains a non-frozen feature mode")
    if set(noise["seed"].astype(int)) != set(EXPECTED_SEEDS):
        raise ValueError("Noise artifact publication seed set mismatch")
    sigma = {round(float(x), 6) for x in noise["sigma_pm_each_wavelength"].unique()}
    if sigma != {round(x, 6) for x in EXPECTED_SIGMA_PM}:
        raise ValueError(f"Noise sigma set mismatch: {sorted(sigma)}")
    if set(noise["noise_draw"].astype(int)) != set(range(EXPECTED_NOISE_DRAWS)):
        raise ValueError("Noise draw set mismatch")

    expected_rows = (
        len(EXPECTED_SEEDS)
        * len(EXPECTED_DIRECTIONS)
        * len(EXPECTED_PROFILES)
        * len(EXPECTED_SIGMA_PM)
        * EXPECTED_NOISE_DRAWS
    )
    if len(noise) != expected_rows:
        raise ValueError(f"Expected {expected_rows} noise rows, found {len(noise)}")
    key = [
        "direction", "held_out_profile", "seed",
        "sigma_pm_each_wavelength", "noise_draw",
    ]
    if noise.duplicated(key).any():
        raise ValueError("Duplicate formal T4 noise rows")


def main() -> None:
    p = argparse.ArgumentParser(description="Aggregate frozen IUW-TCN T4 publication seeds")
    p.add_argument("--input-root", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--bootstrap", type=int, default=20000)
    args = p.parse_args()

    metric_files = sorted(args.input_root.rglob("split_metrics.csv"))
    noise_files = sorted(args.input_root.rglob("fbg_noise_metrics.csv"))
    if len(metric_files) != len(EXPECTED_SEEDS):
        raise FileNotFoundError(
            f"Expected {len(EXPECTED_SEEDS)} split_metrics.csv files, found {len(metric_files)}"
        )
    if len(noise_files) != len(EXPECTED_SEEDS):
        raise FileNotFoundError(
            f"Expected {len(EXPECTED_SEEDS)} fbg_noise_metrics.csv files, found {len(noise_files)}"
        )

    metrics = pd.concat([pd.read_csv(f) for f in metric_files], ignore_index=True)
    noise = pd.concat([pd.read_csv(f) for f in noise_files], ignore_index=True)
    validate_clean_metrics(metrics)
    validate_noise_metrics(noise)

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
            R2_std=("R2", "std"),
            Q95_AE_mean=("Q95_AE", "mean"),
            Q95_AE_std=("Q95_AE", "std"),
            MaxAE_mean=("MaxAE", "mean"),
            MaxAE_std=("MaxAE", "std"),
        )
    )

    profile_summary = (
        metrics.groupby(["direction", "held_out_profile", "model"], as_index=False)
        .agg(
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

    per_seed = (
        metrics.groupby(["seed", "direction", "model"], as_index=False)
        .agg(
            n_profiles=("MAE", "size"),
            MAE_mean=("MAE", "mean"),
            RMSE_mean=("RMSE", "mean"),
            R2_mean=("R2", "mean"),
            Q95_AE_mean=("Q95_AE", "mean"),
            MaxAE_mean=("MaxAE", "mean"),
        )
    )

    bootstrap_rows: list[dict] = []
    scopes = [("overall", metrics)] + [
        (direction, metrics[metrics["direction"] == direction])
        for direction in EXPECTED_DIRECTIONS
    ]
    for scope_idx, (scope, sdf) in enumerate(scopes):
        row = {
            "scope": scope,
            "model": FROZEN_MODEL,
            "n_rows": len(sdf),
            "n_seeds": int(sdf["seed"].nunique()),
            "MAE_mean": float(sdf["MAE"].mean()),
            "MAE_std_rowlevel": float(sdf["MAE"].std(ddof=1)),
        }
        lo, hi = seed_cluster_bootstrap_mean(
            sdf,
            "MAE",
            n_boot=args.bootstrap,
            seed=20260831 + scope_idx,
        )
        row["seed_cluster_bootstrap95_lo"] = lo
        row["seed_cluster_bootstrap95_hi"] = hi
        seed_means = sdf.groupby("seed")["MAE"].mean().to_numpy(dtype=float)
        row["seed_mean_MAE_std"] = float(seed_means.std(ddof=1))
        bootstrap_rows.append(row)
    bootstrap = pd.DataFrame(bootstrap_rows)

    noise_summary = (
        noise.groupby(["direction", "sigma_pm_each_wavelength", "model"], as_index=False)
        .agg(
            n_rows=("MAE", "size"),
            n_seeds=("seed", "nunique"),
            MAE_mean=("MAE", "mean"),
            MAE_std=("MAE", "std"),
            clean_MAE_mean=("clean_MAE", "mean"),
            MAE_increase_mean=("MAE_increase", "mean"),
            MAE_increase_std=("MAE_increase", "std"),
            RMSE_mean=("RMSE", "mean"),
            Q95_AE_mean=("Q95_AE", "mean"),
        )
    )
    noise_profile = (
        noise.groupby(
            ["direction", "held_out_profile", "sigma_pm_each_wavelength", "model"],
            as_index=False,
        )
        .agg(
            n_rows=("MAE", "size"),
            n_seeds=("seed", "nunique"),
            MAE_mean=("MAE", "mean"),
            MAE_increase_mean=("MAE_increase", "mean"),
            MAE_increase_std=("MAE_increase", "std"),
        )
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    metrics.sort_values(["seed", "direction", "held_out_profile"]).to_csv(
        args.out_dir / "all_seed_split_metrics.csv", index=False
    )
    across_seed.to_csv(args.out_dir / "across_seed_summary.csv", index=False)
    profile_summary.to_csv(args.out_dir / "profile_across_seed_summary.csv", index=False)
    per_seed.to_csv(args.out_dir / "per_seed_direction_summary.csv", index=False)
    bootstrap.to_csv(args.out_dir / "seed_cluster_bootstrap.csv", index=False)
    noise.sort_values(
        ["seed", "direction", "held_out_profile", "sigma_pm_each_wavelength", "noise_draw"]
    ).to_csv(args.out_dir / "all_seed_fbg_noise_metrics.csv", index=False)
    noise_summary.to_csv(args.out_dir / "fbg_noise_across_seed_summary.csv", index=False)
    noise_profile.to_csv(args.out_dir / "fbg_noise_profile_summary.csv", index=False)

    print("=== frozen IUW-TCN across-seed summary ===")
    print(across_seed.to_string(index=False))
    print("\n=== seed-cluster bootstrap MAE ===")
    print(bootstrap.to_string(index=False))
    print("\n=== raw-W FBG noise summary ===")
    print(noise_summary.to_string(index=False))


if __name__ == "__main__":
    main()
