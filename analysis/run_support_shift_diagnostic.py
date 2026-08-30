from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wasserstein_distance

from run_representation_conditioning_diagnostic import whitening_matrix
from run_sequence_representation_benchmark import PROFILES, load_sources, normalize_sources, train_normalizer


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=Path, default=Path("data/extracted"))
    p.add_argument("--out", type=Path, default=Path("results/support_shift/support_shift.csv"))
    p.add_argument("--envelope-quantile", type=float, default=0.005)
    p.add_argument("--optical-radius-quantile", type=float, default=0.995)
    args = p.parse_args()

    sources = load_sources(args.data)
    rows = []
    for train_rate, test_rate in (("1C", "2C"), ("2C", "1C")):
        direction = f"{train_rate}_to_{test_rate}"
        for held_out in PROFILES:
            train_raw = [s for s in sources if s["rate"] == train_rate and s["profile"] != held_out]
            test_raw = [s for s in sources if s["rate"] == test_rate and s["profile"] == held_out]
            mean, std = train_normalizer(train_raw)
            train = normalize_sources(train_raw, mean, std)
            test = normalize_sources(test_raw, mean, std)
            tr = np.concatenate([s["x"] for s in train], axis=0).astype(np.float64)
            te = np.concatenate([s["x"] for s in test], axis=0).astype(np.float64)

            q = args.envelope_quantile
            row = {"direction": direction, "held_out_profile": held_out}
            names = ["voltage", "current", "wavelength_1", "wavelength_2", "temperature", "force"]
            for idx, name in enumerate(names):
                lo, hi = np.quantile(tr[:, idx], [q, 1.0 - q])
                row[f"{name}_point_ood_fraction"] = float(np.mean((te[:, idx] < lo) | (te[:, idx] > hi)))
                row[f"{name}_wasserstein_train_std"] = float(wasserstein_distance(tr[:, idx], te[:, idx]))

            white = whitening_matrix(tr[:, (2, 3)])
            ztr = tr[:, (2, 3)] @ white.astype(np.float64).T
            zte = te[:, (2, 3)] @ white.astype(np.float64).T
            rtr = np.linalg.norm(ztr, axis=1)
            rte = np.linalg.norm(zte, axis=1)
            radius_high = float(np.quantile(rtr, args.optical_radius_quantile))
            row["wavelength_white_radius_point_ood_fraction"] = float(np.mean(rte > radius_high))
            row["wavelength_white_radius_wasserstein"] = float(wasserstein_distance(rtr, rte))
            rows.append(row)

    df = pd.DataFrame(rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    print(df.to_string(index=False))

    print("\n=== Direction means ===")
    summary = df.groupby("direction", as_index=False).mean(numeric_only=True)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
