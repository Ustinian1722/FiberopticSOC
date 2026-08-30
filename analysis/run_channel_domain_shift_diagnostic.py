from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp, wasserstein_distance

from run_sequence_representation_benchmark import COLUMNS, load_sources


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("data/extracted"))
    parser.add_argument("--out-dir", type=Path, default=Path("results/channel_domain_shift"))
    parser.add_argument("--envelope-quantile", type=float, default=0.005)
    args = parser.parse_args()

    sources = load_sources(args.data)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    rows = []

    for train_rate, test_rate in (("1C", "2C"), ("2C", "1C")):
        train = np.concatenate([s["x"] for s in sources if s["rate"] == train_rate], axis=0).astype(np.float64)
        test = np.concatenate([s["x"] for s in sources if s["rate"] == test_rate], axis=0).astype(np.float64)
        mean = train.mean(axis=0)
        std = train.std(axis=0)
        std[std < 1e-12] = 1.0
        train_z = (train - mean) / std
        test_z = (test - mean) / std
        q = args.envelope_quantile

        for j, channel in enumerate(COLUMNS):
            low, high = np.quantile(train[:, j], [q, 1.0 - q])
            rows.append(
                {
                    "split": f"{train_rate}_to_{test_rate}",
                    "train_rate": train_rate,
                    "test_rate": test_rate,
                    "channel": channel,
                    "train_q_low": low,
                    "train_q_high": high,
                    "test_fraction_outside_train_envelope": float(
                        np.mean((test[:, j] < low) | (test[:, j] > high))
                    ),
                    "wasserstein_in_train_std_units": float(
                        wasserstein_distance(train_z[:, j], test_z[:, j])
                    ),
                    "ks_statistic": float(
                        ks_2samp(train_z[:, j], test_z[:, j], method="asymp").statistic
                    ),
                }
            )

    result = pd.DataFrame(rows)
    result.to_csv(args.out_dir / "cross_rate_channel_shift.csv", index=False)
    print(result.to_string(index=False))


if __name__ == "__main__":
    main()
