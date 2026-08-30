from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from run_representation_conditioning_diagnostic import whitening_matrix
from run_sequence_representation_benchmark import PROFILES, load_sources, normalize_sources, train_normalizer


def orthogonal_procrustes(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    u, _, vt = np.linalg.svd(a.T @ b, full_matrices=False)
    return u @ vt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("data/extracted"))
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("results/whitened_representation_equivalence"),
    )
    args = parser.parse_args()

    sources = load_sources(args.data)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    rows = []

    for rate in ("1C", "2C"):
        for held_out in PROFILES:
            train_raw = [
                s for s in sources
                if s["rate"] == rate and s["profile"] != held_out
            ]
            mean, std = train_normalizer(train_raw)
            train = normalize_sources(train_raw, mean, std)
            x = np.concatenate([s["x"] for s in train], axis=0).astype(np.float64)

            w = x[:, (2, 3)]
            tf = x[:, (4, 5)]
            w_white = w @ whitening_matrix(w).T
            tf_white = tf @ whitening_matrix(tf).T

            q = orthogonal_procrustes(w_white, tf_white)
            mapped = w_white @ q
            relative_residual = float(
                np.linalg.norm(mapped - tf_white) / max(np.linalg.norm(tf_white), 1e-12)
            )
            orthogonality_error = float(np.linalg.norm(q.T @ q - np.eye(2), ord="fro"))
            rows.append(
                {
                    "rate": rate,
                    "held_out_profile": held_out,
                    "n_train_samples": len(x),
                    "procrustes_relative_residual": relative_residual,
                    "orthogonality_error_fro": orthogonality_error,
                    "rotation_det": float(np.linalg.det(q)),
                    "flattened_correlation_after_mapping": float(
                        np.corrcoef(mapped.ravel(), tf_white.ravel())[0, 1]
                    ),
                }
            )

    result = pd.DataFrame(rows)
    result.to_csv(args.out_dir / "equivalence_by_split.csv", index=False)
    print(result.to_string(index=False))
    print("\nmax Procrustes residual:", result["procrustes_relative_residual"].max())
    print("max orthogonality error:", result["orthogonality_error_fro"].max())


if __name__ == "__main__":
    main()
