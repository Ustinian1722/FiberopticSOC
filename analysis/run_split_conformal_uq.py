from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd


def finite_sample_q(scores: np.ndarray, alpha: float) -> float:
    """Split-conformal finite-sample residual quantile."""
    s = np.sort(np.asarray(scores, dtype=float))
    if len(s) == 0:
        raise ValueError("Calibration set is empty")
    k = int(math.ceil((len(s) + 1) * (1.0 - alpha)))
    k = min(max(k, 1), len(s))
    return float(s[k - 1])


def interval_score(y: np.ndarray, lo: np.ndarray, hi: np.ndarray, alpha: float) -> np.ndarray:
    width = hi - lo
    below = np.maximum(lo - y, 0.0)
    above = np.maximum(y - hi, 0.0)
    return width + (2.0 / alpha) * (below + above)


def metrics(y: np.ndarray, lo: np.ndarray, hi: np.ndarray, alpha: float) -> dict:
    covered = (y >= lo) & (y <= hi)
    width = hi - lo
    # SOC is represented on [0,1], so MPIW is already normalized by the physical target range.
    return {
        "nominal_coverage": 1.0 - alpha,
        "PICP": float(np.mean(covered)),
        "coverage_error": float(np.mean(covered) - (1.0 - alpha)),
        "MPIW": float(np.mean(width)),
        "PINAW": float(np.mean(width)),
        "mean_interval_score": float(np.mean(interval_score(y, lo, hi, alpha))),
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--calibration", type=Path, required=True,
                   help="CSV with source-side calibration predictions")
    p.add_argument("--test", type=Path, required=True,
                   help="CSV with frozen-model final test predictions")
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--alphas", type=float, nargs="+", default=[0.10, 0.05])
    p.add_argument("--y-col", default="y_true")
    p.add_argument("--pred-col", default="y_pred")
    p.add_argument("--current-col", default=None,
                   help="Optional observed current column. Load strata are calibrated from source-side |I| tertiles.")
    p.add_argument("--soc-bin-width", type=float, default=0.10)
    args = p.parse_args()

    cal = pd.read_csv(args.calibration)
    test = pd.read_csv(args.test)
    for df, name in ((cal, "calibration"), (test, "test")):
        missing = {args.y_col, args.pred_col}.difference(df.columns)
        if missing:
            raise ValueError(f"{name} CSV missing columns: {sorted(missing)}")
    if args.current_col is not None:
        for df, name in ((cal, "calibration"), (test, "test")):
            if args.current_col not in df.columns:
                raise ValueError(f"{name} CSV missing current column: {args.current_col}")

    y_cal = cal[args.y_col].to_numpy(float)
    p_cal = cal[args.pred_col].to_numpy(float)
    y_test = test[args.y_col].to_numpy(float)
    p_test = test[args.pred_col].to_numpy(float)
    scores = np.abs(y_cal - p_cal)

    load_thresholds = None
    test_load_group = None
    if args.current_col is not None:
        abs_i_cal = np.abs(cal[args.current_col].to_numpy(float))
        q33, q67 = np.quantile(abs_i_cal, [1.0 / 3.0, 2.0 / 3.0])
        load_thresholds = (float(q33), float(q67))
        abs_i_test = np.abs(test[args.current_col].to_numpy(float))
        # Boundaries come only from calibration/source-side observations.
        test_load_group = pd.cut(
            abs_i_test,
            bins=[-np.inf, q33, q67, np.inf],
            labels=["low", "medium", "high"],
            include_lowest=True,
        )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    interval_frames = []
    summaries = []
    conditional_rows = []
    load_rows = []

    for alpha in args.alphas:
        if not 0.0 < alpha < 1.0:
            raise ValueError(f"alpha must be in (0,1), got {alpha}")
        q = finite_sample_q(scores, alpha)
        lo = np.clip(p_test - q, 0.0, 1.0)
        hi = np.clip(p_test + q, 0.0, 1.0)

        m = metrics(y_test, lo, hi, alpha)
        m.update({"alpha": alpha, "q_abs_residual": q, "n_cal": len(cal), "n_test": len(test)})
        if load_thresholds is not None:
            m["source_abs_current_q33_A"] = load_thresholds[0]
            m["source_abs_current_q67_A"] = load_thresholds[1]
        summaries.append(m)

        frame = test.copy()
        frame["alpha"] = alpha
        frame["lower"] = lo
        frame["upper"] = hi
        frame["covered"] = (y_test >= lo) & (y_test <= hi)
        frame["interval_width"] = hi - lo
        if test_load_group is not None:
            frame["load_group_source_calibrated"] = test_load_group.astype(str)
        interval_frames.append(frame)

        bw = args.soc_bin_width
        edges = np.arange(0.0, 1.0 + bw + 1e-12, bw)
        bins = pd.cut(y_test, bins=edges, right=False, include_lowest=True)
        temp = pd.DataFrame({"bin": bins, "y": y_test, "lo": lo, "hi": hi})
        for b, g in temp.groupby("bin", observed=True):
            yy = g["y"].to_numpy(float)
            ll = g["lo"].to_numpy(float)
            hh = g["hi"].to_numpy(float)
            mm = metrics(yy, ll, hh, alpha)
            conditional_rows.append({"alpha": alpha, "soc_bin": str(b), "n": len(g), **mm})

        if test_load_group is not None:
            temp_load = pd.DataFrame(
                {"load_group": test_load_group, "y": y_test, "lo": lo, "hi": hi}
            )
            for group, g in temp_load.groupby("load_group", observed=True):
                yy = g["y"].to_numpy(float)
                ll = g["lo"].to_numpy(float)
                hh = g["hi"].to_numpy(float)
                mm = metrics(yy, ll, hh, alpha)
                load_rows.append(
                    {
                        "alpha": alpha,
                        "load_group": str(group),
                        "n": len(g),
                        "source_abs_current_q33_A": load_thresholds[0],
                        "source_abs_current_q67_A": load_thresholds[1],
                        **mm,
                    }
                )

    pd.DataFrame(summaries).to_csv(args.out_dir / "uq_summary.csv", index=False)
    pd.concat(interval_frames, ignore_index=True).to_csv(args.out_dir / "test_intervals.csv", index=False)
    pd.DataFrame(conditional_rows).to_csv(args.out_dir / "uq_by_soc_bin.csv", index=False)
    if load_rows:
        pd.DataFrame(load_rows).to_csv(args.out_dir / "uq_by_load_group.csv", index=False)

    print("=== Split-conformal SOC UQ ===")
    print(pd.DataFrame(summaries).to_string(index=False))
    if load_thresholds is not None:
        print(
            f"Source-side load thresholds: |I| q33={load_thresholds[0]:.6f} A, "
            f"q67={load_thresholds[1]:.6f} A"
        )
    print("Calibration scores and load strata are computed only from the supplied calibration/source CSV; test labels are evaluation-only.")


if __name__ == "__main__":
    main()
