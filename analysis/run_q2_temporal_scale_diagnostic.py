from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

SIGNALS = ["Current_A", "Voltage_V", "Wavelength_1", "Wavelength_2", "temperature_℃", "force_N"]


def safe_corr(a: np.ndarray, b: np.ndarray) -> float:
    mask = np.isfinite(a) & np.isfinite(b)
    a, b = a[mask], b[mask]
    if len(a) < 3 or np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def detrend_soc(df: pd.DataFrame, signal: str) -> np.ndarray:
    # Descriptive diagnostic only; SOC is not used by the estimator.
    poly = PolynomialFeatures(5, include_bias=True)
    x = poly.fit_transform(df["SOC"].to_numpy(float).reshape(-1, 1))
    y = df[signal].to_numpy(float)
    return y - LinearRegression().fit(x, y).predict(x)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=Path, default=Path("data/extracted/SiC-18"))
    p.add_argument("--out-dir", type=Path, default=Path("results/q2_temporal_scale"))
    args = p.parse_args()

    rows = []
    jump_rows = []
    files = sorted(args.data.rglob("*.xlsx"))
    if len(files) != 12:
        raise RuntimeError(f"Expected 12 workbooks, got {len(files)}")

    for path in files:
        profile, rate = path.stem.rsplit("_", 1)
        df = pd.read_excel(path)
        t = df["Time_s"].to_numpy(float)
        dt = np.diff(t)
        positive_dt = dt > 0
        one_sec = np.isclose(dt, 1.0)
        current = df["Current_A"].to_numpy(float)
        d_i = np.diff(current)
        current_iqr = float(np.subtract(*np.percentile(current, [75, 25])))
        current_iqr = max(current_iqr, 1e-12)

        for signal in SIGNALS:
            x = df[signal].to_numpy(float)
            dx = np.diff(x)
            iqr = float(np.subtract(*np.percentile(x, [75, 25])))
            iqr = max(iqr, 1e-12)
            speed = np.abs(dx[positive_dt] / dt[positive_dt]) / iqr
            residual = detrend_soc(df, signal) if signal != "Current_A" else x - np.mean(x)
            dres = np.diff(residual)

            rows.append(
                {
                    "file": path.name,
                    "profile": profile,
                    "rate": rate,
                    "signal": signal,
                    "iqr": iqr,
                    "median_normalized_abs_rate_per_s": float(np.median(speed)),
                    "p90_normalized_abs_rate_per_s": float(np.quantile(speed, 0.90)),
                    "lag1_corr_on_1s_pairs": safe_corr(x[:-1][one_sec], x[1:][one_sec]),
                    "soc_detrended_lag1_corr_on_1s_pairs": safe_corr(
                        residual[:-1][one_sec], residual[1:][one_sec]
                    ),
                    "corr_abs_signal_step_vs_abs_current_step_1s": safe_corr(
                        np.abs(dx[one_sec]) / iqr,
                        np.abs(d_i[one_sec]) / current_iqr,
                    ),
                    "corr_abs_detrended_step_vs_abs_current_step_1s": safe_corr(
                        np.abs(dres[one_sec]) / iqr,
                        np.abs(d_i[one_sec]) / current_iqr,
                    ),
                }
            )

        # Current-jump strata: evaluate normalized signal response when current changes sharply.
        valid_di = np.abs(d_i[one_sec])
        if len(valid_di):
            threshold = float(np.quantile(valid_di, 0.90))
            jump_mask = one_sec & (np.abs(d_i) >= threshold)
            quiet_mask = one_sec & (np.abs(d_i) <= np.quantile(valid_di, 0.50))
            for signal in ["Voltage_V", "temperature_℃", "force_N"]:
                x = df[signal].to_numpy(float)
                iqr = max(float(np.subtract(*np.percentile(x, [75, 25]))), 1e-12)
                dxn = np.abs(np.diff(x)) / iqr
                jump_rows.append(
                    {
                        "file": path.name,
                        "profile": profile,
                        "rate": rate,
                        "signal": signal,
                        "current_jump_p90_A": threshold,
                        "median_norm_step_high_current_jump": float(np.median(dxn[jump_mask])) if jump_mask.any() else np.nan,
                        "median_norm_step_quiet_current": float(np.median(dxn[quiet_mask])) if quiet_mask.any() else np.nan,
                        "jump_to_quiet_response_ratio": float(
                            np.median(dxn[jump_mask]) / max(np.median(dxn[quiet_mask]), 1e-12)
                        ) if jump_mask.any() and quiet_mask.any() else np.nan,
                    }
                )

    df = pd.DataFrame(rows)
    jump = pd.DataFrame(jump_rows)
    summary = df.groupby("signal", as_index=False).agg(
        normalized_rate_median=("median_normalized_abs_rate_per_s", "median"),
        normalized_rate_p90_mean=("p90_normalized_abs_rate_per_s", "mean"),
        lag1_corr_mean=("lag1_corr_on_1s_pairs", "mean"),
        detrended_lag1_corr_mean=("soc_detrended_lag1_corr_on_1s_pairs", "mean"),
        current_step_sensitivity_mean=("corr_abs_signal_step_vs_abs_current_step_1s", "mean"),
        detrended_current_step_sensitivity_mean=("corr_abs_detrended_step_vs_abs_current_step_1s", "mean"),
    )
    jump_summary = jump.groupby("signal", as_index=False).agg(
        jump_to_quiet_ratio_mean=("jump_to_quiet_response_ratio", "mean"),
        jump_to_quiet_ratio_median=("jump_to_quiet_response_ratio", "median"),
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out_dir / "per_file_temporal_metrics.csv", index=False)
    summary.to_csv(args.out_dir / "temporal_scale_summary.csv", index=False)
    jump.to_csv(args.out_dir / "current_jump_response.csv", index=False)
    jump_summary.to_csv(args.out_dir / "current_jump_response_summary.csv", index=False)
    print("=== Temporal scale summary ===")
    print(summary.to_string(index=False))
    print("\n=== Current-jump response ===")
    print(jump_summary.to_string(index=False))
    print("SOC is used only for descriptive detrending in this diagnostic, never as an estimator input.")


if __name__ == "__main__":
    main()
