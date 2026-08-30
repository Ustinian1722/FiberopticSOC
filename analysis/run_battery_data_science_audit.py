from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures


SIGNALS = [
    "Voltage_V",
    "Wavelength_1",
    "Wavelength_2",
    "temperature_℃",
    "force_N",
]

K = np.array([[0.0208, 0.00054], [0.0254, 0.00085]], dtype=float)


def load_files(root: Path):
    rows = []
    for path in sorted(root.glob("*.xlsx")):
        stem = path.stem
        profile, rate = stem.rsplit("_", 1)
        df = pd.read_excel(path)
        rows.append((path.name, profile, rate, df))
    if len(rows) != 12:
        raise RuntimeError(f"Expected 12 workbooks, found {len(rows)} in {root}")
    return rows


def file_audit(items):
    rows = []
    for name, profile, rate, df in items:
        t = df["Time_s"].to_numpy(float)
        dt = np.diff(t)
        recon = 1.0 - df["dis_cap"].to_numpy(float) / float(df["dis_cap"].max())
        rows.append(
            {
                "file": name,
                "profile": profile,
                "rate": rate,
                "n": len(df),
                "time_start_s": float(t[0]),
                "time_end_s": float(t[-1]),
                "dt_median_s": float(np.median(dt)),
                "dt_max_s": float(np.max(dt)),
                "fraction_dt_gt_1s": float(np.mean(dt > 1)),
                "fraction_dt_eq_0s": float(np.mean(dt == 0)),
                "capacity_As": float(df["dis_cap"].max()),
                "capacity_Ah_if_As": float(df["dis_cap"].max() / 3600.0),
                "soc_reconstruction_max_abs_error": float(np.max(np.abs(recon - df["SOC"].to_numpy(float)))),
            }
        )
    return pd.DataFrame(rows)


def capacity_increment_audit(items):
    dq_all, pred_all = [], []
    for _, _, _, df in items:
        t = df["Time_s"].to_numpy(float)
        dt = np.diff(t)
        dq = np.diff(df["dis_cap"].to_numpy(float))
        pred = -df["Current_A"].to_numpy(float)[1:] * dt
        dq_all.append(dq)
        pred_all.append(pred)
    dq = np.concatenate(dq_all)
    pred = np.concatenate(pred_all)
    ss_res = float(np.sum((dq - pred) ** 2))
    ss_tot = float(np.sum((dq - dq.mean()) ** 2))
    return {
        "pooled_increment_mae": float(np.mean(np.abs(dq - pred))),
        "pooled_increment_rmse": float(np.sqrt(np.mean((dq - pred) ** 2))),
        "pooled_increment_r2": float(1.0 - ss_res / ss_tot),
        "pooled_increment_corr": float(np.corrcoef(dq, pred)[0, 1]),
    }


def fbg_mapping_audit(items):
    all_df = pd.concat([df for _, _, _, df in items], ignore_index=True)
    X = all_df[["temperature_℃", "force_N"]].to_numpy(float)
    Y = all_df[["Wavelength_1", "Wavelength_2"]].to_numpy(float)
    pred = X @ K.T
    inv = np.linalg.inv(K)
    temp_vec = K[:, 0]
    force_vec = K[:, 1]
    cosine = float(np.dot(temp_vec, force_vec) / (np.linalg.norm(temp_vec) * np.linalg.norm(force_vec)))
    angle_deg = float(np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0))))
    return {
        "forward_max_abs_residual_nm": float(np.max(np.abs(pred - Y))),
        "determinant": float(np.linalg.det(K)),
        "condition_number_2norm": float(np.linalg.cond(K)),
        "sensitivity_vector_angle_deg": angle_deg,
        "inverse_matrix": inv.tolist(),
        "predicted_decoupled_std_from_independent_1pm_noise": {
            "temperature_degC": float(np.linalg.norm(inv[0]) * 0.001),
            "force_N": float(np.linalg.norm(inv[1]) * 0.001),
        },
    }


def rest_step_noise(items):
    rows = []
    for name, profile, rate, df in items:
        t = df["Time_s"].to_numpy(float)
        dt = np.diff(t)
        current = df["Current_A"].to_numpy(float)
        mask = (dt == 1) & (np.abs(current[1:]) < 1e-4) & (np.abs(current[:-1]) < 1e-4)
        row = {"file": name, "profile": profile, "rate": rate, "n_rest_1s_pairs": int(mask.sum())}
        for sig in ["Wavelength_1", "Wavelength_2", "temperature_℃", "force_N"]:
            delta = np.diff(df[sig].to_numpy(float))[mask]
            row[f"median_abs_step_{sig}"] = float(np.median(np.abs(delta))) if len(delta) else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def load_sensitivity_after_soc_trend(items):
    rows = []
    poly = PolynomialFeatures(5, include_bias=True)
    for name, profile, rate, df in items:
        X = poly.fit_transform(df["SOC"].to_numpy(float).reshape(-1, 1))
        current = df["Current_A"].to_numpy(float)
        for sig in SIGNALS:
            y = df[sig].to_numpy(float)
            model = LinearRegression().fit(X, y)
            residual = y - model.predict(X)
            rows.append(
                {
                    "file": name,
                    "profile": profile,
                    "rate": rate,
                    "signal": sig,
                    "soc_only_curve_r2": float(model.score(X, y)),
                    "residual_std": float(np.std(residual)),
                    "residual_corr_current": float(np.corrcoef(residual, current)[0, 1]),
                    "residual_corr_abs_current": float(np.corrcoef(residual, np.abs(current))[0, 1]),
                }
            )
    return pd.DataFrame(rows)


def rate_shift(items, grid=None):
    if grid is None:
        grid = np.linspace(0.02, 0.98, 49)
    by_key = {(profile, rate): df for _, profile, rate, df in items}
    pooled_span = {}
    for sig in SIGNALS + ["Current_A"]:
        vals = np.concatenate([df[sig].to_numpy(float) for _, _, _, df in items])
        pooled_span[sig] = float(vals.max() - vals.min())

    rows = []
    for profile in sorted({profile for _, profile, _, _ in items}):
        d1 = by_key[(profile, "1C")]
        d2 = by_key[(profile, "2C")]
        for sig in SIGNALS + ["Current_A"]:
            curves = []
            for df in (d1, d2):
                g = df.groupby("SOC", as_index=False)[sig].mean().sort_values("SOC")
                curves.append(np.interp(grid, g["SOC"].to_numpy(float), g[sig].to_numpy(float)))
            mad = float(np.mean(np.abs(curves[0] - curves[1])))
            rows.append(
                {
                    "profile": profile,
                    "signal": sig,
                    "mean_abs_1C_2C_difference": mad,
                    "pooled_signal_span": pooled_span[sig],
                    "normalized_rate_shift": mad / pooled_span[sig] if pooled_span[sig] > 0 else np.nan,
                    "curve_corr_1C_2C": float(np.corrcoef(curves[0], curves[1])[0, 1]),
                }
            )
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("data/extracted/SiC-18"))
    parser.add_argument("--out", type=Path, default=Path("results/data_science_audit"))
    args = parser.parse_args()

    items = load_files(args.data)
    args.out.mkdir(parents=True, exist_ok=True)

    files = file_audit(items)
    rest = rest_step_noise(items)
    load = load_sensitivity_after_soc_trend(items)
    rate = rate_shift(items)
    cap = capacity_increment_audit(items)
    fbg = fbg_mapping_audit(items)

    files.to_csv(args.out / "file_sampling_label_audit.csv", index=False)
    rest.to_csv(args.out / "rest_step_noise.csv", index=False)
    load.to_csv(args.out / "soc_trend_load_sensitivity.csv", index=False)
    rate.to_csv(args.out / "rate_shift_by_profile.csv", index=False)

    summary = {
        "n_files": len(items),
        "n_rows": int(sum(len(df) for _, _, _, df in items)),
        "capacity_increment_audit": cap,
        "fbg_mapping_audit": fbg,
        "sampling": {
            "fraction_dt_gt_1s_min": float(files["fraction_dt_gt_1s"].min()),
            "fraction_dt_gt_1s_max": float(files["fraction_dt_gt_1s"].max()),
            "max_dt_s": float(files["dt_max_s"].max()),
        },
        "capacity_Ah_if_dis_cap_is_As": {
            "mean": float(files["capacity_Ah_if_As"].mean()),
            "min": float(files["capacity_Ah_if_As"].min()),
            "max": float(files["capacity_Ah_if_As"].max()),
        },
        "median_rest_step": {
            "W1_nm": float(rest["median_abs_step_Wavelength_1"].median()),
            "W2_nm": float(rest["median_abs_step_Wavelength_2"].median()),
            "temperature_degC": float(rest["median_abs_step_temperature_℃"].median()),
            "force_N": float(rest["median_abs_step_force_N"].median()),
        },
        "mean_soc_curve_r2": load.groupby("signal")["soc_only_curve_r2"].mean().to_dict(),
        "mean_abs_residual_corr_current": load.groupby("signal")["residual_corr_current"].apply(lambda x: float(np.mean(np.abs(x)))).to_dict(),
        "mean_normalized_rate_shift": rate.groupby("signal")["normalized_rate_shift"].mean().to_dict(),
    }
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
