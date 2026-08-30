from __future__ import annotations

from pathlib import Path
import argparse
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

PROFILES = ("HWFET", "LA92", "NEDC", "NYCC", "US06", "WLTC")
RATES = ("1C", "2C")

FEATURE_SETS = {
    "V": ["Voltage_V"],
    "VI": ["Voltage_V", "Current_A"],
    "VI+W": ["Voltage_V", "Current_A", "Wavelength_1", "Wavelength_2"],
    "VI+TF": ["Voltage_V", "Current_A", "temperature_℃", "force_N"],
    "W": ["Wavelength_1", "Wavelength_2"],
    "TF": ["temperature_℃", "force_N"],
}

FORBIDDEN_PREDICTORS = {"SOC", "dis_cap", "Time_s"}


def load_data(root: Path) -> pd.DataFrame:
    frames = []
    for path in sorted(root.glob("*.xlsx")):
        profile, rate = path.stem.rsplit("_", 1)
        if profile not in PROFILES or rate not in RATES:
            continue
        df = pd.read_excel(path)
        df["profile"] = profile
        df["rate"] = rate
        df["source_file"] = path.name
        frames.append(df)
    if not frames:
        raise FileNotFoundError(f"No expected SiC-18 workbooks found in {root}")
    return pd.concat(frames, ignore_index=True)


def validate_feature_sets() -> None:
    for name, columns in FEATURE_SETS.items():
        overlap = FORBIDDEN_PREDICTORS.intersection(columns)
        if overlap:
            raise ValueError(f"Feature set {name} contains forbidden predictors: {sorted(overlap)}")


def fit_predict(train: pd.DataFrame, test: pd.DataFrame, columns: list[str]) -> np.ndarray:
    model = HistGradientBoostingRegressor(
        max_iter=200,
        learning_rate=0.08,
        max_leaf_nodes=31,
        l2_regularization=1e-3,
        random_state=0,
    )
    model.fit(train[columns], train["SOC"])
    return np.clip(model.predict(test[columns]), 0.0, 1.0)


def metrics(y: pd.Series, pred: np.ndarray) -> dict[str, float]:
    err = np.abs(y.to_numpy() - pred)
    return {
        "MAE": mean_absolute_error(y, pred),
        "RMSE": mean_squared_error(y, pred) ** 0.5,
        "R2": r2_score(y, pred),
        "MaxAE": float(err.max()),
        "Q95_AE": float(np.quantile(err, 0.95)),
    }


def run_lopo(data: pd.DataFrame) -> list[dict]:
    records = []
    for rate in RATES:
        rate_df = data[data["rate"] == rate]
        for held_out in PROFILES:
            train = rate_df[rate_df["profile"] != held_out]
            test = rate_df[rate_df["profile"] == held_out]
            for name, columns in FEATURE_SETS.items():
                pred = fit_predict(train, test, columns)
                records.append({
                    "protocol": "LOPO_same_rate",
                    "train": f"{rate}:other5",
                    "test": f"{rate}:{held_out}",
                    "features": name,
                    "n_test": len(test),
                    **metrics(test["SOC"], pred),
                })
    return records


def run_cross_rate(data: pd.DataFrame) -> list[dict]:
    records = []
    for train_rate, test_rate in (("1C", "2C"), ("2C", "1C")):
        train = data[data["rate"] == train_rate]
        test = data[data["rate"] == test_rate]
        for name, columns in FEATURE_SETS.items():
            pred = fit_predict(train, test, columns)
            records.append({
                "protocol": "cross_rate",
                "train": train_rate,
                "test": test_rate,
                "features": name,
                "n_test": len(test),
                **metrics(test["SOC"], pred),
            })
    return records


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("data/raw/SiC-18"))
    parser.add_argument("--out", type=Path, default=Path("results/baseline_results.csv"))
    args = parser.parse_args()

    validate_feature_sets()
    data = load_data(args.data)
    records = run_lopo(data) + run_cross_rate(data)
    result = pd.DataFrame(records)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.out, index=False)

    lopo = (
        result[result["protocol"] == "LOPO_same_rate"]
        .groupby("features")[["MAE", "RMSE", "R2", "MaxAE", "Q95_AE"]]
        .mean()
        .sort_values("MAE")
    )
    print("\nMean leave-one-profile-out results")
    print(lopo.to_string())
    print("\nCross-rate results")
    print(result[result["protocol"] == "cross_rate"].to_string(index=False))


if __name__ == "__main__":
    main()
