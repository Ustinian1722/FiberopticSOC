from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from run_representation_conditioning_diagnostic import PairTCN
from run_sequence_representation_benchmark import (
    WindowDataset,
    count_params,
    load_sources,
    metric_dict,
    normalize_sources,
    predict_model,
    seed_everything,
    train_model,
    train_normalizer,
)

# Exact sensitivity matrix recovered from all released SiC-18 rows:
# [W1, W2]^T = A_TRUE [T, F]^T
# Wavelength columns are in nm, hence 1 pm = 1e-3 nm.
A_TRUE = np.array(
    [
        [0.0208, 0.00054],
        [0.0254, 0.00085],
    ],
    dtype=np.float64,
)
B_TRUE = np.linalg.inv(A_TRUE)


def clone_sources(sources: list[dict]) -> list[dict]:
    out = []
    for s in sources:
        item = dict(s)
        item["x"] = s["x"].copy()
        item["y"] = s["y"].copy()
        out.append(item)
    return out


def recompute_tf_from_w(sources: list[dict], decoder: np.ndarray) -> list[dict]:
    out = clone_sources(sources)
    for item in out:
        w = item["x"][:, (2, 3)].astype(np.float64)
        tf = w @ decoder.T
        item["x"][:, 4] = tf[:, 0].astype(np.float32)
        item["x"][:, 5] = tf[:, 1].astype(np.float32)
    return out


def add_wavelength_noise_and_decode(
    sources: list[dict], sigma_pm: float, rng: np.random.Generator
) -> list[dict]:
    out = clone_sources(sources)
    sigma_nm = sigma_pm * 1e-3
    for item in out:
        if sigma_nm > 0:
            noise = rng.normal(0.0, sigma_nm, size=(len(item["x"]), 2)).astype(np.float32)
            item["x"][:, 2:4] += noise
        w = item["x"][:, (2, 3)].astype(np.float64)
        tf = w @ B_TRUE.T
        item["x"][:, 4] = tf[:, 0].astype(np.float32)
        item["x"][:, 5] = tf[:, 1].astype(np.float32)
    return out


def perturbed_decoder(relative_sigma: float, rng: np.random.Generator) -> np.ndarray:
    if relative_sigma == 0:
        return B_TRUE.copy()
    # Each calibrated sensitivity coefficient receives independent multiplicative error.
    # This is a controlled robustness stress test, not a claim about a specific instrument.
    perturb = rng.normal(0.0, relative_sigma, size=A_TRUE.shape)
    a_hat = A_TRUE * (1.0 + perturb)
    if abs(np.linalg.det(a_hat)) < 1e-10:
        return perturbed_decoder(relative_sigma, rng)
    return np.linalg.inv(a_hat)


def make_loader(
    sources: list[dict], mean: np.ndarray, std: np.ndarray, window: int, stride: int, batch_size: int
) -> DataLoader:
    normed = normalize_sources(sources, mean, std)
    ds = WindowDataset(normed, window, stride)
    return DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)


def fit_clean_models(args, train_raw: list[dict], mean: np.ndarray, std: np.ndarray, device: torch.device):
    train_norm = normalize_sources(train_raw, mean, std)
    train_ds = WindowDataset(train_norm, args.window, args.train_stride)
    loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    models = {}
    for name, pair in (("VI+W", (2, 3)), ("VI+TF", (4, 5))):
        seed_everything(args.seed)
        model = PairTCN(pair)
        print(f"\nTraining clean {name}; params={count_params(model)} windows={len(train_ds)}")
        train_model(model, loader, device, args.epochs, args.lr)
        models[name] = model
    return models


def evaluate(model, loader, device):
    y, pred, _, _ = predict_model(model, loader, device)
    return metric_dict(y, pred)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("data/extracted"))
    parser.add_argument("--out-dir", type=Path, default=Path("results/calibration_noise"))
    parser.add_argument("--window", type=int, default=64)
    parser.add_argument("--train-stride", type=int, default=4)
    parser.add_argument("--test-stride", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--noise-trials", type=int, default=5)
    parser.add_argument("--calibration-trials", type=int, default=10)
    args = parser.parse_args()

    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sources = load_sources(args.data)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    row_norms = np.linalg.norm(B_TRUE, axis=1)
    amplification = pd.DataFrame(
        [
            {
                "quantity": "temperature",
                "decoder_row_norm_per_nm": row_norms[0],
                "std_per_1pm_independent_W_noise": row_norms[0] * 1e-3,
                "unit": "degC",
            },
            {
                "quantity": "force",
                "decoder_row_norm_per_nm": row_norms[1],
                "std_per_1pm_independent_W_noise": row_norms[1] * 1e-3,
                "unit": "N",
            },
        ]
    )
    amplification.to_csv(args.out_dir / "theoretical_noise_amplification.csv", index=False)

    noise_sigmas_pm = (0.0, 0.1, 0.25, 0.5, 1.0, 2.0)
    calibration_sigmas_pct = (0.0, 0.25, 0.5, 1.0, 2.0, 5.0)
    noise_rows = []
    calibration_rows = []

    for train_rate, test_rate in (("1C", "2C"), ("2C", "1C")):
        split = f"{train_rate}_to_{test_rate}"
        train_raw = [s for s in sources if s["rate"] == train_rate]
        test_raw = [s for s in sources if s["rate"] == test_rate]
        mean, std = train_normalizer(train_raw)
        models = fit_clean_models(args, train_raw, mean, std, device)

        # Clean raw optical reference for the calibration sweep.
        clean_loader = make_loader(test_raw, mean, std, args.window, args.test_stride, args.batch_size * 2)
        raw_clean = evaluate(models["VI+W"], clean_loader, device)

        # A) Deployment-time wavelength-noise sweep. Noise is added to W first;
        # T/F are then recomputed through the physical decoder from the same noisy W.
        for sigma_pm in noise_sigmas_pm:
            n_trials = 1 if sigma_pm == 0 else args.noise_trials
            for trial in range(n_trials):
                rng = np.random.default_rng(args.seed + 10000 * (1 + trial) + int(sigma_pm * 1000))
                noisy = add_wavelength_noise_and_decode(test_raw, sigma_pm, rng)
                loader = make_loader(noisy, mean, std, args.window, args.test_stride, args.batch_size * 2)
                for model_name in ("VI+W", "VI+TF"):
                    m = evaluate(models[model_name], loader, device)
                    noise_rows.append(
                        {
                            "split": split,
                            "model": model_name,
                            "sigma_pm": sigma_pm,
                            "trial": trial,
                            **m,
                        }
                    )

        # B) Calibration-matrix uncertainty. Raw W is unaffected; only the decoded T/F
        # branch uses an imperfect sensitivity matrix at deployment.
        calibration_rows.append(
            {
                "split": split,
                "model": "VI+W_reference",
                "calibration_sigma_pct": 0.0,
                "trial": 0,
                **raw_clean,
            }
        )
        for sigma_pct in calibration_sigmas_pct:
            n_trials = 1 if sigma_pct == 0 else args.calibration_trials
            for trial in range(n_trials):
                rng = np.random.default_rng(args.seed + 200000 + 1000 * trial + int(sigma_pct * 100))
                decoder = perturbed_decoder(sigma_pct / 100.0, rng)
                decoded = recompute_tf_from_w(test_raw, decoder)
                loader = make_loader(decoded, mean, std, args.window, args.test_stride, args.batch_size * 2)
                m = evaluate(models["VI+TF"], loader, device)
                calibration_rows.append(
                    {
                        "split": split,
                        "model": "VI+TF",
                        "calibration_sigma_pct": sigma_pct,
                        "trial": trial,
                        **m,
                    }
                )

        for model in models.values():
            del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    noise_df = pd.DataFrame(noise_rows)
    calibration_df = pd.DataFrame(calibration_rows)
    noise_df.to_csv(args.out_dir / "wavelength_noise_trials.csv", index=False)
    calibration_df.to_csv(args.out_dir / "calibration_error_trials.csv", index=False)

    noise_summary = (
        noise_df.groupby(["split", "model", "sigma_pm"])[["MAE", "RMSE", "R2", "MaxAE", "Q95_AE"]]
        .agg(["mean", "std"])
        .reset_index()
    )
    noise_summary.columns = ["_".join([str(x) for x in c if str(x)]) for c in noise_summary.columns]
    noise_summary.to_csv(args.out_dir / "wavelength_noise_summary.csv", index=False)

    cal_tf = calibration_df[calibration_df["model"] == "VI+TF"]
    cal_summary = (
        cal_tf.groupby(["split", "calibration_sigma_pct"])[["MAE", "RMSE", "R2", "MaxAE", "Q95_AE"]]
        .agg(["mean", "std"])
        .reset_index()
    )
    cal_summary.columns = ["_".join([str(x) for x in c if str(x)]) for c in cal_summary.columns]
    cal_summary.to_csv(args.out_dir / "calibration_error_summary.csv", index=False)

    print("\n=== Theoretical decoder noise amplification ===")
    print(amplification.to_string(index=False))
    print("\n=== Wavelength-noise summary (MAE) ===")
    print(
        noise_df.groupby(["split", "model", "sigma_pm"])["MAE"]
        .agg(["mean", "std"])
        .reset_index()
        .to_string(index=False)
    )
    print("\n=== Calibration-error summary for VI+TF (MAE) ===")
    print(
        cal_tf.groupby(["split", "calibration_sigma_pct"])["MAE"]
        .agg(["mean", "std"])
        .reset_index()
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()
