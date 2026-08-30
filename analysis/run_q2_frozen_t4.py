from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from run_nested_epoch_selection import train_fixed_epochs
from run_q2_nested_epoch_selection import MODEL_NAMES, model_factory
from run_sequence_representation_benchmark import (
    PROFILES,
    WindowDataset,
    count_params,
    load_sources,
    metric_dict,
    normalize_sources,
    predict_model,
    seed_everything,
    train_normalizer,
)

EXPECTED_DIRECTIONS = ("1C_to_2C", "2C_to_1C")
K = np.array([[0.0208, 0.00054], [0.0254, 0.00085]], dtype=np.float64)
K_INV = np.linalg.inv(K)
TF_MODELS = {"IUTF-StackedTCN", "DualBranch-Direct-TF", "ETMF-TF"}


def validate_plan(plan: pd.DataFrame, models: tuple[str, ...]) -> None:
    required = {"direction", "held_out_profile", "model", "selected_epoch"}
    missing = required.difference(plan.columns)
    if missing:
        raise ValueError(f"Epoch plan missing columns: {sorted(missing)}")
    plan = plan[plan["model"].isin(models)].copy()
    expected = len(EXPECTED_DIRECTIONS) * len(PROFILES) * len(models)
    if len(plan) != expected:
        raise ValueError(f"Expected {expected} selected rows, found {len(plan)}")
    if plan.duplicated(["direction", "held_out_profile", "model"]).any():
        raise ValueError("Duplicate direction/profile/model rows in epoch plan")
    if not set(plan["direction"]).issubset(EXPECTED_DIRECTIONS):
        raise ValueError(f"Unknown directions: {sorted(set(plan['direction']) - set(EXPECTED_DIRECTIONS))}")
    if (plan["selected_epoch"].astype(int) < 1).any():
        raise ValueError("selected_epoch must be >=1")


def rates_from_direction(direction: str) -> tuple[str, str]:
    if direction == "1C_to_2C":
        return "1C", "2C"
    if direction == "2C_to_1C":
        return "2C", "1C"
    raise ValueError(direction)


def test_metadata(test_raw: list[dict], test_ds: WindowDataset) -> pd.DataFrame:
    rows = []
    for source_id, end in test_ds.index:
        src = test_raw[source_id]
        rows.append(
            {
                "source_name": src["name"],
                "profile": src["profile"],
                "rate": src["rate"],
                "row_end": int(end),
                "current_A": float(src["x"][end, 1]),
                "voltage_V": float(src["x"][end, 0]),
                "soc_true_raw": float(src["y"][end]),
            }
        )
    return pd.DataFrame(rows)


def noisy_fbg_sources(
    sources: list[dict],
    *,
    sigma_pm: float,
    rng: np.random.Generator,
) -> list[dict]:
    """Perturb measured W1/W2 then recompute T/F through the published linear decoupling.

    Noise is added in the measurement coordinate before normalization. This keeps the
    robustness experiment tied to FBG wavelength resolution instead of arbitrary
    percentages of the derived T/F features.
    """
    sigma_nm = float(sigma_pm) * 1e-3
    out = []
    for src in sources:
        item = dict(src)
        x = src["x"].astype(np.float64, copy=True)
        w = x[:, 2:4]
        noisy_w = w + rng.normal(0.0, sigma_nm, size=w.shape)
        noisy_tf = noisy_w @ K_INV.T
        x[:, 2:4] = noisy_w
        x[:, 4:6] = noisy_tf
        item["x"] = x.astype(np.float32)
        out.append(item)
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=Path, default=Path("data/extracted/SiC-18"))
    p.add_argument("--epoch-plan", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--models", type=str, default=",".join(MODEL_NAMES))
    p.add_argument("--window", type=int, default=64)
    p.add_argument("--train-stride", type=int, default=4)
    p.add_argument("--test-stride", type=int, default=1)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--noise-pm", type=float, nargs="*", default=[0.5, 1.0, 2.0])
    p.add_argument("--noise-draws", type=int, default=5)
    args = p.parse_args()

    models = tuple(x.strip() for x in args.models.split(",") if x.strip())
    bad = [m for m in models if m not in MODEL_NAMES]
    if bad:
        raise ValueError(f"Unknown models {bad}")
    if args.noise_draws < 1:
        raise ValueError("noise_draws must be >=1")

    plan = pd.read_csv(args.epoch_plan)
    validate_plan(plan, models)
    plan = plan[plan["model"].isin(models)].copy()
    sources = load_sources(args.data)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    metric_rows: list[dict] = []
    pred_rows: list[pd.DataFrame] = []
    noise_rows: list[dict] = []

    for direction_idx, direction in enumerate(EXPECTED_DIRECTIONS):
        train_rate, test_rate = rates_from_direction(direction)
        for profile_idx, held_out in enumerate(PROFILES):
            train_raw = [
                s for s in sources
                if s["rate"] == train_rate and s["profile"] != held_out
            ]
            test_raw = [
                s for s in sources
                if s["rate"] == test_rate and s["profile"] == held_out
            ]
            if len(train_raw) != 5 or len(test_raw) != 1:
                raise RuntimeError(f"Bad T4 split {direction}/{held_out}")

            mean, std = train_normalizer(train_raw)
            train = normalize_sources(train_raw, mean, std)
            test = normalize_sources(test_raw, mean, std)
            train_ds = WindowDataset(train, args.window, args.train_stride)
            test_ds = WindowDataset(test, args.window, args.test_stride)
            test_loader = DataLoader(
                test_ds, batch_size=args.batch_size * 2, shuffle=False, num_workers=0
            )
            meta = test_metadata(test_raw, test_ds)

            print(
                f"\n=== FINAL T4 seed={args.seed} {direction}/{held_out} "
                f"train={len(train_ds)} test={len(test_ds)} ==="
            )
            for model_name in models:
                selected = plan[
                    (plan["direction"] == direction)
                    & (plan["held_out_profile"] == held_out)
                    & (plan["model"] == model_name)
                ]
                if len(selected) != 1:
                    raise RuntimeError(f"Missing unique epoch for {direction}/{held_out}/{model_name}")
                epoch = int(selected.iloc[0]["selected_epoch"])

                # Fair final RNG start: reset before construction and then build the shuffled loader.
                seed_everything(args.seed)
                model = model_factory(model_name)
                train_loader = DataLoader(
                    train_ds,
                    batch_size=args.batch_size,
                    shuffle=True,
                    num_workers=0,
                )
                params = count_params(model)
                print(f"--- {model_name} epoch={epoch} params={params} ---")
                train_fixed_epochs(model, train_loader, device, epoch, args.lr)
                y, pred, _, gate = predict_model(model, test_loader, device)
                if len(y) != len(meta):
                    raise RuntimeError("Prediction/metadata alignment mismatch")
                clean_metrics = metric_dict(y, pred)
                metric_rows.append(
                    {
                        "protocol": "T4_cross_rate_unseen_profile",
                        "direction": direction,
                        "train_rate": train_rate,
                        "test_rate": test_rate,
                        "held_out_profile": held_out,
                        "model": model_name,
                        "seed": args.seed,
                        "selected_epoch": epoch,
                        "params": params,
                        "n_test": len(y),
                        **clean_metrics,
                    }
                )

                frame = meta.copy()
                frame["protocol"] = "T4_cross_rate_unseen_profile"
                frame["direction"] = direction
                frame["held_out_profile"] = held_out
                frame["model"] = model_name
                frame["seed"] = args.seed
                frame["selected_epoch"] = epoch
                frame["soc_true"] = y
                frame["soc_pred"] = pred
                frame["abs_error"] = np.abs(y - pred)
                frame["signed_error"] = pred - y
                frame["soc_bin_10pct"] = np.minimum((y * 10).astype(int), 9)
                frame["abs_current_A"] = np.abs(frame["current_A"].to_numpy(float))
                frame["alpha"] = gate if gate is not None else np.nan
                pred_rows.append(frame)

                # Test-time FBG measurement-noise robustness. Training data/model stay fixed.
                if model_name in TF_MODELS:
                    for sigma_pm in args.noise_pm:
                        for draw in range(args.noise_draws):
                            noise_seed = (
                                10_000_000 * args.seed
                                + 100_000 * direction_idx
                                + 1_000 * profile_idx
                                + 100 * int(round(sigma_pm * 10))
                                + draw
                            )
                            rng = np.random.default_rng(noise_seed)
                            noisy_raw = noisy_fbg_sources(test_raw, sigma_pm=sigma_pm, rng=rng)
                            noisy_norm = normalize_sources(noisy_raw, mean, std)
                            noisy_ds = WindowDataset(noisy_norm, args.window, args.test_stride)
                            noisy_loader = DataLoader(
                                noisy_ds,
                                batch_size=args.batch_size * 2,
                                shuffle=False,
                                num_workers=0,
                            )
                            yn, pn, _, _ = predict_model(model, noisy_loader, device)
                            nm = metric_dict(yn, pn)
                            noise_rows.append(
                                {
                                    "protocol": "T4_test_time_FBG_noise",
                                    "direction": direction,
                                    "held_out_profile": held_out,
                                    "model": model_name,
                                    "seed": args.seed,
                                    "selected_epoch": epoch,
                                    "sigma_pm_each_wavelength": float(sigma_pm),
                                    "noise_draw": draw,
                                    "noise_seed": noise_seed,
                                    "clean_MAE": float(clean_metrics["MAE"]),
                                    "MAE_increase": float(nm["MAE"] - clean_metrics["MAE"]),
                                    **nm,
                                }
                            )

                del model
                if device.type == "cuda":
                    torch.cuda.empty_cache()

    metrics = pd.DataFrame(metric_rows)
    predictions = pd.concat(pred_rows, ignore_index=True)
    aggregate = (
        metrics.groupby(["direction", "model"], as_index=False)
        .agg(
            n_profiles=("MAE", "size"),
            MAE_mean=("MAE", "mean"),
            MAE_std=("MAE", "std"),
            RMSE_mean=("RMSE", "mean"),
            R2_mean=("R2", "mean"),
            Q95_AE_mean=("Q95_AE", "mean"),
            MaxAE_mean=("MaxAE", "mean"),
        )
    )
    metrics.to_csv(args.out_dir / "split_metrics.csv", index=False)
    predictions.to_csv(args.out_dir / "window_predictions.csv", index=False)
    aggregate.to_csv(args.out_dir / "aggregate_summary.csv", index=False)

    if noise_rows:
        noise_df = pd.DataFrame(noise_rows)
        noise_summary = (
            noise_df.groupby(
                ["direction", "model", "sigma_pm_each_wavelength"], as_index=False
            )
            .agg(
                n_evaluations=("MAE", "size"),
                MAE_mean=("MAE", "mean"),
                MAE_std=("MAE", "std"),
                clean_MAE_mean=("clean_MAE", "mean"),
                MAE_increase_mean=("MAE_increase", "mean"),
                RMSE_mean=("RMSE", "mean"),
                Q95_AE_mean=("Q95_AE", "mean"),
            )
        )
        noise_df.to_csv(args.out_dir / "fbg_noise_metrics.csv", index=False)
        noise_summary.to_csv(args.out_dir / "fbg_noise_summary.csv", index=False)
        print("\n=== test-time FBG wavelength-noise robustness ===")
        print(noise_summary.to_string(index=False))

    print("\n=== final T4 seed summary ===")
    print(aggregate.sort_values(["direction", "MAE_mean"]).to_string(index=False))


if __name__ == "__main__":
    main()
