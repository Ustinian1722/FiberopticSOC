from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from run_representation_conditioning_diagnostic import PairTCN, whitening_matrix
from run_same_rate_lopo_selective_optical import ParameterMatchedVITCN, current_ood_metadata
from run_sequence_representation_benchmark import (
    PROFILES,
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


def nested_profile_shift_threshold(
    train_raw: list[dict],
    *,
    window: int,
    envelope_quantile: float,
    calibration_quantile: float,
) -> tuple[float, pd.DataFrame]:
    """Estimate a normal profile-shift OOD ceiling using training data only.

    For each of the five main-training profiles, build a current envelope from the
    other four profiles and score the pseudo-held-out profile.  The final threshold
    is a high quantile of these internal OOD fractions.  SOC is never consulted.
    """
    calibration_frames: list[pd.DataFrame] = []
    profiles = [s["profile"] for s in train_raw]
    for pseudo_held in profiles:
        inner_train = [s for s in train_raw if s["profile"] != pseudo_held]
        inner_test = [s for s in train_raw if s["profile"] == pseudo_held]
        current = np.concatenate([s["x"][:, 1] for s in inner_train]).astype(np.float64)
        q = envelope_quantile
        q_low, q_high = np.quantile(current, [q, 1.0 - q])
        inner_ds = WindowDataset(inner_test, window, 1)
        meta = current_ood_metadata(inner_test, inner_ds, q_low, q_high)
        meta["pseudo_held_profile"] = pseudo_held
        calibration_frames.append(meta)

    calibration = pd.concat(calibration_frames, ignore_index=True)
    threshold = float(np.quantile(calibration["ood_fraction"].to_numpy(), calibration_quantile))
    return threshold, calibration


def add_result(
    rows: list[dict],
    *,
    direction: str,
    train_rate: str,
    test_rate: str,
    held_out_profile: str,
    model: str,
    seed: int,
    params: int,
    y: np.ndarray,
    pred: np.ndarray,
) -> None:
    rows.append(
        {
            "protocol": "cross_rate_plus_unseen_profile_nested_support_calibration",
            "direction": direction,
            "train_rate": train_rate,
            "test_rate": test_rate,
            "held_out_profile": held_out_profile,
            "model": model,
            "params": params,
            "seed": seed,
            "n_test": len(y),
            **metric_dict(y, pred),
        }
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("data/extracted"))
    parser.add_argument("--out-dir", type=Path, default=Path("results/nested_calibrated_selector"))
    parser.add_argument("--window", type=int, default=64)
    parser.add_argument("--train-stride", type=int, default=4)
    parser.add_argument("--test-stride", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--envelope-quantile", type=float, default=0.005)
    parser.add_argument("--calibration-quantile", type=float, default=0.99)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sources = load_sources(args.data)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    result_rows: list[dict] = []
    gate_rows: list[dict] = []
    calibration_rows: list[dict] = []

    for train_rate, test_rate in (("1C", "2C"), ("2C", "1C")):
        direction = f"{train_rate}_to_{test_rate}"
        for held_out in PROFILES:
            train_raw = [s for s in sources if s["rate"] == train_rate and s["profile"] != held_out]
            test_raw = [s for s in sources if s["rate"] == test_rate and s["profile"] == held_out]
            if len(train_raw) != 5 or len(test_raw) != 1:
                raise RuntimeError(f"Bad split {direction}/{held_out}")

            nested_threshold, calibration = nested_profile_shift_threshold(
                train_raw,
                window=args.window,
                envelope_quantile=args.envelope_quantile,
                calibration_quantile=args.calibration_quantile,
            )
            calibration["direction"] = direction
            calibration["main_held_out_profile"] = held_out
            calibration["nested_threshold"] = nested_threshold
            calibration_rows.extend(calibration.to_dict("records"))

            current = np.concatenate([s["x"][:, 1] for s in train_raw]).astype(np.float64)
            q = args.envelope_quantile
            q_low, q_high = np.quantile(current, [q, 1.0 - q])

            mean, std = train_normalizer(train_raw)
            train_sources = normalize_sources(train_raw, mean, std)
            test_sources = normalize_sources(test_raw, mean, std)
            train_all = np.concatenate([s["x"] for s in train_sources], axis=0)
            w_white = whitening_matrix(train_all[:, (2, 3)])

            train_ds = WindowDataset(train_sources, args.window, args.train_stride)
            test_ds = WindowDataset(test_sources, args.window, args.test_stride)
            train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
            test_loader = DataLoader(test_ds, batch_size=args.batch_size * 2, shuffle=False, num_workers=0)

            specs = (
                ("VI", lambda: ParameterMatchedVITCN()),
                ("VI+W", lambda: PairTCN((2, 3))),
                ("VI+W-white", lambda: PairTCN((2, 3), w_white)),
            )
            predictions: dict[str, np.ndarray] = {}
            params: dict[str, int] = {}
            y_ref: np.ndarray | None = None

            print(
                f"\n=== {direction}/{held_out}: nested_q{args.calibration_quantile:.3f}="
                f"{nested_threshold:.6f} ==="
            )
            for name, factory in specs:
                seed_everything(args.seed)
                model = factory()
                params[name] = count_params(model)
                print(f"--- {name} params={params[name]} ---")
                train_model(model, train_loader, device, args.epochs, args.lr)
                y, pred, _, _ = predict_model(model, test_loader, device)
                if y_ref is None:
                    y_ref = y
                elif not np.allclose(y_ref, y):
                    raise RuntimeError("Target ordering changed")
                predictions[name] = pred
                add_result(
                    result_rows,
                    direction=direction,
                    train_rate=train_rate,
                    test_rate=test_rate,
                    held_out_profile=held_out,
                    model=name,
                    seed=args.seed,
                    params=params[name],
                    y=y,
                    pred=pred,
                )
                del model

            assert y_ref is not None
            meta = current_ood_metadata(test_raw, test_ds, q_low, q_high)
            frac = meta["ood_fraction"].to_numpy()
            any_gate = frac > 0.0
            calibrated_gate = frac > nested_threshold

            for expert in ("VI+W", "VI+W-white"):
                for gate_name, gate in (
                    ("AnyOOD", any_gate),
                    ("NestedCal99", calibrated_gate),
                ):
                    pred = np.where(gate, predictions[expert], predictions["VI"])
                    model_name = f"{gate_name}-VI-or-{expert.removeprefix('VI+') }"
                    add_result(
                        result_rows,
                        direction=direction,
                        train_rate=train_rate,
                        test_rate=test_rate,
                        held_out_profile=held_out,
                        model=model_name,
                        seed=args.seed,
                        params=params[expert],
                        y=y_ref,
                        pred=pred,
                    )
                    gate_rows.append(
                        {
                            "direction": direction,
                            "held_out_profile": held_out,
                            "expert": expert,
                            "gate": gate_name,
                            "seed": args.seed,
                            "nested_threshold": nested_threshold,
                            "fraction_windows_activated": float(gate.mean()),
                            "VI_MAE": metric_dict(y_ref, predictions["VI"])["MAE"],
                            "expert_MAE": metric_dict(y_ref, predictions[expert])["MAE"],
                            "selector_MAE": metric_dict(y_ref, pred)["MAE"],
                        }
                    )

    result_df = pd.DataFrame(result_rows)
    gate_df = pd.DataFrame(gate_rows)
    calibration_df = pd.DataFrame(calibration_rows)

    aggregate = (
        result_df.groupby(["direction", "model"], as_index=False)
        .agg(
            n_splits=("MAE", "size"),
            MAE_mean=("MAE", "mean"),
            MAE_std=("MAE", "std"),
            RMSE_mean=("RMSE", "mean"),
            R2_mean=("R2", "mean"),
            Q95_AE_mean=("Q95_AE", "mean"),
        )
    )

    result_df.to_csv(args.out_dir / "split_metrics.csv", index=False)
    gate_df.to_csv(args.out_dir / "gate_summary.csv", index=False)
    calibration_df.to_csv(args.out_dir / "nested_calibration_windows.csv", index=False)
    aggregate.to_csv(args.out_dir / "aggregate_summary.csv", index=False)

    print("\n=== Nested support-calibrated selector aggregate ===")
    print(aggregate.sort_values(["direction", "MAE_mean"]).to_string(index=False))
    print("\n=== Gate activation summary ===")
    print(
        gate_df.groupby(["direction", "gate", "expert"], as_index=False)[
            ["fraction_windows_activated", "selector_MAE"]
        ].mean().to_string(index=False)
    )


if __name__ == "__main__":
    main()
