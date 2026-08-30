from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from run_nested_calibrated_selector import nested_profile_shift_threshold
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


def optical_radius_ood_fraction(
    raw_sources: list[dict],
    dataset: WindowDataset,
    mean: np.ndarray,
    std: np.ndarray,
    white: np.ndarray,
    radius_high: float,
) -> np.ndarray:
    fractions = []
    for source_id, end in dataset.index:
        src = raw_sources[source_id]
        start = end - dataset.window + 1
        x = ((src["x"][start : end + 1] - mean) / std).astype(np.float64)
        z = x[:, (2, 3)] @ white.astype(np.float64).T
        radius = np.linalg.norm(z, axis=1)
        fractions.append(float(np.mean(radius > radius_high)))
    return np.asarray(fractions, dtype=np.float64)


def nested_optical_support_threshold(
    train_raw: list[dict],
    *,
    window: int,
    radius_quantile: float,
    calibration_quantile: float,
) -> float:
    internal_fractions = []
    profiles = [s["profile"] for s in train_raw]
    for pseudo_held in profiles:
        inner_train_raw = [s for s in train_raw if s["profile"] != pseudo_held]
        inner_test_raw = [s for s in train_raw if s["profile"] == pseudo_held]
        mean, std = train_normalizer(inner_train_raw)
        inner_train = normalize_sources(inner_train_raw, mean, std)
        train_all = np.concatenate([s["x"] for s in inner_train], axis=0)
        white = whitening_matrix(train_all[:, (2, 3)])
        radius = np.linalg.norm(train_all[:, (2, 3)].astype(np.float64) @ white.astype(np.float64).T, axis=1)
        radius_high = float(np.quantile(radius, radius_quantile))
        pseudo_ds = WindowDataset(inner_test_raw, window, 1)
        internal_fractions.append(
            optical_radius_ood_fraction(inner_test_raw, pseudo_ds, mean, std, white, radius_high)
        )
    return float(np.quantile(np.concatenate(internal_fractions), calibration_quantile))


def add_result(rows, *, direction, train_rate, test_rate, held_out, model, seed, params, y, pred):
    rows.append(
        {
            "protocol": "cross_rate_unseen_profile_support_complementarity",
            "direction": direction,
            "train_rate": train_rate,
            "test_rate": test_rate,
            "held_out_profile": held_out,
            "model": model,
            "params": params,
            "seed": seed,
            "n_test": len(y),
            **metric_dict(y, pred),
        }
    )


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=Path, default=Path("data/extracted"))
    p.add_argument("--out-dir", type=Path, default=Path("results/support_complementarity"))
    p.add_argument("--window", type=int, default=64)
    p.add_argument("--train-stride", type=int, default=4)
    p.add_argument("--test-stride", type=int, default=1)
    p.add_argument("--epochs", type=int, default=6)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--envelope-quantile", type=float, default=0.005)
    p.add_argument("--calibration-quantile", type=float, default=0.99)
    p.add_argument("--optical-radius-quantile", type=float, default=0.995)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sources = load_sources(args.data)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    rows, gate_rows = [], []

    for train_rate, test_rate in (("1C", "2C"), ("2C", "1C")):
        direction = f"{train_rate}_to_{test_rate}"
        for held_out in PROFILES:
            train_raw = [s for s in sources if s["rate"] == train_rate and s["profile"] != held_out]
            test_raw = [s for s in sources if s["rate"] == test_rate and s["profile"] == held_out]

            e_threshold, _ = nested_profile_shift_threshold(
                train_raw,
                window=args.window,
                envelope_quantile=args.envelope_quantile,
                calibration_quantile=args.calibration_quantile,
            )
            o_threshold = nested_optical_support_threshold(
                train_raw,
                window=args.window,
                radius_quantile=args.optical_radius_quantile,
                calibration_quantile=args.calibration_quantile,
            )

            current = np.concatenate([s["x"][:, 1] for s in train_raw]).astype(np.float64)
            q_low, q_high = np.quantile(
                current, [args.envelope_quantile, 1.0 - args.envelope_quantile]
            )
            mean, std = train_normalizer(train_raw)
            train_sources = normalize_sources(train_raw, mean, std)
            test_sources = normalize_sources(test_raw, mean, std)
            train_all = np.concatenate([s["x"] for s in train_sources], axis=0)
            w_white = whitening_matrix(train_all[:, (2, 3)])
            radius = np.linalg.norm(
                train_all[:, (2, 3)].astype(np.float64) @ w_white.astype(np.float64).T,
                axis=1,
            )
            radius_high = float(np.quantile(radius, args.optical_radius_quantile))

            train_ds = WindowDataset(train_sources, args.window, args.train_stride)
            test_ds = WindowDataset(test_sources, args.window, args.test_stride)
            train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
            test_loader = DataLoader(test_ds, batch_size=args.batch_size * 2, shuffle=False, num_workers=0)

            specs = (
                ("VI", lambda: ParameterMatchedVITCN()),
                ("VI+W", lambda: PairTCN((2, 3))),
                ("VI+W-white", lambda: PairTCN((2, 3), w_white)),
            )
            predictions, params = {}, {}
            y_ref = None
            print(
                f"\n=== {direction}/{held_out} e_thr={e_threshold:.6f} "
                f"o_thr={o_threshold:.6f} ==="
            )
            for name, factory in specs:
                seed_everything(args.seed)
                model = factory()
                params[name] = count_params(model)
                train_model(model, train_loader, device, args.epochs, args.lr)
                y, pred, _, _ = predict_model(model, test_loader, device)
                if y_ref is None:
                    y_ref = y
                elif not np.allclose(y_ref, y):
                    raise RuntimeError("Target ordering changed")
                predictions[name] = pred
                add_result(
                    rows,
                    direction=direction,
                    train_rate=train_rate,
                    test_rate=test_rate,
                    held_out=held_out,
                    model=name,
                    seed=args.seed,
                    params=params[name],
                    y=y,
                    pred=pred,
                )
                del model

            assert y_ref is not None
            e_meta = current_ood_metadata(test_raw, test_ds, q_low, q_high)
            e_frac = e_meta["ood_fraction"].to_numpy()
            o_frac = optical_radius_ood_fraction(
                test_raw, test_ds, mean, std, w_white, radius_high
            )
            gates = {
                "AnyOOD": e_frac > 0.0,
                "NestedCal99": e_frac > e_threshold,
                "Complementarity": (e_frac > e_threshold) & (o_frac <= o_threshold),
            }

            for expert in ("VI+W", "VI+W-white"):
                for gate_name, gate in gates.items():
                    pred = np.where(gate, predictions[expert], predictions["VI"])
                    name = f"{gate_name}-VI-or-{expert.removeprefix('VI+')}"
                    add_result(
                        rows,
                        direction=direction,
                        train_rate=train_rate,
                        test_rate=test_rate,
                        held_out=held_out,
                        model=name,
                        seed=args.seed,
                        params=params[expert],
                        y=y_ref,
                        pred=pred,
                    )
                    gate_rows.append(
                        {
                            "direction": direction,
                            "held_out_profile": held_out,
                            "seed": args.seed,
                            "expert": expert,
                            "gate": gate_name,
                            "electrical_threshold": e_threshold,
                            "optical_threshold": o_threshold,
                            "electrical_activation": float(np.mean(e_frac > e_threshold)),
                            "optical_in_support": float(np.mean(o_frac <= o_threshold)),
                            "fraction_windows_activated": float(gate.mean()),
                            "selector_MAE": metric_dict(y_ref, pred)["MAE"],
                        }
                    )

    result_df = pd.DataFrame(rows)
    gate_df = pd.DataFrame(gate_rows)
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
    aggregate.to_csv(args.out_dir / "aggregate_summary.csv", index=False)
    print("\n=== Support-complementarity aggregate ===")
    print(aggregate.sort_values(["direction", "MAE_mean"]).to_string(index=False))


if __name__ == "__main__":
    main()
