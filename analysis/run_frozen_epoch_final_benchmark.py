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


def get_epoch(plan: pd.DataFrame, direction: str, held_out: str, model: str) -> int:
    row = plan[
        (plan["direction"] == direction)
        & (plan["held_out_profile"] == held_out)
        & (plan["model"] == model)
    ]
    if len(row) != 1:
        raise RuntimeError(f"Expected one frozen epoch for {direction}/{held_out}/{model}, got {len(row)}")
    epoch = int(row.iloc[0]["selected_epoch"])
    if epoch < 1:
        raise RuntimeError(f"Invalid epoch {epoch}")
    return epoch


def add_row(rows, *, direction, train_rate, test_rate, held_out, model, params, seed, epoch, y, pred):
    rows.append(
        {
            "protocol": "frozen_train_only_epoch_cross_rate_unseen_profile",
            "direction": direction,
            "train_rate": train_rate,
            "test_rate": test_rate,
            "held_out_profile": held_out,
            "model": model,
            "params": params,
            "seed": seed,
            "frozen_epoch": epoch,
            "n_test": len(y),
            **metric_dict(y, pred),
        }
    )


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=Path, default=Path("data/extracted"))
    p.add_argument("--epoch-plan", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--window", type=int, default=64)
    p.add_argument("--train-stride", type=int, default=4)
    p.add_argument("--test-stride", type=int, default=1)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--envelope-quantile", type=float, default=0.005)
    p.add_argument("--calibration-quantile", type=float, default=0.99)
    args = p.parse_args()

    plan = pd.read_csv(args.epoch_plan)
    required = {"direction", "held_out_profile", "model", "selected_epoch"}
    if not required.issubset(plan.columns):
        raise RuntimeError(f"Epoch plan missing columns: {sorted(required - set(plan.columns))}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sources = load_sources(args.data)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    gate_rows: list[dict] = []
    pred_frames: list[pd.DataFrame] = []

    for train_rate, test_rate in (("1C", "2C"), ("2C", "1C")):
        direction = f"{train_rate}_to_{test_rate}"
        for held_out in PROFILES:
            train_raw = [s for s in sources if s["rate"] == train_rate and s["profile"] != held_out]
            test_raw = [s for s in sources if s["rate"] == test_rate and s["profile"] == held_out]
            if len(train_raw) != 5 or len(test_raw) != 1:
                raise RuntimeError(f"Bad split {direction}/{held_out}")

            nested_threshold, _ = nested_profile_shift_threshold(
                train_raw,
                window=args.window,
                envelope_quantile=args.envelope_quantile,
                calibration_quantile=args.calibration_quantile,
            )
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
            test_loader = DataLoader(test_ds, batch_size=args.batch_size * 2, shuffle=False, num_workers=0)

            specs = (
                ("VI", lambda: ParameterMatchedVITCN()),
                ("VI+W", lambda: PairTCN((2, 3))),
                ("VI+W-white", lambda: PairTCN((2, 3), w_white)),
            )
            predictions: dict[str, np.ndarray] = {}
            params_by_model: dict[str, int] = {}
            epochs: dict[str, int] = {}
            y_ref: np.ndarray | None = None

            print(f"\n=== final {direction}/{held_out} seed={args.seed} ===")
            for name, factory in specs:
                epochs[name] = get_epoch(plan, direction, held_out, name)
                seed_everything(args.seed)
                model = factory()
                params_by_model[name] = count_params(model)
                # Rebuild the shuffled loader after resetting the seed so model/data RNG
                # behavior is reproducible per expert without sharing iterator state.
                train_loader = DataLoader(
                    train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0
                )
                print(f"--- {name}: frozen_epoch={epochs[name]} params={params_by_model[name]} ---")
                train_model(model, train_loader, device, epochs[name], args.lr)
                y, pred, _, _ = predict_model(model, test_loader, device)
                if y_ref is None:
                    y_ref = y
                elif not np.allclose(y_ref, y):
                    raise RuntimeError("Target ordering changed")
                predictions[name] = pred
                add_row(
                    rows,
                    direction=direction,
                    train_rate=train_rate,
                    test_rate=test_rate,
                    held_out=held_out,
                    model=name,
                    params=params_by_model[name],
                    seed=args.seed,
                    epoch=epochs[name],
                    y=y,
                    pred=pred,
                )
                del model

            assert y_ref is not None
            meta = current_ood_metadata(test_raw, test_ds, q_low, q_high)
            ood_fraction = meta["ood_fraction"].to_numpy()
            gates = {
                "AnyOOD": ood_fraction > 0.0,
                "NestedCal99": ood_fraction > nested_threshold,
            }

            for gate_name, gate in gates.items():
                pred = np.where(gate, predictions["VI+W-white"], predictions["VI"])
                model_name = f"{gate_name}-VI-or-W-white"
                add_row(
                    rows,
                    direction=direction,
                    train_rate=train_rate,
                    test_rate=test_rate,
                    held_out=held_out,
                    model=model_name,
                    params=params_by_model["VI+W-white"],
                    seed=args.seed,
                    epoch=epochs["VI+W-white"],
                    y=y_ref,
                    pred=pred,
                )
                gate_rows.append(
                    {
                        "direction": direction,
                        "held_out_profile": held_out,
                        "seed": args.seed,
                        "gate": gate_name,
                        "nested_threshold": nested_threshold,
                        "fraction_windows_activated": float(gate.mean()),
                        "VI_MAE": metric_dict(y_ref, predictions["VI"])["MAE"],
                        "Wwhite_MAE": metric_dict(y_ref, predictions["VI+W-white"])["MAE"],
                        "selector_MAE": metric_dict(y_ref, pred)["MAE"],
                    }
                )

            meta.insert(0, "direction", direction)
            meta.insert(1, "held_out_profile", held_out)
            meta.insert(2, "seed", args.seed)
            meta["nested_threshold"] = nested_threshold
            meta["gate_nested"] = gates["NestedCal99"].astype(np.int8)
            meta["y"] = y_ref
            meta["pred_VI"] = predictions["VI"]
            meta["pred_W"] = predictions["VI+W"]
            meta["pred_Wwhite"] = predictions["VI+W-white"]
            meta["pred_NestedCal99"] = np.where(
                gates["NestedCal99"], predictions["VI+W-white"], predictions["VI"]
            )
            pred_frames.append(meta)

    result_df = pd.DataFrame(rows)
    gate_df = pd.DataFrame(gate_rows)
    pred_df = pd.concat(pred_frames, ignore_index=True)
    aggregate = (
        result_df.groupby(["direction", "model"], as_index=False)
        .agg(
            n_splits=("MAE", "size"),
            MAE_mean=("MAE", "mean"),
            MAE_std=("MAE", "std"),
            RMSE_mean=("RMSE", "mean"),
            R2_mean=("R2", "mean"),
            Q95_AE_mean=("Q95_AE", "mean"),
            MaxAE_mean=("MaxAE", "mean"),
        )
    )

    result_df.to_csv(args.out_dir / "split_metrics.csv", index=False)
    gate_df.to_csv(args.out_dir / "gate_summary.csv", index=False)
    pred_df.to_csv(args.out_dir / "window_predictions.csv", index=False)
    aggregate.to_csv(args.out_dir / "aggregate_summary.csv", index=False)
    print("\n=== Frozen-epoch final benchmark ===")
    print(aggregate.sort_values(["direction", "MAE_mean"]).to_string(index=False))


if __name__ == "__main__":
    main()
