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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("data/extracted"))
    parser.add_argument("--out-dir", type=Path, default=Path("results/cross_rate_unseen_profile"))
    parser.add_argument("--window", type=int, default=64)
    parser.add_argument("--train-stride", type=int, default=4)
    parser.add_argument("--test-stride", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--envelope-quantile", type=float, default=0.005)
    args = parser.parse_args()

    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sources = load_sources(args.data)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    split_rows = []
    gate_rows = []
    corr_rows = []
    window_frames = []

    for train_rate, test_rate in (("1C", "2C"), ("2C", "1C")):
        direction = f"{train_rate}_to_{test_rate}"
        for held_out in PROFILES:
            train_raw = [
                s for s in sources
                if s["rate"] == train_rate and s["profile"] != held_out
            ]
            test_raw = [
                s for s in sources
                if s["rate"] == test_rate and s["profile"] == held_out
            ]
            if len(train_raw) != 5 or len(test_raw) != 1:
                raise RuntimeError(f"Bad split {direction}/{held_out}")

            current = np.concatenate([s["x"][:, 1] for s in train_raw]).astype(np.float64)
            q = args.envelope_quantile
            q_low, q_high = np.quantile(current, [q, 1.0 - q])

            mean, std = train_normalizer(train_raw)
            train_sources = normalize_sources(train_raw, mean, std)
            test_sources = normalize_sources(test_raw, mean, std)
            train_all = np.concatenate([s["x"] for s in train_sources], axis=0)
            w_white = whitening_matrix(train_all[:, (2, 3)])
            tf_white = whitening_matrix(train_all[:, (4, 5)])

            train_ds = WindowDataset(train_sources, args.window, args.train_stride)
            test_ds = WindowDataset(test_sources, args.window, args.test_stride)
            train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
            test_loader = DataLoader(test_ds, batch_size=args.batch_size * 2, shuffle=False, num_workers=0)

            configs = (
                ("VI", ParameterMatchedVITCN()),
                ("VI+W", PairTCN((2, 3))),
                ("VI+W-white", PairTCN((2, 3), w_white)),
                ("VI+TF", PairTCN((4, 5))),
                ("VI+TF-white", PairTCN((4, 5), tf_white)),
            )
            predictions = {}
            y_ref = None
            params = {}

            print(
                f"\n=== {direction} held-out={held_out}: "
                f"train_windows={len(train_ds)} test_windows={len(test_ds)} ==="
            )
            for model_name, model in configs:
                seed_everything(args.seed)
                params[model_name] = count_params(model)
                print(f"--- {model_name} params={params[model_name]} ---")
                train_model(model, train_loader, device, args.epochs, args.lr)
                y, pred, _, _ = predict_model(model, test_loader, device)
                if y_ref is None:
                    y_ref = y
                elif not np.allclose(y_ref, y):
                    raise RuntimeError("Target ordering changed")
                predictions[model_name] = pred
                split_rows.append(
                    {
                        "protocol": "cross_rate_plus_unseen_profile",
                        "direction": direction,
                        "train_rate": train_rate,
                        "test_rate": test_rate,
                        "held_out_profile": held_out,
                        "model": model_name,
                        "params": params[model_name],
                        "seed": args.seed,
                        "n_test": len(y),
                        **metric_dict(y, pred),
                    }
                )
                del model

            assert y_ref is not None
            meta = current_ood_metadata(test_raw, test_ds, q_low, q_high)
            any_ood = meta["ood_fraction"].to_numpy() > 0.0
            selective = np.where(any_ood, predictions["VI+W"], predictions["VI"])
            split_rows.append(
                {
                    "protocol": "cross_rate_plus_unseen_profile",
                    "direction": direction,
                    "train_rate": train_rate,
                    "test_rate": test_rate,
                    "held_out_profile": held_out,
                    "model": "OOD-selective-VI-or-W",
                    "params": params["VI+W"],
                    "seed": args.seed,
                    "n_test": len(y_ref),
                    **metric_dict(y_ref, selective),
                }
            )

            vi_ae = np.abs(y_ref - predictions["VI"])
            viw_ae = np.abs(y_ref - predictions["VI+W"])
            gain = vi_ae - viw_ae
            pearson = float(pd.Series(meta["ood_fraction"]).corr(pd.Series(gain), method="pearson"))
            spearman = float(pd.Series(meta["ood_fraction"]).corr(pd.Series(gain), method="spearman"))
            corr_rows.append(
                {
                    "direction": direction,
                    "held_out_profile": held_out,
                    "fraction_windows_any_ood": float(any_ood.mean()),
                    "pearson_ood_vs_optical_gain": pearson,
                    "spearman_ood_vs_optical_gain": spearman,
                    "mean_optical_gain": float(gain.mean()),
                    "mean_optical_gain_ID": float(gain[~any_ood].mean()) if (~any_ood).any() else np.nan,
                    "mean_optical_gain_OOD": float(gain[any_ood].mean()) if any_ood.any() else np.nan,
                }
            )

            vi_mae = float(vi_ae.mean())
            viw_mae = float(viw_ae.mean())
            sel_mae = float(np.abs(y_ref - selective).mean())
            gate_rows.append(
                {
                    "direction": direction,
                    "held_out_profile": held_out,
                    "current_q_low": q_low,
                    "current_q_high": q_high,
                    "fraction_windows_any_ood": float(any_ood.mean()),
                    "VI_MAE": vi_mae,
                    "VIW_MAE": viw_mae,
                    "selective_MAE": sel_mae,
                    "selective_vs_VI_gain": vi_mae - sel_mae,
                    "selective_vs_VIW_gain": viw_mae - sel_mae,
                }
            )

            meta.insert(0, "direction", direction)
            meta.insert(1, "held_out_profile", held_out)
            meta["y"] = y_ref
            meta["pred_VI"] = predictions["VI"]
            meta["pred_VI+W"] = predictions["VI+W"]
            meta["pred_VI+W-white"] = predictions["VI+W-white"]
            meta["pred_VI+TF"] = predictions["VI+TF"]
            meta["pred_VI+TF-white"] = predictions["VI+TF-white"]
            meta["pred_selective"] = selective
            meta["ae_VI"] = vi_ae
            meta["ae_VI+W"] = viw_ae
            meta["optical_gain"] = gain
            meta["gate_uses_W"] = any_ood.astype(np.int8)
            window_frames.append(meta)

    split_df = pd.DataFrame(split_rows)
    gate_df = pd.DataFrame(gate_rows)
    corr_df = pd.DataFrame(corr_rows)
    windows = pd.concat(window_frames, ignore_index=True)

    aggregate = []
    for direction, d0 in list(split_df.groupby("direction")) + [("ALL", split_df)]:
        for model, g in d0.groupby("model"):
            aggregate.append(
                {
                    "direction": direction,
                    "model": model,
                    "n_splits": len(g),
                    "MAE_mean": float(g["MAE"].mean()),
                    "MAE_std": float(g["MAE"].std(ddof=1)) if len(g) > 1 else np.nan,
                    "RMSE_mean": float(g["RMSE"].mean()),
                    "R2_mean": float(g["R2"].mean()),
                    "Q95_AE_mean": float(g["Q95_AE"].mean()),
                }
            )
    aggregate_df = pd.DataFrame(aggregate)

    split_df.to_csv(args.out_dir / "split_metrics.csv", index=False)
    aggregate_df.to_csv(args.out_dir / "aggregate_summary.csv", index=False)
    gate_df.to_csv(args.out_dir / "gate_summary.csv", index=False)
    corr_df.to_csv(args.out_dir / "ood_gain_correlations.csv", index=False)
    windows.to_csv(args.out_dir / "window_predictions.csv", index=False)

    print("\n=== Cross-rate + unseen-profile aggregate ===")
    print(aggregate_df.sort_values(["direction", "MAE_mean"]).to_string(index=False))
    print("\n=== Gate summary ===")
    print(gate_df.to_string(index=False))
    print("\n=== OOD vs optical gain ===")
    print(corr_df.to_string(index=False))


if __name__ == "__main__":
    main()
