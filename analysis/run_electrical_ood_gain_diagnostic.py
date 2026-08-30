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
    load_sources,
    metric_dict,
    normalize_sources,
    predict_model,
    seed_everything,
    train_model,
    train_normalizer,
)


def fit_model(pair, train_loader, args, device):
    seed_everything(args.seed)
    model = PairTCN(pair)
    train_model(model, train_loader, device, args.epochs, args.lr)
    return model


def current_window_metadata(raw_sources, dataset: WindowDataset, q_low: float, q_high: float):
    rows = []
    denom = max(q_high - q_low, 1e-12)
    for source_id, end in dataset.index:
        src = raw_sources[source_id]
        start = end - dataset.window + 1
        current = src["x"][start : end + 1, 1].astype(np.float64)
        below = np.maximum(q_low - current, 0.0)
        above = np.maximum(current - q_high, 0.0)
        exceed = below + above
        rows.append(
            {
                "profile": src["profile"],
                "rate": src["rate"],
                "end_index": end,
                "soc": float(src["y"][end]),
                "current_last": float(current[-1]),
                "current_min_window": float(current.min()),
                "current_max_window": float(current.max()),
                "ood_fraction": float(np.mean(exceed > 0)),
                "ood_max_relative_to_train_iqr": float(exceed.max() / denom),
                "ood_mean_relative_to_train_iqr": float(exceed.mean() / denom),
            }
        )
    return pd.DataFrame(rows)


def ood_bin(x: pd.Series) -> pd.Categorical:
    labels = ["ID", "OOD_0_25", "OOD_25_50", "OOD_50_75", "OOD_75_100"]
    vals = x.to_numpy()
    out = np.empty(len(vals), dtype=object)
    out[vals == 0] = labels[0]
    masks = [
        (vals > 0) & (vals <= 0.25),
        (vals > 0.25) & (vals <= 0.50),
        (vals > 0.50) & (vals <= 0.75),
        vals > 0.75,
    ]
    for label, mask in zip(labels[1:], masks):
        out[mask] = label
    return pd.Categorical(out, categories=labels, ordered=True)


def summarize(df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    rows = []
    for keys, g in df.groupby(group_cols, observed=True):
        if not isinstance(keys, tuple):
            keys = (keys,)
        base = dict(zip(group_cols, keys))
        vi_mae = float(g["ae_VI"].mean())
        w_mae = float(g["ae_VI+W"].mean())
        rows.append(
            {
                **base,
                "n": len(g),
                "VI_MAE": vi_mae,
                "VIW_MAE": w_mae,
                "absolute_gain_VI_minus_VIW": vi_mae - w_mae,
                "relative_gain": (vi_mae - w_mae) / vi_mae if vi_mae > 0 else np.nan,
                "mean_ood_fraction": float(g["ood_fraction"].mean()),
                "mean_soc": float(g["soc"].mean()),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("data/extracted"))
    parser.add_argument("--out-dir", type=Path, default=Path("results/electrical_ood_gain"))
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

    all_window_rows = []
    all_summary = []
    envelope_rows = []

    for train_rate, test_rate in (("1C", "2C"), ("2C", "1C")):
        split = f"{train_rate}_to_{test_rate}"
        train_raw = [s for s in sources if s["rate"] == train_rate]
        test_raw = [s for s in sources if s["rate"] == test_rate]

        train_current = np.concatenate([s["x"][:, 1] for s in train_raw]).astype(np.float64)
        q = args.envelope_quantile
        q_low, q_high = np.quantile(train_current, [q, 1.0 - q])
        envelope_rows.append(
            {
                "split": split,
                "train_rate": train_rate,
                "q": q,
                "current_q_low": q_low,
                "current_q_high": q_high,
                "current_min": float(train_current.min()),
                "current_max": float(train_current.max()),
            }
        )

        mean, std = train_normalizer(train_raw)
        train_sources = normalize_sources(train_raw, mean, std)
        test_sources = normalize_sources(test_raw, mean, std)
        train_ds = WindowDataset(train_sources, args.window, args.train_stride)
        test_ds = WindowDataset(test_sources, args.window, args.test_stride)
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
        test_loader = DataLoader(test_ds, batch_size=args.batch_size * 2, shuffle=False, num_workers=0)

        preds = {}
        y_ref = None
        for name, pair in (("VI", (0, 1)), ("VI+W", (2, 3))):
            print(f"\n=== {split} / {name} ===")
            model = fit_model(pair, train_loader, args, device)
            y, pred, _, _ = predict_model(model, test_loader, device)
            preds[name] = pred
            if y_ref is None:
                y_ref = y
            elif not np.allclose(y_ref, y):
                raise RuntimeError("Target ordering changed between models")

        meta = current_window_metadata(test_raw, test_ds, q_low, q_high)
        if len(meta) != len(y_ref):
            raise RuntimeError("Window metadata length does not match predictions")
        meta.insert(0, "split", split)
        meta["y"] = y_ref
        meta["pred_VI"] = preds["VI"]
        meta["pred_VI+W"] = preds["VI+W"]
        meta["ae_VI"] = np.abs(y_ref - preds["VI"])
        meta["ae_VI+W"] = np.abs(y_ref - preds["VI+W"])
        meta["gain_VI_minus_VIW"] = meta["ae_VI"] - meta["ae_VI+W"]
        meta["ood_bin"] = ood_bin(meta["ood_fraction"])
        meta["soc_bin"] = pd.cut(meta["soc"], bins=np.linspace(0, 1, 11), include_lowest=True)
        all_window_rows.append(meta)

        by_ood = summarize(meta, ["ood_bin"])
        by_ood.insert(0, "split", split)
        by_ood.insert(1, "aggregation", "ood_bin")
        all_summary.append(by_ood)

        by_profile_ood = summarize(meta, ["profile", "ood_bin"])
        by_profile_ood.insert(0, "split", split)
        by_profile_ood.insert(1, "aggregation", "profile_x_ood")
        all_summary.append(by_profile_ood)

        by_soc_ood = summarize(meta, ["soc_bin", "ood_bin"])
        by_soc_ood.insert(0, "split", split)
        by_soc_ood.insert(1, "aggregation", "soc_x_ood")
        all_summary.append(by_soc_ood)

        # Label-free association: does electrical OOD score predict where optical input helps?
        pearson = float(meta[["ood_fraction", "gain_VI_minus_VIW"]].corr(method="pearson").iloc[0, 1])
        spearman = float(meta[["ood_fraction", "gain_VI_minus_VIW"]].corr(method="spearman").iloc[0, 1])
        print(f"{split}: corr(ood_fraction, optical_gain) Pearson={pearson:.4f}, Spearman={spearman:.4f}")

    windows = pd.concat(all_window_rows, ignore_index=True)
    summary = pd.concat(all_summary, ignore_index=True, sort=False)
    envelopes = pd.DataFrame(envelope_rows)
    correlations = []
    for split, g in windows.groupby("split"):
        correlations.append(
            {
                "split": split,
                "pearson_ood_vs_gain": float(g[["ood_fraction", "gain_VI_minus_VIW"]].corr(method="pearson").iloc[0, 1]),
                "spearman_ood_vs_gain": float(g[["ood_fraction", "gain_VI_minus_VIW"]].corr(method="spearman").iloc[0, 1]),
                "fraction_windows_any_ood": float((g["ood_fraction"] > 0).mean()),
                "VI_MAE": float(g["ae_VI"].mean()),
                "VIW_MAE": float(g["ae_VI+W"].mean()),
            }
        )

    windows.to_csv(args.out_dir / "window_predictions_with_ood.csv", index=False)
    summary.to_csv(args.out_dir / "ood_gain_summary.csv", index=False)
    envelopes.to_csv(args.out_dir / "training_current_envelopes.csv", index=False)
    corr_df = pd.DataFrame(correlations)
    corr_df.to_csv(args.out_dir / "ood_gain_correlations.csv", index=False)

    print("\n=== OOD/gain correlations ===")
    print(corr_df.to_string(index=False))
    print("\n=== OOD-bin summary ===")
    print(summary[summary["aggregation"] == "ood_bin"].to_string(index=False))


if __name__ == "__main__":
    main()
