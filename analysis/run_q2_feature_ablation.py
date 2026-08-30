from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader

from run_sequence_representation_benchmark import (
    PROFILES,
    SingleViewTCN,
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

FEATURES = {
    "IU": (0, 1),
    "IUT": (0, 1, 4),
    "IUF": (0, 1, 5),
    "IUTF": (0, 1, 4, 5),
    "IUW": (0, 1, 2, 3),
}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=Path, default=Path("data/extracted/SiC-18"))
    p.add_argument("--out-dir", type=Path, default=Path("results/q2_feature_ablation"))
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--window", type=int, default=64)
    p.add_argument("--train-stride", type=int, default=4)
    p.add_argument("--test-stride", type=int, default=1)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    sources = load_sources(args.data)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows = []

    for held_out in PROFILES:
        train_raw = [
            s for s in sources if s["rate"] == "1C" and s["profile"] != held_out
        ]
        test_raw = [
            s for s in sources if s["rate"] == "2C" and s["profile"] == held_out
        ]
        mean, std = train_normalizer(train_raw)
        train = normalize_sources(train_raw, mean, std)
        test = normalize_sources(test_raw, mean, std)
        train_ds = WindowDataset(train, args.window, args.train_stride)
        test_ds = WindowDataset(test, args.window, args.test_stride)
        test_loader = DataLoader(
            test_ds, batch_size=args.batch_size * 2, shuffle=False, num_workers=0
        )

        for name, idx in FEATURES.items():
            seed_everything(args.seed)
            model = SingleViewTCN(idx)
            train_loader = DataLoader(
                train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0
            )
            train_model(model, train_loader, device, args.epochs, args.lr)
            y, pred, _, _ = predict_model(model, test_loader, device)
            rows.append(
                {
                    "protocol": "development_T4_1C_to_2C_unseen_profile",
                    "held_out_profile": held_out,
                    "feature_set": name,
                    "feature_indices": ",".join(map(str, idx)),
                    "seed": args.seed,
                    "epochs": args.epochs,
                    "params": count_params(model),
                    **metric_dict(y, pred),
                }
            )

    df = pd.DataFrame(rows)
    agg = df.groupby("feature_set", as_index=False).agg(
        n_profiles=("MAE", "size"),
        MAE_mean=("MAE", "mean"),
        MAE_std=("MAE", "std"),
        RMSE_mean=("RMSE", "mean"),
        R2_mean=("R2", "mean"),
        Q95_AE_mean=("Q95_AE", "mean"),
    )
    # Paired per-profile gains against architecture-matched electrical-only IU.
    pivot = df.pivot(index="held_out_profile", columns="feature_set", values="MAE")
    gains = []
    for name in FEATURES:
        if name == "IU":
            continue
        g = pivot["IU"] - pivot[name]
        gains.append(
            {
                "feature_set": name,
                "mean_MAE_gain_vs_IU": float(g.mean()),
                "wins_vs_IU": int((g > 0).sum()),
                "losses_vs_IU": int((g < 0).sum()),
            }
        )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out_dir / "split_metrics.csv", index=False)
    agg.to_csv(args.out_dir / "aggregate_summary.csv", index=False)
    pd.DataFrame(gains).to_csv(args.out_dir / "paired_feature_gains.csv", index=False)
    print(agg.sort_values("MAE_mean").to_string(index=False))
    print(pd.DataFrame(gains).sort_values("mean_MAE_gain_vs_IU", ascending=False).to_string(index=False))
    print("Development-only architecture-matched feature screen; not final publication statistics.")


if __name__ == "__main__":
    main()
