from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from run_sequence_representation_benchmark import (
    TCNEncoder,
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


class PairTCN(nn.Module):
    def __init__(self, pair: tuple[int, int], whiten: np.ndarray | None = None, hidden: int = 24):
        super().__init__()
        self.pair = pair
        if whiten is None:
            self.register_buffer("whiten", torch.empty(0), persistent=False)
        else:
            self.register_buffer("whiten", torch.tensor(whiten, dtype=torch.float32), persistent=True)
        self.encoder = TCNEncoder(4, hidden)
        self.head = nn.Sequential(nn.Linear(hidden, hidden), nn.GELU(), nn.Linear(hidden, 1))

    def forward(self, x: torch.Tensor):
        electrical = x[:, :, (0, 1)]
        optical = x[:, :, self.pair]
        if self.whiten.numel():
            optical = torch.matmul(optical, self.whiten.T)
        z = torch.cat([electrical, optical], dim=-1)
        h = self.encoder(z)
        return self.head(h).squeeze(-1), None


def covariance_condition(x: np.ndarray) -> float:
    cov = np.cov(x.T)
    return float(np.linalg.cond(cov))


def whitening_matrix(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    cov = np.cov(x.T)
    eigval, eigvec = np.linalg.eigh(cov)
    eigval = np.maximum(eigval, eps)
    return (eigvec @ np.diag(1.0 / np.sqrt(eigval)) @ eigvec.T).astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("data/extracted"))
    parser.add_argument("--out-dir", type=Path, default=Path("results/representation_conditioning"))
    parser.add_argument("--window", type=int, default=64)
    parser.add_argument("--train-stride", type=int, default=4)
    parser.add_argument("--test-stride", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sources = load_sources(args.data)
    results = []
    condition_rows = []

    for train_rate, test_rate in (("1C", "2C"), ("2C", "1C")):
        split = f"{train_rate}_to_{test_rate}"
        train_raw = [s for s in sources if s["rate"] == train_rate]
        test_raw = [s for s in sources if s["rate"] == test_rate]
        mean, std = train_normalizer(train_raw)
        train_sources = normalize_sources(train_raw, mean, std)
        test_sources = normalize_sources(test_raw, mean, std)

        train_all = np.concatenate([s["x"] for s in train_sources], axis=0)
        transforms = {}
        for label, pair in (("W", (2, 3)), ("TF", (4, 5))):
            x_pair = train_all[:, pair]
            raw_cond = covariance_condition(x_pair)
            white = whitening_matrix(x_pair)
            white_cond = covariance_condition(x_pair @ white.T)
            transforms[label] = white
            condition_rows.append(
                {
                    "split": split,
                    "representation": label,
                    "condition_before": raw_cond,
                    "condition_after_whitening": white_cond,
                    "corr_before": float(np.corrcoef(x_pair.T)[0, 1]),
                }
            )

        train_ds = WindowDataset(train_sources, args.window, args.train_stride)
        test_ds = WindowDataset(test_sources, args.window, args.test_stride)
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
        test_loader = DataLoader(test_ds, batch_size=args.batch_size * 2, shuffle=False, num_workers=0)

        configs = (
            ("VI+W", (2, 3), None),
            ("VI+W-white", (2, 3), transforms["W"]),
            ("VI+TF", (4, 5), None),
            ("VI+TF-white", (4, 5), transforms["TF"]),
        )
        for name, pair, white in configs:
            seed_everything(args.seed)
            model = PairTCN(pair, white)
            params = count_params(model)
            print(f"\n=== split={split} model={name} params={params} ===")
            train_model(model, train_loader, device, args.epochs, args.lr)
            y, pred, _, _ = predict_model(model, test_loader, device)
            results.append(
                {
                    "split": split,
                    "model": name,
                    "params": params,
                    "window": args.window,
                    "epochs": args.epochs,
                    "seed": args.seed,
                    "n_test": len(y),
                    **metric_dict(y, pred),
                }
            )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    result_df = pd.DataFrame(results)
    cond_df = pd.DataFrame(condition_rows)
    result_df.to_csv(args.out_dir / "whitening_results.csv", index=False)
    cond_df.to_csv(args.out_dir / "representation_condition_numbers.csv", index=False)

    print("\n=== Representation condition numbers ===")
    print(cond_df.to_string(index=False))
    print("\n=== Whitening diagnostic ===")
    print(result_df.sort_values(["split", "MAE"]).to_string(index=False))


if __name__ == "__main__":
    main()
