from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader

from run_nested_epoch_selection import train_fixed_epochs
from run_q2_crossformer_screen import CGAMatched, DualTCNTransformer
from run_q2_final_t1_paper import CNNBaseline, RNNBaseline, TransformerBaseline
from run_representation_conditioning_diagnostic import PairTCN
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

MODELS = (
    "CNN",
    "GRU",
    "LSTM",
    "Transformer",
    "CGA-Matched",
    "DualTCN-Transformer",
    "RA-FBG-TCN",
)
DIRECTIONS = ("1C_to_2C", "2C_to_1C")


def rates(direction: str) -> tuple[str, str]:
    if direction == "1C_to_2C":
        return "1C", "2C"
    if direction == "2C_to_1C":
        return "2C", "1C"
    raise ValueError(direction)


def build_model(name: str, window: int):
    if name == "CNN":
        return CNNBaseline()
    if name == "GRU":
        return RNNBaseline("GRU")
    if name == "LSTM":
        return RNNBaseline("LSTM")
    if name == "Transformer":
        return TransformerBaseline(window)
    if name == "CGA-Matched":
        return CGAMatched()
    if name == "DualTCN-Transformer":
        return DualTCNTransformer((2, 3))
    if name == "RA-FBG-TCN":
        return PairTCN((2, 3), None)
    raise ValueError(name)


def selected_epoch(plan: pd.DataFrame, direction: str, profile: str) -> int:
    g = plan[(plan.direction == direction) & (plan.held_out_profile == profile)]
    if len(g) != 1:
        raise RuntimeError(f"Need one frozen epoch for {direction}/{profile}, got {len(g)}")
    return int(g.iloc[0].selected_epoch)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--epoch-plan", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--direction", choices=DIRECTIONS, required=True)
    p.add_argument("--model", choices=MODELS, default=None,
                   help="Execution-only shard. Omit to run the frozen full model list.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--window", type=int, default=64)
    p.add_argument("--train-stride", type=int, default=4)
    p.add_argument("--test-stride", type=int, default=1)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    args = p.parse_args()

    plan = pd.read_csv(args.epoch_plan)
    sources = load_sources(args.data)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_rate, test_rate = rates(args.direction)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    models_to_run = (args.model,) if args.model else MODELS

    rows: list[dict] = []
    for profile in PROFILES:
        train_raw = [s for s in sources if s["rate"] == train_rate and s["profile"] != profile]
        test_raw = [s for s in sources if s["rate"] == test_rate and s["profile"] == profile]
        if len(train_raw) != 5 or len(test_raw) != 1:
            raise RuntimeError(f"Bad strict split {args.direction}/{profile}")

        mean, std = train_normalizer(train_raw)
        train = normalize_sources(train_raw, mean, std)
        test = normalize_sources(test_raw, mean, std)
        train_ds = WindowDataset(train, args.window, args.train_stride)
        test_ds = WindowDataset(test, args.window, args.test_stride)
        test_loader = DataLoader(test_ds, batch_size=args.batch_size * 2, shuffle=False, num_workers=0)
        epoch = selected_epoch(plan, args.direction, profile)

        print(f"\n=== {args.direction}/{profile} epoch={epoch} train={len(train_ds)} test={len(test_ds)} ===")
        for model_name in models_to_run:
            seed_everything(args.seed)
            model = build_model(model_name, args.window)
            train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
            n_params = count_params(model)
            print(f"--- {model_name} params={n_params} ---")
            train_fixed_epochs(model, train_loader, device, epoch, args.lr)
            y, pred, _, gate = predict_model(model, test_loader, device)
            if gate is not None:
                raise RuntimeError(f"Unexpected gate from {model_name}")
            rows.append(
                {
                    "protocol": "T4_strict_backbone_benchmark",
                    "direction": args.direction,
                    "train_rate": train_rate,
                    "test_rate": test_rate,
                    "held_out_profile": profile,
                    "model": model_name,
                    "input": "V/I/W1/W2",
                    "seed": args.seed,
                    "selected_epoch": epoch,
                    "params": n_params,
                    "n_train": len(train_ds),
                    "n_test": len(y),
                    **metric_dict(y, pred),
                }
            )
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()

    df = pd.DataFrame(rows)
    df.to_csv(args.out_dir / "split_metrics.csv", index=False)
    summary = (
        df.groupby(["direction", "model", "params"], as_index=False)
        .agg(
            n_profiles=("MAE", "size"),
            MAE_mean=("MAE", "mean"),
            MAE_std=("MAE", "std"),
            RMSE_mean=("RMSE", "mean"),
            R2_mean=("R2", "mean"),
            Q95_AE_mean=("Q95_AE", "mean"),
            MaxAE_mean=("MaxAE", "mean"),
        )
        .sort_values("MAE_mean")
    )
    summary.to_csv(args.out_dir / "direction_summary.csv", index=False)
    print("\n=== strict backbone summary ===")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
