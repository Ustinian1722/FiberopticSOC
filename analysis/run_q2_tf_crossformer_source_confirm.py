from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader

from run_q2_crossformer_screen import EOCrossFormer
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
    train_model,
    train_normalizer,
)

MODEL_ORDER = ("IUW-TCN", "EO-CrossFormer-TF")


def build_model(name: str) -> torch.nn.Module:
    if name == "IUW-TCN":
        return PairTCN((2, 3), None)
    if name == "EO-CrossFormer-TF":
        return EOCrossFormer((4, 5))
    raise ValueError(name)


def run_fold(args, sources: list[dict], device: torch.device) -> pd.DataFrame:
    held_out = args.held_out_profile
    train_raw = [s for s in sources if s["rate"] == "1C" and s["profile"] != held_out]
    val_raw = [s for s in sources if s["rate"] == "1C" and s["profile"] == held_out]
    if len(train_raw) != 5 or len(val_raw) != 1:
        raise RuntimeError(f"Bad source-side split for {held_out}: train={len(train_raw)} val={len(val_raw)}")
    # Hard guard: this confirmation must not even normalize with 2C data.
    if any(s["rate"] != "1C" for s in train_raw + val_raw):
        raise RuntimeError("2C data entered source-side confirmation")

    mean, std = train_normalizer(train_raw)
    train = normalize_sources(train_raw, mean, std)
    val = normalize_sources(val_raw, mean, std)
    train_ds = WindowDataset(train, args.window, args.train_stride)
    val_ds = WindowDataset(val, args.window, args.val_stride)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size * 2, shuffle=False, num_workers=0)

    rows = []
    for name in MODEL_ORDER:
        seed_everything(args.seed)
        model = build_model(name)
        params = count_params(model)
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
        print(f"\n=== 1C source confirm held_out={held_out} model={name} params={params} train={len(train_ds)} val={len(val_ds)} ===")
        train_model(model, train_loader, device, args.epochs, args.lr)
        y, pred, _, aux = predict_model(model, val_loader, device)
        if aux is not None:
            raise RuntimeError(f"Unexpected auxiliary output from {name}")
        rows.append(
            {
                "protocol": "source_only_1C_same_rate_LOPO",
                "rate": "1C",
                "held_out_profile": held_out,
                "model": name,
                "representation": "raw_W" if name == "IUW-TCN" else "physics_decoupled_TF",
                "seed": args.seed,
                "epochs": args.epochs,
                "window": args.window,
                "train_stride": args.train_stride,
                "val_stride": args.val_stride,
                "params": params,
                "n_train": len(train_ds),
                "n_val": len(y),
                "source_only": True,
                "two_c_loaded_for_decision": False,
                **metric_dict(y, pred),
            }
        )
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return pd.DataFrame(rows)


def main() -> None:
    p = argparse.ArgumentParser(description="Source-only confirmation of EO-CrossFormer-TF")
    p.add_argument("--data", type=Path, default=Path("data/extracted/SiC-18"))
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--held-out-profile", choices=PROFILES, required=True)
    p.add_argument("--window", type=int, default=64)
    p.add_argument("--train-stride", type=int, default=4)
    p.add_argument("--val-stride", type=int, default=1)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}")
    sources = load_sources(args.data)
    # Dataset structure can contain 2C, but run_fold explicitly filters/guards 1C only.
    result = run_fold(args, sources, device)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    out = args.out_dir / f"source_confirm_{args.held_out_profile}.csv"
    result.to_csv(out, index=False)
    print("\n=== source-only confirmation fold ===")
    print(result.sort_values("MAE").to_string(index=False))


if __name__ == "__main__":
    main()
