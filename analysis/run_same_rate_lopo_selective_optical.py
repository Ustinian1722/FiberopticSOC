from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from run_representation_conditioning_diagnostic import PairTCN, whitening_matrix
from run_sequence_representation_benchmark import (
    PROFILES,
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


class ParameterMatchedVITCN(nn.Module):
    """VI control with exactly the same 4-channel encoder size as optical models.

    The final two channels are fixed zeros rather than duplicated V/I. This keeps the
    parameter count matched without adding information to the electrical-only control.
    """

    def __init__(self, hidden: int = 24):
        super().__init__()
        self.encoder = TCNEncoder(4, hidden)
        self.head = nn.Sequential(nn.Linear(hidden, hidden), nn.GELU(), nn.Linear(hidden, 1))

    def forward(self, x: torch.Tensor):
        electrical = x[:, :, (0, 1)]
        z = torch.cat([electrical, torch.zeros_like(electrical)], dim=-1)
        h = self.encoder(z)
        return self.head(h).squeeze(-1), None


def current_ood_metadata(
    raw_sources: list[dict],
    dataset: WindowDataset,
    q_low: float,
    q_high: float,
) -> pd.DataFrame:
    rows = []
    span = max(q_high - q_low, 1e-12)
    for source_id, end in dataset.index:
        src = raw_sources[source_id]
        start = end - dataset.window + 1
        current = src["x"][start : end + 1, 1].astype(np.float64)
        exceed = np.maximum(q_low - current, 0.0) + np.maximum(current - q_high, 0.0)
        rows.append(
            {
                "profile": src["profile"],
                "rate": src["rate"],
                "end_index": end,
                "soc": float(src["y"][end]),
                "ood_fraction": float(np.mean(exceed > 0.0)),
                "ood_max_relative_to_train_span": float(exceed.max() / span),
                "ood_mean_relative_to_train_span": float(exceed.mean() / span),
            }
        )
    return pd.DataFrame(rows)


def build_config_factories(train_sources: list[dict]):
    train_all = np.concatenate([s["x"] for s in train_sources], axis=0)
    tf_white = whitening_matrix(train_all[:, (4, 5)])
    return (
        ("VI", lambda: ParameterMatchedVITCN()),
        ("VI+W", lambda: PairTCN((2, 3))),
        ("VI+TF", lambda: PairTCN((4, 5))),
        ("VI+TF-white", lambda: PairTCN((4, 5), tf_white)),
    )


def add_metric_row(
    rows: list[dict],
    *,
    rate: str,
    held_out_profile: str,
    model: str,
    y: np.ndarray,
    pred: np.ndarray,
    params: int | None,
    args,
) -> None:
    rows.append(
        {
            "protocol": "same_rate_leave_one_profile_out",
            "rate": rate,
            "held_out_profile": held_out_profile,
            "model": model,
            "params": params,
            "window": args.window,
            "train_stride": args.train_stride,
            "test_stride": args.test_stride,
            "epochs": args.epochs,
            "seed": args.seed,
            "n_test": len(y),
            **metric_dict(y, pred),
        }
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("data/extracted"))
    parser.add_argument("--out-dir", type=Path, default=Path("results/same_rate_lopo"))
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

    split_rows: list[dict] = []
    region_rows: list[dict] = []
    gate_rows: list[dict] = []
    prediction_rows: list[pd.DataFrame] = []

    for rate in ("1C", "2C"):
        rate_sources = [s for s in sources if s["rate"] == rate]
        available_profiles = {s["profile"] for s in rate_sources}
        if available_profiles != set(PROFILES):
            raise RuntimeError(f"{rate}: expected profiles {PROFILES}, got {sorted(available_profiles)}")

        for held_out in PROFILES:
            split_name = f"{rate}_LOPO_{held_out}"
            train_raw = [s for s in rate_sources if s["profile"] != held_out]
            test_raw = [s for s in rate_sources if s["profile"] == held_out]
            if len(train_raw) != 5 or len(test_raw) != 1:
                raise RuntimeError(f"Unexpected split sizes for {split_name}")

            train_current = np.concatenate([s["x"][:, 1] for s in train_raw]).astype(np.float64)
            q = args.envelope_quantile
            q_low, q_high = np.quantile(train_current, [q, 1.0 - q])

            mean, std = train_normalizer(train_raw)
            train_sources = normalize_sources(train_raw, mean, std)
            test_sources = normalize_sources(test_raw, mean, std)
            train_ds = WindowDataset(train_sources, args.window, args.train_stride)
            test_ds = WindowDataset(test_sources, args.window, args.test_stride)
            train_loader = DataLoader(
                train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0
            )
            test_loader = DataLoader(
                test_ds, batch_size=args.batch_size * 2, shuffle=False, num_workers=0
            )

            print(
                f"\n=== {split_name}: train_windows={len(train_ds)} "
                f"test_windows={len(test_ds)} current_envelope=[{q_low:.6g},{q_high:.6g}] ==="
            )

            predictions: dict[str, np.ndarray] = {}
            y_ref: np.ndarray | None = None
            params_by_model: dict[str, int] = {}

            for model_name, factory in build_config_factories(train_sources):
                seed_everything(args.seed)
                model = factory()
                params = count_params(model)
                params_by_model[model_name] = params
                print(f"\n--- {model_name}, params={params} ---")
                train_model(model, train_loader, device, args.epochs, args.lr)
                y, pred, _, _ = predict_model(model, test_loader, device)
                if y_ref is None:
                    y_ref = y
                elif not np.allclose(y_ref, y):
                    raise RuntimeError("Target ordering changed between models")
                predictions[model_name] = pred
                add_metric_row(
                    split_rows,
                    rate=rate,
                    held_out_profile=held_out,
                    model=model_name,
                    y=y,
                    pred=pred,
                    params=params,
                    args=args,
                )
                del model
                if device.type == "cuda":
                    torch.cuda.empty_cache()

            assert y_ref is not None
            meta = current_ood_metadata(test_raw, test_ds, q_low, q_high)
            if len(meta) != len(y_ref):
                raise RuntimeError("OOD metadata and prediction lengths differ")
            any_ood = meta["ood_fraction"].to_numpy() > 0.0

            selective = np.where(any_ood, predictions["VI+W"], predictions["VI"])
            add_metric_row(
                split_rows,
                rate=rate,
                held_out_profile=held_out,
                model="OOD-selective-VI-or-W",
                y=y_ref,
                pred=selective,
                params=params_by_model["VI+W"],
                args=args,
            )

            meta.insert(0, "split", split_name)
            meta["y"] = y_ref
            for name, pred in predictions.items():
                meta[f"pred_{name}"] = pred
                meta[f"ae_{name}"] = np.abs(y_ref - pred)
            meta["pred_OOD-selective-VI-or-W"] = selective
            meta["ae_OOD-selective-VI-or-W"] = np.abs(y_ref - selective)
            meta["gate_uses_W"] = any_ood.astype(np.int8)
            prediction_rows.append(meta)

            for region_name, mask in (("ID", ~any_ood), ("OOD", any_ood)):
                if not mask.any():
                    continue
                for model_name in (
                    "VI",
                    "VI+W",
                    "VI+TF",
                    "VI+TF-white",
                    "OOD-selective-VI-or-W",
                ):
                    pred = (
                        selective
                        if model_name == "OOD-selective-VI-or-W"
                        else predictions[model_name]
                    )
                    region_rows.append(
                        {
                            "rate": rate,
                            "held_out_profile": held_out,
                            "region": region_name,
                            "model": model_name,
                            "n": int(mask.sum()),
                            "mean_ood_fraction": float(meta.loc[mask, "ood_fraction"].mean()),
                            **metric_dict(y_ref[mask], pred[mask]),
                        }
                    )

            vi_mae = metric_dict(y_ref, predictions["VI"])["MAE"]
            viw_mae = metric_dict(y_ref, predictions["VI+W"])["MAE"]
            sel_mae = metric_dict(y_ref, selective)["MAE"]
            gate_rows.append(
                {
                    "rate": rate,
                    "held_out_profile": held_out,
                    "current_q_low": q_low,
                    "current_q_high": q_high,
                    "fraction_windows_any_ood": float(any_ood.mean()),
                    "VI_MAE": vi_mae,
                    "VIW_MAE": viw_mae,
                    "selective_MAE": sel_mae,
                    "selective_vs_VI_gain": vi_mae - sel_mae,
                    "selective_vs_VIW_gain": viw_mae - sel_mae,
                    "selective_beats_or_ties_VI": bool(sel_mae <= vi_mae + 1e-12),
                    "selective_beats_or_ties_VIW": bool(sel_mae <= viw_mae + 1e-12),
                }
            )

    split_df = pd.DataFrame(split_rows)
    region_df = pd.DataFrame(region_rows)
    gate_df = pd.DataFrame(gate_rows)
    pred_df = pd.concat(prediction_rows, ignore_index=True)

    aggregate_rows = []
    for rate_label, g0 in list(split_df.groupby("rate")) + [("ALL", split_df)]:
        for model_name, g in g0.groupby("model"):
            aggregate_rows.append(
                {
                    "rate": rate_label,
                    "model": model_name,
                    "n_splits": len(g),
                    "MAE_mean": float(g["MAE"].mean()),
                    "MAE_std": float(g["MAE"].std(ddof=1)) if len(g) > 1 else np.nan,
                    "RMSE_mean": float(g["RMSE"].mean()),
                    "R2_mean": float(g["R2"].mean()),
                    "Q95_AE_mean": float(g["Q95_AE"].mean()),
                    "MaxAE_mean": float(g["MaxAE"].mean()),
                }
            )
    aggregate_df = pd.DataFrame(aggregate_rows)

    split_df.to_csv(args.out_dir / "lopo_split_metrics.csv", index=False)
    region_df.to_csv(args.out_dir / "lopo_region_metrics.csv", index=False)
    gate_df.to_csv(args.out_dir / "lopo_gate_summary.csv", index=False)
    aggregate_df.to_csv(args.out_dir / "lopo_aggregate_summary.csv", index=False)
    pred_df.to_csv(args.out_dir / "lopo_window_predictions.csv", index=False)

    print("\n=== Same-rate LOPO aggregate ===")
    print(aggregate_df.sort_values(["rate", "MAE_mean"]).to_string(index=False))
    print("\n=== Selective gate split summary ===")
    print(gate_df.to_string(index=False))
    print("\n=== Safety counts ===")
    print(
        gate_df.groupby("rate")[[
            "selective_beats_or_ties_VI",
            "selective_beats_or_ties_VIW",
        ]].sum().to_string()
    )


if __name__ == "__main__":
    main()
