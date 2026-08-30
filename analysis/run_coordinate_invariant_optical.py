from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from run_representation_conditioning_diagnostic import PairTCN, whitening_matrix
from run_same_rate_lopo_selective_optical import ParameterMatchedVITCN, current_ood_metadata
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


class InvariantOpticalTCN(nn.Module):
    """Four-channel VI + two O(2)-invariant optical features.

    The optical pair is first train-only whitened.  Any invertible two-channel
    representation of the same FBG degrees of freedom then differs only by an
    orthogonal transform.  Norms, Euclidean step lengths, and pairwise cosines
    are invariant to that orthogonal coordinate choice.
    """

    def __init__(
        self,
        pair: tuple[int, int],
        whiten: np.ndarray,
        mode: str = "step",
        hidden: int = 24,
        eps: float = 1e-6,
    ):
        super().__init__()
        if mode not in {"step", "cos"}:
            raise ValueError(f"Unknown invariant mode: {mode}")
        self.pair = pair
        self.mode = mode
        self.eps = eps
        self.register_buffer("whiten", torch.tensor(whiten, dtype=torch.float32), persistent=True)
        self.encoder = TCNEncoder(4, hidden)
        self.head = nn.Sequential(nn.Linear(hidden, hidden), nn.GELU(), nn.Linear(hidden, 1))

    def invariant_features(self, x: torch.Tensor) -> torch.Tensor:
        z = torch.matmul(x[:, :, self.pair], self.whiten.T)
        radius = torch.linalg.vector_norm(z, dim=-1)
        prev = torch.cat([z[:, :1], z[:, :-1]], dim=1)
        if self.mode == "step":
            second = torch.linalg.vector_norm(z - prev, dim=-1)
        else:
            denom = torch.linalg.vector_norm(z, dim=-1) * torch.linalg.vector_norm(prev, dim=-1)
            second = (z * prev).sum(dim=-1) / torch.clamp(denom, min=self.eps)
        return torch.stack([radius, second], dim=-1)

    def forward(self, x: torch.Tensor):
        electrical = x[:, :, (0, 1)]
        optical = self.invariant_features(x)
        z = torch.cat([electrical, optical], dim=-1)
        h = self.encoder(z)
        return self.head(h).squeeze(-1), None


def numpy_invariants(x: np.ndarray, pair: tuple[int, int], white: np.ndarray, mode: str) -> np.ndarray:
    z = x[:, pair].astype(np.float64) @ white.astype(np.float64).T
    radius = np.linalg.norm(z, axis=1)
    prev = np.vstack([z[:1], z[:-1]])
    if mode == "step":
        second = np.linalg.norm(z - prev, axis=1)
    elif mode == "cos":
        denom = np.linalg.norm(z, axis=1) * np.linalg.norm(prev, axis=1)
        second = np.sum(z * prev, axis=1) / np.maximum(denom, 1e-12)
    else:
        raise ValueError(mode)
    return np.column_stack([radius, second])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("data/extracted"))
    parser.add_argument("--out-dir", type=Path, default=Path("results/coordinate_invariant_optical"))
    parser.add_argument("--window", type=int, default=64)
    parser.add_argument("--train-stride", type=int, default=4)
    parser.add_argument("--test-stride", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--envelope-quantile", type=float, default=0.005)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sources = load_sources(args.data)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    equivalence_rows: list[dict] = []
    gate_rows: list[dict] = []

    for train_rate, test_rate in (("1C", "2C"), ("2C", "1C")):
        direction = f"{train_rate}_to_{test_rate}"
        for held_out in PROFILES:
            train_raw = [s for s in sources if s["rate"] == train_rate and s["profile"] != held_out]
            test_raw = [s for s in sources if s["rate"] == test_rate and s["profile"] == held_out]
            if len(train_raw) != 5 or len(test_raw) != 1:
                raise RuntimeError(f"Bad split {direction}/{held_out}")

            current = np.concatenate([s["x"][:, 1] for s in train_raw]).astype(np.float64)
            q_low, q_high = np.quantile(current, [args.envelope_quantile, 1.0 - args.envelope_quantile])

            mean, std = train_normalizer(train_raw)
            train_sources = normalize_sources(train_raw, mean, std)
            test_sources = normalize_sources(test_raw, mean, std)
            train_all = np.concatenate([s["x"] for s in train_sources], axis=0)
            test_all = np.concatenate([s["x"] for s in test_sources], axis=0)
            w_white = whitening_matrix(train_all[:, (2, 3)])
            tf_white = whitening_matrix(train_all[:, (4, 5)])

            # Pipeline-level numerical equivalence check.  W-white and TF-white may
            # differ by an orthogonal map, but the proposed scalar invariants should agree.
            for mode in ("step", "cos"):
                w_train_inv = numpy_invariants(train_all, (2, 3), w_white, mode)
                tf_train_inv = numpy_invariants(train_all, (4, 5), tf_white, mode)
                w_test_inv = numpy_invariants(test_all, (2, 3), w_white, mode)
                tf_test_inv = numpy_invariants(test_all, (4, 5), tf_white, mode)
                equivalence_rows.append(
                    {
                        "direction": direction,
                        "held_out_profile": held_out,
                        "mode": mode,
                        "train_max_abs_diff": float(np.max(np.abs(w_train_inv - tf_train_inv))),
                        "train_mean_abs_diff": float(np.mean(np.abs(w_train_inv - tf_train_inv))),
                        "test_max_abs_diff": float(np.max(np.abs(w_test_inv - tf_test_inv))),
                        "test_mean_abs_diff": float(np.mean(np.abs(w_test_inv - tf_test_inv))),
                    }
                )

            train_ds = WindowDataset(train_sources, args.window, args.train_stride)
            test_ds = WindowDataset(test_sources, args.window, args.test_stride)
            train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
            test_loader = DataLoader(test_ds, batch_size=args.batch_size * 2, shuffle=False, num_workers=0)

            specs = (
                ("VI", lambda: ParameterMatchedVITCN()),
                ("VI+W", lambda: PairTCN((2, 3))),
                ("VI+W-white", lambda: PairTCN((2, 3), w_white)),
                ("VI+OI-step-W", lambda: InvariantOpticalTCN((2, 3), w_white, "step")),
                ("VI+OI-step-TF", lambda: InvariantOpticalTCN((4, 5), tf_white, "step")),
                ("VI+OI-cos-W", lambda: InvariantOpticalTCN((2, 3), w_white, "cos")),
                ("VI+OI-cos-TF", lambda: InvariantOpticalTCN((4, 5), tf_white, "cos")),
            )

            predictions: dict[str, np.ndarray] = {}
            params_by_model: dict[str, int] = {}
            y_ref = None
            print(f"\n=== {direction} held-out={held_out} ===")
            for name, factory in specs:
                seed_everything(args.seed)
                model = factory()
                params_by_model[name] = count_params(model)
                print(f"--- {name} params={params_by_model[name]} ---")
                train_model(model, train_loader, device, args.epochs, args.lr)
                y, pred, _, _ = predict_model(model, test_loader, device)
                if y_ref is None:
                    y_ref = y
                elif not np.allclose(y_ref, y):
                    raise RuntimeError("Target ordering changed")
                predictions[name] = pred
                rows.append(
                    {
                        "direction": direction,
                        "held_out_profile": held_out,
                        "model": name,
                        "seed": args.seed,
                        "params": params_by_model[name],
                        "n_test": len(y),
                        **metric_dict(y, pred),
                    }
                )
                del model

            assert y_ref is not None
            meta = current_ood_metadata(test_raw, test_ds, q_low, q_high)
            use_optical = meta["ood_fraction"].to_numpy() > 0.0
            for expert in ("VI+W", "VI+W-white", "VI+OI-step-W", "VI+OI-cos-W"):
                selective = np.where(use_optical, predictions[expert], predictions["VI"])
                name = f"OOD-selective-{expert}"
                rows.append(
                    {
                        "direction": direction,
                        "held_out_profile": held_out,
                        "model": name,
                        "seed": args.seed,
                        "params": params_by_model[expert],
                        "n_test": len(y_ref),
                        **metric_dict(y_ref, selective),
                    }
                )
                gate_rows.append(
                    {
                        "direction": direction,
                        "held_out_profile": held_out,
                        "expert": expert,
                        "fraction_windows_any_ood": float(use_optical.mean()),
                        "VI_MAE": metric_dict(y_ref, predictions["VI"])["MAE"],
                        "expert_MAE": metric_dict(y_ref, predictions[expert])["MAE"],
                        "selective_MAE": metric_dict(y_ref, selective)["MAE"],
                    }
                )

    result_df = pd.DataFrame(rows)
    equiv_df = pd.DataFrame(equivalence_rows)
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
    aggregate.to_csv(args.out_dir / "aggregate_summary.csv", index=False)
    equiv_df.to_csv(args.out_dir / "invariance_equivalence.csv", index=False)
    gate_df.to_csv(args.out_dir / "gate_summary.csv", index=False)

    print("\n=== Coordinate-invariant optical aggregate ===")
    print(aggregate.sort_values(["direction", "MAE_mean"]).to_string(index=False))
    print("\n=== W-vs-TF invariant equivalence ===")
    print(
        equiv_df.groupby("mode")[[
            "train_max_abs_diff",
            "train_mean_abs_diff",
            "test_max_abs_diff",
            "test_mean_abs_diff",
        ]].max().to_string()
    )


if __name__ == "__main__":
    main()
