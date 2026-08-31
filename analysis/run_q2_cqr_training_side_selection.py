from __future__ import annotations

import argparse
import copy
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from q2_protocols import blocked_mixed_condition_split
from run_q2_matched_dt_source_validation import DT_INDEX, load_sources_with_dt
from run_sequence_representation_benchmark import (
    TCNEncoder,
    WindowDataset,
    load_sources,
    normalize_sources,
    seed_everything,
    train_normalizer,
)

QUANTILES = (0.025, 0.05, 0.95, 0.975)


class FrozenPointTCN(nn.Module):
    """Frozen IUW-TCN body, optionally with preregistered causal delta-t."""

    def __init__(self, use_dt: bool, hidden: int = 24):
        super().__init__()
        self.use_dt = bool(use_dt)
        self.encoder = TCNEncoder(5 if self.use_dt else 4, hidden)
        self.head = nn.Sequential(nn.Linear(hidden, hidden), nn.GELU(), nn.Linear(hidden, 1))

    def forward(self, x: torch.Tensor):
        idx = (0, 1, 2, 3, DT_INDEX) if self.use_dt else (0, 1, 2, 3)
        h = self.encoder(x[:, :, idx])
        return self.head(h).squeeze(-1)


class FrozenQuantileTCN(nn.Module):
    """Same frozen encoder geometry with a four-quantile prediction head."""

    def __init__(self, use_dt: bool, hidden: int = 24):
        super().__init__()
        self.use_dt = bool(use_dt)
        self.encoder = TCNEncoder(5 if self.use_dt else 4, hidden)
        self.head = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, len(QUANTILES)),
        )

    def forward(self, x: torch.Tensor):
        idx = (0, 1, 2, 3, DT_INDEX) if self.use_dt else (0, 1, 2, 3)
        h = self.encoder(x[:, :, idx])
        return self.head(h)


def pinball_plus_crossing(pred: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    q = torch.tensor(QUANTILES, dtype=pred.dtype, device=pred.device)[None, :]
    err = y[:, None] - pred
    pinball = torch.maximum(q * err, (q - 1.0) * err).mean()
    crossing = F.relu(pred[:, :-1] - pred[:, 1:]).mean()
    return pinball + crossing


@torch.no_grad()
def predict_point(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    ys, ps = [], []
    for x, y, _ in loader:
        ps.append(model(x.to(device)).cpu().numpy())
        ys.append(y.numpy())
    return np.concatenate(ys), np.concatenate(ps)


@torch.no_grad()
def predict_quantile(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    ys, qs = [], []
    for x, y, _ in loader:
        qs.append(model(x.to(device)).cpu().numpy())
        ys.append(y.numpy())
    return np.concatenate(ys), np.concatenate(qs)


def train_point_earlystop(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    *,
    max_epochs: int,
    min_epochs: int,
    patience: int,
    min_delta: float,
    lr: float,
) -> tuple[list[dict], int]:
    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    loss_fn = nn.MSELoss()
    best = float("inf")
    anchor = float("inf")
    stale = 0
    best_epoch = 1
    best_state = copy.deepcopy(model.state_dict())
    rows: list[dict] = []

    for epoch in range(1, max_epochs + 1):
        model.train()
        total = 0.0
        n = 0
        for x, y, _ in train_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad(set_to_none=True)
            pred = model(x)
            loss = loss_fn(pred, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            opt.step()
            total += float(loss.detach()) * len(y)
            n += len(y)

        yv, pv = predict_point(model, val_loader, device)
        mae = float(np.mean(np.abs(yv - pv)))
        rows.append({"epoch": epoch, "train_mse": total / max(n, 1), "val_MAE": mae})
        if mae < best:
            best = mae
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
        if mae < anchor - min_delta:
            anchor = mae
            stale = 0
        else:
            stale += 1
        if epoch >= min_epochs and stale >= patience:
            break

    model.load_state_dict(best_state)
    return rows, best_epoch


def train_quantile_earlystop(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    *,
    max_epochs: int,
    min_epochs: int,
    patience: int,
    min_delta: float,
    lr: float,
) -> tuple[list[dict], int]:
    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    best = float("inf")
    anchor = float("inf")
    stale = 0
    best_epoch = 1
    best_state = copy.deepcopy(model.state_dict())
    rows: list[dict] = []

    for epoch in range(1, max_epochs + 1):
        model.train()
        total = 0.0
        n = 0
        for x, y, _ in train_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad(set_to_none=True)
            q = model(x)
            loss = pinball_plus_crossing(q, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            opt.step()
            total += float(loss.detach()) * len(y)
            n += len(y)

        yv, qv = predict_quantile(model, val_loader, device)
        val_loss = float(
            pinball_plus_crossing(torch.from_numpy(qv), torch.from_numpy(yv))
        )
        rows.append({"epoch": epoch, "train_loss": total / max(n, 1), "val_pinball_plus_crossing": val_loss})
        if val_loss < best:
            best = val_loss
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
        if val_loss < anchor - min_delta:
            anchor = val_loss
            stale = 0
        else:
            stale += 1
        if epoch >= min_epochs and stale >= patience:
            break

    model.load_state_dict(best_state)
    return rows, best_epoch


def finite_sample_q(scores: np.ndarray, alpha: float) -> float:
    scores = np.sort(np.asarray(scores, dtype=float))
    if len(scores) == 0:
        raise ValueError("Empty conformal calibration score set")
    k = int(math.ceil((len(scores) + 1) * (1.0 - alpha)))
    k = min(max(k, 1), len(scores))
    return float(scores[k - 1])


def interval_metrics(y: np.ndarray, lo: np.ndarray, hi: np.ndarray, alpha: float) -> dict:
    lo = np.asarray(lo, dtype=float)
    hi = np.asarray(hi, dtype=float)
    y = np.asarray(y, dtype=float)
    if np.any(hi < lo):
        raise ValueError("Invalid interval with upper < lower")
    covered = (y >= lo) & (y <= hi)
    width = hi - lo
    miss = np.maximum(lo - y, 0.0) + np.maximum(y - hi, 0.0)
    score = width + 2.0 / alpha * miss
    return {
        "nominal_coverage": 1.0 - alpha,
        "PICP": float(np.mean(covered)),
        "coverage_error": float(np.mean(covered) - (1.0 - alpha)),
        "MPIW": float(np.mean(width)),
        "mean_interval_score": float(np.mean(score)),
    }


def _parent_name(src: dict) -> str:
    return str(src.get("parent_name", src["name"]))


def partition_t1_train_segments(train_segments: list[dict]) -> tuple[dict[str, list[dict]], pd.DataFrame]:
    """Split only formal T1-train segments into fit/val/cal/selection by whole segment.

    The frozen T1 block pattern yields only a small number of disjoint train segments per
    physical trajectory (three for the current SiC-18 release). Therefore roles are
    assigned across trajectories rather than requiring every parent trajectory to contain
    all four roles. Assignment uses only deterministic parent ordering and within-parent
    segment order; SOC labels, predictions, residuals and uncertainty statistics are never
    consulted. A 5-slot phase-rotated pattern gives fit twice the weight of each held-out
    role while spreading every role across both rates and all six drive profiles.
    """
    grouped: dict[str, list[dict]] = defaultdict(list)
    for seg in train_segments:
        grouped[_parent_name(seg)].append(seg)

    buckets = {"fit": [], "validation": [], "calibration": [], "selection": []}
    audit_rows: list[dict] = []
    role_pattern = ("fit", "validation", "fit", "calibration", "selection")

    for parent_idx, parent in enumerate(sorted(grouped)):
        segs = sorted(grouped[parent], key=lambda s: int(s.get("segment_start", 0)))
        if not segs:
            raise RuntimeError(f"No formal T1-train segments for {parent}")
        phase = (2 * parent_idx) % len(role_pattern)

        for seg_idx, seg in enumerate(segs):
            role = role_pattern[(seg_idx + phase) % len(role_pattern)]
            buckets[role].append(seg)
            audit_rows.append(
                {
                    "parent_name": parent,
                    "profile": str(seg["profile"]),
                    "rate": str(seg["rate"]),
                    "segment_index_within_parent": seg_idx,
                    "segment_start": int(seg.get("segment_start", -1)),
                    "segment_stop": int(seg.get("segment_stop", -1)),
                    "role": role,
                    "assignment_used_labels": False,
                    "assignment_pattern": "fit,val,fit,cal,selection_phase2parent",
                }
            )

    expected_rates = {"1C", "2C"}
    expected_profiles = {"HWFET", "LA92", "NEDC", "NYCC", "US06", "WLTC"}
    for role, items in buckets.items():
        if not items:
            raise RuntimeError(f"No segments assigned to {role}")
        rates = {str(s["rate"]) for s in items}
        profiles = {str(s["profile"]) for s in items}
        if rates != expected_rates:
            raise RuntimeError(f"Training-side role {role} does not cover both rates: {sorted(rates)}")
        if profiles != expected_profiles:
            raise RuntimeError(
                f"Training-side role {role} does not cover all six profiles: {sorted(profiles)}"
            )

    return buckets, pd.DataFrame(audit_rows)


def main() -> None:
    p = argparse.ArgumentParser(description="Training-side CQR KEEP/DROP selection for frozen Q2 IUW-TCN")
    p.add_argument("--data", type=Path, default=Path("data/extracted/SiC-18"))
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--feature-mode", choices=("raw_w", "raw_w_dt"), required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--window", type=int, default=64)
    p.add_argument("--train-stride", type=int, default=2)
    p.add_argument("--eval-stride", type=int, default=1)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--max-epochs", type=int, default=50)
    p.add_argument("--min-epochs", type=int, default=10)
    p.add_argument("--patience", type=int, default=7)
    p.add_argument("--min-delta", type=float, default=5e-5)
    p.add_argument("--lr", type=float, default=1e-3)
    args = p.parse_args()

    use_dt = args.feature_mode == "raw_w_dt"
    raw = load_sources_with_dt(args.data) if use_dt else load_sources(args.data)

    # Frozen formal T1 split. Only .train is allowed to enter this selection program.
    formal = blocked_mixed_condition_split(raw, window=args.window)
    buckets_raw, assignment = partition_t1_train_segments(formal.train)

    mean, std = train_normalizer(buckets_raw["fit"])
    buckets = {
        role: normalize_sources(items, mean, std)
        for role, items in buckets_raw.items()
    }

    train_ds = WindowDataset(buckets["fit"], args.window, args.train_stride)
    val_ds = WindowDataset(buckets["validation"], args.window, args.eval_stride)
    cal_ds = WindowDataset(buckets["calibration"], args.window, args.eval_stride)
    sel_ds = WindowDataset(buckets["selection"], args.window, args.eval_stride)
    if min(len(train_ds), len(val_ds), len(cal_ds), len(sel_ds)) <= 0:
        raise RuntimeError("Training-side UQ split produced an empty window set")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size * 2, shuffle=False, num_workers=0)
    cal_loader = DataLoader(cal_ds, batch_size=args.batch_size * 2, shuffle=False, num_workers=0)
    sel_loader = DataLoader(sel_ds, batch_size=args.batch_size * 2, shuffle=False, num_workers=0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Simple residual split-conformal baseline.
    seed_everything(args.seed)
    point = FrozenPointTCN(use_dt=use_dt)
    point_curve, point_best_epoch = train_point_earlystop(
        point,
        train_loader,
        val_loader,
        device,
        max_epochs=args.max_epochs,
        min_epochs=args.min_epochs,
        patience=args.patience,
        min_delta=args.min_delta,
        lr=args.lr,
    )
    ycal_p, pcal = predict_point(point, cal_loader, device)
    ysel_p, psel = predict_point(point, sel_loader, device)

    # CQR candidate.
    seed_everything(args.seed)
    cqr = FrozenQuantileTCN(use_dt=use_dt)
    cqr_curve, cqr_best_epoch = train_quantile_earlystop(
        cqr,
        train_loader,
        val_loader,
        device,
        max_epochs=args.max_epochs,
        min_epochs=args.min_epochs,
        patience=args.patience,
        min_delta=args.min_delta,
        lr=args.lr,
    )
    ycal_q, qcal = predict_quantile(cqr, cal_loader, device)
    ysel_q, qsel = predict_quantile(cqr, sel_loader, device)
    if not np.allclose(ycal_p, ycal_q) or not np.allclose(ysel_p, ysel_q):
        raise RuntimeError("Point and CQR loaders produced inconsistent target ordering")

    qcal = np.sort(qcal, axis=1)
    qsel = np.sort(qsel, axis=1)
    rows: list[dict] = []
    interval_specs = ((0.10, 1, 2), (0.05, 0, 3))
    for alpha, lo_idx, hi_idx in interval_specs:
        # Residual split conformal.
        resid_q = finite_sample_q(np.abs(ycal_p - pcal), alpha)
        base_lo = np.clip(psel - resid_q, 0.0, 1.0)
        base_hi = np.clip(psel + resid_q, 0.0, 1.0)
        base_m = interval_metrics(ysel_p, base_lo, base_hi, alpha)
        rows.append(
            {
                "method": "residual_split_conformal",
                "alpha": alpha,
                "qhat": resid_q,
                **base_m,
            }
        )

        # Conformalized quantile regression.
        cal_lo = qcal[:, lo_idx]
        cal_hi = qcal[:, hi_idx]
        cqr_scores = np.maximum(cal_lo - ycal_q, ycal_q - cal_hi)
        cqr_qhat = finite_sample_q(cqr_scores, alpha)
        cqr_lo = np.clip(qsel[:, lo_idx] - cqr_qhat, 0.0, 1.0)
        cqr_hi = np.clip(qsel[:, hi_idx] + cqr_qhat, 0.0, 1.0)
        cqr_m = interval_metrics(ysel_q, cqr_lo, cqr_hi, alpha)
        rows.append(
            {
                "method": "CQR",
                "alpha": alpha,
                "qhat": cqr_qhat,
                **cqr_m,
            }
        )

    metrics = pd.DataFrame(rows).sort_values(["alpha", "method"])
    cqr_m = metrics[metrics.method == "CQR"].set_index("alpha")
    base_m = metrics[metrics.method == "residual_split_conformal"].set_index("alpha")

    c1 = bool(cqr_m.loc[0.10, "PICP"] >= 0.87)
    c2 = bool(cqr_m.loc[0.05, "PICP"] >= 0.92)
    c3 = bool((cqr_m["mean_interval_score"] <= base_m["mean_interval_score"]).all())
    rel_improve = (
        (base_m["mean_interval_score"] - cqr_m["mean_interval_score"])
        / base_m["mean_interval_score"].replace(0.0, np.nan)
    )
    avg_rel_improve = float(rel_improve.mean())
    c4 = bool(avg_rel_improve >= 0.02)
    width_ratio = cqr_m["MPIW"] / base_m["MPIW"].replace(0.0, np.nan)
    c5 = bool((width_ratio <= 1.10).all())
    keep = bool(c1 and c2 and c3 and c4 and c5)

    decision = pd.DataFrame(
        [
            {
                "frozen_model": "IUW-TCN",
                "feature_mode": args.feature_mode,
                "candidate": "CQR",
                "baseline": "residual_split_conformal",
                "criterion_picp90_at_least_087": c1,
                "criterion_picp95_at_least_092": c2,
                "criterion_mis_no_worse_both_levels": c3,
                "average_relative_MIS_improvement": avg_rel_improve,
                "criterion_average_MIS_improvement_at_least_2pct": c4,
                "max_CQR_to_baseline_width_ratio": float(width_ratio.max()),
                "criterion_width_not_over_10pct_wider": c5,
                "decision": "KEEP" if keep else "DROP",
                "formal_T1_validation_used_for_decision": False,
                "formal_T1_calibration_used_for_decision": False,
                "formal_T1_test_used_for_decision": False,
                "T4_target_used_for_decision": False,
                "point_best_epoch_inner": point_best_epoch,
                "cqr_best_epoch_inner": cqr_best_epoch,
                "n_fit_windows": len(train_ds),
                "n_validation_windows": len(val_ds),
                "n_calibration_windows": len(cal_ds),
                "n_selection_windows": len(sel_ds),
            }
        ]
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    assignment.to_csv(args.out_dir / "training_side_segment_assignment.csv", index=False)
    pd.DataFrame(point_curve).to_csv(args.out_dir / "point_validation_curve.csv", index=False)
    pd.DataFrame(cqr_curve).to_csv(args.out_dir / "cqr_validation_curve.csv", index=False)
    metrics.to_csv(args.out_dir / "selection_interval_metrics.csv", index=False)
    decision.to_csv(args.out_dir / "cqr_decision.csv", index=False)

    print("=== Training-side CQR selection ===")
    print(metrics.to_string(index=False))
    print(decision.to_string(index=False))
    print("Formal T1 validation/calibration/test and all T4 targets were excluded from this decision.")


if __name__ == "__main__":
    main()
