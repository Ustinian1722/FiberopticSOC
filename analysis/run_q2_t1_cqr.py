from __future__ import annotations

import argparse
import copy
import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from q2_protocols import blocked_mixed_condition_split
from run_q2_etmf_prototype import ETMFNet
from run_sequence_representation_benchmark import (
    WindowDataset,
    load_sources,
    normalize_sources,
    seed_everything,
    train_normalizer,
)

QUANTILES = (0.025, 0.05, 0.95, 0.975)


class ETMFQuantile(nn.Module):
    """ETMF backbone with four quantile outputs for 90% and 95% CQR intervals."""

    def __init__(self, hidden: int = 24):
        super().__init__()
        base = ETMFNet(hidden=hidden)
        self.electrical = base.electrical
        self.thermomechanical = base.thermomechanical
        self.m_to_e = base.m_to_e
        self.e_to_m = base.e_to_m
        self.e_cross_gate = base.e_cross_gate
        self.m_cross_gate = base.m_cross_gate
        self.e_norm = base.e_norm
        self.m_norm = base.m_norm
        self.mix_gate = base.mix_gate
        self.head = nn.Sequential(
            nn.Linear(hidden * 3, hidden * 2),
            nn.GELU(),
            nn.Linear(hidden * 2, hidden),
            nn.GELU(),
            nn.Linear(hidden, len(QUANTILES)),
        )

    def forward(self, x: torch.Tensor):
        e = self.electrical(x[:, :, (0, 1)])
        m = self.thermomechanical(x[:, :, (4, 5)])
        e_refined = self.e_norm(e + self.e_cross_gate(m) * self.m_to_e(m))
        m_refined = self.m_norm(m + self.m_cross_gate(e) * self.e_to_m(e))
        interaction = torch.abs(e_refined - m_refined)
        alpha = self.mix_gate(torch.cat([e_refined, m_refined, interaction], dim=-1))
        fused = alpha * e_refined + (1.0 - alpha) * m_refined
        q = self.head(torch.cat([fused, e_refined * m_refined, interaction], dim=-1))
        return q, alpha.squeeze(-1)


def pinball_loss(pred: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    target = y[:, None]
    q = torch.tensor(QUANTILES, dtype=pred.dtype, device=pred.device)[None, :]
    error = target - pred
    loss = torch.maximum(q * error, (q - 1.0) * error).mean()
    # Soft crossing penalty keeps q025 <= q05 <= q95 <= q975 without changing labels.
    crossing = F.relu(pred[:, :-1] - pred[:, 1:]).mean()
    return loss + crossing


@torch.no_grad()
def predict(model: nn.Module, loader: DataLoader, device: torch.device):
    model.eval()
    ys, qs, alphas = [], [], []
    for x, y, _ in loader:
        q, alpha = model(x.to(device))
        ys.append(y.numpy())
        qs.append(q.cpu().numpy())
        alphas.append(alpha.cpu().numpy())
    return np.concatenate(ys), np.concatenate(qs), np.concatenate(alphas)


def train_earlystop(
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
):
    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    best = float("inf")
    anchor = float("inf")
    best_epoch = 1
    stale = 0
    best_state = copy.deepcopy(model.state_dict())
    rows = []
    for epoch in range(1, max_epochs + 1):
        model.train()
        total, n = 0.0, 0
        for x, y, _ in train_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad(set_to_none=True)
            q, _ = model(x)
            loss = pinball_loss(q, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            opt.step()
            total += float(loss.detach()) * len(y)
            n += len(y)
        yv, qv, _ = predict(model, val_loader, device)
        target = torch.from_numpy(yv)
        qp = torch.from_numpy(qv)
        val_loss = float(pinball_loss(qp, target))
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
    k = int(math.ceil((len(scores) + 1) * (1.0 - alpha)))
    k = min(max(k, 1), len(scores))
    return float(scores[k - 1])


def interval_metrics(y: np.ndarray, lo: np.ndarray, hi: np.ndarray, alpha: float) -> dict:
    covered = (y >= lo) & (y <= hi)
    width = hi - lo
    below = np.maximum(lo - y, 0.0)
    above = np.maximum(y - hi, 0.0)
    score = width + 2.0 / alpha * (below + above)
    return {
        "nominal_coverage": 1.0 - alpha,
        "PICP": float(np.mean(covered)),
        "coverage_error": float(np.mean(covered) - (1.0 - alpha)),
        "MPIW": float(np.mean(width)),
        "PINAW": float(np.mean(width)),
        "mean_interval_score": float(np.mean(score)),
    }


def make_meta(raw_sources: list[dict], ds: WindowDataset) -> pd.DataFrame:
    rows = []
    for source_id, end in ds.index:
        src = raw_sources[source_id]
        rows.append(
            {
                "source_name": src["name"],
                "profile": src["profile"],
                "rate": src["rate"],
                "current_A": float(src["x"][end, 1]),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=Path, default=Path("data/extracted/SiC-18"))
    p.add_argument("--out-dir", type=Path, required=True)
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

    raw = load_sources(args.data)
    split = blocked_mixed_condition_split(raw, window=args.window)
    mean, std = train_normalizer(split.train)
    train = normalize_sources(split.train, mean, std)
    val = normalize_sources(split.validation, mean, std)
    cal = normalize_sources(split.calibration, mean, std)
    test = normalize_sources(split.test, mean, std)

    train_ds = WindowDataset(train, args.window, args.train_stride)
    val_ds = WindowDataset(val, args.window, args.eval_stride)
    cal_ds = WindowDataset(cal, args.window, args.eval_stride)
    test_ds = WindowDataset(test, args.window, args.eval_stride)
    seed_everything(args.seed)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size * 2, shuffle=False)
    cal_loader = DataLoader(cal_ds, batch_size=args.batch_size * 2, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size * 2, shuffle=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = ETMFQuantile()
    curves, best_epoch = train_earlystop(
        model, train_loader, val_loader, device,
        max_epochs=args.max_epochs, min_epochs=args.min_epochs,
        patience=args.patience, min_delta=args.min_delta, lr=args.lr,
    )
    ycal, qcal, acal = predict(model, cal_loader, device)
    ytest, qtest, atest = predict(model, test_loader, device)

    # Adjacent quantiles are already crossing-penalized; enforce order only for interval reporting.
    qcal_sorted = np.sort(qcal, axis=1)
    qtest_sorted = np.sort(qtest, axis=1)
    interval_specs = [
        (0.10, 1, 2),  # q05, q95
        (0.05, 0, 3),  # q025, q975
    ]
    summaries = []
    interval_frames = []
    test_meta = make_meta(split.test, test_ds)
    cal_meta = make_meta(split.calibration, cal_ds)

    for alpha, lo_idx, hi_idx in interval_specs:
        cal_lo = qcal_sorted[:, lo_idx]
        cal_hi = qcal_sorted[:, hi_idx]
        scores = np.maximum(cal_lo - ycal, ycal - cal_hi)
        qhat = finite_sample_q(scores, alpha)
        lo = np.clip(qtest_sorted[:, lo_idx] - qhat, 0.0, 1.0)
        hi = np.clip(qtest_sorted[:, hi_idx] + qhat, 0.0, 1.0)
        m = interval_metrics(ytest, lo, hi, alpha)
        m.update({"alpha": alpha, "qhat": qhat, "best_epoch": best_epoch, "n_cal": len(ycal), "n_test": len(ytest)})
        summaries.append(m)
        frame = test_meta.copy()
        frame["alpha"] = alpha
        frame["y_true"] = ytest
        frame["raw_lower_quantile"] = qtest_sorted[:, lo_idx]
        frame["raw_upper_quantile"] = qtest_sorted[:, hi_idx]
        frame["lower"] = lo
        frame["upper"] = hi
        frame["covered"] = (ytest >= lo) & (ytest <= hi)
        frame["interval_width"] = hi - lo
        frame["alpha_fusion"] = atest
        interval_frames.append(frame)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(curves).to_csv(args.out_dir / "validation_curve.csv", index=False)
    pd.DataFrame(summaries).to_csv(args.out_dir / "cqr_summary.csv", index=False)
    pd.concat(interval_frames, ignore_index=True).to_csv(args.out_dir / "cqr_test_intervals.csv", index=False)
    cal_out = cal_meta.copy()
    cal_out["y_true"] = ycal
    for i, q in enumerate(QUANTILES):
        cal_out[f"q_{q:g}"] = qcal_sorted[:, i]
    cal_out["alpha_fusion"] = acal
    cal_out.to_csv(args.out_dir / "cqr_calibration_predictions.csv", index=False)
    print(pd.DataFrame(summaries).to_string(index=False))
    print("CQR is a UQ candidate only; it does not participate in point-model selection.")


if __name__ == "__main__":
    main()
