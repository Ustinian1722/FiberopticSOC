from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from torch.utils.data import DataLoader, Dataset

CELLS = ("A1", "A2", "P1", "P2")
MODELS = ("VI-TCN", "VI-S5rel-TCN")
FEATURE_INDEX = {"VI-TCN": (0, 1), "VI-S5rel-TCN": (0, 1, 2)}


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


class CausalConv1d(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int, dilation: int = 1):
        super().__init__()
        self.left = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size, dilation=dilation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(F.pad(x, (self.left, 0)))


class TCNBlock(nn.Module):
    def __init__(self, channels: int, dilation: int):
        super().__init__()
        self.conv1 = CausalConv1d(channels, channels, 3, dilation=dilation)
        self.conv2 = CausalConv1d(channels, channels, 3, dilation=dilation)
        self.norm1 = nn.GroupNorm(1, channels)
        self.norm2 = nn.GroupNorm(1, channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = F.gelu(self.norm1(self.conv1(x)))
        z = self.norm2(self.conv2(z))
        return F.gelu(x + z)


class ExternalTCN(nn.Module):
    def __init__(self, in_dim: int, hidden: int = 24):
        super().__init__()
        self.proj = nn.Conv1d(in_dim, hidden, 1)
        self.blocks = nn.Sequential(TCNBlock(hidden, 1), TCNBlock(hidden, 2), TCNBlock(hidden, 4))
        self.head = nn.Sequential(nn.Linear(hidden, hidden), nn.GELU(), nn.Linear(hidden, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.proj(x.transpose(1, 2))
        z = self.blocks(z)
        return self.head(z[:, :, -1]).squeeze(-1)


class WindowDataset(Dataset):
    def __init__(self, segments: list[dict], window: int, stride: int, feature_idx: tuple[int, ...]):
        self.segments = segments
        self.window = window
        self.feature_idx = feature_idx
        self.index: list[tuple[int, int]] = []
        for seg_id, seg in enumerate(segments):
            for end in range(window - 1, len(seg["y"]), stride):
                self.index.append((seg_id, end))

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, idx: int):
        seg_id, end = self.index[idx]
        seg = self.segments[seg_id]
        start = end - self.window + 1
        x = seg["x"][start : end + 1, self.feature_idx]
        return torch.from_numpy(x), torch.tensor(seg["y"][end], dtype=torch.float32), seg_id


def metric_dict(y: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    ae = np.abs(y - pred)
    return {
        "MAE": float(mean_absolute_error(y, pred)),
        "RMSE": float(mean_squared_error(y, pred) ** 0.5),
        "R2": float(r2_score(y, pred)),
        "MaxAE": float(ae.max()),
        "Q95_AE": float(np.quantile(ae, 0.95)),
    }


def load_segments(npz_path: Path, metadata_path: Path) -> list[dict]:
    meta = pd.read_csv(metadata_path)
    data = np.load(npz_path)
    segments = []
    for row in meta.itertuples(index=False):
        x = data[f"{row.key}__x"].astype(np.float32)
        y = data[f"{row.key}__y"].astype(np.float32)
        if len(x) != int(row.n_samples) or len(y) != int(row.n_samples):
            raise RuntimeError(f"length mismatch for {row.key}")
        segments.append({
            "key": row.key,
            "cell": row.cell,
            "rate_C": float(row.rate_C),
            "segment_id": int(row.segment_id),
            "x": x,
            "y": y,
        })
    return segments


def fit_normalizer(train_segments: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    x = np.concatenate([s["x"] for s in train_segments], axis=0).astype(np.float64)
    mean = x.mean(axis=0)
    std = x.std(axis=0)
    std[std < 1e-12] = 1.0
    return mean.astype(np.float32), std.astype(np.float32)


def apply_normalizer(segments: list[dict], mean: np.ndarray, std: np.ndarray) -> list[dict]:
    out = []
    for seg in segments:
        s = dict(seg)
        s["x"] = ((seg["x"] - mean) / std).astype(np.float32)
        out.append(s)
    return out


def train(model: nn.Module, loader: DataLoader, device: torch.device, epochs: int, lr: float) -> None:
    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    loss_fn = nn.MSELoss()
    for epoch in range(epochs):
        model.train()
        total = 0.0
        n = 0
        for x, y, _ in loader:
            x = x.to(device)
            y = y.to(device)
            opt.zero_grad(set_to_none=True)
            pred = model(x)
            loss = loss_fn(pred, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            opt.step()
            total += float(loss.detach()) * len(y)
            n += len(y)
        print(f"epoch={epoch+1:02d} train_mse={total/max(n,1):.8f}")


@torch.no_grad()
def predict(model: nn.Module, loader: DataLoader, device: torch.device):
    model.eval()
    ys, preds, seg_ids = [], [], []
    for x, y, sid in loader:
        pred = model(x.to(device)).cpu().numpy()
        ys.append(y.numpy())
        preds.append(pred)
        seg_ids.append(sid.numpy())
    y = np.concatenate(ys)
    pred = np.clip(np.concatenate(preds), 0.0, 1.0)
    sid = np.concatenate(seg_ids)
    return y, pred, sid


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", type=Path, required=True)
    p.add_argument("--metadata", type=Path, required=True)
    p.add_argument("--held-out-cell", choices=CELLS, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--window", type=int, default=64)
    p.add_argument("--train-stride", type=int, default=4)
    p.add_argument("--test-stride", type=int, default=1)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    all_segments = load_segments(args.dataset, args.metadata)
    train_raw = [s for s in all_segments if s["cell"] != args.held_out_cell]
    test_raw = [s for s in all_segments if s["cell"] == args.held_out_cell]
    if not train_raw or not test_raw:
        raise RuntimeError("invalid LOCO split")
    mean, std = fit_normalizer(train_raw)
    train_segments = apply_normalizer(train_raw, mean, std)
    test_segments = apply_normalizer(test_raw, mean, std)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    np.savez(args.out_dir / "source_normalizer.npz", mean=mean, std=std)

    fold_rows = []
    rate_rows = []
    segment_rows = []
    prediction_rows = []

    for model_name in MODELS:
        feature_idx = FEATURE_INDEX[model_name]
        train_ds = WindowDataset(train_segments, args.window, args.train_stride, feature_idx)
        test_ds = WindowDataset(test_segments, args.window, args.test_stride, feature_idx)
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
        test_loader = DataLoader(test_ds, batch_size=args.batch_size * 2, shuffle=False, num_workers=0)
        seed_everything(args.seed)
        model = ExternalTCN(len(feature_idx), hidden=24)
        params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"\n=== held_out={args.held_out_cell} model={model_name} params={params} train_windows={len(train_ds)} test_windows={len(test_ds)} ===")
        train(model, train_loader, device, args.epochs, args.lr)
        y, pred, seg_ids = predict(model, test_loader, device)
        fold_rows.append({
            "protocol": "external_E1_leave_one_cell_out",
            "held_out_cell": args.held_out_cell,
            "model": model_name,
            "seed": args.seed,
            "epochs": args.epochs,
            "window": args.window,
            "train_stride": args.train_stride,
            "test_stride": args.test_stride,
            "params": params,
            "n_train_segments": len(train_segments),
            "n_test_segments": len(test_segments),
            "n_train_windows": len(train_ds),
            "n_test_windows": len(test_ds),
            "source_only_normalization": True,
            "held_out_cell_used_for_training": False,
            "held_out_cell_used_for_selection": False,
            **metric_dict(y, pred),
        })

        for local_sid, seg in enumerate(test_segments):
            mask = seg_ids == local_sid
            if not mask.any():
                continue
            m = metric_dict(y[mask], pred[mask])
            segment_rows.append({
                "held_out_cell": args.held_out_cell,
                "model": model_name,
                "segment_key": seg["key"],
                "rate_C": seg["rate_C"],
                "n_windows": int(mask.sum()),
                **m,
            })
            for yy, pp in zip(y[mask], pred[mask]):
                prediction_rows.append({
                    "held_out_cell": args.held_out_cell,
                    "model": model_name,
                    "segment_key": seg["key"],
                    "rate_C": seg["rate_C"],
                    "y_true": float(yy),
                    "y_pred": float(pp),
                    "abs_error": float(abs(yy - pp)),
                })

        seg_df = pd.DataFrame([r for r in segment_rows if r["held_out_cell"] == args.held_out_cell and r["model"] == model_name])
        for rate in sorted(seg_df.rate_C.unique()):
            p_df = pd.DataFrame([r for r in prediction_rows if r["held_out_cell"] == args.held_out_cell and r["model"] == model_name and r["rate_C"] == rate])
            rate_rows.append({
                "held_out_cell": args.held_out_cell,
                "model": model_name,
                "rate_C": float(rate),
                "n_segments": int((seg_df.rate_C == rate).sum()),
                "n_windows": int(len(p_df)),
                **metric_dict(p_df.y_true.to_numpy(), p_df.y_pred.to_numpy()),
            })

        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    fold_df = pd.DataFrame(fold_rows)
    rate_df = pd.DataFrame(rate_rows)
    segment_df = pd.DataFrame(segment_rows)
    pred_df = pd.DataFrame(prediction_rows)
    fold_df.to_csv(args.out_dir / "e1_fold_metrics.csv", index=False)
    rate_df.to_csv(args.out_dir / "e1_rate_metrics.csv", index=False)
    segment_df.to_csv(args.out_dir / "e1_segment_metrics.csv", index=False)
    pred_df.to_csv(args.out_dir / "e1_predictions.csv", index=False)

    print("\n=== fold metrics ===")
    print(fold_df.to_string(index=False))
    print("\n=== rate metrics ===")
    print(rate_df.to_string(index=False))
    print("\n=== source-only normalizer ===")
    print(json.dumps({"mean": mean.tolist(), "std": std.tolist()}, indent=2))


if __name__ == "__main__":
    main()
