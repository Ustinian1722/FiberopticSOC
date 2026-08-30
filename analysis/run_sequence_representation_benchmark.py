from __future__ import annotations

import argparse
import math
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from torch.utils.data import DataLoader, Dataset

PROFILES = ("HWFET", "LA92", "NEDC", "NYCC", "US06", "WLTC")
RATES = ("1C", "2C")
COLUMNS = (
    "Voltage_V",
    "Current_A",
    "Wavelength_1",
    "Wavelength_2",
    "temperature_℃",
    "force_N",
)
FORBIDDEN = {"SOC", "dis_cap", "Time_s"}

FEATURE_INDEX = {
    "VI": (0, 1),
    "VI+W": (0, 1, 2, 3),
    "VI+TF": (0, 1, 4, 5),
    # Explicit redundancy control only; W and TF are two invertible views of the same FBG DoF.
    "VI+W+TF": (0, 1, 2, 3, 4, 5),
}


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(False)


def discover_workbooks(root: Path) -> list[Path]:
    files = []
    for path in sorted(root.rglob("*.xlsx")):
        stem = path.stem
        if "_" not in stem:
            continue
        profile, rate = stem.rsplit("_", 1)
        if profile in PROFILES and rate in RATES:
            files.append(path)
    if len(files) != 12:
        raise FileNotFoundError(f"Expected 12 SiC-18 workbooks under {root}, found {len(files)}")
    return files


def load_sources(root: Path) -> list[dict]:
    sources = []
    for path in discover_workbooks(root):
        profile, rate = path.stem.rsplit("_", 1)
        df = pd.read_excel(path)
        required = set(COLUMNS) | {"SOC"}
        missing = required.difference(df.columns)
        if missing:
            raise ValueError(f"{path.name} missing columns: {sorted(missing)}")
        if FORBIDDEN.intersection(COLUMNS):
            raise RuntimeError("Forbidden predictor entered feature definition")
        values = df[list(COLUMNS)].to_numpy(dtype=np.float32)
        target = df["SOC"].to_numpy(dtype=np.float32)
        if not np.isfinite(values).all() or not np.isfinite(target).all():
            raise ValueError(f"Non-finite values found in {path.name}")
        sources.append(
            {
                "name": path.name,
                "profile": profile,
                "rate": rate,
                "x": values,
                "y": target,
            }
        )
    return sources


def train_normalizer(sources: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    x = np.concatenate([s["x"] for s in sources], axis=0).astype(np.float64)
    mean = x.mean(axis=0)
    std = x.std(axis=0)
    std[std < 1e-12] = 1.0
    return mean.astype(np.float32), std.astype(np.float32)


def normalize_sources(sources: list[dict], mean: np.ndarray, std: np.ndarray) -> list[dict]:
    out = []
    for s in sources:
        item = dict(s)
        item["x"] = ((s["x"] - mean) / std).astype(np.float32)
        out.append(item)
    return out


class WindowDataset(Dataset):
    def __init__(self, sources: list[dict], window: int, stride: int):
        self.sources = sources
        self.window = window
        self.index: list[tuple[int, int]] = []
        for source_id, src in enumerate(sources):
            for end in range(window - 1, len(src["y"]), stride):
                self.index.append((source_id, end))

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, idx: int):
        source_id, end = self.index[idx]
        src = self.sources[source_id]
        start = end - self.window + 1
        x = torch.from_numpy(src["x"][start : end + 1])
        y = torch.tensor(src["y"][end], dtype=torch.float32)
        return x, y, source_id


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


class TCNEncoder(nn.Module):
    def __init__(self, in_dim: int, hidden: int):
        super().__init__()
        self.proj = nn.Conv1d(in_dim, hidden, 1)
        self.blocks = nn.Sequential(TCNBlock(hidden, 1), TCNBlock(hidden, 2), TCNBlock(hidden, 4))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # [B,T,F] -> [B,F,T], all convolutions are left-padded only.
        z = self.proj(x.transpose(1, 2))
        z = self.blocks(z)
        return z[:, :, -1]


class SingleViewTCN(nn.Module):
    def __init__(self, feature_idx: tuple[int, ...], hidden: int = 24):
        super().__init__()
        self.feature_idx = feature_idx
        self.encoder = TCNEncoder(len(feature_idx), hidden)
        self.head = nn.Sequential(nn.Linear(hidden, hidden), nn.GELU(), nn.Linear(hidden, 1))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:
        h = self.encoder(x[:, :, self.feature_idx])
        pred = self.head(h).squeeze(-1)
        return pred, None


class RepresentationAdaptiveTCN(nn.Module):
    """Convexly selects raw-FBG vs physics-decoupled FBG latent views.

    W1/W2 and T/F are information-equivalent in this release; the gate is intended to
    adapt feature geometry/conditioning under domain shift, not to claim extra sensing DoF.
    """

    def __init__(self, electrical_hidden: int = 24, optical_hidden: int = 16):
        super().__init__()
        self.electrical = TCNEncoder(2, electrical_hidden)
        self.raw_optical = TCNEncoder(2, optical_hidden)
        self.physics = TCNEncoder(2, optical_hidden)
        self.gate = nn.Sequential(
            nn.Linear(electrical_hidden + 2 * optical_hidden, 24),
            nn.GELU(),
            nn.Linear(24, 1),
        )
        self.head = nn.Sequential(
            nn.Linear(electrical_hidden + optical_hidden, 32),
            nn.GELU(),
            nn.Linear(32, 1),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        e = self.electrical(x[:, :, (0, 1)])
        w = self.raw_optical(x[:, :, (2, 3)])
        tf = self.physics(x[:, :, (4, 5)])
        g = torch.sigmoid(self.gate(torch.cat([e, w, tf], dim=-1)))
        optical = g * w + (1.0 - g) * tf
        pred = self.head(torch.cat([e, optical], dim=-1)).squeeze(-1)
        return pred, g.squeeze(-1)


def build_model(name: str) -> nn.Module:
    if name == "RA-TCN":
        return RepresentationAdaptiveTCN()
    return SingleViewTCN(FEATURE_INDEX[name])


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def train_model(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    epochs: int,
    lr: float,
) -> None:
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    loss_fn = nn.MSELoss()
    model.train()
    for epoch in range(epochs):
        total = 0.0
        n = 0
        for x, y, _ in loader:
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad(set_to_none=True)
            pred, _ = model(x)
            loss = loss_fn(pred, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()
            total += float(loss.detach()) * len(y)
            n += len(y)
        print(f"epoch={epoch+1:02d} train_mse={total/max(n,1):.8f}")


@torch.no_grad()
def predict_model(model: nn.Module, loader: DataLoader, device: torch.device):
    model.eval()
    ys, preds, source_ids, gates = [], [], [], []
    for x, y, src in loader:
        x = x.to(device)
        pred, gate = model(x)
        ys.append(y.numpy())
        preds.append(pred.cpu().numpy())
        source_ids.append(src.numpy())
        if gate is not None:
            gates.append(gate.cpu().numpy())
    y = np.concatenate(ys)
    pred = np.clip(np.concatenate(preds), 0.0, 1.0)
    src = np.concatenate(source_ids)
    gate = np.concatenate(gates) if gates else None
    return y, pred, src, gate


def metric_dict(y: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    ae = np.abs(y - pred)
    return {
        "MAE": float(mean_absolute_error(y, pred)),
        "RMSE": float(mean_squared_error(y, pred) ** 0.5),
        "R2": float(r2_score(y, pred)),
        "MaxAE": float(ae.max()),
        "Q95_AE": float(np.quantile(ae, 0.95)),
    }


@dataclass(frozen=True)
class Split:
    name: str
    train_rate: str
    test_rate: str


def run_cross_rate(args, all_sources: list[dict], device: torch.device):
    models = ["VI", "VI+W", "VI+TF", "VI+W+TF", "RA-TCN"]
    records = []
    profile_records = []
    gate_records = []

    for split in (Split("1C_to_2C", "1C", "2C"), Split("2C_to_1C", "2C", "1C")):
        train_raw = [s for s in all_sources if s["rate"] == split.train_rate]
        test_raw = [s for s in all_sources if s["rate"] == split.test_rate]
        mean, std = train_normalizer(train_raw)
        train_sources = normalize_sources(train_raw, mean, std)
        test_sources = normalize_sources(test_raw, mean, std)

        train_ds = WindowDataset(train_sources, args.window, args.train_stride)
        test_ds = WindowDataset(test_sources, args.window, args.test_stride)
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
        test_loader = DataLoader(test_ds, batch_size=args.batch_size * 2, shuffle=False, num_workers=0)

        print(f"\n=== {split.name}: train_windows={len(train_ds)}, test_windows={len(test_ds)} ===")
        for model_name in models:
            seed_everything(args.seed)
            model = build_model(model_name)
            params = count_params(model)
            print(f"\n--- model={model_name}, params={params} ---")
            train_model(model, train_loader, device, args.epochs, args.lr)
            y, pred, source_ids, gate = predict_model(model, test_loader, device)
            overall = metric_dict(y, pred)
            records.append(
                {
                    "protocol": "cross_rate_causal_window",
                    "split": split.name,
                    "train_rate": split.train_rate,
                    "test_rate": split.test_rate,
                    "model": model_name,
                    "window": args.window,
                    "train_stride": args.train_stride,
                    "test_stride": args.test_stride,
                    "epochs": args.epochs,
                    "seed": args.seed,
                    "params": params,
                    "n_test": len(y),
                    **overall,
                }
            )

            for source_id, src in enumerate(test_sources):
                mask = source_ids == source_id
                if not mask.any():
                    continue
                m = metric_dict(y[mask], pred[mask])
                profile_records.append(
                    {
                        "split": split.name,
                        "model": model_name,
                        "profile": src["profile"],
                        "rate": src["rate"],
                        "n_test": int(mask.sum()),
                        **m,
                    }
                )
                if gate is not None:
                    g = gate[mask]
                    gate_records.append(
                        {
                            "split": split.name,
                            "profile": src["profile"],
                            "rate": src["rate"],
                            "gate_raw_mean": float(g.mean()),
                            "gate_raw_std": float(g.std()),
                            "gate_raw_q10": float(np.quantile(g, 0.10)),
                            "gate_raw_q50": float(np.quantile(g, 0.50)),
                            "gate_raw_q90": float(np.quantile(g, 0.90)),
                        }
                    )
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()

    return pd.DataFrame(records), pd.DataFrame(profile_records), pd.DataFrame(gate_records)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("data/extracted"))
    parser.add_argument("--out-dir", type=Path, default=Path("results/sequence_representation"))
    parser.add_argument("--window", type=int, default=64)
    parser.add_argument("--train-stride", type=int, default=4)
    parser.add_argument("--test-stride", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}")
    sources = load_sources(args.data)
    results, by_profile, gates = run_cross_rate(args, sources, device)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    results.to_csv(args.out_dir / "cross_rate_results.csv", index=False)
    by_profile.to_csv(args.out_dir / "cross_rate_by_profile.csv", index=False)
    gates.to_csv(args.out_dir / "ra_tcn_gate_by_profile.csv", index=False)

    print("\n=== Cross-rate causal sequence summary ===")
    print(results.sort_values(["split", "MAE"]).to_string(index=False))
    if not gates.empty:
        print("\n=== RA-TCN raw-view gate (1=raw W, 0=physics TF) ===")
        print(gates.to_string(index=False))


if __name__ == "__main__":
    main()
