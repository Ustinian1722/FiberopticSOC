from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from run_representation_conditioning_diagnostic import PairTCN
from run_sequence_representation_benchmark import (
    PROFILES,
    CausalConv1d,
    TCNBlock,
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


MODEL_ORDER = (
    "IUW-TCN",
    "VI-Mamba",
    "VIW-Mamba",
    "DualMS-Mamba",
    "EO-Gated-TCN",
    "EO-Gated-Mamba",
    "EO-Gated-Mamba-TF",
)


class CausalMultiScaleEncoder(nn.Module):
    """Causal multi-scale local encoder that preserves the full sequence."""

    def __init__(self, in_dim: int, hidden: int = 24):
        super().__init__()
        if hidden % 3 != 0:
            raise ValueError("hidden must be divisible by 3")
        branch = hidden // 3
        self.branches = nn.ModuleList(
            [CausalConv1d(in_dim, branch, kernel_size=k) for k in (3, 5, 9)]
        )
        self.mix = nn.Conv1d(hidden, hidden, kernel_size=1)
        self.norm = nn.LayerNorm(hidden)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = x.transpose(1, 2)
        multi = torch.cat([F.gelu(conv(z)) for conv in self.branches], dim=1)
        multi = self.mix(multi).transpose(1, 2)
        return self.norm(F.gelu(multi))


class MambaSelectiveBlock(nn.Module):
    """Pure-PyTorch Mamba-style selective SSM block.

    This follows the core diagonal selective state-space mechanism: causal local
    convolution, input-dependent delta/B/C, negative diagonal A, recurrent selective
    scan, gated output projection, and a residual feed-forward block. It intentionally
    avoids CUDA-specific fused kernels so the research screen is reproducible on CPU CI.
    """

    def __init__(
        self,
        d_model: int,
        d_state: int = 8,
        dt_rank: int | None = None,
        conv_kernel: int = 3,
        expansion: int = 2,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.dt_rank = dt_rank or max(1, math.ceil(d_model / 16))
        self.norm = nn.LayerNorm(d_model)
        self.in_proj = nn.Linear(d_model, 2 * d_model)
        self.dw_conv = nn.Conv1d(
            d_model,
            d_model,
            kernel_size=conv_kernel,
            groups=d_model,
            bias=True,
        )
        self.left = conv_kernel - 1
        self.x_proj = nn.Linear(d_model, self.dt_rank + 2 * d_state, bias=False)
        self.dt_proj = nn.Linear(self.dt_rank, d_model, bias=True)
        nn.init.normal_(self.dt_proj.weight, mean=0.0, std=0.02)
        dt0 = 0.10
        with torch.no_grad():
            self.dt_proj.bias.fill_(math.log(math.expm1(dt0)))

        base = torch.arange(1, d_state + 1, dtype=torch.float32)
        self.A_log = nn.Parameter(torch.log(base).repeat(d_model, 1))
        self.D = nn.Parameter(torch.ones(d_model))
        self.out_proj = nn.Linear(d_model, d_model)

        ff_hidden = expansion * d_model
        self.ff_norm = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, ff_hidden),
            nn.GELU(),
            nn.Linear(ff_hidden, d_model),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        h = self.norm(x)
        u, z = self.in_proj(h).chunk(2, dim=-1)
        u = self.dw_conv(F.pad(u.transpose(1, 2), (self.left, 0))).transpose(1, 2)
        u = F.silu(u)

        params = self.x_proj(u)
        dt_raw = params[..., : self.dt_rank]
        b_t = params[..., self.dt_rank : self.dt_rank + self.d_state]
        c_t = params[..., self.dt_rank + self.d_state :]
        delta = F.softplus(self.dt_proj(dt_raw)) + 1e-4

        # Stable continuous-time diagonal state matrix.
        A = -torch.exp(self.A_log).to(dtype=u.dtype, device=u.device)
        state = torch.zeros(
            u.shape[0], self.d_model, self.d_state, dtype=u.dtype, device=u.device
        )
        outputs = []
        for t in range(u.shape[1]):
            dt = delta[:, t, :, None]
            delta_A = torch.exp(dt * A[None, :, :])
            delta_B_u = (
                dt
                * b_t[:, t, None, :]
                * u[:, t, :, None]
            )
            state = delta_A * state + delta_B_u
            y_t = (state * c_t[:, t, None, :]).sum(dim=-1)
            y_t = y_t + self.D[None, :] * u[:, t, :]
            outputs.append(y_t)

        y = torch.stack(outputs, dim=1)
        y = y * F.silu(z)
        x = residual + self.out_proj(y)
        x = x + self.ff(self.ff_norm(x))
        return x


class MambaBackbone(nn.Module):
    def __init__(self, d_model: int = 32, depth: int = 2, d_state: int = 8):
        super().__init__()
        self.blocks = nn.ModuleList(
            [MambaSelectiveBlock(d_model=d_model, d_state=d_state) for _ in range(depth)]
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            x = block(x)
        return self.norm(x)


class SequenceTCNBackbone(nn.Module):
    def __init__(self, d_model: int = 32):
        super().__init__()
        self.blocks = nn.Sequential(
            TCNBlock(d_model, 1),
            TCNBlock(d_model, 2),
            TCNBlock(d_model, 4),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.blocks(x.transpose(1, 2))
        return z.transpose(1, 2)


class SingleStreamMamba(nn.Module):
    def __init__(self, feature_idx: tuple[int, ...], d_model: int = 32):
        super().__init__()
        self.feature_idx = feature_idx
        self.input_proj = nn.Sequential(nn.Linear(len(feature_idx), d_model), nn.GELU())
        self.backbone = MambaBackbone(d_model=d_model, depth=2, d_state=8)
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model), nn.GELU(), nn.Linear(d_model, 1)
        )

    def forward(self, x: torch.Tensor):
        z = self.input_proj(x[:, :, self.feature_idx])
        h = self.backbone(z)[:, -1]
        return self.head(h).squeeze(-1), None


class DualModalEstimator(nn.Module):
    def __init__(
        self,
        *,
        optical_pair: tuple[int, int] = (2, 3),
        use_gate: bool = True,
        backbone: str = "mamba",
        branch_hidden: int = 24,
        d_model: int = 32,
    ):
        super().__init__()
        self.optical_pair = optical_pair
        self.use_gate = use_gate
        self.electrical = CausalMultiScaleEncoder(2, branch_hidden)
        self.optical = CausalMultiScaleEncoder(2, branch_hidden)
        if use_gate:
            self.gate = nn.Sequential(
                nn.Linear(2 * branch_hidden, branch_hidden),
                nn.GELU(),
                nn.Linear(branch_hidden, branch_hidden),
            )
        else:
            self.gate = None
        self.fusion = nn.Sequential(
            nn.Linear(2 * branch_hidden, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model),
        )
        if backbone == "mamba":
            self.backbone = MambaBackbone(d_model=d_model, depth=2, d_state=8)
        elif backbone == "tcn":
            self.backbone = SequenceTCNBackbone(d_model=d_model)
        else:
            raise ValueError(backbone)
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model), nn.GELU(), nn.Linear(d_model, 1)
        )

    def forward(self, x: torch.Tensor):
        e = self.electrical(x[:, :, (0, 1)])
        o = self.optical(x[:, :, self.optical_pair])
        if self.gate is not None:
            gate = torch.sigmoid(self.gate(torch.cat([e, o], dim=-1)))
            o_used = gate * o
            gate_summary = gate.mean(dim=(1, 2))
        else:
            o_used = o
            gate_summary = None
        fused = self.fusion(torch.cat([e, o_used], dim=-1))
        h = self.backbone(fused)[:, -1]
        pred = self.head(h).squeeze(-1)
        return pred, gate_summary


def build_model(name: str) -> nn.Module:
    if name == "IUW-TCN":
        return PairTCN((2, 3), None)
    if name == "VI-Mamba":
        return SingleStreamMamba((0, 1))
    if name == "VIW-Mamba":
        return SingleStreamMamba((0, 1, 2, 3))
    if name == "DualMS-Mamba":
        return DualModalEstimator(use_gate=False, backbone="mamba")
    if name == "EO-Gated-TCN":
        return DualModalEstimator(use_gate=True, backbone="tcn")
    if name == "EO-Gated-Mamba":
        return DualModalEstimator(use_gate=True, backbone="mamba")
    if name == "EO-Gated-Mamba-TF":
        return DualModalEstimator(optical_pair=(4, 5), use_gate=True, backbone="mamba")
    raise KeyError(name)


def representation_name(model_name: str) -> str:
    if model_name == "VI-Mamba":
        return "electrical_only"
    if model_name.endswith("-TF"):
        return "physics_decoupled_TF"
    return "raw_W"


def aggregate_screen(metrics: pd.DataFrame) -> pd.DataFrame:
    agg = (
        metrics.groupby("model", as_index=False)
        .agg(
            n_profiles=("MAE", "size"),
            params=("params", "first"),
            MAE_mean=("MAE", "mean"),
            MAE_std=("MAE", "std"),
            RMSE_mean=("RMSE", "mean"),
            R2_mean=("R2", "mean"),
            Q95_AE_mean=("Q95_AE", "mean"),
            MaxAE_mean=("MaxAE", "mean"),
        )
    )
    baseline = metrics.loc[metrics["model"] == "IUW-TCN", ["held_out_profile", "MAE"]].rename(
        columns={"MAE": "baseline_MAE"}
    )
    paired = metrics.merge(baseline, on="held_out_profile", how="left")
    paired["delta_MAE_vs_IUW_TCN"] = paired["MAE"] - paired["baseline_MAE"]
    paired["win_vs_IUW_TCN"] = paired["delta_MAE_vs_IUW_TCN"] < 0
    paired_summary = (
        paired.groupby("model", as_index=False)
        .agg(
            wins_vs_IUW_TCN=("win_vs_IUW_TCN", "sum"),
            mean_delta_MAE_vs_IUW_TCN=("delta_MAE_vs_IUW_TCN", "mean"),
        )
    )
    out = agg.merge(paired_summary, on="model", how="left")
    base_mean = float(out.loc[out["model"] == "IUW-TCN", "MAE_mean"].iloc[0])
    out["relative_MAE_improvement_vs_IUW_TCN"] = (
        base_mean - out["MAE_mean"]
    ) / base_mean
    order = {name: i for i, name in enumerate(MODEL_ORDER)}
    out["model_order"] = out["model"].map(order)
    out = out.sort_values(["MAE_mean", "Q95_AE_mean", "model_order"]).drop(columns="model_order")
    return out


def main() -> None:
    p = argparse.ArgumentParser(description="Equal-budget EO-Gated-Mamba architecture screen")
    p.add_argument("--data", type=Path, default=Path("data/extracted/SiC-18"))
    p.add_argument("--out-dir", type=Path, default=Path("results/eo_gated_mamba_screen"))
    p.add_argument("--window", type=int, default=64)
    p.add_argument("--train-stride", type=int, default=4)
    p.add_argument("--test-stride", type=int, default=1)
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--profiles", nargs="*", default=list(PROFILES))
    p.add_argument("--models", nargs="*", default=list(MODEL_ORDER))
    args = p.parse_args()

    unknown_profiles = set(args.profiles).difference(PROFILES)
    unknown_models = set(args.models).difference(MODEL_ORDER)
    if unknown_profiles:
        raise ValueError(f"Unknown profiles: {sorted(unknown_profiles)}")
    if unknown_models:
        raise ValueError(f"Unknown models: {sorted(unknown_models)}")
    if "IUW-TCN" not in args.models:
        raise ValueError("IUW-TCN must be present as the paired strong baseline")

    sources = load_sources(args.data)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    gate_rows: list[dict] = []

    # Architecture development deliberately uses only the previously development-exposed
    # 1C -> 2C direction. Reverse 2C -> 1C is not touched here.
    for held_out in args.profiles:
        train_raw = [
            s for s in sources if s["rate"] == "1C" and s["profile"] != held_out
        ]
        test_raw = [
            s for s in sources if s["rate"] == "2C" and s["profile"] == held_out
        ]
        if len(train_raw) != 5 or len(test_raw) != 1:
            raise RuntimeError(f"Bad compound-shift split for {held_out}")

        mean, std = train_normalizer(train_raw)
        train = normalize_sources(train_raw, mean, std)
        test = normalize_sources(test_raw, mean, std)
        train_ds = WindowDataset(train, args.window, args.train_stride)
        test_ds = WindowDataset(test, args.window, args.test_stride)
        test_loader = DataLoader(
            test_ds, batch_size=args.batch_size * 2, shuffle=False, num_workers=0
        )

        for model_name in args.models:
            seed_everything(args.seed)
            train_loader = DataLoader(
                train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0
            )
            model = build_model(model_name)
            params = count_params(model)
            print(
                f"\n=== held_out={held_out} model={model_name} params={params} "
                f"train={len(train_ds)} test={len(test_ds)} ==="
            )
            train_model(model, train_loader, device, args.epochs, args.lr)
            y, pred, _, gate = predict_model(model, test_loader, device)
            m = metric_dict(y, pred)
            rows.append(
                {
                    "protocol": "development_1C_to_2C_unseen_profile",
                    "train_rate": "1C",
                    "test_rate": "2C",
                    "held_out_profile": held_out,
                    "model": model_name,
                    "representation": representation_name(model_name),
                    "seed": args.seed,
                    "epochs": args.epochs,
                    "window": args.window,
                    "train_stride": args.train_stride,
                    "test_stride": args.test_stride,
                    "params": params,
                    "n_train": len(train_ds),
                    "n_test": len(y),
                    **m,
                }
            )
            if gate is not None:
                gate_rows.append(
                    {
                        "held_out_profile": held_out,
                        "model": model_name,
                        "representation": representation_name(model_name),
                        "gate_mean": float(gate.mean()),
                        "gate_std": float(gate.std()),
                        "gate_q10": float(np.quantile(gate, 0.10)),
                        "gate_q50": float(np.quantile(gate, 0.50)),
                        "gate_q90": float(np.quantile(gate, 0.90)),
                    }
                )
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()

    metrics = pd.DataFrame(rows)
    ranking = aggregate_screen(metrics)
    gates = pd.DataFrame(gate_rows)

    metrics.to_csv(args.out_dir / "profile_metrics.csv", index=False)
    ranking.to_csv(args.out_dir / "screen_ranking.csv", index=False)
    gates.to_csv(args.out_dir / "gate_by_profile.csv", index=False)

    print("\n=== EO-Gated-Mamba development ranking ===")
    print(ranking.to_string(index=False))
    if not gates.empty:
        print("\n=== optical gate summaries ===")
        print(gates.to_string(index=False))


if __name__ == "__main__":
    main()
