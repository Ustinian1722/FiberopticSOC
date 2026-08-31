from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

CELLS = ("A1", "A2", "P1", "P2")
TIME_COL = "Time / yyyy-mm-ddTHH:MM:SS.FFF"
MAX_DT_S = 30.0


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--validation-csv", type=Path, required=True)
    p.add_argument("--segments", type=Path, required=True)
    p.add_argument("--freeze", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--chunksize", type=int, default=150000)
    return p.parse_args()


def cumulative_discharge_ah(times: np.ndarray, current: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if len(times) == 0:
        raise ValueError("empty segment")
    if len(times) == 1:
        return np.zeros(1, dtype=np.float64), np.zeros(0, dtype=np.float64)
    dt = np.diff(times.astype("datetime64[ns]").astype(np.int64)).astype(np.float64) / 1e9
    if (dt <= 0).any() or (dt > MAX_DT_S).any():
        bad = dt[(dt <= 0) | (dt > MAX_DT_S)]
        raise ValueError(f"invalid intra-segment dt values: {bad[:10]}")
    increments = 0.5 * (current[:-1] + current[1:]) * dt / 3600.0
    cum = np.concatenate([np.zeros(1, dtype=np.float64), np.cumsum(increments)])
    return cum, dt


def main() -> None:
    args = parse_args()
    freeze = json.loads(args.freeze.read_text(encoding="utf-8"))
    segments = pd.read_csv(args.segments)
    if freeze["status"] != "FROZEN_EXTERNAL_LABEL_RECONSTRUCTION_V2":
        raise RuntimeError("unexpected freeze status")
    if len(segments) != int(freeze["eligible_validation_discharges"]):
        raise RuntimeError(f"expected {freeze['eligible_validation_discharges']} frozen segments, got {len(segments)}")

    segments["start_ts"] = pd.to_datetime(segments["start_time"], errors="raise")
    segments["end_ts"] = pd.to_datetime(segments["end_time"], errors="raise")
    segments["key"] = segments.apply(lambda r: f"{r.cell}_{int(r.segment_id)}", axis=1)
    if segments["key"].duplicated().any():
        raise RuntimeError("duplicate frozen segment key")

    q_ref = {k: float(v) for k, v in freeze["q_ref_Ah"].items()}
    for c in CELLS:
        vals = segments.loc[segments.cell == c, "Q_ref_Ah"].to_numpy(dtype=float)
        if len(vals) == 0 or not np.allclose(vals, q_ref[c], rtol=0, atol=1e-10):
            raise RuntimeError(f"Q_ref mismatch for {c}")

    usecols = [TIME_COL]
    for c in CELLS:
        usecols.extend([f"I_{c} / A", f"U_{c} / V", f"{c}S5 / nm"])

    buffers: dict[str, dict[str, list[np.ndarray]]] = {
        key: {"time": [], "i": [], "u": [], "s5": []} for key in segments["key"]
    }
    seg_by_cell = {c: segments[segments.cell == c].copy() for c in CELLS}

    for chunk_idx, chunk in enumerate(
        pd.read_csv(args.validation_csv, usecols=usecols, chunksize=args.chunksize, low_memory=False)
    ):
        times = pd.to_datetime(chunk[TIME_COL], errors="coerce")
        valid_t = times.notna()
        if not valid_t.any():
            continue
        tmin, tmax = times[valid_t].min(), times[valid_t].max()
        for c in CELLS:
            current = pd.to_numeric(chunk[f"I_{c} / A"], errors="coerce").to_numpy(dtype=float)
            voltage = pd.to_numeric(chunk[f"U_{c} / V"], errors="coerce").to_numpy(dtype=float)
            s5 = pd.to_numeric(chunk[f"{c}S5 / nm"], errors="coerce").to_numpy(dtype=float)
            candidates = seg_by_cell[c]
            candidates = candidates[(candidates.start_ts <= tmax) & (candidates.end_ts >= tmin)]
            if candidates.empty:
                continue
            t_np = times.to_numpy(dtype="datetime64[ns]")
            for row in candidates.itertuples(index=False):
                mask = (times >= row.start_ts) & (times <= row.end_ts)
                mask_np = mask.to_numpy(dtype=bool)
                mask_np &= np.isfinite(current) & np.isfinite(voltage) & np.isfinite(s5) & (current > 0.05)
                if not mask_np.any():
                    continue
                b = buffers[row.key]
                b["time"].append(t_np[mask_np])
                b["i"].append(current[mask_np])
                b["u"].append(voltage[mask_np])
                b["s5"].append(s5[mask_np])
        print(f"processed chunk {chunk_idx + 1}, rows={len(chunk)}")

    arrays: dict[str, np.ndarray] = {}
    meta_rows: list[dict] = []
    for row in segments.itertuples(index=False):
        b = buffers[row.key]
        if not b["time"]:
            raise RuntimeError(f"no rows extracted for {row.key}")
        t = np.concatenate(b["time"])
        i = np.concatenate(b["i"]).astype(np.float64)
        u = np.concatenate(b["u"]).astype(np.float64)
        s5 = np.concatenate(b["s5"]).astype(np.float64)
        order = np.argsort(t.astype(np.int64), kind="stable")
        t, i, u, s5 = t[order], i[order], u[order], s5[order]

        expected_n = int(row.n_samples)
        if len(t) != expected_n:
            raise RuntimeError(f"{row.key}: extracted {len(t)} rows != frozen n_samples {expected_n}")
        cum_ah, dt = cumulative_discharge_ah(t, i)
        q = q_ref[row.cell]
        soc = 1.0 - cum_ah / q
        s5_rel = s5 - s5[0]
        x = np.column_stack([u, i, s5_rel]).astype(np.float32)
        y = soc.astype(np.float32)
        if not np.isfinite(x).all() or not np.isfinite(y).all():
            raise RuntimeError(f"non-finite model array in {row.key}")

        terminal_ah = float(cum_ah[-1])
        terminal_soc = float(soc[-1])
        if not np.isclose(terminal_ah, float(row.integrated_ah), rtol=2e-5, atol=2e-4):
            raise RuntimeError(f"{row.key}: Ah mismatch {terminal_ah} vs frozen {row.integrated_ah}")
        if not np.isclose(terminal_soc, float(row.SOC_end_from_Qref), rtol=0, atol=3e-5):
            raise RuntimeError(f"{row.key}: terminal SOC mismatch {terminal_soc} vs frozen {row.SOC_end_from_Qref}")

        arrays[f"{row.key}__x"] = x
        arrays[f"{row.key}__y"] = y
        meta_rows.append(
            {
                "key": row.key,
                "cell": row.cell,
                "segment_id": int(row.segment_id),
                "rate_C": float(row.assigned_rate_C),
                "n_samples": int(len(y)),
                "Q_ref_Ah": q,
                "terminal_Ah": terminal_ah,
                "terminal_SOC": terminal_soc,
                "median_dt_s": float(np.median(dt)) if len(dt) else np.nan,
                "p95_dt_s": float(np.quantile(dt, 0.95)) if len(dt) else np.nan,
                "s5_start_nm": float(s5[0]),
                "s5_abs_min_nm": float(s5.min()),
                "s5_abs_max_nm": float(s5.max()),
                "s5_rel_min_nm": float(s5_rel.min()),
                "s5_rel_max_nm": float(s5_rel.max()),
                "soc_min": float(y.min()),
                "soc_max": float(y.max()),
            }
        )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out_dir / "external_e1_dataset.npz", **arrays)
    meta = pd.DataFrame(meta_rows).sort_values(["cell", "rate_C", "segment_id"])
    meta.to_csv(args.out_dir / "external_e1_segment_metadata.csv", index=False)

    summary = {
        "n_segments": int(len(meta)),
        "n_samples": int(meta.n_samples.sum()),
        "cells": {c: int((meta.cell == c).sum()) for c in CELLS},
        "rates": {str(r): int((meta.rate_C == r).sum()) for r in (0.2, 0.5, 1.0)},
        "median_segment_dt_s": float(meta.median_dt_s.median()),
        "input_columns": ["Voltage_V", "Current_A", "S5_rel_nm"],
        "soc_definition": freeze["soc_definition"],
        "s5_definition": freeze["sensor_mainline"],
        "frozen_source_run_id": freeze["source_run_id"],
        "frozen_source_artifact_id": freeze["source_artifact_id"],
    }
    (args.out_dir / "external_e1_dataset_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(meta.groupby(["cell", "rate_C"], as_index=False).agg(n_segments=("key", "size"), n_samples=("n_samples", "sum"), mean_terminal_SOC=("terminal_SOC", "mean")).to_string(index=False))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
