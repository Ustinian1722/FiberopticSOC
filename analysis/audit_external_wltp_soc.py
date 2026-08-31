from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

CELLS = ("P1", "P2")
TIME_COL = "Time / yyyy-mm-ddTHH:MM:SS.FFF"
REST_THRESHOLDS_A = (0.05, 0.1, 0.2, 0.5, 1.0)
LONG_REST_MIN_S = 3600.0


def integrate_signed_ah(t: pd.Series, i: np.ndarray) -> float:
    dt = t.diff().dt.total_seconds().to_numpy(float)
    total = 0.0
    for k in range(1, len(i)):
        dtk = dt[k]
        if np.isfinite(dtk) and 0 < dtk <= 30.0 and np.isfinite(i[k - 1]) and np.isfinite(i[k]):
            total += 0.5 * (float(i[k - 1]) + float(i[k])) * float(dtk) / 3600.0
    return float(total)


def contiguous_runs(mask: np.ndarray) -> list[tuple[int, int]]:
    if len(mask) == 0:
        return []
    changes = np.r_[True, mask[1:] != mask[:-1]]
    starts = np.flatnonzero(changes)
    ends = np.r_[starts[1:] - 1, len(mask) - 1]
    return [(int(a), int(b)) for a, b in zip(starts, ends) if bool(mask[a])]


def summarize_interval(cell: str, phase_id: int, t: pd.Series, i: np.ndarray, u: np.ndarray, s5: np.ndarray, a: int, b: int, kind: str) -> dict:
    ti = t.iloc[a:b + 1].reset_index(drop=True)
    ii, uu, ss = i[a:b + 1], u[a:b + 1], s5[a:b + 1]
    return {
        "cell": cell,
        "phase_id": phase_id,
        "kind": kind,
        "start_time": str(ti.iloc[0]),
        "end_time": str(ti.iloc[-1]),
        "n_samples": int(len(ii)),
        "duration_s": float((ti.iloc[-1] - ti.iloc[0]).total_seconds()),
        "signed_Ah_positive_discharge": integrate_signed_ah(ti, ii),
        "current_mean_A": float(np.nanmean(ii)),
        "current_std_A": float(np.nanstd(ii)),
        "current_min_A": float(np.nanmin(ii)),
        "current_max_A": float(np.nanmax(ii)),
        "positive_current_fraction": float(np.mean(ii > 0.05)),
        "negative_current_fraction": float(np.mean(ii < -0.05)),
        "voltage_start_V": float(uu[0]),
        "voltage_end_V": float(uu[-1]),
        "voltage_min_V": float(np.nanmin(uu)),
        "voltage_max_V": float(np.nanmax(uu)),
        "S5_start_nm": float(ss[0]),
        "S5_end_nm": float(ss[-1]),
        "S5_delta_end_nm": float(ss[-1] - ss[0]),
    }


def audit_cell(df: pd.DataFrame, cell: str, q_ref: float) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    i0 = pd.to_numeric(df[f"I_{cell} / A"], errors="coerce").to_numpy(float)
    u0 = pd.to_numeric(df[f"U_{cell} / V"], errors="coerce").to_numpy(float)
    s0 = pd.to_numeric(df[f"{cell}S5 / nm"], errors="coerce").to_numpy(float)
    t0 = pd.to_datetime(df[TIME_COL], errors="coerce")
    valid = np.isfinite(i0) & np.isfinite(u0) & np.isfinite(s0) & t0.notna().to_numpy()
    idx = np.flatnonzero(valid)
    if len(idx) < 2:
        raise RuntimeError(f"insufficient WLTP data for {cell}")
    i, u, s5 = i0[idx], u0[idx], s0[idx]
    t = t0.iloc[idx].reset_index(drop=True)
    dt = t.diff().dt.total_seconds().to_numpy(float)
    finite_dt = dt[np.isfinite(dt) & (dt > 0) & (dt <= 30.0)]

    q = np.quantile(np.abs(i), [0, .01, .05, .10, .25, .50, .75, .90, .95, .99, 1.0])
    summary = {
        "cell": cell,
        "Q_ref_Ah": q_ref,
        "n_valid_samples": int(len(i)),
        "median_dt_s": float(np.median(finite_dt)),
        "p95_dt_s": float(np.quantile(finite_dt, .95)),
        "voltage_min_V": float(np.nanmin(u)),
        "voltage_max_V": float(np.nanmax(u)),
        "current_min_A": float(np.nanmin(i)),
        "current_max_A": float(np.nanmax(i)),
        "abs_current_quantiles_A": {k: float(v) for k, v in zip(["p0","p1","p5","p10","p25","p50","p75","p90","p95","p99","p100"], q)},
    }

    rest_rows = []
    all_threshold_phase_rows = []
    for threshold in REST_THRESHOLDS_A:
        mask = np.abs(i) <= threshold
        long_rests = []
        for a, b in contiguous_runs(mask):
            dur = float((t.iloc[b] - t.iloc[a]).total_seconds())
            if dur >= LONG_REST_MIN_S:
                long_rests.append((a, b, dur))
                rest_rows.append({
                    "cell": cell,
                    "threshold_A": threshold,
                    "start_time": str(t.iloc[a]),
                    "end_time": str(t.iloc[b]),
                    "duration_s": dur,
                    "n_samples": int(b - a + 1),
                    "current_mean_A": float(np.nanmean(i[a:b+1])),
                    "current_abs_max_A": float(np.nanmax(np.abs(i[a:b+1]))),
                    "voltage_start_V": float(u[a]),
                    "voltage_end_V": float(u[b]),
                })

        # Build non-rest phases only for each threshold to see which threshold reproduces repeated WLTP blocks.
        bounds = []
        cur = 0
        for a, b, _ in long_rests:
            if a > cur:
                bounds.append((cur, a - 1))
            cur = b + 1
        if cur < len(i):
            bounds.append((cur, len(i) - 1))
        for phase_id, (a, b) in enumerate(bounds, 1):
            if b - a + 1 < 100:
                continue
            row = summarize_interval(cell, phase_id, t, i, u, s5, a, b, "between_long_rests")
            row["rest_threshold_A"] = threshold
            row["looks_WLTP_dynamic"] = bool(
                row["n_samples"] >= 500
                and row["current_min_A"] < -5.0
                and row["current_max_A"] > 5.0
                and row["current_std_A"] >= 3.0
            )
            all_threshold_phase_rows.append(row)

    summary["long_rest_counts"] = {
        str(th): int(sum(1 for r in rest_rows if abs(r["threshold_A"] - th) < 1e-12))
        for th in REST_THRESHOLDS_A
    }
    return summary, pd.DataFrame(rest_rows), pd.DataFrame(all_threshold_phase_rows)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--wltp-csv", type=Path, required=True)
    p.add_argument("--freeze", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    args = p.parse_args()

    freeze = json.loads(args.freeze.read_text(encoding="utf-8"))
    qmap = freeze["q_ref_Ah"]
    usecols = [TIME_COL]
    for c in CELLS:
        usecols += [f"I_{c} / A", f"U_{c} / V", f"{c}S5 / nm"]
    df = pd.read_csv(args.wltp_csv, usecols=usecols, low_memory=False)

    summaries, rests_all, phases_all = [], [], []
    for c in CELLS:
        summary, rests, phases = audit_cell(df, c, float(qmap[c]))
        summaries.append(summary)
        rests_all.append(rests)
        phases_all.append(phases)

    rests_df = pd.concat(rests_all, ignore_index=True) if rests_all else pd.DataFrame()
    phases_df = pd.concat(phases_all, ignore_index=True) if phases_all else pd.DataFrame()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([{k: v for k, v in s.items() if k != "abs_current_quantiles_A"} | {f"absI_{q}": val for q, val in s["abs_current_quantiles_A"].items()} for s in summaries]).to_csv(args.out_dir / "wltp_cell_summary.csv", index=False)
    rests_df.to_csv(args.out_dir / "wltp_long_rest_threshold_scan.csv", index=False)
    phases_df.to_csv(args.out_dir / "wltp_phase_threshold_scan.csv", index=False)
    (args.out_dir / "wltp_audit_summary.json").write_text(json.dumps({"cells": summaries}, indent=2), encoding="utf-8")
    print("=== WLTP summary ===")
    print(json.dumps({"cells": summaries}, indent=2))
    print("=== long rest threshold scan ===")
    print(rests_df.to_string(index=False))
    print("=== phases between long rests ===")
    print(phases_df.to_string(index=False))


if __name__ == "__main__":
    main()
