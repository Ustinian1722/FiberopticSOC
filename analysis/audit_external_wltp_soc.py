from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

CELLS = ("P1", "P2")
TIME_COL = "Time / yyyy-mm-ddTHH:MM:SS.FFF"
BLOCK_GAP_S = 300.0
MIN_BLOCK_SAMPLES = 50


def integrate_signed_ah(t: pd.Series, i: np.ndarray) -> float:
    dt = t.diff().dt.total_seconds().to_numpy(float)
    total = 0.0
    for k in range(1, len(i)):
        dtk = dt[k]
        if np.isfinite(dtk) and 0 < dtk <= 30.0 and np.isfinite(i[k - 1]) and np.isfinite(i[k]):
            total += 0.5 * (float(i[k - 1]) + float(i[k])) * float(dtk) / 3600.0
    return float(total)


def scan_cell_blocks(df: pd.DataFrame, cell: str, q_ref: float) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    i = pd.to_numeric(df[f"I_{cell} / A"], errors="coerce").to_numpy(float)
    u = pd.to_numeric(df[f"U_{cell} / V"], errors="coerce").to_numpy(float)
    s5 = pd.to_numeric(df[f"{cell}S5 / nm"], errors="coerce").to_numpy(float)
    t = pd.to_datetime(df[TIME_COL], errors="coerce")
    valid = np.isfinite(i) & np.isfinite(u) & np.isfinite(s5) & t.notna().to_numpy()
    idx = np.flatnonzero(valid)
    if len(idx) < 2:
        raise RuntimeError(f"insufficient valid WLTP data for {cell}")

    i = i[idx]
    u = u[idx]
    s5 = s5[idx]
    t = t.iloc[idx].reset_index(drop=True)
    dt = t.diff().dt.total_seconds().to_numpy(float)

    new_block = np.ones(len(t), dtype=bool)
    if len(t) > 1:
        new_block[1:] = (~np.isfinite(dt[1:])) | (dt[1:] <= 0) | (dt[1:] > BLOCK_GAP_S)
    block_id = np.cumsum(new_block) - 1

    rows = []
    unique_blocks = np.unique(block_id)
    for bid in unique_blocks:
        mask = block_id == bid
        pos = np.flatnonzero(mask)
        if len(pos) < MIN_BLOCK_SAMPLES:
            continue
        a, b = int(pos[0]), int(pos[-1])
        ib, ub, sb = i[a:b + 1], u[a:b + 1], s5[a:b + 1]
        tb = t.iloc[a:b + 1].reset_index(drop=True)
        duration = float((tb.iloc[-1] - tb.iloc[0]).total_seconds())
        signed_ah = integrate_signed_ah(tb, ib)
        current_std = float(np.nanstd(ib))
        imin, imax = float(np.nanmin(ib)), float(np.nanmax(ib))
        pos_frac = float(np.mean(ib > 0.05))
        neg_frac = float(np.mean(ib < -0.05))
        looks_dynamic = bool(
            len(ib) >= 500
            and duration >= 300.0
            and imin < -5.0
            and imax > 5.0
            and current_std >= 3.0
        )
        gap_before_s = float(dt[a]) if a > 0 and np.isfinite(dt[a]) else np.nan
        rows.append({
            "cell": cell,
            "block_id": int(bid),
            "start_time": str(tb.iloc[0]),
            "end_time": str(tb.iloc[-1]),
            "n_samples": int(len(ib)),
            "duration_s": duration,
            "gap_before_s": gap_before_s,
            "signed_Ah_positive_discharge": signed_ah,
            "current_mean_A": float(np.nanmean(ib)),
            "current_std_A": current_std,
            "current_min_A": imin,
            "current_max_A": imax,
            "positive_current_fraction": pos_frac,
            "negative_current_fraction": neg_frac,
            "voltage_start_V": float(ub[0]),
            "voltage_end_V": float(ub[-1]),
            "voltage_min_V": float(np.nanmin(ub)),
            "voltage_max_V": float(np.nanmax(ub)),
            "S5_start_nm": float(sb[0]),
            "S5_end_nm": float(sb[-1]),
            "S5_delta_end_nm": float(sb[-1] - sb[0]),
            "looks_dynamic_WLTP": looks_dynamic,
        })

    blocks = pd.DataFrame(rows).sort_values("start_time").reset_index(drop=True)
    dynamic = blocks[blocks.looks_dynamic_WLTP].copy().reset_index(drop=True)
    if dynamic.empty:
        raise RuntimeError(f"no dynamic WLTP-like blocks detected for {cell}")

    soc_cursor = 1.0
    episode_rows = []
    prev_end = None
    for j, r in dynamic.iterrows():
        net_ah = float(r.signed_Ah_positive_discharge)
        soc_start = soc_cursor
        soc_end = soc_start - net_ah / q_ref
        gap_from_prev = np.nan
        if prev_end is not None:
            gap_from_prev = (pd.Timestamp(r.start_time) - prev_end).total_seconds()
        episode_rows.append({
            "cell": cell,
            "episode": int(j + 1),
            "source_block_id": int(r.block_id),
            "start_time": r.start_time,
            "end_time": r.end_time,
            "gap_from_previous_dynamic_s": gap_from_prev,
            "n_samples": int(r.n_samples),
            "duration_s": float(r.duration_s),
            "net_Ah_positive_discharge": net_ah,
            "SOC_start_from_Qref": soc_start,
            "SOC_end_from_Qref": soc_end,
            "voltage_start_V": float(r.voltage_start_V),
            "voltage_end_V": float(r.voltage_end_V),
            "voltage_min_V": float(r.voltage_min_V),
            "current_min_A": float(r.current_min_A),
            "current_max_A": float(r.current_max_A),
            "S5_start_nm": float(r.S5_start_nm),
            "S5_end_nm": float(r.S5_end_nm),
            "S5_delta_end_nm": float(r.S5_delta_end_nm),
            "net_discharge_is_positive": bool(net_ah > 0),
        })
        soc_cursor = soc_end
        prev_end = pd.Timestamp(r.end_time)

    episodes = pd.DataFrame(episode_rows)
    all_good_dt = dt[np.isfinite(dt) & (dt > 0) & (dt <= 30.0)]
    summary = {
        "cell": cell,
        "Q_ref_Ah": float(q_ref),
        "n_valid_samples": int(len(t)),
        "median_contiguous_dt_s": float(np.median(all_good_dt)),
        "p95_contiguous_dt_s": float(np.quantile(all_good_dt, 0.95)),
        "n_blocks_ge_50_samples": int(len(blocks)),
        "n_dynamic_WLTP_blocks": int(len(dynamic)),
        "all_dynamic_blocks_net_discharge_positive": bool((episodes.net_Ah_positive_discharge > 0).all()),
        "dynamic_total_net_discharge_Ah": float(episodes.net_Ah_positive_discharge.sum()),
        "dynamic_final_SOC_from_Qref": float(episodes.SOC_end_from_Qref.iloc[-1]),
        "first_dynamic_voltage_start_V": float(episodes.voltage_start_V.iloc[0]),
        "last_dynamic_voltage_end_V": float(episodes.voltage_end_V.iloc[-1]),
        "last_dynamic_voltage_min_V": float(episodes.voltage_min_V.iloc[-1]),
        "max_abs_dynamic_current_A": float(max(abs(episodes.current_min_A.min()), abs(episodes.current_max_A.max()))),
        "max_abs_dynamic_C_rate_from_Qref": float(max(abs(episodes.current_min_A.min()), abs(episodes.current_max_A.max())) / q_ref),
        "S5_valid_fraction": 1.0,
    }
    return summary, blocks, episodes


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

    summaries = []
    blocks_all = []
    episodes_all = []
    for c in CELLS:
        summary, blocks, episodes = scan_cell_blocks(df, c, float(qmap[c]))
        summaries.append(summary)
        blocks_all.append(blocks)
        episodes_all.append(episodes)

    summary_df = pd.DataFrame(summaries)
    blocks_df = pd.concat(blocks_all, ignore_index=True)
    episodes_df = pd.concat(episodes_all, ignore_index=True)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(args.out_dir / "wltp_cell_summary.csv", index=False)
    blocks_df.to_csv(args.out_dir / "wltp_all_contiguous_blocks.csv", index=False)
    episodes_df.to_csv(args.out_dir / "wltp_dynamic_episode_summary.csv", index=False)
    (args.out_dir / "wltp_audit_summary.json").write_text(
        json.dumps({"cells": summaries}, indent=2), encoding="utf-8"
    )
    print("=== WLTP cell summary ===")
    print(summary_df.to_string(index=False))
    print("=== all contiguous blocks ===")
    print(blocks_df.to_string(index=False))
    print("=== dynamic WLTP blocks with cumulative SOC ===")
    print(episodes_df.to_string(index=False))


if __name__ == "__main__":
    main()
