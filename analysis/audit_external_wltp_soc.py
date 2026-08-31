from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

CELLS = ("P1", "P2")
TIME_COL = "Time / yyyy-mm-ddTHH:MM:SS.FFF"
CURRENT_THRESHOLD_A = 0.05
LONG_REST_S = 3600.0


def summarize_cell(df: pd.DataFrame, cell: str, q_ref: float) -> tuple[dict, pd.DataFrame]:
    i = pd.to_numeric(df[f"I_{cell} / A"], errors="coerce").to_numpy(float)
    u = pd.to_numeric(df[f"U_{cell} / V"], errors="coerce").to_numpy(float)
    s5 = pd.to_numeric(df[f"{cell}S5 / nm"], errors="coerce").to_numpy(float)
    t = pd.to_datetime(df[TIME_COL], errors="coerce")
    valid = np.isfinite(i) & np.isfinite(u) & np.isfinite(s5) & t.notna().to_numpy()
    idx = np.flatnonzero(valid)
    if len(idx) < 2:
        raise RuntimeError(f"insufficient valid WLTP data for {cell}")

    i2, u2, s2 = i[idx], u[idx], s5[idx]
    t2 = t.iloc[idx].reset_index(drop=True)

    active = np.abs(i2) > CURRENT_THRESHOLD_A
    active_idx = np.flatnonzero(active)
    if len(active_idx) == 0:
        raise RuntimeError(f"no active WLTP current for {cell}")
    first, last = int(active_idx[0]), int(active_idx[-1])

    # Work only on the continuous experiment envelope from first to last active sample.
    i3 = i2[first:last+1]
    u3 = u2[first:last+1]
    s3 = s2[first:last+1]
    t3 = t2.iloc[first:last+1].reset_index(drop=True)
    dt3 = t3.diff().dt.total_seconds().to_numpy(float)

    d_ah = np.zeros(len(i3), dtype=float)
    for k in range(1, len(i3)):
        dtk = dt3[k]
        if np.isfinite(dtk) and 0 < dtk <= 30.0 and np.isfinite(i3[k-1]) and np.isfinite(i3[k]):
            # Positive current means discharge; negative current is regenerative charge.
            d_ah[k] = 0.5 * (i3[k-1] + i3[k]) * dtk / 3600.0
    cum_net_ah = np.cumsum(d_ah)
    soc = 1.0 - cum_net_ah / q_ref

    # Find long rests using contiguous near-zero-current spans inside the experiment envelope.
    rest = np.abs(i3) <= CURRENT_THRESHOLD_A
    change = np.r_[True, rest[1:] != rest[:-1]]
    starts = np.flatnonzero(change)
    ends = np.r_[starts[1:] - 1, len(rest) - 1]
    long_rest_rows = []
    for a, b in zip(starts, ends):
        if not rest[a]:
            continue
        dur = (t3.iloc[b] - t3.iloc[a]).total_seconds()
        if dur >= LONG_REST_S:
            long_rest_rows.append({
                "cell": cell,
                "rest_start": str(t3.iloc[a]),
                "rest_end": str(t3.iloc[b]),
                "duration_s": float(dur),
                "soc_start": float(soc[a]),
                "soc_end": float(soc[b]),
            })

    # Split repeated-WLTP episodes at long rests for descriptive coverage only.
    split_points = []
    for r in long_rest_rows:
        rs = pd.Timestamp(r["rest_start"])
        re = pd.Timestamp(r["rest_end"])
        a = int(np.searchsorted(t3.to_numpy(), np.datetime64(rs), side="left"))
        b = int(np.searchsorted(t3.to_numpy(), np.datetime64(re), side="right")) - 1
        split_points.append((a, b))

    episode_ranges = []
    cur = 0
    for a, b in split_points:
        if a > cur:
            episode_ranges.append((cur, a - 1))
        cur = b + 1
    if cur < len(i3):
        episode_ranges.append((cur, len(i3) - 1))

    episodes = []
    for j, (a, b) in enumerate(episode_ranges, 1):
        act = np.abs(i3[a:b+1]) > CURRENT_THRESHOLD_A
        if act.sum() < 100:
            continue
        aa = a + int(np.flatnonzero(act)[0])
        bb = a + int(np.flatnonzero(act)[-1])
        episodes.append({
            "cell": cell,
            "episode": j,
            "start_time": str(t3.iloc[aa]),
            "end_time": str(t3.iloc[bb]),
            "n_samples": int(bb-aa+1),
            "duration_s": float((t3.iloc[bb]-t3.iloc[aa]).total_seconds()),
            "soc_start": float(soc[aa]),
            "soc_end": float(soc[bb]),
            "net_Ah": float(cum_net_ah[bb] - cum_net_ah[aa]),
            "voltage_min_V": float(np.nanmin(u3[aa:bb+1])),
            "current_min_A": float(np.nanmin(i3[aa:bb+1])),
            "current_max_A": float(np.nanmax(i3[aa:bb+1])),
            "S5_rel_min_nm": float(np.nanmin(s3[aa:bb+1] - s3[aa])),
            "S5_rel_max_nm": float(np.nanmax(s3[aa:bb+1] - s3[aa])),
        })

    finite_dt3 = dt3[np.isfinite(dt3) & (dt3 > 0) & (dt3 <= 30.0)]
    summary = {
        "cell": cell,
        "Q_ref_Ah": float(q_ref),
        "experiment_start": str(t3.iloc[0]),
        "experiment_end": str(t3.iloc[-1]),
        "n_samples": int(len(i3)),
        "median_dt_s": float(np.median(finite_dt3)),
        "p95_dt_s": float(np.quantile(finite_dt3, 0.95)),
        "S5_valid_fraction": float(np.mean(np.isfinite(s3))),
        "current_min_A": float(np.nanmin(i3)),
        "current_max_A": float(np.nanmax(i3)),
        "max_abs_C_rate_from_Qref": float(np.nanmax(np.abs(i3)) / q_ref),
        "voltage_start_V": float(u3[0]),
        "voltage_end_V": float(u3[-1]),
        "voltage_min_V": float(np.nanmin(u3)),
        "net_discharge_Ah": float(cum_net_ah[-1]),
        "SOC_end_from_Qref": float(soc[-1]),
        "SOC_min": float(np.nanmin(soc)),
        "SOC_max": float(np.nanmax(soc)),
        "n_long_rests_ge_1h": int(len(long_rest_rows)),
        "n_active_episodes_between_long_rests": int(len(episodes)),
    }
    return summary, pd.DataFrame(episodes), pd.DataFrame(long_rest_rows)


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
    episodes_all = []
    rests_all = []
    for c in CELLS:
        summary, episodes, rests = summarize_cell(df, c, float(qmap[c]))
        summaries.append(summary)
        episodes_all.append(episodes)
        rests_all.append(rests)

    summary_df = pd.DataFrame(summaries)
    episode_df = pd.concat(episodes_all, ignore_index=True) if episodes_all else pd.DataFrame()
    rest_df = pd.concat(rests_all, ignore_index=True) if rests_all else pd.DataFrame()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(args.out_dir / "wltp_cell_summary.csv", index=False)
    episode_df.to_csv(args.out_dir / "wltp_episode_summary.csv", index=False)
    rest_df.to_csv(args.out_dir / "wltp_long_rests.csv", index=False)
    (args.out_dir / "wltp_audit_summary.json").write_text(
        json.dumps({"cells": summaries}, indent=2), encoding="utf-8"
    )
    print("=== WLTP cell summary ===")
    print(summary_df.to_string(index=False))
    print("=== WLTP episodes ===")
    print(episode_df.to_string(index=False))
    print("=== long rests ===")
    print(rest_df.to_string(index=False))


if __name__ == "__main__":
    main()
