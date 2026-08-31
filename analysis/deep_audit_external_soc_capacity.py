from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import pandas as pd

CELLS = ("A1", "A2", "P1", "P2")
TIME_COL = "Time / yyyy-mm-ddTHH:MM:SS.FFF"
CURRENT_THRESHOLD_A = 0.05
MAX_CONTIGUOUS_DT_S = 30.0

# Protocol-defined nominal cell capacity and current levels from the external paper.
NOMINAL_CELL_CAPACITY_AH = 10.0
CALIBRATION_RATE_C = 0.04
CALIBRATION_CURRENT_A = NOMINAL_CELL_CAPACITY_AH * CALIBRATION_RATE_C  # 0.4 A
CALIBRATION_CURRENT_REL_TOL = 0.10
CALIBRATION_MIN_DURATION_S = 20.0 * 3600.0
CALIBRATION_MIN_CAPACITY_AH = 8.0

VALIDATION_RATE_TO_CURRENT_A = {0.2: 2.0, 0.5: 5.0, 1.0: 10.0}
VALIDATION_CURRENT_REL_TOL = 0.10
VALIDATION_MIN_CAPACITY_FRACTION = 0.65
VALIDATION_MIN_S5_FRACTION = 0.95
VALIDATION_MAX_VOLTAGE_MIN_V = 3.05
VALIDATION_MIN_START_VOLTAGE_V = 3.8


@dataclass
class Segment:
    dataset: str
    cell: str
    state: str
    segment_id: int
    start_time: str
    end_time: str
    n_samples: int = 0
    duration_s: float = 0.0
    integrated_ah: float = 0.0
    current_sum: float = 0.0
    current_abs_sum: float = 0.0
    current_count: int = 0
    voltage_start: float = np.nan
    voltage_end: float = np.nan
    voltage_min: float = np.nan
    voltage_max: float = np.nan
    s5_valid: int = 0
    s5_count: int = 0

    def update_scalar(self, time_str: str, current: float, voltage: float, s5: float) -> None:
        self.end_time = time_str
        self.n_samples += 1
        if np.isfinite(current):
            self.current_sum += float(current)
            self.current_abs_sum += abs(float(current))
            self.current_count += 1
        if np.isfinite(voltage):
            if not np.isfinite(self.voltage_start):
                self.voltage_start = float(voltage)
            self.voltage_end = float(voltage)
            self.voltage_min = float(voltage) if not np.isfinite(self.voltage_min) else min(self.voltage_min, float(voltage))
            self.voltage_max = float(voltage) if not np.isfinite(self.voltage_max) else max(self.voltage_max, float(voltage))
        self.s5_count += 1
        self.s5_valid += int(np.isfinite(s5))

    def to_row(self) -> dict:
        row = asdict(self)
        row["mean_current_A"] = self.current_sum / self.current_count if self.current_count else np.nan
        row["mean_abs_current_A"] = self.current_abs_sum / self.current_count if self.current_count else np.nan
        row["s5_valid_fraction"] = self.s5_valid / self.s5_count if self.s5_count else np.nan
        return row


def state_from_current(i: float) -> str:
    if not np.isfinite(i):
        return "missing"
    if i > CURRENT_THRESHOLD_A:
        return "discharge"
    if i < -CURRENT_THRESHOLD_A:
        return "charge"
    return "rest"


class CellTracker:
    def __init__(self, dataset: str, cell: str):
        self.dataset = dataset
        self.cell = cell
        self.current_segment: Segment | None = None
        self.segment_counter = 0
        self.rows: list[dict] = []
        self.prev_time: pd.Timestamp | None = None
        self.prev_i: float = np.nan
        self.prev_state: str | None = None
        self.dt_values: list[float] = []
        self.n_total = 0
        self.n_valid_current = 0

    def close_segment(self) -> None:
        if self.current_segment is not None:
            self.rows.append(self.current_segment.to_row())
            self.current_segment = None

    def process(self, ts: pd.Timestamp | pd.NaT, time_str: str, i: float, u: float, s5: float) -> None:
        self.n_total += 1
        if np.isfinite(i):
            self.n_valid_current += 1
        state = state_from_current(i)
        if state in ("missing", "rest"):
            self.close_segment()
            self.prev_time = ts if not pd.isna(ts) else None
            self.prev_i = i
            self.prev_state = state
            return

        dt = np.nan
        if self.prev_time is not None and not pd.isna(ts):
            dt = (ts - self.prev_time).total_seconds()
            if np.isfinite(dt) and 0 < dt <= MAX_CONTIGUOUS_DT_S:
                self.dt_values.append(float(dt))

        need_new = (
            self.current_segment is None
            or self.current_segment.state != state
            or not np.isfinite(dt)
            or dt <= 0
            or dt > MAX_CONTIGUOUS_DT_S
        )
        if need_new:
            self.close_segment()
            self.segment_counter += 1
            self.current_segment = Segment(
                dataset=self.dataset,
                cell=self.cell,
                state=state,
                segment_id=self.segment_counter,
                start_time=time_str,
                end_time=time_str,
            )

        assert self.current_segment is not None
        self.current_segment.update_scalar(time_str, i, u, s5)

        if (
            self.prev_state == state
            and np.isfinite(dt)
            and 0 < dt <= MAX_CONTIGUOUS_DT_S
            and np.isfinite(self.prev_i)
            and np.isfinite(i)
        ):
            self.current_segment.duration_s += float(dt)
            self.current_segment.integrated_ah += 0.5 * (abs(float(self.prev_i)) + abs(float(i))) * float(dt) / 3600.0

        self.prev_time = ts if not pd.isna(ts) else None
        self.prev_i = i
        self.prev_state = state

    def finish(self) -> tuple[list[dict], dict]:
        self.close_segment()
        meta = {
            "dataset": self.dataset,
            "cell": self.cell,
            "n_total_rows_seen": self.n_total,
            "n_valid_current": self.n_valid_current,
            "median_contiguous_dt_s": float(np.median(self.dt_values)) if self.dt_values else np.nan,
            "p95_contiguous_dt_s": float(np.quantile(self.dt_values, 0.95)) if self.dt_values else np.nan,
        }
        return self.rows, meta


def scan_file(path: Path, dataset: str, chunksize: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    usecols = [TIME_COL]
    for c in CELLS:
        usecols += [f"I_{c} / A", f"U_{c} / V", f"{c}S5 / nm"]

    trackers = {c: CellTracker(dataset, c) for c in CELLS}
    for chunk_i, chunk in enumerate(pd.read_csv(path, usecols=usecols, chunksize=chunksize, low_memory=False)):
        times_raw = chunk[TIME_COL].astype(str)
        times = pd.to_datetime(times_raw, errors="coerce")
        for c in CELLS:
            i_arr = pd.to_numeric(chunk[f"I_{c} / A"], errors="coerce").to_numpy(dtype=float)
            u_arr = pd.to_numeric(chunk[f"U_{c} / V"], errors="coerce").to_numpy(dtype=float)
            s_arr = pd.to_numeric(chunk[f"{c}S5 / nm"], errors="coerce").to_numpy(dtype=float)
            tracker = trackers[c]
            for j in range(len(chunk)):
                tracker.process(times.iloc[j], times_raw.iloc[j], i_arr[j], u_arr[j], s_arr[j])
        print(f"{dataset}: processed chunk {chunk_i + 1}, rows={len(chunk)}")

    segment_rows: list[dict] = []
    meta_rows: list[dict] = []
    for c in CELLS:
        rows, meta = trackers[c].finish()
        segment_rows.extend(rows)
        meta_rows.append(meta)
    return pd.DataFrame(segment_rows), pd.DataFrame(meta_rows)


def select_full_calibration_discharges(segments: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    dis = segments[(segments.dataset == "initial_calibration") & (segments.state == "discharge")].copy()
    dis["cal_current_rel_error"] = (dis.mean_abs_current_A - CALIBRATION_CURRENT_A).abs() / CALIBRATION_CURRENT_A
    dis["is_protocol_0p04C"] = (
        (dis.cal_current_rel_error <= CALIBRATION_CURRENT_REL_TOL)
        & (dis.duration_s >= CALIBRATION_MIN_DURATION_S)
        & (dis.integrated_ah >= CALIBRATION_MIN_CAPACITY_AH)
        & (dis.voltage_min <= VALIDATION_MAX_VOLTAGE_MIN_V)
    )
    selected = dis[dis.is_protocol_0p04C].copy()

    counts = selected.groupby("cell").size().to_dict()
    for c in CELLS:
        if counts.get(c, 0) != 3:
            candidates = dis[dis.cell == c][
                ["segment_id", "mean_abs_current_A", "duration_s", "integrated_ah", "voltage_min", "cal_current_rel_error", "is_protocol_0p04C"]
            ]
            raise RuntimeError(
                f"Expected exactly three protocol 0.04C discharges for {c}, found {counts.get(c, 0)}.\n"
                + candidates.to_string(index=False)
            )

    cap_rows = []
    for c in CELLS:
        s = selected[selected.cell == c].sort_values("start_time")
        capacities = s.integrated_ah.to_numpy(dtype=float)
        currents = s.mean_abs_current_A.to_numpy(dtype=float)
        cap_rows.append(
            {
                "cell": c,
                "n_reference_discharges": len(s),
                "mean_reference_current_A": float(np.mean(currents)),
                "Q_ref_Ah_mean": float(np.mean(capacities)),
                "Q_ref_Ah_std": float(np.std(capacities, ddof=1)),
                "Q_ref_Ah_min": float(np.min(capacities)),
                "Q_ref_Ah_max": float(np.max(capacities)),
                "relative_range": float((np.max(capacities) - np.min(capacities)) / np.mean(capacities)),
                "source": "exactly three long ~0.4 A discharge segments from Initial_referenceless_strain_calibration.csv",
                "paper_protocol": "three 0.04C cycles at 25C; SOC referenced to initial discharge capacity",
            }
        )
    return selected, pd.DataFrame(cap_rows)


def assign_validation_rate(mean_abs_current_A: float) -> tuple[float, float, float]:
    if not np.isfinite(mean_abs_current_A):
        return np.nan, np.nan, np.nan
    rates = np.array(list(VALIDATION_RATE_TO_CURRENT_A.keys()), dtype=float)
    currents = np.array(list(VALIDATION_RATE_TO_CURRENT_A.values()), dtype=float)
    idx = int(np.argmin(np.abs(currents - mean_abs_current_A)))
    assigned_rate = float(rates[idx])
    expected_current = float(currents[idx])
    rel_error = abs(float(mean_abs_current_A) - expected_current) / expected_current
    if rel_error > VALIDATION_CURRENT_REL_TOL:
        return np.nan, expected_current, rel_error
    return assigned_rate, expected_current, rel_error


def annotate_validation(segments: pd.DataFrame, capacities: pd.DataFrame) -> pd.DataFrame:
    val = segments[(segments.dataset == "constant_current_validation") & (segments.state == "discharge")].copy()
    qmap = capacities.set_index("cell").Q_ref_Ah_mean.to_dict()
    val["Q_ref_Ah"] = val.cell.map(qmap)

    assigned = val.mean_abs_current_A.map(assign_validation_rate)
    val["assigned_rate_C"] = assigned.map(lambda x: x[0])
    val["expected_current_A"] = assigned.map(lambda x: x[1])
    val["current_rel_error"] = assigned.map(lambda x: x[2])
    val["capacity_fraction_of_Qref"] = val.integrated_ah / val.Q_ref_Ah
    val["SOC_end_from_Qref"] = 1.0 - val.capacity_fraction_of_Qref

    val["eligible_full_validation_discharge"] = (
        val.assigned_rate_C.notna()
        & (val.current_rel_error <= VALIDATION_CURRENT_REL_TOL)
        & (val.capacity_fraction_of_Qref >= VALIDATION_MIN_CAPACITY_FRACTION)
        & (val.s5_valid_fraction >= VALIDATION_MIN_S5_FRACTION)
        & (val.voltage_min <= VALIDATION_MAX_VOLTAGE_MIN_V)
        & (val.voltage_start >= VALIDATION_MIN_START_VOLTAGE_V)
    )
    return val


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--initial-calibration", type=Path, required=True)
    p.add_argument("--constant-current-validation", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--chunksize", type=int, default=200000)
    args = p.parse_args()

    cal_seg, cal_meta = scan_file(args.initial_calibration, "initial_calibration", args.chunksize)
    val_seg, val_meta = scan_file(args.constant_current_validation, "constant_current_validation", args.chunksize)
    all_seg = pd.concat([cal_seg, val_seg], ignore_index=True)

    selected, capacities = select_full_calibration_discharges(all_seg)
    val_dis = annotate_validation(all_seg, capacities)

    eligible = val_dis[val_dis.eligible_full_validation_discharge].copy()
    coverage = (
        eligible.groupby(["cell", "assigned_rate_C"], as_index=False)
        .agg(
            n_discharges=("segment_id", "size"),
            mean_observed_current_A=("mean_abs_current_A", "mean"),
            mean_current_rel_error=("current_rel_error", "mean"),
            mean_capacity_fraction_of_Qref=("capacity_fraction_of_Qref", "mean"),
            min_capacity_fraction_of_Qref=("capacity_fraction_of_Qref", "min"),
            mean_end_SOC=("SOC_end_from_Qref", "mean"),
            mean_S5_valid_fraction=("s5_valid_fraction", "mean"),
            mean_voltage_min_V=("voltage_min", "mean"),
        )
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    all_seg.to_csv(args.out_dir / "all_active_segments.csv", index=False)
    pd.concat([cal_meta, val_meta], ignore_index=True).to_csv(args.out_dir / "sampling_meta.csv", index=False)
    selected.to_csv(args.out_dir / "selected_initial_reference_discharges.csv", index=False)
    capacities.to_csv(args.out_dir / "initial_reference_capacities.csv", index=False)
    val_dis.to_csv(args.out_dir / "constant_current_discharge_audit.csv", index=False)
    coverage.to_csv(args.out_dir / "external_E1_E2_coverage.csv", index=False)

    summary = {
        "current_sign": "positive=discharge",
        "current_threshold_A": CURRENT_THRESHOLD_A,
        "max_contiguous_dt_s": MAX_CONTIGUOUS_DT_S,
        "nominal_cell_capacity_Ah": NOMINAL_CELL_CAPACITY_AH,
        "calibration_protocol": {
            "rate_C": CALIBRATION_RATE_C,
            "expected_current_A": CALIBRATION_CURRENT_A,
            "current_relative_tolerance": CALIBRATION_CURRENT_REL_TOL,
            "minimum_duration_s": CALIBRATION_MIN_DURATION_S,
            "minimum_capacity_Ah": CALIBRATION_MIN_CAPACITY_AH,
        },
        "validation_protocol": {
            "rate_to_expected_current_A": {str(k): v for k, v in VALIDATION_RATE_TO_CURRENT_A.items()},
            "current_relative_tolerance": VALIDATION_CURRENT_REL_TOL,
            "minimum_capacity_fraction_of_Qref": VALIDATION_MIN_CAPACITY_FRACTION,
            "minimum_S5_valid_fraction": VALIDATION_MIN_S5_FRACTION,
            "maximum_voltage_min_V": VALIDATION_MAX_VOLTAGE_MIN_V,
        },
        "capacity_reference": "mean of exactly three long ~0.4 A initial 0.04C discharge capacities per cell",
        "reference_capacities_Ah": capacities.set_index("cell").Q_ref_Ah_mean.to_dict(),
        "eligible_validation_discharges": int(len(eligible)),
        "coverage_rows": coverage.to_dict("records"),
        "label_definition": "SOC(t)=1-cumulative positive-current discharge Ah / Q_ref for each eligible full validation discharge starting fully charged",
    }
    (args.out_dir / "external_soc_reconstruction_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("=== selected 0.04C reference segments ===")
    print(selected[["cell", "segment_id", "mean_abs_current_A", "duration_s", "integrated_ah", "voltage_min", "cal_current_rel_error"]].to_string(index=False))
    print("=== reference capacities ===")
    print(capacities.to_string(index=False))
    print("=== eligible constant-current coverage ===")
    print(coverage.to_string(index=False))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
