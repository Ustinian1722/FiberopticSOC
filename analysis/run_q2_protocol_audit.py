from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from q2_protocols import (
    assert_no_parent_overlap,
    blocked_mixed_condition_split,
    cross_rate_splits,
    cross_rate_unseen_profile_splits,
    same_rate_unseen_profile_splits,
)
from run_sequence_representation_benchmark import WindowDataset, load_sources


def summary_row(split, part: str, items: list[dict], window: int) -> dict:
    n_rows = sum(len(x["y"]) for x in items)
    n_windows = len(WindowDataset(items, window, 1)) if items else 0
    ys = [x["y"] for x in items]
    return {
        "protocol": split.name,
        "split_id": split.split_id,
        "part": part,
        "n_segments_or_trajectories": len(items),
        "n_rows": n_rows,
        "n_windows_stride1": n_windows,
        "soc_min": min(float(y.min()) for y in ys) if ys else float("nan"),
        "soc_max": max(float(y.max()) for y in ys) if ys else float("nan"),
        "profiles": ",".join(sorted({str(x["profile"]) for x in items})),
        "rates": ",".join(sorted({str(x["rate"]) for x in items})),
    }


def assert_t1_segments_disjoint(split) -> None:
    by_parent: dict[str, list[tuple[int, int, str]]] = {}
    for part in ("train", "validation", "calibration", "test"):
        for x in getattr(split, part):
            parent = str(x["parent_name"])
            by_parent.setdefault(parent, []).append(
                (int(x["segment_start"]), int(x["segment_stop"]), part)
            )
    for parent, ranges in by_parent.items():
        ranges = sorted(ranges)
        for i in range(len(ranges)):
            a0, a1, ap = ranges[i]
            for j in range(i + 1, len(ranges)):
                b0, b1, bp = ranges[j]
                if ap == bp:
                    continue
                if max(a0, b0) < min(a1, b1):
                    raise AssertionError(
                        f"T1 overlap in {parent}: {ap}[{a0},{a1}) vs {bp}[{b0},{b1})"
                    )


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=Path, default=Path("data/extracted/SiC-18"))
    p.add_argument("--out-dir", type=Path, default=Path("results/q2_protocol_audit"))
    p.add_argument("--window", type=int, default=64)
    args = p.parse_args()

    sources = load_sources(args.data)
    t1 = blocked_mixed_condition_split(sources, window=args.window)
    assert_t1_segments_disjoint(t1)

    splits = [t1]
    splits += same_rate_unseen_profile_splits(sources)
    splits += cross_rate_splits(sources)
    splits += cross_rate_unseen_profile_splits(sources)

    rows = []
    for split in splits:
        assert_no_parent_overlap(split)
        for part in ("train", "validation", "calibration", "test"):
            rows.append(summary_row(split, part, getattr(split, part), args.window))

    df = pd.DataFrame(rows)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out_dir / "protocol_counts.csv", index=False)

    # Frozen cardinality checks.
    assert len(same_rate_unseen_profile_splits(sources)) == 12
    assert len(cross_rate_splits(sources)) == 2
    assert len(cross_rate_unseen_profile_splits(sources)) == 12
    assert all(x > 0 for x in df.loc[df["part"].isin(["train", "test"]), "n_windows_stride1"])

    print("=== T1 ===")
    print(df[df["split_id"] == t1.split_id].to_string(index=False))
    print("\nT2 splits=12, T3 splits=2, T4 splits=12")
    print("Protocol audit passed: no forbidden parent overlap outside T1 and no T1 row overlap across sets.")


if __name__ == "__main__":
    main()
