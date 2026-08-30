from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from run_sequence_representation_benchmark import PROFILES


@dataclass(frozen=True)
class ProtocolSplit:
    name: str
    split_id: str
    train: list[dict]
    validation: list[dict]
    calibration: list[dict]
    test: list[dict]


def _slice_source(src: dict, start: int, stop: int, suffix: str) -> dict:
    item = dict(src)
    item["name"] = f"{src['name']}::{suffix}::{start}:{stop}"
    item["x"] = src["x"][start:stop].copy()
    item["y"] = src["y"][start:stop].copy()
    item["parent_name"] = src["name"]
    item["segment_start"] = int(start)
    item["segment_stop"] = int(stop)
    return item


def _runs(labels: np.ndarray) -> list[tuple[int, int, str]]:
    out = []
    start = 0
    for i in range(1, len(labels) + 1):
        if i == len(labels) or labels[i] != labels[start]:
            out.append((start, i, str(labels[start])))
            start = i
    return out


def blocked_mixed_condition_split(
    sources: list[dict],
    *,
    window: int = 64,
    n_blocks: int = 16,
) -> ProtocolSplit:
    """T1 interpolation split with deterministic contiguous blocks and guard gaps.

    Every physical trajectory contributes to train/validation/calibration/test, but no
    random rows are used. Block identities repeat a fixed pattern over the trajectory.
    At every set boundary, `window-1` rows are removed from each neighboring side so
    causal windows from different sets cannot share samples or immediate context.

    This is intentionally an interpolation benchmark, not a generalization claim.
    """
    if n_blocks < 8:
        raise ValueError("n_blocks must be >= 8")
    guard = window - 1
    # 10 train, 2 validation, 2 calibration, 2 test blocks per 16-block trajectory.
    pattern = np.array(
        ["train", "train", "test", "train", "cal", "train", "train", "val",
         "train", "test", "train", "train", "cal", "train", "val", "train"],
        dtype=object,
    )
    if n_blocks != len(pattern):
        raise ValueError("Current frozen T1 protocol uses n_blocks=16")

    buckets = {"train": [], "val": [], "cal": [], "test": []}
    for src in sources:
        n = len(src["y"])
        edges = np.linspace(0, n, n_blocks + 1, dtype=int)
        labels = np.empty(n, dtype=object)
        for b in range(n_blocks):
            labels[edges[b]:edges[b + 1]] = pattern[b]

        for start, stop, label in _runs(labels):
            left_trim = guard if start > 0 and labels[start - 1] != label else 0
            right_trim = guard if stop < n and labels[stop] != label else 0
            a, b = start + left_trim, stop - right_trim
            # Each isolated segment must still support at least one causal window.
            if b - a < window:
                continue
            short = {"validation": "val", "calibration": "cal"}.get(label, label)
            buckets[short].append(_slice_source(src, a, b, label))

    for key, value in buckets.items():
        if not value:
            raise RuntimeError(f"T1 produced no {key} segments")
    return ProtocolSplit(
        name="T1_mixed_condition_blocked_interpolation",
        split_id="T1_block16_guard_window",
        train=buckets["train"],
        validation=buckets["val"],
        calibration=buckets["cal"],
        test=buckets["test"],
    )


def same_rate_unseen_profile_splits(sources: list[dict]) -> list[ProtocolSplit]:
    """T2: five complete profiles train, sixth complete profile test, separately by rate."""
    out = []
    for rate in ("1C", "2C"):
        for held_out in PROFILES:
            train = [s for s in sources if s["rate"] == rate and s["profile"] != held_out]
            test = [s for s in sources if s["rate"] == rate and s["profile"] == held_out]
            if len(train) != 5 or len(test) != 1:
                raise RuntimeError(f"Bad T2 split {rate}/{held_out}")
            out.append(
                ProtocolSplit(
                    name="T2_same_rate_unseen_profile",
                    split_id=f"T2_{rate}_{held_out}",
                    train=train,
                    validation=[],
                    calibration=[],
                    test=test,
                )
            )
    return out


def cross_rate_splits(sources: list[dict]) -> list[ProtocolSplit]:
    """T3: all six source-rate profiles train; all six same profile identities test at target rate."""
    out = []
    for train_rate, test_rate in (("1C", "2C"), ("2C", "1C")):
        train = [s for s in sources if s["rate"] == train_rate]
        test = [s for s in sources if s["rate"] == test_rate]
        if len(train) != 6 or len(test) != 6:
            raise RuntimeError(f"Bad T3 split {train_rate}->{test_rate}")
        out.append(
            ProtocolSplit(
                name="T3_cross_rate_seen_profiles",
                split_id=f"T3_{train_rate}_to_{test_rate}",
                train=train,
                validation=[],
                calibration=[],
                test=test,
            )
        )
    return out


def cross_rate_unseen_profile_splits(sources: list[dict]) -> list[ProtocolSplit]:
    """T4: strict profile+rate extrapolation; held-out identity absent at source rate as well."""
    out = []
    for train_rate, test_rate in (("1C", "2C"), ("2C", "1C")):
        for held_out in PROFILES:
            train = [
                s for s in sources
                if s["rate"] == train_rate and s["profile"] != held_out
            ]
            test = [
                s for s in sources
                if s["rate"] == test_rate and s["profile"] == held_out
            ]
            if len(train) != 5 or len(test) != 1:
                raise RuntimeError(f"Bad T4 split {train_rate}->{test_rate}/{held_out}")
            out.append(
                ProtocolSplit(
                    name="T4_cross_rate_unseen_profile",
                    split_id=f"T4_{train_rate}_to_{test_rate}_{held_out}",
                    train=train,
                    validation=[],
                    calibration=[],
                    test=test,
                )
            )
    return out


def assert_no_parent_overlap(split: ProtocolSplit, *, allow_t1_shared_parent: bool = True) -> None:
    """Sanity check parent-trajectory overlap according to protocol semantics."""
    def parents(items: Iterable[dict]) -> set[str]:
        return {str(x.get("parent_name", x["name"])) for x in items}

    if split.name.startswith("T1_") and allow_t1_shared_parent:
        # T1 deliberately shares parent trajectories, but uses disjoint guarded segments.
        return
    train_parents = parents(split.train)
    test_parents = parents(split.test)
    if train_parents & test_parents:
        raise AssertionError(
            f"Unexpected train/test parent overlap in {split.split_id}: "
            f"{sorted(train_parents & test_parents)}"
        )
