from __future__ import annotations

import argparse
import json
import math
import shutil
import tarfile
import zipfile
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

CANONICAL = [
    "time",
    "current",
    "voltage",
    "dis_cap",
    "wavelength_1",
    "wavelength_2",
    "temperature",
    "force",
    "soc",
]

ALIASES = {
    "time": {"time", "times", "timesec", "times", "t"},
    "current": {"current", "currenta", "i", "amp", "amps"},
    "voltage": {"voltage", "voltagev", "v", "volt"},
    "dis_cap": {"discap", "dischargecapacity", "capacity", "discapacity", "dischargecap"},
    "wavelength_1": {"wavelength1", "wave1", "lambda1", "wavelength01"},
    "wavelength_2": {"wavelength2", "wave2", "lambda2", "wavelength02"},
    "temperature": {"temperature", "temp", "temperaturec", "tempc"},
    "force": {"force", "forcen", "stress", "mechanicalforce"},
    "soc": {"soc", "stateofcharge", "statecharge"},
}


def norm_col(name: object) -> str:
    return "".join(ch.lower() for ch in str(name) if ch.isalnum())


def canonical_mapping(columns: Iterable[object]) -> dict[object, str]:
    mapping: dict[object, str] = {}
    used: set[str] = set()
    for col in columns:
        token = norm_col(col)
        for target, aliases in ALIASES.items():
            if target not in used and token in aliases:
                mapping[col] = target
                used.add(target)
                break
    return mapping


def safe_extract_archives(raw_dir: Path, extracted_dir: Path) -> list[Path]:
    extracted_dir.mkdir(parents=True, exist_ok=True)
    extracted_roots: list[Path] = []
    for path in raw_dir.rglob("*"):
        if not path.is_file() or path.name == "manifest.json":
            continue
        suffix = path.suffix.lower()
        out = extracted_dir / path.stem
        if suffix == ".zip":
            out.mkdir(parents=True, exist_ok=True)
            with zipfile.ZipFile(path) as zf:
                zf.extractall(out)
            extracted_roots.append(out)
        elif suffix in {".tar", ".gz", ".tgz", ".bz2", ".xz"}:
            try:
                out.mkdir(parents=True, exist_ok=True)
                with tarfile.open(path, "r:*") as tf:
                    tf.extractall(out)
                extracted_roots.append(out)
            except tarfile.ReadError:
                pass
    return extracted_roots


def read_csv_flexible(path: Path) -> pd.DataFrame:
    errors = []
    for encoding in ("utf-8-sig", "utf-8", "gb18030", "latin1"):
        try:
            return pd.read_csv(path, sep=None, engine="python", encoding=encoding)
        except Exception as exc:
            errors.append(f"{encoding}: {exc}")
    raise RuntimeError("; ".join(errors[-2:]))


def load_tables(search_roots: list[Path]) -> tuple[list[tuple[str, pd.DataFrame]], list[dict[str, str]]]:
    tables: list[tuple[str, pd.DataFrame]] = []
    failures: list[dict[str, str]] = []
    seen: set[Path] = set()
    candidates: list[Path] = []
    for root in search_roots:
        for path in root.rglob("*"):
            if path.is_file() and path not in seen:
                seen.add(path)
                candidates.append(path)

    for path in sorted(candidates):
        suffix = path.suffix.lower()
        try:
            if suffix in {".csv", ".txt", ".tsv"}:
                df = read_csv_flexible(path)
                tables.append((str(path), df))
            elif suffix in {".xlsx", ".xlsm"}:
                sheets = pd.read_excel(path, sheet_name=None, engine="openpyxl")
                for sheet, df in sheets.items():
                    tables.append((f"{path}::{sheet}", df))
            elif suffix == ".xls":
                sheets = pd.read_excel(path, sheet_name=None)
                for sheet, df in sheets.items():
                    tables.append((f"{path}::{sheet}", df))
        except Exception as exc:
            failures.append({"source": str(path), "error": repr(exc)})
    return tables, failures


def standardize(name: str, df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, object]]:
    df = df.copy()
    df = df.dropna(axis=1, how="all").dropna(axis=0, how="all")
    mapping = canonical_mapping(df.columns)
    df = df.rename(columns=mapping)
    for col in CANONICAL:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    meta = {
        "source": name,
        "rows": int(len(df)),
        "columns": int(df.shape[1]),
        "original_columns": [str(c) for c in mapping.keys()],
        "mapped_columns": {str(k): v for k, v in mapping.items()},
        "canonical_present": [c for c in CANONICAL if c in df.columns],
        "duplicate_rows": int(df.duplicated().sum()) if len(df) else 0,
    }
    return df, meta


def numeric_profile(source: str, df: pd.DataFrame) -> list[dict[str, object]]:
    rows = []
    for col in CANONICAL:
        if col not in df.columns:
            continue
        s = pd.to_numeric(df[col], errors="coerce")
        valid = s.dropna()
        rows.append({
            "source": source,
            "variable": col,
            "n": int(valid.size),
            "missing_fraction": float(s.isna().mean()) if len(s) else math.nan,
            "min": float(valid.min()) if len(valid) else math.nan,
            "mean": float(valid.mean()) if len(valid) else math.nan,
            "std": float(valid.std()) if len(valid) > 1 else math.nan,
            "median": float(valid.median()) if len(valid) else math.nan,
            "max": float(valid.max()) if len(valid) else math.nan,
        })
    return rows


def soc_correlations(source: str, df: pd.DataFrame) -> list[dict[str, object]]:
    if "soc" not in df.columns:
        return []
    out = []
    for col in CANONICAL:
        if col == "soc" or col not in df.columns:
            continue
        pair = df[[col, "soc"]].dropna()
        if len(pair) < 3 or pair[col].nunique() < 2 or pair["soc"].nunique() < 2:
            continue
        out.append({
            "source": source,
            "feature": col,
            "n": int(len(pair)),
            "pearson_r": float(pair[col].corr(pair["soc"], method="pearson")),
            "spearman_rho": float(pair[col].corr(pair["soc"], method="spearman")),
        })
    return out


def sample_frame(df: pd.DataFrame, max_rows: int = 150_000) -> pd.DataFrame:
    if len(df) <= max_rows:
        return df.copy()
    return df.sample(max_rows, random_state=42).sort_index()


def save_trace_plot(source: str, df: pd.DataFrame, out: Path) -> None:
    x_col = "time" if "time" in df.columns else None
    x = df[x_col] if x_col else np.arange(len(df))
    channels = [c for c in ["soc", "voltage", "current", "wavelength_1", "wavelength_2", "temperature", "force"] if c in df.columns]
    if not channels:
        return
    fig, axes = plt.subplots(len(channels), 1, figsize=(11, max(2.2 * len(channels), 4)), sharex=True)
    if len(channels) == 1:
        axes = [axes]
    stride = max(1, len(df) // 25_000)
    for ax, col in zip(axes, channels):
        ax.plot(np.asarray(x)[::stride], df[col].to_numpy()[::stride], linewidth=0.8)
        ax.set_ylabel(col)
        ax.grid(alpha=0.2)
    axes[-1].set_xlabel(x_col or "sample index")
    fig.suptitle(f"Representative trace: {Path(source.split('::')[0]).name}")
    fig.tight_layout()
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_corr_heatmap(df: pd.DataFrame, out: Path) -> None:
    cols = [c for c in CANONICAL if c in df.columns and df[c].nunique(dropna=True) > 1]
    if len(cols) < 2:
        return
    corr = df[cols].corr(method="spearman")
    fig, ax = plt.subplots(figsize=(9, 7))
    im = ax.imshow(corr.to_numpy(), vmin=-1, vmax=1, cmap="coolwarm")
    ax.set_xticks(range(len(cols)), cols, rotation=45, ha="right")
    ax.set_yticks(range(len(cols)), cols)
    for i in range(len(cols)):
        for j in range(len(cols)):
            value = corr.iloc[i, j]
            if np.isfinite(value):
                ax.text(j, i, f"{value:.2f}", ha="center", va="center", fontsize=7)
    fig.colorbar(im, ax=ax, label="Spearman correlation")
    ax.set_title("Pooled channel correlation")
    fig.tight_layout()
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_soc_optical_scatter(df: pd.DataFrame, out: Path) -> None:
    if "soc" not in df.columns:
        return
    optical = [c for c in ["wavelength_1", "wavelength_2", "force", "temperature"] if c in df.columns]
    if not optical:
        return
    fig, axes = plt.subplots(len(optical), 1, figsize=(8, 3 * len(optical)), sharex=True)
    if len(optical) == 1:
        axes = [axes]
    sampled = sample_frame(df, 60_000)
    for ax, col in zip(axes, optical):
        pair = sampled[["soc", col]].dropna()
        ax.scatter(pair["soc"], pair[col], s=4, alpha=0.18)
        ax.set_ylabel(col)
        ax.grid(alpha=0.2)
    axes[-1].set_xlabel("SOC")
    fig.suptitle("Optical / thermo-mechanical channels versus SOC")
    fig.tight_layout()
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_soc_binned_profiles(df: pd.DataFrame, out: Path) -> pd.DataFrame | None:
    if "soc" not in df.columns:
        return None
    features = [c for c in ["voltage", "wavelength_1", "wavelength_2", "temperature", "force"] if c in df.columns]
    if not features:
        return None
    work = df[["soc"] + features].dropna(subset=["soc"]).copy()
    if work.empty:
        return None
    soc = work["soc"]
    if soc.max() > 1.5:
        soc = soc / 100.0
    work["soc_fraction"] = soc.clip(0, 1)
    work["soc_bin"] = pd.cut(work["soc_fraction"], bins=np.linspace(0, 1, 21), include_lowest=True)
    grouped = work.groupby("soc_bin", observed=True)[features + ["soc_fraction"]].mean(numeric_only=True).reset_index(drop=True)
    if grouped.empty:
        return None
    fig, ax = plt.subplots(figsize=(9, 5))
    for col in features:
        s = grouped[col]
        sd = s.std()
        z = (s - s.mean()) / sd if np.isfinite(sd) and sd > 0 else s * 0
        ax.plot(grouped["soc_fraction"], z, marker="o", linewidth=1.1, label=col)
    ax.set_xlabel("SOC")
    ax.set_ylabel("Mean channel response (z-score)")
    ax.set_title("SOC-binned channel profiles")
    ax.grid(alpha=0.2)
    ax.legend(ncol=2, fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return grouped


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", default="data/raw")
    parser.add_argument("--extracted", default="data/extracted")
    parser.add_argument("--out", default="results")
    args = parser.parse_args()

    raw_dir = Path(args.raw)
    extracted_dir = Path(args.extracted)
    out_dir = Path(args.out)
    tables_dir = out_dir / "tables"
    figs_dir = out_dir / "figures"
    tables_dir.mkdir(parents=True, exist_ok=True)
    figs_dir.mkdir(parents=True, exist_ok=True)

    if extracted_dir.exists():
        shutil.rmtree(extracted_dir)
    safe_extract_archives(raw_dir, extracted_dir)

    tables, failures = load_tables([raw_dir, extracted_dir])
    if not tables:
        raise RuntimeError("No CSV/XLSX/XLS data tables were discovered after download/extraction")

    metadata = []
    profiles = []
    corrs = []
    standardized: list[tuple[str, pd.DataFrame]] = []
    pooled_samples = []

    for source, raw_df in tables:
        df, meta = standardize(source, raw_df)
        metadata.append(meta)
        standardized.append((source, df))
        profiles.extend(numeric_profile(source, df))
        corrs.extend(soc_correlations(source, df))
        keep = [c for c in CANONICAL if c in df.columns]
        if keep:
            sampled = sample_frame(df[keep], 150_000)
            sampled["__source"] = source
            pooled_samples.append(sampled)

    meta_df = pd.DataFrame(metadata).sort_values("rows", ascending=False)
    profile_df = pd.DataFrame(profiles)
    corr_df = pd.DataFrame(corrs)
    meta_df.to_csv(tables_dir / "table_inventory.csv", index=False)
    profile_df.to_csv(tables_dir / "numeric_profile.csv", index=False)
    corr_df.to_csv(tables_dir / "soc_feature_correlations_by_table.csv", index=False)
    pd.DataFrame(failures).to_csv(tables_dir / "load_failures.csv", index=False)

    pooled = pd.concat(pooled_samples, ignore_index=True, sort=False) if pooled_samples else pd.DataFrame()
    pooled_numeric = pooled[[c for c in CANONICAL if c in pooled.columns]] if not pooled.empty else pooled
    pooled_corrs = soc_correlations("POOLED_SAMPLED", pooled_numeric) if not pooled.empty else []
    pooled_corr_df = pd.DataFrame(pooled_corrs)
    if not pooled_corr_df.empty:
        pooled_corr_df["abs_spearman"] = pooled_corr_df["spearman_rho"].abs()
        pooled_corr_df = pooled_corr_df.sort_values("abs_spearman", ascending=False)
    pooled_corr_df.to_csv(tables_dir / "soc_feature_correlations_pooled.csv", index=False)

    # Representative trace comes from the largest table with at least one canonical channel.
    source0, df0 = max(standardized, key=lambda item: len(item[1]))
    save_trace_plot(source0, df0, figs_dir / "representative_multichannel_trace.png")
    if not pooled_numeric.empty:
        save_corr_heatmap(pooled_numeric, figs_dir / "channel_correlation_heatmap.png")
        save_soc_optical_scatter(pooled_numeric, figs_dir / "soc_vs_optical_channels.png")
        binned = save_soc_binned_profiles(pooled_numeric, figs_dir / "soc_binned_channel_profiles.png")
        if binned is not None:
            binned.to_csv(tables_dir / "soc_binned_channel_profiles.csv", index=False)

    canonical_coverage = {c: int(sum(c in df.columns for _, df in standardized)) for c in CANONICAL}
    total_rows = int(sum(len(df) for _, df in standardized))
    summary = {
        "table_count": len(standardized),
        "total_rows_across_tables": total_rows,
        "pooled_sample_rows_for_global_diagnostics": int(len(pooled_numeric)),
        "canonical_coverage_table_count": canonical_coverage,
        "load_failures": failures,
        "representative_table": source0,
    }
    (out_dir / "analysis_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    top_corr_md = "No usable SOC correlations were computed."
    if not pooled_corr_df.empty:
        display = pooled_corr_df[["feature", "n", "pearson_r", "spearman_rho"]].head(12).copy()
        top_corr_md = display.to_markdown(index=False, floatfmt=".4f")

    coverage_lines = "\n".join(f"- `{k}`: present in {v}/{len(standardized)} tables" for k, v in canonical_coverage.items())
    failure_text = "None" if not failures else "\n".join(f"- {f['source']}: {f['error']}" for f in failures[:10])

    report = f"""# FiberopticSOC initial data audit

## Dataset ingestion

- Tables discovered: **{len(standardized)}**
- Total table rows: **{total_rows:,}**
- Rows used in pooled diagnostics: **{len(pooled_numeric):,}** (deterministic capped sample for memory-safe global plots)
- Representative largest table: `{source0}`

## Canonical channel coverage

{coverage_lines}

## Pooled feature association with SOC

{top_corr_md}

## Research-integrity / leakage audit

1. **Do not use `dis_cap` as an SOC-model input for the primary benchmark.** The source dataset documentation states that SOC is calculated from discharge capacity by charge integration, so `dis_cap` is a direct target-construction variable and can make an SOC model look artificially accurate.
2. Treat **time** cautiously. Under a repeated or fixed-duration discharge protocol, time can become a proxy for SOC. It is useful for diagnostics but should be excluded or explicitly ablated in the main causal/online SOC input set.
3. The scientifically interesting comparison is therefore electrical-only (`V`, `I`, optional measured temperature) versus **raw FBG wavelengths** versus **decoupled thermo-mechanical channels** (`temperature`, `force`), with identical train/test splits.
4. Raw wavelength channels and their decoupled temperature/force outputs are not independent information sources. Report ablations separately rather than combining all four and claiming four independent sensing modalities.
5. Any future train/test split should be based on condition / run / battery identity where available, not random row splitting, to avoid temporal-neighbor leakage.

## Immediate modeling experiments suggested by this audit

- E0: `V + I`
- E1: `V + I + measured/decoupled temperature`
- E2: `V + I + force`
- E3: `V + I + wavelength_1 + wavelength_2`
- E4: `V + I + temperature + force`
- E5: raw-FBG versus decoupled-FBG comparison under the same split
- Robustness: sensor noise, wavelength drift/offset, channel dropout, and cross-condition generalization

## Load failures

{failure_text}

Generated tables are under `results/tables/`; figures are under `results/figures/`.
"""
    (out_dir / "analysis_report.md").write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
