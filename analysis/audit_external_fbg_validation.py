from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

import pandas as pd

TEXT_EXT = {".csv", ".txt", ".tsv", ".dat", ".log"}
EXCEL_EXT = {".xlsx", ".xls"}
CELLS = ("A1", "A2", "P1", "P2")


def decode_head(path: Path, n: int = 131072) -> str:
    raw = path.read_bytes()[:n]
    for enc in ("utf-8-sig", "utf-8", "cp1252", "latin1"):
        try:
            return raw.decode(enc)
        except UnicodeDecodeError:
            continue
    return raw.decode("latin1", errors="replace")


def text_header(path: Path) -> tuple[str | None, list[str]]:
    text = decode_head(path)
    lines = [ln for ln in text.splitlines() if ln.strip()]
    if not lines:
        return None, []
    sample = "\n".join(lines[:20])
    delim = None
    try:
        delim = csv.Sniffer().sniff(sample, delimiters=",;\t|").delimiter
    except csv.Error:
        for d in (";", "\t", ",", "|"):
            if d in lines[0]:
                delim = d
                break
    if delim is None:
        return None, [lines[0].strip()]
    return delim, [x.strip().strip('"') for x in lines[0].split(delim)]


def classify_path(rel: str) -> dict[str, bool]:
    low = rel.lower()
    return {
        "looks_constant_current": any(k in low for k in ("constant", "current", "cc", "0.2c", "0.5c", "1c")),
        "looks_wltp": "wltp" in low,
        "looks_validation": "valid" in low,
    }


def inventory(root: Path) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    rows = []
    col_rows = []
    for path in sorted(p for p in root.rglob("*") if p.is_file()):
        rel = path.relative_to(root).as_posix()
        ext = path.suffix.lower()
        row = {
            "path": rel,
            "extension": ext or "<none>",
            "size_bytes": path.stat().st_size,
            **classify_path(rel),
            "cells_in_path": ",".join(c for c in CELLS if c.lower() in rel.lower()),
            "parse_status": "not_parsed",
            "delimiter": "",
            "sheet_count": 0,
            "header_column_count": 0,
        }
        headers: list[tuple[str, list[str]]] = []
        try:
            if ext in TEXT_EXT:
                delim, cols = text_header(path)
                row["parse_status"] = "text_header_ok"
                row["delimiter"] = repr(delim) if delim is not None else ""
                row["header_column_count"] = len(cols)
                headers.append(("", cols))
            elif ext in EXCEL_EXT:
                xls = pd.ExcelFile(path)
                row["parse_status"] = "excel_header_ok"
                row["sheet_count"] = len(xls.sheet_names)
                for sheet in xls.sheet_names[:10]:
                    sample = pd.read_excel(path, sheet_name=sheet, nrows=2)
                    headers.append((str(sheet), [str(c) for c in sample.columns]))
                row["header_column_count"] = max((len(c) for _, c in headers), default=0)
        except Exception as exc:  # audit must survive odd files
            row["parse_status"] = f"parse_error:{type(exc).__name__}"
        rows.append(row)

        for sheet, cols in headers:
            for col in cols:
                low = col.lower()
                matched_cells = [c for c in CELLS if c.lower() in low]
                sensor_match = re.search(r"(?i)s\s*([1-9])", col)
                col_rows.append(
                    {
                        "path": rel,
                        "sheet": sheet,
                        "column": col,
                        "cell_tokens": ",".join(matched_cells),
                        "sensor_position": f"S{sensor_match.group(1)}" if sensor_match else "",
                        "is_S5": bool(re.search(r"(?i)s\s*5", col)),
                        "looks_time": any(k in low for k in ("time", "timestamp", "date")),
                        "looks_voltage": any(k in low for k in ("voltage", "volt", "u_", " u")),
                        "looks_current": any(k in low for k in ("current", "amp", " i", "i_")),
                        "looks_temperature": any(k in low for k in ("temp", "pt100", "chamber", "t_")),
                        "looks_wavelength": any(k in low for k in ("wavelength", "bragg", "lambda", "fbg")),
                        "looks_soc": "soc" in low,
                    }
                )

    manifest = pd.DataFrame(rows)
    columns = pd.DataFrame(col_rows)
    summary = {
        "n_files": int(len(manifest)),
        "total_bytes": int(manifest.size_bytes.sum()) if len(manifest) else 0,
        "extensions": manifest.extension.value_counts().to_dict() if len(manifest) else {},
        "constant_current_candidate_files": manifest.loc[manifest.looks_constant_current, "path"].tolist() if len(manifest) else [],
        "wltp_candidate_files": manifest.loc[manifest.looks_wltp, "path"].tolist() if len(manifest) else [],
        "s5_columns": columns.loc[columns.is_S5, ["path", "sheet", "column", "cell_tokens"]].to_dict("records") if len(columns) else [],
        "cells_seen_in_paths": {c: bool(manifest.cells_in_path.str.contains(c, regex=False).any()) for c in CELLS} if len(manifest) else {},
        "cells_seen_in_headers": {c: bool(columns.cell_tokens.str.contains(c, regex=False).any()) for c in CELLS} if len(columns) else {},
    }
    return manifest, columns, summary


def write_report(out: Path, manifest: pd.DataFrame, columns: pd.DataFrame, summary: dict) -> None:
    lines = [
        "# External FBG validation archive audit",
        "",
        f"- files: {summary['n_files']}",
        f"- total extracted bytes: {summary['total_bytes']}",
        f"- extensions: `{json.dumps(summary['extensions'], ensure_ascii=False)}`",
        f"- cells in paths: `{summary['cells_seen_in_paths']}`",
        f"- cells in headers: `{summary['cells_seen_in_headers']}`",
        f"- S5 header matches: {len(summary['s5_columns'])}",
        f"- constant-current candidate files: {len(summary['constant_current_candidate_files'])}",
        f"- WLTP candidate files: {len(summary['wltp_candidate_files'])}",
        "",
        "## Constant-current candidates",
    ]
    lines += [f"- `{p}`" for p in summary["constant_current_candidate_files"][:80]] or ["- none detected by filename"]
    lines += ["", "## WLTP candidates"]
    lines += [f"- `{p}`" for p in summary["wltp_candidate_files"][:80]] or ["- none detected by filename"]
    lines += ["", "## S5 columns"]
    for item in summary["s5_columns"][:120]:
        lines.append(f"- `{item['path']}` | `{item['sheet']}` | `{item['column']}` | cell={item['cell_tokens']}")
    if not summary["s5_columns"]:
        lines.append("- none detected in parsable headers")
    lines += ["", "## Signal-like columns"]
    if len(columns):
        sig = columns[
            columns[["looks_time", "looks_voltage", "looks_current", "looks_temperature", "looks_wavelength", "looks_soc"]].any(axis=1)
        ]
        for _, r in sig.head(150).iterrows():
            flags = [k.replace("looks_", "") for k in ("looks_time", "looks_voltage", "looks_current", "looks_temperature", "looks_wavelength", "looks_soc") if bool(r[k])]
            lines.append(f"- `{r.path}` | `{r.sheet}` | `{r.column}` -> {','.join(flags)}")
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    args = p.parse_args()
    manifest, columns, summary = inventory(args.root)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(args.out_dir / "external_validation_file_manifest.csv", index=False)
    columns.to_csv(args.out_dir / "external_validation_column_inventory.csv", index=False)
    (args.out_dir / "external_validation_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    write_report(args.out_dir / "external_validation_audit.md", manifest, columns, summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
