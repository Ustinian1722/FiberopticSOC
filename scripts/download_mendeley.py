from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

DATASET_ID = "ft6rtwt8vm"
VERSION = 1
BASE_API = "https://data.mendeley.com/public-api/datasets"


def build_session() -> requests.Session:
    retry = Retry(
        total=5,
        connect=5,
        read=5,
        backoff_factor=1.0,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset(["GET"]),
    )
    session = requests.Session()
    session.headers.update({"User-Agent": "FiberopticSOC/1.0 (GitHub Actions research workflow)"})
    session.mount("https://", HTTPAdapter(max_retries=retry))
    return session


def normalize_file_list(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        for key in ("results", "files", "items"):
            value = payload.get(key)
            if isinstance(value, list):
                return value
    raise TypeError(f"Unexpected Mendeley response shape: {type(payload).__name__}")


def list_files(session: requests.Session, dataset_id: str, version: int) -> list[dict[str, Any]]:
    url = f"{BASE_API}/{dataset_id}/files"
    all_files: list[dict[str, Any]] = []
    start = 0
    limit = 100

    # Most Mendeley datasets expose files at root. If that request fails, retry
    # without folder_id because the public API has changed behavior over time.
    strategies = [
        {"folder_id": "root", "version": version},
        {"version": version},
    ]

    last_error: Exception | None = None
    for base_params in strategies:
        all_files.clear()
        start = 0
        try:
            while True:
                params = dict(base_params)
                params.update({"$start": start, "$limit": limit})
                resp = session.get(url, params=params, timeout=60)
                resp.raise_for_status()
                batch = normalize_file_list(resp.json())
                all_files.extend(batch)
                if len(batch) < limit:
                    break
                start += len(batch)
            if all_files:
                return all_files
        except Exception as exc:  # pragma: no cover - network behavior
            last_error = exc

    if last_error:
        raise RuntimeError("Unable to enumerate Mendeley dataset files") from last_error
    raise RuntimeError("Mendeley API returned no files")


def safe_filename(name: str) -> str:
    name = name.strip().replace("\\", "_").replace("/", "_")
    name = re.sub(r"[^A-Za-z0-9._()\-\u4e00-\u9fff]+", "_", name)
    return name or "unnamed_file"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def download_file(session: requests.Session, item: dict[str, Any], out_dir: Path) -> dict[str, Any]:
    details = item.get("content_details") or {}
    url = details.get("download_url") or item.get("download_url")
    if not url:
        raise KeyError(f"No download_url for Mendeley file: {item}")

    original_name = item.get("filename") or item.get("name") or details.get("filename") or item.get("id", "file")
    filename = safe_filename(str(original_name))
    target = out_dir / filename

    # Disambiguate duplicate names without silently overwriting data.
    if target.exists():
        stem, suffix = target.stem, target.suffix
        idx = 2
        while target.exists():
            target = out_dir / f"{stem}_{idx}{suffix}"
            idx += 1

    with session.get(url, stream=True, timeout=120, allow_redirects=True) as resp:
        resp.raise_for_status()
        with target.open("wb") as f:
            for chunk in resp.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)

    actual_sha = sha256_file(target)
    expected_sha = details.get("sha256_hash") or item.get("sha256_hash")
    if expected_sha and str(expected_sha).lower() != actual_sha.lower():
        raise RuntimeError(f"SHA-256 mismatch for {filename}")

    return {
        "mendeley_id": item.get("id"),
        "filename": filename,
        "original_filename": original_name,
        "bytes": target.stat().st_size,
        "sha256": actual_sha,
        "expected_sha256": expected_sha,
        "content_type": details.get("content_type") or item.get("mime_type"),
        "source_download_url": url,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-id", default=DATASET_ID)
    parser.add_argument("--version", type=int, default=VERSION)
    parser.add_argument("--out", default="data/raw")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    session = build_session()

    files = list_files(session, args.dataset_id, args.version)
    records = []
    for idx, item in enumerate(files, start=1):
        print(f"[{idx}/{len(files)}] downloading {item.get('filename') or item.get('name') or item.get('id')}")
        records.append(download_file(session, item, out_dir))

    manifest = {
        "dataset_id": args.dataset_id,
        "version": args.version,
        "doi": f"10.17632/{args.dataset_id}.{args.version}",
        "dataset_page": f"https://data.mendeley.com/datasets/{args.dataset_id}/{args.version}",
        "file_count": len(records),
        "total_bytes": sum(r["bytes"] for r in records),
        "files": records,
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({k: manifest[k] for k in ("doi", "file_count", "total_bytes")}, indent=2))


if __name__ == "__main__":
    main()
