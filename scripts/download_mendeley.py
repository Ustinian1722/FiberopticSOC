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
PUBLIC_API = "https://data.mendeley.com/public-api/datasets"
DATA_API = "https://api.data.mendeley.com/datasets"
JINA_READER = "https://r.jina.ai/"


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
    session.headers.update({
        "User-Agent": (
            "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/140.0 Safari/537.36"
        ),
        "Accept-Language": "en-US,en;q=0.9",
    })
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


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def safe_filename(name: str) -> str:
    name = name.strip().replace("\\", "_").replace("/", "_")
    name = re.sub(r"[^A-Za-z0-9._()\-\u4e00-\u9fff]+", "_", name)
    return name or "unnamed_file"


def stream_to_file(resp: requests.Response, target: Path) -> None:
    with target.open("wb") as f:
        for chunk in resp.iter_content(chunk_size=1024 * 1024):
            if chunk:
                f.write(chunk)


def looks_like_zip(path: Path) -> bool:
    try:
        with path.open("rb") as f:
            return f.read(4).startswith(b"PK")
    except OSError:
        return False


def try_direct_dataset_zip(
    session: requests.Session, dataset_id: str, version: int, out_dir: Path
) -> dict[str, Any] | None:
    candidates = [
        f"{DATA_API}/{dataset_id}/zip/file_downloaded?version={version}",
        f"{DATA_API}/{dataset_id}/zip?version={version}",
    ]
    errors: list[str] = []

    for url in candidates:
        print(f"Trying Mendeley dataset ZIP endpoint: {url}")
        try:
            resp = session.get(
                url,
                headers={"Accept": "application/zip, application/json, */*"},
                timeout=180,
                allow_redirects=True,
                stream=True,
            )
        except requests.RequestException as exc:
            errors.append(f"{url}: {exc!r}")
            continue

        content_type = (resp.headers.get("content-type") or "").lower()
        print(f"  -> HTTP {resp.status_code}; content-type={content_type}; final={resp.url}")
        if not resp.ok:
            body = resp.text[:300].replace("\n", " ")
            errors.append(f"{url}: HTTP {resp.status_code}: {body}")
            continue

        if "zip" in content_type or "octet-stream" in content_type:
            target = out_dir / "mendeley_dataset.zip"
            stream_to_file(resp, target)
            if looks_like_zip(target):
                return {
                    "mendeley_id": "dataset-zip",
                    "filename": target.name,
                    "original_filename": target.name,
                    "bytes": target.stat().st_size,
                    "sha256": sha256_file(target),
                    "expected_sha256": None,
                    "content_type": content_type,
                    "source_download_url": resp.url,
                }
            target.unlink(missing_ok=True)
            errors.append(f"{url}: binary response was not a ZIP archive")
            continue

        try:
            payload = resp.json()
        except Exception:
            text = resp.text[:300].replace("\n", " ")
            errors.append(f"{url}: HTTP 200 but unsupported payload: {text}")
            continue

        if isinstance(payload, dict):
            signed_url = (
                payload.get("url")
                or payload.get("download_url")
                or (payload.get("content_details") or {}).get("download_url")
            )
            status = str(payload.get("status") or "").upper()
            print(f"  -> JSON archive status={status or 'UNKNOWN'}; signed_url={bool(signed_url)}")
            if signed_url:
                archive_resp = session.get(str(signed_url), stream=True, timeout=180, allow_redirects=True)
                archive_resp.raise_for_status()
                target = out_dir / "mendeley_dataset.zip"
                stream_to_file(archive_resp, target)
                if not looks_like_zip(target):
                    target.unlink(missing_ok=True)
                    errors.append(f"{url}: signed URL did not return ZIP bytes")
                    continue
                actual_sha = sha256_file(target)
                expected_sha = payload.get("sha256_hash") or (payload.get("content_details") or {}).get("sha256_hash")
                if expected_sha and str(expected_sha).lower() != actual_sha.lower():
                    raise RuntimeError("SHA-256 mismatch for dataset ZIP")
                return {
                    "mendeley_id": "dataset-zip",
                    "filename": target.name,
                    "original_filename": target.name,
                    "bytes": target.stat().st_size,
                    "sha256": actual_sha,
                    "expected_sha256": expected_sha,
                    "content_type": archive_resp.headers.get("content-type"),
                    "source_download_url": str(signed_url),
                    "archive_status": status or None,
                }

        errors.append(f"{url}: no downloadable archive URL in JSON response")

    print("Direct dataset ZIP endpoints were unavailable:")
    for error in errors:
        print(f"  - {error}")
    return None


def extract_json_from_reader_text(text: str) -> Any:
    """Extract a JSON object/array from a text-proxy response."""
    stripped = text.strip()
    candidates = [stripped]
    fenced = re.findall(r"```(?:json)?\s*(.*?)```", stripped, flags=re.S | re.I)
    candidates.extend(fenced)
    for open_ch, close_ch in (("[", "]"), ("{", "}")):
        start = stripped.find(open_ch)
        end = stripped.rfind(close_ch)
        if start >= 0 and end > start:
            candidates.append(stripped[start : end + 1])
    for candidate in candidates:
        try:
            return json.loads(candidate)
        except Exception:
            pass
    raise ValueError("No parseable JSON payload found in reader response")


def list_files_via_reader(session: requests.Session, dataset_id: str, version: int) -> list[dict[str, Any]]:
    """Metadata-only fallback through a public text reader.

    This is used only to recover the public file metadata/UUIDs when Mendeley's
    Cloudflare challenge blocks GitHub-hosted runner IPs. Actual dataset bytes are
    still downloaded from Mendeley's own public file URLs.
    """
    origin = f"https://data.mendeley.com/public-api/datasets/{dataset_id}/files?folder_id=root&version={version}"
    proxy_url = JINA_READER + origin
    print(f"Trying metadata-only reader fallback: {proxy_url}")
    resp = session.get(proxy_url, headers={"Accept": "text/plain, */*"}, timeout=120)
    print(f"  -> HTTP {resp.status_code}; content-type={resp.headers.get('content-type')}")
    resp.raise_for_status()
    payload = extract_json_from_reader_text(resp.text)
    files = normalize_file_list(payload)
    if not files:
        raise RuntimeError("Reader fallback returned an empty Mendeley file list")
    print(f"  -> recovered metadata for {len(files)} public file(s)")
    return files


def list_files(session: requests.Session, dataset_id: str, version: int) -> tuple[list[dict[str, Any]], str]:
    dataset_page = f"https://data.mendeley.com/datasets/{dataset_id}/{version}"
    api_url = f"{PUBLIC_API}/{dataset_id}/files"

    try:
        session.get(dataset_page, headers={"Accept": "text/html,application/xhtml+xml"}, timeout=60).raise_for_status()
    except requests.RequestException:
        pass

    strategies = [
        {"folder_id": "root", "version": version},
        {"version": version},
    ]
    errors: list[str] = []

    for params in strategies:
        resp = session.get(
            api_url,
            params=params,
            headers={
                "Accept": "application/json, text/plain, */*",
                "Referer": dataset_page,
                "Origin": "https://data.mendeley.com",
            },
            timeout=60,
        )
        if resp.ok:
            files = normalize_file_list(resp.json())
            if files:
                return files, "public_file_api"
            errors.append(f"{resp.url}: HTTP 200 but no files")
            continue
        body = resp.text[:300].replace("\n", " ")
        errors.append(f"{resp.url}: HTTP {resp.status_code}: {body}")

    print("Mendeley web file API unavailable from this runner:")
    for error in errors:
        print(f"  - {error}")

    files = list_files_via_reader(session, dataset_id, version)
    return files, "reader_metadata_plus_mendeley_public_files"


def download_file(
    session: requests.Session, item: dict[str, Any], out_dir: Path, dataset_id: str, version: int
) -> dict[str, Any]:
    details = item.get("content_details") or {}
    file_id = item.get("id") or details.get("id")
    url = details.get("download_url") or item.get("download_url")

    # Prefer the stable Mendeley public-files route when a UUID is known. It avoids
    # another metadata/API request and preserves attribution to the original host.
    if file_id:
        url = f"https://data.mendeley.com/public-files/datasets/{dataset_id}/files/{file_id}/file_downloaded?version={version}"
    if not url:
        raise KeyError(f"No download URL or file UUID for Mendeley file: {item}")

    original_name = item.get("filename") or item.get("name") or details.get("filename") or file_id or "file"
    filename = safe_filename(str(original_name))
    target = out_dir / filename

    if target.exists():
        stem, suffix = target.stem, target.suffix
        idx = 2
        while target.exists():
            target = out_dir / f"{stem}_{idx}{suffix}"
            idx += 1

    print(f"Downloading public file from Mendeley: {url}")
    with session.get(url, stream=True, timeout=300, allow_redirects=True) as resp:
        print(f"  -> HTTP {resp.status_code}; final={resp.url}; type={resp.headers.get('content-type')}")
        resp.raise_for_status()
        stream_to_file(resp, target)

    actual_sha = sha256_file(target)
    expected_sha = details.get("sha256_hash") or item.get("sha256_hash")
    if expected_sha and str(expected_sha).lower() != actual_sha.lower():
        raise RuntimeError(f"SHA-256 mismatch for {filename}")

    return {
        "mendeley_id": file_id,
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

    records: list[dict[str, Any]] = []
    archive_record = try_direct_dataset_zip(session, args.dataset_id, args.version, out_dir)
    if archive_record is not None:
        records.append(archive_record)
        acquisition_mode = "dataset_zip_api"
    else:
        files, acquisition_mode = list_files(session, args.dataset_id, args.version)
        for idx, item in enumerate(files, start=1):
            print(f"[{idx}/{len(files)}] {item.get('filename') or item.get('name') or item.get('id')}")
            records.append(download_file(session, item, out_dir, args.dataset_id, args.version))

    manifest = {
        "dataset_id": args.dataset_id,
        "version": args.version,
        "doi": f"10.17632/{args.dataset_id}.{args.version}",
        "dataset_page": f"https://data.mendeley.com/datasets/{args.dataset_id}/{args.version}",
        "acquisition_mode": acquisition_mode,
        "file_count": len(records),
        "total_bytes": sum(r["bytes"] for r in records),
        "files": records,
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({k: manifest[k] for k in ("doi", "acquisition_mode", "file_count", "total_bytes")}, indent=2))


if __name__ == "__main__":
    main()
