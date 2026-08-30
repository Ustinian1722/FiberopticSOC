from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from playwright.sync_api import TimeoutError as PlaywrightTimeoutError
from playwright.sync_api import sync_playwright

DATASET_ID = "ft6rtwt8vm"
VERSION = 1


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-id", default=DATASET_ID)
    parser.add_argument("--version", type=int, default=VERSION)
    parser.add_argument("--out", default="data/raw")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    dataset_url = f"https://data.mendeley.com/datasets/{args.dataset_id}/{args.version}"
    target = out_dir / "mendeley_dataset.zip"

    with sync_playwright() as p:
        browser = p.chromium.launch(
            headless=True,
            args=[
                "--disable-blink-features=AutomationControlled",
                "--no-sandbox",
                "--disable-dev-shm-usage",
            ],
        )
        context = browser.new_context(
            accept_downloads=True,
            locale="en-US",
            viewport={"width": 1440, "height": 1000},
            user_agent=(
                "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                "(KHTML, like Gecko) Chrome/140.0.0.0 Safari/537.36"
            ),
        )
        page = context.new_page()

        print(f"Opening public Mendeley dataset page in Chromium: {dataset_url}")
        page.goto(dataset_url, wait_until="domcontentloaded", timeout=120_000)

        # Managed Cloudflare challenges often complete after JS executes for a few
        # seconds. Poll the title/body rather than relying on a fixed redirect.
        for attempt in range(12):
            title = page.title()
            body_text = page.locator("body").inner_text(timeout=10_000)[:1000]
            print(f"Browser probe {attempt + 1}: title={title!r}; url={page.url}")
            if "Download All" in body_text or "Multi-condition Battery" in body_text:
                break
            page.wait_for_timeout(5_000)
        else:
            page.screenshot(path=str(out_dir / "mendeley_browser_blocked.png"), full_page=True)
            raise RuntimeError(
                "Chromium could not reach the public dataset page after Cloudflare challenge; "
                f"final title={page.title()!r}, url={page.url}"
            )

        button = page.get_by_role("button", name="Download All")
        if button.count() == 0:
            # Accessible-name markup may differ by frontend revision.
            button = page.get_by_text("Download All", exact=True)
        if button.count() == 0:
            page.screenshot(path=str(out_dir / "mendeley_no_download_button.png"), full_page=True)
            raise RuntimeError("Dataset page loaded but Download All control was not found")

        print("Dataset page loaded; triggering Download All...")
        try:
            with page.expect_download(timeout=180_000) as download_info:
                button.first.click(timeout=30_000)
            download = download_info.value
            print(f"Suggested filename: {download.suggested_filename}")
            download.save_as(str(target))
        except PlaywrightTimeoutError as exc:
            page.screenshot(path=str(out_dir / "mendeley_download_timeout.png"), full_page=True)
            raise RuntimeError("Download All did not produce a browser download") from exc

        browser.close()

    if not target.exists() or target.stat().st_size == 0:
        raise RuntimeError("Browser download produced no dataset bytes")

    # ZIP archives begin with one of the standard PK signatures.
    with target.open("rb") as f:
        signature = f.read(4)
    if not signature.startswith(b"PK"):
        raise RuntimeError(f"Downloaded file is not a ZIP archive; signature={signature!r}")

    manifest = {
        "dataset_id": args.dataset_id,
        "version": args.version,
        "doi": f"10.17632/{args.dataset_id}.{args.version}",
        "dataset_page": dataset_url,
        "acquisition_mode": "chromium_download_all",
        "file_count": 1,
        "total_bytes": target.stat().st_size,
        "files": [
            {
                "mendeley_id": "dataset-zip",
                "filename": target.name,
                "original_filename": target.name,
                "bytes": target.stat().st_size,
                "content_type": "application/zip",
                "source_download_url": dataset_url,
            }
        ],
    }
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps({k: manifest[k] for k in ("doi", "acquisition_mode", "total_bytes")}, indent=2))


if __name__ == "__main__":
    main()
