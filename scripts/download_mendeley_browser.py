from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from urllib.parse import urljoin

from playwright.sync_api import TimeoutError as PlaywrightTimeoutError
from playwright.sync_api import sync_playwright

DATASET_ID = "ft6rtwt8vm"
VERSION = 1


def save_download(page, locator, target: Path, label: str) -> bool:
    """Click a locator and save the resulting browser download."""
    try:
        print(f"Trying browser download via {label}...")
        with page.expect_download(timeout=120_000) as download_info:
            locator.click(timeout=30_000)
        download = download_info.value
        print(f"  -> suggested filename: {download.suggested_filename}")
        download.save_as(str(target))
        return True
    except PlaywrightTimeoutError as exc:
        print(f"  -> {label} did not emit a browser download: {exc}")
        return False


def save_via_browser_request(context, url: str, target: Path) -> bool:
    """Fetch a download URL through Playwright's browser context/cookies."""
    print(f"Trying browser-context request: {url}")
    response = context.request.get(url, timeout=180_000, fail_on_status_code=False)
    print(f"  -> HTTP {response.status}; content-type={response.headers.get('content-type')}")
    if not response.ok:
        return False
    body = response.body()
    if not body:
        return False
    target.write_bytes(body)
    return True


def zip_ok(path: Path) -> bool:
    if not path.exists() or path.stat().st_size == 0:
        return False
    with path.open("rb") as f:
        return f.read(4).startswith(b"PK")


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

        for attempt in range(12):
            title = page.title()
            body_text = page.locator("body").inner_text(timeout=10_000)[:1500]
            print(f"Browser probe {attempt + 1}: title={title!r}; url={page.url}")
            if "Download All" in body_text and "Multi-condition Battery" in body_text:
                break
            page.wait_for_timeout(5_000)
        else:
            page.screenshot(path=str(out_dir / "mendeley_browser_blocked.png"), full_page=True)
            raise RuntimeError(
                "Chromium could not reach the public dataset page after Cloudflare challenge; "
                f"final title={page.title()!r}, url={page.url}"
            )

        # Remove the cookie banner so it cannot intercept a later pointer click.
        cookie_button = page.get_by_role("button", name="Accept all cookies")
        if cookie_button.count() and cookie_button.first.is_visible():
            try:
                cookie_button.first.click(timeout=10_000)
                print("Accepted cookie banner.")
            except PlaywrightTimeoutError:
                print("Cookie banner could not be clicked; continuing.")

        # Mendeley initially renders Download All as disabled, then replaces/enables
        # it after file metadata arrives. Wait for the *current enabled DOM node*.
        print("Waiting for Mendeley file metadata / enabled Download All button...")
        try:
            page.wait_for_function(
                """() => {
                    const b = document.querySelector('#download-all');
                    return !!b && !b.disabled && !b.hasAttribute('disabled');
                }""",
                timeout=120_000,
            )
            button = page.locator("#download-all:not([disabled])")
            button.wait_for(state="visible", timeout=30_000)
            print("Enabled Download All control:", button.evaluate("el => el.outerHTML"))
        except PlaywrightTimeoutError:
            button = None
            print("Download All never became enabled; will try file-level fallbacks.")

        acquired = False
        if button is not None:
            acquired = save_download(page, button, target, "enabled #download-all")
            if acquired and not zip_ok(target):
                print("Download All returned non-ZIP bytes; discarding and falling back.")
                target.unlink(missing_ok=True)
                acquired = False

        # The dataset currently contains one ZIP file. If the aggregate control is
        # implemented as an async frontend action rather than a native download,
        # recover the file-level download URL rendered by the page.
        if not acquired:
            html = page.content()
            candidates: list[str] = []
            patterns = [
                r'https?://[^"\'<>\s]+/file_downloaded[^"\'<>\s]*',
                r'[^"\'<>\s]+/public-files/datasets/[^"\'<>\s]+',
            ]
            for pattern in patterns:
                for match in re.findall(pattern, html):
                    url = urljoin(page.url, match.replace("&amp;", "&"))
                    if url not in candidates:
                        candidates.append(url)

            print(f"Found {len(candidates)} direct-looking file URL candidate(s) in DOM.")
            for url in candidates[:20]:
                if save_via_browser_request(context, url, target) and zip_ok(target):
                    acquired = True
                    break
                target.unlink(missing_ok=True)

        # Last UI fallback: click file-level download anchors/buttons. This is
        # intentionally broad but scoped to download-like controls on the dataset page.
        if not acquired:
            selectors = [
                "a[href*='file_downloaded']",
                "a[download]",
                "a[aria-label*='download' i]",
                "button[aria-label*='download' i]",
                "[title*='download' i]",
            ]
            for selector in selectors:
                loc = page.locator(selector)
                count = loc.count()
                print(f"UI fallback selector {selector!r}: {count} match(es)")
                for idx in range(min(count, 10)):
                    item = loc.nth(idx)
                    try:
                        if not item.is_visible():
                            continue
                        print("  candidate:", item.evaluate("el => el.outerHTML"))
                        if save_download(page, item, target, f"{selector}[{idx}]") and zip_ok(target):
                            acquired = True
                            break
                        target.unlink(missing_ok=True)
                    except Exception as exc:
                        print(f"  candidate failed: {exc!r}")
                if acquired:
                    break

        if not acquired:
            page.screenshot(path=str(out_dir / "mendeley_download_timeout.png"), full_page=True)
            # Save a compact DOM diagnostic as text so future failures can be fixed
            # from the artifact without guessing frontend markup.
            (out_dir / "mendeley_download_dom.txt").write_text(
                page.content(), encoding="utf-8", errors="replace"
            )
            raise RuntimeError("Dataset page loaded, but no usable ZIP download could be acquired")

        print(f"Dataset ZIP acquired: {target.stat().st_size} bytes")
        browser.close()

    if not zip_ok(target):
        raise RuntimeError("Browser acquisition did not produce a valid ZIP archive")

    manifest = {
        "dataset_id": args.dataset_id,
        "version": args.version,
        "doi": f"10.17632/{args.dataset_id}.{args.version}",
        "dataset_page": dataset_url,
        "acquisition_mode": "chromium_dataset_page",
        "file_count": 1,
        "total_bytes": target.stat().st_size,
        "files": [
            {
                "mendeley_id": "dataset-zip",
                "filename": target.name,
                "original_filename": "SiC-18.zip",
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
