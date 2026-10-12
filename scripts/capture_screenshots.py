"""Click through the app in a real browser and save screenshots for the README.

Start the app first (scripts\\run_app.bat, or streamlit run app\\Home.py on port 8501), then:

    python scripts/capture_screenshots.py [--url http://localhost:8501] [--out docs/img]

The script follows the same path a person would: Home, then Analyze (preset area, dates,
no concessions, offline demo), then Results, then Models. It fails loudly if a step is
missing, so a broken page shows up as an error rather than a bad screenshot.
Requires: pip install playwright and python -m playwright install chromium.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from playwright.sync_api import TimeoutError as PWTimeout
from playwright.sync_api import expect, sync_playwright

REPO_DIR = Path(__file__).resolve().parents[1]


def wait_for_text(page, text: str, timeout: int = 60_000) -> None:
    expect(page.get_by_text(text, exact=False).first).to_be_visible(timeout=timeout)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://localhost:8501")
    parser.add_argument("--out", default=str(REPO_DIR / "docs" / "img"))
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(viewport={"width": 1400, "height": 900})
        page.goto(args.url)
        wait_for_text(page, "Illegal Mining Detector")
        page.wait_for_timeout(1500)
        page.screenshot(path=str(out / "home.png"), full_page=True)
        print("home ok")

        page.get_by_role("button", name="Analyze my area").click()
        wait_for_text(page, "Step 1 of 4")
        page.screenshot(path=str(out / "analyze.png"), full_page=True)
        print("analyze step 1 ok")

        page.get_by_role("button", name="Next: dates").click()
        wait_for_text(page, "Step 2 of 4")
        page.get_by_role("button", name="Next: concessions").click()
        wait_for_text(page, "Step 3 of 4")
        page.get_by_role("button", name="Skip (every alert will be 'illegal')").click()
        wait_for_text(page, "Step 4 of 4")
        page.get_by_text("Offline demo instead", exact=False).click()
        page.get_by_role("button", name="Run the demo").click()
        print("demo started")

        try:
            wait_for_text(page, "Results", timeout=180_000)
            wait_for_text(page, "This is DEMO output", timeout=60_000)
        except PWTimeout:
            print("ERROR: did not reach Results with the DEMO warning", file=sys.stderr)
            page.screenshot(path=str(out / "_failed_analyze.png"), full_page=True)
            browser.close()
            return 1
        # The map is an iframe that loads lazily, so scroll to it and wait before capturing.
        frame = page.locator("iframe").first
        frame.scroll_into_view_if_needed()
        page.wait_for_timeout(4000)
        frame.screenshot(path=str(out / "results_map.png"))
        page.evaluate("window.scrollTo(0, 0)")
        page.wait_for_timeout(1000)
        page.screenshot(path=str(out / "results.png"), full_page=True)
        print("results with DEMO warning ok")

        page.get_by_role("link", name="Models").click()
        wait_for_text(page, "Deep-learning models")
        page.wait_for_timeout(2500)
        page.screenshot(path=str(out / "models.png"), full_page=True)
        print("models ok")
        browser.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
