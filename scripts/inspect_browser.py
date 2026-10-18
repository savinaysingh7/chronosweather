import os
import time
from playwright.sync_api import sync_playwright

SCREENSHOT_DIR = "artifacts/browser_inspection"
os.makedirs(SCREENSHOT_DIR, exist_ok=True)

console_messages = []
network_errors = []

def run_inspection():
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        context = browser.new_context(viewport={"width": 1440, "height": 900})
        page = context.new_page()

        page.on("console", lambda msg: console_messages.append(f"[{msg.type}] {msg.text}"))
        page.on("requestfailed", lambda req: network_errors.append(f"{req.url}: {req.failure}"))

        print("Navigating to http://localhost:3001...")
        page.goto("http://localhost:3001", wait_until="networkidle")
        time.sleep(2)

        # 1. Overview / Command Center
        page.screenshot(path=f"{SCREENSHOT_DIR}/01_overview.png", full_page=True)
        print("Captured 01_overview.png")

        # Test Cohort Switcher on Command Center
        if page.locator("button:has-text('T1: Diurnal')").is_visible():
            page.click("button:has-text('T1: Diurnal')")
            time.sleep(1.5)
            page.screenshot(path=f"{SCREENSHOT_DIR}/01_overview_t1_diurnal.png", full_page=True)
            print("Captured 01_overview_t1_diurnal.png")
            page.click("button:has-text('T3: Squall')")
            time.sleep(1.5)

        # 2. Live Streaming Ops
        page.click("button:has-text('Live Stream Ops')")
        time.sleep(2)
        page.screenshot(path=f"{SCREENSHOT_DIR}/02_live_single.png", full_page=True)
        print("Captured 02_live_single.png")

        # Toggle to Side-by-Side Dual Stream & Inject Telemetry
        if page.locator("button:has-text('Side-by-Side Dual Grid Stream')").is_visible():
            page.click("button:has-text('Side-by-Side Dual Grid Stream')")
            time.sleep(1.5)
            if page.locator("button:has-text('Inject Telemetry')").is_visible():
                page.click("button:has-text('Inject Telemetry')")
                time.sleep(2)
            page.screenshot(path=f"{SCREENSHOT_DIR}/02_live_dual.png", full_page=True)
            print("Captured 02_live_dual.png")

        # 3. SHAP Explainability Studio
        page.click("button:has-text('SHAP Explainability')")
        time.sleep(2)
        page.screenshot(path=f"{SCREENSHOT_DIR}/03_shap.png", full_page=True)
        print("Captured 03_shap.png")

        # 4. Conformal Prediction Lab
        page.click("button:has-text('Conformal Prediction')")
        time.sleep(2)
        page.screenshot(path=f"{SCREENSHOT_DIR}/04_conformal.png", full_page=True)
        print("Captured 04_conformal.png")

        # 5. Cohort Drift Suite
        page.click("button:has-text('Cohort Drift Suite')")
        time.sleep(2)
        page.screenshot(path=f"{SCREENSHOT_DIR}/05_cohorts.png", full_page=True)
        print("Captured 05_cohorts.png")

        # 6. Cost & Retrain ROI
        page.click("button:has-text('Cost & Retrain ROI')")
        time.sleep(2)
        page.screenshot(path=f"{SCREENSHOT_DIR}/06_roi.png", full_page=True)
        print("Captured 06_roi.png")

        # 7. Beginner's Guide (ELI5)
        page.click("button:has-text('Beginner')")
        time.sleep(2)
        page.screenshot(path=f"{SCREENSHOT_DIR}/07_eli5.png", full_page=True)
        print("Captured 07_eli5.png")

        browser.close()

    print("\n--- Console Messages ---")
    for msg in console_messages:
        print(msg)

    print("\n--- Network Errors ---")
    if network_errors:
        for err in network_errors:
            print(err)
    else:
        print("None! All network requests succeeded.")

if __name__ == "__main__":
    run_inspection()
