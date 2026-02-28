from playwright.sync_api import sync_playwright

def run(playwright):
    browser = playwright.chromium.launch(headless=True)
    context = browser.new_context()
    page = context.new_page()

    print("Navigating to http://localhost:3000/simulation")
    # Navigate to the simulation page
    try:
        page.goto("http://localhost:3000/simulation")
    except Exception as e:
        print(f"Failed to load page: {e}")
        browser.close()
        return

    # Verify title or key elements
    try:
        page.wait_for_selector("text=Simulation Control", timeout=5000)
    except Exception as e:
        print(f"Simulation Control not found: {e}")
        page.screenshot(path="page_load_fail.png")
        browser.close()
        return

    print("Clicking Start Simulation")
    # Click Start Simulation
    page.click("text=Start Simulation")

    # Wait for completion message or metrics to appear
    # The metrics card has title "Backtest Metrics"
    print("Waiting for metrics...")
    try:
        page.wait_for_selector("text=Backtest Metrics", timeout=10000)
        print("Metrics found!")
    except Exception as e:
        print(f"Timeout waiting for metrics: {e}")
        # Take screenshot anyway to see what happened
        page.screenshot(path="frontend_error.png")
        browser.close()
        return

    # Wait a bit for chart to render
    page.wait_for_timeout(2000)

    # Take screenshot
    page.screenshot(path="simulation_result.png")
    print("Screenshot saved to simulation_result.png")

    browser.close()

if __name__ == "__main__":
    with sync_playwright() as playwright:
        run(playwright)
