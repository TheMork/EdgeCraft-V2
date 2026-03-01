from playwright.sync_api import Page, expect, sync_playwright

def verify_dashboard_ui(page: Page):
    # Navigate to the dashboard
    page.goto("http://localhost:3000/simulation")

    # Wait for the main elements to load
    page.wait_for_selector("h1:has-text('Crypto Quant Dashboard')")

    # Optional: ensure cards are visible
    page.wait_for_selector("h3:has-text('Simulation Control')")
    page.wait_for_selector("h3:has-text('Price Chart')")

    # Expand or show some elements if needed

    # Give the page a moment to render the charts and styles fully
    page.wait_for_timeout(2000)

    # Take a full page screenshot
    page.screenshot(path="/home/jules/verification/dashboard_redesign.png", full_page=True)

if __name__ == "__main__":
    with sync_playwright() as p:
        # Increase viewport size to simulate a large desktop monitor
        browser = p.chromium.launch(headless=True)
        context = browser.new_context(viewport={'width': 1920, 'height': 1080})
        page = context.new_page()
        try:
            verify_dashboard_ui(page)
            print("Verification screenshot saved.")
        except Exception as e:
            print(f"Error during verification: {e}")
            page.screenshot(path="/home/jules/verification/error.png")
        finally:
            browser.close()