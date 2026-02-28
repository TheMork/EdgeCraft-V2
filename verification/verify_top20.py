from playwright.sync_api import sync_playwright
import time

def verify_ui():
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()

        print("Navigating to simulation page...")
        try:
            page.goto("http://localhost:3000/simulation", timeout=60000)
        except Exception as e:
            print(f"Failed to load page: {e}")
            browser.close()
            return

        page.wait_for_load_state("networkidle")

        # Take initial screenshot
        page.screenshot(path="verification/initial_state.png")
        print("Initial screenshot saved.")

        # Check for Sync Top 20 button
        if page.get_by_role("button", name="Sync Top 20").is_visible():
            print("Sync Top 20 button is visible.")
        else:
            print("Sync Top 20 button NOT visible.")

        # Verify initial symbol "BTC/USDT" in main content
        # There should be a span with BTC/USDT.
        # Check text content.
        content = page.content()
        if "BTC/USDT" in content:
            print("Found BTC/USDT in page content.")
        else:
            print("BTC/USDT NOT found in page content.")

        # Interact with Sidebar
        print("Interacting with Sidebar...")
        # Try to find the select by its options or position
        # Using nth(1) assuming it's the second select on the page (Strategy is first)
        # Strategy select has options "Moving Average Crossover"

        selects = page.locator("select")
        count = selects.count()
        print(f"Found {count} select elements.")

        symbol_select = None
        for i in range(count):
            txt = selects.nth(i).inner_text()
            if "BTC/USDT" in txt:
                symbol_select = selects.nth(i)
                print(f"Found symbol select at index {i}")
                break

        if symbol_select:
            symbol_select.select_option("ETH/USDT")
            print("Selected ETH/USDT in sidebar.")
            time.sleep(1) # Wait for React update

            # Verify update
            page.screenshot(path="verification/after_selection.png")

            # Check if ETH/USDT is now displayed in the simulation control card
            # The control card shows the selected symbol in a span
            # We can look for the text "ETH/USDT" in a specific area if needed, or just generally
            if page.get_by_text("ETH/USDT").count() > 0:
                 print("ETH/USDT found on page after selection.")
            else:
                 print("ETH/USDT NOT found after selection.")

        else:
            print("Could not find symbol select.")

        browser.close()

if __name__ == "__main__":
    verify_ui()
