
import sys
import os
sys.path.append(os.getcwd())
try:
    from src.data_manager import SyncManager
except ImportError:
    print("Could not import SyncManager.")
    sys.exit(1)

def populate():
    print("Starting data population...")
    sync = SyncManager()
    # Sync just 2 days for speed testing
    symbol = "BTC/USDT"
    start = "2024-01-01T00:00:00Z"
    end = "2024-01-03T00:00:00Z" 
    try:
        sync.sync_data(symbol, start, end)
        print("Data population complete.")
    except Exception as e:
        print(f"Data population failed: {e}")

if __name__ == "__main__":
    populate()
