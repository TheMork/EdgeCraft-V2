
import sys
import os
import pandas as pd
sys.path.append(os.getcwd())
try:
    from src.database import QuestDBManager
except ImportError:
    print("Could not import QuestDBManager.")
    sys.exit(1)

def test_agg():
    db = QuestDBManager()
    symbol = "BTC/USDT"
    start = "2024-01-01T00:00:00Z"
    end = "2024-01-02T00:00:00Z" # 1 day = 1440 candles
    
    print(f"Fetching OHLCV for {symbol} {start} to {end}...")
    df = db.get_ohlcv(symbol, start, end)
    
    if df.empty:
        print("Error: DataFrame is empty.")
    else:
        print(f"Success! Retrieved {len(df)} candles.")
        print(df.head())
        print(df.tail())

if __name__ == "__main__":
    test_agg()
