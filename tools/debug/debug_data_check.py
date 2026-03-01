import sys
import os
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))
from src.database import QuestDBManager

def check():
    db = QuestDBManager()
    symbols = ["BTC/USDT", "SOL/USDT", "ADA/USDT", "AVAX/USDT", "LINK/USDT"]
    timeframe = "4h"
    start_date = "2024-01-01T00:00:00"
    end_date = "2026-01-02T00:00:00"
    
    print(f"Checking data for {timeframe} from {start_date} to {end_date}...\n")
    
    for symbol in symbols:
        try:
            df = db.get_ohlcv(symbol, start_date, end_date, timeframe=timeframe)
            if df.empty:
                print(f"{symbol:10}: EMPTY")
            else:
                print(f"{symbol:10}: {len(df):5} rows | {df.index.min()} to {df.index.max()}")
        except Exception as e:
            print(f"{symbol:10}: ERROR: {e}")

if __name__ == "__main__":
    check()
