
import sys
import os
import pandas as pd
from datetime import datetime
sys.path.append(os.getcwd())
try:
    from src.database import QuestDBManager
except ImportError:
    print("Could not import QuestDBManager.")
    sys.exit(1)

def test_frontend_query():
    db = QuestDBManager()
    symbol = "BTC/USDT"
    # Exact format from user request
    start = "2024-01-01T00:00:00" 
    end = "2024-01-02T00:00:00"
    
    print(f"Testing query with Naive format: {start} to {end}")
    
    try:
        df = db.get_ohlcv(symbol, start, end)
        if df.empty:
            print("Result: EMPTY DataFrame")
        else:
            print(f"Result: {len(df)} rows")
            print(df.head(1))
    except Exception as e:
        print(f"Query failed: {e}")

if __name__ == "__main__":
    test_frontend_query()
