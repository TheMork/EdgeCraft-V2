
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

def insert_dummy_2024():
    db = QuestDBManager()
    timestamp = datetime(2024, 1, 1, 0, 0, 0)
    
    # TRADES
    df = pd.DataFrame({
        'timestamp': [timestamp],
        'id': ['dummy_2024_1'],
        'symbol': ['BTC/USDT'],
        'side': ['BUY'],
        'price': [45000.0],
        'amount': [1.0]
    })
    try:
        db.insert_trades('BTC/USDT', df)
        print("Inserted dummy trade for 2024-01-01.")
    except Exception as e:
        print(f"Failed to insert trades: {e}")

    # OHLCV
    df_ohlcv = pd.DataFrame({
        'timestamp': [timestamp],
        'open': [45000.0],
        'high': [45100.0],
        'low': [44900.0],
        'close': [45050.0],
        'volume': [100.0]
    })
    try:
        db.insert_ohlcv('BTC/USDT', df_ohlcv)
        print("Inserted dummy OHLCV for 2024-01-01.")
    except Exception as e:
        print(f"Failed to insert OHLCV: {e}")

if __name__ == "__main__":
    insert_dummy_2024()
