
import sys
import os
import pandas as pd
from datetime import datetime
sys.path.append(os.getcwd())
try:
    from src.database import QuestDBManager
    from questdb.ingress import Sender
except ImportError:
    print("Could not import dependencies. Run in venv.")
    sys.exit(1)

def init_db():
    print("Initializing QuestDB...")
    db = QuestDBManager(host='127.0.0.1')
    
    print("Creating tables...")
    try:
        db.create_ohlcv_table()
        db.create_funding_table()
        db.create_trades_table()
        print("Tables created.")
    except Exception as e:
        print(f"Error creating tables: {e}")

    print("Testing insertion (Sender)...")
    try:
        # Create dummy dataframe
        df = pd.DataFrame({
            'timestamp': [datetime.utcnow()],
            'id': ['test_1'],
            'symbol': ['BTC/USDT'],
            'side': ['BUY'],
            'price': [50000.0],
            'amount': [0.1]
        })
        
        # Test insert_trades
        db.insert_trades('BTC/USDT', df)
        print("Insertion successful!")
    except Exception as e:
        print(f"Insertion failed: {e}")
        # Debug Sender if needed
        try:
             print(f"Sender signature test...")
             with Sender('127.0.0.1', 9009) as sender:
                 pass
             print("Sender init successful.")
        except Exception as se:
             print(f"Sender init failed: {se}")

if __name__ == "__main__":
    init_db()
