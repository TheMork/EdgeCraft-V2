
import sys
import os
sys.path.append(os.getcwd())
try:
    from src.database import QuestDBManager
except ImportError:
    print("Could not import QuestDBManager.")
    sys.exit(1)

def check_counts():
    db = QuestDBManager()
    print("Checking table counts...")
    
    trades_count = db._query_sql("SELECT count() FROM trades")
    ohlcv_count = db._query_sql("SELECT count() FROM ohlcv")
    
    print(f"Trades count: {trades_count['dataset'][0][0] if trades_count and trades_count['dataset'] else 'Error'}")
    print(f"OHLCV count: {ohlcv_count['dataset'][0][0] if ohlcv_count and ohlcv_count['dataset'] else 'Error'}")

if __name__ == "__main__":
    check_counts()
