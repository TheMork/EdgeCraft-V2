
import sys
import os
sys.path.append(os.getcwd())
try:
    from src.database import QuestDBManager
except ImportError:
    print("Could not import QuestDBManager.")
    sys.exit(1)

def truncate():
    db = QuestDBManager()
    print("Truncating ohlcv table...")
    # QuestDB TRUNCATE TABLE ohlcv;
    db._execute_sql("TRUNCATE TABLE ohlcv;")
    print("Truncated.")

if __name__ == "__main__":
    truncate()
