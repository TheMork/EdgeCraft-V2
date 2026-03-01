
import sys
import os
sys.path.append(os.getcwd())
from src.database import QuestDBManager

def test_connection():
    try:
        db = QuestDBManager()
        print(f"Testing connection to {db.host}:{db.rest_port}...")
        # 'show tables' is a valid QuestDB query
        res = db._query_sql("show tables") 
        if res is not None:
             print("Connection successful!")
             print("Response:", res)
        else:
             print("Connection failed (no response)")
    except Exception as e:
        print(f"Connection failed with error: {e}")

if __name__ == "__main__":
    test_connection()
