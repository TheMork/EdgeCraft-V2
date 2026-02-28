import pytest
import subprocess
import time
import requests
import pandas as pd
import shutil
from datetime import datetime
from src.database import QuestDBManager

@pytest.fixture(scope="module")
def questdb_service():
    """Starts QuestDB via Docker for integration testing."""
    if not shutil.which("docker-compose"):
        pytest.skip("docker-compose not found")

    try:
        # Check if docker is running
        subprocess.run(["docker", "ps"], check=True, stdout=subprocess.DEVNULL)

        # Start compose
        print("Starting QuestDB container...")
        subprocess.run(["docker-compose", "up", "-d"], check=True)

        # Wait for healthcheck (port 9000)
        start_time = time.time()
        while time.time() - start_time < 30:
            try:
                requests.get("http://localhost:9000")
                print("QuestDB is up!")
                break
            except requests.ConnectionError:
                time.sleep(1)
        else:
            pytest.fail("QuestDB failed to start in 30 seconds")

        yield "http://localhost:9000"

    except (subprocess.CalledProcessError, FileNotFoundError):
        pytest.skip("Docker not available, skipping integration tests")
    finally:
        subprocess.run(["docker-compose", "down"], check=False)

def test_db_manager_integration(questdb_service):
    # Connect to the service
    db = QuestDBManager() # defaults are localhost

    # 1. Create tables
    db.create_ohlcv_table()
    db.create_funding_table()

    # 2. Insert Data
    ts = pd.Timestamp.now()
    df_ohlcv = pd.DataFrame([{
        'timestamp': ts,
        'open': 100.0, 'high': 105.0, 'low': 95.0, 'close': 102.0, 'volume': 500.0
    }])

    db.insert_ohlcv('BTC-TEST', df_ohlcv)

    # 3. Verify via REST API
    # ILP is async and eventual consistent (usually fast). Wait a bit.
    time.sleep(2)

    query = "SELECT * FROM ohlcv WHERE symbol = 'BTC-TEST'"
    resp = requests.get(f"{questdb_service}/exec", params={'query': query})
    assert resp.status_code == 200
    data = resp.json()

    # If table exists but empty, 'dataset' might be empty
    assert 'dataset' in data
    assert len(data['dataset']) >= 1
    # Check symbol - column 0 is symbol based on create statement
    # create statement: symbol, open, high, low, close, volume, timestamp
    # select * usually returns in order
    assert data['dataset'][0][0] == 'BTC-TEST'

def test_funding_integration(questdb_service):
    db = QuestDBManager()
    db.create_funding_table()

    ts = pd.Timestamp.now()
    df_fund = pd.DataFrame([{
        'timestamp': ts,
        'fundingRate': 0.0001
    }])

    db.insert_funding('BTC-TEST', df_fund)
    time.sleep(2)

    query = "SELECT * FROM funding_rates WHERE symbol = 'BTC-TEST'"
    resp = requests.get(f"{questdb_service}/exec", params={'query': query})
    assert resp.status_code == 200
    data = resp.json()
    assert len(data['dataset']) >= 1
