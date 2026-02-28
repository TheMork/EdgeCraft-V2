from fastapi.testclient import TestClient
from src.api.main import app
import json
from unittest.mock import patch, MagicMock
import pandas as pd

def test_websocket_simulation():
    client = TestClient(app)

    # Mock QuestDBManager in runner
    with patch('src.simulation.runner.QuestDBManager') as MockDBManager:
        mock_db = MockDBManager.return_value
        dates = pd.date_range(start='2024-01-01', periods=5, freq='1min')
        df = pd.DataFrame({
            'open': [100.0] * 5,
            'high': [105.0] * 5,
            'low': [95.0] * 5,
            'close': [102.0] * 5,
            'volume': [1000.0] * 5
        }, index=dates)
        mock_db.get_ohlcv.return_value = df

        with client.websocket_connect("/api/v1/simulation/ws?symbol=BTC/USDT&start_date=2024-01-01&end_date=2024-01-01T00:05:00") as websocket:
            messages = []
            try:
                while True:
                    # TestClient websocket.receive_json() is synchronous
                    data = websocket.receive_json()
                    messages.append(data)
                    if data.get("type") == "status" and data.get("payload") == "simulation_complete":
                        break
            except Exception as e:
                # If disconnect or other error
                pass

            # Verify we received messages
            # Note: We expect 5 data messages + 1 status message
            assert len(messages) >= 6
            assert messages[0]['payload']['symbol'] == 'BTC/USDT'
            # Check if last message is completion status
            assert messages[-1]['type'] == 'status'
            assert messages[-1]['payload'] == 'simulation_complete'
