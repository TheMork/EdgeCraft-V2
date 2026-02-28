import pytest
from unittest.mock import MagicMock, patch
import pandas as pd
from src.harvester import DataHarvester

@pytest.fixture
def mock_ccxt():
    with patch('ccxt.binance') as mock:
        yield mock

def test_init(mock_ccxt):
    harvester = DataHarvester()
    assert harvester.exchange_id == 'binance'
    mock_ccxt.assert_called()

def test_fetch_ohlcv(mock_ccxt):
    # Setup mock exchange
    mock_exchange_instance = MagicMock()
    mock_ccxt.return_value = mock_exchange_instance
    mock_exchange_instance.load_markets.return_value = None
    mock_exchange_instance.markets = {'BTC/USDT': {}}

    # Mock parse8601 and milliseconds
    mock_exchange_instance.parse8601.return_value = 1000
    mock_exchange_instance.milliseconds.return_value = 2000

    # Mock fetch_ohlcv response
    # [timestamp, open, high, low, close, volume]
    mock_data = [
        [1000, 100, 105, 95, 102, 10],
        [1060, 102, 107, 100, 105, 12] # Next minute
    ]
    # First call returns data, second call returns empty to stop loop
    mock_exchange_instance.fetch_ohlcv.side_effect = [mock_data, []]

    harvester = DataHarvester()
    chunks = list(harvester.fetch_historical_ohlcv('BTC/USDT', '1m', '2024-01-01'))

    assert len(chunks) == 1
    df = chunks[0]
    assert len(df) == 2
    assert df.iloc[0]['close'] == 102.0

def test_fetch_funding(mock_ccxt):
    mock_exchange_instance = MagicMock()
    mock_ccxt.return_value = mock_exchange_instance
    mock_exchange_instance.load_markets.return_value = None
    mock_exchange_instance.markets = {'BTC/USDT': {}}

    mock_exchange_instance.parse8601.return_value = 1000
    mock_exchange_instance.milliseconds.return_value = 50000

    # Mock fetch_funding_rate_history response
    mock_data = [
        {'timestamp': 1000, 'fundingRate': 0.0001, 'symbol': 'BTC/USDT'},
        {'timestamp': 29800, 'fundingRate': 0.0002, 'symbol': 'BTC/USDT'}
    ]
    # First call returns data, second returns empty
    mock_exchange_instance.fetch_funding_rate_history.side_effect = [mock_data, []]

    harvester = DataHarvester()
    chunks = list(harvester.fetch_funding_rate_history('BTC/USDT', '2024-01-01'))

    assert len(chunks) == 1
    df = chunks[0]
    assert len(df) == 2
    assert df.iloc[0]['fundingRate'] == 0.0001
