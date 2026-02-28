from datetime import datetime, timezone
from unittest.mock import MagicMock

from src.data_manager import SyncManager


def _make_sync_manager_stub() -> SyncManager:
    manager = object.__new__(SyncManager)
    manager.db = MagicMock()
    manager.db.is_available.return_value = True
    manager.db.create_trades_table.return_value = None
    manager.db.create_ohlcv_table.return_value = None
    manager.db.create_all_ohlcv_tables.return_value = None
    manager.db.parse_timeframe.side_effect = lambda tf: tf
    manager.ORDERED_TIMEFRAMES = ["1m", "1h", "4h"]
    manager._sync_candles_only = MagicMock()
    return manager


def test_sync_data_candles_all_routes_to_all_timeframes_sync():
    manager = _make_sync_manager_stub()
    manager.sync_all_candles = MagicMock()

    manager.sync_data(
        symbol="BTC/USDT",
        start_date_str="2024-01-01T00:00:00Z",
        end_date_str="2024-01-02T00:00:00Z",
        sync_mode="candles_all",
        timeframe="1m",
    )

    assert manager.sync_all_candles.call_count == 1
    args, kwargs = manager.sync_all_candles.call_args
    assert args[0] == "BTC/USDT"
    assert isinstance(args[1], datetime) and args[1].utcoffset() == timezone.utc.utcoffset(args[1])
    assert isinstance(args[2], datetime) and args[2].utcoffset() == timezone.utc.utcoffset(args[2])
    manager.db.create_all_ohlcv_tables.assert_called_once()


def test_sync_top20_candles_all_forwards_mode_and_progress():
    manager = _make_sync_manager_stub()
    manager.get_top_20_symbols = MagicMock(return_value=["BTC/USDT", "ETH/USDT"])

    calls = []

    def fake_sync_data(symbol, start_date_str, end_date_str=None, sync_mode="trades", timeframe="1m", progress_callback=None):
        calls.append((symbol, sync_mode, timeframe))
        if progress_callback:
            progress_callback(50, "half")

    manager.sync_data = fake_sync_data

    progress_events = []

    manager.sync_top_20(
        start_date_str="2024-01-01T00:00:00Z",
        end_date_str="2024-01-02T00:00:00Z",
        sync_mode="candles_all",
        timeframe="1m",
        progress_callback=lambda p, m: progress_events.append((p, m)),
    )

    assert calls == [
        ("BTC/USDT", "candles_all", "1m"),
        ("ETH/USDT", "candles_all", "1m"),
    ]
    assert any("BTC/USDT" in message for _, message in progress_events)
    assert any("ETH/USDT" in message for _, message in progress_events)
