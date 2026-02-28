from datetime import datetime, timezone
from unittest.mock import MagicMock

import numpy as np
import pandas as pd

from src.simulation.models import Position, Side
from src.simulation.strategies.multi_divergence import MultiIndicatorDivergenceStrategy


class MockBroker:
    def __init__(self):
        self.balance = 10_000.0
        self.leverage = 2
        self.positions = {}
        self.orders = []
        self.open_orders = {}
        self.trades = []
        self.equity = self.balance

    def get_position(self, symbol):
        return self.positions.get(symbol, Position(symbol, 0.0, 0.0, self.leverage, 0.0, 0.0, 0.0, 0.0))

    def get_available_balance(self):
        return self.balance

    def submit_order(self, order):
        order.id = f"order_{len(self.orders) + 1}"
        order.status = "NEW"
        self.orders.append(order)
        self.open_orders[order.id] = order
        return order

    def cancel_order(self, order_id):
        order = self.open_orders.pop(order_id, None)
        if order:
            order.status = "CANCELED"
            return True
        return False


def _bar(close: float = 100.0):
    return {
        "symbol": "BTC/USDT",
        "timestamp": datetime(2024, 1, 1, tzinfo=timezone.utc),
        "open": close,
        "high": close * 1.01,
        "low": close * 0.99,
        "close": close,
        "volume": 1500.0,
    }


def test_divergence_components_detect_regular_and_hidden():
    s = MultiIndicatorDivergenceStrategy()

    price = np.array([100.0, 95.0], dtype=float)
    indicator = np.array([30.0, 38.0], dtype=float)
    reg, hid = s._evaluate_divergence_components(price, indicator, (0, 1), bullish=True)
    assert reg is True and hid is False

    price_h = np.array([100.0, 105.0], dtype=float)
    indicator_h = np.array([35.0, 30.0], dtype=float)
    reg_h, hid_h = s._evaluate_divergence_components(price_h, indicator_h, (0, 1), bullish=True)
    assert reg_h is False and hid_h is True


def test_unknown_indicator_list_falls_back_to_defaults():
    s = MultiIndicatorDivergenceStrategy(indicators=("foo", "bar"))
    assert s.indicators == MultiIndicatorDivergenceStrategy.DEFAULT_INDICATORS


def test_on_bar_opens_long_when_bullish_confluence_reached():
    s = MultiIndicatorDivergenceStrategy(required_bullish=3, required_bullish_score=3.0, enable_shorts=True)
    s.set_broker(MockBroker())
    s._build_df = MagicMock(return_value=pd.DataFrame({"dummy": [1.0]}))
    s._compute_indicator_frame = MagicMock(side_effect=lambda frame: frame)
    s._divergence_snapshot = MagicMock(
        return_value=(3, 0, 3.4, 0.0, 100.0, 1.2, 25.0, 101.0, 100.0, 2200.0, 1000.0, {})
    )

    s.on_bar(_bar(100.0))
    assert s.broker.orders, "expected a long entry order"
    assert s.broker.orders[-1].side == Side.BUY


def test_on_bar_opens_short_when_bearish_confluence_reached():
    s = MultiIndicatorDivergenceStrategy(required_bearish=3, required_bearish_score=3.0, enable_shorts=True)
    s.set_broker(MockBroker())
    s._build_df = MagicMock(return_value=pd.DataFrame({"dummy": [1.0]}))
    s._compute_indicator_frame = MagicMock(side_effect=lambda frame: frame)
    s._divergence_snapshot = MagicMock(
        return_value=(0, 3, 0.0, 3.6, 100.0, 1.2, 25.0, 99.0, 100.0, 2200.0, 1000.0, {})
    )

    s.on_bar(_bar(100.0))
    assert s.broker.orders, "expected a short entry order"
    assert s.broker.orders[-1].side == Side.SELL
