from datetime import datetime, timedelta, timezone

import numpy as np

from src.simulation.models import Position, Side
from src.simulation.strategies.pair_arbitrage import PairArbitrageStrategy


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


def _make_bar(ts, close):
    return {
        "symbol": "ETH/USDT",
        "timestamp": ts,
        "open": close,
        "high": close * 1.001,
        "low": close * 0.999,
        "close": close,
        "volume": 1000.0,
    }


def _build_series(bias: str):
    base = datetime(2024, 1, 1, tzinfo=timezone.utc)
    ref = []
    asset = []
    for i in range(120):
        ref_close = 100.0 + 0.1 * i
        premium = 1.5
        if 45 <= i <= 58:
            premium += (i - 45) * 0.18 if bias == "sell" else -(i - 45) * 0.18
        if 59 <= i <= 70:
            premium += 2.3 - (i - 59) * 0.32 if bias == "sell" else -2.3 + (i - 59) * 0.32
        close = ref_close + premium + 0.03 * np.sin(i / 3)
        ts = base + timedelta(hours=i)
        ref.append((ts, ref_close))
        asset.append(_make_bar(ts, float(close)))
    return ref, asset


def test_pair_arbitrage_v1_generates_short_entry_signal():
    ref, bars = _build_series("sell")
    strategy = PairArbitrageStrategy(
        timeframe="1h",
        variant="v1",
        ols_window=20,
        z_window=16,
        z_entry=0.9,
        cooldown_bars=0,
        enable_shorts=True,
        min_leverage=1,
        max_leverage=3,
    )
    strategy.set_broker(MockBroker())
    strategy.set_reference_series(dict(ref))
    strategy.on_start()

    for bar in bars:
        strategy.on_bar(bar)

    assert strategy.broker.orders, "expected at least one order"
    assert any(order.side == Side.SELL for order in strategy.broker.orders), "expected a short entry order"


def test_pair_arbitrage_v4_generates_signals_and_orders():
    ref, bars = _build_series("buy")
    strategy = PairArbitrageStrategy(
        timeframe="1h",
        variant="v4",
        z_window=8,
        z_entry=0.2,
        ou_window=30,
        ou_theta_min=0.0,
        v4_min_entry_z=0.2,
        v4_stop_z=6.0,
        v4_mean_exit_z=0.4,
        v4_trend_hurst_min=2.0,
        cooldown_bars=0,
        enable_shorts=True,
        min_leverage=1,
        max_leverage=3,
    )
    strategy.set_broker(MockBroker())
    strategy.set_reference_series(dict(ref))
    strategy.on_start()

    for bar in bars:
        strategy.on_bar(bar)

    assert len(strategy.z_scores) > 20, "expected v4 to compute z-scores"
    assert strategy.broker.orders, "expected at least one order from v4 signal path"


def test_pair_arbitrage_v1_has_no_implicit_time_stop():
    strategy = PairArbitrageStrategy(
        timeframe="1h",
        variant="v1",
        cooldown_bars=0,
        enable_shorts=True,
    )
    strategy.set_broker(MockBroker())
    strategy._schedule_action("OPEN_LONG", signal_z=1.0, max_hold_bars=None)
    strategy._execute_pending_action("ETH/USDT", datetime(2024, 1, 1, tzinfo=timezone.utc), exec_price=100.0)
    assert strategy.broker.orders, "expected entry order"
    assert strategy.active_max_hold_bars is None, "v1 should not enforce implicit time stop"
