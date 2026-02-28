from datetime import datetime, timedelta, timezone

import pytest

from src.simulation.models import OrderType, Position, Side, Trade
from src.simulation.strategies.double_rsi_divergence import DoubleRSIDivergenceStrategy


class MockBroker:
    def __init__(self):
        self.balance = 10000.0
        self.leverage = 2
        self.positions = {}
        self.orders = []
        self.open_orders = {}
        self.trades = []
        self.equity = 10000.0

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

    def process_fill(self, order, price, timestamp):
        self.open_orders.pop(order.id, None)
        order.status = "FILLED"
        side = getattr(order.side, "value", str(order.side))

        trade = Trade(
            id=f"trade_{len(self.trades) + 1}",
            order_id=order.id,
            symbol=order.symbol,
            side=order.side,
            quantity=order.quantity,
            price=price,
            timestamp=timestamp,
            fee=0.0,
            pnl=0.0,
        )
        self.trades.append(trade)

        pos = self.get_position(order.symbol)
        size = float(pos.size)
        entry = float(pos.entry_price)
        qty = float(order.quantity)

        if side == "BUY":
            if size >= 0:
                new_size = size + qty
                new_entry = ((size * entry) + (qty * price)) / new_size if new_size > 0 else 0.0
            else:
                remaining = size + qty
                if remaining > 0:
                    new_size = remaining
                    new_entry = price
                else:
                    new_size = remaining
                    new_entry = entry if remaining < 0 else 0.0
        else:
            if size <= 0:
                new_size = size - qty
                new_entry = ((abs(size) * entry) + (qty * price)) / abs(new_size) if abs(new_size) > 0 else 0.0
            else:
                remaining = size - qty
                if remaining < 0:
                    new_size = remaining
                    new_entry = price
                else:
                    new_size = remaining
                    new_entry = entry if remaining > 0 else 0.0

        self.positions[order.symbol] = Position(
            order.symbol,
            float(new_size),
            float(new_entry),
            self.leverage,
            0.0,
            0.0,
            0.0,
            0.0,
        )
        return trade


def _bar(ts, open_, high, low, close, rsi, atr=2.0):
    return {
        "symbol": "BTC/USDT",
        "timestamp": ts,
        "open": open_,
        "high": high,
        "low": low,
        "close": close,
        "volume": 1000.0,
        "rsi14": rsi,
        "atr14": atr,
    }


@pytest.fixture
def strategy():
    s = DoubleRSIDivergenceStrategy(
        timeframe="1h",
        pivot_lookback=1,
        min_pivot_separation_bars=2,
        min_rsi_delta=1.0,
        cooldown_bars=0,
        enable_shorts=True,
        use_structure_break_trigger=False,
        enable_regime_filter=False,
        min_leverage=1,
        max_leverage=5,
    )
    s.set_broker(MockBroker())
    return s


def test_bearish_double_divergence_enters_short_and_places_stop(strategy):
    base = datetime(2024, 1, 1, 0, 0, tzinfo=timezone.utc)
    bars = [
        _bar(base + timedelta(hours=0), 94.0, 95.0, 90.0, 93.0, 60.0),
        _bar(base + timedelta(hours=1), 98.0, 100.0, 95.0, 99.0, 70.0),  # H1
        _bar(base + timedelta(hours=2), 94.0, 96.0, 91.0, 93.0, 58.0),
        _bar(base + timedelta(hours=3), 101.0, 105.0, 97.0, 104.0, 65.0),  # H2
        _bar(base + timedelta(hours=4), 96.0, 97.0, 92.0, 95.0, 57.0),
        _bar(base + timedelta(hours=5), 107.0, 108.0, 103.0, 106.0, 60.0),  # H3
        _bar(base + timedelta(hours=6), 105.0, 106.0, 100.0, 102.0, 54.0),  # bearish trigger
    ]

    for bar in bars:
        strategy.on_bar(bar)

    assert strategy.broker.orders, "Expected short entry order."
    entry = strategy.broker.orders[-1]
    assert entry.side == Side.SELL
    assert entry.order_type == OrderType.MARKET

    trade = strategy.broker.process_fill(entry, bars[-1]["close"], bars[-1]["timestamp"])
    strategy.on_fill(trade)

    stop_orders = [o for o in strategy.broker.open_orders.values() if o.order_type == OrderType.STOP]
    assert stop_orders, "Expected protective stop after short fill."
    stop = stop_orders[-1]
    assert stop.side == Side.BUY
    assert float(stop.stop_price) > bars[-1]["close"]


def test_bullish_double_divergence_enters_long_and_places_stop(strategy):
    base = datetime(2024, 1, 2, 0, 0, tzinfo=timezone.utc)
    bars = [
        _bar(base + timedelta(hours=0), 109.0, 111.0, 105.0, 108.0, 45.0),
        _bar(base + timedelta(hours=1), 103.0, 105.0, 100.0, 101.0, 30.0),  # T1
        _bar(base + timedelta(hours=2), 108.0, 110.0, 106.0, 107.0, 34.0),
        _bar(base + timedelta(hours=3), 97.0, 99.0, 95.0, 96.0, 35.0),  # T2
        _bar(base + timedelta(hours=4), 104.0, 106.0, 102.0, 103.0, 36.0),
        _bar(base + timedelta(hours=5), 92.0, 94.0, 90.0, 91.0, 40.0),  # T3
        _bar(base + timedelta(hours=6), 92.0, 97.0, 91.0, 96.0, 44.0),  # bullish trigger
    ]

    for bar in bars:
        strategy.on_bar(bar)

    assert strategy.broker.orders, "Expected long entry order."
    entry = strategy.broker.orders[-1]
    assert entry.side == Side.BUY
    assert entry.order_type == OrderType.MARKET

    trade = strategy.broker.process_fill(entry, bars[-1]["close"], bars[-1]["timestamp"])
    strategy.on_fill(trade)

    stop_orders = [o for o in strategy.broker.open_orders.values() if o.order_type == OrderType.STOP]
    assert stop_orders, "Expected protective stop after long fill."
    stop = stop_orders[-1]
    assert stop.side == Side.SELL
    assert float(stop.stop_price) < bars[-1]["close"]
