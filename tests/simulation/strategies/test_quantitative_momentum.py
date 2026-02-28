import pytest
from src.simulation.strategies.quantitative_momentum import QuantitativeMomentumStrategy
from src.simulation.models import Side, OrderType, Order, Position, Trade
from datetime import datetime, timezone


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
                closed = min(abs(size), qty)
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


def _bar(
    ts=None,
    close=50000.0,
    rsi=55.0,
    rsi_prev=52.0,
    ema50=49500.0,
    ema20=49800.0,
    vwap=49700.0,
    atr=300.0,
    adx=25.0,
    vol=5000.0,
    vol_sma=2000.0,
    regime_4h=48000.0,
    regime_daily=47000.0,
    ready_4h=True,
    ready_daily=True,
    corr_long=True,
    corr_short=False,
):
    if ts is None:
        ts = datetime(2024, 1, 1, 0, 0, tzinfo=timezone.utc)
    return {
        "symbol": "BTC/USDT",
        "timestamp": ts,
        "open": close,
        "high": close * 1.01,
        "low": close * 0.99,
        "close": close,
        "volume": vol,
        "rsi14": rsi,
        "rsi14_prev": rsi_prev,
        "ema50": ema50,
        "ema20": ema20,
        "vwap50": vwap,
        "atr14": atr,
        "adx14": adx,
        "vol_sma20": vol_sma,
        "has_correction_long_prev10": corr_long,
        "has_correction_short_prev10": corr_short,
        "regime_filter_4h": regime_4h,
        "regime_filter_daily": regime_daily,
        "ready_4h": ready_4h,
        "ready_daily": ready_daily,
    }


@pytest.fixture
def strategy():
    s = QuantitativeMomentumStrategy(
        timeframe="1h",
        risk_per_trade=0.01,
        sl_atr_multiplier=2.5,
        trailing_sl_atr_multiplier=1.5,
        adx_threshold=18.0,
        cooldown_bars=3,
        enable_shorts=True,
    )
    s.set_broker(MockBroker())
    return s


def test_long_entry_and_protective_stop(strategy):
    bar = _bar()
    strategy.on_bar(bar)

    assert strategy.broker.orders, "Expected long entry order."
    entry_order = strategy.broker.orders[-1]
    assert entry_order.side == Side.BUY
    assert entry_order.order_type == OrderType.MARKET

    trade = strategy.broker.process_fill(entry_order, bar["close"], bar["timestamp"])
    strategy.on_fill(trade)

    open_stop_orders = [o for o in strategy.broker.open_orders.values() if o.order_type == OrderType.STOP]
    assert open_stop_orders, "Expected protective stop after fill."
    stop = open_stop_orders[-1]
    assert stop.side == Side.SELL
    assert stop.stop_price < bar["close"]


def test_short_entry_and_protective_stop(strategy):
    bar = _bar(
        close=48000.0,
        rsi=45.0,
        rsi_prev=48.0,
        ema50=49000.0,
        ema20=48500.0,
        vwap=48900.0,
        regime_4h=50000.0,
        regime_daily=50500.0,
        corr_long=False,
        corr_short=True,
    )
    strategy.on_bar(bar)

    assert strategy.broker.orders, "Expected short entry order."
    entry_order = strategy.broker.orders[-1]
    assert entry_order.side == Side.SELL
    assert entry_order.order_type == OrderType.MARKET

    trade = strategy.broker.process_fill(entry_order, bar["close"], bar["timestamp"])
    strategy.on_fill(trade)

    open_stop_orders = [o for o in strategy.broker.open_orders.values() if o.order_type == OrderType.STOP]
    assert open_stop_orders, "Expected protective stop after short fill."
    stop = open_stop_orders[-1]
    assert stop.side == Side.BUY
    assert stop.stop_price > bar["close"]


def test_no_double_exit_same_bar(strategy):
    symbol = "BTC/USDT"
    strategy.broker.positions[symbol] = Position(symbol, 2.0, 50000.0, 2, 0.0, 0.0, 0.0, 0.0)
    strategy.trade_state[symbol] = {
        "direction": "long",
        "sl_price": 49000.0,
        "sl_order_id": None,
        "partial_exit_done": False,
        "entry_price": 50000.0,
        "entry_atr": 300.0,
    }

    start_orders = len(strategy.broker.orders)
    bar = _bar(close=49500.0, ema20=50000.0, rsi=85.0, rsi_prev=70.0)
    strategy.on_bar(bar)

    new_orders = strategy.broker.orders[start_orders:]
    assert len(new_orders) == 1
    assert new_orders[0].side == Side.SELL
    assert new_orders[0].order_type == OrderType.MARKET
    assert abs(float(new_orders[0].quantity) - 2.0) < 1e-9


def test_trailing_stop_tightens_and_replaces(strategy):
    symbol = "BTC/USDT"
    strategy.broker.positions[symbol] = Position(symbol, 1.0, 50000.0, 2, 0.0, 0.0, 0.0, 0.0)
    strategy.trade_state[symbol] = {
        "direction": "long",
        "sl_price": 49000.0,
        "sl_order_id": None,
        "partial_exit_done": True,
        "entry_price": 50000.0,
        "entry_atr": 300.0,
    }

    strategy.on_bar(_bar(close=51000.0, ema20=50500.0, atr=200.0, rsi=60.0, rsi_prev=58.0))
    stops_1 = [o for o in strategy.broker.open_orders.values() if o.order_type == OrderType.STOP]
    assert len(stops_1) == 1
    first_stop = stops_1[0]

    strategy.on_bar(_bar(close=52000.0, ema20=51000.0, atr=200.0, rsi=62.0, rsi_prev=60.0))
    stops_2 = [o for o in strategy.broker.open_orders.values() if o.order_type == OrderType.STOP]
    assert len(stops_2) == 1
    second_stop = stops_2[0]

    assert second_stop.id != first_stop.id
    assert second_stop.stop_price > first_stop.stop_price


def test_cooldown_blocks_immediate_reentry(strategy):
    symbol = "BTC/USDT"
    strategy.last_exit_bar[symbol] = 0
    strategy.bar_index = 1

    strategy.on_bar(_bar())
    assert not strategy.broker.orders


def test_entry_leverage_respects_min_max_bounds():
    s = QuantitativeMomentumStrategy(
        timeframe="1h",
        min_leverage=2,
        max_leverage=5,
        adx_threshold=18.0,
        enable_shorts=True,
    )
    broker = MockBroker()
    s.set_broker(broker)

    s.on_bar(_bar(adx=45.0))
    assert broker.orders
    long_order = broker.orders[-1]
    assert 2 <= int(long_order.leverage) <= 5
    assert int(long_order.leverage) == 5

    # Reset to flat for short test.
    broker.orders.clear()
    broker.open_orders.clear()
    broker.positions["BTC/USDT"] = Position("BTC/USDT", 0.0, 0.0, broker.leverage, 0.0, 0.0, 0.0, 0.0)
    s.trade_state.clear()

    s.on_bar(
        _bar(
            close=47000.0,
            rsi=40.0,
            rsi_prev=45.0,
            ema50=49000.0,
            ema20=48000.0,
            vwap=48500.0,
            regime_4h=50000.0,
            regime_daily=50500.0,
            corr_long=False,
            corr_short=True,
            adx=18.0,  # at threshold -> min leverage
        )
    )
    assert broker.orders
    short_order = broker.orders[-1]
    assert int(short_order.leverage) == 2
