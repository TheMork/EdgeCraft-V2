import pytest
from datetime import datetime, timezone
from src.simulation.broker import Broker
from src.simulation.models import Order, OrderType, Side, OrderStatus, Position, Trade, Event
from simulation_core import Event as CoreEvent

# Helper to create Event
def create_event(payload, type_id=1):
    return Event(datetime.now(timezone.utc), type_id, payload)

class TestBroker:
    @pytest.fixture
    def broker(self):
        return Broker(initial_balance=10000.0, leverage=1)

    def test_initial_state(self, broker):
        assert broker.balance == 10000.0
        assert broker.equity == 10000.0
        assert broker.positions == {}
        assert broker.open_orders == {}
        assert broker.trades == []

    def test_submit_market_order_buy(self, broker):
        order = Order(
            id="1",
            symbol="BTC/USDT",
            side=Side.BUY,
            order_type=OrderType.MARKET,
            quantity=0.1
        )
        submitted = broker.submit_order(order)
        # Note: Rust uses String for status, so comparison with Enum works if Enum inherits str
        assert submitted.status == OrderStatus.NEW
        # submitted.id might have been generated if empty, but we passed "1"
        assert submitted.id in broker.open_orders
        assert len(broker.open_orders) == 1

    def test_process_market_data_market_buy(self, broker):
        # Submit Order
        order = Order(
            id="1",
            symbol="BTC/USDT",
            side=Side.BUY,
            order_type=OrderType.MARKET,
            quantity=0.1
        )
        broker.submit_order(order)

        # Process Event
        event_payload = {'symbol': 'BTC/USDT', 'close': 50000.0, 'high': 51000.0, 'low': 49000.0}
        event = create_event(event_payload)

        trades = broker.process_market_data(event)

        assert len(trades) == 1
        trade = trades[0]
        assert trade.price == 50000.0 # Filled at Close for Market
        assert trade.quantity == 0.1
        assert trade.side == Side.BUY

        # Check Position
        # broker.get_position returns Py<Position> wrapper
        pos = broker.get_position("BTC/USDT")
        assert pos is not None
        assert pos.size == 0.1
        assert pos.entry_price == 50000.0

    def test_insufficient_margin(self, broker):
        # Limit Order with price 50000, quantity 1.0 -> 50000 USD required
        # Balance 10000. Leverage 1.
        order = Order(
            id="1",
            symbol="BTC/USDT",
            side=Side.BUY,
            order_type=OrderType.LIMIT,
            quantity=1.0,
            price=50000.0
        )
        submitted = broker.submit_order(order)
        assert submitted.status == OrderStatus.REJECTED
        assert len(broker.open_orders) == 0

    def test_position_pnl_update(self, broker):
        # Setup existing position
        pos = Position(symbol="BTC/USDT", size=0.1, entry_price=50000.0, leverage=1)
        broker.add_position(pos)
        broker.balance = 10000.0

        # Process Event with price increase
        event_payload = {'symbol': 'BTC/USDT', 'close': 55000.0}
        event = create_event(event_payload)

        broker.process_market_data(event)

        # Re-fetch position to get updated state (since Python object might be stale if implementation changed,
        # but with Py<Position> it should be updated in place)
        pos = broker.get_position("BTC/USDT")

        # PnL = (55000 - 50000) * 0.1 = 500
        assert pos.unrealized_pnl == 500.0
        # Equity = Balance + PnL
        assert broker.equity == 10000.0 + 500.0

    def test_cancel_order(self, broker):
        order = Order(
            id="1",
            symbol="BTC/USDT",
            side=Side.BUY,
            order_type=OrderType.LIMIT,
            quantity=0.1,
            price=40000.0
        )
        broker.submit_order(order)
        assert len(broker.open_orders) == 1

        success = broker.cancel_order(order.id)
        assert success is True
        assert len(broker.open_orders) == 0

        # Note: Order object status update depends on whether we hold the reference to the object in Rust
        # Rust broker modifies the stored Py<Order>.
        # If 'order' variable here refers to the same object, it is updated.
        assert order.status == OrderStatus.CANCELED

    def test_close_position(self, broker):
        # Open Long 0.1 @ 50000
        pos = Position(symbol="BTC/USDT", size=0.1, entry_price=50000.0, leverage=1)
        broker.add_position(pos)
        broker.balance = 10000.0

        # Sell 0.1 Market
        order = Order(
            id="2",
            symbol="BTC/USDT",
            side=Side.SELL,
            order_type=OrderType.MARKET,
            quantity=0.1
        )
        broker.submit_order(order)

        # Process at 55000
        event_payload = {'symbol': 'BTC/USDT', 'close': 55000.0}
        event = create_event(event_payload)

        trades = broker.process_market_data(event)

        # Trade PnL = (55000 - 50000) * 0.1 = 500
        # Fee = 0.1 * 55000 * 0.0004 = 2.2
        # Balance = 10000 + 500 - 2.2 = 10497.8

        assert len(trades) == 1
        pos = broker.get_position("BTC/USDT")
        assert pos.size == 0
        assert broker.balance == 10497.8
