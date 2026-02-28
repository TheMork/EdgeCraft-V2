from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, TYPE_CHECKING
from datetime import datetime
from .models import Order, OrderType, Side, Trade, Position

if TYPE_CHECKING:
    from .broker import Broker

class Strategy(ABC):
    def __init__(self):
        self.broker: Optional['Broker'] = None
        self.equity_curve: List[Dict[str, Any]] = []

    def set_broker(self, broker: 'Broker'):
        self.broker = broker

    @property
    def balance(self) -> float:
        return self.broker.balance if self.broker else 0.0

    @property
    def positions(self) -> Dict[str, Position]:
        return self.broker.positions if self.broker else {}

    @property
    def trades(self) -> List[Trade]:
        return self.broker.trades if self.broker else []

    @abstractmethod
    def on_start(self):
        """Called when the simulation starts."""
        pass

    @abstractmethod
    def on_bar(self, bar: Dict[str, Any]):
        """Called when a new candle/bar is closed."""
        pass

    def on_event(self, event: Any):
        """Generic event handler (optional)."""
        pass

    @abstractmethod
    def on_stop(self):
        """Called when simulation ends."""
        pass

    def on_fill(self, trade: Trade):
        """Called when an order is filled."""
        pass

    def update_equity(self, timestamp: datetime):
        """Called by Runner to record equity curve."""
        if self.broker:
            self.equity_curve.append({
                "timestamp": timestamp,
                "equity": self.broker.equity
            })

    def submit_order(self, order: Order) -> Optional[Order]:
        if self.broker:
            return self.broker.submit_order(order)
        return None

    def cancel_order(self, order_id: str) -> bool:
        if self.broker:
            return self.broker.cancel_order(order_id)
        return False

    def get_position_size(self, symbol: str) -> float:
        """Helper to get position size (quantity) for a symbol."""
        if self.broker:
            pos = self.broker.get_position(symbol)
            return pos.size if pos else 0.0
        return 0.0

    # Backward compatibility wrappers
    def buy(self, symbol: str, quantity: float, price: float, timestamp: datetime, type: OrderType = OrderType.MARKET):
        """
        Submits a Buy Order.
        Note: 'price' is ignored for Market orders in the broker logic,
        but kept for API compatibility.
        """
        order = Order(
            id="",
            symbol=symbol,
            side=Side.BUY,
            order_type=type,
            quantity=quantity,
            price=price if type == OrderType.LIMIT else None,
            timestamp=timestamp
        )
        self.submit_order(order)

    def sell(self, symbol: str, quantity: float, price: float, timestamp: datetime, type: OrderType = OrderType.MARKET):
        """
        Submits a Sell Order.
        """
        order = Order(
            id="",
            symbol=symbol,
            side=Side.SELL,
            order_type=type,
            quantity=quantity,
            price=price if type == OrderType.LIMIT else None,
            timestamp=timestamp
        )
        self.submit_order(order)
