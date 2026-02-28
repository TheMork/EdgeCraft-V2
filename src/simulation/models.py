from enum import Enum, auto
from datetime import datetime, timezone
from typing import Optional
import simulation_core

# Re-export Rust classes
Order = simulation_core.Order
Trade = simulation_core.Trade
Position = simulation_core.Position
Event = simulation_core.Event # Also exported here for convenience

class Side(str, Enum):
    BUY = "BUY"
    SELL = "SELL"

class OrderType(str, Enum):
    MARKET = "MARKET"
    LIMIT = "LIMIT"
    STOP = "STOP"
    STOP_LIMIT = "STOP_LIMIT"
    TRAILING_STOP = "TRAILING_STOP"

class OrderStatus(str, Enum):
    NEW = "NEW"
    PARTIALLY_FILLED = "PARTIALLY_FILLED"
    FILLED = "FILLED"
    CANCELED = "CANCELED"
    REJECTED = "REJECTED"
    EXPIRED = "EXPIRED"
