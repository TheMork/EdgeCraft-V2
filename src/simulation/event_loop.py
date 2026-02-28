import simulation_core
from dataclasses import dataclass, field
from enum import IntEnum, auto
from typing import Any, Optional, List
from datetime import datetime

class EventType(IntEnum):
    # Lower value = Higher Priority
    MARKET_DATA = 0  # Process market updates first
    FUNDING = 1      # Process funding payments
    FILL = 2         # Process order fills
    ORDER = 3        # Process new orders
    SIGNAL = 4       # Strategy signals (process last for the same timestamp)

# Alias Rust implementations
Event = simulation_core.Event
EventLoop = simulation_core.EventLoop

# Simple implementation of components to demonstrate interaction
class SynchronizationBuffer:
    """
    Ensures that for a given timestamp, all market data events
    are processed before any strategy logic is triggered.
    This is implicitly handled by the EventLoop priority queue
    (MarketData priority < Signal priority).
    """
    pass
