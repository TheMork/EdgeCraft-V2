import pytest
from datetime import datetime, timedelta
from src.simulation.event_loop import EventLoop, Event, EventType

class TestEventLoop:
    def test_priority_processing(self):
        loop = EventLoop()
        processed = []

        def handler(event):
            processed.append(event.type)

        loop.subscribe(EventType.MARKET_DATA, handler)
        loop.subscribe(EventType.SIGNAL, handler)

        t1 = datetime(2023, 1, 1, 12, 0, 0)

        # Add Signal first, then Market Data (same timestamp)
        loop.add_event(Event(t1, EventType.SIGNAL, "Signal"))
        loop.add_event(Event(t1, EventType.MARKET_DATA, "Market"))

        loop.run()

        # Market Data should be processed before Signal because of priority
        assert processed == [EventType.MARKET_DATA, EventType.SIGNAL]

    def test_latency_simulation(self):
        loop = EventLoop(latency_ms=100)
        processed_times = []

        def handler(event):
            processed_times.append(event.timestamp)

        loop.subscribe(EventType.ORDER, handler)

        t1 = datetime(2023, 1, 1, 12, 0, 0)
        event = Event(t1, EventType.ORDER, "Order")

        # Schedule with delay
        loop.schedule_delayed(event)

        loop.run()

        assert len(processed_times) == 1
        expected_time = t1 + timedelta(milliseconds=100)
        assert processed_times[0] == expected_time

    def test_time_travel_protection(self):
        loop = EventLoop()
        t1 = datetime(2023, 1, 1, 12, 0, 0)
        t2 = datetime(2023, 1, 1, 11, 0, 0) # Back in time

        # Process first event to set time
        loop.add_event(Event(t1, EventType.MARKET_DATA, "Tick 1"))
        loop.run()

        # Add past event
        loop.add_event(Event(t2, EventType.MARKET_DATA, "Tick 2"))

        with pytest.raises(ValueError, match="Time travel detected"):
            loop.run()
