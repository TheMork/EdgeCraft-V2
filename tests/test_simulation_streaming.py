import unittest
from unittest.mock import MagicMock, patch
import pandas as pd
from datetime import datetime, timedelta
from src.simulation.runner import SimulationRunner
from src.simulation.strategies.demo import DemoStrategy
from src.simulation.event_loop import EventType

class TestSimulationRunnerStreaming(unittest.TestCase):
    @patch('src.simulation.runner.QuestDBManager')
    def test_run_with_callback(self, MockDBManager):
        # Setup mock data
        mock_db = MockDBManager.return_value
        dates = pd.date_range(start='2024-01-01', periods=5, freq='1min')
        df = pd.DataFrame({
            'open': [100.0] * 5,
            'high': [105.0] * 5,
            'low': [95.0] * 5,
            'close': [102.0] * 5,
            'volume': [1000.0] * 5
        }, index=dates)
        mock_db.get_ohlcv.return_value = df

        # Setup strategy and runner
        strategy = DemoStrategy()
        runner = SimulationRunner(strategy, 'BTC/USDT', '2024-01-01', '2024-01-01T00:05:00')

        # Callback to capture events
        captured_events = []
        def on_event(event):
            captured_events.append(event)

        # Run simulation
        runner.run(on_event_callback=on_event)

        # Verify
        self.assertEqual(len(captured_events), 5)
        self.assertEqual(captured_events[0].type, int(EventType.MARKET_DATA))
        self.assertEqual(captured_events[0].payload['symbol'], 'BTC/USDT')

if __name__ == '__main__':
    unittest.main()
