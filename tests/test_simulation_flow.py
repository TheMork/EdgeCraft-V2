import unittest
from unittest.mock import MagicMock
import pandas as pd
from datetime import datetime, timedelta
from src.simulation.runner import SimulationRunner
from src.simulation.strategy import Strategy
from src.simulation.models import OrderType, Side, Order, OrderStatus

class TestStrategy(Strategy):
    def on_start(self):
        pass

    def on_bar(self, bar):
        # Submit order on first bar if not already active
        if not self.broker.open_orders and not self.broker.positions:
            price = bar['close']
            limit_price = price * 0.95 # Buy 5% below
            # Use specific timestamp
            self.buy("BTC/USDT", 1.0, limit_price, bar['timestamp'], OrderType.LIMIT)

    def on_stop(self):
        pass

class TestSimulationFlow(unittest.TestCase):
    def setUp(self):
        # Setup Mock Data
        self.dates = [datetime(2023, 1, 1, 10, i) for i in range(3)]
        data = {
            'timestamp': self.dates,
            'open': [100.0, 100.0, 92.0],
            'high': [105.0, 100.0, 115.0],
            'low': [98.0, 90.0, 91.0],
            'close': [100.0, 92.0, 110.0],
            'volume': [1000, 1000, 1000]
        }
        self.df = pd.DataFrame(data)
        # Note: In runner.load_data, we iterate rows. QuestDB returns DataFrame.
        # If we return a DataFrame with 'timestamp' as index or column?
        # Runner: `for timestamp, row in df.iterrows():` implies Index is timestamp usually.
        # But logic: `ts = timestamp.to_pydatetime() if isinstance(timestamp, pd.Timestamp) else timestamp`
        # implies index is timestamp.
        self.df.set_index('timestamp', inplace=True)

        self.mock_db = MagicMock()
        self.mock_db.get_ohlcv.return_value = self.df

    def test_limit_order_fill(self):
        strategy = TestStrategy()
        runner = SimulationRunner(strategy, "BTC/USDT", "2023-01-01", "2023-01-02", initial_balance=10000.0)
        runner.db = self.mock_db

        runner.run()
        broker = runner.broker

        # Check Trades
        # Order submitted at Bar 1 (Close 100, Limit 95).
        # Bar 2 Low is 90. Should fill.
        self.assertEqual(len(broker.trades), 1)
        trade = broker.trades[0]
        self.assertEqual(trade.side, Side.BUY)
        self.assertEqual(trade.price, 95.0) # Limit Price
        self.assertEqual(trade.quantity, 1.0)

        # Check Position
        pos = broker.get_position("BTC/USDT")
        self.assertIsNotNone(pos)
        self.assertEqual(pos.size, 1.0)
        self.assertEqual(pos.entry_price, 95.0)

        # Check PnL
        # Last price (Bar 3 Close) is 110. Entry 95. PnL = 15.
        self.assertAlmostEqual(pos.unrealized_pnl, 15.0)

        # Check Equity
        expected_balance = 10000.0 - (1.0 * 95.0 * 0.0002)
        self.assertAlmostEqual(broker.balance, expected_balance)
        self.assertAlmostEqual(broker.equity, expected_balance + 15.0)

    def test_insufficient_margin(self):
        # Strategy that tries to buy too much
        class GreedyStrategy(Strategy):
            def on_start(self): pass
            def on_stop(self): pass
            def on_bar(self, bar):
                if not self.broker.open_orders:
                    # Price 100. Try to buy 200 units (Value 20000). Balance 10000. Leverage 1.
                    # Should fail.
                    self.buy("BTC/USDT", 200.0, 100.0, bar['timestamp'], OrderType.LIMIT)

        strategy = GreedyStrategy()
        runner = SimulationRunner(strategy, "BTC/USDT", "2023-01-01", "2023-01-02", initial_balance=10000.0, leverage=1)
        runner.db = self.mock_db

        runner.run()
        broker = runner.broker

        # Check Order History
        self.assertTrue(len(broker.order_history) > 0)
        order = broker.order_history[0]
        self.assertEqual(order.status, OrderStatus.REJECTED)
        self.assertEqual(len(broker.trades), 0)

    def test_leverage_margin(self):
        # Strategy using Leverage 2x
        class LeverageStrategy(Strategy):
            def on_start(self): pass
            def on_stop(self): pass
            def on_bar(self, bar):
                if not self.broker.open_orders and not self.broker.positions:
                    # Price 100. Buy 150 units (Value 15000). Balance 10000. Leverage 2.
                    # Margin Req = 7500. Should Pass.
                    self.buy("BTC/USDT", 150.0, 95.0, bar['timestamp'], OrderType.LIMIT)

        strategy = LeverageStrategy()
        runner = SimulationRunner(strategy, "BTC/USDT", "2023-01-01", "2023-01-02", initial_balance=10000.0, leverage=2)
        runner.db = self.mock_db

        runner.run()
        broker = runner.broker

        # Check Trades
        self.assertEqual(len(broker.trades), 1)
        pos = broker.get_position("BTC/USDT")
        self.assertEqual(pos.size, 150.0)
        # Margin Used check?
        # Initial Margin = 150 * 95 / 2 = 7125.
        # Used Margin reported by broker logic is based on Entry Price usually.
        # My broker logic: abs(pos.size * pos.entry_price) / pos.leverage
        self.assertAlmostEqual(broker.get_used_margin(), 7125.0)

if __name__ == '__main__':
    unittest.main()
