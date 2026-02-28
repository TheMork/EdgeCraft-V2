import unittest
from src.optimization.optimizer import GridSearchOptimizer
from src.simulation.strategy import Strategy
from src.simulation.models import OrderType

class TestStrategy(Strategy):
    def __init__(self, leverage=1, buy_factor=1.0):
        super().__init__()
        self.leverage = leverage
        self.buy_factor = buy_factor

    def on_start(self): pass
    def on_stop(self): pass
    def on_bar(self, bar):
        # Use buy_factor to decide quantity
        if not self.broker.positions:
            self.buy("BTC/USDT", 0.1 * self.buy_factor, bar['close'], bar['timestamp'], OrderType.MARKET)

class TestOptimizer(unittest.TestCase):
    def test_grid_search(self):
        optimizer = GridSearchOptimizer(
            TestStrategy,
            "BTC/USDT",
            "2023-01-01",
            "2023-01-02",
            db_host='mock'
        )

        param_grid = {
            'leverage': [1, 2],
            'buy_factor': [1.0, 2.0]
        }

        results = optimizer.optimize(param_grid, processes=2)

        self.assertEqual(len(results['all_results']), 4)
        self.assertIsNotNone(results['best_parameters'])
        self.assertIsNotNone(results['best_metrics'])

        params_seen = [r['parameters'] for r in results['all_results']]
        self.assertTrue({'leverage': 1, 'buy_factor': 1.0} in params_seen)
        self.assertTrue({'leverage': 2, 'buy_factor': 2.0} in params_seen)

if __name__ == '__main__':
    unittest.main()
