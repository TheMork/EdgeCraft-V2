import unittest
from src.simulation.parallel_runner import ParallelSimulationRunner
from src.simulation.strategy import Strategy
from src.simulation.models import OrderType

class SimpleStrategy(Strategy):
    def on_start(self): pass
    def on_stop(self): pass
    def on_bar(self, bar):
        # Just buy once
        if not self.broker.positions:
            self.buy("BTC/USDT", 0.1, bar['close'], bar['timestamp'], OrderType.MARKET)

class TestParallelRunner(unittest.TestCase):
    def test_run_parallel(self):
        # Config
        runner = ParallelSimulationRunner(
            SimpleStrategy,
            "BTC/USDT",
            "2023-01-01",
            "2023-01-02",
            db_host='mock'
        )

        # We vary leverage, but SimpleStrategy ignores it (broker handles it)
        # However, run_simulation_task sets leverage in SimulationRunner
        params = [
            {'leverage': 1},
            {'leverage': 2}
        ]

        results = runner.run_parallel(params, processes=2)

        self.assertEqual(len(results), 2)
        for res in results:
            self.assertIn('metrics', res)
            self.assertIn('parameters', res)
            # Check metrics exist
            metrics = res['metrics']
            # We expect some trades
            self.assertGreater(metrics.get('total_trades', 0), 0)

if __name__ == '__main__':
    unittest.main()
