from src.simulation.optimizer import BacktestOptimizer
from src.simulation.strategies.demo import DemoStrategy
from unittest.mock import MagicMock, patch

def verify_optimizer():
    optimizer = BacktestOptimizer(DemoStrategy, 'BTC/USDT', '2024-01-01', '2024-01-02')

    param_grid = {
        'initial_balance': [10000, 20000]
    }

    # Mock multiprocessing.Pool
    with patch('src.simulation.optimizer.multiprocessing.Pool') as MockPool:
        mock_pool_instance = MockPool.return_value
        mock_pool_instance.__enter__.return_value = mock_pool_instance

        # Mock map return value
        mock_pool_instance.map.return_value = [
            {'params': {'initial_balance': 10000}, 'metrics': {'sharpe': 1.0}},
            {'params': {'initial_balance': 20000}, 'metrics': {'sharpe': 2.0}}
        ]

        results = optimizer.optimize(param_grid, processes=2)

        print(f"Results: {results}")
        assert len(results) == 2
        assert results[0]['metrics']['sharpe'] == 1.0

if __name__ == "__main__":
    verify_optimizer()
