from src.simulation.runner import SimulationRunner
from src.simulation.strategies.demo import DemoStrategy
from unittest.mock import MagicMock
import pandas as pd
from datetime import datetime, timedelta

def verify_oms():
    # Mock data
    dates = pd.date_range(start='2024-01-01', periods=10, freq='h')
    data = {
        'open': [100, 101, 102, 103, 102, 101, 100, 99, 98, 97],
        'high': [105] * 10,
        'low': [95] * 10,
        'close': [101, 102, 103, 102, 101, 100, 99, 98, 97, 96],
        'volume': [1000] * 10
    }
    df = pd.DataFrame(data, index=dates)

    strategy = DemoStrategy()
    runner = SimulationRunner(strategy, 'BTC/USDT', '2024-01-01', '2024-01-02', initial_balance=10000.0)

    # Mock DB
    runner.db.get_ohlcv = MagicMock(return_value=df)

    runner.run()

    print(f"Trades: {len(strategy.trades)}")
    print(f"Equity Curve: {len(strategy.equity_curve)}")
    print(f"Final Balance: {strategy.balance}")

    for trade in strategy.trades:
        print(trade)

    assert len(strategy.trades) > 0
    assert len(strategy.equity_curve) == 10

if __name__ == "__main__":
    verify_oms()
