import sys
import os
import pandas as pd
from datetime import datetime

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))
from src.simulation.runner import SimulationRunner
from src.simulation.strategies.demo import DemoStrategy

def test_sync():
    symbol = "ADA/USDT"
    timeframe = "4h"
    start_date = "2024-01-01T00:00:00"
    end_date = "2025-01-01T00:00:00"
    
    print(f"Testing manual runner trigger for {symbol} to check auto-sync...")
    
    strategy = DemoStrategy()
    runner = SimulationRunner(
        strategy=strategy,
        symbol=symbol,
        start_date=start_date,
        end_date=end_date,
        timeframe=timeframe
    )
    
    # This should trigger load_data which triggers auto-sync
    runner.load_data()

if __name__ == "__main__":
    test_sync()
