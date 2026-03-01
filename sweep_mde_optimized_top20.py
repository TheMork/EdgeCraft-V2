import warnings

warnings.warn("This script is deprecated. Use the new SweepEngine via API instead.", DeprecationWarning)

import sys
import os
import pandas as pd
from typing import Dict, List, Any

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))

from src.simulation.optimizer import BacktestOptimizer
from src.simulation.strategies.mde_mad_v2_optimized import MDEMADV2OptimizedStrategy

def main():
    symbols = [
        "BTC/USDT", "ETH/USDT", "BNB/USDT", "SOL/USDT", "XRP/USDT", "DOGE/USDT", 
        "ADA/USDT", "AVAX/USDT", "LINK/USDT", "BCH/USDT", "LTC/USDT"
    ]
    start_date = "2024-01-01T00:00:00"
    end_date = "2025-01-01T00:00:00"
    timeframe = "1h"
    
    # Define the optimized parameters we found
    param_grid = {
        "lookback_bars": [100],
        "risk_aversion": [2.0],
        "entropy_weight": [0.1],
        "turnover_penalty": [0.05],
        "trend_filter_period": [200],
        "enable_shorts": [True],
        "min_rebalance_weight_delta": [0.05],
        "be_trigger_percent": [0.02],
        "trailing_stop_percent": [0.05],
        "timeframe": [timeframe],
        "initial_balance": [10000.0]
    }
    
    all_results = []
    
    for symbol in symbols:
        print(f"\n--- Testing {symbol} ---")
        try:
            optimizer = BacktestOptimizer(MDEMADV2OptimizedStrategy, symbol, start_date, end_date)
            results = optimizer.optimize(param_grid, processes=1)
            
            if results and len(results) > 0:
                r = results[0]["metrics"]
                all_results.append({
                    "Symbol": symbol,
                    "Total Return": f"{r.get('total_return', 0)*100:.2f}%",
                    "Win Rate": f"{r.get('win_rate', 0)*100:.2f}%",
                    "Sharpe": f"{r.get('sharpe_ratio', 0):.2f}",
                    "Max DD": f"{r.get('max_drawdown', 0)*100:.2f}%",
                    "Trades": r.get('total_trades', 0)
                })
        except Exception as e:
            print(f"Failed on {symbol}: {e}")
            
    df = pd.DataFrame(all_results)
    print("\n=== Optimized Portfolio Test Results ===")
    print(df.to_string(index=False))

if __name__ == "__main__":
    main()
