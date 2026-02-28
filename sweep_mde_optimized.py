import sys
import os
import pandas as pd
from typing import Dict, List, Any

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))

from src.simulation.optimizer import BacktestOptimizer
from src.simulation.strategies.mde_mad_v2_optimized import MDEMADV2OptimizedStrategy

def main():
    symbol = "ETH/USDT"
    start_date = "2024-01-01T00:00:00"
    end_date = "2025-01-01T00:00:00"
    timeframe = "1h"
    
    # Define the parameter grid for the OPTIMIZED version
    param_grid = {
        "lookback_bars": [100],
        "risk_aversion": [2.0],
        "entropy_weight": [0.1],
        "turnover_penalty": [0.05],
        "trend_filter_period": [200],
        "enable_shorts": [True],
        "min_rebalance_weight_delta": [0.015, 0.03, 0.05], # Rebalance Sensitivities
        "be_trigger_percent": [0.015, 0.02], # Break-Even
        "trailing_stop_percent": [0.03, 0.05], # Trailing Stop
        "timeframe": [timeframe],
        "initial_balance": [10000.0]
    }
    
    print(f"Starting Parameter Sweep for OPTIMIZED MDE Strategy on {symbol} ({timeframe})...")
    
    optimizer = BacktestOptimizer(
        MDEMADV2OptimizedStrategy,
        symbol,
        start_date,
        end_date
    )
    
    results = optimizer.optimize(param_grid, processes=8)
    
    processed_results = []
    for r in results:
        row = r["params"].copy()
        metrics = r["metrics"]
        row.update({
            "total_return": metrics.get("total_return", 0),
            "sharpe_ratio": metrics.get("sharpe_ratio", 0),
            "max_drawdown": metrics.get("max_drawdown", 0),
            "total_trades": metrics.get("total_trades", 0),
            "win_rate": metrics.get("win_rate", 0)
        })
        processed_results.append(row)
        
    df = pd.DataFrame(processed_results)
    df = df.sort_values(by="sharpe_ratio", ascending=False)
    
    print("\nOptimized Sweep Results (Sorted by Sharpe Ratio):")
    print(df[['min_rebalance_weight_delta', 'be_trigger_percent', 'trailing_stop_percent', 'total_return', 'sharpe_ratio', 'max_drawdown', 'win_rate', 'total_trades']].head(10).to_string(index=False))

if __name__ == "__main__":
    main()
