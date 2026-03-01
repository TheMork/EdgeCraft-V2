import warnings

warnings.warn("This script is deprecated. Use the new SweepEngine via API instead.", DeprecationWarning)

import sys
import os
import pandas as pd
from typing import Dict, List, Any

# Add the project root to sys.path to allow imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))

from src.simulation.optimizer import BacktestOptimizer
from src.simulation.strategies.mde_mad_entropy import MDEMADEntropyStrategy

def main():
    symbol = "BTC/USDT"
    start_date = "2024-01-01T00:00:00"
    end_date = "2025-01-01T00:00:00"
    timeframe = "1h"
    
    # Define the parameter grid for the IMPROVED version
    param_grid = {
        "lookback_bars": [50, 100, 150],
        "risk_aversion": [2.0, 4.0],
        "entropy_weight": [0.1, 0.2],
        "turnover_penalty": [0.01, 0.05, 0.1],
        "trend_filter_period": [200],
        "timeframe": [timeframe],
        "initial_balance": [10000.0],
        "leverage": [1]
    }
    
    print(f"Starting Parameter Sweep for IMPROVED MDE Strategy on {symbol} ({timeframe})...")
    print(f"Filters: EMA 200 Trend Filter & Turnover Penalty")
    
    # Initialize Optimizer
    optimizer = BacktestOptimizer(
        MDEMADEntropyStrategy,
        symbol,
        start_date,
        end_date
    )
    
    # Run optimization
    results = optimizer.optimize(param_grid, processes=8)
    
    # Process results
    processed_results = []
    for r in results:
        row = r["params"].copy()
        metrics = r["metrics"]
        row.update({
            "total_return": metrics.get("total_return", 0),
            "sharpe_ratio": metrics.get("sharpe_ratio", 0),
            "max_drawdown": metrics.get("max_drawdown", 0),
            "total_trades": metrics.get("total_trades", 0)
        })
        processed_results.append(row)
        
    df = pd.DataFrame(processed_results)
    df = df.sort_values(by="sharpe_ratio", ascending=False)
    
    print("\nImproved Sweep Results (Sorted by Sharpe Ratio):")
    print(df.head(20).to_string(index=False))
    
    df.to_csv("sweep_improved_mde_results.csv", index=False)
    print("\nFull results saved to sweep_improved_mde_results.csv")

if __name__ == "__main__":
    main()
