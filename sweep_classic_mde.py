import warnings

warnings.warn("This script is deprecated. Use the new SweepEngine via API instead.", DeprecationWarning)

import sys
import os
import pandas as pd
from typing import Dict, List, Any

# Add the project root to sys.path to allow imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))

from src.simulation.optimizer import BacktestOptimizer
from src.simulation.strategies.classic_mde_mad_entropy import ClassicMDEMADEntropyStrategy

def main():
    symbol = "BTC/USDT"
    start_date = "2024-01-01T00:00:00"
    end_date = "2025-01-01T00:00:00"
    timeframe = "1h"
    
    # Define the parameter grid
    param_grid = {
        "lookback_bars": [20, 30, 50, 100],
        "risk_aversion": [1.0, 2.0, 3.0],
        "entropy_weight": [0.01, 0.1, 0.5],
        "timeframe": [timeframe],  # Pass timeframe through init
        "initial_balance": [10000.0],
        "leverage": [1]
    }
    
    print(f"Starting Parameter Sweep for {symbol} ({timeframe})...")
    print(f"Range: {start_date} to {end_date}")
    
    # Initialize Optimizer
    optimizer = BacktestOptimizer(
        ClassicMDEMADEntropyStrategy,
        symbol,
        start_date,
        end_date
    )
    
    # Run optimization (using a pool of processes)
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
    
    # Sort by Sharpe Ratio (descending)
    df = df.sort_values(by="sharpe_ratio", ascending=False)
    
    print("\nSweep Results (Sorted by Sharpe Ratio):")
    print(df.head(20).to_string(index=False))
    
    # Save results to CSV
    df.to_csv("sweep_classic_mde_results.csv", index=False)
    print("\nFull results saved to sweep_classic_mde_results.csv")

if __name__ == "__main__":
    main()
