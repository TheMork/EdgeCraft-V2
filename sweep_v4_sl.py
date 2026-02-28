import sys
import os
import pandas as pd
from typing import Dict, List, Any

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))

from src.simulation.optimizer import BacktestOptimizer
from src.simulation.strategies.mde_mad_v4_sl_test import MDEMADV4SLTestStrategy

def main():
    symbol = "BTC/USDT"
    start_date = "2024-01-01T00:00:00"
    end_date = "2026-01-02T00:00:00"
    timeframe = "4h"
    
    param_grid = {
        "timeframe": [timeframe],
        "lookback_bars": [100],
        "risk_aversion": [2.5],
        "entropy_weight": [0.1],
        "turnover_penalty": [0.1],
        "sl_atr_multiplier": [0.5, 1.0, 2.0, 3.0, 5.0],
        "enable_shorts": [True],
        "initial_balance": [10000.0],
        "leverage": [1]
    }
    
    print(f"Starting V4 Stop-Loss Sweep for {symbol} (2024-2026)...")
    
    optimizer = BacktestOptimizer(MDEMADV4SLTestStrategy, symbol, start_date, end_date)
    results = optimizer.optimize(param_grid, processes=5)
    
    processed = []
    for r in results:
        row = r["params"].copy()
        m = r["metrics"]
        
        # Check if any trades were STOP orders
        # Note: Need to check result["trades"] but it's not in the run_backtest return currently.
        # Let's just output the metrics for now.
        
        row.update({
            "Return": m.get("total_return", 0),
            "Sharpe": m.get("sharpe_ratio", 0),
            "MaxDD": m.get("max_drawdown", 0),
            "Trades": m.get("total_trades", 0)
        })
        processed.append(row)
        
    df = pd.DataFrame(processed)
    df = df.sort_values(by="Sharpe", ascending=False)
    
    print("\nV4 Stop-Loss Sweep Results:")
    cols = ["sl_atr_multiplier", "Return", "MaxDD", "Sharpe", "Trades"]
    print(df[cols].to_string(index=False))
    
    df.to_csv("sweep_v4_sl_results.csv", index=False)
    print("\nResults saved to sweep_v4_sl_results.csv")

if __name__ == "__main__":
    main()
