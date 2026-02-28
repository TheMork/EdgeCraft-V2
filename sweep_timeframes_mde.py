import sys
import os
import pandas as pd
import numpy as np
from typing import Dict, List, Any

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))

from src.simulation.optimizer import BacktestOptimizer
from src.simulation.strategies.mde_mad_modular import MDEMADModularStrategy

def main():
    symbol = "BTC/USDT"
    # Testing from 2024 to beginning of 2026
    start_date = "2024-01-01T00:00:00"
    end_date = "2026-01-02T00:00:00"
    
    param_grid = {
        "timeframe": ["1h", "4h", "12h", "1d"],
        "use_trend_filter": [True, False],
        "use_turnover_penalty": [True, False],
        "use_dynamic_lookback": [True, False],
        "lookback_bars": [100],
        "risk_aversion": [2.0],
        "entropy_weight": [0.1],
        "turnover_penalty_value": [0.05],
        "trend_filter_period": [200],
        "initial_balance": [10000.0],
        "leverage": [1]
    }
    
    print(f"Starting Multi-Timeframe Feature Sweep for {symbol} (2024-2026)...")
    
    optimizer = BacktestOptimizer(MDEMADModularStrategy, symbol, start_date, end_date)
    # Reducing processes slightly to be safe with memory over longer periods
    results = optimizer.optimize(param_grid, processes=4)
    
    processed = []
    for r in results:
        row = r["params"].copy()
        m = r["metrics"]
        row.update({
            "Return": m.get("total_return", 0),
            "Sharpe": m.get("sharpe_ratio", 0),
            "MaxDD": m.get("max_drawdown", 0),
            "Trades": m.get("total_trades", 0)
        })
        processed.append(row)
        
    df = pd.DataFrame(processed)
    df = df.sort_values(by=["timeframe", "Sharpe"], ascending=[True, False])
    
    print("\nSweep Results (2024-2026) by Timeframe:")
    cols = ["timeframe", "use_trend_filter", "use_turnover_penalty", "use_dynamic_lookback", "Return", "MaxDD", "Sharpe"]
    print(df[cols].to_string(index=False))
    
    df.to_csv("sweep_timeframe_2024_2026_results.csv", index=False)
    
    print("\n--- Feature Impact Analysis (2024-2026) ---")
    for feat in ["use_trend_filter", "use_turnover_penalty", "use_dynamic_lookback"]:
        impact_ret = df.groupby(feat)["Return"].mean()
        impact_dd = df.groupby(feat)["MaxDD"].mean()
        print(f"\nImpact of {feat}:")
        print(f"  Return when True: {impact_ret[True]:.4f}, False: {impact_ret[False]:.4f}")
        print(f"  MaxDD  when True: {impact_dd[True]:.4f}, False: {impact_dd[False]:.4f}")

if __name__ == "__main__":
    main()
