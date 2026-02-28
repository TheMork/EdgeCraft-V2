import itertools
from typing import List, Dict, Any, Type, Optional
from src.simulation.strategy import Strategy
from src.simulation.parallel_runner import ParallelSimulationRunner
import pandas as pd

class GridSearchOptimizer:
    def __init__(self, strategy_cls: Type[Strategy], symbol: str, start_date: str, end_date: str, initial_balance: float = 10000.0, db_host: str = 'localhost'):
        self.strategy_cls = strategy_cls
        self.symbol = symbol
        self.start_date = start_date
        self.end_date = end_date
        self.initial_balance = initial_balance
        self.db_host = db_host

    def optimize(self, param_grid: Dict[str, List[Any]], target_metric: str = 'total_return', processes: int = 4) -> Dict[str, Any]:
        """
        Runs grid search optimization.
        param_grid: {'ma_window': [10, 20, 50], 'leverage': [1, 2]}
        target_metric: Metric to maximize (e.g., 'total_return', 'sharpe_ratio').
        """
        # Generate all combinations
        keys = param_grid.keys()
        values = param_grid.values()
        combinations = list(itertools.product(*values))

        param_list = [dict(zip(keys, v)) for v in combinations]

        runner = ParallelSimulationRunner(
            self.strategy_cls,
            self.symbol,
            self.start_date,
            self.end_date,
            initial_balance=self.initial_balance,
            db_host=self.db_host
        )

        results = runner.run_parallel(param_list, processes=processes)

        # Find best
        best_result = None
        best_metric_value = -float('inf')

        all_results = []

        for res in results:
            metrics = res['metrics']
            val = metrics.get(target_metric, 0.0)

            # Handle NaN/None
            if val is None or pd.isna(val):
                val = -float('inf')

            # For Drawdown, usually we want to minimize (it's negative number?).
            # In metrics.py: max_drawdown is calculated as (equity - peak) / peak. It's negative (e.g., -0.20).
            # So maximizing -0.20 (-20%) vs -0.50 (-50%) prefers -0.20. Correct.

            res['optimization_score'] = val
            all_results.append(res)

            if val > best_metric_value:
                best_metric_value = val
                best_result = res

        return {
            "best_parameters": best_result['parameters'] if best_result else None,
            "best_metrics": best_result['metrics'] if best_result else None,
            "all_results": all_results
        }
