import itertools
import multiprocessing
from typing import List, Dict, Type, Any
from src.simulation.runner import SimulationRunner
from src.simulation.strategy import Strategy

def run_backtest(args):
    strategy_cls, params, symbol, start_date, end_date = args

    # Extract runner params (metrics related)
    # We copy params to avoid modifying the original dict if reused
    strategy_params = params.copy()
    initial_balance = strategy_params.pop('initial_balance', 10000.0)
    leverage = strategy_params.pop('leverage', 1)
    timeframe = strategy_params.pop('timeframe', '1m')

    # Initialize strategy with remaining params
    try:
        strategy = strategy_cls(**strategy_params)
    except TypeError:
        # Fallback: init default and set attributes
        # This handles cases where strategy doesn't accept all params in __init__
        # but we want to inject them as attributes.
        strategy = strategy_cls()
        for k, v in strategy_params.items():
            setattr(strategy, k, v)

    runner = SimulationRunner(
        strategy, 
        symbol, 
        start_date, 
        end_date, 
        timeframe=timeframe, 
        initial_balance=initial_balance, 
        leverage=leverage
    )
    # Silence stdout/stderr for parallel runs usually, but let's keep it for now or redirect
    result = runner.run()
    return {
        "params": params,
        "metrics": result["metrics"]
    }

class BacktestOptimizer:
    def __init__(self, strategy_cls: Type[Strategy], symbol: str, start_date: str, end_date: str):
        self.strategy_cls = strategy_cls
        self.symbol = symbol
        self.start_date = start_date
        self.end_date = end_date

    def optimize(self, param_grid: Dict[str, List[Any]], processes: int = 4):
        # Generate combinations
        keys = param_grid.keys()
        values = param_grid.values()
        combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]

        # Use multiprocessing
        # Note: QuestDBManager inside Runner might have issues with pickling or connections
        # if not handled correctly. Each process creates its own Runner and QuestDBManager,
        # so it should be fine as long as connection is established in __init__ or run.

        with multiprocessing.Pool(processes=processes) as pool:
            args = [(self.strategy_cls, params, self.symbol, self.start_date, self.end_date) for params in combinations]
            results = pool.map(run_backtest, args)

        return results
