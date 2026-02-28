import multiprocessing
from typing import List, Dict, Any, Type
from src.simulation.runner import SimulationRunner
from src.simulation.strategy import Strategy

def run_simulation_task(config: Dict[str, Any], params: Dict[str, Any]) -> Dict[str, Any]:
    strategy_cls = config['strategy_cls']
    try:
        strategy = strategy_cls(**params)
    except TypeError:
        strategy = strategy_cls()
        for k, v in params.items():
            setattr(strategy, k, v)

    runner = SimulationRunner(
        strategy,
        config['symbol'],
        config['start_date'],
        config['end_date'],
        db_host=config.get('db_host', 'localhost'),
        initial_balance=config.get('initial_balance', 10000.0),
        leverage=config.get('leverage', 1)
    )

    result = runner.run()

    return {
        "metrics": result['metrics'],
        "parameters": params,
    }

class ParallelSimulationRunner:
    def __init__(self, strategy_cls: Type[Strategy], symbol: str, start_date: str, end_date: str, initial_balance: float = 10000.0, leverage: int = 1, db_host: str = 'localhost'):
        self.config = {
            'strategy_cls': strategy_cls,
            'symbol': symbol,
            'start_date': start_date,
            'end_date': end_date,
            'initial_balance': initial_balance,
            'leverage': leverage,
            'db_host': db_host
        }

    def run_parallel(self, parameter_list: List[Dict[str, Any]], processes: int = 4) -> List[Dict[str, Any]]:
        items = [(self.config, params) for params in parameter_list]
        with multiprocessing.Pool(processes=processes) as pool:
            results = pool.starmap(run_simulation_task, items)
        return results
