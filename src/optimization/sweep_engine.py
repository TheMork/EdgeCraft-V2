import sqlite3
import json
import uuid
import itertools
import multiprocessing
import optuna
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from datetime import datetime

@dataclass
class SweepConfig:
    strategy_name: str
    symbol: str
    start_date: str
    end_date: str
    timeframe: str
    initial_balance: float
    leverage: int
    param_grid: Dict[str, List[Any]]

class SweepEngine:
    def __init__(self, db_path: str = "results/sweeps.db"):
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        import os
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS sweep_jobs (
                    job_id TEXT PRIMARY KEY,
                    strategy_name TEXT,
                    status TEXT,
                    progress INTEGER,
                    total_combinations INTEGER,
                    created_at TEXT
                )
            ''')
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS sweep_results (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    job_id TEXT,
                    parameters TEXT,
                    sharpe_ratio REAL,
                    total_return REAL,
                    max_drawdown REAL,
                    win_rate REAL,
                    total_trades INTEGER,
                    FOREIGN KEY(job_id) REFERENCES sweep_jobs(job_id)
                )
            ''')
            conn.commit()

    def create_job(self, config: SweepConfig, total_combinations: int) -> str:
        job_id = str(uuid.uuid4())
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO sweep_jobs (job_id, strategy_name, status, progress, total_combinations, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (job_id, config.strategy_name, "running", 0, total_combinations, datetime.now().isoformat()))
            conn.commit()
        return job_id

    def update_job_status(self, job_id: str, status: str, progress: Optional[int] = None):
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            if progress is not None:
                cursor.execute('UPDATE sweep_jobs SET status = ?, progress = ? WHERE job_id = ?', (status, progress, job_id))
            else:
                cursor.execute('UPDATE sweep_jobs SET status = ? WHERE job_id = ?', (status, job_id))
            conn.commit()

    def save_result(self, job_id: str, parameters: Dict[str, Any], metrics: Dict[str, Any]):
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO sweep_results (job_id, parameters, sharpe_ratio, total_return, max_drawdown, win_rate, total_trades)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (
                job_id,
                json.dumps(parameters),
                metrics.get('sharpe_ratio', 0.0),
                metrics.get('total_return', 0.0),
                metrics.get('max_drawdown', 0.0),
                metrics.get('win_rate', 0.0),
                metrics.get('total_trades', 0)
            ))
            conn.commit()

    @staticmethod
    def _run_task(args):
        from src.simulation.parallel_runner import run_simulation_task
        config, params = args
        return run_simulation_task(config, params)

    def run_grid_search(self, job_id: str, config: SweepConfig, strategy_cls: Any, processes: int = 4, db_host: str = 'localhost'):
        from src.simulation.parallel_runner import ParallelSimulationRunner

        keys = config.param_grid.keys()
        values = config.param_grid.values()
        combinations = list(itertools.product(*values))
        param_list = [dict(zip(keys, v)) for v in combinations]

        runner_config = {
            'strategy_cls': strategy_cls,
            'symbol': config.symbol,
            'start_date': config.start_date,
            'end_date': config.end_date,
            'initial_balance': config.initial_balance,
            'leverage': config.leverage,
            'db_host': db_host
        }

        try:
            items = [(runner_config, params) for params in param_list]
            completed = 0
            with multiprocessing.Pool(processes=processes) as pool:
                for result in pool.imap_unordered(SweepEngine._run_task, items):
                    # Check for cancellation
                    if completed % max(1, len(param_list) // 100) == 0:
                        with sqlite3.connect(self.db_path) as conn:
                            cursor = conn.cursor()
                            cursor.execute('SELECT status FROM sweep_jobs WHERE job_id = ?', (job_id,))
                            row = cursor.fetchone()
                            if row and row[0] == "cancelled":
                                pool.terminate()
                                return

                    completed += 1
                    self.save_result(job_id, result['parameters'], result['metrics'])
                    if completed % 10 == 0 or completed == len(param_list):
                        self.update_job_status(job_id, "running", progress=completed)


            self.update_job_status(job_id, "completed", progress=len(param_list))

        except Exception as e:
            self.update_job_status(job_id, f"failed: {str(e)}")

    def run_bayesian_optimization(self, job_id: str, config: SweepConfig, strategy_cls: Any, n_trials: int = 100, db_host: str = 'localhost'):
        from src.simulation.runner import SimulationRunner

        def objective(trial):
            params = {}
            for key, values in config.param_grid.items():
                if isinstance(values[0], int):
                    params[key] = trial.suggest_int(key, min(values), max(values))
                elif isinstance(values[0], float):
                    params[key] = trial.suggest_float(key, min(values), max(values))
                else:
                    params[key] = trial.suggest_categorical(key, values)

            try:
                strategy = strategy_cls(**params)
            except TypeError:
                strategy = strategy_cls()
                for k, v in params.items():
                    setattr(strategy, k, v)
            runner = SimulationRunner(
                strategy,
                config.symbol,
                config.start_date,
                config.end_date,
                db_host=db_host,
                initial_balance=config.initial_balance,
                leverage=config.leverage
            )

            # Check for cancellation
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('SELECT status FROM sweep_jobs WHERE job_id = ?', (job_id,))
                row = cursor.fetchone()
                if row and row[0] == "cancelled":
                    trial.study.stop()
                    return -float('inf')

            result = runner.run()


            self.save_result(job_id, params, result['metrics'])
            self.update_job_status(job_id, "running", progress=trial.number + 1)

            val = result['metrics'].get('sharpe_ratio', 0.0)
            if val is None:
                val = -float('inf')
            return val

        try:
            study = optuna.create_study(direction="maximize")
            study.optimize(objective, n_trials=n_trials)
            self.update_job_status(job_id, "completed", progress=n_trials)
        except Exception as e:
            self.update_job_status(job_id, f"failed: {str(e)}")

sweep_engine = SweepEngine()
