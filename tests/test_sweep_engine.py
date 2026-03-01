import pytest
import sqlite3
import json
import os
from src.optimization.sweep_engine import SweepEngine, SweepConfig

class DummyStrategy:
    def __init__(self, **kwargs):
        self.params = kwargs

@pytest.fixture
def test_db_path(tmp_path):
    return str(tmp_path / "test_sweeps.db")

@pytest.fixture
def engine(test_db_path):
    return SweepEngine(db_path=test_db_path)

@pytest.fixture
def config():
    return SweepConfig(
        strategy_name="demo",
        symbol="BTC/USDT",
        start_date="2023-01-01",
        end_date="2023-01-02",
        timeframe="1h",
        initial_balance=10000.0,
        leverage=1,
        param_grid={"param1": [1, 2], "param2": [0.1, 0.2]}
    )

def test_sweep_engine_initialization(engine, test_db_path):
    assert os.path.exists(test_db_path)
    with sqlite3.connect(test_db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='sweep_jobs'")
        assert cursor.fetchone() is not None
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='sweep_results'")
        assert cursor.fetchone() is not None

def test_db_serialization(engine, config):
    job_id = engine.create_job(config, 4)

    with sqlite3.connect(engine.db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT status, total_combinations FROM sweep_jobs WHERE job_id = ?", (job_id,))
        row = cursor.fetchone()
        assert row[0] == "running"
        assert row[1] == 4

    engine.update_job_status(job_id, "completed", progress=4)
    with sqlite3.connect(engine.db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT status, progress FROM sweep_jobs WHERE job_id = ?", (job_id,))
        row = cursor.fetchone()
        assert row[0] == "completed"
        assert row[1] == 4

    params = {"param1": 1, "param2": 0.1}
    metrics = {"sharpe_ratio": 1.5, "total_return": 10.0, "max_drawdown": -2.0, "win_rate": 0.6, "total_trades": 100}
    engine.save_result(job_id, params, metrics)

    with sqlite3.connect(engine.db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT parameters, sharpe_ratio FROM sweep_results WHERE job_id = ?", (job_id,))
        row = cursor.fetchone()
        assert json.loads(row[0]) == params
        assert row[1] == 1.5

# Top-level function for multiprocessing pickle
def mock_run_simulation_task(args):
    conf, params = args
    return {
        "parameters": params,
        "metrics": {
            "sharpe_ratio": params["param1"] + params["param2"]
        }
    }

def test_process_pool_grid_execution(engine, config, monkeypatch):
    monkeypatch.setattr("src.optimization.sweep_engine.SweepEngine._run_task", mock_run_simulation_task)

    job_id = engine.create_job(config, 4)
    engine.run_grid_search(job_id, config, DummyStrategy, processes=1)

    with sqlite3.connect(engine.db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT status FROM sweep_jobs WHERE job_id = ?", (job_id,))
        status = cursor.fetchone()[0]
        assert status == "completed", f"Job failed with status: {status}"

        cursor.execute("SELECT count(*) FROM sweep_results WHERE job_id = ?", (job_id,))
        assert cursor.fetchone()[0] == 4
