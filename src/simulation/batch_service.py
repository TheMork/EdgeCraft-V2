import uuid
import itertools
import os
import threading
import math
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from typing import List, Dict, Any, Optional
from datetime import datetime

# Import strategies dynamically to avoid circular imports or giant if-else blocks in the worker
from src.simulation.strategies.demo import DemoStrategy
from src.simulation.strategies.quantitative_momentum import QuantitativeMomentumStrategy
from src.simulation.strategies.double_rsi_divergence import DoubleRSIDivergenceStrategy
from src.simulation.strategies.multi_divergence import MultiIndicatorDivergenceStrategy
from src.simulation.strategies.pair_arbitrage import PairArbitrageStrategy
from src.simulation.strategies.mde_mad_entropy import MDEMADEntropyStrategy
from src.simulation.strategies.classic_mde_mad_entropy import ClassicMDEMADEntropyStrategy
from src.simulation.strategies.mde_mad_v2 import MDEMADV2Strategy
from src.simulation.strategies.mde_mad_v2_leverage import MDEMADV2LeverageStrategy
from src.simulation.strategies.mde_mad_v3 import MDEMADV3Strategy
from src.simulation.strategies.mde_mad_v3_1 import MDEMADV3_1Strategy
from src.simulation.strategies.mde_mad_v4 import MDEMADV4Strategy

from src.simulation.runner import SimulationRunner

# Map strategy names to classes
STRATEGY_MAP = {
    "demo": DemoStrategy,
    "momentum": QuantitativeMomentumStrategy,
    "double_rsi_divergence": DoubleRSIDivergenceStrategy,
    "multi_divergence": MultiIndicatorDivergenceStrategy,
    "pair_arbitrage_v1": PairArbitrageStrategy, # Requires variant handling
    "pair_arbitrage_v2": PairArbitrageStrategy,
    "pair_arbitrage_v3": PairArbitrageStrategy,
    "pair_arbitrage_v4": PairArbitrageStrategy,
    "mde_mad_entropy": MDEMADEntropyStrategy,
    "mde_mad_classic": ClassicMDEMADEntropyStrategy,
    "mde_mad_v2": MDEMADV2Strategy,
    "mde_mad_v2_leverage": MDEMADV2LeverageStrategy,
    "mde_mad-v2_leverage": MDEMADV2LeverageStrategy,
    "mde_mad_v3": MDEMADV3Strategy,
    "mde_mad_v3_1": MDEMADV3_1Strategy,
    "mde_mad_v4": MDEMADV4Strategy,
}

def _worker_run_simulation(task_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Worker function to run a single simulation in a background worker.
    """
    try:
        symbol = task_data["symbol"]
        strategy_name = task_data["strategy_name"]
        timeframe = task_data["timeframe"]
        start_date = task_data["start_date"]
        end_date = task_data["end_date"]
        initial_balance = task_data.get("initial_balance", 10000.0)
        leverage = task_data.get("leverage", 1)
        
        # Strategy instantiation logic
        strategy_cls = STRATEGY_MAP.get(strategy_name)
        if not strategy_cls:
            return {"status": "failed", "error": f"Unknown strategy: {strategy_name}", "task": task_data}

        # Handle specific strategy params (defaults for now, could be extended)
        strategy_kwargs = {
            "timeframe": timeframe, # Most strategies accept this
            "min_leverage": leverage,
            "max_leverage": max(leverage, 3),
            "enable_shorts": True
        }
        
        # Special handling for Pair Arbitrage variants
        if "pair_arbitrage" in strategy_name:
            variant = strategy_name.split("_")[-1]
            strategy_kwargs["variant"] = variant
            strategy_kwargs["reference_symbol"] = "BTC/USDT" # Default reference
            strategy_kwargs["asset_symbol"] = symbol
            strategy_kwargs["start_date"] = start_date
            strategy_kwargs["end_date"] = end_date

        # Filter kwargs based on what the strategy actually accepts would be safer,
        # but Python allows passing extra kwargs if **kwargs is in init, or we rely on shared base.
        # For now, we assume our strategies are somewhat uniform or tolerant.
        # If instantiation fails, we catch it.
        
        try:
            strategy = strategy_cls(**strategy_kwargs)
        except TypeError:
            # Fallback for strategies with different signatures
            # Try minimal init
            try:
                strategy = strategy_cls()
                strategy.timeframe = timeframe # Inject manually
            except Exception as e:
                return {"status": "failed", "error": f"Strategy init failed: {e}", "task": task_data}

        runner = SimulationRunner(
            strategy=strategy,
            symbol=symbol,
            start_date=start_date,
            end_date=end_date,
            timeframe=timeframe,
            initial_balance=initial_balance,
            leverage=leverage,
            auto_sync_on_missing_data=False,
            allow_trade_backfill=False,
        )
        
        # Suppress stdout/stderr during run to keep logs clean? 
        # Or redirect to a file? For now, let it print to worker logs.
        result = runner.run()

        metrics = result.get("metrics") if isinstance(result, dict) else None
        if not metrics:
            return {
                "status": "failed",
                "task": task_data,
                "error": (
                    f"No usable OHLCV data for {symbol} ({timeframe}) in requested range "
                    f"{start_date} - {end_date}. Please sync candles first."
                ),
            }
        return {
            "status": "success",
            "task": task_data,
            "metrics": metrics,
            # We don't return the full trades/equity curve to save memory in batch results,
            # unless specifically requested (future feature).
        }

    except Exception as e:
        return {"status": "failed", "error": str(e), "task": task_data}


class BatchJob:
    def __init__(self, job_id: str, total_tasks: int):
        self.job_id = job_id
        self.status = "queued"  # queued, running, completed, failed
        self.message = "Queued."
        self.created_at = datetime.now().isoformat()
        self.completed_at = None
        self.total_tasks = total_tasks
        self.completed_tasks = 0
        self.results: List[Dict[str, Any]] = []
        self.error = None

class BatchService:
    def __init__(self):
        self.jobs: Dict[str, BatchJob] = {}
        self._lock = threading.Lock()

    def create_job(
        self,
        symbols: List[str],
        strategies: List[str],
        timeframes: List[str],
        start_date: str,
        end_date: str,
        initial_balance: float = 10000.0,
        leverage: int = 1
    ) -> str:
        job_id = str(uuid.uuid4())

        # Create combinatorial tasks
        tasks = []
        # Interleave symbols across strategy/timeframe combinations to avoid
        # "all BTC first, then next symbol" behavior when worker count is low.
        for strat, tf, sym in itertools.product(strategies, timeframes, symbols):
            tasks.append({
                "symbol": sym,
                "strategy_name": strat,
                "timeframe": tf,
                "start_date": start_date,
                "end_date": end_date,
                "initial_balance": initial_balance,
                "leverage": leverage
            })

        with self._lock:
            self.jobs[job_id] = BatchJob(job_id, len(tasks))

        # Run in background thread to avoid blocking FastAPI.
        t = threading.Thread(target=self._process_batch, args=(job_id, tasks), name=f"batch-{job_id[:8]}")
        t.daemon = True
        t.start()

        return job_id

    def _determine_worker_count(self, task_count: int) -> int:
        configured = os.getenv("EDGECRAFT_BATCH_MAX_WORKERS")
        if configured:
            try:
                parsed = int(configured)
                if parsed > 0:
                    return min(parsed, max(1, task_count))
            except ValueError:
                pass
        cpu_count = os.cpu_count() or 1
        # Keep defaults conservative to protect QuestDB from timeout spikes.
        return min(max(2, cpu_count // 4), 4, max(1, task_count))

    def _executor_mode(self) -> str:
        mode = (os.getenv("EDGECRAFT_BATCH_EXECUTOR") or "thread").strip().lower()
        if mode in {"process", "thread"}:
            return mode
        return "thread"

    def _is_transient_error(self, error: Any) -> bool:
        text = str(error or "").lower()
        if not text:
            return False
        transient_markers = (
            "timed out",
            "connection refused",
            "failed to establish a new connection",
            "remote end closed connection",
            "questdb unavailable",
        )
        return any(marker in text for marker in transient_markers)

    def _run_tasks_parallel(
        self,
        job_id: str,
        tasks: List[Dict[str, Any]],
        workers: int,
        mode_override: Optional[str] = None,
    ) -> None:
        mode = mode_override or self._executor_mode()
        if mode == "thread":
            executor = ThreadPoolExecutor(max_workers=workers, thread_name_prefix=f"batch-{job_id[:8]}")
        else:
            # Spawn avoids fork-related crashes with DB/network clients in worker children.
            mp_ctx = multiprocessing.get_context("spawn")
            executor = ProcessPoolExecutor(max_workers=workers, mp_context=mp_ctx)

        with executor:
            futures = {executor.submit(_worker_run_simulation, task): task for task in tasks}
            for future in as_completed(futures):
                task = futures[future]
                try:
                    result = future.result()
                except Exception as e:
                    result = {"status": "failed", "error": f"Worker crashed: {e}", "task": task}

                # Retry one time inline for transient infra/database failures.
                if result.get("status") != "success" and self._is_transient_error(result.get("error")):
                    with self._lock:
                        job = self.jobs.get(job_id)
                        if job:
                            job.message = (
                                f"Transient error for {task.get('symbol')} ({task.get('timeframe')}), retrying once..."
                            )
                    retry_result = _worker_run_simulation(task)
                    if retry_result.get("status") == "success":
                        result = retry_result
                    else:
                        result["error"] = (
                            f"{result.get('error')} | retry failed: {retry_result.get('error')}"
                        )

                with self._lock:
                    job = self.jobs.get(job_id)
                    if not job:
                        return
                    job.results.append(result)
                    job.completed_tasks += 1
                    job.message = f"Completed {job.completed_tasks}/{job.total_tasks} task(s)."

    def _to_json_safe(self, value: Any) -> Any:
        if isinstance(value, float):
            return value if math.isfinite(value) else None
        if isinstance(value, datetime):
            return value.isoformat()
        if isinstance(value, dict):
            return {str(k): self._to_json_safe(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [self._to_json_safe(v) for v in value]
        if hasattr(value, "item"):
            try:
                return self._to_json_safe(value.item())
            except Exception:
                return str(value)
        return value

    def _process_batch(self, job_id: str, tasks: List[Dict[str, Any]]):
        try:
            with self._lock:
                job = self.jobs.get(job_id)
                if not job:
                    return
                job.status = "running"
                job.message = f"Running {job.total_tasks} task(s)..."

            if not tasks:
                with self._lock:
                    job = self.jobs.get(job_id)
                    if not job:
                        return
                    job.status = "completed"
                    job.message = "No tasks to run."
                    job.completed_at = datetime.now().isoformat()
                return

            workers = self._determine_worker_count(len(tasks))
            self._run_tasks_parallel(job_id, tasks, workers)

            # If process workers all crashed, retry once in threads for resilience.
            with self._lock:
                job = self.jobs.get(job_id)
                if not job:
                    return
                only_worker_crashes = (
                    bool(job.results)
                    and all(
                        item.get("status") == "failed"
                        and str(item.get("error", "")).startswith("Worker crashed:")
                        for item in job.results
                    )
                )
                if only_worker_crashes and self._executor_mode() == "process":
                    job.results = []
                    job.completed_tasks = 0
                    job.message = "Process workers crashed, retrying in thread mode..."

            if only_worker_crashes and self._executor_mode() == "process":
                self._run_tasks_parallel(job_id, tasks, workers, mode_override="thread")

            with self._lock:
                job = self.jobs.get(job_id)
                if not job:
                    return
                failed_tasks = sum(1 for item in job.results if item.get("status") != "success")
                if failed_tasks == job.total_tasks and job.total_tasks > 0:
                    job.status = "failed"
                    job.error = f"All tasks failed ({failed_tasks}/{job.total_tasks})."
                    job.message = job.error
                else:
                    job.status = "completed"
                    job.error = f"{failed_tasks}/{job.total_tasks} tasks failed." if failed_tasks > 0 else None
                    job.message = "Batch completed with errors." if failed_tasks > 0 else "Batch completed successfully."
                job.completed_at = datetime.now().isoformat()

        except Exception as e:
            with self._lock:
                job = self.jobs.get(job_id)
                if not job:
                    return
                job.status = "failed"
                job.error = str(e)
                job.message = str(e)
                job.completed_at = datetime.now().isoformat()

    def get_job(self, job_id: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            job = self.jobs.get(job_id)
            if not job:
                return None
            total = job.total_tasks
            completed = job.completed_tasks
            results = list(job.results)

        return {
            "job_id": job.job_id,
            "status": job.status,
            "message": job.message,
            "progress": {
                "total": total,
                "completed": completed,
                "percent": (
                    max(1, int((completed / total) * 100))
                    if total > 0 and job.status == "running"
                    else int((completed / total) * 100) if total > 0 else 0
                ),
            },
            "created_at": job.created_at,
            "completed_at": job.completed_at,
            "error": self._to_json_safe(job.error),
            "results": self._to_json_safe(results)
        }

# Global instance
batch_service = BatchService()
