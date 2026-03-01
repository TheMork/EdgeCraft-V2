from fastapi import APIRouter, HTTPException, BackgroundTasks, WebSocket, WebSocketDisconnect
from typing import List, Dict, Any
import asyncio
import json
import sqlite3

from src.api.schemas import (
    SweepRequest,
    SweepStartResponse,
    SweepStatusResponse,
    SweepResultsResponse,
    SweepResultData
)
from src.optimization.sweep_engine import sweep_engine, SweepConfig

router = APIRouter()

# Need to import all strategies to resolve name to class
from src.simulation.strategies.quantitative_momentum import QuantitativeMomentumStrategy
from src.simulation.strategies.double_rsi_divergence import DoubleRSIDivergenceStrategy
from src.simulation.strategies.mde_mad_entropy import MDEMADEntropyStrategy
from src.simulation.strategies.classic_mde_mad_entropy import ClassicMDEMADEntropyStrategy
from src.simulation.strategies.mde_mad_v2 import MDEMADV2Strategy
from src.simulation.strategies.mde_mad_v2_leverage import MDEMADV2LeverageStrategy
from src.simulation.strategies.mde_mad_v3 import MDEMADV3Strategy
from src.simulation.strategies.mde_mad_v3_1 import MDEMADV3_1Strategy
from src.simulation.strategies.mde_mad_v4 import MDEMADV4Strategy
from src.simulation.strategies.multi_divergence import MultiIndicatorDivergenceStrategy
from src.simulation.strategies.pair_arbitrage import PairArbitrageStrategy
from src.simulation.strategies.demo import DemoStrategy

STRATEGY_MAP = {
    "momentum": QuantitativeMomentumStrategy,
    "double_rsi_divergence": DoubleRSIDivergenceStrategy,
    "mde_mad_entropy": MDEMADEntropyStrategy,
    "mde_mad_classic": ClassicMDEMADEntropyStrategy,
    "mde_mad_v2": MDEMADV2Strategy,
    "mde_mad_v2_leverage": MDEMADV2LeverageStrategy,
    "mde_mad_v3": MDEMADV3Strategy,
    "mde_mad_v3_1": MDEMADV3_1Strategy,
    "mde_mad_v4": MDEMADV4Strategy,
    "multi_divergence": MultiIndicatorDivergenceStrategy,
    "pair_arbitrage": PairArbitrageStrategy,
    "demo": DemoStrategy,
}

@router.post("/start", response_model=SweepStartResponse)
async def start_sweep(request: SweepRequest, background_tasks: BackgroundTasks):
    strategy_cls = STRATEGY_MAP.get(request.strategy_name)
    if not strategy_cls:
        raise HTTPException(status_code=400, detail=f"Strategy {request.strategy_name} not found")

    config = SweepConfig(
        strategy_name=request.strategy_name,
        symbol=request.symbol,
        start_date=request.start_date,
        end_date=request.end_date,
        timeframe=request.timeframe,
        initial_balance=request.initial_balance,
        leverage=request.leverage,
        param_grid=request.param_grid
    )

    if request.method == "grid":
        import itertools
        total_combinations = len(list(itertools.product(*config.param_grid.values())))
    else:
        total_combinations = request.n_trials or 100

    job_id = sweep_engine.create_job(config, total_combinations)

    if request.method == "grid":
        background_tasks.add_task(
            sweep_engine.run_grid_search,
            job_id,
            config,
            strategy_cls,
            processes=request.processes or 4
        )
    else:
        background_tasks.add_task(
            sweep_engine.run_bayesian_optimization,
            job_id,
            config,
            strategy_cls,
            n_trials=request.n_trials or 100
        )

    return SweepStartResponse(job_id=job_id, message="Sweep job started")

@router.get("/{job_id}/status", response_model=SweepStatusResponse)
async def get_sweep_status(job_id: str):
    with sqlite3.connect(sweep_engine.db_path) as conn:
        cursor = conn.cursor()
        cursor.execute('SELECT status, progress, total_combinations FROM sweep_jobs WHERE job_id = ?', (job_id,))
        row = cursor.fetchone()

    if not row:
        raise HTTPException(status_code=404, detail="Sweep job not found")

    status, progress, total_combinations = row
    return SweepStatusResponse(
        job_id=job_id,
        status=status,
        progress=progress,
        total_combinations=total_combinations
    )

@router.get("/{job_id}/results", response_model=SweepResultsResponse)
async def get_sweep_results(job_id: str):
    with sqlite3.connect(sweep_engine.db_path) as conn:
        cursor = conn.cursor()
        cursor.execute('''
            SELECT parameters, sharpe_ratio, total_return, max_drawdown, win_rate, total_trades
            FROM sweep_results
            WHERE job_id = ?
            ORDER BY sharpe_ratio DESC
        ''', (job_id,))
        rows = cursor.fetchall()

    results = []
    for row in rows:
        results.append(SweepResultData(
            parameters=json.loads(row[0]),
            sharpe_ratio=row[1],
            total_return=row[2],
            max_drawdown=row[3],
            win_rate=row[4],
            total_trades=row[5]
        ))

    return SweepResultsResponse(job_id=job_id, results=results)

@router.delete("/{job_id}")
async def cancel_sweep(job_id: str):
    sweep_engine.update_job_status(job_id, "cancelled")
    return {"message": "Sweep job cancelled. Note: Currently running processes might finish their current task."}

@router.websocket("/ws/{job_id}")
async def sweep_websocket(websocket: WebSocket, job_id: str):
    await websocket.accept()
    try:
        while True:
            with sqlite3.connect(sweep_engine.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('SELECT status, progress, total_combinations FROM sweep_jobs WHERE job_id = ?', (job_id,))
                row = cursor.fetchone()

            if not row:
                await websocket.send_json({"error": "Job not found"})
                break

            status, progress, total_combinations = row
            await websocket.send_json({
                "job_id": job_id,
                "status": status,
                "progress": progress,
                "total_combinations": total_combinations
            })

            if status in ["completed", "cancelled"] or status.startswith("failed"):
                break

            await asyncio.sleep(1)

    except WebSocketDisconnect:
        print(f"WebSocket client disconnected for job {job_id}")
    except Exception as e:
        print(f"WebSocket error for job {job_id}: {e}")
