from fastapi import APIRouter, HTTPException, Query, BackgroundTasks, WebSocket, WebSocketDisconnect
from typing import List, Literal, Optional
import asyncio
import re
from pydantic import BaseModel
from src.database import QuestDBManager
from src.data_manager import SyncManager
from src.api.schemas import (
    OHLCVResponse,
    SyncRequest,
    SyncResponse,
    SyncJobStatusResponse,
    DataCoverageResponse,
)
from src.api.sync_jobs import SyncJobStore

router = APIRouter()

class SyncTop20Request(BaseModel):
    start_date: str
    end_date: Optional[str] = None
    sync_mode: Literal["trades", "candles", "candles_1m", "candles_all"] = "trades"
    timeframe: Literal["1m", "3m", "5m", "15m", "30m", "1h", "2h", "4h", "6h", "8h", "12h", "1d", "3d", "1w", "1M"] = "1m"
db = QuestDBManager()
sync_manager = SyncManager(db_manager=db)
sync_job_store = SyncJobStore()


def _run_multi_sync_job(
    job_id: str,
    symbols: List[str],
    start_date: str,
    end_date: Optional[str],
    sync_mode: str,
    timeframes: List[str],
) -> None:
    sync_job_store.update(
        job_id,
        status="running",
        progress=1,
        message=f"Starting multi-sync for {len(symbols)} symbols...",
    )

    total_tasks = len(symbols) * len(timeframes)
    completed_tasks = 0

    try:
        for symbol in symbols:
            for timeframe in timeframes:
                def _progress(progress: int, message: str) -> None:
                    # Map individual task progress (0-100) to overall progress
                    overall_base = (completed_tasks / total_tasks) * 100
                    overall_progress = overall_base + (progress / total_tasks)

                    sync_job_store.update(
                        job_id,
                        status="running",
                        progress=int(overall_progress),
                        message=f"[{symbol} {timeframe}] {message}",
                    )

                sync_manager.sync_data(
                    symbol,
                    start_date,
                    end_date,
                    sync_mode=sync_mode,
                    timeframe=timeframe,
                    progress_callback=_progress,
                )
                completed_tasks += 1

        sync_job_store.update(
            job_id,
            status="completed",
            progress=100,
            message=f"Sync completed for {len(symbols)} symbols.",
        )
    except Exception as e:
        sync_job_store.update(
            job_id,
            status="failed",
            progress=100,
            message=f"Sync failed: {e}",
        )


def _run_top20_sync_job(
    job_id: str,
    start_date: str,
    end_date: Optional[str],
    sync_mode: str,
    timeframe: str,
) -> None:
    sync_job_store.update(
        job_id,
        status="running",
        progress=1,
        message="Starting Top 20 sync...",
    )

    def _progress(progress: int, message: str) -> None:
        sync_job_store.update(
            job_id,
            status="running",
            progress=progress,
            message=message,
        )

    try:
        sync_manager.sync_top_20(
            start_date,
            end_date,
            sync_mode=sync_mode,
            timeframe=timeframe,
            progress_callback=_progress,
        )
        sync_job_store.update(
            job_id,
            status="completed",
            progress=100,
            message="Top 20 sync completed.",
        )
    except Exception as e:
        sync_job_store.update(
            job_id,
            status="failed",
            progress=100,
            message=f"Top 20 sync failed: {e}",
        )

@router.post("/sync", response_model=SyncResponse)
async def sync_data_endpoint(request: SyncRequest, background_tasks: BackgroundTasks):
    """
    Synchronizes historical trade data for the given symbols.
    Checks existing data and downloads missing parts using Bulk Downloader (ZIPs) and CCXT (API).
    """
    # Sanitize symbols
    for sym in request.symbols:
        if not re.match(r"^[a-zA-Z0-9/_:-]+$", sym):
            raise HTTPException(status_code=400, detail=f"Invalid symbol format: {sym}")

    if not request.symbols or not request.timeframes:
        raise HTTPException(status_code=400, detail="Symbols and timeframes lists cannot be empty")

    try:
        job = sync_job_store.create(
            message=f"Data sync queued for {len(request.symbols)} symbols",
            details={
                "symbols": request.symbols,
                "start_date": request.start_date,
                "end_date": request.end_date,
                "sync_mode": request.sync_mode,
                "timeframes": request.timeframes,
                "type": "multi",
            },
        )
        # Run in background to avoid blocking
        background_tasks.add_task(
            _run_multi_sync_job,
            job.job_id,
            request.symbols,
            request.start_date,
            request.end_date,
            request.sync_mode,
            request.timeframes,
        )
        return SyncResponse(
            status="success",
            message=f"Data sync started for {len(request.symbols)} symbols",
            job_id=job.job_id,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/sync/top20", response_model=SyncResponse)
async def sync_top_20_endpoint(request: SyncTop20Request, background_tasks: BackgroundTasks):
    """
    Synchronizes historical trade data for the top 20 symbols by volume.
    """
    try:
        job = sync_job_store.create(
            message="Top 20 data sync queued.",
            details={
                "start_date": request.start_date,
                "end_date": request.end_date,
                "sync_mode": request.sync_mode,
                "timeframe": request.timeframe,
                "type": "top20",
            },
        )
        background_tasks.add_task(
            _run_top20_sync_job,
            job.job_id,
            request.start_date,
            request.end_date,
            request.sync_mode,
            request.timeframe,
        )
        return SyncResponse(
            status="success",
            message=f"Top 20 data sync started ({request.sync_mode}, {request.timeframe}).",
            job_id=job.job_id,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/sync/jobs/{job_id}", response_model=SyncJobStatusResponse)
async def sync_job_status_endpoint(job_id: str):
    job = sync_job_store.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Sync job not found")

    return SyncJobStatusResponse(
        job_id=job.job_id,
        status=job.status,
        progress=job.progress,
        message=job.message,
        details=job.details,
        created_at=job.created_at,
        updated_at=job.updated_at,
    )

@router.websocket("/ws/data-sync/{job_id}")
async def websocket_data_sync(websocket: WebSocket, job_id: str):
    await websocket.accept()
    try:
        while True:
            job = sync_job_store.get(job_id)
            if job is None:
                await websocket.send_json({"error": "Sync job not found"})
                break

            await websocket.send_json({
                "job_id": job.job_id,
                "status": job.status,
                "progress": job.progress,
                "message": job.message,
            })

            if job.status in ["completed", "failed"]:
                break

            await asyncio.sleep(1) # Poll every second
    except WebSocketDisconnect:
        pass
    except Exception as e:
        print(f"WebSocket error: {e}")
        try:
             await websocket.close()
        except:
             pass

@router.get("/coverage", response_model=DataCoverageResponse)
async def get_data_coverage(
    symbol: str = Query(..., description="Trading symbol (e.g. BTC/USDT)"),
    timeframe: str = Query("1m", description="Candle timeframe"),
):
    """
    Returns the available data range in DB for the given symbol.
    """
    if not re.match(r"^[a-zA-Z0-9/_:-]+$", symbol):
        raise HTTPException(status_code=400, detail="Invalid symbol format")
    safe_timeframe = db.parse_timeframe(timeframe)
    if safe_timeframe is None:
        raise HTTPException(status_code=400, detail=f"Invalid timeframe: {timeframe}")

    try:
        trades_start, trades_end = db.get_trade_min_max(symbol)
        ohlcv_start, ohlcv_end = db.get_ohlcv_min_max(symbol, timeframe=safe_timeframe)
        ohlcv_ranges_raw = db.get_ohlcv_ranges(symbol)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    ohlcv_ranges = {
        tf: {"start": start, "end": end}
        for tf, (start, end) in ohlcv_ranges_raw.items()
    }

    starts = [ts for ts in (trades_start, ohlcv_start) if ts is not None]
    ends = [ts for ts in (trades_end, ohlcv_end) if ts is not None]

    return DataCoverageResponse(
        symbol=symbol,
        timeframe=safe_timeframe,
        available_start=min(starts) if starts else None,
        available_end=max(ends) if ends else None,
        trades_start=trades_start,
        trades_end=trades_end,
        ohlcv_start=ohlcv_start,
        ohlcv_end=ohlcv_end,
        ohlcv_ranges=ohlcv_ranges,
    )

@router.get("/history", response_model=List[OHLCVResponse])
async def get_history(
    symbol: str = Query(..., description="Trading symbol (e.g. BTC/USDT)"),
    start_date: str = Query(..., description="ISO 8601 start date (e.g., 2024-01-01T00:00:00Z)"),
    end_date: str = Query(..., description="ISO 8601 end date (e.g., 2024-01-02T00:00:00Z)"),
    timeframe: str = Query("1m", description="Candle timeframe"),
):
    """
    Fetch historical OHLCV data from QuestDB.
    """
    # Sanitize symbol
    if not re.match(r"^[a-zA-Z0-9/_:-]+$", symbol):
        raise HTTPException(status_code=400, detail="Invalid symbol format")
    safe_timeframe = db.parse_timeframe(timeframe)
    if safe_timeframe is None:
        raise HTTPException(status_code=400, detail=f"Invalid timeframe: {timeframe}")

    try:
        df = db.get_ohlcv(symbol, start_date, end_date, timeframe=safe_timeframe)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    if df.empty:
        return []

    # Convert DataFrame to list of dicts
    # reset_index to include timestamp as a column
    df = df.reset_index()

    # Ensure all required columns exist
    required = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
    if not all(col in df.columns for col in required):
         raise HTTPException(status_code=500, detail="Database schema mismatch")

    return df.to_dict(orient='records')
