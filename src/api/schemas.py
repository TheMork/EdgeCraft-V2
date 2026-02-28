from pydantic import BaseModel
from typing import Any, Dict, List, Literal, Optional
from datetime import datetime

class OHLCVResponse(BaseModel):
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float

class HealthCheckResponse(BaseModel):
    status: str

class SyncRequest(BaseModel):
    symbols: List[str]
    start_date: str
    end_date: Optional[str] = None
    sync_mode: Literal["trades", "candles", "candles_1m", "candles_all"] = "trades"
    timeframes: List[Literal["1m", "3m", "5m", "15m", "30m", "1h", "2h", "4h", "6h", "8h", "12h", "1d", "3d", "1w", "1M"]] = ["1m"]

class SyncResponse(BaseModel):
    status: str
    message: str
    job_id: Optional[str] = None


class SyncJobStatusResponse(BaseModel):
    job_id: str
    status: str
    progress: int
    message: str
    details: Dict[str, Any]
    created_at: datetime
    updated_at: datetime


class DataCoverageResponse(BaseModel):
    symbol: str
    timeframe: str
    available_start: Optional[datetime]
    available_end: Optional[datetime]
    trades_start: Optional[datetime]
    trades_end: Optional[datetime]
    ohlcv_start: Optional[datetime]
    ohlcv_end: Optional[datetime]
    ohlcv_ranges: Dict[str, Dict[str, Optional[datetime]]]
