from fastapi import APIRouter, HTTPException, Query, Body
from typing import List, Optional
from pydantic import BaseModel
from src.simulation.batch_service import batch_service

router = APIRouter()

class BatchRequest(BaseModel):
    symbols: List[str]
    strategies: List[str]
    timeframes: List[str]
    start_date: str
    end_date: str
    initial_balance: float = 10000.0
    leverage: int = 1

class BatchResponse(BaseModel):
    job_id: str
    message: str

@router.post("/run", response_model=BatchResponse)
async def run_batch_simulation(request: BatchRequest):
    if not request.symbols or not request.strategies or not request.timeframes:
        raise HTTPException(status_code=400, detail="Symbols, strategies, and timeframes must not be empty.")
    
    # Validation could be added here (e.g. check if strategies exist)
    
    job_id = batch_service.create_job(
        request.symbols,
        request.strategies,
        request.timeframes,
        request.start_date,
        request.end_date,
        request.initial_balance,
        request.leverage
    )
    
    return BatchResponse(job_id=job_id, message="Batch simulation started.")

@router.get("/status/{job_id}")
async def get_batch_status(job_id: str):
    job = batch_service.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Batch job not found")
    return job
