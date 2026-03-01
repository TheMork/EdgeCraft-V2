from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from src.api.routes.data import router as data_router
from src.api.routes.simulation import router as simulation_router
from src.api.routes.batch import router as batch_router
from src.api.routes.sweep import router as sweep_router
from src.api.schemas import HealthCheckResponse

app = FastAPI(
    title="Institutional Crypto Backtesting Engine",
    description="API for controlling simulations and fetching historical data.",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(data_router, prefix="/api/v1/data", tags=["data"])
app.include_router(simulation_router, prefix="/api/v1/simulation", tags=["simulation"])
app.include_router(batch_router, prefix="/api/v1/batch", tags=["batch"])
app.include_router(sweep_router, prefix="/api/v1/sweep", tags=["sweep"])

@app.get("/health", response_model=HealthCheckResponse)
async def health_check():
    return {"status": "ok"}
