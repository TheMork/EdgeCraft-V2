# EdgeCraft

EdgeCraft is a high-performance cryptocurrency trading backtesting station and strategy research platform.

## Architecture

- **Database**: TimescaleDB (PostgreSQL) for time-series data storage.
- **Backend**: FastAPI (Python) for data ingestion, backtesting logic, and sweep management.
- **Broker Simulation**: Fast simulation core via a Rust extension.
- **Frontend**: Next.js UI for a professional research experience.

## Setup Instructions

1. **Install dependencies:**
   `pip install -r requirements.txt`
   For frontend, navigate to `frontend/` and run `npm install`.

2. **Run infrastructure:**
   Ensure Docker is installed, then run `docker-compose up -d` to start TimescaleDB.

3. **Start services:**
   Run `./start_dev.sh` to start all required services in the correct order.

## Strategy Development

Strategies are implemented as drop-in Python scripts in `src/simulation/strategies/`.
See [docs/strategy_development.md](docs/strategy_development.md) for a guide on how to create and register new strategies.
