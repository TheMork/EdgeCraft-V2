#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FRONTEND_DIR="$ROOT_DIR/frontend"
API_LOG="$ROOT_DIR/api.log"
FRONTEND_LOG="$FRONTEND_DIR/frontend.log"

start_detached() {
    local log_file="$1"
    shift

    if command -v setsid >/dev/null 2>&1; then
        setsid "$@" > "$log_file" 2>&1 < /dev/null &
    else
        nohup "$@" > "$log_file" 2>&1 < /dev/null &
    fi

    local pid=$!
    disown "$pid" 2>/dev/null || true
    echo "$pid"
}

cleanup() {
    echo "Stopping log tail. Services continue in background."
    exit 0
}

trap cleanup SIGINT SIGTERM

echo "Starting EdgeCraft development environment..."
cd "$ROOT_DIR"

echo "Starting TimescaleDB (PostgreSQL)..."
if [ "${EDGECRAFT_SKIP_DOCKER:-0}" = "1" ]; then
    echo "Skipping Database startup because EDGECRAFT_SKIP_DOCKER=1."
elif command -v docker >/dev/null 2>&1; then
    if docker compose up -d || docker-compose up -d; then
        echo "Database started via docker compose."
    elif command -v sudo >/dev/null 2>&1; then
        echo "Docker requires elevated privileges. Trying sudo..."
        if sudo docker compose up -d || sudo docker-compose up -d; then
            echo "Database started via sudo docker compose."
        else
            echo "Warning: Could not start Database via docker. Start it manually and continue."
        fi
    else
        echo "Warning: Docker is available but compose start failed and sudo is unavailable."
    fi
else
    echo "Docker is not available. Database was not started by this script."
fi

if [ "${EDGECRAFT_SKIP_DOCKER:-0}" != "1" ]; then
    # Wait for TimescaleDB to be ready
    echo "Waiting for TimescaleDB to be ready..."
    for i in {1..30}; do
        if docker exec edgecraft-timescaledb pg_isready -U postgres > /dev/null 2>&1; then
            echo "TimescaleDB is ready!"
            break
        fi
        sleep 1
    done
fi

echo "Setting up backend..."
if [ ! -d "$ROOT_DIR/venv" ]; then
    python3 -m venv "$ROOT_DIR/venv"
fi
"$ROOT_DIR/venv/bin/pip" install -r "$ROOT_DIR/requirements.txt"

echo "Building Rust extension..."
cd "$ROOT_DIR/rust_extension"
"$ROOT_DIR/venv/bin/maturin" build --release
cd "$ROOT_DIR"
"$ROOT_DIR/venv/bin/pip" install "$ROOT_DIR"/rust_extension/target/wheels/*.whl --force-reinstall

echo "Starting backend API..."
pkill -f "python start_api.py" || true
BACKEND_PID="$(start_detached "$API_LOG" env \
    PYTHONPATH="$ROOT_DIR" \
    EDGECRAFT_BATCH_EXECUTOR="${EDGECRAFT_BATCH_EXECUTOR:-thread}" \
    EDGECRAFT_BATCH_MAX_WORKERS="${EDGECRAFT_BATCH_MAX_WORKERS:-2}" \
    EDGECRAFT_DB_REQUEST_TIMEOUT_SECONDS="${EDGECRAFT_DB_REQUEST_TIMEOUT_SECONDS:-20}" \
    "$ROOT_DIR/venv/bin/python" "$ROOT_DIR/start_api.py")"
echo "$BACKEND_PID" > "$ROOT_DIR/.api.pid"
echo "Backend PID: $BACKEND_PID"

echo "Waiting for backend health check..."
for i in {1..30}; do
    if curl -s http://127.0.0.1:8000/health > /dev/null; then
        echo "Backend is ready!"
        break
    fi
    sleep 1
done

echo "Setting up frontend..."
cd "$FRONTEND_DIR"
if [ ! -d "$FRONTEND_DIR/node_modules" ]; then
    npm install
fi

echo "Ensuring frontend uses port 3000..."
# Stop old EdgeCraft Next.js processes first.
pkill -f "$FRONTEND_DIR/node_modules/.bin/next dev" || true
pkill -f "next-server" || true

# Stop any remaining listener on 3000.
EXISTING_PIDS="$(lsof -nP -iTCP:3000 -sTCP:LISTEN -t || true)"
if [ -n "$EXISTING_PIDS" ]; then
    echo "Stopping existing listener(s) on :3000 (PID(s): $EXISTING_PIDS)"
    kill $EXISTING_PIDS || true
    sleep 1
    STILL_LISTENING="$(lsof -nP -iTCP:3000 -sTCP:LISTEN -t || true)"
    if [ -n "$STILL_LISTENING" ]; then
        echo "Force-stopping remaining listener(s) on :3000 (PID(s): $STILL_LISTENING)"
        kill -9 $STILL_LISTENING || true
        sleep 1
    fi
fi

FRONTEND_PID="$(start_detached "$FRONTEND_LOG" env PORT=3000 npm run dev)"
echo "$FRONTEND_PID" > "$FRONTEND_DIR/.frontend.pid"
echo "Frontend PID: $FRONTEND_PID"

echo "Waiting for frontend readiness..."
for _ in {1..20}; do
    if curl -s -o /tmp/edgecraft_front_status -w "%{http_code}" "http://127.0.0.1:3000/simulation" | grep -q "200"; then
        break
    fi
    sleep 1
done

echo "Services started:"
echo "Backend:  http://127.0.0.1:8000"
echo "Frontend: http://127.0.0.1:3000/simulation"
echo "QuestDB:  http://127.0.0.1:9000"

echo "--- Backend log tail ---"
tail -n 10 "$API_LOG" || true
echo "--- Frontend log tail ---"
tail -n 10 "$FRONTEND_LOG" || true

echo "Tailing logs (Ctrl+C to stop tailing)..."
tail -f "$API_LOG" "$FRONTEND_LOG"
