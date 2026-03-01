from typing import Literal

from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Query
import asyncio
import queue
import math
from typing import List, Optional
from pydantic import BaseModel
from src.simulation.runner import SimulationRunner
from src.simulation.multi_runner import MultiAssetSimulationRunner
from src.simulation.strategies.demo import DemoStrategy
from src.simulation.strategies.double_rsi_divergence import DoubleRSIDivergenceStrategy
from src.simulation.strategies.multi_divergence import MultiIndicatorDivergenceStrategy
from src.simulation.strategies.pair_arbitrage import PairArbitrageStrategy
from src.simulation.strategies.quantitative_momentum import QuantitativeMomentumStrategy
from src.simulation.strategies.mde_mad_entropy import MDEMADEntropyStrategy
from src.simulation.strategies.classic_mde_mad_entropy import ClassicMDEMADEntropyStrategy
from src.simulation.strategies.mde_mad_v2 import MDEMADV2Strategy
from src.simulation.strategies.mde_mad_v2_leverage import MDEMADV2LeverageStrategy
from src.simulation.strategies.mde_mad_v3 import MDEMADV3Strategy
from src.simulation.strategies.mde_mad_v3_1 import MDEMADV3_1Strategy
from src.simulation.strategies.mde_mad_v4 import MDEMADV4Strategy
from src.database import QuestDBManager
from datetime import datetime
import json
import pandas as pd

router = APIRouter()
db = QuestDBManager()

class DateTimeEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (datetime, pd.Timestamp)):
            return obj.isoformat()
        return super().default(obj)


def _json_safe(value):
    if isinstance(value, (datetime, pd.Timestamp)):
        return value.isoformat()
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    # Handle pandas/numpy scalar values
    if hasattr(value, "item"):
        try:
            return _json_safe(value.item())
        except Exception:
            pass
    try:
        if pd.isna(value):
            return None
    except Exception:
        pass
    return value

class MultiSimulationRequest(BaseModel):
    symbols: List[str]
    start_date: str
    end_date: str
    timeframe: str = "1m"
    strategy_name: str = "demo"
    initial_balance: float = 10000.0
    leverage: int = 1
    slippage_bps: float = 0.0

@router.post("/multi")
async def multi_simulation(request: MultiSimulationRequest):
    # Setup strategy
    if request.strategy_name == "momentum":
        strategy_class = QuantitativeMomentumStrategy
    elif request.strategy_name == "double_rsi_divergence":
        strategy_class = DoubleRSIDivergenceStrategy
    elif request.strategy_name == "multi_divergence":
        strategy_class = MultiIndicatorDivergenceStrategy
    elif request.strategy_name == "mde_mad_entropy":
        strategy_class = MDEMADEntropyStrategy
    elif request.strategy_name == "mde_mad_classic":
        strategy_class = ClassicMDEMADEntropyStrategy
    elif request.strategy_name == "mde_mad_v2":
        strategy_class = MDEMADV2Strategy
    elif request.strategy_name in {"mde_mad_v2_leverage", "mde_mad-v2_leverage"}:
        strategy_class = MDEMADV2LeverageStrategy
    elif request.strategy_name == "mde_mad_v3":
        strategy_class = MDEMADV3Strategy
    elif request.strategy_name == "mde_mad_v3_1":
        strategy_class = MDEMADV3_1Strategy
    elif request.strategy_name == "mde_mad_v4":
        strategy_class = MDEMADV4Strategy
    elif request.strategy_name.startswith("pair_arbitrage"):
        strategy_class = PairArbitrageStrategy
    else:
        strategy_class = DemoStrategy

    runner = MultiAssetSimulationRunner(
        strategy_class=strategy_class,
        symbols=request.symbols,
        start_date=request.start_date,
        end_date=request.end_date,
        timeframe=request.timeframe,
        initial_balance=request.initial_balance,
        leverage=request.leverage,
        slippage_bps=request.slippage_bps,
    )

    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(None, runner.run)

    return {
        "status": "success",
        "metrics": _json_safe(result.get("metrics", {})),
        "equity_curve": _json_safe(result.get("equity_curve", []))
    }

@router.websocket("/ws")
async def websocket_endpoint(
    websocket: WebSocket,
    symbol: str = Query(..., description="Symbol to simulate"),
    start_date: str = Query(..., description="Start date ISO"),
    end_date: str = Query(..., description="End date ISO"),
    strategy_name: Literal[
        "demo",
        "momentum",
        "double_rsi_divergence",
        "multi_divergence",
        "pair_arbitrage_v1",
        "pair_arbitrage_v2",
        "pair_arbitrage_v3",
        "pair_arbitrage_v4",
        "mde_mad_entropy",
        "mde_mad_classic",
        "mde_mad_v2",
        "mde_mad_v2_leverage",
        "mde_mad-v2_leverage",
        "mde_mad_v3",
        "mde_mad_v3_1",
        "mde_mad_v4",
    ] = Query(
        "demo",
        alias="strategy",
        description=(
            "Strategy name "
            "(demo, momentum, double_rsi_divergence, multi_divergence, "
            "pair_arbitrage_v1, pair_arbitrage_v2, pair_arbitrage_v3, pair_arbitrage_v4, "
            "mde_mad_entropy, mde_mad_classic, mde_mad_v2, mde_mad_v2_leverage, mde_mad-v2_leverage, "
            "mde_mad_v3, mde_mad_v3_1, mde_mad_v4)"
        ),
    ),
    timeframe: str = Query("1m", description="Candle timeframe"),
    leverage: int = Query(1, ge=1, le=125, description="Broker leverage"),
    min_leverage: int = Query(1, ge=1, le=125, description="Min strategy leverage"),
    max_leverage: int = Query(3, ge=1, le=125, description="Max strategy leverage"),
    enable_shorts: bool = Query(True, description="Enable short positions for compatible strategies"),
    stream: bool = Query(True, description="Stream candle events during simulation"),
    multi_indicators: str = Query(
        "rsi,macd_hist,stoch_k,williams_r,cci,mfi",
        description="Comma-separated indicators for multi_divergence strategy",
    ),
    multi_required_bullish: int = Query(3, ge=1, le=10, description="Min bullish divergence count"),
    multi_required_bearish: int = Query(3, ge=1, le=10, description="Min bearish divergence count"),
    multi_required_bullish_score: float = Query(3.0, ge=0.1, le=20.0, description="Min bullish divergence score"),
    multi_required_bearish_score: float = Query(3.0, ge=0.1, le=20.0, description="Min bearish divergence score"),
    multi_include_regular: bool = Query(True, description="Include regular divergences"),
    multi_include_hidden: bool = Query(True, description="Include hidden divergences"),
    multi_hidden_score_multiplier: float = Query(1.15, ge=1.0, le=5.0, description="Hidden divergence score multiplier"),
    multi_min_adx_for_entry: float = Query(10.0, ge=0.0, le=100.0, description="Minimum ADX for entries"),
    multi_require_regime_filter: bool = Query(True, description="Require EMA regime alignment"),
    multi_require_volume_confirmation: bool = Query(True, description="Require volume confirmation"),
    multi_pivot_lookback: int = Query(3, ge=1, le=20, description="Pivot lookback bars"),
    multi_min_pivot_separation_bars: int = Query(5, ge=1, le=300, description="Minimum bars between pivots"),
    multi_max_pivot_age_bars: int = Query(120, ge=5, le=3000, description="Max age of latest pivot"),
    multi_min_price_move_pct: float = Query(0.001, ge=0.0, le=1.0, description="Minimum price move between pivots"),
    multi_min_indicator_delta: float = Query(0.0, ge=0.0, le=1000.0, description="Minimum indicator delta between pivots"),
    multi_risk_per_trade: float = Query(0.01, ge=0.0001, le=0.2, description="Risk per trade"),
    multi_sl_atr_multiplier: float = Query(2.5, ge=0.1, le=20.0, description="Stop loss ATR multiplier"),
    multi_analysis_window_bars: int = Query(1200, ge=200, le=10000, description="Rolling analysis window for indicators"),
):
    await websocket.accept()

    # Create a thread-safe queue
    event_queue = queue.Queue()

    safe_timeframe = db.parse_timeframe(timeframe) or "1m"
    lev = max(1, int(leverage))
    min_lev = max(1, int(min_leverage))
    max_lev = max(min_lev, int(max_leverage))
    # Create strategy and runner
    if strategy_name == "momentum":
        strategy = QuantitativeMomentumStrategy(
            timeframe=safe_timeframe,
            enable_shorts=enable_shorts,
            min_leverage=min_lev,
            max_leverage=max_lev,
        )
    elif strategy_name == "double_rsi_divergence":
        strategy = DoubleRSIDivergenceStrategy(
            timeframe=safe_timeframe,
            enable_shorts=enable_shorts,
            min_leverage=min_lev,
            max_leverage=max_lev,
        )
    elif strategy_name == "multi_divergence":
        supported_indicators = set(MultiIndicatorDivergenceStrategy.SUPPORTED_INDICATORS)
        indicator_list = tuple(
            part.strip().lower()
            for part in (multi_indicators or "").split(",")
            if part and part.strip() and part.strip().lower() in supported_indicators
        ) or MultiIndicatorDivergenceStrategy.DEFAULT_INDICATORS
        strategy = MultiIndicatorDivergenceStrategy(
            timeframe=safe_timeframe,
            indicators=indicator_list,
            required_bullish=multi_required_bullish,
            required_bearish=multi_required_bearish,
            required_bullish_score=multi_required_bullish_score,
            required_bearish_score=multi_required_bearish_score,
            include_regular=multi_include_regular,
            include_hidden=multi_include_hidden,
            hidden_score_multiplier=multi_hidden_score_multiplier,
            min_adx_for_entry=multi_min_adx_for_entry,
            require_regime_filter=multi_require_regime_filter,
            require_volume_confirmation=multi_require_volume_confirmation,
            pivot_lookback=multi_pivot_lookback,
            min_pivot_separation_bars=multi_min_pivot_separation_bars,
            max_pivot_age_bars=multi_max_pivot_age_bars,
            min_price_move_pct=multi_min_price_move_pct,
            min_indicator_delta=multi_min_indicator_delta,
            risk_per_trade=multi_risk_per_trade,
            sl_atr_multiplier=multi_sl_atr_multiplier,
            analysis_window_bars=multi_analysis_window_bars,
            enable_shorts=enable_shorts,
            min_leverage=min_lev,
            max_leverage=max_lev,
        )
    elif strategy_name.startswith("pair_arbitrage_"):
        variant = strategy_name.rsplit("_", 1)[-1]
        strategy = PairArbitrageStrategy(
            timeframe=safe_timeframe,
            variant=variant,
            reference_symbol="BTC/USDT",
            asset_symbol=symbol,
            start_date=start_date,
            end_date=end_date,
            enable_shorts=enable_shorts,
            min_leverage=min_lev,
            max_leverage=max_lev,
        )
    elif strategy_name == "mde_mad_entropy":
        strategy = MDEMADEntropyStrategy(
            timeframe=safe_timeframe,
            enable_shorts=enable_shorts,
            min_leverage=min_lev,
            max_leverage=max_lev,
        )
    elif strategy_name == "mde_mad_classic":
        strategy = ClassicMDEMADEntropyStrategy(
            timeframe=safe_timeframe,
            enable_shorts=enable_shorts,
            min_leverage=min_lev,
            max_leverage=max_lev,
        )
    elif strategy_name == "mde_mad_v2":
        strategy = MDEMADV2Strategy(
            timeframe=safe_timeframe,
            enable_shorts=enable_shorts,
            min_leverage=min_lev,
            max_leverage=max_lev,
        )
    elif strategy_name in {"mde_mad_v2_leverage", "mde_mad-v2_leverage"}:
        strategy = MDEMADV2LeverageStrategy(
            timeframe=safe_timeframe,
            enable_shorts=enable_shorts,
            min_leverage=min_lev,
            max_leverage=max_lev,
        )
    elif strategy_name == "mde_mad_v3":
        strategy = MDEMADV3Strategy(
            timeframe=safe_timeframe,
            enable_shorts=enable_shorts,
            min_leverage=min_lev,
            max_leverage=max_lev,
        )
    elif strategy_name == "mde_mad_v3_1":
        strategy = MDEMADV3_1Strategy(
            timeframe=safe_timeframe,
            enable_shorts=enable_shorts,
            min_leverage=min_lev,
            max_leverage=max_lev,
        )
    elif strategy_name == "mde_mad_v4":
        strategy = MDEMADV4Strategy(
            timeframe=safe_timeframe,
            enable_shorts=enable_shorts,
            min_leverage=min_lev,
            max_leverage=max_lev,
        )
    else:
        strategy = DemoStrategy()

    runner = SimulationRunner(
        strategy,
        symbol,
        start_date,
        end_date,
        timeframe=safe_timeframe,
        leverage=lev,
    )

    # Callback to push events to queue
    def on_event(event):
        try:
            # Construct message
            # We assume event.payload is a dictionary
            data = {
                "type": event.type,
                "timestamp": event.timestamp.isoformat(),
                "payload": event.payload
            }
            event_queue.put(data)
        except Exception as e:
            print(f"Error processing event for WS: {e}")

    # Run simulation in a separate thread
    loop = asyncio.get_event_loop()

    # We use a future to track if simulation is done
    # Note: run_in_executor returns a Future
    callback = on_event if stream else None
    simulation_task = loop.run_in_executor(None, runner.run, callback)

    try:
        while not simulation_task.done() or not event_queue.empty():
            try:
                # Poll the queue
                # We check queue size to avoid busy waiting too hard, but strictly speaking
                # queue.get_nowait() is fast.
                try:
                    event_data = event_queue.get_nowait()
                    # Use custom encoder for serialization
                    json_str = json.dumps(_json_safe(event_data), cls=DateTimeEncoder, allow_nan=False)
                    await websocket.send_text(json_str)
                except queue.Empty:
                    await asyncio.sleep(0.01)

            except Exception as e:
                print(f"Error sending data: {e}")
                break

        # Get result
        metrics = {}
        equity_curve = []
        try:
            if simulation_task.done():
                result = simulation_task.result()
                if isinstance(result, dict):
                    metrics = result.get("metrics", {})
                    equity_curve = result.get("equity_curve", [])
        except Exception as e:
            print(f"Error getting simulation result: {e}")

        # Send a completion message
        await websocket.send_text(
            json.dumps(
                _json_safe({
                    "type": "status",
                    "payload": "simulation_complete",
                    "metrics": metrics,
                    "equity_curve": equity_curve,
                }),
                cls=DateTimeEncoder,
                allow_nan=False,
            )
        )

    except WebSocketDisconnect:
        print("Client disconnected")
    except Exception as e:
        print(f"WebSocket error: {e}")
    finally:
        # If simulation is still running (e.g. infinite loop which is not the case here),
        # we can't easily kill it unless we add a stop flag to runner.
        pass

@router.get("/strategies")
def list_strategies():
    """Returns a list of available strategy classes with metadata."""
    from src.simulation.registry import registry
    return {
        "strategies": registry.get_strategy_metadata()
    }
