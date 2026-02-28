import contextlib
import io
import os
import sys
import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple
from pathlib import Path

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.simulation.metrics import calculate_metrics
from src.database import QuestDBManager
from verification.data_cache import load_many
from verification.backtest_turnover_penalty import (
    ServerLikeMdeMadV2Strategy,
    SharedPortfolioBroker,
    ExecutionModel,
    LEGACY_EXECUTION,
    STRICT_EXECUTION,
    SYMBOLS,
    INITIAL_BALANCE,
    BROKER_LEVERAGE,
    _build_events,
    _normalize_index,
)

START_DATE = os.getenv("EDGECRAFT_START_DATE", "2024-02-22T00:00:00")
END_DATE = os.getenv("EDGECRAFT_END_DATE", "2026-02-22T00:00:00")

def _gross_exposure(broker: SharedPortfolioBroker) -> float:
    return sum(
        abs(float(pos.size) * float(broker.last_prices.get(sym, float(pos.entry_price))))
        for sym, pos in broker.positions.items()
    )

@dataclass
class SweepRunConfig:
    name: str
    phase: int
    timeframe: str
    lookback_bars: int
    risk_aversion: float
    turnover_penalty: float
    entropy_weight: float
    trend_filter_period: int
    trend_filter_type: str
    max_leverage: float
    enable_shorts: bool
    execution: ExecutionModel

@dataclass
class SweepRunResult:
    run: str
    phase: int
    execution: str
    timeframe: str
    lookback_bars: int
    risk_aversion: float
    turnover_penalty: float
    entropy_weight: float
    trend_filter_period: int
    trend_filter_type: str
    max_leverage: float
    enable_shorts: bool
    active_symbols: int
    missing_symbols: int
    total_return: float
    final_equity: float
    max_drawdown: float
    max_drawdown_close: float
    max_drawdown_worst: float
    sharpe_ratio: float
    sortino_ratio: float
    total_trades: int
    avg_gross_exposure: float
    max_gross_exposure: float
    score_balanced: float

def _load_tf_data(timeframe: str, symbols: List[str], force_refresh: bool = False):
    bars_by_symbol, missing = load_many(
        symbols,
        timeframe=timeframe,
        start=START_DATE,
        end=END_DATE,
        force_refresh=force_refresh,
        verbose=False,
    )
    funding_schedule: Dict[pd.Timestamp, List[Tuple[str, float]]] = {}
    db = QuestDBManager()
    for sym in symbols:
        if sym in missing:
            continue
        try:
            funding = db.get_funding_rates(sym, START_DATE, END_DATE)
            funding = _normalize_index(funding)
            if not funding.empty and "funding_rate" in funding.columns:
                rates = pd.to_numeric(funding["funding_rate"], errors="coerce").dropna()
                for ts, rate in rates.items():
                    ts_dt = ts.to_pydatetime()
                    funding_schedule.setdefault(ts_dt, []).append((sym, float(rate)))
        except Exception:
            pass
    return bars_by_symbol, funding_schedule, missing

def _simulate_sweep(
    config: SweepRunConfig,
    bars_by_symbol: Dict[str, pd.DataFrame],
    funding_schedule: Dict[datetime, List[Tuple[str, float]]],
    missing_symbols: List[str],
) -> SweepRunResult:
    active_symbols = sorted(list(bars_by_symbol.keys()))
    events_by_ts = _build_events(bars_by_symbol)
    timestamps = sorted(events_by_ts.keys())

    broker = SharedPortfolioBroker(
        initial_balance=INITIAL_BALANCE,
        leverage=BROKER_LEVERAGE,
        execution=config.execution,
    )

    strategies: Dict[str, ServerLikeMdeMadV2Strategy] = {}
    equity_curve: List[Dict[str, Any]] = []
    gross_exposure_points: List[float] = []

    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        for sym in active_symbols:
            strat = ServerLikeMdeMadV2Strategy(
                lookback_bars=config.lookback_bars,
                trend_filter_period=config.trend_filter_period,
                risk_aversion=config.risk_aversion,
                entropy_weight=config.entropy_weight,
                turnover_penalty=config.turnover_penalty,
                max_leverage=config.max_leverage,
                enable_shorts=config.enable_shorts,
                optimization_steps=41,
                entry_weight_threshold=0.10,
                trend_filter_type=config.trend_filter_type,
            )
            strat.set_broker(broker)
            strat.on_start()
            strategies[sym] = strat

        for ts in timestamps:
            events = events_by_ts[ts]

            for bar in events:
                sym = str(bar["symbol"])
                fills = broker.process_bar(bar)
                for tr in fills:
                    strat = strategies.get(str(tr.symbol))
                    if strat is not None:
                        strat.on_fill(tr)
                strat = strategies.get(sym)
                if strat is not None:
                    strat.on_bar(bar)

            # Rebalance
            weights = np.array([
                getattr(strat, "latest_target_weight", None) or 0.0
                for strat in strategies.values()
            ], dtype=float)
            total_exposure = float(np.sum(np.abs(weights)))
            scale = min(1.0, 3.0 / total_exposure) if total_exposure > 3.0 else 1.0

            for (sym, strat), raw_w in zip(strategies.items(), weights):
                if getattr(strat, "latest_target_weight", None) is None:
                    continue
                final_target = float(raw_w) * scale
                strat._rebalance_server(sym, final_target, strat.latest_close, strat.latest_ts)
                if abs(final_target) >= strat.entry_weight_threshold:
                    strat.current_weight = final_target
                strat.latest_target_weight = None

            # Funding
            for sym, rate in funding_schedule.get(ts, []):
                broker.process_funding(sym, rate)

            broker._update_pnl()
            ge = _gross_exposure(broker)
            eq = max(1e-9, broker.equity)
            gross_exposure_points.append(ge / eq)
            equity_curve.append(
                {
                    "timestamp": ts,
                    "equity": float(broker.equity),
                    "equity_worst": float(broker.equity),
                }
            )

        for strat in strategies.values():
            strat.on_stop()

    metrics = calculate_metrics(broker.trades, equity_curve)
    ret = float(metrics.get("total_return", 0.0))
    dd = float(metrics.get("max_drawdown", 0.0))
    score = ret - 0.5 * dd

    return SweepRunResult(
        run=config.name,
        phase=config.phase,
        execution=config.execution.name,
        timeframe=config.timeframe,
        lookback_bars=config.lookback_bars,
        risk_aversion=config.risk_aversion,
        turnover_penalty=config.turnover_penalty,
        entropy_weight=config.entropy_weight,
        trend_filter_period=config.trend_filter_period,
        trend_filter_type=config.trend_filter_type,
        max_leverage=config.max_leverage,
        enable_shorts=config.enable_shorts,
        active_symbols=len(active_symbols),
        missing_symbols=len(missing_symbols),
        total_return=ret,
        final_equity=float(metrics.get("final_equity", INITIAL_BALANCE)),
        max_drawdown=dd,
        max_drawdown_close=float(metrics.get("max_drawdown_close", 0.0)),
        max_drawdown_worst=float(metrics.get("max_drawdown_worst", 0.0)),
        sharpe_ratio=float(metrics.get("sharpe_ratio", 0.0)),
        sortino_ratio=float(metrics.get("sortino_ratio", 0.0)),
        total_trades=int(metrics.get("total_trades", 0)),
        avg_gross_exposure=float(np.mean(gross_exposure_points)) if gross_exposure_points else 0.0,
        max_gross_exposure=float(np.max(gross_exposure_points)) if gross_exposure_points else 0.0,
        score_balanced=score,
    )

def _worker(args: Tuple) -> Dict[str, Any]:
    config_dict, bars_by_symbol, funding_schedule, missing_symbols = args
    exec_dict = config_dict.pop("execution")
    execution = ExecutionModel(**exec_dict)
    config = SweepRunConfig(**config_dict, execution=execution)

    res = _simulate_sweep(config, bars_by_symbol, funding_schedule, missing_symbols)
    return res.__dict__

def run_phase(configs: List[SweepRunConfig], worker_count: int) -> List[SweepRunResult]:
    results = []
    
    tf_groups = {}
    for c in configs:
        tf_groups.setdefault(c.timeframe, []).append(c)
        
    for tf, tf_configs in tf_groups.items():
        print(f"\n--- Loading data for TF {tf} ({len(tf_configs)} configs) ---")
        bars_by_symbol, funding_schedule, missing = _load_tf_data(tf, SYMBOLS)
        
        tasks = []
        for c in tf_configs:
            c_dict = c.__dict__.copy()
            c_dict["execution"] = c.execution.__dict__
            tasks.append((c_dict, bars_by_symbol, funding_schedule, missing))
            
        print(f"Running {len(tasks)} tasks...")
        
        if worker_count <= 1:
            for i, t in enumerate(tasks, 1):
                print(f"[{i}/{len(tasks)}] {t[0]['name']} ...", end=" ", flush=True)
                res_dict = _worker(t)
                print(f"✓ ret={res_dict['total_return']:.2%} score={res_dict['score_balanced']:.4f}")
                results.append(SweepRunResult(**res_dict))
        else:
            import multiprocessing
            mp_ctx = multiprocessing.get_context("spawn")
            with ProcessPoolExecutor(max_workers=worker_count, mp_context=mp_ctx) as executor:
                future_to_name = {executor.submit(_worker, t): t[0]["name"] for t in tasks}
                for i, future in enumerate(as_completed(future_to_name), 1):
                    name = future_to_name[future]
                    try:
                        res_dict = future.result()
                        print(f"[{i}/{len(tasks)}] {name} ✓ ret={res_dict['total_return']:.2%} score={res_dict['score_balanced']:.4f}")
                        results.append(SweepRunResult(**res_dict))
                    except Exception as e:
                        print(f"[{i}/{len(tasks)}] {name} ✗ ERROR: {e}")
                        
    return results

def main():
    cpu = os.cpu_count() or 2
    max_workers_env = os.getenv("EDGECRAFT_MAX_WORKERS", "").strip()
    if max_workers_env.isdigit() and int(max_workers_env) > 0:
        max_workers = int(max_workers_env)
    else:
        max_workers = max(2, cpu // 2)

    all_results = []
    
    print("=== PHASE 1: Timeframe & Core Parameters ===")
    timeframes = ["4h", "6h", "8h", "12h", "1d"]
    lookbacks = [15, 30, 50, 80]
    risk_avers = [1.0, 2.0, 3.0, 5.0]
    penalties = [0.01, 0.03, 0.05, 0.08, 0.10]
    
    p1_configs = []
    i = 0
    for tf in timeframes:
        for lb in lookbacks:
            for ra in risk_avers:
                for pen in penalties:
                    i += 1
                    p1_configs.append(SweepRunConfig(
                        name=f"p1_{tf}_lb{lb}_ra{ra}_pen{pen}",
                        phase=1,
                        timeframe=tf,
                        lookback_bars=lb,
                        risk_aversion=ra,
                        turnover_penalty=pen,
                        entropy_weight=0.1,
                        trend_filter_period=200,
                        trend_filter_type="EMA",
                        max_leverage=3.0,
                        enable_shorts=True,
                        execution=LEGACY_EXECUTION,
                    ))
                    
    p1_results = run_phase(p1_configs, max_workers)
    all_results.extend(p1_results)
    
    # Sort Phase 1 results by score
    p1_results.sort(key=lambda x: x.score_balanced, reverse=True)
    top_p1 = p1_results[:5]
    
    print("\n--- Top 5 from Phase 1 ---")
    for idx, r in enumerate(top_p1, 1):
        print(f"#{idx}: {r.run} | tf={r.timeframe}, lb={r.lookback_bars}, ra={r.risk_aversion}, pen={r.turnover_penalty} | ret={r.total_return:.2%} score={r.score_balanced:.4f}")
        
    print("\n=== PHASE 2: Fine-Tuning Top 5 ===")
    entropy_weights = [0.0, 0.1, 0.2]
    trend_options = [(0, "EMA"), (100, "EMA"), (200, "EMA"), (200, "HMA")]
    max_levs = [2.0, 3.0]
    shorts_opts = [True, False]
    
    p2_configs = []
    for base_i, base in enumerate(top_p1, 1):
        for ew in entropy_weights:
            for tp, tt in trend_options:
                for ml in max_levs:
                    for shorts in shorts_opts:
                        p2_configs.append(SweepRunConfig(
                            name=f"p2_b{base_i}_ew{ew}_tp{tp}{tt}_ml{ml}_sh{shorts}",
                            phase=2,
                            timeframe=base.timeframe,
                            lookback_bars=base.lookback_bars,
                            risk_aversion=base.risk_aversion,
                            turnover_penalty=base.turnover_penalty,
                            entropy_weight=ew,
                            trend_filter_period=tp,
                            trend_filter_type=tt,
                            max_leverage=ml,
                            enable_shorts=shorts,
                            execution=LEGACY_EXECUTION,
                        ))
                        
    p2_results = run_phase(p2_configs, max_workers)
    all_results.extend(p2_results)
    
    p2_results.sort(key=lambda x: x.score_balanced, reverse=True)
    top_p2 = p2_results[:10]
    
    print("\n--- Top 10 from Phase 2 ---")
    for idx, r in enumerate(top_p2, 1):
        print(f"#{idx}: {r.run} | score={r.score_balanced:.4f} ret={r.total_return:.2%}")
        
    print("\n=== PHASE 3: STRICT Execution Validation ===")
    p3_configs = []
    for idx, base in enumerate(top_p2, 1):
        config_kwargs = {k: getattr(base, k) for k in [
            "timeframe", "lookback_bars", "risk_aversion", "turnover_penalty", 
            "entropy_weight", "trend_filter_period", "trend_filter_type", 
            "max_leverage", "enable_shorts"
        ]}
        p3_configs.append(SweepRunConfig(
            name=f"p3_strict_top{idx}",
            phase=3,
            execution=STRICT_EXECUTION,
            **config_kwargs
        ))
        
    p3_results = run_phase(p3_configs, max_workers)
    all_results.extend(p3_results)
    
    p3_results.sort(key=lambda x: x.score_balanced, reverse=True)
    
    print("\n=== FINAL STRICT RESULTS ===")
    for idx, r in enumerate(p3_results, 1):
        print(f"#{idx} {r.run} | tf={r.timeframe} score={r.score_balanced:.4f} ret={r.total_return:.2%} sr={r.sharpe_ratio:.2f} dd={r.max_drawdown:.2%}")

    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_dir = Path("verification")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    df = pd.DataFrame([r.__dict__ for r in all_results])
    out_path = out_dir / f"mde_v2_comprehensive_sweep_{ts}.csv"
    df.to_csv(out_path, index=False)
    print(f"\nSaved all results to {out_path}")

if __name__ == "__main__":
    main()
