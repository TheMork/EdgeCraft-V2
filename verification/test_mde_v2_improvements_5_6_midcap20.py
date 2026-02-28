from __future__ import annotations

import contextlib
import io
import os
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.simulation.runner import SimulationRunner
from src.simulation.strategies.mde_mad_v2_leverage import MDEMADV2LeverageStrategy


SYMBOLS: List[str] = [
    "SUI/USDT",
    "APT/USDT",
    "ARB/USDT",
    "SEI/USDT",
    "WLD/USDT",
    "AAVE/USDT",
    "ICP/USDT",
    "GRT/USDT",
    "CRV/USDT",
    "INJ/USDT",
    "ONDO/USDT",
    "QNT/USDT",
    "JUP/USDT",
    "STX/USDT",
    "FET/USDT",
    "OP/USDT",
    "CAKE/USDT",
    "CFX/USDT",
    "ENS/USDT",
    "TIA/USDT",
]

START_DATE = "2024-02-22T00:00:00"
END_DATE = "2026-02-22T00:00:00"
TIMEFRAME = "4h"
INITIAL_BALANCE = 1000.0
BROKER_LEVERAGE = 3


@dataclass
class Config:
    name: str
    kwargs: Dict[str, Any]


def _to_series(equity_curve: List[Dict[str, Any]]) -> pd.Series:
    if not equity_curve:
        return pd.Series(dtype=float)
    df = pd.DataFrame(equity_curve)
    if df.empty or "timestamp" not in df.columns or "equity" not in df.columns:
        return pd.Series(dtype=float)
    ts = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    eq = pd.to_numeric(df["equity"], errors="coerce")
    out = pd.Series(eq.values, index=ts).dropna()
    out = out[~out.index.isna()].sort_index()
    if out.empty:
        return pd.Series(dtype=float)
    return out


def _equity_at_or_before(series: pd.Series, ts: pd.Timestamp) -> float | None:
    if series.empty:
        return None
    pos = series.index.searchsorted(ts, side="right") - 1
    if pos < 0:
        return None
    return float(series.iloc[pos])


def _window_score(series: pd.Series, t_end: pd.Timestamp, lookback_days: int) -> float:
    t_start = t_end - pd.Timedelta(days=lookback_days)
    e0 = _equity_at_or_before(series, t_start)
    e1 = _equity_at_or_before(series, t_end)
    if e0 is None or e1 is None or e0 <= 0.0:
        return -1e9
    window = series[(series.index >= t_start) & (series.index <= t_end)]
    if window.empty:
        return -1e9
    total_ret = (e1 / e0) - 1.0
    running_max = window.cummax()
    dd = float(((running_max - window) / running_max).max())
    return float(total_ret - 0.5 * dd)


def _period_return(series: pd.Series, t0: pd.Timestamp, t1: pd.Timestamp) -> float:
    e0 = _equity_at_or_before(series, t0)
    e1 = _equity_at_or_before(series, t1)
    if e0 is None or e1 is None or e0 <= 0.0:
        return 0.0
    return float((e1 / e0) - 1.0)


def _max_drawdown_from_values(values: List[float]) -> float:
    if not values:
        return 0.0
    arr = np.asarray(values, dtype=float)
    running_max = np.maximum.accumulate(arr)
    dd = (running_max - arr) / np.maximum(running_max, 1e-12)
    return float(np.max(dd))


def _simulate_dynamic_portfolio(
    series_map: Dict[str, pd.Series],
    lookback_days: int,
    rebalance_days: int,
    top_n: int,
    use_hysteresis: bool,
    hold_buffer: int = 4,
    min_hold_rebalances: int = 2,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    start_ts = pd.Timestamp(START_DATE, tz="UTC")
    end_ts = pd.Timestamp(END_DATE, tz="UTC")

    rebalance_times: List[pd.Timestamp] = []
    t = start_ts + pd.Timedelta(days=lookback_days)
    while t < end_ts:
        rebalance_times.append(t)
        t += pd.Timedelta(days=rebalance_days)
    if not rebalance_times:
        return {
            "final_capital": INITIAL_BALANCE,
            "total_return": 0.0,
            "max_drawdown": 0.0,
            "rebalances": 0,
            "win_rate_period": 0.0,
        }, []

    selected: List[str] = []
    hold_age: Dict[str, int] = {}
    records: List[Dict[str, Any]] = []
    portfolio_equity = INITIAL_BALANCE
    equity_points = [portfolio_equity]
    period_returns: List[float] = []

    for i, t0 in enumerate(rebalance_times):
        t1 = rebalance_times[i + 1] if i + 1 < len(rebalance_times) else end_ts

        scores = {
            sym: _window_score(series_map[sym], t0, lookback_days)
            for sym in SYMBOLS
        }
        ranked = sorted(SYMBOLS, key=lambda s: scores[s], reverse=True)

        if not selected:
            selected = ranked[:top_n]
            hold_age = {s: 0 for s in selected}
        elif use_hysteresis:
            rank_pos = {sym: idx + 1 for idx, sym in enumerate(ranked)}
            kept: List[str] = []
            for sym in selected:
                age = hold_age.get(sym, 0)
                if age < min_hold_rebalances:
                    kept.append(sym)
                    continue
                if rank_pos.get(sym, 9999) <= (top_n + hold_buffer):
                    kept.append(sym)

            slots = max(0, top_n - len(kept))
            add_candidates = [sym for sym in ranked if sym not in kept][:slots]
            selected = kept + add_candidates

            new_hold_age: Dict[str, int] = {}
            for sym in selected:
                if sym in hold_age:
                    new_hold_age[sym] = hold_age[sym] + 1
                else:
                    new_hold_age[sym] = 0
            hold_age = new_hold_age
        else:
            selected = ranked[:top_n]
            hold_age = {s: hold_age.get(s, -1) + 1 for s in selected}

        interval_returns = [_period_return(series_map[sym], t0, t1) for sym in selected]
        period_ret = float(np.mean(interval_returns)) if interval_returns else 0.0
        portfolio_equity *= (1.0 + period_ret)
        equity_points.append(portfolio_equity)
        period_returns.append(period_ret)

        records.append(
            {
                "rebalance_ts": t0.isoformat(),
                "next_ts": t1.isoformat(),
                "picked": ",".join(selected),
                "period_return": period_ret,
                "portfolio_equity": portfolio_equity,
                "mode": "hysteresis" if use_hysteresis else "no_hysteresis",
            }
        )

    summary = {
        "final_capital": float(portfolio_equity),
        "total_return": float((portfolio_equity - INITIAL_BALANCE) / INITIAL_BALANCE),
        "max_drawdown": _max_drawdown_from_values(equity_points),
        "rebalances": len(rebalance_times),
        "win_rate_period": float(np.mean([r > 0.0 for r in period_returns])) if period_returns else 0.0,
    }
    return summary, records


def _run_config(cfg: Config) -> Tuple[List[Dict[str, Any]], Dict[str, pd.Series]]:
    rows: List[Dict[str, Any]] = []
    series_map: Dict[str, pd.Series] = {}
    for sym in SYMBOLS:
        strategy = MDEMADV2LeverageStrategy(
            timeframe=TIMEFRAME,
            enable_shorts=True,
            min_leverage=1,
            max_leverage=3,
            **cfg.kwargs,
        )
        runner = SimulationRunner(
            strategy=strategy,
            symbol=sym,
            start_date=START_DATE,
            end_date=END_DATE,
            timeframe=TIMEFRAME,
            initial_balance=INITIAL_BALANCE,
            leverage=BROKER_LEVERAGE,
            auto_sync_on_missing_data=False,
            allow_trade_backfill=False,
        )
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            res = runner.run()
        metrics = res.get("metrics", {})
        curve = res.get("equity_curve", [])
        series_map[sym] = _to_series(curve)
        rows.append(
            {
                "config": cfg.name,
                "symbol": sym,
                "total_return": float(metrics.get("total_return", 0.0)),
                "sharpe_ratio": float(metrics.get("sharpe_ratio", 0.0)),
                "max_drawdown": float(metrics.get("max_drawdown", 0.0)),
                "total_trades": int(metrics.get("total_trades", 0)),
                "final_equity": float(metrics.get("final_equity", INITIAL_BALANCE)),
            }
        )
    return rows, series_map


def main() -> None:
    configs = [
        Config(
            name="adaptive_no_lev_default",
            kwargs={
                "target_leverage_multiplier": 1.0,
                "max_effective_leverage": 1.0,
                "enable_volatility_targeting": True,
                "volatility_lookback_bars": 60,
                "target_annual_volatility": 0.90,
                "min_volatility_scale": 0.35,
                "max_volatility_scale": 1.25,
                "enable_adaptive_leverage": True,
                "flat_outside_regime": True,
                "strong_trend_slope_threshold": 0.0015,
                "strong_trend_distance_threshold": 0.02,
            },
        ),
        Config(
            name="adaptive_no_lev_strict_turnover",
            kwargs={
                "target_leverage_multiplier": 1.0,
                "max_effective_leverage": 1.0,
                "enable_volatility_targeting": True,
                "volatility_lookback_bars": 60,
                "target_annual_volatility": 0.90,
                "min_volatility_scale": 0.35,
                "max_volatility_scale": 1.25,
                "enable_adaptive_leverage": True,
                "flat_outside_regime": True,
                "strong_trend_slope_threshold": 0.0015,
                "strong_trend_distance_threshold": 0.02,
                # Improvement 5: stricter trading filters
                "turnover_penalty": 0.12,
                "min_rebalance_weight_delta": 0.04,
                "min_rebalance_notional": 40.0,
                "min_edge_over_cost_ratio": 2.0,
            },
        ),
    ]

    all_symbol_rows: List[Dict[str, Any]] = []
    summary_rows: List[Dict[str, Any]] = []
    rebalance_rows: List[Dict[str, Any]] = []

    for cfg in configs:
        print(f"Running config: {cfg.name}")
        symbol_rows, series_map = _run_config(cfg)
        all_symbol_rows.extend(symbol_rows)
        symbol_df = pd.DataFrame(symbol_rows)

        static_final = float(symbol_df["final_equity"].sum() / 20.0)
        static_ret = float((static_final - INITIAL_BALANCE) / INITIAL_BALANCE)
        static_dd = float(symbol_df["max_drawdown"].mean())
        static_sh = float(symbol_df["sharpe_ratio"].mean())
        static_wr = float((symbol_df["total_return"] > 0.0).mean())
        static_trades = int(symbol_df["total_trades"].sum())
        summary_rows.append(
            {
                "config": cfg.name,
                "mode": "static_all20",
                "final_capital": static_final,
                "total_return": static_ret,
                "max_drawdown": static_dd,
                "avg_sharpe": static_sh,
                "win_rate": static_wr,
                "sum_trades": static_trades,
                "rebalances": 0,
            }
        )

        dyn_plain, rec_plain = _simulate_dynamic_portfolio(
            series_map=series_map,
            lookback_days=90,
            rebalance_days=30,
            top_n=12,
            use_hysteresis=False,
        )
        for row in rec_plain:
            row["config"] = cfg.name
            rebalance_rows.append(row)
        summary_rows.append(
            {
                "config": cfg.name,
                "mode": "dynamic_top12_no_hysteresis",
                "final_capital": dyn_plain["final_capital"],
                "total_return": dyn_plain["total_return"],
                "max_drawdown": dyn_plain["max_drawdown"],
                "avg_sharpe": np.nan,
                "win_rate": dyn_plain["win_rate_period"],
                "sum_trades": np.nan,
                "rebalances": dyn_plain["rebalances"],
            }
        )

        dyn_hyst, rec_hyst = _simulate_dynamic_portfolio(
            series_map=series_map,
            lookback_days=90,
            rebalance_days=30,
            top_n=12,
            use_hysteresis=True,
            hold_buffer=4,
            min_hold_rebalances=2,
        )
        for row in rec_hyst:
            row["config"] = cfg.name
            rebalance_rows.append(row)
        summary_rows.append(
            {
                "config": cfg.name,
                "mode": "dynamic_top12_hysteresis",
                "final_capital": dyn_hyst["final_capital"],
                "total_return": dyn_hyst["total_return"],
                "max_drawdown": dyn_hyst["max_drawdown"],
                "avg_sharpe": np.nan,
                "win_rate": dyn_hyst["win_rate_period"],
                "sum_trades": np.nan,
                "rebalances": dyn_hyst["rebalances"],
            }
        )

        print(
            f"  static ret={static_ret:.4f}, dd={static_dd:.4f} | "
            f"dyn_plain ret={dyn_plain['total_return']:.4f}, dd={dyn_plain['max_drawdown']:.4f} | "
            f"dyn_hyst ret={dyn_hyst['total_return']:.4f}, dd={dyn_hyst['max_drawdown']:.4f}"
        )

    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_dir = Path("verification")
    out_dir.mkdir(parents=True, exist_ok=True)

    symbol_path = out_dir / f"mde_v2_improvements_5_6_symbol_results_{ts}.csv"
    summary_path = out_dir / f"mde_v2_improvements_5_6_summary_{ts}.csv"
    rebalance_path = out_dir / f"mde_v2_improvements_6_rebalances_{ts}.csv"

    pd.DataFrame(all_symbol_rows).to_csv(symbol_path, index=False)
    pd.DataFrame(summary_rows).to_csv(summary_path, index=False)
    pd.DataFrame(rebalance_rows).to_csv(rebalance_path, index=False)

    print("\nSummary:")
    print(pd.DataFrame(summary_rows).to_string(index=False))
    print(f"\nSaved: {summary_path}")
    print(f"Saved: {symbol_path}")
    print(f"Saved: {rebalance_path}")


if __name__ == "__main__":
    main()
