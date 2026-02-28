from __future__ import annotations

import contextlib
import io
import math
import os
import sys
from collections import deque
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.simulation.metrics import calculate_metrics
from verification.backtest_turnover_penalty import (
    BROKER_LEVERAGE,
    INITIAL_BALANCE,
    STRICT_EXECUTION,
    SYMBOLS,
    ServerLikeMdeMadV2Strategy,
    SharedPortfolioBroker,
    _build_events,
)
from verification.data_cache import load_many

START_DATE = os.getenv("EDGECRAFT_START_DATE", "2024-02-22T00:00:00")
END_DATE = os.getenv("EDGECRAFT_END_DATE", "2026-02-22T00:00:00")


@dataclass
class Config:
    name: str
    timeframe: str
    lookback_bars: int
    risk_aversion: float
    turnover_penalty: float
    entropy_weight: float
    trend_filter_period: int
    trend_filter_type: str
    max_leverage: float
    enable_shorts: bool
    execution: str = STRICT_EXECUTION.name
    vol_target_annual: float = 0.0
    position_cap: float = 1.0
    vol_lookback: int = 30
    max_holding_bars: int = 0
    atr_period: int = 14
    atr_stop_mult: float = 0.0
    rsi_period: int = 14
    rsi_overbought: float = 100.0
    rsi_oversold: float = 0.0
    dd_guard_threshold: float = 0.0
    dd_guard_cooldown_bars: int = 0


def _hard_score(metrics: Dict[str, Any], turnover_penalty: float) -> float:
    ret = float(metrics.get("total_return", 0.0))
    dd = float(metrics.get("max_drawdown", 0.0))
    sharpe = float(metrics.get("sharpe_ratio", 0.0))
    trades = int(metrics.get("total_trades", 0))
    score = ret - (0.70 * dd) + (0.20 * sharpe) - (0.90 * float(turnover_penalty)) - (0.00003 * trades)
    if trades <= 0:
        score -= 2.0
    return float(score)


def _load_cached_bars(timeframe: str) -> Tuple[Dict[str, pd.DataFrame], List[str]]:
    bars, missing = load_many(
        SYMBOLS,
        timeframe=timeframe,
        start=START_DATE,
        end=END_DATE,
        force_refresh=False,
        verbose=False,
    )
    return bars, missing


def _slice_bars(
    bars_by_symbol: Dict[str, pd.DataFrame],
    start_ts: pd.Timestamp,
    end_ts: pd.Timestamp,
) -> Dict[str, pd.DataFrame]:
    sliced: Dict[str, pd.DataFrame] = {}
    for sym, df in bars_by_symbol.items():
        part = df[(df.index >= start_ts) & (df.index <= end_ts)]
        if not part.empty:
            sliced[sym] = part
    return sliced


def _compute_rsi(closes: Deque[float], period: int) -> Optional[float]:
    if period <= 1 or len(closes) < period + 1:
        return None
    arr = np.asarray(closes, dtype=float)
    deltas = np.diff(arr[-(period + 1) :])
    gains = np.where(deltas > 0, deltas, 0.0)
    losses = np.where(deltas < 0, -deltas, 0.0)
    avg_gain = float(np.mean(gains))
    avg_loss = float(np.mean(losses))
    if avg_loss <= 1e-12:
        return 100.0
    rs = avg_gain / avg_loss
    return float(100.0 - (100.0 / (1.0 + rs)))


def _compute_atr(tr_values: Deque[float], period: int) -> Optional[float]:
    if period <= 1 or len(tr_values) < period:
        return None
    arr = np.asarray(tr_values, dtype=float)
    return float(np.mean(arr[-period:]))


def _position_sign(size: float) -> int:
    if size > 1e-12:
        return 1
    if size < -1e-12:
        return -1
    return 0


def _has_open_order_for_symbol(broker: SharedPortfolioBroker, symbol: str) -> bool:
    for o in broker.open_orders.values():
        if str(o.symbol) == symbol:
            return True
    return False


def _simulate(config: Config, bars_by_symbol: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
    active_symbols = sorted(list(bars_by_symbol.keys()))
    if not active_symbols:
        return {
            **asdict(config),
            "active_symbols": 0,
            "total_return": 0.0,
            "max_drawdown": 0.0,
            "sharpe_ratio": 0.0,
            "sortino_ratio": 0.0,
            "total_trades": 0,
            "final_equity": INITIAL_BALANCE,
            "hard_score": -2.0,
            "avg_gross_exposure": 0.0,
            "max_gross_exposure": 0.0,
        }

    events_by_ts = _build_events(bars_by_symbol)
    timestamps = sorted(events_by_ts.keys())
    broker = SharedPortfolioBroker(
        initial_balance=INITIAL_BALANCE,
        leverage=BROKER_LEVERAGE,
        execution=STRICT_EXECUTION,
    )

    hist_len = max(200, config.lookback_bars + 10, config.atr_period + 10, config.rsi_period + 10)
    close_hist: Dict[str, Deque[float]] = {sym: deque(maxlen=hist_len) for sym in active_symbols}
    tr_hist: Dict[str, Deque[float]] = {sym: deque(maxlen=hist_len) for sym in active_symbols}
    symbol_bar_idx: Dict[str, int] = {sym: 0 for sym in active_symbols}
    pos_state: Dict[str, Dict[str, float]] = {
        sym: {"sign": 0.0, "entry_bar": -1.0, "peak": 0.0, "trough": 0.0}
        for sym in active_symbols
    }

    strategies: Dict[str, ServerLikeMdeMadV2Strategy] = {}
    equity_curve: List[Dict[str, Any]] = []
    gross_exposure_points: List[float] = []
    equity_points: List[float] = []
    equity_peak = float(INITIAL_BALANCE)
    dd_cooldown = 0

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
            for bar in events_by_ts[ts]:
                sym = str(bar["symbol"])
                fills = broker.process_bar(bar)
                for tr in fills:
                    fill_strat = strategies.get(str(tr.symbol))
                    if fill_strat is not None:
                        fill_strat.on_fill(tr)
                strat = strategies.get(sym)
                if strat is not None:
                    strat.on_bar(bar)

                close = float(bar["close"])
                high = float(bar.get("high", close))
                low = float(bar.get("low", close))
                prev_close = close_hist[sym][-1] if close_hist[sym] else close
                close_hist[sym].append(close)
                tr = max(high - low, abs(high - prev_close), abs(low - prev_close))
                tr_hist[sym].append(float(tr))
                symbol_bar_idx[sym] += 1

                pos = broker.get_position(sym)
                size = float(pos.size) if pos is not None else 0.0
                sign = _position_sign(size)
                st = pos_state[sym]
                prev_sign = int(st["sign"])
                if sign != prev_sign:
                    if sign == 0:
                        st["entry_bar"] = -1.0
                        st["peak"] = 0.0
                        st["trough"] = 0.0
                    else:
                        st["entry_bar"] = float(symbol_bar_idx[sym])
                        st["peak"] = close
                        st["trough"] = close
                else:
                    if sign > 0:
                        st["peak"] = max(float(st["peak"]), close)
                    elif sign < 0:
                        if st["trough"] == 0.0:
                            st["trough"] = close
                        st["trough"] = min(float(st["trough"]), close)
                st["sign"] = float(sign)

            ordered_symbols = list(strategies.keys())
            raw_weights = np.array(
                [getattr(strategies[sym], "latest_target_weight", None) or 0.0 for sym in ordered_symbols],
                dtype=float,
            )
            rebalance_eligible = [
                getattr(strategies[sym], "latest_target_weight", None) is not None for sym in ordered_symbols
            ]
            force_exit_flags = [False for _ in ordered_symbols]

            total_abs = float(np.sum(np.abs(raw_weights)))
            vol_scale = 1.0
            if config.vol_target_annual > 0 and len(equity_points) > (config.vol_lookback + 2):
                arr = np.asarray(equity_points[-(config.vol_lookback + 1) :], dtype=float)
                arr = np.maximum(arr, 1e-9)
                rets = np.diff(np.log(arr))
                if len(rets) > 5:
                    realized_ann = float(np.std(rets) * np.sqrt(365.0))
                    if math.isfinite(realized_ann) and realized_ann > 1e-9:
                        vol_scale = float(np.clip(config.vol_target_annual / realized_ann, 0.35, 1.25))

            gross_target = float(config.max_leverage) * vol_scale
            scale = min(1.0, gross_target / total_abs) if total_abs > gross_target and total_abs > 0 else 1.0
            final_weights = raw_weights * scale

            if config.position_cap > 0:
                final_weights = np.clip(final_weights, -config.position_cap, config.position_cap)
                final_abs = float(np.sum(np.abs(final_weights)))
                if final_abs > gross_target and final_abs > 0:
                    final_weights = final_weights * (gross_target / final_abs)

            # Portfolio drawdown guard: flatten and cooldown after a large peak-to-trough drop.
            broker._update_pnl()
            pre_eq = max(1e-9, float(broker.equity))
            equity_peak = max(equity_peak, pre_eq)
            cur_dd = 1.0 - (pre_eq / max(1e-9, equity_peak))
            if config.dd_guard_threshold > 0 and config.dd_guard_cooldown_bars > 0:
                if cur_dd >= float(config.dd_guard_threshold) and dd_cooldown <= 0:
                    dd_cooldown = int(config.dd_guard_cooldown_bars)
                if dd_cooldown > 0:
                    final_weights[:] = 0.0
                    dd_cooldown -= 1

            for idx, sym in enumerate(ordered_symbols):
                target = float(final_weights[idx])
                rsi = _compute_rsi(close_hist[sym], config.rsi_period)
                if rsi is not None:
                    if target > 0 and rsi >= float(config.rsi_overbought):
                        target = 0.0
                    elif target < 0 and rsi <= float(config.rsi_oversold):
                        target = 0.0

                st = pos_state[sym]
                sign = int(st["sign"])
                if sign != 0:
                    bar_i = symbol_bar_idx[sym]
                    entry_bar = int(st["entry_bar"])
                    hold_exit = False
                    if config.max_holding_bars > 0 and entry_bar >= 0:
                        hold_exit = (bar_i - entry_bar) >= int(config.max_holding_bars)

                    atr_exit = False
                    if config.atr_stop_mult > 0:
                        atr = _compute_atr(tr_hist[sym], config.atr_period)
                        if atr is not None and atr > 1e-12 and close_hist[sym]:
                            px = float(close_hist[sym][-1])
                            if sign > 0:
                                stop_px = float(st["peak"]) - float(config.atr_stop_mult) * atr
                                atr_exit = px <= stop_px
                            else:
                                stop_px = float(st["trough"]) + float(config.atr_stop_mult) * atr
                                atr_exit = px >= stop_px

                    if hold_exit or atr_exit:
                        target = 0.0
                        force_exit_flags[idx] = True

                final_weights[idx] = target

            for idx, sym in enumerate(ordered_symbols):
                strat = strategies[sym]
                should_rebalance = rebalance_eligible[idx] or force_exit_flags[idx]
                if not should_rebalance:
                    continue
                if _has_open_order_for_symbol(broker, sym):
                    strat.latest_target_weight = None
                    continue
                target = float(final_weights[idx]) if rebalance_eligible[idx] else 0.0
                strat._rebalance_server(sym, target, strat.latest_close, strat.latest_ts)
                if abs(target) >= strat.entry_weight_threshold:
                    strat.current_weight = target
                elif abs(target) < 1e-9:
                    strat.current_weight = 0.0
                strat.latest_target_weight = None

            broker._update_pnl()
            eq = max(1e-9, float(broker.equity))
            ge = sum(
                abs(float(pos.size) * float(broker.last_prices.get(sym, float(pos.entry_price))))
                for sym, pos in broker.positions.items()
            )
            gross_exposure_points.append(ge / eq)
            equity_points.append(eq)
            equity_curve.append({"timestamp": ts, "equity": eq, "equity_worst": eq})

        for strat in strategies.values():
            strat.on_stop()

    metrics = calculate_metrics(broker.trades, equity_curve)
    out = {
        **asdict(config),
        "active_symbols": len(active_symbols),
        "total_return": float(metrics.get("total_return", 0.0)),
        "max_drawdown": float(metrics.get("max_drawdown", 0.0)),
        "sharpe_ratio": float(metrics.get("sharpe_ratio", 0.0)),
        "sortino_ratio": float(metrics.get("sortino_ratio", 0.0)),
        "total_trades": int(metrics.get("total_trades", 0)),
        "final_equity": float(metrics.get("final_equity", INITIAL_BALANCE)),
        "avg_gross_exposure": float(np.mean(gross_exposure_points)) if gross_exposure_points else 0.0,
        "max_gross_exposure": float(np.max(gross_exposure_points)) if gross_exposure_points else 0.0,
    }
    out["hard_score"] = _hard_score(out, config.turnover_penalty)
    return out


def _rolling_windows(timestamps: List[pd.Timestamp]) -> List[Tuple[pd.Timestamp, pd.Timestamp]]:
    n = len(timestamps)
    if n < 120:
        return [(timestamps[0], timestamps[-1])]
    test_len = max(60, int(n * 0.20))
    step = max(45, int(n * 0.15))
    start = max(0, int(n * 0.35))
    out: List[Tuple[pd.Timestamp, pd.Timestamp]] = []
    while start + test_len <= n:
        s = timestamps[start]
        e = timestamps[start + test_len - 1]
        out.append((s, e))
        start += step
    return out[-4:] if len(out) > 4 else out


def _run_configs(configs: List[Config], bars_by_symbol: Dict[str, pd.DataFrame], tag: str) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    total = len(configs)
    for i, cfg in enumerate(configs, 1):
        r = _simulate(cfg, bars_by_symbol)
        rows.append(r)
        if i % 12 == 0 or i == total:
            print(f"[{tag}] {i}/{total} done")
    df = pd.DataFrame(rows)
    return df.sort_values("hard_score", ascending=False).reset_index(drop=True)


def main() -> None:
    out_dir = Path("verification")
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

    print("Phase 0: Timeframe-Check (strict_next_open)")
    tf_rows: List[Dict[str, Any]] = []
    for tf in ["4h", "8h", "12h", "1d"]:
        bars, missing = _load_cached_bars(tf)
        base_cfg = Config(
            name=f"tf_check_{tf}",
            timeframe=tf,
            lookback_bars=80,
            risk_aversion=4.5,
            turnover_penalty=0.09,
            entropy_weight=0.08,
            trend_filter_period=0,
            trend_filter_type="EMA",
            max_leverage=3.0,
            enable_shorts=True,
            vol_target_annual=0.0,
            position_cap=0.10,
            max_holding_bars=0,
            atr_period=14,
            atr_stop_mult=0.0,
            rsi_period=14,
            rsi_overbought=100.0,
            rsi_oversold=0.0,
        )
        row = _simulate(base_cfg, bars)
        row["missing_symbols"] = len(missing)
        tf_rows.append(row)

    tf_df = pd.DataFrame(tf_rows).sort_values("hard_score", ascending=False).reset_index(drop=True)
    tf_df.to_csv(out_dir / f"mde_v2_phase0_timeframe_check_{ts}.csv", index=False)
    tf_valid = tf_df[(tf_df["active_symbols"] > 0) & (tf_df["total_trades"] > 0)]
    if tf_valid.empty:
        raise RuntimeError("Kein gueltiger Timeframe mit aktiven Trades gefunden.")
    best_tf = str(tf_valid.iloc[0]["timeframe"])
    print(f"Best valid timeframe: {best_tf}")

    bars_full, missing_full = _load_cached_bars(best_tf)
    if not bars_full:
        raise RuntimeError(f"Keine Daten geladen fuer {best_tf}.")
    print(f"Loaded {len(bars_full)} active symbols, {len(missing_full)} missing.")

    print("Phase A: Local strict fine sweep")
    p1_configs: List[Config] = []
    i = 0
    for lb in [70, 80, 90]:
        for ra in [4.5, 5.0, 5.5]:
            for pen in [0.09, 0.10, 0.11]:
                for ew in [0.08, 0.10]:
                    for ml in [2.5, 3.0]:
                        i += 1
                        p1_configs.append(
                            Config(
                                name=f"p1_{i:03d}",
                                timeframe=best_tf,
                                lookback_bars=lb,
                                risk_aversion=ra,
                                turnover_penalty=pen,
                                entropy_weight=ew,
                                trend_filter_period=0,
                                trend_filter_type="EMA",
                                max_leverage=ml,
                                enable_shorts=True,
                                vol_target_annual=0.0,
                                position_cap=0.10,
                            )
                        )
    p1_df = _run_configs(p1_configs, bars_full, "phaseA")
    p1_df.to_csv(out_dir / f"mde_v2_phaseA_local_strict_{ts}.csv", index=False)
    top_p1 = p1_df[(p1_df["total_trades"] > 0)].head(4)
    if top_p1.empty:
        raise RuntimeError("Phase A hatte keine Trades.")

    print("Phase B: Exits + RSI filter sweep")
    p2_configs: List[Config] = []
    j = 0
    rsi_modes = [
        (100.0, 0.0),   # off
        (70.0, 30.0),   # standard veto
        (65.0, 35.0),   # strict veto
    ]
    for _, r in top_p1.iterrows():
        for hold_bars in [0, 20, 30, 45]:
            for atr_mult in [0.0, 2.5, 3.0]:
                for rsi_ob, rsi_os in rsi_modes:
                    j += 1
                    p2_configs.append(
                        Config(
                            name=f"p2_{j:03d}",
                            timeframe=best_tf,
                            lookback_bars=int(r["lookback_bars"]),
                            risk_aversion=float(r["risk_aversion"]),
                            turnover_penalty=float(r["turnover_penalty"]),
                            entropy_weight=float(r["entropy_weight"]),
                            trend_filter_period=int(r["trend_filter_period"]),
                            trend_filter_type=str(r["trend_filter_type"]),
                            max_leverage=float(r["max_leverage"]),
                            enable_shorts=bool(r["enable_shorts"]),
                            vol_target_annual=0.0,
                            position_cap=0.10,
                            max_holding_bars=int(hold_bars),
                            atr_period=14,
                            atr_stop_mult=float(atr_mult),
                            rsi_period=14,
                            rsi_overbought=float(rsi_ob),
                            rsi_oversold=float(rsi_os),
                        )
                    )
    p2_df = _run_configs(p2_configs, bars_full, "phaseB")
    p2_df.to_csv(out_dir / f"mde_v2_phaseB_exits_rsi_{ts}.csv", index=False)

    print("Phase C: Rolling walk-forward (strict)")
    union_ts = sorted({idx for df in bars_full.values() for idx in df.index})
    windows = _rolling_windows(union_ts)
    if not windows:
        raise RuntimeError("Keine Rolling-Fenster erzeugt.")

    wf_candidates = p2_df[(p2_df["total_trades"] > 0)].head(6)
    wf_rows: List[Dict[str, Any]] = []
    for _, cand in wf_candidates.iterrows():
        cfg = Config(
            name=str(cand["name"]),
            timeframe=best_tf,
            lookback_bars=int(cand["lookback_bars"]),
            risk_aversion=float(cand["risk_aversion"]),
            turnover_penalty=float(cand["turnover_penalty"]),
            entropy_weight=float(cand["entropy_weight"]),
            trend_filter_period=int(cand["trend_filter_period"]),
            trend_filter_type=str(cand["trend_filter_type"]),
            max_leverage=float(cand["max_leverage"]),
            enable_shorts=bool(cand["enable_shorts"]),
            vol_target_annual=float(cand["vol_target_annual"]),
            position_cap=float(cand["position_cap"]),
            max_holding_bars=int(cand["max_holding_bars"]),
            atr_period=int(cand["atr_period"]),
            atr_stop_mult=float(cand["atr_stop_mult"]),
            rsi_period=int(cand["rsi_period"]),
            rsi_overbought=float(cand["rsi_overbought"]),
            rsi_oversold=float(cand["rsi_oversold"]),
        )
        for wi, (ws, we) in enumerate(windows, 1):
            part = _slice_bars(bars_full, ws, we)
            res = _simulate(cfg, part)
            res["window_id"] = wi
            res["window_start"] = ws
            res["window_end"] = we
            wf_rows.append(res)
        print(f"[walkforward] {cfg.name} finished")

    wf_df = pd.DataFrame(wf_rows)
    wf_df.to_csv(out_dir / f"mde_v2_phaseC_walkforward_detail_{ts}.csv", index=False)

    grp = wf_df.groupby("name", as_index=False).agg(
        median_hard_score=("hard_score", "median"),
        median_return=("total_return", "median"),
        median_drawdown=("max_drawdown", "median"),
        worst_return=("total_return", "min"),
        windows=("window_id", "nunique"),
    )
    wf_cfg_meta = p2_df.drop_duplicates(subset=["name"])[
        [
            "name",
            "timeframe",
            "lookback_bars",
            "risk_aversion",
            "turnover_penalty",
            "entropy_weight",
            "trend_filter_period",
            "trend_filter_type",
            "max_leverage",
            "enable_shorts",
            "max_holding_bars",
            "atr_period",
            "atr_stop_mult",
            "rsi_period",
            "rsi_overbought",
            "rsi_oversold",
        ]
    ]
    wf_summary = grp.merge(wf_cfg_meta, on="name", how="left")
    wf_summary = wf_summary.sort_values(
        ["median_hard_score", "median_return", "worst_return"],
        ascending=[False, False, False],
    ).reset_index(drop=True)
    wf_summary.to_csv(out_dir / f"mde_v2_phaseC_walkforward_summary_{ts}.csv", index=False)

    print("\nTop 10 Phase B (exits + rsi):")
    print(
        p2_df.head(10)[
            [
                "name",
                "lookback_bars",
                "risk_aversion",
                "turnover_penalty",
                "entropy_weight",
                "max_holding_bars",
                "atr_stop_mult",
                "rsi_overbought",
                "rsi_oversold",
                "total_return",
                "max_drawdown",
                "sharpe_ratio",
                "total_trades",
                "hard_score",
            ]
        ].to_string(index=False)
    )

    print("\nTop Walk-forward by median score:")
    print(
        wf_summary.head(10)[
            [
                "name",
                "median_hard_score",
                "median_return",
                "median_drawdown",
                "worst_return",
                "lookback_bars",
                "risk_aversion",
                "turnover_penalty",
                "entropy_weight",
                "max_holding_bars",
                "atr_stop_mult",
                "rsi_overbought",
                "rsi_oversold",
                "windows",
            ]
        ].to_string(index=False)
    )

    best = wf_summary.iloc[0].to_dict()
    print("\nRecommended config:")
    print(best)
    print(f"\nSaved outputs with timestamp: {ts}")


if __name__ == "__main__":
    main()
