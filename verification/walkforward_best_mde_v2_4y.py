from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from verification.data_cache import load_many
from verification.optimize_mde_v2_strict_rolling import Config, _simulate
from verification.backtest_turnover_penalty import SYMBOLS
from src.data_manager import SyncManager
from src.database import QuestDBManager


GLOBAL_START = pd.Timestamp("2022-02-22T00:00:00Z")
GLOBAL_END = pd.Timestamp("2026-02-22T00:00:00Z")
TRAIN_END = pd.Timestamp("2024-02-22T00:00:00Z")
TEST_START = pd.Timestamp("2024-02-22T00:00:00Z")
TEST_END = pd.Timestamp("2026-02-22T00:00:00Z")
TIMEFRAME = "1d"


BEST_CONFIG = Config(
    name="best_cfg_20260227",
    timeframe=TIMEFRAME,
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
    max_holding_bars=20,
    atr_period=14,
    atr_stop_mult=3.0,
    rsi_period=14,
    rsi_overbought=100.0,
    rsi_oversold=0.0,
)


def _slice_bars(
    bars_by_symbol: Dict[str, pd.DataFrame],
    start_ts: pd.Timestamp,
    end_ts: pd.Timestamp,
) -> Dict[str, pd.DataFrame]:
    out: Dict[str, pd.DataFrame] = {}
    for sym, df in bars_by_symbol.items():
        part = df[(df.index >= start_ts) & (df.index <= end_ts)]
        if not part.empty:
            out[sym] = part
    return out


def _quarterly_windows(start: pd.Timestamp, end: pd.Timestamp) -> List[Tuple[pd.Timestamp, pd.Timestamp]]:
    windows: List[Tuple[pd.Timestamp, pd.Timestamp]] = []
    cur = start
    while cur < end:
        nxt = cur + pd.DateOffset(months=3)
        wnd_end = min(nxt, end) - pd.Timedelta(seconds=1)
        windows.append((cur, wnd_end))
        cur = nxt
    return windows


def _parse_symbols_env(raw: str) -> List[str]:
    out: List[str] = []
    for part in raw.split(","):
        sym = part.strip().upper()
        if not sym:
            continue
        if "/" not in sym and sym.endswith("USDT") and len(sym) > 4:
            sym = f"{sym[:-4]}/USDT"
        if sym not in out:
            out.append(sym)
    return out


def _resolve_symbols() -> List[str]:
    raw_symbols = os.getenv("EDGECRAFT_SYMBOLS", "").strip()
    if raw_symbols:
        parsed = _parse_symbols_env(raw_symbols)
        if parsed:
            return parsed

    universe = os.getenv("EDGECRAFT_SYMBOL_UNIVERSE", "default").strip().lower()
    if universe in {"top20_market_cap", "market_cap_top20", "coingecko_top20"}:
        sm = SyncManager(db_manager=QuestDBManager())
        symbols = sm.get_top_20_symbols(source="market_cap")
        if symbols:
            return symbols
    if universe in {"top20", "top20_quote_volume", "quote_volume_top20"}:
        sm = SyncManager(db_manager=QuestDBManager())
        symbols = sm.get_top_20_symbols(source="quote_volume")
        if symbols:
            return symbols

    return SYMBOLS


def _config_from_env(base: Config) -> Config:
    cfg = Config(**base.__dict__)

    def _f(name: str, cur: float) -> float:
        raw = os.getenv(name, "").strip()
        if not raw:
            return cur
        try:
            return float(raw)
        except ValueError:
            return cur

    def _i(name: str, cur: int) -> int:
        raw = os.getenv(name, "").strip()
        if not raw:
            return cur
        try:
            return int(raw)
        except ValueError:
            return cur

    cfg.risk_aversion = _f("EDGECRAFT_CFG_RISK_AVERSION", cfg.risk_aversion)
    cfg.turnover_penalty = _f("EDGECRAFT_CFG_TURNOVER_PENALTY", cfg.turnover_penalty)
    cfg.entropy_weight = _f("EDGECRAFT_CFG_ENTROPY_WEIGHT", cfg.entropy_weight)
    cfg.max_leverage = _f("EDGECRAFT_CFG_MAX_LEVERAGE", cfg.max_leverage)
    cfg.position_cap = _f("EDGECRAFT_CFG_POSITION_CAP", cfg.position_cap)
    cfg.max_holding_bars = _i("EDGECRAFT_CFG_MAX_HOLDING_BARS", cfg.max_holding_bars)
    cfg.atr_period = _i("EDGECRAFT_CFG_ATR_PERIOD", cfg.atr_period)
    cfg.atr_stop_mult = _f("EDGECRAFT_CFG_ATR_STOP_MULT", cfg.atr_stop_mult)
    cfg.dd_guard_threshold = _f("EDGECRAFT_CFG_DD_GUARD_THRESHOLD", cfg.dd_guard_threshold)
    cfg.dd_guard_cooldown_bars = _i("EDGECRAFT_CFG_DD_GUARD_COOLDOWN_BARS", cfg.dd_guard_cooldown_bars)
    return cfg


def main() -> None:
    force_refresh = os.getenv("EDGECRAFT_CACHE_REFRESH", "").strip().lower() in {"1", "true", "yes"}
    min_train_bars_raw = os.getenv("EDGECRAFT_MIN_TRAIN_BARS", "0").strip()
    try:
        min_train_bars = max(0, int(min_train_bars_raw))
    except ValueError:
        min_train_bars = 0
    cfg = _config_from_env(BEST_CONFIG)
    symbols = _resolve_symbols()
    print(
        "Walk-forward setup: "
        f"global={GLOBAL_START.date()}..{GLOBAL_END.date()} | "
        f"train={GLOBAL_START.date()}..{TRAIN_END.date()} | "
        f"test={TEST_START.date()}..{TEST_END.date()} | "
        f"cache_refresh={force_refresh} | symbols={len(symbols)} | min_train_bars={min_train_bars}"
    )
    print(f"Symbol universe: {', '.join(symbols)}")

    bars, missing = load_many(
        symbols,
        timeframe=TIMEFRAME,
        start=GLOBAL_START.strftime("%Y-%m-%dT%H:%M:%S"),
        end=GLOBAL_END.strftime("%Y-%m-%dT%H:%M:%S"),
        force_refresh=force_refresh,
        verbose=False,
    )
    print(f"Loaded symbols: active={len(bars)} missing={len(missing)}")

    if min_train_bars > 0:
        filtered: Dict[str, pd.DataFrame] = {}
        for sym, df in bars.items():
            train_part = df[(df.index >= GLOBAL_START) & (df.index < TRAIN_END)]
            test_part = df[(df.index >= TEST_START) & (df.index < TEST_END)]
            if len(train_part) >= min_train_bars and len(test_part) > 0:
                filtered[sym] = df
        bars = filtered
        print(f"After min_train_bars filter: active={len(bars)}")

    train_bars = _slice_bars(bars, GLOBAL_START, TRAIN_END - pd.Timedelta(seconds=1))
    test_bars = _slice_bars(bars, TEST_START, TEST_END - pd.Timedelta(seconds=1))

    train_res = _simulate(cfg, train_bars)
    train_res["phase"] = "train"
    train_res["period_start"] = GLOBAL_START
    train_res["period_end"] = TRAIN_END - pd.Timedelta(seconds=1)

    test_res = _simulate(cfg, test_bars)
    test_res["phase"] = "test_full"
    test_res["period_start"] = TEST_START
    test_res["period_end"] = TEST_END - pd.Timedelta(seconds=1)

    wf_rows: List[Dict] = []
    for i, (ws, we) in enumerate(_quarterly_windows(TEST_START, TEST_END), 1):
        fold_bars = _slice_bars(bars, ws, we)
        r = _simulate(cfg, fold_bars)
        r["phase"] = "test_fold"
        r["fold"] = i
        r["period_start"] = ws
        r["period_end"] = we
        wf_rows.append(r)

    wf_df = pd.DataFrame(wf_rows)
    summary = {
        "folds": int(len(wf_df)),
        "mean_fold_return": float(wf_df["total_return"].mean()) if not wf_df.empty else 0.0,
        "median_fold_return": float(wf_df["total_return"].median()) if not wf_df.empty else 0.0,
        "worst_fold_return": float(wf_df["total_return"].min()) if not wf_df.empty else 0.0,
        "best_fold_return": float(wf_df["total_return"].max()) if not wf_df.empty else 0.0,
        "mean_fold_drawdown": float(wf_df["max_drawdown"].mean()) if not wf_df.empty else 0.0,
        "worst_fold_drawdown": float(wf_df["max_drawdown"].max()) if not wf_df.empty else 0.0,
        "mean_fold_sharpe": float(wf_df["sharpe_ratio"].mean()) if not wf_df.empty else 0.0,
        "median_fold_hard_score": float(wf_df["hard_score"].median()) if not wf_df.empty else 0.0,
    }

    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_dir = Path("verification")
    out_dir.mkdir(parents=True, exist_ok=True)

    detail_rows = [train_res, test_res] + wf_rows
    detail_df = pd.DataFrame(detail_rows)
    detail_path = out_dir / f"walkforward_best_mde_v2_4y_detail_{ts}.csv"
    summary_path = out_dir / f"walkforward_best_mde_v2_4y_summary_{ts}.csv"
    cfg_path = out_dir / f"walkforward_best_mde_v2_4y_config_{ts}.json"

    detail_df.to_csv(detail_path, index=False)
    pd.DataFrame(
        [
            {
                **summary,
                "train_total_return": train_res["total_return"],
                "train_max_drawdown": train_res["max_drawdown"],
                "train_sharpe_ratio": train_res["sharpe_ratio"],
                "test_total_return": test_res["total_return"],
                "test_max_drawdown": test_res["max_drawdown"],
                "test_sharpe_ratio": test_res["sharpe_ratio"],
                "test_hard_score": test_res["hard_score"],
            }
        ]
    ).to_csv(summary_path, index=False)

    cfg_path.write_text(pd.Series(asdict(cfg)).to_json(indent=2), encoding="utf-8")

    print("\nTrain (2y):")
    print(
        f"return={train_res['total_return']:.2%} dd={train_res['max_drawdown']:.2%} "
        f"sharpe={train_res['sharpe_ratio']:.3f} trades={train_res['total_trades']}"
    )
    print("Test (2y):")
    print(
        f"return={test_res['total_return']:.2%} dd={test_res['max_drawdown']:.2%} "
        f"sharpe={test_res['sharpe_ratio']:.3f} trades={test_res['total_trades']}"
    )
    print("Test walk-forward folds (quarterly across 2y):")
    print(
        f"folds={summary['folds']} mean_ret={summary['mean_fold_return']:.2%} "
        f"median_ret={summary['median_fold_return']:.2%} worst_ret={summary['worst_fold_return']:.2%} "
        f"mean_dd={summary['mean_fold_drawdown']:.2%} worst_dd={summary['worst_fold_drawdown']:.2%} "
        f"median_score={summary['median_fold_hard_score']:.4f}"
    )
    print(f"\nSaved detail: {detail_path}")
    print(f"Saved summary: {summary_path}")
    print(f"Saved config: {cfg_path}")


if __name__ == "__main__":
    main()
