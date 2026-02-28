from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from verification.data_cache import load_many
from verification.optimize_mde_v2_strict_rolling import Config, _simulate


START = pd.Timestamp("2022-02-22T00:00:00Z")
END = pd.Timestamp("2026-02-22T00:00:00Z")
TRAIN_END = pd.Timestamp("2024-02-22T00:00:00Z")
TEST_START = pd.Timestamp("2024-02-22T00:00:00Z")
TEST_END = pd.Timestamp("2026-02-22T00:00:00Z")
TIMEFRAME = "1d"

# Fixed top20-by-market-cap universe used in previous run.
TOP20_MARKET_CAP_SYMBOLS = [
    "BTC/USDT", "ETH/USDT", "XRP/USDT", "BNB/USDT", "SOL/USDT", "TRX/USDT", "DOGE/USDT", "ADA/USDT",
    "BCH/USDT", "HYPE/USDT", "CC/USDT", "XMR/USDT", "LINK/USDT", "XLM/USDT", "HBAR/USDT", "LTC/USDT",
    "AVAX/USDT", "ZEC/USDT", "SUI/USDT", "1000SHIB/USDT",
]


BASE_CFG = Config(
    name="base",
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
    dd_guard_threshold=0.0,
    dd_guard_cooldown_bars=0,
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
    out: List[Tuple[pd.Timestamp, pd.Timestamp]] = []
    cur = start
    while cur < end:
        nxt = cur + pd.DateOffset(months=3)
        wnd_end = min(nxt, end) - pd.Timedelta(seconds=1)
        out.append((cur, wnd_end))
        cur = nxt
    return out


def _robust_score(
    train_return: float,
    test_return: float,
    test_dd: float,
    median_fold_return: float,
    worst_fold_return: float,
    worst_fold_dd: float,
    symbol_count: int,
) -> float:
    # Robustness-weighted objective: prefer stable OOS behavior over single huge outlier windows.
    score = (
        0.15 * train_return
        + 0.35 * test_return
        + 0.30 * median_fold_return
        + 0.20 * worst_fold_return
        - 0.35 * test_dd
        - 0.35 * worst_fold_dd
        - 0.005 * max(0, 20 - symbol_count)
    )
    return float(score)


def main() -> None:
    print("Loading bars for fixed top20 market-cap universe...")
    bars_all, missing = load_many(
        TOP20_MARKET_CAP_SYMBOLS,
        timeframe=TIMEFRAME,
        start=START.strftime("%Y-%m-%dT%H:%M:%S"),
        end=END.strftime("%Y-%m-%dT%H:%M:%S"),
        force_refresh=False,
        verbose=False,
    )
    print(f"Loaded active={len(bars_all)} missing={len(missing)}")
    if missing:
        print(f"Missing: {', '.join(missing)}")

    windows = _quarterly_windows(TEST_START, TEST_END)
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_dir = Path("verification")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Robustness filter sweep.
    min_train_bars_options = [0, 365, 600]
    dd_guards = [(0.0, 0), (0.18, 20), (0.15, 30)]
    position_caps = [0.08, 0.10]
    atr_stops = [2.5, 3.0]

    rows: List[Dict] = []
    fold_rows: List[Dict] = []

    cfg_id = 0
    total_cfg = len(min_train_bars_options) * len(dd_guards) * len(position_caps) * len(atr_stops)
    for min_train_bars in min_train_bars_options:
        eligible: Dict[str, pd.DataFrame] = {}
        for sym, df in bars_all.items():
            train_part = df[(df.index >= START) & (df.index < TRAIN_END)]
            test_part = df[(df.index >= TEST_START) & (df.index < TEST_END)]
            if len(train_part) >= min_train_bars and len(test_part) > 0:
                eligible[sym] = df

        if not eligible:
            continue

        for dd_thr, dd_cd in dd_guards:
            for cap in position_caps:
                for atr_stop in atr_stops:
                    cfg_id += 1
                    cfg = Config(
                        **{
                            **BASE_CFG.__dict__,
                            "name": f"robust_{cfg_id:03d}",
                            "position_cap": cap,
                            "atr_stop_mult": atr_stop,
                            "dd_guard_threshold": dd_thr,
                            "dd_guard_cooldown_bars": dd_cd,
                        }
                    )

                    train_bars = _slice_bars(eligible, START, TRAIN_END - pd.Timedelta(seconds=1))
                    test_bars = _slice_bars(eligible, TEST_START, TEST_END - pd.Timedelta(seconds=1))
                    train_res = _simulate(cfg, train_bars)
                    test_res = _simulate(cfg, test_bars)

                    fold_metrics: List[Dict] = []
                    for wi, (ws, we) in enumerate(windows, 1):
                        fold_bars = _slice_bars(eligible, ws, we)
                        r = _simulate(cfg, fold_bars)
                        r["cfg_name"] = cfg.name
                        r["fold"] = wi
                        r["period_start"] = ws
                        r["period_end"] = we
                        fold_rows.append(r)
                        fold_metrics.append(r)

                    fold_df = pd.DataFrame(fold_metrics)
                    median_fold_ret = float(fold_df["total_return"].median())
                    worst_fold_ret = float(fold_df["total_return"].min())
                    worst_fold_dd = float(fold_df["max_drawdown"].max())

                    robust = _robust_score(
                        train_return=float(train_res["total_return"]),
                        test_return=float(test_res["total_return"]),
                        test_dd=float(test_res["max_drawdown"]),
                        median_fold_return=median_fold_ret,
                        worst_fold_return=worst_fold_ret,
                        worst_fold_dd=worst_fold_dd,
                        symbol_count=len(eligible),
                    )
                    row = {
                        "cfg_name": cfg.name,
                        "symbol_count": len(eligible),
                        "min_train_bars": min_train_bars,
                        "position_cap": cap,
                        "atr_stop_mult": atr_stop,
                        "dd_guard_threshold": dd_thr,
                        "dd_guard_cooldown_bars": dd_cd,
                        "train_return": float(train_res["total_return"]),
                        "train_dd": float(train_res["max_drawdown"]),
                        "train_sharpe": float(train_res["sharpe_ratio"]),
                        "test_return": float(test_res["total_return"]),
                        "test_dd": float(test_res["max_drawdown"]),
                        "test_sharpe": float(test_res["sharpe_ratio"]),
                        "median_fold_return": median_fold_ret,
                        "mean_fold_return": float(fold_df["total_return"].mean()),
                        "worst_fold_return": worst_fold_ret,
                        "worst_fold_dd": worst_fold_dd,
                        "median_fold_score": float(fold_df["hard_score"].median()),
                        "robust_score": robust,
                    }
                    rows.append(row)
                    print(
                        f"[{cfg_id}/{total_cfg}] {cfg.name} symbols={len(eligible)} "
                        f"test_ret={row['test_return']:.2%} med_fold={row['median_fold_return']:.2%} "
                        f"worst_fold={row['worst_fold_return']:.2%} robust={robust:.4f}"
                    )

    summary_df = pd.DataFrame(rows).sort_values("robust_score", ascending=False).reset_index(drop=True)
    fold_df_all = pd.DataFrame(fold_rows)

    summary_path = out_dir / f"robustness_search_top20_summary_{ts}.csv"
    detail_path = out_dir / f"robustness_search_top20_folds_{ts}.csv"
    summary_df.to_csv(summary_path, index=False)
    fold_df_all.to_csv(detail_path, index=False)

    print("\nTop 10 robust configs:")
    print(
        summary_df.head(10)[
            [
                "cfg_name",
                "symbol_count",
                "min_train_bars",
                "position_cap",
                "atr_stop_mult",
                "dd_guard_threshold",
                "dd_guard_cooldown_bars",
                "train_return",
                "test_return",
                "test_dd",
                "median_fold_return",
                "worst_fold_return",
                "worst_fold_dd",
                "robust_score",
            ]
        ].to_string(index=False)
    )
    print(f"\nSaved summary: {summary_path}")
    print(f"Saved fold details: {detail_path}")


if __name__ == "__main__":
    main()
