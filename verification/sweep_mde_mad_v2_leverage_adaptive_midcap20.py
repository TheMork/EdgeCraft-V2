from __future__ import annotations

import contextlib
import io
import itertools
import os
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple

import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.simulation.runner import SimulationRunner
from src.simulation.strategies.mde_mad_v2 import MDEMADV2Strategy
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
class SweepConfig:
    label: str
    strategy_type: str
    kwargs: Dict[str, Any]


def _run_single_symbol(
    symbol: str,
    strategy_factory: Callable[[], Any],
) -> Dict[str, Any]:
    strategy = strategy_factory()
    runner = SimulationRunner(
        strategy=strategy,
        symbol=symbol,
        start_date=START_DATE,
        end_date=END_DATE,
        timeframe=TIMEFRAME,
        initial_balance=INITIAL_BALANCE,
        leverage=BROKER_LEVERAGE,
        auto_sync_on_missing_data=False,
        allow_trade_backfill=False,
    )
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        result = runner.run()

    metrics = result.get("metrics") if isinstance(result, dict) else None
    if not metrics:
        return {"symbol": symbol, "status": "missing"}

    return {
        "symbol": symbol,
        "status": "success",
        "total_return": float(metrics.get("total_return", 0.0)),
        "sharpe_ratio": float(metrics.get("sharpe_ratio", 0.0)),
        "max_drawdown": float(metrics.get("max_drawdown", 0.0)),
        "total_trades": int(metrics.get("total_trades", 0)),
        "final_equity": float(metrics.get("final_equity", INITIAL_BALANCE)),
    }


def _evaluate_config(cfg: SweepConfig) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    if cfg.strategy_type == "v2":
        strategy_factory = lambda: MDEMADV2Strategy(
            timeframe=TIMEFRAME,
            enable_shorts=True,
            min_leverage=1,
            max_leverage=3,
        )
    elif cfg.strategy_type == "lev":
        strategy_factory = lambda: MDEMADV2LeverageStrategy(
            timeframe=TIMEFRAME,
            enable_shorts=True,
            min_leverage=1,
            max_leverage=3,
            **cfg.kwargs,
        )
    else:
        raise ValueError(f"Unknown strategy_type: {cfg.strategy_type}")

    symbol_rows: List[Dict[str, Any]] = []
    for symbol in SYMBOLS:
        row = _run_single_symbol(symbol, strategy_factory)
        row["config"] = cfg.label
        symbol_rows.append(row)

    ok = [r for r in symbol_rows if r["status"] == "success"]
    missing = len(symbol_rows) - len(ok)
    if not ok:
        summary = {
            "config": cfg.label,
            "success": 0,
            "missing": missing,
            "portfolio_final_equal20": float("nan"),
            "portfolio_return_equal20": float("nan"),
            "avg_symbol_return": float("nan"),
            "avg_sharpe": float("nan"),
            "avg_max_drawdown": float("nan"),
            "sum_trades": 0,
            "win_rate": float("nan"),
            "score_balanced": float("nan"),
            "score_conservative": float("nan"),
            "calmar_like": float("nan"),
        }
        return summary, symbol_rows

    final_capital = sum(r["final_equity"] for r in ok) / 20.0
    port_ret = (final_capital - INITIAL_BALANCE) / INITIAL_BALANCE
    avg_ret = sum(r["total_return"] for r in ok) / len(ok)
    avg_sh = sum(r["sharpe_ratio"] for r in ok) / len(ok)
    avg_dd = sum(r["max_drawdown"] for r in ok) / len(ok)
    sum_trades = sum(r["total_trades"] for r in ok)
    win_rate = sum(1 for r in ok if r["total_return"] > 0.0) / len(ok)

    summary = {
        "config": cfg.label,
        "success": len(ok),
        "missing": missing,
        "portfolio_final_equal20": final_capital,
        "portfolio_return_equal20": port_ret,
        "avg_symbol_return": avg_ret,
        "avg_sharpe": avg_sh,
        "avg_max_drawdown": avg_dd,
        "sum_trades": sum_trades,
        "win_rate": win_rate,
        "score_balanced": port_ret - (0.50 * avg_dd),
        "score_conservative": port_ret - (0.80 * avg_dd),
        "calmar_like": port_ret / max(avg_dd, 1e-9),
    }
    return summary, symbol_rows


def _build_configs() -> List[SweepConfig]:
    configs: List[SweepConfig] = [
        SweepConfig(label="baseline_mde_mad_v2", strategy_type="v2", kwargs={}),
        SweepConfig(
            label="baseline_leverage_static_x2",
            strategy_type="lev",
            kwargs={
                "target_leverage_multiplier": 2.0,
                "max_effective_leverage": 3.0,
                "enable_volatility_targeting": False,
                "enable_adaptive_leverage": False,
            },
        ),
        SweepConfig(
            label="baseline_leverage_adaptive_default",
            strategy_type="lev",
            kwargs={
                "target_leverage_multiplier": 2.0,
                "max_effective_leverage": 3.0,
                "enable_volatility_targeting": True,
                "enable_adaptive_leverage": True,
                "target_annual_volatility": 0.90,
                "min_volatility_scale": 0.35,
                "max_volatility_scale": 1.25,
                "strong_trend_slope_threshold": 0.0015,
                "strong_trend_distance_threshold": 0.02,
                "flat_outside_regime": True,
            },
        ),
    ]

    target_leverage_multiplier_grid = [1.8, 2.0, 2.2]
    target_annual_volatility_grid = [0.70, 0.90, 1.10]
    min_volatility_scale_grid = [0.25, 0.35]

    for lev, targ_vol, min_scale in itertools.product(
        target_leverage_multiplier_grid,
        target_annual_volatility_grid,
        min_volatility_scale_grid,
    ):
        label = (
            f"sweep_x{lev:.1f}_tvol{targ_vol:.2f}_"
            f"minscale{min_scale:.2f}"
        )
        configs.append(
            SweepConfig(
                label=label,
                strategy_type="lev",
                kwargs={
                    "target_leverage_multiplier": lev,
                    "max_effective_leverage": 3.0,
                    "enable_volatility_targeting": True,
                    "volatility_lookback_bars": 60,
                    "target_annual_volatility": targ_vol,
                    "min_volatility_scale": min_scale,
                    "max_volatility_scale": 1.25,
                    "enable_adaptive_leverage": True,
                    "flat_outside_regime": True,
                    "strong_trend_slope_threshold": 0.0015,
                    "strong_trend_distance_threshold": 0.02,
                },
            )
        )
    return configs


def main() -> None:
    configs = _build_configs()
    print(f"Starting sweep with {len(configs)} configs across {len(SYMBOLS)} symbols...")
    print(f"Range: {START_DATE} to {END_DATE}, timeframe={TIMEFRAME}, initial={INITIAL_BALANCE}")

    summary_rows: List[Dict[str, Any]] = []
    symbol_rows: List[Dict[str, Any]] = []

    for idx, cfg in enumerate(configs, start=1):
        started = datetime.utcnow()
        print(f"[{idx:02d}/{len(configs):02d}] {cfg.label} ...")
        summary, rows = _evaluate_config(cfg)
        summary_rows.append(summary)
        symbol_rows.extend(rows)
        elapsed = (datetime.utcnow() - started).total_seconds()
        print(
            f"  done in {elapsed:.1f}s | ret={summary['portfolio_return_equal20']:.4f} "
            f"| dd={summary['avg_max_drawdown']:.4f} | sharpe={summary['avg_sharpe']:.4f}"
        )

    summary_df = pd.DataFrame(summary_rows).sort_values(
        by=["score_balanced", "portfolio_return_equal20"],
        ascending=False,
    )
    symbol_df = pd.DataFrame(symbol_rows)

    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    out_dir = Path("verification")
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir / f"mde_v2_leverage_adaptive_sweep_midcap20_summary_{ts}.csv"
    symbol_path = out_dir / f"mde_v2_leverage_adaptive_sweep_midcap20_symbols_{ts}.csv"
    summary_df.to_csv(summary_path, index=False)
    symbol_df.to_csv(symbol_path, index=False)

    print("\nTop 10 by score_balanced:")
    print(
        summary_df.head(10)[
            [
                "config",
                "portfolio_return_equal20",
                "avg_max_drawdown",
                "avg_sharpe",
                "win_rate",
                "score_balanced",
                "calmar_like",
            ]
        ].to_string(index=False)
    )

    dd_cap = 0.50
    under_cap = summary_df[summary_df["avg_max_drawdown"] <= dd_cap].copy()
    if not under_cap.empty:
        best_under_cap = under_cap.sort_values(
            by=["portfolio_return_equal20", "avg_sharpe"],
            ascending=False,
        ).iloc[0]
        print(
            f"\nBest config under avg_max_drawdown <= {dd_cap:.2f}: "
            f"{best_under_cap['config']} "
            f"(ret={best_under_cap['portfolio_return_equal20']:.4f}, "
            f"dd={best_under_cap['avg_max_drawdown']:.4f}, "
            f"sh={best_under_cap['avg_sharpe']:.4f})"
        )
    else:
        print(f"\nNo config found under avg_max_drawdown <= {dd_cap:.2f}.")

    print(f"\nSaved summary: {summary_path}")
    print(f"Saved per-symbol results: {symbol_path}")


if __name__ == "__main__":
    main()
