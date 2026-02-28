import json
import os
import random
import time
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from statistics import mean, pstdev
from typing import Any, Dict, List, Tuple

from src.simulation.runner import SimulationRunner
from src.simulation.strategies.double_rsi_divergence import DoubleRSIDivergenceStrategy

SYMBOL = "BTC/USDT"
TIMEFRAMES = ["1h", "4h", "12h", "1d"]

# Expanding-window walk-forward folds.
FOLDS: List[Dict[str, str]] = [
    {
        "train_start": "2024-01-01T00:00:00Z",
        "train_end": "2024-09-30T23:59:59Z",
        "test_start": "2024-10-01T00:00:00Z",
        "test_end": "2024-12-31T23:59:59Z",
    },
    {
        "train_start": "2024-01-01T00:00:00Z",
        "train_end": "2024-12-31T23:59:59Z",
        "test_start": "2025-01-01T00:00:00Z",
        "test_end": "2025-03-31T23:59:59Z",
    },
    {
        "train_start": "2024-01-01T00:00:00Z",
        "train_end": "2025-03-31T23:59:59Z",
        "test_start": "2025-04-01T00:00:00Z",
        "test_end": "2025-06-30T23:59:59Z",
    },
    {
        "train_start": "2024-01-01T00:00:00Z",
        "train_end": "2025-06-30T23:59:59Z",
        "test_start": "2025-07-01T00:00:00Z",
        "test_end": "2026-01-02T00:00:00Z",
    },
]


def _safe_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _score(metrics: Dict[str, Any]) -> float:
    ret = _safe_float(metrics.get("total_return", 0.0))
    sharpe = _safe_float(metrics.get("sharpe_ratio", 0.0))
    max_dd = _safe_float(metrics.get("max_drawdown", 0.0))
    trades = int(_safe_float(metrics.get("total_trades", 0)))
    score = (ret * 100.0) + (sharpe * 7.0) + (max_dd * 18.0)
    if trades == 0:
        score -= 20.0
    elif trades < 2:
        score -= 6.0
    else:
        score += min(trades, 30) * 0.1
    return score


def _run_backtest(
    timeframe: str,
    start_date: str,
    end_date: str,
    params: Dict[str, Any],
) -> Dict[str, Any]:
    strategy = DoubleRSIDivergenceStrategy(
        timeframe=timeframe,
        enable_shorts=True,
        min_leverage=1,
        max_leverage=3,
        **params,
    )
    runner = SimulationRunner(
        strategy,
        SYMBOL,
        start_date,
        end_date,
        timeframe=timeframe,
        leverage=2,
    )
    with redirect_stdout(StringIO()), redirect_stderr(StringIO()):
        result = runner.run()
    return result.get("metrics", {}) if isinstance(result, dict) else {}


def _sample_params(space: Dict[str, List[Any]], sample_limit: int, seed: int) -> List[Dict[str, Any]]:
    random.seed(seed)
    keys = list(space.keys())
    picked: List[Dict[str, Any]] = []
    seen = set()
    max_attempts = sample_limit * 80
    attempts = 0

    while len(picked) < sample_limit and attempts < max_attempts:
        attempts += 1
        candidate = {k: random.choice(space[k]) for k in keys}
        signature = tuple((k, candidate[k]) for k in keys)
        if signature in seen:
            continue
        seen.add(signature)
        picked.append(candidate)

    return picked


def evaluate_params_for_timeframe(timeframe: str, params: Dict[str, Any]) -> Dict[str, Any]:
    train_scores: List[float] = []
    test_scores: List[float] = []
    test_returns: List[float] = []
    test_trades: List[int] = []
    fold_results: List[Dict[str, Any]] = []

    for fold in FOLDS:
        train_metrics = _run_backtest(
            timeframe,
            fold["train_start"],
            fold["train_end"],
            params,
        )
        test_metrics = _run_backtest(
            timeframe,
            fold["test_start"],
            fold["test_end"],
            params,
        )
        train_s = _score(train_metrics)
        test_s = _score(test_metrics)
        train_scores.append(train_s)
        test_scores.append(test_s)
        test_returns.append(_safe_float(test_metrics.get("total_return", 0.0)))
        test_trades.append(int(_safe_float(test_metrics.get("total_trades", 0.0))))
        fold_results.append(
            {
                "train_score": train_s,
                "test_score": test_s,
                "train_metrics": {
                    "total_return": _safe_float(train_metrics.get("total_return", 0.0)),
                    "sharpe_ratio": _safe_float(train_metrics.get("sharpe_ratio", 0.0)),
                    "max_drawdown": _safe_float(train_metrics.get("max_drawdown", 0.0)),
                    "total_trades": int(_safe_float(train_metrics.get("total_trades", 0.0))),
                },
                "test_metrics": {
                    "total_return": _safe_float(test_metrics.get("total_return", 0.0)),
                    "sharpe_ratio": _safe_float(test_metrics.get("sharpe_ratio", 0.0)),
                    "max_drawdown": _safe_float(test_metrics.get("max_drawdown", 0.0)),
                    "total_trades": int(_safe_float(test_metrics.get("total_trades", 0.0))),
                },
            }
        )

    mean_train = mean(train_scores)
    mean_test = mean(test_scores)
    worst_test = min(test_scores)
    ret_stability_penalty = pstdev(test_returns) * 45.0 if len(test_returns) > 1 else 0.0
    avg_test_trades = mean(test_trades) if test_trades else 0.0
    sparse_penalty = 10.0 if avg_test_trades < 1.0 else (4.0 if avg_test_trades < 2.0 else 0.0)
    robust_score = (0.55 * mean_test) + (0.30 * worst_test) + (0.15 * mean_train) - ret_stability_penalty - sparse_penalty

    return {
        "timeframe": timeframe,
        "params": params,
        "robust_score": robust_score,
        "summary": {
            "mean_train_score": mean_train,
            "mean_test_score": mean_test,
            "worst_test_score": worst_test,
            "test_return_stddev": pstdev(test_returns) if len(test_returns) > 1 else 0.0,
            "avg_test_trades": avg_test_trades,
        },
        "folds": fold_results,
    }


def _profile_from_params(params: Dict[str, Any]) -> Dict[str, Any]:
    keep_keys = [
        "pivot_lookback",
        "min_pivot_separation_bars",
        "min_pivot_strength_atr",
        "min_retrace_between_pivots_pct",
        "double_top_bottom_tolerance",
        "min_rsi_delta",
        "min_price_move_pct",
        "stop_buffer_atr_multiplier",
        "rr_target",
        "risk_per_trade",
        "use_structure_break_trigger",
        "enable_regime_filter",
        "min_adx_for_entry",
        "require_ema_trend_alignment",
        "max_setup_age_bars",
    ]
    return {k: params[k] for k in keep_keys if k in params}


def main() -> None:
    sample_limit = int(os.getenv("SAMPLE_LIMIT", "120"))
    seed = int(os.getenv("RANDOM_SEED", "42"))
    output_path = os.getenv("OUTPUT_PATH", "verification/walkforward_double_rsi_results.json")

    # Includes trigger/regime layer so walk-forward can pick safer configurations.
    param_space: Dict[str, List[Any]] = {
        "pivot_lookback": [1, 2, 3],
        "min_pivot_separation_bars": [1, 2, 3, 4],
        "min_pivot_strength_atr": [0.0, 0.2, 0.6, 1.0],
        "min_retrace_between_pivots_pct": [0.0, 0.001, 0.004],
        "double_top_bottom_tolerance": [0.0015, 0.003],
        "min_rsi_delta": [0.2, 0.6, 1.0],
        "min_price_move_pct": [0.0, 0.0005, 0.001],
        "stop_buffer_atr_multiplier": [0.1, 0.2],
        "rr_target": [1.5, 2.0, 3.0],
        "risk_per_trade": [0.005, 0.01],
        "use_structure_break_trigger": [False, True],
        "enable_regime_filter": [False, True],
        "min_adx_for_entry": [10.0, 14.0, 18.0],
        "require_ema_trend_alignment": [False, True],
        "max_setup_age_bars": [12, 24],
    }

    samples = _sample_params(param_space, sample_limit=sample_limit, seed=seed)
    print(f"samples={len(samples)} seed={seed} folds={len(FOLDS)}")

    started = time.time()
    payload: Dict[str, Any] = {
        "meta": {
            "symbol": SYMBOL,
            "timeframes": TIMEFRAMES,
            "folds": FOLDS,
            "sample_limit": sample_limit,
            "seed": seed,
            "generated_at_epoch": time.time(),
        },
        "per_timeframe": {},
        "recommended_profiles": {},
    }

    for timeframe in TIMEFRAMES:
        print(f"\n=== timeframe={timeframe} ===")
        rows: List[Dict[str, Any]] = []
        for idx, params in enumerate(samples, 1):
            rows.append(evaluate_params_for_timeframe(timeframe, params))
            if idx % max(1, len(samples) // 10) == 0:
                elapsed = time.time() - started
                print(f"{timeframe} progress={idx}/{len(samples)} elapsed_sec={elapsed:.1f}")

        rows.sort(key=lambda item: item["robust_score"], reverse=True)
        top5 = rows[:5]
        payload["per_timeframe"][timeframe] = top5
        payload["recommended_profiles"][timeframe] = _profile_from_params(top5[0]["params"])

        best = top5[0]
        print(
            f"{timeframe} best robust_score={best['robust_score']:.4f} "
            f"mean_test={best['summary']['mean_test_score']:.4f} "
            f"worst_test={best['summary']['worst_test_score']:.4f} "
            f"avg_test_trades={best['summary']['avg_test_trades']:.2f}"
        )
        print(json.dumps(payload["recommended_profiles"][timeframe], indent=2))

    with open(output_path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)

    print(f"\nWrote {output_path}")


if __name__ == "__main__":
    main()
