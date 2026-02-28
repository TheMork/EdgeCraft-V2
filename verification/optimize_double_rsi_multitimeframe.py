import itertools
import json
import os
import random
import time
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from statistics import mean
from typing import Any, Dict, List

from src.simulation.runner import SimulationRunner
from src.simulation.strategies.double_rsi_divergence import DoubleRSIDivergenceStrategy


SYMBOL = "BTC/USDT"
START = "2024-01-01T00:00:00Z"
END = "2026-01-02T00:00:00Z"
TIMEFRAMES = ["1h", "4h", "12h", "1d"]


def _safe_metric(metrics: Dict[str, Any], key: str) -> float:
    value = metrics.get(key, 0.0) if metrics else 0.0
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _score_metrics(metrics: Dict[str, Any]) -> float:
    ret = _safe_metric(metrics, "total_return")
    sharpe = _safe_metric(metrics, "sharpe_ratio")
    max_dd = _safe_metric(metrics, "max_drawdown")
    trades = int(_safe_metric(metrics, "total_trades"))
    score = (ret * 100.0) + (sharpe * 8.0) + (max_dd * 25.0)
    if trades == 0:
        score -= 20.0
    elif trades < 2:
        score -= 5.0
    return score


def _run_backtest(timeframe: str, params: Dict[str, Any]) -> Dict[str, Any]:
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
        START,
        END,
        timeframe=timeframe,
        leverage=2,
    )
    with redirect_stdout(StringIO()), redirect_stderr(StringIO()):
        result = runner.run()
    metrics = result.get("metrics", {}) if isinstance(result, dict) else {}
    return {
        "total_return": _safe_metric(metrics, "total_return"),
        "sharpe_ratio": _safe_metric(metrics, "sharpe_ratio"),
        "max_drawdown": _safe_metric(metrics, "max_drawdown"),
        "final_equity": _safe_metric(metrics, "final_equity"),
        "total_trades": int(_safe_metric(metrics, "total_trades")),
    }


def evaluate_params(params: Dict[str, Any]) -> Dict[str, Any]:
    by_tf: Dict[str, Dict[str, Any]] = {}
    tf_scores: List[float] = []
    for tf in TIMEFRAMES:
        metrics = _run_backtest(tf, params)
        by_tf[tf] = metrics
        tf_scores.append(_score_metrics(metrics))

    overall_score = (mean(tf_scores) * 0.4) + (min(tf_scores) * 0.6)
    return {
        "params": params,
        "score": overall_score,
        "tf_scores": {tf: tf_scores[i] for i, tf in enumerate(TIMEFRAMES)},
        "by_tf": by_tf,
    }


def main() -> None:
    grid = {
        "pivot_lookback": [1, 2, 3],
        "min_pivot_separation_bars": [1, 2, 3, 4],
        "min_pivot_strength_atr": [0.0, 0.2, 0.6, 1.0],
        "min_retrace_between_pivots_pct": [0.0, 0.001],
        "double_top_bottom_tolerance": [0.0015, 0.003],
        "min_rsi_delta": [0.2, 0.6, 1.0],
        "min_price_move_pct": [0.0, 0.0005, 0.001],
        "stop_buffer_atr_multiplier": [0.1, 0.2],
        "rr_target": [1.5, 2.0, 3.0],
        "risk_per_trade": [0.005, 0.01],
    }

    keys = list(grid.keys())
    all_combinations = [dict(zip(keys, values)) for values in itertools.product(*[grid[k] for k in keys])]
    sample_limit = int(os.getenv("SAMPLE_LIMIT", "400"))
    random_seed = int(os.getenv("RANDOM_SEED", "42"))
    if sample_limit > 0 and len(all_combinations) > sample_limit:
        random.seed(random_seed)
        combinations = random.sample(all_combinations, sample_limit)
    else:
        combinations = all_combinations

    print(f"all_combinations={len(all_combinations)} sampled={len(combinations)} seed={random_seed}")

    results: List[Dict[str, Any]] = []
    started = time.time()
    for idx, params in enumerate(combinations, 1):
        results.append(evaluate_params(params))
        if idx % max(1, len(combinations) // 10) == 0:
            elapsed = time.time() - started
            print(f"progress={idx}/{len(combinations)} elapsed_sec={elapsed:.1f}")

    results.sort(key=lambda item: item["score"], reverse=True)
    print("top10=")
    print(json.dumps(results[:10], indent=2))


if __name__ == "__main__":
    main()
