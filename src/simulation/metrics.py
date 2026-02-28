import pandas as pd
import numpy as np
from typing import List, Dict, Any
from src.simulation.models import Trade

def calculate_sharpe_ratio(returns: pd.Series, risk_free_rate: float = 0.0, periods_per_year: float = 365 * 24) -> float:
    if returns.empty or returns.std() == 0:
        return 0.0
    excess_returns = returns - risk_free_rate / periods_per_year
    return (excess_returns.mean() / returns.std()) * np.sqrt(periods_per_year)

def calculate_sortino_ratio(returns: pd.Series, risk_free_rate: float = 0.0, periods_per_year: float = 365 * 24) -> float:
    if returns.empty:
        return 0.0
    excess_returns = returns - risk_free_rate / periods_per_year
    downside_returns = excess_returns[excess_returns < 0]
    if downside_returns.empty or downside_returns.std() == 0:
        return 0.0
    return (excess_returns.mean() / downside_returns.std()) * np.sqrt(periods_per_year)

def calculate_max_drawdown(equity_curve: pd.Series) -> float:
    if equity_curve.empty:
        return 0.0
    running_max = equity_curve.cummax()
    drawdown = (running_max - equity_curve) / running_max
    return float(drawdown.max())

def calculate_metrics(trades: List[Trade], equity_curve: List[Dict[str, Any]]) -> Dict[str, float]:
    if not equity_curve:
        return {}

    df = pd.DataFrame(equity_curve)
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df.set_index('timestamp', inplace=True)

    # Calculate returns
    df['returns'] = df['equity'].pct_change().fillna(0)

    periods_per_year = 365 * 24

    # Estimate frequency
    if len(df) > 1:
        time_diff = (df.index[1] - df.index[0]).total_seconds()
        if time_diff > 0:
            periods_per_year = 31536000 / time_diff # Seconds in year / seconds per bar

    total_return = (df['equity'].iloc[-1] - df['equity'].iloc[0]) / df['equity'].iloc[0]
    sharpe = calculate_sharpe_ratio(df['returns'], periods_per_year=periods_per_year)
    sortino = calculate_sortino_ratio(df['returns'], periods_per_year=periods_per_year)
    max_drawdown_close = calculate_max_drawdown(df['equity'])
    max_drawdown_worst = max_drawdown_close
    if 'equity_worst' in df.columns:
        worst_series = pd.to_numeric(df['equity_worst'], errors='coerce').fillna(df['equity'])
        max_drawdown_worst = calculate_max_drawdown(worst_series)
    max_drawdown = max(max_drawdown_close, max_drawdown_worst)

    return {
        "total_return": total_return,
        "sharpe_ratio": sharpe,
        "sortino_ratio": sortino,
        "max_drawdown": max_drawdown,
        "max_drawdown_close": max_drawdown_close,
        "max_drawdown_worst": max_drawdown_worst,
        "final_equity": df['equity'].iloc[-1],
        "total_trades": len(trades)
    }
