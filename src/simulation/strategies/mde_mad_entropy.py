from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from src.simulation.models import Order, OrderType, Side, Trade
from src.simulation.strategy import Strategy

class MDEMADEntropyStrategy(Strategy):
    """
    Mathematical Logic (MDE-Model) Strategy:
    - Logarithmic Returns: Calculates returns in log space.
    - Risk Measure: Mean Absolute Deviation (MAD) instead of variance.
    - Diversification: Shannon Entropy of portfolio weights as a penalty.
    - Optimization: fast bounded search over candidate weights.

    Note: This strategy currently operates on a single symbol in the Runner's context,
    but it computes the 'optimal' allocation between Cash and Asset based on its 
    rolling MAD and Expected Return.
    """
    NAME: str = "MDEMADEntropy"
    DESCRIPTION: str = "Auto-generated description for MDEMADEntropy"
    VERSION: str = "1.0.0"
    AUTHOR: str = "EdgeCraft"
    SUPPORTED_TIMEFRAMES: list = ["1h", "4h", "1d"]



    EPS = 1e-9


    @classmethod
    def get_param_schema(cls):
        return {}

    def __init__(
        self,
        timeframe: str = "1d",
        lookback_bars: int = 30,
        risk_aversion: float = 2.0,
        entropy_weight: float = 0.1,
        turnover_penalty: float = 0.05,
        trend_filter_period: int = 200,
        enable_shorts: bool = False,
        min_leverage: int = 1,
        max_leverage: int = 1,
        optimization_steps: int = 41,
    ):
        super().__init__()
        self.timeframe = timeframe
        self.lookback_bars = max(10, int(lookback_bars))
        self.risk_aversion = float(risk_aversion)
        self.entropy_weight = float(entropy_weight)
        self.turnover_penalty = float(turnover_penalty)
        self.trend_filter_period = int(trend_filter_period)
        self.enable_shorts = bool(enable_shorts)
        self.min_leverage = max(1, int(min_leverage))
        self.max_leverage = max(self.min_leverage, int(max_leverage))
        self.optimization_steps = max(9, int(optimization_steps))

        self.bar_index = 0
        self.history: List[float] = []
        self.timestamps: List[datetime] = []
        self.current_weight = 0.0

    def on_start(self):
        print(
            f"MDE-MAD-Entropy Improved started (lb={self.lookback_bars}, "
            f"risk_a={self.risk_aversion}, entropy_w={self.entropy_weight}, "
            f"turnover_p={self.turnover_penalty}, trend_f={self.trend_filter_period})."
        )

    def on_stop(self):
        print("MDE-MAD-Entropy Strategy stopped.")

    def _calculate_log_returns(self, prices: np.ndarray) -> np.ndarray:
        if len(prices) < 2:
            return np.array([])
        return np.log(prices[1:] / prices[:-1])

    def _calculate_mad(self, returns: np.ndarray) -> float:
        if len(returns) == 0:
            return 0.0
        return np.mean(np.abs(returns - np.mean(returns)))

    def _calculate_ema(self, prices: List[float], period: int) -> float:
        if len(prices) < period:
            return prices[-1]
        alpha = 2 / (period + 1)
        ema = prices[0]
        for p in prices[1:]:
            ema = alpha * p + (1 - alpha) * ema
        return ema

    def _shannon_entropy(self, weights: np.ndarray) -> float:
        """
        Weights is [w_asset, w_cash].
        Entropy = - sum(p * log(p))
        """
        # Ensure positive for log and normalize
        w_abs = np.abs(weights)
        w = np.clip(w_abs, self.EPS, None)
        p = w / np.sum(w)
        return -np.sum(p * np.log(p))

    def _utility(self, weights: np.ndarray, expected_return: float, mad: float, current_w: float) -> np.ndarray:
        w_abs = np.abs(weights)
        w_cash = np.maximum(0.0, 1.0 - w_abs)
        p_asset = np.clip(w_abs, self.EPS, None)
        p_cash = np.clip(w_cash, self.EPS, None)
        p_sum = p_asset + p_cash
        p_asset /= p_sum
        p_cash /= p_sum
        entropy = -(p_asset * np.log(p_asset) + p_cash * np.log(p_cash))
        turnover = np.abs(weights - current_w)
        return (
            (weights * expected_return)
            - (self.risk_aversion * w_abs * mad)
            + (self.entropy_weight * entropy)
            - (self.turnover_penalty * turnover)
        )

    def _optimize_weight(self, returns: np.ndarray, is_uptrend: bool) -> float:
        if len(returns) < 5:
            return 0.0
        
        expected_return = np.mean(returns)
        mad = self._calculate_mad(returns)
        
        if mad <= self.EPS:
            if is_uptrend and expected_return > 0:
                return 1.0
            return 0.0

        # Bounds: [0, max_lev] if uptrend, [min_short, 0] if downtrend and shorts enabled
        if is_uptrend:
            lower_bound = 0.0
            upper_bound = float(self.max_leverage)
        else:
            lower_bound = -float(self.max_leverage) if self.enable_shorts else 0.0
            upper_bound = 0.0
        if upper_bound <= lower_bound:
            return float(lower_bound)

        coarse = np.linspace(lower_bound, upper_bound, num=self.optimization_steps, dtype=float)
        clipped_current = float(np.clip(self.current_weight, lower_bound, upper_bound))
        candidates = np.unique(np.concatenate([coarse, np.array([clipped_current], dtype=float)]))
        utilities = self._utility(candidates, expected_return, mad, self.current_weight)
        best_idx = int(np.argmax(utilities))
        best_weight = float(candidates[best_idx])

        # One local refinement step around the best coarse candidate.
        if len(candidates) > 1:
            step = float((upper_bound - lower_bound) / max(1, self.optimization_steps - 1))
            if step > 0:
                local_lower = max(lower_bound, best_weight - step)
                local_upper = min(upper_bound, best_weight + step)
                if local_upper > local_lower:
                    local = np.linspace(local_lower, local_upper, num=11, dtype=float)
                    local_util = self._utility(local, expected_return, mad, self.current_weight)
                    best_weight = float(local[int(np.argmax(local_util))])

        return best_weight

    def on_bar(self, bar: Dict[str, Any]):
        self.bar_index += 1
        symbol = bar.get("symbol")
        close = bar.get("close")
        ts = bar.get("timestamp")
        
        if close is None or ts is None:
            return
            
        self.history.append(float(close))
        self.timestamps.append(ts)
        
        # We need enough history for BOTH lookback and trend filter
        min_required = max(self.lookback_bars, self.trend_filter_period) + 1
        if len(self.history) < min_required:
            return
            
        # Limit history to save memory but keep enough for trend filter
        if len(self.history) > min_required + 50:
            self.history = self.history[-(min_required + 50):]
            
        # 1. Trend Filter (EMA 200)
        ema_trend = self._calculate_ema(self.history, self.trend_filter_period)
        is_uptrend = float(close) >= ema_trend
        
        # 2. Optimization
        prices_for_returns = np.array(self.history[-(self.lookback_bars + 1):])
        returns = self._calculate_log_returns(prices_for_returns)
        
        target_weight = self._optimize_weight(returns, is_uptrend)
        
        # Execute rebalance if significantly different (Turnover penalty already handles small drifts in objective)
        if abs(target_weight - self.current_weight) > 0.005:
            self._rebalance(symbol, target_weight, float(close), ts)

    def _rebalance(self, symbol: str, target_weight: float, price: float, timestamp: Any):
        if not self.broker:
            return
            
        equity = self.broker.equity
        target_notional = equity * target_weight
        current_qty = self.get_position_size(symbol)
        current_notional = current_qty * price
        
        diff_notional = target_notional - current_notional
        diff_qty = diff_notional / price
        
        if abs(diff_qty) < 1e-6:
            return
            
        side = Side.BUY if diff_qty > 0 else Side.SELL
        order = Order(
            id="",
            symbol=symbol,
            side=side,
            order_type=OrderType.MARKET,
            quantity=abs(float(diff_qty)),
            price=None,
            timestamp=timestamp,
            leverage=max(1, int(abs(target_weight)))
        )
        
        if self.submit_order(order):
            self.current_weight = target_weight

    def on_fill(self, trade: Trade):
        pass
