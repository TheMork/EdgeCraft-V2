from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from src.simulation.models import Order, OrderType, Side, Trade
from src.simulation.strategy import Strategy

class MDEMADModularStrategy(Strategy):
    """
    MDE-MAD Modular Strategy:
    Allows toggling features to analyze their impact on Profit and Drawdown.
    """
    NAME: str = "MDEMADModular"
    DESCRIPTION: str = "Auto-generated description for MDEMADModular"
    VERSION: str = "1.0.0"
    AUTHOR: str = "EdgeCraft"
    SUPPORTED_TIMEFRAMES: list = ["1h", "4h", "1d"]



    EPS = 1e-9


    @classmethod
    def get_param_schema(cls):
        return {}

    def __init__(
        self,
        timeframe: str = "1h",
        lookback_bars: int = 100,
        use_trend_filter: bool = False,
        use_turnover_penalty: bool = False,
        use_dynamic_lookback: bool = False,
        min_lookback: int = 20,
        max_lookback: int = 150,
        risk_aversion: float = 2.0,
        entropy_weight: float = 0.1,
        turnover_penalty_value: float = 0.05,
        trend_filter_period: int = 200,
        enable_shorts: bool = True,
        min_leverage: int = 1,
        max_leverage: int = 1,
    ):
        super().__init__()
        self.timeframe = timeframe
        self.lookback_bars = int(lookback_bars)
        
        # Features
        self.use_trend_filter = bool(use_trend_filter)
        self.use_turnover_penalty = bool(use_turnover_penalty)
        self.use_dynamic_lookback = bool(use_dynamic_lookback)
        
        # Params
        self.min_lookback = int(min_lookback)
        self.max_lookback = int(max_lookback)
        self.risk_aversion = float(risk_aversion)
        self.entropy_weight = float(entropy_weight)
        self.turnover_penalty_value = float(turnover_penalty_value)
        self.trend_filter_period = int(trend_filter_period)
        
        self.enable_shorts = bool(enable_shorts)
        self.min_leverage = max(1, int(min_leverage))
        self.max_leverage = max(self.min_leverage, int(max_leverage))

        self.bar_index = 0
        self.history: List[float] = []
        self.timestamps: List[datetime] = []
        self.current_weight = 0.0
        self.returns_history: List[float] = []

    def on_start(self):
        pass

    def on_stop(self):
        pass

    def _calculate_log_returns(self, prices: np.ndarray) -> np.ndarray:
        if len(prices) < 2: return np.array([])
        return np.log(prices[1:] / prices[:-1])

    def _calculate_mad(self, returns: np.ndarray) -> float:
        if len(returns) == 0: return 0.0
        return np.mean(np.abs(returns - np.mean(returns)))

    def _calculate_ema(self, prices: List[float], period: int) -> float:
        if len(prices) < period: return prices[-1]
        alpha = 2 / (period + 1)
        ema = prices[0]
        for p in prices[1:]:
            ema = alpha * p + (1 - alpha) * ema
        return ema

    def _shannon_entropy(self, weights: np.ndarray) -> float:
        w_abs = np.abs(weights)
        w = np.clip(w_abs, self.EPS, None)
        p = w / np.sum(w)
        return -np.sum(p * np.log(p))

    def _objective(self, w_asset_arr: np.ndarray, expected_return: float, mad: float, current_w: float) -> float:
        w_asset = w_asset_arr[0]
        w_cash = 1.0 - abs(w_asset)
        weights = np.array([abs(w_asset), w_cash])
        
        portfolio_return = w_asset * expected_return
        portfolio_risk = abs(w_asset) * mad
        entropy = self._shannon_entropy(weights)
        
        # Turnover Penalty Feature
        penalty = 0.0
        if self.use_turnover_penalty:
            penalty = self.turnover_penalty_value * abs(w_asset - current_w)
        
        utility = (
            portfolio_return 
            - (self.risk_aversion * portfolio_risk) 
            + (self.entropy_weight * entropy)
            - penalty
        )
        return -utility

    def _optimize_weight(self, returns: np.ndarray, is_uptrend: bool) -> float:
        if len(returns) < 5: return 0.0
        expected_return = np.mean(returns)
        mad = self._calculate_mad(returns)
        
        if mad <= self.EPS:
            if self.use_trend_filter:
                if is_uptrend and expected_return > 0: return 1.0
                return 0.0
            return 1.0 if expected_return > 0 else 0.0

        # Bounds logic
        if self.use_trend_filter:
            if is_uptrend:
                lower_bound, upper_bound = 0.0, float(self.max_leverage)
            else:
                lower_bound = -float(self.max_leverage) if self.enable_shorts else 0.0
                upper_bound = 0.0
        else:
            lower_bound = -float(self.max_leverage) if self.enable_shorts else 0.0
            upper_bound = float(self.max_leverage)
        
        res = minimize(
            self._objective,
            x0=np.array([self.current_weight]),
            args=(expected_return, mad, self.current_weight),
            bounds=[(lower_bound, upper_bound)],
            method='L-BFGS-B'
        )
        return float(res.x[0]) if res.success else self.current_weight

    def on_bar(self, bar: Dict[str, Any]):
        self.bar_index += 1
        symbol, close, ts = bar.get("symbol"), bar.get("close"), bar.get("timestamp")
        if close is None or ts is None: return
            
        self.history.append(float(close))
        self.timestamps.append(ts)
        if len(self.history) >= 2:
            self.returns_history.append(np.log(self.history[-1] / self.history[-2]))
        
        min_required = max(self.max_lookback, self.trend_filter_period, 100) + 1
        if len(self.history) < min_required: return
        if len(self.history) > min_required + 50:
            self.history = self.history[-(min_required + 50):]
            self.returns_history = self.returns_history[-(min_required + 50):]
            
        # 1. Trend Filter Feature
        is_uptrend = True
        if self.use_trend_filter:
            ema_trend = self._calculate_ema(self.history, self.trend_filter_period)
            is_uptrend = float(close) >= ema_trend
        
        # 2. Dynamic Lookback Feature
        active_lookback = self.lookback_bars
        if self.use_dynamic_lookback:
            recent_vol = np.std(self.returns_history[-30:]) if len(self.returns_history) > 0 else self.EPS
            long_term_vol = np.std(self.returns_history[-100:]) if len(self.returns_history) > 0 else self.EPS
            vol_ratio = recent_vol / max(self.EPS, long_term_vol)
            norm_ratio = np.clip(vol_ratio, 0.5, 2.0)
            lookback_factor = (norm_ratio - 0.5) / 1.5 
            active_lookback = int(self.max_lookback - (lookback_factor * (self.max_lookback - self.min_lookback)))
        
        # 3. Optimization
        prices_for_returns = np.array(self.history[-(active_lookback + 1):])
        returns = self._calculate_log_returns(prices_for_returns)
        target_weight = self._optimize_weight(returns, is_uptrend)
        
        if abs(target_weight - self.current_weight) > 0.005:
            self._rebalance(symbol, target_weight, float(close), ts)

    def _rebalance(self, symbol: str, target_weight: float, price: float, timestamp: Any):
        if not self.broker: return
        equity = self.broker.equity
        target_notional = equity * target_weight
        current_qty = self.get_position_size(symbol)
        diff_qty = (target_notional - (current_qty * price)) / price
        if abs(diff_qty) < 1e-6: return
            
        side = Side.BUY if diff_qty > 0 else Side.SELL
        order = Order(
            id="", symbol=symbol, side=side, order_type=OrderType.MARKET,
            quantity=abs(float(diff_qty)), price=None, timestamp=timestamp,
            leverage=max(1, int(abs(target_weight)))
        )
        if self.submit_order(order):
            self.current_weight = target_weight

    def on_fill(self, trade: Trade): pass
