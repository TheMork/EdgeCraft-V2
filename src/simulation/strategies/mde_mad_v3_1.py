from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from src.simulation.models import Order, OrderType, Side, Trade
from src.simulation.strategy import Strategy

class MDEMADV3_1Strategy(Strategy):
    """
    MDE-MAD Version 3.1 Strategy:
    - Based on V1 (Aggressive Classic).
    - NO Trend Filter (Trade everything).
    - Turnover Penalty (Light).
    - DYNAMIC Volatility-Adjusted Lookback:
      Short lookback in high vol, long lookback in low vol.
    """

    EPS = 1e-9

    def __init__(
        self,
        timeframe: str = "1h",
        min_lookback: int = 20,
        max_lookback: int = 150,
        risk_aversion: float = 2.0,
        entropy_weight: float = 0.1,
        turnover_penalty: float = 0.01, # Light penalty from V1 context
        enable_shorts: bool = True,
        min_leverage: int = 1,
        max_leverage: int = 1,
    ):
        super().__init__()
        self.timeframe = timeframe
        self.min_lookback = int(min_lookback)
        self.max_lookback = int(max_lookback)
        self.risk_aversion = float(risk_aversion)
        self.entropy_weight = float(entropy_weight)
        self.turnover_penalty = float(turnover_penalty)
        self.enable_shorts = bool(enable_shorts)
        self.min_leverage = max(1, int(min_leverage))
        self.max_leverage = max(self.min_leverage, int(max_leverage))

        self.bar_index = 0
        self.history: List[float] = []
        self.timestamps: List[datetime] = []
        self.current_weight = 0.0
        self.returns_history: List[float] = []

    def on_start(self):
        print(
            f"MDE-MAD-V3.1 (V1 + Dynamic) started (lb={self.min_lookback}-{self.max_lookback}, "
            f"risk_a={self.risk_aversion}, turnover_p={self.turnover_penalty})."
        )

    def on_stop(self):
        print("MDE-MAD-V3.1 Strategy stopped.")

    def _calculate_log_returns(self, prices: np.ndarray) -> np.ndarray:
        if len(prices) < 2:
            return np.array([])
        return np.log(prices[1:] / prices[:-1])

    def _calculate_mad(self, returns: np.ndarray) -> float:
        if len(returns) == 0:
            return 0.0
        return np.mean(np.abs(returns - np.mean(returns)))

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
        turnover = abs(w_asset - current_w)
        
        utility = (
            portfolio_return 
            - (self.risk_aversion * portfolio_risk) 
            + (self.entropy_weight * entropy)
            - (self.turnover_penalty * turnover)
        )
        return -utility

    def _optimize_weight(self, returns: np.ndarray) -> float:
        if len(returns) < 5:
            return 0.0
        
        expected_return = np.mean(returns)
        mad = self._calculate_mad(returns)
        
        if mad <= self.EPS:
            return float(self.max_leverage) if expected_return > 0 else 0.0

        # Aggressive Bounds like V1
        lower_bound = -float(self.max_leverage) if self.enable_shorts else 0.0
        upper_bound = float(self.max_leverage)
        
        res = minimize(
            self._objective,
            x0=np.array([self.current_weight]),
            args=(expected_return, mad, self.current_weight),
            bounds=[(lower_bound, upper_bound)],
            method='L-BFGS-B'
        )
        
        if res.success:
            return float(res.x[0])
        return self.current_weight

    def on_bar(self, bar: Dict[str, Any]):
        self.bar_index += 1
        symbol = bar.get("symbol")
        close = bar.get("close")
        ts = bar.get("timestamp")
        
        if close is None or ts is None:
            return
            
        self.history.append(float(close))
        self.timestamps.append(ts)
        
        if len(self.history) >= 2:
            self.returns_history.append(np.log(self.history[-1] / self.history[-2]))
        
        # Min bars required for Volatility calculation
        min_required = max(self.max_lookback, 100) + 1
        if len(self.history) < min_required:
            return
            
        if len(self.history) > min_required + 100:
            self.history = self.history[-(min_required + 100):]
            self.returns_history = self.returns_history[-(min_required + 100):]
            
        # 1. Dynamic Lookback based on Volatility
        recent_returns = np.array(self.returns_history[-30:])
        long_term_returns = np.array(self.returns_history[-100:])
        
        recent_vol = np.std(recent_returns) if len(recent_returns) > 0 else self.EPS
        long_term_vol = np.std(long_term_returns) if len(long_term_returns) > 0 else self.EPS
        
        vol_ratio = recent_vol / max(self.EPS, long_term_vol)
        
        lb_range = self.max_lookback - self.min_lookback
        norm_ratio = np.clip(vol_ratio, 0.5, 2.0)
        lookback_factor = (norm_ratio - 0.5) / 1.5 
        dynamic_lookback = int(self.max_lookback - (lookback_factor * lb_range))
        
        # 2. Aggressive Optimization (No Trend Filter)
        prices_for_returns = np.array(self.history[-(dynamic_lookback + 1):])
        returns = self._calculate_log_returns(prices_for_returns)
        
        target_weight = self._optimize_weight(returns)
        
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
            id="", symbol=symbol, side=side, order_type=OrderType.MARKET,
            quantity=abs(float(diff_qty)), price=None, timestamp=timestamp,
            leverage=max(1, int(abs(target_weight)))
        )
        if self.submit_order(order):
            self.current_weight = target_weight

    def on_fill(self, trade: Trade):
        pass
