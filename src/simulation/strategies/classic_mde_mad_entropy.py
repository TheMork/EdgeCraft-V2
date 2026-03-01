from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from src.simulation.models import Order, OrderType, Side, Trade
from src.simulation.strategy import Strategy

class ClassicMDEMADEntropyStrategy(Strategy):
    """
    Classic Mathematical Logic (MDE-Model) Strategy:
    - Logarithmic Returns: Calculates returns in log space.
    - Risk Measure: Mean Absolute Deviation (MAD) instead of variance.
    - Diversification: Shannon Entropy of portfolio weights as a penalty.
    - Optimization: scipy.optimize.minimize to find optimal weights.

    Note: This strategy currently operates on a single symbol in the Runner's context,
    but it computes the 'optimal' allocation between Cash and Asset based on its 
    rolling MAD and Expected Return.
    """
    NAME: str = "ClassicMDEMADEntropy"
    DESCRIPTION: str = "Auto-generated description for ClassicMDEMADEntropy"
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
        enable_shorts: bool = False,
        min_leverage: int = 1,
        max_leverage: int = 1,
    ):
        super().__init__()
        self.timeframe = timeframe
        self.lookback_bars = max(10, int(lookback_bars))
        self.risk_aversion = float(risk_aversion)
        self.entropy_weight = float(entropy_weight)
        self.enable_shorts = bool(enable_shorts)
        self.min_leverage = max(1, int(min_leverage))
        self.max_leverage = max(self.min_leverage, int(max_leverage))

        self.bar_index = 0
        self.history: List[float] = []
        self.timestamps: List[datetime] = []
        self.current_weight = 0.0

    def on_start(self):
        print(
            f"MDE-MAD-Entropy Strategy started (lb={self.lookback_bars}, "
            f"risk_a={self.risk_aversion}, entropy_w={self.entropy_weight})."
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

    def _shannon_entropy(self, weights: np.ndarray) -> float:
        """
        Weights is [w_asset, w_cash].
        Entropy = - sum(p * log(p))
        """
        # Ensure positive for log
        w = np.clip(weights, self.EPS, 1.0)
        # Normalize to sum to 1 for entropy calculation
        p = w / np.sum(w)
        return -np.sum(p * np.log(p))

    def _objective(self, w_asset: float, expected_return: float, mad: float) -> float:
        """
        Objective: Maximize Return - Risk + Entropy
        We minimize the negative: - (Return - Risk_Aversion * Risk + Entropy_Weight * Entropy)
        """
        # weights = [w_asset, 1 - w_asset]
        w_cash = 1.0 - w_asset
        weights = np.array([w_asset, w_cash])
        
        portfolio_return = w_asset * expected_return
        portfolio_risk = abs(w_asset) * mad
        entropy = self._shannon_entropy(weights)
        
        # Utility = Return - RiskPenalty + EntropyBonus
        utility = portfolio_return - (self.risk_aversion * portfolio_risk) + (self.entropy_weight * entropy)
        return -utility

    def _optimize_weight(self, returns: np.ndarray) -> float:
        if len(returns) < 5:
            return 0.0
        
        expected_return = np.mean(returns)
        mad = self._calculate_mad(returns)
        
        if mad <= self.EPS:
            return 1.0 if expected_return > 0 else 0.0

        # Bounds: [0, 1] if no shorts, [-1, 1] if shorts enabled (or higher if leverage > 1)
        lower_bound = -float(self.max_leverage) if self.enable_shorts else 0.0
        upper_bound = float(self.max_leverage)
        
        res = minimize(
            self._objective,
            x0=np.array([self.current_weight]),
            args=(expected_return, mad),
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
        
        if len(self.history) < self.lookback_bars + 1:
            return
            
        # Limit history
        if len(self.history) > self.lookback_bars + 1:
            self.history = self.history[-(self.lookback_bars + 1):]
            
        prices = np.array(self.history)
        returns = self._calculate_log_returns(prices)
        
        target_weight = self._optimize_weight(returns)
        
        # Execute rebalance if significantly different
        if abs(target_weight - self.current_weight) > 0.01:
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
