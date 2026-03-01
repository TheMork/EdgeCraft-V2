from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from src.simulation.models import Order, OrderType, Side, Trade
from src.simulation.strategy import Strategy

class MDEMADV2Strategy(Strategy):
    """
    MDE-MAD Version 2 Strategy:
    - Added EMA 200 Trend Filter.
    - Added Turnover Penalty to reduce unnecessary trades.
    """
    NAME: str = "MDEMADV2"
    DESCRIPTION: str = "Auto-generated description for MDEMADV2"
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
        cooldown_hours: float = 1.0,
        min_rebalance_weight_delta: float = 0.02,
        min_rebalance_notional: float = 25.0,
        enable_cost_gate: bool = True,
        fee_rate: float = 0.0004,
        slippage_rate: float = 0.0002,
        edge_horizon_bars: int = 3,
        min_edge_over_cost_ratio: float = 1.25,
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

        self.bar_index = 0
        self.history: List[float] = []
        self.timestamps: List[datetime] = []
        self.current_weight = 0.0
        
        self.cooldown_hours = float(cooldown_hours)
        self.cooldown_until: Optional[datetime] = None
        self.closing_order_ids: set = set()
        self.min_rebalance_weight_delta = max(0.0, float(min_rebalance_weight_delta))
        self.min_rebalance_notional = max(0.0, float(min_rebalance_notional))
        self.enable_cost_gate = bool(enable_cost_gate)
        self.fee_rate = max(0.0, float(fee_rate))
        self.slippage_rate = max(0.0, float(slippage_rate))
        self.edge_horizon_bars = max(1, int(edge_horizon_bars))
        self.min_edge_over_cost_ratio = max(0.0, float(min_edge_over_cost_ratio))
        self._last_expected_return = 0.0

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

    def _objective(self, w_asset_arr: np.ndarray, expected_return: float, mad: float, current_w: float) -> float:
        """
        Objective: Maximize Return - Risk + Entropy - TurnoverPenalty
        """
        w_asset = w_asset_arr[0]
        w_cash = 1.0 - abs(w_asset)
        weights = np.array([abs(w_asset), w_cash])
        
        portfolio_return = w_asset * expected_return
        portfolio_risk = abs(w_asset) * mad
        entropy = self._shannon_entropy(weights)
        
        # Turnover Penalty: Penalty for changing the weight
        turnover = abs(w_asset - current_w)
        
        # Utility = Return - RiskPenalty + EntropyBonus - TurnoverPenalty
        utility = (
            portfolio_return 
            - (self.risk_aversion * portfolio_risk) 
            + (self.entropy_weight * entropy)
            - (self.turnover_penalty * turnover)
        )
        return -utility

    def _optimize_weight(self, returns: np.ndarray, is_uptrend: bool) -> float:
        if len(returns) < 5:
            self._last_expected_return = 0.0
            return 0.0
        
        expected_return = np.mean(returns)
        self._last_expected_return = float(expected_return)
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

    def _normalize_timestamp(self, ts: Any) -> Optional[datetime]:
        try:
            parsed = pd.Timestamp(ts)
        except Exception:
            return None
        if pd.isna(parsed):
            return None
        if parsed.tzinfo is None:
            parsed = parsed.tz_localize(timezone.utc)
        else:
            parsed = parsed.tz_convert(timezone.utc)
        return parsed.to_pydatetime()

    def _is_cooldown_active(self, ts: Any) -> bool:
        if self.cooldown_until is None:
            return False
        ts_utc = self._normalize_timestamp(ts)
        cooldown_until_utc = self._normalize_timestamp(self.cooldown_until)
        if ts_utc is None or cooldown_until_utc is None:
            self.cooldown_until = None
            return False
        if ts_utc < cooldown_until_utc:
            return True
        self.cooldown_until = None
        return False

    def _passes_cost_gate(self, trade_notional: float, expected_return: float) -> bool:
        if not self.enable_cost_gate:
            return True
        notional = abs(float(trade_notional))
        if notional <= self.EPS:
            return False
        est_edge = notional * abs(float(expected_return)) * float(self.edge_horizon_bars)
        round_trip_cost = notional * 2.0 * (self.fee_rate + self.slippage_rate)
        required_edge = round_trip_cost * self.min_edge_over_cost_ratio
        return est_edge >= required_edge

    def on_bar(self, bar: Dict[str, Any]):
        self.bar_index += 1
        symbol = bar.get("symbol")
        close = bar.get("close")
        ts = bar.get("timestamp")
        
        if close is None or ts is None:
            return
        ts_utc = self._normalize_timestamp(ts)
        if ts_utc is None:
            return
            
        self.history.append(float(close))
        self.timestamps.append(ts_utc)
        
        # We need enough history for BOTH lookback and trend filter
        min_required = max(self.lookback_bars, self.trend_filter_period) + 1
        if len(self.history) < min_required:
            return
            
        # Limit history to save memory but keep enough for trend filter
        if len(self.history) > min_required + 50:
            self.history = self.history[-(min_required + 50):]
            
        # 0. Cooldown Filter
        is_cooldown_active = self._is_cooldown_active(ts_utc)
            
        # 1. Trend Filter (EMA 200)
        ema_trend = self._calculate_ema(self.history, self.trend_filter_period)
        is_uptrend = float(close) >= ema_trend
        
        # 2. Optimization
        prices_for_returns = np.array(self.history[-(self.lookback_bars + 1):])
        returns = self._calculate_log_returns(prices_for_returns)
        
        target_weight = self._optimize_weight(returns, is_uptrend)
        
        # Override target weight to 0.0 if cooldown is active to avoid opening new positions
        if is_cooldown_active:
            target_weight = 0.0
        
        # Execute rebalance if significantly different (Turnover penalty already handles small drifts in objective)
        if abs(target_weight - self.current_weight) > self.min_rebalance_weight_delta:
            self._rebalance(symbol, target_weight, float(close), ts_utc)

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
        
        # Check if closing/reducing
        is_closing = False
        if current_qty != 0.0 and ((current_qty > 0 and diff_qty < 0) or (current_qty < 0 and diff_qty > 0)):
            is_closing = True

        trade_notional = abs(diff_notional)
        if trade_notional < self.min_rebalance_notional:
            return
        if not is_closing and not self._passes_cost_gate(trade_notional, self._last_expected_return):
            return
            
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
        
        placed_order = self.submit_order(order)
        if placed_order:
            self.current_weight = target_weight
            if is_closing:
                self.closing_order_ids.add(placed_order.id)

    def on_fill(self, trade: Trade):
        if trade.order_id in self.closing_order_ids:
            self.closing_order_ids.remove(trade.order_id)
            if float(trade.pnl) <= 0.0:
                trade_ts = self._normalize_timestamp(trade.timestamp)
                if trade_ts is None:
                    return
                self.cooldown_until = trade_ts + timedelta(hours=self.cooldown_hours)
                print(f"[{trade.timestamp}] Cooldown triggered for {self.cooldown_hours}h due to closing trade PnL={trade.pnl:.2f}")
