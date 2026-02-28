from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from src.simulation.models import Order, OrderType, Side, Trade
from src.simulation.strategy import Strategy
from src.simulation.indicators import calculate_atr

class MDEMADV4SLTestStrategy(Strategy):
    """
    Test version of V4 with an added ATR-based hard Stop-Loss.
    """

    EPS = 1e-9

    def __init__(
        self,
        timeframe: str = "4h",
        lookback_bars: int = 100,
        risk_aversion: float = 2.5,
        entropy_weight: float = 0.1,
        turnover_penalty: float = 0.1,
        sl_atr_multiplier: float = 3.0, # The multiplier to test
        enable_shorts: bool = True,
        min_leverage: int = 1,
        max_leverage: int = 1,
    ):
        super().__init__()
        self.timeframe = timeframe
        self.lookback_bars = int(lookback_bars)
        self.risk_aversion = float(risk_aversion)
        self.entropy_weight = float(entropy_weight)
        self.turnover_penalty = float(turnover_penalty)
        self.sl_atr_multiplier = float(sl_atr_multiplier)
        self.enable_shorts = bool(enable_shorts)
        self.min_leverage = max(1, int(min_leverage))
        self.max_leverage = max(self.min_leverage, int(max_leverage))

        self.bar_index = 0
        self.history_df = pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
        self.current_weight = 0.0
        self.active_stop_id: Optional[str] = None

    def on_start(self):
        pass

    def on_stop(self):
        pass

    def _calculate_log_returns(self, prices: np.ndarray) -> np.ndarray:
        if len(prices) < 2: return np.array([])
        # Ensure prices is a float array
        p = prices.astype(float)
        return np.log(p[1:] / p[:-1])

    def _calculate_mad(self, returns: np.ndarray) -> float:
        if len(returns) == 0: return 0.0
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
        utility = portfolio_return - (self.risk_aversion * portfolio_risk) + (self.entropy_weight * entropy) - (self.turnover_penalty * turnover)
        return -utility

    def _optimize_weight(self, returns: np.ndarray) -> float:
        if len(returns) < 5: return 0.0
        expected_return = np.mean(returns)
        mad = self._calculate_mad(returns)
        if mad <= self.EPS:
            return float(self.max_leverage) if expected_return > 0 else 0.0
        lower_bound = -float(self.max_leverage) if self.enable_shorts else 0.0
        upper_bound = float(self.max_leverage)
        res = minimize(self._objective, x0=np.array([self.current_weight]), args=(expected_return, mad, self.current_weight), bounds=[(lower_bound, upper_bound)], method='L-BFGS-B')
        return float(res.x[0]) if res.success else self.current_weight

    def on_bar(self, bar: Dict[str, Any]):
        self.bar_index += 1
        symbol, close, ts = bar.get("symbol"), bar.get("close"), bar.get("timestamp")
        if close is None or ts is None: return
        
        # Check if we were stopped out
        if self.broker:
            pos = self.get_position_size(symbol)
            if abs(pos) < self.EPS and self.current_weight != 0:
                # We had a weight but no position -> likely stopped out
                self.current_weight = 0.0
                self.active_stop_id = None
            
        new_row = pd.DataFrame([{"open": bar["open"], "high": bar["high"], "low": bar["low"], "close": bar["close"], "volume": bar.get("volume", 0.0)}])
        self.history_df = pd.concat([self.history_df, new_row], ignore_index=True)
        if len(self.history_df) > self.lookback_bars + 20:
            self.history_df = self.history_df.iloc[- (self.lookback_bars + 20):]
            
        if len(self.history_df) < self.lookback_bars + 1: return
            
        prices = self.history_df["close"].values[-(self.lookback_bars + 1):]
        returns = self._calculate_log_returns(prices)
        target_weight = self._optimize_weight(returns)
        
        if abs(target_weight - self.current_weight) > 0.005:
            # Calculate ATR for Stop Loss
            atr_series = calculate_atr(self.history_df["high"], self.history_df["low"], self.history_df["close"], 14)
            current_atr = float(atr_series.iloc[-1]) if not atr_series.empty else close * 0.02
            
            self._rebalance(symbol, target_weight, float(close), ts, current_atr)

    def _rebalance(self, symbol: str, target_weight: float, price: float, timestamp: Any, atr: float):
        if not self.broker: return
        
        # 1. Cancel existing Stop orders for this symbol
        if self.active_stop_id:
            self.cancel_order(self.active_stop_id)
            self.active_stop_id = None

        equity = self.broker.equity
        target_notional = equity * target_weight
        current_qty = self.get_position_size(symbol)
        diff_qty = (target_notional - (current_qty * price)) / price
        
        # 2. Execute Market Rebalance
        if abs(diff_qty) >= 1e-6:
            side = Side.BUY if diff_qty > 0 else Side.SELL
            order = Order(
                id="", symbol=symbol, side=side, order_type=OrderType.MARKET,
                quantity=abs(float(diff_qty)), price=None, timestamp=timestamp,
                leverage=max(1, int(abs(target_weight)))
            )
            if self.submit_order(order):
                self.current_weight = target_weight

        # 3. Set new Stop Loss Order if we have a position
        final_qty = self.get_position_size(symbol)
        if abs(final_qty) > 1e-6:
            sl_side = Side.SELL if final_qty > 0 else Side.BUY
            if final_qty > 0:
                stop_price = price - (self.sl_atr_multiplier * atr)
            else:
                stop_price = price + (self.sl_atr_multiplier * atr)
            
            stop_order = Order(
                id="", symbol=symbol, side=sl_side, order_type=OrderType.STOP,
                quantity=abs(float(final_qty)), stop_price=float(stop_price), timestamp=timestamp
            )
            placed_stop = self.submit_order(stop_order)
            if placed_stop:
                self.active_stop_id = placed_stop.id
                # print(f"Placed SL at {stop_price} for {final_qty}")

    def on_fill(self, trade: Trade):
        # If a STOP order was filled, our weight is now effectively 0
        if self.active_stop_id and getattr(trade, "order_id", None) == self.active_stop_id:
            print(f"!!! STOP LOSS FILLED at {trade.price} !!!")
            self.current_weight = 0.0
            self.active_stop_id = None
