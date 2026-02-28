"""
backtest_turnover_penalty.py – Optimierte Version

Verbesserungen gegenüber Original:
  1. Parquet-Cache (data_cache.py): 8h-Candles werden direkt geladen und
     lokal gecacht. Kein Resample-Overhead, kein QuestDB-Timeout beim 2. Run.
  2. Parallele Penalty-Runs: alle RunConfigs laufen gleichzeitig via
     ProcessPoolExecutor (max. cpu_count//2, mind. 2 Worker).
  3. NumPy-Optimierungen im Rebalancing-Scaling-Schritt.

Umgebungsvariablen:
  EDGECRAFT_START_DATE   (default: 2024-02-22T00:00:00)
  EDGECRAFT_END_DATE     (default: 2026-02-22T00:00:00)
  EDGECRAFT_TIMEFRAME    (default: 8h)
  EDGECRAFT_SYMBOLS      (kommagetrennte Liste, default: DEFAULT_SYMBOLS)
  EDGECRAFT_MAX_WORKERS  (default: cpu_count//2, mind. 2)
  EDGECRAFT_CACHE_REFRESH  (1 = Cache neu aufbauen)
"""
from __future__ import annotations

import contextlib
import io
import math
import os
import sys
import uuid
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.simulation.metrics import calculate_metrics
from src.simulation.models import Order, OrderStatus, OrderType, Position, Side, Trade
from src.simulation.strategy import Strategy
from verification.data_cache import load_many


DEFAULT_SYMBOLS: List[str] = [
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

START_DATE = os.getenv("EDGECRAFT_START_DATE", "2024-02-22T00:00:00")
END_DATE = os.getenv("EDGECRAFT_END_DATE", "2026-02-22T00:00:00")
TIMEFRAME = os.getenv("EDGECRAFT_TIMEFRAME", "8h")
INITIAL_BALANCE = 1000.0
BROKER_LEVERAGE = 3


def _parse_symbols_env() -> List[str]:
    raw = os.getenv("EDGECRAFT_SYMBOLS", "").strip()
    if not raw:
        return DEFAULT_SYMBOLS
    symbols = [s.strip().upper() for s in raw.split(",") if s.strip()]
    deduped: List[str] = []
    seen = set()
    for s in symbols:
        if s not in seen:
            seen.add(s)
            deduped.append(s)
    return deduped or DEFAULT_SYMBOLS


SYMBOLS: List[str] = _parse_symbols_env()


@dataclass
class ExecutionModel:
    name: str
    market_fill_price: str = "close"  # close | open
    order_delay_bars: int = 1
    half_spread_bps: float = 0.0
    taker_slippage_bps: float = 0.0
    impact_bps_at_100pct_volume: float = 0.0
    max_bar_participation: float = 1.0


LEGACY_EXECUTION = ExecutionModel(
    name="legacy_close",
    market_fill_price="close",
    order_delay_bars=1,
    half_spread_bps=0.0,
    taker_slippage_bps=0.0,
    impact_bps_at_100pct_volume=0.0,
    max_bar_participation=1.0,
)

STRICT_EXECUTION = ExecutionModel(
    name="strict_next_open",
    market_fill_price="open",
    order_delay_bars=1,
    half_spread_bps=1.5,
    taker_slippage_bps=1.0,
    impact_bps_at_100pct_volume=25.0,
    max_bar_participation=0.15,
)


# ---------------------------------------------------------------------------
# Strategie
# ---------------------------------------------------------------------------

class ServerLikeMdeMadV2Strategy(Strategy):
    """
    Server-nahe Approximation der mde_mad_v2-Strategie:
    - Diskrete Optimierung (coarse + local search)
    - Kein Cooldown / Cost Gate
    - Rebalance-Schwelle = 0.5 % des aktuellen Eigenkapitals
    - enableShorts default True, maxLeverage default 3
    """

    EPS = 1e-9

    def __init__(
        self,
        lookback_bars: int = 30,
        trend_filter_period: int = 200,
        risk_aversion: float = 2.0,
        entropy_weight: float = 0.1,
        turnover_penalty: float = 0.02,
        max_leverage: float = 3.0,
        enable_shorts: bool = True,
        optimization_steps: int = 41,
        entry_weight_threshold: float = 0.10,
        trend_filter_type: str = "EMA",
    ):
        super().__init__()
        self.lookback_bars = int(lookback_bars)
        self.trend_filter_period = int(trend_filter_period)
        self.risk_aversion = float(risk_aversion)
        self.entropy_weight = float(entropy_weight)
        self.turnover_penalty = float(turnover_penalty)
        self.max_leverage = float(max_leverage)
        self.enable_shorts = bool(enable_shorts)
        self.optimization_steps = int(optimization_steps)
        self.entry_weight_threshold = float(entry_weight_threshold)
        self.trend_filter_type = trend_filter_type.upper()

        self.history: List[float] = []
        self.current_weight = 0.0
        self.latest_target_weight: Optional[float] = None
        self.latest_close: float = 0.0
        self.latest_ts: Any = None

    def on_start(self):
        pass

    def on_stop(self):
        pass

    def _calculate_log_returns(self, prices: List[float]) -> np.ndarray:
        arr = np.asarray(prices, dtype=float)
        if len(arr) < 2:
            return np.empty(0)
        return np.log(arr[1:] / arr[:-1])

    def _calculate_mad(self, returns: np.ndarray) -> float:
        if len(returns) == 0:
            return 0.0
        return float(np.mean(np.abs(returns - returns.mean())))

    # --- Trend-Hilfsfunktionen (EMA, SMA, WMA, HMA, KAMA) ---

    def _ema_series(self, values: List[float], period: int) -> np.ndarray:
        arr = np.asarray(values, dtype=float)
        out = np.zeros(len(arr))
        if len(arr) < period:
            return out
        k = 2.0 / (period + 1.0)
        out[period - 1] = arr[:period].mean()
        for i in range(period, len(arr)):
            out[i] = arr[i] * k + out[i - 1] * (1.0 - k)
        return out

    def _ema_value(self, values: List[float], period: int) -> float:
        if not values:
            return 0.0
        if len(values) < period:
            return values[-1]
        series = self._ema_series(values, period)
        last = series[-1]
        if np.isfinite(last) and last > 0:
            return float(last)
        return float(values[-1])

    def _sma_value(self, values: List[float], period: int) -> float:
        if not values:
            return 0.0
        if len(values) < period:
            period = len(values)
        return float(np.mean(values[-period:]))

    def _wma_series(self, values: List[float], period: int) -> np.ndarray:
        arr = np.asarray(values, dtype=float)
        out = np.zeros(len(arr))
        if len(arr) < period:
            return out
        weights = np.arange(1, period + 1, dtype=float)
        wsum = weights.sum()
        for i in range(period - 1, len(arr)):
            out[i] = np.dot(arr[i - period + 1 : i + 1], weights) / wsum
        return out

    def _hma_series(self, values: List[float], period: int) -> np.ndarray:
        if len(values) < period:
            v = values[-1] if values else 0.0
            return np.full(len(values), v)
        half = max(1, period // 2)
        sqrtp = max(1, int(math.sqrt(period)))
        w_half = self._wma_series(values, half)
        w_full = self._wma_series(values, period)
        diff = (2.0 * w_half - w_full).tolist()
        return self._wma_series(diff, sqrtp)

    def _kama_series(self, values: List[float], period: int) -> np.ndarray:
        arr = np.asarray(values, dtype=float)
        out = np.zeros(len(arr))
        if len(arr) < period:
            if len(arr):
                out[:] = arr[-1]
            return out
        fast_sc = 2.0 / 3.0
        slow_sc = 2.0 / 31.0
        out[period - 1] = arr[period - 1]
        for i in range(period, len(arr)):
            change = abs(arr[i] - arr[i - period])
            volatility = np.sum(np.abs(np.diff(arr[i - period : i + 1])))
            er = 0.0 if volatility == 0 else change / volatility
            sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
            out[i] = out[i - 1] + sc * (arr[i] - out[i - 1])
        return out

    def _trend_value(self, closes: List[float]) -> float:
        t = self.trend_filter_type
        if t == "HMA":
            return float(self._hma_series(closes, self.trend_filter_period)[-1])
        if t == "KAMA":
            return float(self._kama_series(closes, self.trend_filter_period)[-1])
        if t == "SMA":
            return self._sma_value(closes, self.trend_filter_period)
        return self._ema_value(closes, self.trend_filter_period)

    # --- Utility / Optimierung ---

    def _utility(self, weights: np.ndarray, expected_return: float, mad: float) -> np.ndarray:
        w_abs = np.abs(weights)
        w_cash = np.maximum(0.0, 1.0 - w_abs)
        p_asset = np.maximum(w_abs, self.EPS)
        p_cash = np.maximum(w_cash, self.EPS)
        p_sum = p_asset + p_cash
        pa = p_asset / p_sum
        pc = p_cash / p_sum
        entropy = -((pa * np.log(pa)) + (pc * np.log(pc)))
        turnover = np.abs(weights - self.current_weight)
        return (
            weights * expected_return
            - self.risk_aversion * w_abs * mad
            + self.entropy_weight * entropy
            - self.turnover_penalty * turnover
        )

    def _optimize_weight(self, returns: np.ndarray, is_uptrend: bool) -> float:
        if len(returns) < 5:
            return 0.0
        expected_return = float(returns.mean())
        mad = self._calculate_mad(returns)

        if mad <= self.EPS:
            if is_uptrend and expected_return > 0:
                return 1.0
            if (not is_uptrend) and self.enable_shorts and expected_return < 0:
                return -1.0
            return 0.0

        lower = 0.0
        upper = self.max_leverage if is_uptrend else 0.0
        if (not is_uptrend) and self.enable_shorts:
            lower = -self.max_leverage
            upper = 0.0
        if upper <= lower:
            return 0.0

        steps = max(9, self.optimization_steps)
        coarse = np.linspace(lower, upper, steps)
        # Aktuelles Gewicht als Kandidat hinzufügen
        cur_clipped = min(upper, max(lower, self.current_weight))
        coarse = np.unique(np.append(coarse, cur_clipped))

        utils = self._utility(coarse, expected_return, mad)
        best_idx = int(np.argmax(utils))
        best_w = float(coarse[best_idx])

        step_size = (upper - lower) / max(1, steps - 1)
        local = np.linspace(
            max(lower, best_w - step_size),
            min(upper, best_w + step_size),
            11,
        )
        local_u = self._utility(local, expected_return, mad)
        best_w = float(local[int(np.argmax(local_u))])
        return best_w

    def on_bar(self, bar: Dict[str, Any]):
        close = bar.get("close")
        symbol = bar.get("symbol")
        ts = bar.get("timestamp")
        if close is None or symbol is None or ts is None:
            return
        self.history.append(float(close))
        min_required = max(self.lookback_bars, self.trend_filter_period if self.trend_filter_period > 0 else 0) + 1
        if len(self.history) < min_required:
            return
        window = max(min_required + 120, 500)
        if len(self.history) > window:
            self.history = self.history[-window:]

        closes = [x for x in self.history if math.isfinite(x) and x > 0]
        if len(closes) < 5:
            return

        price_window = closes[-(self.lookback_bars + 1):]
        returns = self._calculate_log_returns(price_window)
        if len(returns) < 5:
            return

        if self.trend_filter_period > 0:
            trend_value = self._trend_value(closes)
            is_uptrend = closes[-1] >= trend_value
        else:
            # Kein Trend-Filter: Richtung basierend auf expected return (Long/Short frei)
            is_uptrend = float(returns.mean()) >= 0

        target_weight = self._optimize_weight(returns, is_uptrend)

        self.latest_target_weight = target_weight
        self.latest_close = float(close)
        self.latest_ts = ts

    def _rebalance_server(self, symbol: str, target_weight: float, price: float, timestamp: Any):
        if not self.broker:
            return
        equity = float(self.broker.equity)
        target_notional = equity * float(target_weight)
        current_qty = float(self.get_position_size(symbol))
        current_notional = current_qty * price
        diff_notional = target_notional - current_notional

        if abs(diff_notional) < equity * 0.005:
            return

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
            leverage=max(1, int(abs(target_weight))),
        )
        self.submit_order(order)

    def on_fill(self, trade: Trade):
        pass


# ---------------------------------------------------------------------------
# Broker
# ---------------------------------------------------------------------------

class SharedPortfolioBroker:
    def __init__(
        self,
        initial_balance: float,
        leverage: int,
        execution: ExecutionModel,
        taker_fee_rate: float = 0.0004,
        maker_fee_rate: float = 0.0002,
    ):
        self.balance = float(initial_balance)
        self.equity = float(initial_balance)
        self.leverage = int(max(1, leverage))
        self.execution = execution
        self.taker_fee_rate = float(taker_fee_rate)
        self.maker_fee_rate = float(maker_fee_rate)

        self.positions: Dict[str, Position] = {}
        self.open_orders: Dict[str, Order] = {}
        self.order_history: List[Order] = []
        self.trades: List[Trade] = []
        self.last_prices: Dict[str, float] = {}

        self._bar_seq_by_symbol: Dict[str, int] = {}
        self._order_created_seq: Dict[str, int] = {}

    def _as_str(self, value: Any) -> str:
        return str(getattr(value, "value", value))

    def get_position(self, symbol: str) -> Optional[Position]:
        return self.positions.get(symbol)

    def get_used_margin(self) -> float:
        pos_margin = sum(
            abs(float(p.size) * float(p.entry_price)) / max(1, int(p.leverage))
            for p in self.positions.values()
        )
        order_margin = 0.0
        for order in self.open_orders.values():
            if order.price and float(order.price) > 0:
                remaining = max(0.0, float(order.quantity) - float(order.filled_quantity))
                order_margin += (remaining * float(order.price)) / max(1, int(order.leverage))
        return pos_margin + order_margin

    def get_margin_balance(self) -> float:
        return self.balance + sum(float(p.unrealized_pnl) for p in self.positions.values())

    def get_available_balance(self) -> float:
        return self.get_margin_balance() - self.get_used_margin()

    def submit_order(self, order: Order) -> Optional[Order]:
        if int(order.leverage) <= 1 and self.leverage != 1:
            order.leverage = self.leverage
        if (order.price is not None) and float(order.price) > 0:
            required_margin = (float(order.quantity) * float(order.price)) / max(1, int(order.leverage))
            if required_margin > self.get_available_balance():
                order.status = OrderStatus.REJECTED.value
                self.order_history.append(order)
                return order
        if not order.id:
            order.id = str(uuid.uuid4())
        self.open_orders[order.id] = order
        self.order_history.append(order)
        self._order_created_seq[order.id] = self._bar_seq_by_symbol.get(order.symbol, 0)
        return order

    def cancel_order(self, order_id: str) -> bool:
        order = self.open_orders.pop(order_id, None)
        if order is None:
            return False
        order.status = OrderStatus.CANCELED.value
        self._order_created_seq.pop(order_id, None)
        return True

    def _update_pnl(self):
        total_unrealized = 0.0
        for pos in self.positions.values():
            if abs(float(pos.size)) <= 1e-12:
                pos.unrealized_pnl = 0.0
                continue
            px = self.last_prices.get(pos.symbol, float(pos.entry_price))
            pnl = (float(px) - float(pos.entry_price)) * float(pos.size)
            pos.unrealized_pnl = pnl
            pos.initial_margin = (abs(float(pos.size)) * float(pos.entry_price)) / max(1, int(pos.leverage))
            pos.maintenance_margin = abs(float(pos.size) * float(px)) * 0.005
            total_unrealized += pnl
        self.equity = self.balance + total_unrealized

    def process_funding(self, symbol: str, funding_rate: float):
        pos = self.positions.get(symbol)
        if pos is None or abs(float(pos.size)) <= 1e-12:
            return
        px = self.last_prices.get(symbol, float(pos.entry_price))
        payment = float(pos.size) * float(px) * float(funding_rate)
        self.balance -= payment
        self._update_pnl()

    def _apply_taker_adjustments(self, raw_price: float, side: str, qty: float, bar_volume: float) -> float:
        ex = self.execution
        if ex.half_spread_bps <= 0.0 and ex.taker_slippage_bps <= 0.0 and ex.impact_bps_at_100pct_volume <= 0.0:
            return raw_price
        volume = max(1e-12, float(bar_volume))
        participation = float(qty) / volume
        impact_bps = float(ex.impact_bps_at_100pct_volume) * min(1.0, max(0.0, participation))
        total_bps = float(ex.half_spread_bps) + float(ex.taker_slippage_bps) + impact_bps
        if side == Side.BUY.value:
            return raw_price * (1.0 + total_bps / 10_000.0)
        return raw_price * (1.0 - total_bps / 10_000.0)

    def _passes_margin_check(self, symbol: str, side: str, qty: float, price: float, leverage: int, fee: float) -> bool:
        leverage = max(1, int(leverage))
        pos = self.positions.get(symbol)
        cur_size = float(pos.size) if pos else 0.0
        cur_entry = float(pos.entry_price) if pos else 0.0
        new_size, new_entry, realized_pnl = cur_size, cur_entry, 0.0

        if side == Side.BUY.value:
            if cur_size >= 0.0:
                combined = cur_size + qty
                new_entry = ((cur_size * cur_entry) + (qty * price)) / combined if combined > 0 else 0.0
                new_size = combined
            else:
                closed_qty = min(abs(cur_size), qty)
                realized_pnl = (cur_entry - price) * closed_qty
                remaining = cur_size + qty
                new_size = remaining
                new_entry = (0.0 if abs(remaining) <= 1e-12 else price if remaining > 0.0 else cur_entry)
        else:
            if cur_size <= 0.0:
                combined = cur_size - qty
                new_entry = ((abs(cur_size) * cur_entry) + (qty * price)) / abs(combined) if abs(combined) > 0 else 0.0
                new_size = combined
            else:
                closed_qty = min(cur_size, qty)
                realized_pnl = (price - cur_entry) * closed_qty
                remaining = cur_size - qty
                new_size = remaining
                new_entry = (0.0 if abs(remaining) <= 1e-12 else price if remaining < 0.0 else cur_entry)

        if abs(new_size) <= abs(cur_size) + 1e-12:
            return True

        balance_after = self.balance - fee + realized_pnl
        used_margin_after = 0.0
        unrealized_after = 0.0
        for sym, p in self.positions.items():
            size = float(p.size)
            entry = float(p.entry_price)
            lev = max(1, int(p.leverage))
            if sym == symbol:
                size, entry, lev = new_size, new_entry, leverage
            used_margin_after += abs(size * entry) / lev
            mark = price if sym == symbol else self.last_prices.get(sym, entry)
            unrealized_after += (mark - entry) * size

        if symbol not in self.positions:
            used_margin_after += abs(new_size * new_entry) / leverage
            unrealized_after += (price - new_entry) * new_size

        return balance_after + unrealized_after + 1e-9 >= used_margin_after

    def _execute_trade(
        self,
        order: Order,
        qty: float,
        price: float,
        timestamp: datetime,
        is_taker: bool,
        skip_margin_check: bool = False,
    ) -> Optional[Trade]:
        qty = float(qty)
        if qty <= 1e-12:
            return None
        side = self._as_str(order.side)
        notional_value = qty * price
        fee_rate = self.taker_fee_rate if is_taker else self.maker_fee_rate
        fee = notional_value * fee_rate

        if (not skip_margin_check) and (not self._passes_margin_check(order.symbol, side, qty, price, int(order.leverage), fee)):
            return None

        self.balance -= fee

        if order.symbol not in self.positions:
            self.positions[order.symbol] = Position(
                symbol=order.symbol, size=0.0, entry_price=0.0, leverage=max(1, int(order.leverage))
            )

        pos = self.positions[order.symbol]
        pos.leverage = max(1, int(order.leverage))
        realized_pnl = 0.0

        if side == Side.BUY.value:
            if float(pos.size) >= 0.0:
                new_size = float(pos.size) + qty
                total_cost = float(pos.size) * float(pos.entry_price) + qty * price
                pos.entry_price = (total_cost / new_size) if new_size > 0 else 0.0
                pos.size = new_size
            else:
                closed_qty = min(abs(float(pos.size)), qty)
                realized_pnl = (float(pos.entry_price) - price) * closed_qty
                remaining_qty = float(pos.size) + qty
                self.balance += realized_pnl
                pos.size = remaining_qty
                if abs(float(pos.size)) <= 1e-12:
                    pos.size = 0.0
                    pos.entry_price = 0.0
                elif float(pos.size) > 0.0:
                    pos.entry_price = price
        else:
            if float(pos.size) <= 0.0:
                new_size = float(pos.size) - qty
                total_cost = abs(float(pos.size)) * float(pos.entry_price) + qty * price
                pos.entry_price = (total_cost / abs(new_size)) if abs(new_size) > 0 else 0.0
                pos.size = new_size
            else:
                closed_qty = min(float(pos.size), qty)
                realized_pnl = (price - float(pos.entry_price)) * closed_qty
                remaining_qty = float(pos.size) - qty
                self.balance += realized_pnl
                pos.size = remaining_qty
                if abs(float(pos.size)) <= 1e-12:
                    pos.size = 0.0
                    pos.entry_price = 0.0
                elif float(pos.size) < 0.0:
                    pos.entry_price = price

        trade = Trade(
            id=str(uuid.uuid4()),
            order_id=order.id,
            symbol=order.symbol,
            side=side,
            quantity=qty,
            price=float(price),
            timestamp=timestamp,
            fee=float(fee),
            pnl=float(realized_pnl),
        )
        self.trades.append(trade)
        self._update_pnl()
        return trade

    def _liquidation_needed(self) -> bool:
        self._update_pnl()
        maintenance = sum(float(p.maintenance_margin) for p in self.positions.values())
        return self.equity < maintenance

    def _liquidate_all(self, timestamp: datetime) -> List[Trade]:
        trades: List[Trade] = []
        for sym, pos in list(self.positions.items()):
            size = float(pos.size)
            if abs(size) <= 1e-12:
                continue
            px = float(self.last_prices.get(sym, float(pos.entry_price)))
            side = Side.SELL.value if size > 0 else Side.BUY.value
            dummy = Order(
                id="LIQUIDATION", symbol=sym, side=side, order_type=OrderType.MARKET,
                quantity=abs(size), price=None, timestamp=timestamp, leverage=max(1, int(pos.leverage)),
            )
            tr = self._execute_trade(dummy, abs(size), px, timestamp, is_taker=True, skip_margin_check=True)
            if tr is not None:
                trades.append(tr)
        return trades

    def process_bar(self, bar: Dict[str, Any]) -> List[Trade]:
        symbol = str(bar["symbol"])
        ts = bar["timestamp"]
        if isinstance(ts, pd.Timestamp):
            ts = ts.to_pydatetime()

        close = float(bar["close"])
        open_px = float(bar.get("open", close))
        high_px = float(bar.get("high", close))
        low_px = float(bar.get("low", close))
        volume = max(0.0, float(bar.get("volume", 0.0)))

        self._bar_seq_by_symbol[symbol] = self._bar_seq_by_symbol.get(symbol, 0) + 1
        cur_seq = self._bar_seq_by_symbol[symbol]
        self.last_prices[symbol] = close
        self._update_pnl()

        new_fills: List[Trade] = []
        if self._liquidation_needed():
            new_fills.extend(self._liquidate_all(ts))

        filled_ids: List[str] = []
        rejected_ids: List[str] = []

        for order_id, order in list(self.open_orders.items()):
            if order.symbol != symbol:
                continue
            created_seq = self._order_created_seq.get(order_id, 0)
            if cur_seq < (created_seq + max(1, int(self.execution.order_delay_bars))):
                continue

            order_type = self._as_str(order.order_type)
            side = self._as_str(order.side)
            remaining_qty = max(0.0, float(order.quantity) - float(order.filled_quantity))
            if remaining_qty <= 1e-12:
                filled_ids.append(order_id)
                continue

            fill_price: Optional[float] = None
            is_taker = True

            if order_type == OrderType.MARKET.value:
                fill_price = open_px if self.execution.market_fill_price == "open" else close
                is_taker = True
            elif order_type == OrderType.LIMIT.value:
                limit_px = None if order.price is None else float(order.price)
                if limit_px is not None:
                    if side == Side.BUY.value:
                        if open_px <= limit_px:
                            fill_price, is_taker = open_px, False
                        elif low_px <= limit_px:
                            fill_price, is_taker = limit_px, False
                    elif side == Side.SELL.value:
                        if open_px >= limit_px:
                            fill_price, is_taker = open_px, False
                        elif high_px >= limit_px:
                            fill_price, is_taker = limit_px, False
            elif order_type == OrderType.STOP.value:
                stop_px = None if order.stop_price is None else float(order.stop_price)
                if stop_px is not None:
                    if side == Side.BUY.value:
                        if open_px >= stop_px:
                            fill_price = open_px
                        elif high_px >= stop_px:
                            fill_price = stop_px
                    elif side == Side.SELL.value:
                        if open_px <= stop_px:
                            fill_price = open_px
                        elif low_px <= stop_px:
                            fill_price = stop_px

            if fill_price is None:
                continue

            max_fill_qty = remaining_qty
            if volume > 0.0 and self.execution.max_bar_participation < 1.0:
                max_fill_qty = min(remaining_qty, volume * max(0.0, float(self.execution.max_bar_participation)))
            if max_fill_qty <= 1e-12:
                continue

            exec_price = float(fill_price)
            if is_taker:
                exec_price = self._apply_taker_adjustments(exec_price, side, max_fill_qty, volume)

            tr = self._execute_trade(order, max_fill_qty, exec_price, ts, is_taker=is_taker)
            if tr is None:
                order.status = OrderStatus.REJECTED.value
                rejected_ids.append(order_id)
                continue

            prev_filled = float(order.filled_quantity)
            new_filled = prev_filled + max_fill_qty
            order.filled_quantity = new_filled
            if new_filled > 0.0:
                order.average_fill_price = (
                    (prev_filled * float(order.average_fill_price) + max_fill_qty * exec_price) / new_filled
                )
            if new_filled + 1e-12 >= float(order.quantity):
                order.status = OrderStatus.FILLED.value
                filled_ids.append(order_id)
            else:
                order.status = OrderStatus.PARTIALLY_FILLED.value
            new_fills.append(tr)

        for oid in filled_ids:
            self.open_orders.pop(oid, None)
            self._order_created_seq.pop(oid, None)
        for oid in rejected_ids:
            self.open_orders.pop(oid, None)
            self._order_created_seq.pop(oid, None)

        return new_fills


# ---------------------------------------------------------------------------
# Konfigurations-Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class RunConfig:
    name: str
    penalty: float


@dataclass
class RunResult:
    run: str
    execution: str
    timeframe: str
    active_symbols: int
    missing_symbols: int
    total_return: float
    final_equity: float
    max_drawdown: float
    max_drawdown_close: float
    max_drawdown_worst: float
    sharpe_ratio: float
    sortino_ratio: float
    total_trades: int
    avg_gross_exposure: float
    max_gross_exposure: float


# ---------------------------------------------------------------------------
# Factory / Hilfsfunktionen
# ---------------------------------------------------------------------------

def _make_strategy(penalty: float) -> Strategy:
    no_trend_filter = os.getenv("EDGECRAFT_NO_TREND_FILTER", "").strip() in {"1", "true", "yes"}
    trend_period = 0 if no_trend_filter else int(os.getenv("EDGECRAFT_TREND_FILTER_PERIOD", "200"))
    return ServerLikeMdeMadV2Strategy(
        lookback_bars=30,
        trend_filter_period=trend_period,
        risk_aversion=2.0,
        entropy_weight=0.1,
        turnover_penalty=penalty,
        max_leverage=3.0,
        enable_shorts=True,
        optimization_steps=41,
        entry_weight_threshold=0.10,
        trend_filter_type="EMA",
    )


def _normalize_index(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    out = df.copy()
    idx = pd.to_datetime(out.index, utc=True, errors="coerce")
    out.index = idx
    return out[~out.index.isna()].sort_index()


def _load_data(
    symbols: List[str],
    force_refresh: bool = False,
) -> Tuple[Dict[str, pd.DataFrame], Dict[pd.Timestamp, List[Tuple[str, float]]], List[str]]:
    """
    Lädt 8h-OHLCV direkt (kein Resample-Overhead) via Parquet-Cache.
    Funding-Rates kommen weiterhin live aus QuestDB (klein, schnell).
    """
    from src.database import QuestDBManager

    # OHLCV via Cache
    bars_by_symbol, missing = load_many(
        symbols,
        timeframe=TIMEFRAME,
        start=START_DATE,
        end=END_DATE,
        force_refresh=force_refresh,
        verbose=True,
    )

    # Funding-Rates (kein Cache nötig – viel weniger Daten als OHLCV)
    funding_schedule: Dict[pd.Timestamp, List[Tuple[str, float]]] = {}
    db = QuestDBManager()
    for sym in symbols:
        if sym in missing:
            continue
        try:
            funding = db.get_funding_rates(sym, START_DATE, END_DATE)
            funding = _normalize_index(funding)
            if not funding.empty and "funding_rate" in funding.columns:
                rates = pd.to_numeric(funding["funding_rate"], errors="coerce").dropna()
                for ts, rate in rates.items():
                    funding_schedule.setdefault(ts, []).append((sym, float(rate)))
        except Exception as exc:
            print(f"  [funding] Warnung: {sym}: {exc}")

    return bars_by_symbol, funding_schedule, missing


def _build_events(
    bars_by_symbol: Dict[str, pd.DataFrame],
) -> Dict[pd.Timestamp, List[Dict[str, Any]]]:
    events: Dict[pd.Timestamp, List[Dict[str, Any]]] = {}
    for sym, df in bars_by_symbol.items():
        for ts, row in df.iterrows():
            payload = {
                "symbol": sym,
                "timestamp": ts.to_pydatetime(),
                "open": float(row["open"]),
                "high": float(row["high"]),
                "low": float(row["low"]),
                "close": float(row["close"]),
                "volume": float(row["volume"]),
            }
            events.setdefault(ts, []).append(payload)
    for ts in list(events.keys()):
        events[ts].sort(key=lambda x: str(x["symbol"]))
    return events


def _gross_exposure(broker: SharedPortfolioBroker) -> float:
    return sum(
        abs(float(pos.size) * float(broker.last_prices.get(sym, float(pos.entry_price))))
        for sym, pos in broker.positions.items()
    )


# ---------------------------------------------------------------------------
# Simulation (ein Run = eine RunConfig + ein ExecutionModel)
# ---------------------------------------------------------------------------

def _simulate_portfolio(
    run: RunConfig,
    execution: ExecutionModel,
    bars_by_symbol: Dict[str, pd.DataFrame],
    funding_schedule: Dict[pd.Timestamp, List[Tuple[str, float]]],
    missing_symbols: List[str],
) -> Tuple[RunResult, pd.DataFrame]:
    active_symbols = sorted(list(bars_by_symbol.keys()))
    events_by_ts = _build_events(bars_by_symbol)
    timestamps = sorted(events_by_ts.keys())

    broker = SharedPortfolioBroker(
        initial_balance=INITIAL_BALANCE,
        leverage=BROKER_LEVERAGE,
        execution=execution,
    )

    strategies: Dict[str, Strategy] = {}
    equity_curve: List[Dict[str, Any]] = []
    gross_exposure_points: List[float] = []

    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        for sym in active_symbols:
            strat = _make_strategy(run.penalty)
            strat.set_broker(broker)
            strat.on_start()
            strategies[sym] = strat

        for ts in timestamps:
            events = events_by_ts[ts]

            for bar in events:
                sym = str(bar["symbol"])
                fills = broker.process_bar(bar)
                for tr in fills:
                    strat = strategies.get(str(tr.symbol))
                    if strat is not None:
                        strat.on_fill(tr)
                strat = strategies.get(sym)
                if strat is not None:
                    strat.on_bar(bar)

            # Skalierung: Gesamt-Exposure darf max_leverage nicht übersteigen
            weights = np.array([
                getattr(strat, "latest_target_weight", None) or 0.0
                for strat in strategies.values()
            ], dtype=float)
            total_exposure = float(np.sum(np.abs(weights)))
            scale = min(1.0, 3.0 / total_exposure) if total_exposure > 3.0 else 1.0

            for (sym, strat), raw_w in zip(strategies.items(), weights):
                if getattr(strat, "latest_target_weight", None) is None:
                    continue
                final_target = float(raw_w) * scale
                strat._rebalance_server(sym, final_target, strat.latest_close, strat.latest_ts)
                if abs(final_target) >= strat.entry_weight_threshold:
                    strat.current_weight = final_target
                strat.latest_target_weight = None

            for sym, rate in funding_schedule.get(ts, []):
                broker.process_funding(sym, rate)

            broker._update_pnl()
            ge = _gross_exposure(broker)
            gross_exposure_points.append(ge / max(1e-9, broker.equity))
            equity_curve.append(
                {
                    "timestamp": ts.to_pydatetime(),
                    "equity": float(broker.equity),
                    "equity_worst": float(broker.equity),
                }
            )

        for strat in strategies.values():
            strat.on_stop()

    metrics = calculate_metrics(broker.trades, equity_curve)

    result = RunResult(
        run=run.name,
        execution=execution.name,
        timeframe=TIMEFRAME,
        active_symbols=len(active_symbols),
        missing_symbols=len(missing_symbols),
        total_return=float(metrics.get("total_return", 0.0)),
        final_equity=float(metrics.get("final_equity", INITIAL_BALANCE)),
        max_drawdown=float(metrics.get("max_drawdown", 0.0)),
        max_drawdown_close=float(metrics.get("max_drawdown_close", 0.0)),
        max_drawdown_worst=float(metrics.get("max_drawdown_worst", 0.0)),
        sharpe_ratio=float(metrics.get("sharpe_ratio", 0.0)),
        sortino_ratio=float(metrics.get("sortino_ratio", 0.0)),
        total_trades=int(metrics.get("total_trades", 0)),
        avg_gross_exposure=float(np.mean(gross_exposure_points)) if gross_exposure_points else 0.0,
        max_gross_exposure=float(np.max(gross_exposure_points)) if gross_exposure_points else 0.0,
    )

    symbol_pnl_rows: List[Dict[str, Any]] = []
    if broker.trades:
        df_trades = pd.DataFrame(
            {
                "symbol": [str(t.symbol) for t in broker.trades],
                "pnl": [float(t.pnl) for t in broker.trades],
                "fee": [float(t.fee) for t in broker.trades],
            }
        )
        grouped = df_trades.groupby("symbol", as_index=False).agg(
            realized_pnl=("pnl", "sum"),
            fees=("fee", "sum"),
        )
        grouped["net_pnl"] = grouped["realized_pnl"] - grouped["fees"]
        grouped["run"] = run.name
        grouped["execution"] = execution.name
        symbol_pnl_rows = grouped.to_dict(orient="records")

    return result, pd.DataFrame(symbol_pnl_rows)


# ---------------------------------------------------------------------------
# Parallel-Worker-Funktion (muss auf Modul-Ebene stehen für multiprocessing)
# ---------------------------------------------------------------------------

def _worker(args: Tuple) -> Tuple[Dict[str, Any], pd.DataFrame]:
    """Wird in einem separaten Prozess ausgeführt."""
    run_dict, exec_dict, bars_by_symbol, funding_schedule, missing_symbols = args

    run = RunConfig(**run_dict)
    execution = ExecutionModel(**exec_dict)

    result, pnl_df = _simulate_portfolio(run, execution, bars_by_symbol, funding_schedule, missing_symbols)
    return result.__dict__, pnl_df


# ---------------------------------------------------------------------------
# Haupt-Funktion
# ---------------------------------------------------------------------------

def main() -> None:
    force_refresh = os.getenv("EDGECRAFT_CACHE_REFRESH", "").strip() in {"1", "true", "yes"}

    penalties = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.08, 0.10]
    runs = [RunConfig(name=f"penalty_{p:.2f}", penalty=p) for p in penalties]
    exec_env = os.getenv("EDGECRAFT_EXECUTIONS", "strict").strip().lower()
    if exec_env == "both":
        executions = [LEGACY_EXECUTION, STRICT_EXECUTION]
    elif exec_env == "legacy":
        executions = [LEGACY_EXECUTION]
    else:
        executions = [STRICT_EXECUTION]

    # --- Daten einmalig laden (Cache) ---
    print(f"Lade OHLCV-Daten ({TIMEFRAME}, Cache: {'erzwinge Refresh' if force_refresh else 'aktiv'}) …")
    bars_by_symbol, funding_schedule, missing_symbols = _load_data(SYMBOLS, force_refresh=force_refresh)
    if not bars_by_symbol:
        raise RuntimeError("Keine OHLCV-Daten gefunden. Prüfe QuestDB und Symbole.")

    print(f"\n{len(bars_by_symbol)} Symbole geladen, {len(missing_symbols)} fehlend.")
    if missing_symbols:
        print(f"Fehlende Symbole: {', '.join(missing_symbols)}")

    # --- Parallel-Runs bestimmen ---
    cpu = os.cpu_count() or 2
    max_workers_env = os.getenv("EDGECRAFT_MAX_WORKERS", "").strip()
    if max_workers_env.isdigit() and int(max_workers_env) > 0:
        max_workers = int(max_workers_env)
    else:
        max_workers = max(2, cpu // 2)

    tasks = [
        (run.__dict__, execution.__dict__, bars_by_symbol, funding_schedule, missing_symbols)
        for run in runs
        for execution in executions
    ]

    print(f"\nStarte {len(tasks)} Run(s) mit max. {max_workers} Worker(n) …\n")

    summary_rows: List[Dict[str, Any]] = []
    pnl_rows: List[pd.DataFrame] = []

    if max_workers <= 1 or len(tasks) == 1:
        # Seriell (kein Mehraufwand für Prozess-Spawning bei wenigen Tasks)
        for task in tasks:
            run_d = task[0]
            exec_d = task[1]
            print(f"  {run_d['name']} | {exec_d['name']} …", end=" ", flush=True)
            res_dict, pnl_df = _worker(task)
            print(f"✓  Return={res_dict['total_return']:.2%}  Trades={res_dict['total_trades']}")
            summary_rows.append(res_dict)
            if not pnl_df.empty:
                pnl_rows.append(pnl_df)
    else:
        import multiprocessing
        mp_ctx = multiprocessing.get_context("spawn")
        with ProcessPoolExecutor(max_workers=max_workers, mp_context=mp_ctx) as executor:
            future_to_name = {}
            for task in tasks:
                run_name = task[0]["name"]
                exec_name = task[1]["name"]
                label = f"{run_name}|{exec_name}"
                future_to_name[executor.submit(_worker, task)] = label

            for future in as_completed(future_to_name):
                label = future_to_name[future]
                try:
                    res_dict, pnl_df = future.result()
                    print(f"  ✓ {label}  Return={res_dict['total_return']:.2%}  Trades={res_dict['total_trades']}")
                    summary_rows.append(res_dict)
                    if not pnl_df.empty:
                        pnl_rows.append(pnl_df)
                except Exception as exc:
                    print(f"  ✗ {label}  FEHLER: {exc}")

    # --- Ergebnisse speichern ---
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_dir = Path("verification")
    out_dir.mkdir(parents=True, exist_ok=True)

    summary_df = pd.DataFrame(summary_rows)
    summary_path = out_dir / f"turnover_penalty_optimization_summary_{ts}.csv"
    summary_df.to_csv(summary_path, index=False)

    pnl_path = out_dir / f"turnover_penalty_optimization_symbol_pnl_{ts}.csv"
    if pnl_rows:
        pd.concat(pnl_rows, ignore_index=True).to_csv(pnl_path, index=False)
    else:
        pd.DataFrame(columns=["run", "execution", "symbol", "realized_pnl", "fees", "net_pnl"]).to_csv(
            pnl_path, index=False
        )

    coverage_path = out_dir / f"turnover_penalty_optimization_coverage_{ts}.csv"
    pd.DataFrame(
        {
            "requested_symbols": SYMBOLS,
            "status": ["missing" if s in missing_symbols else "used" for s in SYMBOLS],
        }
    ).to_csv(coverage_path, index=False)

    print("\n" + summary_df.to_string(index=False))
    print(f"\nGespeichert: {summary_path}")
    print(f"Gespeichert: {pnl_path}")
    print(f"Gespeichert: {coverage_path}")


if __name__ == "__main__":
    main()
