from __future__ import annotations

import contextlib
import io
import math
import os
import sys
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.database import QuestDBManager
from src.simulation.metrics import calculate_metrics
from src.simulation.models import Order, OrderStatus, OrderType, Position, Side, Trade
from src.simulation.strategies.mde_mad_v2 import MDEMADV2Strategy
from src.simulation.strategy import Strategy


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


class ServerLikeMdeMadV2Strategy(Strategy):
    """
    Server container mde_mad_v2 behavior approximation:
    - discrete optimization (coarse + local search)
    - no cooldown/cost gate
    - rebalance threshold = 0.5% of current equity
    - enableShorts default false, maxLeverage default 10
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
        trend_filter_type: str = "SMA",
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

    def _calculate_log_returns(self, prices: List[float]) -> List[float]:
        out: List[float] = []
        for i in range(1, len(prices)):
            prev = prices[i - 1]
            cur = prices[i]
            if prev > 0 and cur > 0:
                out.append(float(math.log(cur / prev)))
        return out

    def _calculate_mad(self, returns: List[float]) -> float:
        if not returns:
            return 0.0
        mean = float(sum(returns) / len(returns))
        return float(sum(abs(x - mean) for x in returns) / len(returns))

    def _ema_series(self, values: List[float], period: int) -> List[float]:
        out = [0.0] * len(values)
        if len(values) < period:
            return out
        k = 2.0 / (period + 1.0)
        out[period - 1] = float(sum(values[:period]) / period)
        for i in range(period, len(values)):
            out[i] = (values[i] * k) + (out[i - 1] * (1.0 - k))
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
        return float(sum(values[-period:]) / period)

    def _wma_series(self, values: List[float], period: int) -> List[float]:
        out = [0.0] * len(values)
        if len(values) < period:
            return out
        wsum = period * (period + 1) / 2
        for i in range(period - 1, len(values)):
            s = 0.0
            for j in range(period):
                s += values[i - period + 1 + j] * (j + 1)
            out[i] = s / wsum
        return out

    def _hma_series(self, values: List[float], period: int) -> List[float]:
        if len(values) < period:
            return [values[-1] if values else 0.0] * len(values)
        half = max(1, period // 2)
        sqrtp = max(1, int(math.sqrt(period)))
        w_half = self._wma_series(values, half)
        w_full = self._wma_series(values, period)
        diff = [(2.0 * w_half[i]) - w_full[i] for i in range(len(values))]
        return self._wma_series(diff, sqrtp)

    def _kama_series(self, values: List[float], period: int) -> List[float]:
        out = [0.0] * len(values)
        if len(values) < period:
            if values:
                out = [values[-1]] * len(values)
            return out
        fast_sc = 2.0 / 3.0
        slow_sc = 2.0 / 31.0
        out[period - 1] = values[period - 1]
        for i in range(period, len(values)):
            change = abs(values[i] - values[i - period])
            volatility = 0.0
            for j in range(i - period + 1, i + 1):
                volatility += abs(values[j] - values[j - 1])
            er = 0.0 if volatility == 0 else change / volatility
            sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
            out[i] = out[i - 1] + sc * (values[i] - out[i - 1])
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

    def _utility(self, weights: List[float], expected_return: float, mad: float) -> List[float]:
        out: List[float] = []
        for w in weights:
            w_abs = abs(w)
            w_cash = max(0.0, 1.0 - w_abs)
            p_asset_raw = max(w_abs, self.EPS)
            p_cash_raw = max(w_cash, self.EPS)
            p_sum = p_asset_raw + p_cash_raw
            p_asset = p_asset_raw / p_sum
            p_cash = p_cash_raw / p_sum
            entropy = -((p_asset * math.log(p_asset)) + (p_cash * math.log(p_cash)))
            turnover = abs(w - self.current_weight)
            out.append(
                (w * expected_return)
                - (self.risk_aversion * w_abs * mad)
                + (self.entropy_weight * entropy)
                - (self.turnover_penalty * turnover)
            )
        return out

    def _optimize_weight(self, returns: List[float], is_uptrend: bool) -> float:
        expected_return = float(sum(returns) / len(returns))
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

        candidates = set()
        steps = max(9, self.optimization_steps)
        step_size = (upper - lower) / max(1, steps - 1)
        for i in range(steps):
            candidates.add(lower + i * step_size)
        candidates.add(min(upper, max(lower, self.current_weight)))

        coarse = sorted(candidates)
        coarse_u = self._utility(coarse, expected_return, mad)
        best_idx = int(np.argmax(np.asarray(coarse_u, dtype=float)))
        best_w = float(coarse[best_idx])

        local_step = step_size
        local_lower = max(lower, best_w - local_step)
        local_upper = min(upper, best_w + local_step)
        if local_upper > local_lower:
            local = [local_lower + ((local_upper - local_lower) * i / 10.0) for i in range(11)]
            local_u = self._utility(local, expected_return, mad)
            best_local_idx = int(np.argmax(np.asarray(local_u, dtype=float)))
            best_w = float(local[best_local_idx])
        return best_w

    def on_bar(self, bar: Dict[str, Any]):
        close = bar.get("close")
        symbol = bar.get("symbol")
        ts = bar.get("timestamp")
        if close is None or symbol is None or ts is None:
            return
        self.history.append(float(close))
        min_required = max(self.lookback_bars, self.trend_filter_period) + 1
        if len(self.history) < min_required:
            return
        if len(self.history) > max(min_required + 120, 500):
            self.history = self.history[-max(min_required + 120, 500):]

        closes = [x for x in self.history if np.isfinite(x) and x > 0]
        if len(closes) < 5:
            return
        trend_value = self._trend_value(closes)
        is_uptrend = closes[-1] >= trend_value
        price_window = closes[-(self.lookback_bars + 1):]
        returns = self._calculate_log_returns(price_window)
        if len(returns) < 5:
            return
        target_weight = self._optimize_weight(returns, is_uptrend)

        if abs(target_weight) < self.entry_weight_threshold:
            # Under threshold, it acts as low conviction. We will still let it rebalance server side logic.
            pass

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

        min_rebalance_threshold = equity * 0.005
        if abs(diff_notional) < min_rebalance_threshold:
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
        pos_margin = 0.0
        for pos in self.positions.values():
            pos_margin += abs(float(pos.size) * float(pos.entry_price)) / max(1, int(pos.leverage))

        order_margin = 0.0
        for order in self.open_orders.values():
            if order.price and float(order.price) > 0:
                remaining_qty = max(0.0, float(order.quantity) - float(order.filled_quantity))
                order_margin += (remaining_qty * float(order.price)) / max(1, int(order.leverage))

        return pos_margin + order_margin

    def get_margin_balance(self) -> float:
        total_unrealized = 0.0
        for pos in self.positions.values():
            total_unrealized += float(pos.unrealized_pnl)
        return self.balance + total_unrealized

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
        created_seq = self._bar_seq_by_symbol.get(order.symbol, 0)
        self._order_created_seq[order.id] = created_seq
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

    def _apply_taker_adjustments(
        self,
        raw_price: float,
        side: str,
        qty: float,
        bar_volume: float,
    ) -> float:
        if self.execution.half_spread_bps <= 0.0 and self.execution.taker_slippage_bps <= 0.0 and self.execution.impact_bps_at_100pct_volume <= 0.0:
            return raw_price

        volume = max(1e-12, float(bar_volume))
        participation = float(qty) / volume
        impact_bps = float(self.execution.impact_bps_at_100pct_volume) * min(1.0, max(0.0, participation))
        total_bps = float(self.execution.half_spread_bps) + float(self.execution.taker_slippage_bps) + impact_bps

        if side == Side.BUY.value:
            return raw_price * (1.0 + total_bps / 10_000.0)
        return raw_price * (1.0 - total_bps / 10_000.0)

    def _passes_margin_check(self, symbol: str, side: str, qty: float, price: float, leverage: int, fee: float) -> bool:
        leverage = max(1, int(leverage))
        pos = self.positions.get(symbol)
        cur_size = float(pos.size) if pos else 0.0
        cur_entry = float(pos.entry_price) if pos else 0.0

        new_size = cur_size
        new_entry = cur_entry
        realized_pnl = 0.0

        if side == Side.BUY.value:
            if cur_size >= 0.0:
                combined = cur_size + qty
                total_cost = (cur_size * cur_entry) + (qty * price)
                new_entry = (total_cost / combined) if combined > 0 else 0.0
                new_size = combined
            else:
                closed_qty = min(abs(cur_size), qty)
                realized_pnl = (cur_entry - price) * closed_qty
                remaining = cur_size + qty
                new_size = remaining
                if abs(remaining) <= 1e-12:
                    new_entry = 0.0
                elif remaining > 0.0:
                    new_entry = price
                else:
                    new_entry = cur_entry
        else:
            if cur_size <= 0.0:
                combined = cur_size - qty
                total_cost = (abs(cur_size) * cur_entry) + (qty * price)
                new_entry = (total_cost / abs(combined)) if abs(combined) > 0 else 0.0
                new_size = combined
            else:
                closed_qty = min(cur_size, qty)
                realized_pnl = (price - cur_entry) * closed_qty
                remaining = cur_size - qty
                new_size = remaining
                if abs(remaining) <= 1e-12:
                    new_entry = 0.0
                elif remaining < 0.0:
                    new_entry = price
                else:
                    new_entry = cur_entry

        # Always allow pure risk-reducing trades so we can unwind risk/liquidate.
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
                size = new_size
                entry = new_entry
                lev = leverage
            used_margin_after += abs(size * entry) / lev
            mark = price if sym == symbol else self.last_prices.get(sym, entry)
            unrealized_after += (mark - entry) * size

        if symbol not in self.positions:
            used_margin_after += abs(new_size * new_entry) / leverage
            unrealized_after += (price - new_entry) * new_size

        margin_balance_after = balance_after + unrealized_after
        return margin_balance_after + 1e-9 >= used_margin_after

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
                symbol=order.symbol,
                size=0.0,
                entry_price=0.0,
                leverage=max(1, int(order.leverage)),
            )

        pos = self.positions[order.symbol]
        pos.leverage = max(1, int(order.leverage))
        realized_pnl = 0.0

        if side == Side.BUY.value:
            if float(pos.size) >= 0.0:
                new_size = float(pos.size) + qty
                total_cost = (float(pos.size) * float(pos.entry_price)) + (qty * price)
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
                total_cost = (abs(float(pos.size)) * float(pos.entry_price)) + (qty * price)
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
        maintenance = 0.0
        for pos in self.positions.values():
            maintenance += float(pos.maintenance_margin)
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
                id="LIQUIDATION",
                symbol=sym,
                side=side,
                order_type=OrderType.MARKET,
                quantity=abs(size),
                price=None,
                timestamp=timestamp,
                leverage=max(1, int(pos.leverage)),
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
                            fill_price = open_px
                            is_taker = False
                        elif low_px <= limit_px:
                            fill_price = limit_px
                            is_taker = False
                    elif side == Side.SELL.value:
                        if open_px >= limit_px:
                            fill_price = open_px
                            is_taker = False
                        elif high_px >= limit_px:
                            fill_price = limit_px
                            is_taker = False
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
                max_qty_from_bar = volume * max(0.0, float(self.execution.max_bar_participation))
                max_fill_qty = min(remaining_qty, max_qty_from_bar)

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
                    ((prev_filled * float(order.average_fill_price)) + (max_fill_qty * exec_price)) / new_filled
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


@dataclass
class RunConfig:
    name: str
    kind: str


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


def _make_strategy(kind: str) -> Strategy:
    if kind == "local_default":
        return MDEMADV2Strategy(
            timeframe=TIMEFRAME,
            enable_shorts=True,
            min_leverage=1,
            max_leverage=3,
        )
    if kind == "local_server_params":
        return MDEMADV2Strategy(
            timeframe=TIMEFRAME,
            lookback_bars=30,
            risk_aversion=2.0,
            entropy_weight=0.1,
            turnover_penalty=0.05,
            trend_filter_period=200,
            enable_shorts=False,
            min_leverage=1,
            max_leverage=10,
            cooldown_hours=0.0,
            min_rebalance_weight_delta=0.0,
            min_rebalance_notional=0.0,
            enable_cost_gate=False,
        )
    if kind == "server_like":
        return ServerLikeMdeMadV2Strategy(
            lookback_bars=30,
            trend_filter_period=200,
            risk_aversion=2.0,
            entropy_weight=0.1,
            turnover_penalty=0.05,
            max_leverage=10.0,
            enable_shorts=False,
            optimization_steps=41,
            entry_weight_threshold=0.10,
            trend_filter_type="EMA",
        )
    raise ValueError(kind)


def _normalize_index(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    out = df.copy()
    idx = pd.to_datetime(out.index, utc=True, errors="coerce")
    out.index = idx
    out = out[~out.index.isna()]
    out = out.sort_index()
    return out


def _load_data(symbols: List[str]) -> Tuple[Dict[str, pd.DataFrame], Dict[pd.Timestamp, List[Tuple[str, float]]], List[str]]:
    db = QuestDBManager()
    bars_by_symbol: Dict[str, pd.DataFrame] = {}
    funding_schedule: Dict[pd.Timestamp, List[Tuple[str, float]]] = {}
    missing: List[str] = []

    for sym in symbols:
        bars = db.get_ohlcv(
            sym,
            START_DATE,
            END_DATE,
            timeframe=TIMEFRAME,
            allow_trade_backfill=False,
        )
        bars = _normalize_index(bars)
        if bars.empty:
            missing.append(sym)
            continue

        cols = ["open", "high", "low", "close", "volume"]
        for col in cols:
            bars[col] = pd.to_numeric(bars[col], errors="coerce")
        bars = bars.dropna(subset=cols)
        if bars.empty:
            missing.append(sym)
            continue

        bars_by_symbol[sym] = bars[cols]

        funding = db.get_funding_rates(sym, START_DATE, END_DATE)
        funding = _normalize_index(funding)
        if not funding.empty and "funding_rate" in funding.columns:
            rates = pd.to_numeric(funding["funding_rate"], errors="coerce").dropna()
            for ts, rate in rates.items():
                funding_schedule.setdefault(ts, []).append((sym, float(rate)))

    return bars_by_symbol, funding_schedule, missing


def _build_events(bars_by_symbol: Dict[str, pd.DataFrame]) -> Dict[pd.Timestamp, List[Dict[str, Any]]]:
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
    total = 0.0
    for sym, pos in broker.positions.items():
        px = broker.last_prices.get(sym, float(pos.entry_price))
        total += abs(float(pos.size) * float(px))
    return total


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
            strat = _make_strategy(run.kind)
            strat.set_broker(broker)
            strat.on_start()
            strategies[sym] = strat

        for ts in timestamps:
            events = events_by_ts[ts]

            for bar in events:
                sym = str(bar["symbol"])
                fills = broker.process_bar(bar)
                for tr in fills:
                    tr_sym = str(tr.symbol)
                    strat = strategies.get(tr_sym)
                    if strat is not None:
                        strat.on_fill(tr)

                strat = strategies.get(sym)
                if strat is not None:
                    strat.on_bar(bar)

            if run.kind == "server_like":
                total_exposure = 0.0
                for strat in strategies.values():
                    if getattr(strat, "latest_target_weight", None) is not None:
                        total_exposure += abs(strat.latest_target_weight)
                
                scale = 1.0
                if total_exposure > 3.0:
                    scale = 3.0 / total_exposure
                
                for sym, strat in strategies.items():
                    if getattr(strat, "latest_target_weight", None) is not None:
                        final_target = strat.latest_target_weight * scale
                        strat._rebalance_server(sym, final_target, strat.latest_close, strat.latest_ts)
                        
                        if abs(final_target) >= strat.entry_weight_threshold:
                            strat.current_weight = float(final_target)
                        
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


def main() -> None:
    all_runs = [
        RunConfig(name="local_mde_mad_v2_default", kind="local_default"),
        RunConfig(name="local_mde_mad_v2_server_params", kind="local_server_params"),
        RunConfig(name="server_like_mde_mad_v2", kind="server_like"),
    ]
    run_by_name = {r.name: r for r in all_runs}

    all_executions = [LEGACY_EXECUTION, STRICT_EXECUTION]
    exec_by_name = {e.name: e for e in all_executions}

    selected_runs_raw = os.getenv("EDGECRAFT_COMPARE_RUNS", "").strip()
    if selected_runs_raw:
        runs = [run_by_name[name.strip()] for name in selected_runs_raw.split(",") if name.strip() in run_by_name]
    else:
        runs = all_runs

    selected_execs_raw = os.getenv("EDGECRAFT_COMPARE_EXECUTIONS", "").strip()
    if selected_execs_raw:
        executions = [
            exec_by_name[name.strip()]
            for name in selected_execs_raw.split(",")
            if name.strip() in exec_by_name
        ]
    else:
        executions = all_executions

    if not runs:
        raise RuntimeError("No runs selected. Check EDGECRAFT_COMPARE_RUNS.")
    if not executions:
        raise RuntimeError("No executions selected. Check EDGECRAFT_COMPARE_EXECUTIONS.")

    print("Loading OHLCV/Funding data once for all runs ...")
    bars_by_symbol, funding_schedule, missing_symbols = _load_data(SYMBOLS)
    if not bars_by_symbol:
        raise RuntimeError("No symbols with usable OHLCV data found in requested range.")

    summary_rows: List[Dict[str, Any]] = []
    pnl_rows: List[pd.DataFrame] = []

    for run in runs:
        for execution in executions:
            print(f"Running {run.name} | execution={execution.name} ...")
            res, pnl_df = _simulate_portfolio(
                run=run,
                execution=execution,
                bars_by_symbol=bars_by_symbol,
                funding_schedule=funding_schedule,
                missing_symbols=missing_symbols,
            )
            summary_rows.append(res.__dict__)
            if not pnl_df.empty:
                pnl_rows.append(pnl_df)

    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_dir = Path("verification")
    out_dir.mkdir(parents=True, exist_ok=True)

    summary_df = pd.DataFrame(summary_rows)
    summary_path = out_dir / f"server_vs_local_mde_mad_v2_portfolio_summary_{ts}.csv"
    summary_df.to_csv(summary_path, index=False)

    pnl_path = out_dir / f"server_vs_local_mde_mad_v2_portfolio_symbol_pnl_{ts}.csv"
    if pnl_rows:
        pd.concat(pnl_rows, ignore_index=True).to_csv(pnl_path, index=False)
    else:
        pd.DataFrame(columns=["run", "execution", "symbol", "realized_pnl", "fees", "net_pnl"]).to_csv(pnl_path, index=False)

    coverage_path = out_dir / f"server_vs_local_mde_mad_v2_portfolio_coverage_{ts}.csv"
    pd.DataFrame(
        {
            "requested_symbols": SYMBOLS,
            "status": ["missing" if s in missing_symbols else "used" for s in SYMBOLS],
        }
    ).to_csv(coverage_path, index=False)

    print(summary_df.to_string(index=False))
    print(f"Saved: {summary_path}")
    print(f"Saved: {pnl_path}")
    print(f"Saved: {coverage_path}")


if __name__ == "__main__":
    main()
