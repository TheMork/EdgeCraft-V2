from src.simulation.strategy import Strategy
from src.simulation.models import OrderType, Side, Trade, Order
from src.simulation.indicators import (
    calculate_sma,
    calculate_ema,
    calculate_rsi,
    calculate_atr,
    calculate_rolling_vwap,
    calculate_adx,
)
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional
from datetime import datetime, timezone


class QuantitativeMomentumStrategy(Strategy):
    """
    Quantitative Momentum strategy with:
    - long and short entries,
    - ATR-based risk sizing + notional cap,
    - partial TP + full trend exit,
    - dynamic breakeven/trailing stop management,
    - cooldown to reduce overtrading.
    """

    EPS = 1e-9

    def __init__(
        self,
        timeframe: str = "1h",
        risk_per_trade: float = 0.01,
        sl_atr_multiplier: float = 2.5,
        trailing_sl_atr_multiplier: float = 1.5,
        partial_exit_ratio: float = 0.5,
        max_notional_fraction: float = 0.9,
        min_order_qty: float = 1e-6,
        adx_threshold: float = 18.0,
        cooldown_bars: int = 5,
        breakeven_r_multiple: float = 1.0,
        enable_shorts: bool = True,
        min_leverage: int = 1,
        max_leverage: int = 3,
    ):
        super().__init__()
        self.timeframe = timeframe
        self.risk_per_trade = risk_per_trade
        self.sl_atr_multiplier = sl_atr_multiplier
        self.trailing_sl_atr_multiplier = trailing_sl_atr_multiplier
        self.partial_exit_ratio = partial_exit_ratio
        self.max_notional_fraction = max_notional_fraction
        self.min_order_qty = min_order_qty
        self.adx_threshold = adx_threshold
        self.cooldown_bars = cooldown_bars
        self.breakeven_r_multiple = breakeven_r_multiple
        self.enable_shorts = enable_shorts
        self.min_leverage = int(max(1, min_leverage))
        self.max_leverage = int(max(self.min_leverage, max_leverage))

        self.history = []
        self.bar_index = 0
        self.last_exit_bar: Dict[str, int] = {}
        # {symbol: {'direction', 'sl_price', 'sl_order_id', 'partial_exit_done', 'entry_price', 'entry_atr'}}
        self.trade_state: Dict[str, Dict[str, Any]] = {}

    def on_start(self):
        print(
            "QuantitativeMomentumStrategy started "
            f"(timeframe={self.timeframe}, shorts={self.enable_shorts}, "
            f"lev_range={self.min_leverage}-{self.max_leverage})."
        )

    def on_stop(self):
        print("QuantitativeMomentumStrategy stopped.")

    def _as_float(self, value: Any) -> Optional[float]:
        if value is None:
            return None
        try:
            parsed = float(value)
        except (TypeError, ValueError):
            return None
        if not np.isfinite(parsed):
            return None
        return parsed

    def _timeframe_to_minutes(self, timeframe: str) -> Optional[int]:
        raw = (timeframe or "").strip()
        if not raw:
            return None
        try:
            if raw.endswith("m") and raw[:-1].isdigit():
                return int(raw[:-1])
            if raw.endswith("h") and raw[:-1].isdigit():
                return int(raw[:-1]) * 60
            if raw.endswith("d") and raw[:-1].isdigit():
                return int(raw[:-1]) * 60 * 24
            if raw.endswith("w") and raw[:-1].isdigit():
                return int(raw[:-1]) * 60 * 24 * 7
            if raw.endswith("M") and raw[:-1].isdigit():
                return int(raw[:-1]) * 60 * 24 * 30
        except ValueError:
            return None
        return None

    def _use_daily_regime(self) -> bool:
        minutes = self._timeframe_to_minutes(self.timeframe)
        if minutes is None:
            return False
        return minutes >= 24 * 60

    def _can_use_fast_path(self, bar: Dict[str, Any]) -> bool:
        required = (
            "close",
            "volume",
            "rsi14",
            "rsi14_prev",
            "ema50",
            "ema20",
            "vwap50",
            "atr14",
            "adx14",
            "vol_sma20",
            "has_correction_long_prev10",
            "has_correction_short_prev10",
            "regime_filter_4h",
            "regime_filter_daily",
            "ready_4h",
            "ready_daily",
        )
        return all(key in bar for key in required)

    def _iter_open_orders(self):
        if not self.broker:
            return []
        orders = getattr(self.broker, "open_orders", {})
        if isinstance(orders, dict):
            return list(orders.values())
        try:
            return list(orders.values())
        except Exception:
            return []

    def _enumish_to_str(self, value: Any) -> str:
        if value is None:
            return ""
        if hasattr(value, "value"):
            return str(getattr(value, "value")).upper()
        raw = str(value)
        if "." in raw:
            raw = raw.split(".")[-1]
        return raw.upper()

    def _find_open_order(self, order_id: Optional[str]):
        if not order_id:
            return None
        for order in self._iter_open_orders():
            if getattr(order, "id", None) == order_id:
                return order
        return None

    def _has_pending_market_or_limit_order(self, symbol: str) -> bool:
        for order in self._iter_open_orders():
            if getattr(order, "symbol", None) != symbol:
                continue
            order_type = self._enumish_to_str(getattr(order, "order_type", ""))
            if order_type in {"MARKET", "LIMIT"}:
                return True
        return False

    def _submit_market_order(
        self,
        symbol: str,
        side: Side,
        quantity: float,
        timestamp: Any,
        leverage: Optional[int] = None,
    ) -> Optional[Order]:
        if quantity <= self.min_order_qty:
            return None
        ts = self._normalize_timestamp(timestamp)
        lev = int(leverage if leverage is not None else getattr(self.broker, "leverage", 1) or 1)
        lev = max(1, lev)
        order = Order(
            id="",
            symbol=symbol,
            side=side,
            order_type=OrderType.MARKET,
            quantity=float(quantity),
            price=None,
            timestamp=ts,
            leverage=lev,
        )
        placed = self.submit_order(order)
        if not placed:
            return None
        if self._enumish_to_str(getattr(placed, "status", "")) == "REJECTED":
            return None
        return placed

    def _clamp_leverage(self, leverage: int) -> int:
        return int(max(self.min_leverage, min(self.max_leverage, int(max(1, leverage)))))

    def _resolve_entry_leverage(self, adx: float) -> int:
        """
        Maps ADX strength into [min_leverage, max_leverage].
        """
        lo = float(self.adx_threshold)
        hi = max(lo + 1.0, 35.0)
        if adx <= lo:
            return self.min_leverage
        if adx >= hi:
            return self.max_leverage
        ratio = (adx - lo) / (hi - lo)
        lev = self.min_leverage + ratio * (self.max_leverage - self.min_leverage)
        return self._clamp_leverage(int(round(lev)))

    def _calculate_position_qty(self, close: float, sl_price: float, leverage: int) -> Optional[float]:
        risk_per_unit = abs(close - sl_price)
        if risk_per_unit <= self.EPS:
            return None

        risk_amount = max(0.0, self.balance * self.risk_per_trade)
        risk_qty = risk_amount / risk_per_unit

        lev = float(self._clamp_leverage(leverage))
        try:
            available_balance = float(self.broker.get_available_balance()) if self.broker else self.balance
        except Exception:
            available_balance = self.balance
        max_notional = max(0.0, available_balance * lev * self.max_notional_fraction)
        max_qty = max_notional / close if close > self.EPS else 0.0

        qty = min(risk_qty, max_qty)
        if qty <= self.min_order_qty:
            return None
        return qty

    def _in_cooldown(self, symbol: str) -> bool:
        last_exit = self.last_exit_bar.get(symbol)
        if last_exit is None:
            return False
        return (self.bar_index - last_exit) < self.cooldown_bars

    def _cancel_stop_order_if_any(self, symbol: str) -> None:
        state = self.trade_state.get(symbol)
        if not state:
            return
        stop_order_id = state.get("sl_order_id")
        if stop_order_id:
            self.cancel_order(stop_order_id)
        state["sl_order_id"] = None

    def _cleanup_flat_state(self, symbol: str) -> None:
        pos = self.get_position_size(symbol)
        if abs(pos) > self.EPS:
            return
        if symbol in self.trade_state:
            self._cancel_stop_order_if_any(symbol)
            self.trade_state.pop(symbol, None)

    def _sync_stop_order(self, symbol: str, timestamp: Any) -> None:
        pos_size = self.get_position_size(symbol)
        if abs(pos_size) <= self.EPS:
            return

        state = self.trade_state.get(symbol)
        if not state:
            return

        sl_price = self._as_float(state.get("sl_price"))
        if sl_price is None:
            return

        direction = state.get("direction")
        desired_side = Side.SELL if direction == "long" else Side.BUY
        qty = abs(pos_size)
        order_id = state.get("sl_order_id")
        existing = self._find_open_order(order_id)

        needs_replace = False
        if existing is None:
            needs_replace = True
        else:
            existing_qty = self._as_float(getattr(existing, "quantity", None))
            existing_stop = self._as_float(getattr(existing, "stop_price", None))
            existing_side = self._enumish_to_str(getattr(existing, "side", ""))
            if (
                existing_qty is None
                or abs(existing_qty - qty) > self.EPS
                or existing_stop is None
                or abs(existing_stop - sl_price) > self.EPS
                or existing_side != self._enumish_to_str(desired_side)
            ):
                self.cancel_order(getattr(existing, "id", ""))
                needs_replace = True

        if not needs_replace:
            return

        ts = self._normalize_timestamp(timestamp)
        pos = self.broker.get_position(symbol) if self.broker else None
        pos_lev = self._clamp_leverage(int(getattr(pos, "leverage", getattr(self.broker, "leverage", 1) or 1)))
        stop_order = Order(
            id="",
            symbol=symbol,
            side=desired_side,
            order_type=OrderType.STOP,
            quantity=qty,
            price=None,
            stop_price=sl_price,
            timestamp=ts,
            leverage=pos_lev,
        )
        placed = self.submit_order(stop_order)
        if placed and self._enumish_to_str(getattr(placed, "status", "")) != "REJECTED":
            state["sl_order_id"] = placed.id
            self.trade_state[symbol] = state

    def _open_long(self, symbol: str, timestamp: Any, close: float, atr: float, leverage: int) -> None:
        sl_price = close - (self.sl_atr_multiplier * atr)
        qty = self._calculate_position_qty(close, sl_price, leverage=leverage)
        if qty is None:
            return

        self.trade_state[symbol] = {
            "direction": "long",
            "sl_price": sl_price,
            "sl_order_id": None,
            "partial_exit_done": False,
            "entry_price": close,
            "entry_atr": atr,
        }
        placed = self._submit_market_order(symbol, Side.BUY, qty, timestamp, leverage=leverage)
        if not placed:
            self.trade_state.pop(symbol, None)

    def _open_short(self, symbol: str, timestamp: Any, close: float, atr: float, leverage: int) -> None:
        sl_price = close + (self.sl_atr_multiplier * atr)
        qty = self._calculate_position_qty(close, sl_price, leverage=leverage)
        if qty is None:
            return

        self.trade_state[symbol] = {
            "direction": "short",
            "sl_price": sl_price,
            "sl_order_id": None,
            "partial_exit_done": False,
            "entry_price": close,
            "entry_atr": atr,
        }
        placed = self._submit_market_order(symbol, Side.SELL, qty, timestamp, leverage=leverage)
        if not placed:
            self.trade_state.pop(symbol, None)

    def _manage_long_position(
        self,
        symbol: str,
        timestamp: Any,
        position_size: float,
        close: float,
        ema20: float,
        regime_filter: float,
        rsi: float,
        atr: float,
    ) -> None:
        state = self.trade_state.get(symbol, {})
        if state.get("direction") != "long":
            state = {
                "direction": "long",
                "sl_price": close - (self.sl_atr_multiplier * atr),
                "sl_order_id": None,
                "partial_exit_done": False,
                "entry_price": close,
                "entry_atr": atr,
            }
            self.trade_state[symbol] = state

        # Full exit first; prevents partial+full double-exit in same bar.
        if close < ema20 or close < regime_filter:
            if not self._has_pending_market_or_limit_order(symbol):
                self._submit_market_order(symbol, Side.SELL, abs(position_size), timestamp)
            return

        if (not state.get("partial_exit_done")) and rsi > 75 and position_size > (2 * self.min_order_qty):
            qty_to_sell = max(self.min_order_qty, abs(position_size) * self.partial_exit_ratio)
            placed = self._submit_market_order(symbol, Side.SELL, qty_to_sell, timestamp)
            if placed:
                state["partial_exit_done"] = True
                self.trade_state[symbol] = state
            return

        current_sl = self._as_float(state.get("sl_price"))
        if current_sl is None:
            current_sl = close - (self.sl_atr_multiplier * atr)

        new_sl = max(current_sl, close - (self.trailing_sl_atr_multiplier * atr))
        entry_price = self._as_float(state.get("entry_price")) or close
        entry_atr = self._as_float(state.get("entry_atr")) or atr
        if entry_atr > self.EPS and (close - entry_price) >= (self.breakeven_r_multiple * entry_atr):
            new_sl = max(new_sl, entry_price)

        if new_sl > current_sl + self.EPS:
            state["sl_price"] = new_sl
            self.trade_state[symbol] = state

        self._sync_stop_order(symbol, timestamp)

    def _manage_short_position(
        self,
        symbol: str,
        timestamp: Any,
        position_size: float,
        close: float,
        ema20: float,
        regime_filter: float,
        rsi: float,
        atr: float,
    ) -> None:
        state = self.trade_state.get(symbol, {})
        if state.get("direction") != "short":
            state = {
                "direction": "short",
                "sl_price": close + (self.sl_atr_multiplier * atr),
                "sl_order_id": None,
                "partial_exit_done": False,
                "entry_price": close,
                "entry_atr": atr,
            }
            self.trade_state[symbol] = state

        # Full exit first; prevents partial+full double-exit in same bar.
        if close > ema20 or close > regime_filter:
            if not self._has_pending_market_or_limit_order(symbol):
                self._submit_market_order(symbol, Side.BUY, abs(position_size), timestamp)
            return

        if (not state.get("partial_exit_done")) and rsi < 25 and abs(position_size) > (2 * self.min_order_qty):
            qty_to_cover = max(self.min_order_qty, abs(position_size) * self.partial_exit_ratio)
            placed = self._submit_market_order(symbol, Side.BUY, qty_to_cover, timestamp)
            if placed:
                state["partial_exit_done"] = True
                self.trade_state[symbol] = state
            return

        current_sl = self._as_float(state.get("sl_price"))
        if current_sl is None:
            current_sl = close + (self.sl_atr_multiplier * atr)

        new_sl = min(current_sl, close + (self.trailing_sl_atr_multiplier * atr))
        entry_price = self._as_float(state.get("entry_price")) or close
        entry_atr = self._as_float(state.get("entry_atr")) or atr
        if entry_atr > self.EPS and (entry_price - close) >= (self.breakeven_r_multiple * entry_atr):
            new_sl = min(new_sl, entry_price)

        if new_sl < current_sl - self.EPS:
            state["sl_price"] = new_sl
            self.trade_state[symbol] = state

        self._sync_stop_order(symbol, timestamp)

    def _on_bar_fast(self, bar: Dict[str, Any], symbol: str, timestamp: Any):
        self._cleanup_flat_state(symbol)

        ready_key = "ready_daily" if self._use_daily_regime() else "ready_4h"
        regime_key = "regime_filter_daily" if self._use_daily_regime() else "regime_filter_4h"
        if not bool(bar.get(ready_key, False)):
            return

        current_close = self._as_float(bar.get("close"))
        current_filter = self._as_float(bar.get(regime_key))
        current_rsi = self._as_float(bar.get("rsi14"))
        prev_rsi = self._as_float(bar.get("rsi14_prev"))
        current_vol = self._as_float(bar.get("volume"))
        current_vol_sma = self._as_float(bar.get("vol_sma20"))
        current_vwap = self._as_float(bar.get("vwap50"))
        current_ema50 = self._as_float(bar.get("ema50"))
        current_ema20 = self._as_float(bar.get("ema20"))
        current_atr = self._as_float(bar.get("atr14"))
        current_adx = self._as_float(bar.get("adx14"))
        has_correction_long = bool(bar.get("has_correction_long_prev10", False))
        has_correction_short = bool(bar.get("has_correction_short_prev10", False))

        if any(
            v is None
            for v in (
                current_close,
                current_filter,
                current_rsi,
                prev_rsi,
                current_vol,
                current_vol_sma,
                current_vwap,
                current_ema50,
                current_ema20,
                current_atr,
                current_adx,
            )
        ):
            return

        if current_atr <= self.EPS:
            return

        position_size = self.get_position_size(symbol)

        if abs(position_size) <= self.EPS:
            if self._in_cooldown(symbol):
                return
            if self._has_pending_market_or_limit_order(symbol):
                return

            adx_confirmed = current_adx >= self.adx_threshold
            vol_confirmed = current_vol > current_vol_sma

            bullish_regime = current_close > current_filter
            trend_long = current_close > current_ema20 > current_ema50
            reclaim_long = (current_close > current_vwap) and (current_close > current_ema50)
            rsi_rising = current_rsi > prev_rsi
            long_signal = (
                bullish_regime
                and trend_long
                and reclaim_long
                and rsi_rising
                and vol_confirmed
                and adx_confirmed
                and has_correction_long
            )

            bearish_regime = current_close < current_filter
            trend_short = current_close < current_ema20 < current_ema50
            reclaim_short = (current_close < current_vwap) and (current_close < current_ema50)
            rsi_falling = current_rsi < prev_rsi
            short_signal = (
                bearish_regime
                and trend_short
                and reclaim_short
                and rsi_falling
                and vol_confirmed
                and adx_confirmed
                and has_correction_short
            )
            entry_leverage = self._resolve_entry_leverage(current_adx)

            if long_signal:
                self._open_long(symbol, timestamp, current_close, current_atr, leverage=entry_leverage)
                return

            if self.enable_shorts and short_signal:
                self._open_short(symbol, timestamp, current_close, current_atr, leverage=entry_leverage)
                return

            return

        if position_size > self.EPS:
            self._manage_long_position(
                symbol=symbol,
                timestamp=timestamp,
                position_size=position_size,
                close=current_close,
                ema20=current_ema20,
                regime_filter=current_filter,
                rsi=current_rsi,
                atr=current_atr,
            )
            return

        if position_size < -self.EPS:
            self._manage_short_position(
                symbol=symbol,
                timestamp=timestamp,
                position_size=position_size,
                close=current_close,
                ema20=current_ema20,
                regime_filter=current_filter,
                rsi=current_rsi,
                atr=current_atr,
            )

    def _on_bar_legacy(self, bar: Dict[str, Any], symbol: str, timestamp: Any):
        # Fallback for non-enriched payloads (e.g., old tests/tools).
        self.history.append(bar)
        if len(self.history) > 3000:
            self.history = self.history[-3000:]
        if len(self.history) < 250:
            return

        df = pd.DataFrame(self.history)
        for col in ["open", "high", "low", "close", "volume"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df = df.dropna(subset=["open", "high", "low", "close", "volume"])
        if len(df) < 250:
            return

        close = df["close"]
        high = df["high"]
        low = df["low"]
        volume = df["volume"]

        rsi = calculate_rsi(close, 14)
        ema50 = calculate_ema(close, 50)
        ema20 = calculate_ema(close, 20)
        vwap50 = calculate_rolling_vwap(high, low, close, volume, window=50)
        atr14 = calculate_atr(high, low, close, 14)
        adx14 = calculate_adx(high, low, close, 14)
        vol_sma20 = calculate_sma(volume, 20)
        rsi_prev = rsi.shift(1)
        has_corr_long = rsi_prev.rolling(window=10, min_periods=1).min() < 50
        has_corr_short = rsi_prev.rolling(window=10, min_periods=1).max() > 50
        regime_4h = calculate_ema(close, 1100)
        regime_daily = calculate_sma(close, 200)

        enriched_bar = dict(bar)
        enriched_bar.update(
            {
                "rsi14": rsi.iloc[-1],
                "rsi14_prev": rsi_prev.iloc[-1],
                "ema50": ema50.iloc[-1],
                "ema20": ema20.iloc[-1],
                "vwap50": vwap50.iloc[-1],
                "atr14": atr14.iloc[-1],
                "adx14": adx14.iloc[-1],
                "vol_sma20": vol_sma20.iloc[-1],
                "has_correction_long_prev10": bool(has_corr_long.iloc[-1]),
                "has_correction_short_prev10": bool(has_corr_short.iloc[-1]),
                "regime_filter_4h": regime_4h.iloc[-1],
                "regime_filter_daily": regime_daily.iloc[-1],
                "ready_4h": len(df) >= 1100,
                "ready_daily": len(df) >= 200,
            }
        )
        self._on_bar_fast(enriched_bar, symbol, timestamp)

    def _normalize_timestamp(self, timestamp: Any) -> datetime:
        if isinstance(timestamp, datetime):
            return timestamp
        if hasattr(timestamp, "to_pydatetime"):
            try:
                return timestamp.to_pydatetime()
            except Exception:
                pass
        if isinstance(timestamp, str):
            parsed = pd.to_datetime(timestamp, utc=True, errors="coerce")
            if pd.notna(parsed):
                try:
                    return parsed.to_pydatetime()
                except Exception:
                    pass
        return datetime.now(timezone.utc)

    def on_bar(self, bar: Dict[str, Any]):
        self.bar_index += 1

        symbol = bar.get("symbol")
        timestamp = bar.get("timestamp")
        if not symbol or not timestamp:
            return

        if self._can_use_fast_path(bar):
            self._on_bar_fast(bar, symbol, timestamp)
            return

        self._on_bar_legacy(bar, symbol, timestamp)

    def on_fill(self, trade: Trade):
        symbol = trade.symbol
        pos_size = self.get_position_size(symbol)

        if abs(pos_size) <= self.EPS:
            self._cancel_stop_order_if_any(symbol)
            self.trade_state.pop(symbol, None)
            self.last_exit_bar[symbol] = self.bar_index
            return

        direction = "long" if pos_size > 0 else "short"
        state = self.trade_state.get(symbol)

        if state and state.get("direction") != direction:
            self._cancel_stop_order_if_any(symbol)
            state = None

        if not state:
            default_sl = (
                trade.price - (self.sl_atr_multiplier * max(trade.price * 0.01, self.EPS))
                if direction == "long"
                else trade.price + (self.sl_atr_multiplier * max(trade.price * 0.01, self.EPS))
            )
            state = {
                "direction": direction,
                "sl_price": default_sl,
                "sl_order_id": None,
                "partial_exit_done": False,
                "entry_price": trade.price,
                "entry_atr": 0.0,
            }
        else:
            state["direction"] = direction
            if state.get("entry_price") is None:
                state["entry_price"] = trade.price
            if state.get("entry_atr") is None:
                state["entry_atr"] = 0.0

        self.trade_state[symbol] = state
        self._sync_stop_order(symbol, trade.timestamp)
