from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from src.simulation.models import Order, OrderType, Side, Trade
from src.simulation.strategy import Strategy


class DoubleRSIDivergenceStrategy(Strategy):
    """
    Double RSI Divergence Strategy.

    Short setup:
    - Price forms three highs (H1, H2, H3 with H3 >= H2 or higher),
    - RSI forms three lower highs,
    - Entry on weakness after H3 (bearish confirmation candle).

    Long setup:
    - Price forms three lows (T1, T2, T3 with T3 <= T2 or lower),
    - RSI forms three higher lows,
    - Entry on strength after T3 (bullish confirmation candle).
    """
    NAME: str = "DoubleRSIDivergence"
    DESCRIPTION: str = "Auto-generated description for DoubleRSIDivergence"
    VERSION: str = "1.0.0"
    AUTHOR: str = "EdgeCraft"
    SUPPORTED_TIMEFRAMES: list = ["1h", "4h", "1d"]



    EPS = 1e-9
    TIMEFRAME_PROFILES: Dict[str, Dict[str, Any]] = {
        # Robust fallback profile (best minimum score across 1h/4h/12h/1d search)
        "default": {
            "pivot_lookback": 2,
            "min_pivot_separation_bars": 4,
            "min_pivot_strength_atr": 0.6,
            "min_retrace_between_pivots_pct": 0.0,
            "double_top_bottom_tolerance": 0.003,
            "min_rsi_delta": 0.6,
            "min_price_move_pct": 0.0,
            "stop_buffer_atr_multiplier": 0.2,
            "rr_target": 1.5,
            "risk_per_trade": 0.005,
            "use_structure_break_trigger": False,
            "enable_regime_filter": False,
            "min_adx_for_entry": 14.0,
            "require_ema_trend_alignment": True,
            "max_setup_age_bars": 24,
        },
        # Timeframe-specific tuned profiles
        "1h": {
            "pivot_lookback": 2,
            "min_pivot_separation_bars": 4,
            "min_pivot_strength_atr": 0.0,
            "min_retrace_between_pivots_pct": 0.0,
            "double_top_bottom_tolerance": 0.003,
            "min_rsi_delta": 1.0,
            "min_price_move_pct": 0.0,
            "stop_buffer_atr_multiplier": 0.1,
            "rr_target": 1.5,
            "risk_per_trade": 0.01,
            "use_structure_break_trigger": True,
            "enable_regime_filter": True,
            "min_adx_for_entry": 18.0,
            "require_ema_trend_alignment": True,
            "max_setup_age_bars": 24,
        },
        "4h": {
            "pivot_lookback": 1,
            "min_pivot_separation_bars": 2,
            "min_pivot_strength_atr": 0.6,
            "min_retrace_between_pivots_pct": 0.004,
            "double_top_bottom_tolerance": 0.0015,
            "min_rsi_delta": 0.6,
            "min_price_move_pct": 0.0005,
            "stop_buffer_atr_multiplier": 0.2,
            "rr_target": 1.5,
            "risk_per_trade": 0.005,
            "use_structure_break_trigger": False,
            "enable_regime_filter": False,
            "min_adx_for_entry": 14.0,
            "require_ema_trend_alignment": False,
            "max_setup_age_bars": 24,
        },
        "12h": {
            "pivot_lookback": 1,
            "min_pivot_separation_bars": 4,
            "min_pivot_strength_atr": 0.2,
            "min_retrace_between_pivots_pct": 0.001,
            "double_top_bottom_tolerance": 0.003,
            "min_rsi_delta": 0.6,
            "min_price_move_pct": 0.0005,
            "stop_buffer_atr_multiplier": 0.2,
            "rr_target": 3.0,
            "risk_per_trade": 0.005,
            "use_structure_break_trigger": False,
            "enable_regime_filter": False,
            "min_adx_for_entry": 10.0,
            "require_ema_trend_alignment": True,
            "max_setup_age_bars": 24,
        },
        "1d": {
            "pivot_lookback": 2,
            "min_pivot_separation_bars": 4,
            "min_pivot_strength_atr": 0.6,
            "min_retrace_between_pivots_pct": 0.0,
            "double_top_bottom_tolerance": 0.003,
            "min_rsi_delta": 0.6,
            "min_price_move_pct": 0.0,
            "stop_buffer_atr_multiplier": 0.2,
            "rr_target": 3.0,
            "risk_per_trade": 0.005,
            "use_structure_break_trigger": False,
            "enable_regime_filter": False,
            "min_adx_for_entry": 10.0,
            "require_ema_trend_alignment": True,
            "max_setup_age_bars": 24,
        },
    }


    @classmethod
    def get_param_schema(cls):
        return {}

    def __init__(
        self,
        timeframe: str = "1h",
        risk_per_trade: Optional[float] = None,
        pivot_lookback: Optional[int] = None,
        min_price_move_pct: Optional[float] = None,
        double_top_bottom_tolerance: Optional[float] = None,
        min_rsi_delta: Optional[float] = None,
        min_pivot_separation_bars: Optional[int] = None,
        min_pivot_strength_atr: Optional[float] = None,
        min_retrace_between_pivots_pct: Optional[float] = None,
        stop_buffer_atr_multiplier: Optional[float] = None,
        rr_target: Optional[float] = None,
        cooldown_bars: int = 5,
        min_order_qty: float = 1e-6,
        max_notional_fraction: float = 0.9,
        enable_shorts: bool = True,
        use_structure_break_trigger: Optional[bool] = None,
        enable_regime_filter: Optional[bool] = None,
        min_adx_for_entry: Optional[float] = None,
        require_ema_trend_alignment: Optional[bool] = None,
        max_setup_age_bars: Optional[int] = None,
        min_leverage: int = 1,
        max_leverage: int = 3,
    ):
        super().__init__()
        self.timeframe = timeframe
        profile = self.TIMEFRAME_PROFILES.get(timeframe, self.TIMEFRAME_PROFILES["default"])
        self.risk_per_trade = float(profile["risk_per_trade"] if risk_per_trade is None else risk_per_trade)
        self.pivot_lookback = max(1, int(profile["pivot_lookback"] if pivot_lookback is None else pivot_lookback))
        self.min_price_move_pct = float(
            profile["min_price_move_pct"] if min_price_move_pct is None else min_price_move_pct
        )
        self.double_top_bottom_tolerance = float(
            profile["double_top_bottom_tolerance"]
            if double_top_bottom_tolerance is None
            else double_top_bottom_tolerance
        )
        self.min_rsi_delta = float(profile["min_rsi_delta"] if min_rsi_delta is None else min_rsi_delta)
        self.min_pivot_separation_bars = max(
            1,
            int(profile["min_pivot_separation_bars"] if min_pivot_separation_bars is None else min_pivot_separation_bars),
        )
        self.min_pivot_strength_atr = max(
            0.0,
            float(profile["min_pivot_strength_atr"] if min_pivot_strength_atr is None else min_pivot_strength_atr),
        )
        self.min_retrace_between_pivots_pct = max(
            0.0,
            float(
                profile["min_retrace_between_pivots_pct"]
                if min_retrace_between_pivots_pct is None
                else min_retrace_between_pivots_pct
            ),
        )
        self.stop_buffer_atr_multiplier = float(
            profile["stop_buffer_atr_multiplier"] if stop_buffer_atr_multiplier is None else stop_buffer_atr_multiplier
        )
        self.rr_target = float(profile["rr_target"] if rr_target is None else rr_target)
        self.cooldown_bars = cooldown_bars
        self.min_order_qty = min_order_qty
        self.max_notional_fraction = max_notional_fraction
        self.enable_shorts = enable_shorts
        self.use_structure_break_trigger = bool(
            profile.get("use_structure_break_trigger", False)
            if use_structure_break_trigger is None
            else use_structure_break_trigger
        )
        self.enable_regime_filter = bool(
            profile.get("enable_regime_filter", False)
            if enable_regime_filter is None
            else enable_regime_filter
        )
        self.min_adx_for_entry = max(
            0.0,
            float(profile.get("min_adx_for_entry", 18.0) if min_adx_for_entry is None else min_adx_for_entry),
        )
        self.require_ema_trend_alignment = bool(
            profile.get("require_ema_trend_alignment", True)
            if require_ema_trend_alignment is None
            else require_ema_trend_alignment
        )
        self.max_setup_age_bars = max(
            1,
            int(profile.get("max_setup_age_bars", 24) if max_setup_age_bars is None else max_setup_age_bars),
        )
        self.min_leverage = max(1, int(min_leverage))
        self.max_leverage = max(self.min_leverage, int(max_leverage))

        self.bar_index = 0
        self.last_exit_bar: Dict[str, int] = {}
        self.last_pivot_eval_idx: Dict[str, int] = {}
        self.last_setup_pivot_idx: Dict[str, int] = {}

        self.history_by_symbol: Dict[str, List[Dict[str, Any]]] = {}
        self.pivot_highs: Dict[str, List[Dict[str, Any]]] = {}
        self.pivot_lows: Dict[str, List[Dict[str, Any]]] = {}
        self.setup_state: Dict[str, Dict[str, Any]] = {}
        # {symbol: {'direction', 'sl_price', 'sl_order_id', 'entry_price', 'risk_per_unit'}}
        self.trade_state: Dict[str, Dict[str, Any]] = {}

    def on_start(self):
        print(
            "DoubleRSIDivergenceStrategy started "
            f"(timeframe={self.timeframe}, shorts={self.enable_shorts}, "
            f"lev_range={self.min_leverage}-{self.max_leverage}, "
            f"pivot_sep={self.min_pivot_separation_bars}, "
            f"pivot_strength_atr={self.min_pivot_strength_atr:.2f}, "
            f"structure_trigger={self.use_structure_break_trigger}, "
            f"regime_filter={self.enable_regime_filter})."
        )

    def on_stop(self):
        print("DoubleRSIDivergenceStrategy stopped.")

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

    def _enumish_to_str(self, value: Any) -> str:
        if value is None:
            return ""
        if hasattr(value, "value"):
            return str(getattr(value, "value")).upper()
        raw = str(value)
        if "." in raw:
            raw = raw.split(".")[-1]
        return raw.upper()

    def _iter_open_orders(self) -> List[Any]:
        if not self.broker:
            return []
        orders = getattr(self.broker, "open_orders", {})
        if isinstance(orders, dict):
            return list(orders.values())
        try:
            return list(orders.values())
        except Exception:
            return []

    def _has_pending_entry_order(self, symbol: str) -> bool:
        for order in self._iter_open_orders():
            if getattr(order, "symbol", None) != symbol:
                continue
            if self._enumish_to_str(getattr(order, "order_type", "")) in {"MARKET", "LIMIT"}:
                return True
        return False

    def _find_open_order(self, order_id: Optional[str]) -> Optional[Any]:
        if not order_id:
            return None
        for order in self._iter_open_orders():
            if getattr(order, "id", None) == order_id:
                return order
        return None

    def _clamp_leverage(self, leverage: int) -> int:
        lev = max(1, int(leverage))
        return int(max(self.min_leverage, min(self.max_leverage, lev)))

    def _resolve_entry_leverage(self, symbol: str) -> int:
        # Keep leverage deterministic and conservative for this setup.
        base = int(getattr(self.broker, "leverage", 1) or 1) if self.broker else 1
        return self._clamp_leverage(base)

    def _calculate_position_qty(self, close: float, stop_price: float, leverage: int) -> Optional[float]:
        risk_per_unit = abs(close - stop_price)
        if risk_per_unit <= self.EPS:
            return None

        risk_amount = max(0.0, self.balance * self.risk_per_trade)
        risk_qty = risk_amount / risk_per_unit

        lev = float(self._clamp_leverage(leverage))
        try:
            available = float(self.broker.get_available_balance()) if self.broker else self.balance
        except Exception:
            available = self.balance

        max_notional = max(0.0, available * lev * self.max_notional_fraction)
        max_qty = max_notional / close if close > self.EPS else 0.0
        qty = min(risk_qty, max_qty)
        if qty <= self.min_order_qty:
            return None
        return qty

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
        lev = self._clamp_leverage(leverage if leverage is not None else int(getattr(self.broker, "leverage", 1) or 1))
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

        existing = self._find_open_order(state.get("sl_order_id"))
        replace = False
        if existing is None:
            replace = True
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
                replace = True

        if not replace:
            return

        ts = self._normalize_timestamp(timestamp)
        pos = self.broker.get_position(symbol) if self.broker else None
        lev = self._clamp_leverage(int(getattr(pos, "leverage", getattr(self.broker, "leverage", 1) or 1)))
        order = Order(
            id="",
            symbol=symbol,
            side=desired_side,
            order_type=OrderType.STOP,
            quantity=qty,
            price=None,
            stop_price=sl_price,
            timestamp=ts,
            leverage=lev,
        )
        placed = self.submit_order(order)
        if placed and self._enumish_to_str(getattr(placed, "status", "")) != "REJECTED":
            state["sl_order_id"] = placed.id
            self.trade_state[symbol] = state

    def _cancel_stop_order_if_any(self, symbol: str) -> None:
        state = self.trade_state.get(symbol)
        if not state:
            return
        stop_id = state.get("sl_order_id")
        if stop_id:
            self.cancel_order(stop_id)
        state["sl_order_id"] = None

    def _in_cooldown(self, symbol: str) -> bool:
        last = self.last_exit_bar.get(symbol)
        if last is None:
            return False
        return (self.bar_index - last) < self.cooldown_bars

    def _append_bar(self, symbol: str, bar: Dict[str, Any]) -> None:
        bars = self.history_by_symbol.setdefault(symbol, [])
        bars.append(bar)
        if len(bars) > 5000:
            del bars[: len(bars) - 5000]

    def _add_pivot(self, pivots: List[Dict[str, Any]], pivot: Dict[str, Any], is_high: bool) -> None:
        if pivots and pivots[-1]["idx"] == pivot["idx"]:
            return
        if pivots and (pivot["idx"] - pivots[-1]["idx"]) < self.min_pivot_separation_bars:
            prev = pivots[-1]
            prev_price = float(prev["price"])
            new_price = float(pivot["price"])
            prev_strength = float(prev.get("strength", 0.0))
            new_strength = float(pivot.get("strength", 0.0))
            better_price = new_price > prev_price if is_high else new_price < prev_price
            if better_price or new_strength > prev_strength:
                pivots[-1] = pivot
            return
        pivots.append(pivot)
        if len(pivots) > 20:
            del pivots[: len(pivots) - 20]

    def _lowest_pivot_between(
        self,
        pivots: List[Dict[str, Any]],
        left_idx: int,
        right_idx: int,
    ) -> Optional[Dict[str, Any]]:
        candidates = [p for p in pivots if left_idx < p["idx"] < right_idx]
        if not candidates:
            return None
        return min(candidates, key=lambda p: p["price"])

    def _highest_pivot_between(
        self,
        pivots: List[Dict[str, Any]],
        left_idx: int,
        right_idx: int,
    ) -> Optional[Dict[str, Any]]:
        candidates = [p for p in pivots if left_idx < p["idx"] < right_idx]
        if not candidates:
            return None
        return max(candidates, key=lambda p: p["price"])

    def _bars_between(self, symbol: str, left_idx: int, right_idx: int) -> List[Dict[str, Any]]:
        bars = self.history_by_symbol.get(symbol, [])
        if not bars:
            return []
        left = max(0, min(len(bars), left_idx + 1))
        right = max(left, min(len(bars), right_idx))
        return bars[left:right]

    def _structure_trigger_price(self, symbol: str, side: str, left_idx: int, right_idx: int) -> Optional[float]:
        if side == "short":
            lows = self.pivot_lows.get(symbol, [])
            low_pivot = self._lowest_pivot_between(lows, left_idx, right_idx)
            if low_pivot is not None:
                return float(low_pivot["price"])
            segment = self._bars_between(symbol, left_idx, right_idx)
            if segment:
                return float(min(b["low"] for b in segment))
            return None

        highs = self.pivot_highs.get(symbol, [])
        high_pivot = self._highest_pivot_between(highs, left_idx, right_idx)
        if high_pivot is not None:
            return float(high_pivot["price"])
        segment = self._bars_between(symbol, left_idx, right_idx)
        if segment:
            return float(max(b["high"] for b in segment))
        return None

    def _passes_regime_filter(self, symbol: str, direction: str) -> bool:
        if not self.enable_regime_filter:
            return True

        bars = self.history_by_symbol.get(symbol, [])
        if not bars:
            return False
        current = bars[-1]
        adx = self._as_float(current.get("adx14"))
        if adx is None or adx < self.min_adx_for_entry:
            return False

        if not self.require_ema_trend_alignment:
            return True

        close = self._as_float(current.get("close"))
        ema = self._as_float(current.get("ema50"))
        prev_ema = self._as_float(bars[-2].get("ema50")) if len(bars) >= 2 else None
        if close is None or ema is None:
            return False

        if direction == "long":
            if close < ema:
                return False
            if prev_ema is not None and ema < prev_ema:
                return False
            return True

        if close > ema:
            return False
        if prev_ema is not None and ema > prev_ema:
            return False
        return True

    def _expire_stale_setup(self, symbol: str) -> None:
        setup = self.setup_state.get(symbol)
        if not setup:
            return
        armed_at_bar = int(setup.get("armed_at_bar", self.bar_index))
        if (self.bar_index - armed_at_bar) > self.max_setup_age_bars:
            self.setup_state.pop(symbol, None)

    def _update_pivots(self, symbol: str) -> None:
        bars = self.history_by_symbol.get(symbol, [])
        lb = self.pivot_lookback
        if len(bars) < (2 * lb + 1):
            return

        candidate_idx = len(bars) - lb - 1
        if self.last_pivot_eval_idx.get(symbol) == candidate_idx:
            return
        self.last_pivot_eval_idx[symbol] = candidate_idx
        if candidate_idx < lb:
            return

        window = bars[candidate_idx - lb : candidate_idx + lb + 1]
        candidate = bars[candidate_idx]
        highs = [b["high"] for b in window]
        lows = [b["low"] for b in window]

        is_pivot_high = candidate["high"] >= max(highs) and any(candidate["high"] > h for i, h in enumerate(highs) if i != lb)
        is_pivot_low = candidate["low"] <= min(lows) and any(candidate["low"] < l for i, l in enumerate(lows) if i != lb)

        rsi = self._as_float(candidate.get("rsi14"))
        if rsi is None:
            return
        atr = self._as_float(candidate.get("atr14"))
        if atr is None or atr <= self.EPS:
            atr = max(float(candidate["close"]) * 0.001, self.EPS)
        high_strength = float(candidate["high"] - min(lows))
        low_strength = float(max(highs) - candidate["low"])
        high_threshold = max(float(candidate["high"]) * self.min_price_move_pct, atr * self.min_pivot_strength_atr)
        low_threshold = max(float(candidate["low"]) * self.min_price_move_pct, atr * self.min_pivot_strength_atr)

        if is_pivot_high and high_strength >= high_threshold:
            self._add_pivot(
                self.pivot_highs.setdefault(symbol, []),
                {
                    "idx": candidate_idx,
                    "price": float(candidate["high"]),
                    "rsi": rsi,
                    "strength": high_strength,
                    "timestamp": candidate["timestamp"],
                },
                is_high=True,
            )
        if is_pivot_low and low_strength >= low_threshold:
            self._add_pivot(
                self.pivot_lows.setdefault(symbol, []),
                {
                    "idx": candidate_idx,
                    "price": float(candidate["low"]),
                    "rsi": rsi,
                    "strength": low_strength,
                    "timestamp": candidate["timestamp"],
                },
                is_high=False,
            )

    def _is_bearish_double_divergence(self, symbol: str, h1: Dict[str, Any], h2: Dict[str, Any], h3: Dict[str, Any]) -> bool:
        higher_h2 = h2["price"] > h1["price"] * (1 + self.min_price_move_pct)
        h3_higher_or_equal = h3["price"] >= h2["price"] * (1 - self.double_top_bottom_tolerance)
        rsi_falling = (h1["rsi"] - h2["rsi"]) >= self.min_rsi_delta and (h2["rsi"] - h3["rsi"]) >= self.min_rsi_delta
        spacing_ok = (
            (h2["idx"] - h1["idx"]) >= self.min_pivot_separation_bars
            and (h3["idx"] - h2["idx"]) >= self.min_pivot_separation_bars
        )
        lows = self.pivot_lows.get(symbol, [])
        low_12 = self._lowest_pivot_between(lows, h1["idx"], h2["idx"])
        low_23 = self._lowest_pivot_between(lows, h2["idx"], h3["idx"])
        if low_12 is None or low_23 is None:
            return False
        retrace_12 = (h1["price"] - low_12["price"]) / max(h1["price"], self.EPS)
        retrace_23 = (h2["price"] - low_23["price"]) / max(h2["price"], self.EPS)
        retracement_ok = (
            retrace_12 >= self.min_retrace_between_pivots_pct
            and retrace_23 >= self.min_retrace_between_pivots_pct
        )
        return higher_h2 and h3_higher_or_equal and rsi_falling and spacing_ok and retracement_ok

    def _is_bullish_double_divergence(self, symbol: str, t1: Dict[str, Any], t2: Dict[str, Any], t3: Dict[str, Any]) -> bool:
        lower_t2 = t2["price"] < t1["price"] * (1 - self.min_price_move_pct)
        t3_lower_or_equal = t3["price"] <= t2["price"] * (1 + self.double_top_bottom_tolerance)
        rsi_rising = (t2["rsi"] - t1["rsi"]) >= self.min_rsi_delta and (t3["rsi"] - t2["rsi"]) >= self.min_rsi_delta
        spacing_ok = (
            (t2["idx"] - t1["idx"]) >= self.min_pivot_separation_bars
            and (t3["idx"] - t2["idx"]) >= self.min_pivot_separation_bars
        )
        highs = self.pivot_highs.get(symbol, [])
        high_12 = self._highest_pivot_between(highs, t1["idx"], t2["idx"])
        high_23 = self._highest_pivot_between(highs, t2["idx"], t3["idx"])
        if high_12 is None or high_23 is None:
            return False
        rebound_12 = (high_12["price"] - t1["price"]) / max(t1["price"], self.EPS)
        rebound_23 = (high_23["price"] - t2["price"]) / max(t2["price"], self.EPS)
        retracement_ok = (
            rebound_12 >= self.min_retrace_between_pivots_pct
            and rebound_23 >= self.min_retrace_between_pivots_pct
        )
        return lower_t2 and t3_lower_or_equal and rsi_rising and spacing_ok and retracement_ok

    def _refresh_setup_state(self, symbol: str) -> None:
        if symbol in self.setup_state:
            return

        highs = self.pivot_highs.get(symbol, [])
        lows = self.pivot_lows.get(symbol, [])
        last_setup_idx = self.last_setup_pivot_idx.get(symbol, -1)

        if self.enable_shorts and len(highs) >= 3:
            h1, h2, h3 = highs[-3], highs[-2], highs[-1]
            if h3["idx"] > last_setup_idx and self._is_bearish_double_divergence(symbol, h1, h2, h3):
                trigger_price = self._structure_trigger_price(symbol, "short", h2["idx"], h3["idx"])
                self.setup_state[symbol] = {
                    "side": "short",
                    "pivot_idx": h3["idx"],
                    "pivot_price": h3["price"],
                    "trigger_price": trigger_price,
                    "armed_at_bar": self.bar_index,
                }
                self.last_setup_pivot_idx[symbol] = h3["idx"]
                return

        if len(lows) >= 3:
            t1, t2, t3 = lows[-3], lows[-2], lows[-1]
            if t3["idx"] > last_setup_idx and self._is_bullish_double_divergence(symbol, t1, t2, t3):
                trigger_price = self._structure_trigger_price(symbol, "long", t2["idx"], t3["idx"])
                self.setup_state[symbol] = {
                    "side": "long",
                    "pivot_idx": t3["idx"],
                    "pivot_price": t3["price"],
                    "trigger_price": trigger_price,
                    "armed_at_bar": self.bar_index,
                }
                self.last_setup_pivot_idx[symbol] = t3["idx"]

    def _setup_triggered(self, symbol: str) -> bool:
        bars = self.history_by_symbol.get(symbol, [])
        setup = self.setup_state.get(symbol)
        if not setup or len(bars) < 2:
            return False

        current = bars[-1]
        prev = bars[-2]
        trigger_price = self._as_float(setup.get("trigger_price"))

        if self.use_structure_break_trigger and trigger_price is not None:
            if setup["side"] == "short":
                return (
                    current["close"] < trigger_price
                    and current["low"] < trigger_price
                    and current["close"] < prev["close"]
                )
            return (
                current["close"] > trigger_price
                and current["high"] > trigger_price
                and current["close"] > prev["close"]
            )

        if setup["side"] == "short":
            return (current["close"] < current["open"]) and (current["close"] < prev["close"])
        return (current["close"] > current["open"]) and (current["close"] > prev["close"])

    def _enter_position(self, symbol: str, timestamp: Any) -> None:
        setup = self.setup_state.get(symbol)
        bars = self.history_by_symbol.get(symbol, [])
        if not setup or not bars:
            return

        current = bars[-1]
        close = current["close"]
        atr = current["atr14"]
        if atr <= self.EPS:
            self.setup_state.pop(symbol, None)
            return

        buffer = max(close * 0.0003, atr * self.stop_buffer_atr_multiplier)
        if setup["side"] == "short":
            stop_price = setup["pivot_price"] + buffer
            side = Side.SELL
            direction = "short"
        else:
            stop_price = setup["pivot_price"] - buffer
            side = Side.BUY
            direction = "long"

        leverage = self._resolve_entry_leverage(symbol)
        qty = self._calculate_position_qty(close, stop_price, leverage=leverage)
        if qty is None:
            self.setup_state.pop(symbol, None)
            return

        placed = self._submit_market_order(symbol, side, qty, timestamp, leverage=leverage)
        if not placed:
            self.setup_state.pop(symbol, None)
            return

        self.trade_state[symbol] = {
            "direction": direction,
            "sl_price": stop_price,
            "sl_order_id": None,
            "entry_price": close,
            "risk_per_unit": abs(close - stop_price),
        }
        self.setup_state.pop(symbol, None)

    def _manage_open_position(self, symbol: str, timestamp: Any) -> None:
        size = self.get_position_size(symbol)
        if abs(size) <= self.EPS:
            return
        bars = self.history_by_symbol.get(symbol, [])
        if not bars:
            return
        current = bars[-1]
        close = current["close"]
        state = self.trade_state.get(symbol, {})

        entry_price = self._as_float(state.get("entry_price"))
        risk_per_unit = self._as_float(state.get("risk_per_unit"))
        if entry_price is None or risk_per_unit is None or risk_per_unit <= self.EPS:
            self._sync_stop_order(symbol, timestamp)
            return

        target_price = (
            entry_price + self.rr_target * risk_per_unit
            if size > 0
            else entry_price - self.rr_target * risk_per_unit
        )

        if size > 0 and close >= target_price and not self._has_pending_entry_order(symbol):
            self._submit_market_order(symbol, Side.SELL, abs(size), timestamp)
            return
        if size < 0 and close <= target_price and not self._has_pending_entry_order(symbol):
            self._submit_market_order(symbol, Side.BUY, abs(size), timestamp)
            return

        self._sync_stop_order(symbol, timestamp)

    def on_bar(self, bar: Dict[str, Any]):
        self.bar_index += 1

        symbol = bar.get("symbol")
        timestamp = bar.get("timestamp")
        if not symbol or not timestamp:
            return

        open_price = self._as_float(bar.get("open"))
        high = self._as_float(bar.get("high"))
        low = self._as_float(bar.get("low"))
        close = self._as_float(bar.get("close"))
        rsi = self._as_float(bar.get("rsi14"))
        atr = self._as_float(bar.get("atr14"))
        adx = self._as_float(bar.get("adx14"))
        ema50 = self._as_float(bar.get("ema50"))
        if any(v is None for v in (open_price, high, low, close)):
            return

        # Fallback for non-enriched payloads
        if rsi is None:
            bars = self.history_by_symbol.get(symbol, [])
            closes = [b["close"] for b in bars[-50:]] + [close]
            if len(closes) >= 15:
                s = pd.Series(closes)
                delta = s.diff()
                gain = delta.where(delta > 0, 0.0).fillna(0.0)
                loss = (-delta.where(delta < 0, 0.0)).fillna(0.0)
                avg_gain = gain.ewm(alpha=1 / 14, min_periods=14, adjust=False).mean()
                avg_loss = loss.ewm(alpha=1 / 14, min_periods=14, adjust=False).mean()
                rs = avg_gain / avg_loss.replace(0, np.nan)
                rsi = float((100 - (100 / (1 + rs))).iloc[-1])
        if atr is None:
            atr = max(close * 0.001, self.EPS)
        if rsi is None:
            return

        normalized = {
            "symbol": symbol,
            "timestamp": self._normalize_timestamp(timestamp),
            "open": open_price,
            "high": high,
            "low": low,
            "close": close,
            "rsi14": rsi,
            "atr14": atr,
            "adx14": adx,
            "ema50": ema50,
        }
        self._append_bar(symbol, normalized)

        # Housekeeping on flat state
        pos = self.get_position_size(symbol)
        if abs(pos) <= self.EPS and symbol in self.trade_state:
            self._cancel_stop_order_if_any(symbol)
            self.trade_state.pop(symbol, None)

        self._update_pivots(symbol)
        self._refresh_setup_state(symbol)
        self._expire_stale_setup(symbol)

        if abs(pos) <= self.EPS:
            if self._in_cooldown(symbol):
                return
            if self._has_pending_entry_order(symbol):
                return
            if self._setup_triggered(symbol):
                setup = self.setup_state.get(symbol)
                if not setup:
                    return
                if not self._passes_regime_filter(symbol, setup["side"]):
                    return
                self._enter_position(symbol, normalized["timestamp"])
            return

        self._manage_open_position(symbol, normalized["timestamp"])

    def on_fill(self, trade: Trade):
        symbol = trade.symbol
        pos_size = self.get_position_size(symbol)

        if abs(pos_size) <= self.EPS:
            self._cancel_stop_order_if_any(symbol)
            self.trade_state.pop(symbol, None)
            self.setup_state.pop(symbol, None)
            self.last_exit_bar[symbol] = self.bar_index
            return

        direction = "long" if pos_size > 0 else "short"
        state = self.trade_state.get(symbol)
        if not state:
            default_risk = max(trade.price * 0.002, self.EPS)
            state = {
                "direction": direction,
                "sl_price": trade.price - default_risk if direction == "long" else trade.price + default_risk,
                "sl_order_id": None,
                "entry_price": trade.price,
                "risk_per_unit": default_risk,
            }
        else:
            state["direction"] = direction

        self.trade_state[symbol] = state
        self._sync_stop_order(symbol, trade.timestamp)
