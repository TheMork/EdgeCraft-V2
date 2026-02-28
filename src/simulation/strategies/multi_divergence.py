from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from src.simulation.indicators import calculate_adx, calculate_atr, calculate_ema, calculate_rsi
from src.simulation.models import Order, OrderType, Side, Trade
from src.simulation.strategy import Strategy


class MultiIndicatorDivergenceStrategy(Strategy):
    """
    Multi-indicator divergence strategy with regular + hidden divergence support.

    Core idea:
    - Build divergence signals on multiple indicators.
    - Require a confluence threshold (count and score) for entries.
    - Apply quality filters (trend, ADX, volume, ATR regime).
    """

    EPS = 1e-9
    SUPPORTED_INDICATORS = (
        "rsi",
        "macd_hist",
        "cci",
        "mfi",
        "stoch_k",
        "williams_r",
        "roc",
        "obv",
    )
    DEFAULT_INDICATORS = ("rsi", "macd_hist", "stoch_k", "williams_r", "cci", "mfi")

    def __init__(
        self,
        timeframe: str = "1h",
        indicators: Sequence[str] = DEFAULT_INDICATORS,
        required_bullish: int = 3,
        required_bearish: int = 3,
        required_bullish_score: float = 3.0,
        required_bearish_score: float = 3.0,
        include_regular: bool = True,
        include_hidden: bool = True,
        hidden_score_multiplier: float = 1.15,
        pivot_lookback: int = 3,
        min_pivot_separation_bars: int = 5,
        max_pivot_age_bars: int = 120,
        min_price_move_pct: float = 0.001,
        min_indicator_delta: float = 0.0,
        risk_per_trade: float = 0.01,
        atr_period: int = 14,
        sl_atr_multiplier: float = 2.5,
        cooldown_bars: int = 5,
        min_order_qty: float = 1e-6,
        max_notional_fraction: float = 0.9,
        enable_shorts: bool = True,
        min_leverage: int = 1,
        max_leverage: int = 3,
        require_regime_filter: bool = True,
        regime_fast_ema: int = 50,
        regime_slow_ema: int = 200,
        min_adx_for_entry: float = 10.0,
        require_volume_confirmation: bool = True,
        volume_sma_period: int = 20,
        atr_ratio_floor: float = 0.001,
        atr_ratio_cap: float = 0.2,
        indicator_weights: Optional[Dict[str, float]] = None,
        analysis_window_bars: int = 1200,
    ):
        super().__init__()
        self.timeframe = timeframe
        requested_indicators = [i.strip().lower() for i in indicators if i and i.strip()]
        allowed = set(self.SUPPORTED_INDICATORS)
        normalized_indicators: List[str] = []
        for name in requested_indicators:
            if name in allowed and name not in normalized_indicators:
                normalized_indicators.append(name)
        if not normalized_indicators:
            normalized_indicators = list(self.DEFAULT_INDICATORS)
        self.indicators = tuple(normalized_indicators)

        self.required_bullish = max(1, int(required_bullish))
        self.required_bearish = max(1, int(required_bearish))
        self.required_bullish_score = max(0.1, float(required_bullish_score))
        self.required_bearish_score = max(0.1, float(required_bearish_score))
        self.include_regular = bool(include_regular)
        self.include_hidden = bool(include_hidden)
        self.hidden_score_multiplier = max(1.0, float(hidden_score_multiplier))

        self.pivot_lookback = max(1, int(pivot_lookback))
        self.min_pivot_separation_bars = max(1, int(min_pivot_separation_bars))
        self.max_pivot_age_bars = max(5, int(max_pivot_age_bars))
        self.min_price_move_pct = max(0.0, float(min_price_move_pct))
        self.min_indicator_delta = max(0.0, float(min_indicator_delta))

        self.risk_per_trade = max(0.0, float(risk_per_trade))
        self.atr_period = max(2, int(atr_period))
        self.sl_atr_multiplier = max(0.1, float(sl_atr_multiplier))
        self.cooldown_bars = max(0, int(cooldown_bars))
        self.min_order_qty = float(min_order_qty)
        self.max_notional_fraction = float(max_notional_fraction)
        self.enable_shorts = bool(enable_shorts)
        self.min_leverage = max(1, int(min_leverage))
        self.max_leverage = max(self.min_leverage, int(max_leverage))

        self.require_regime_filter = bool(require_regime_filter)
        self.regime_fast_ema = max(2, int(regime_fast_ema))
        self.regime_slow_ema = max(self.regime_fast_ema + 1, int(regime_slow_ema))
        self.min_adx_for_entry = max(0.0, float(min_adx_for_entry))
        self.require_volume_confirmation = bool(require_volume_confirmation)
        self.volume_sma_period = max(2, int(volume_sma_period))
        self.atr_ratio_floor = max(0.0, float(atr_ratio_floor))
        self.atr_ratio_cap = max(self.atr_ratio_floor + 1e-6, float(atr_ratio_cap))
        self.analysis_window_bars = max(200, int(analysis_window_bars))

        default_weights = {
            "rsi": 1.0,
            "macd_hist": 1.0,
            "cci": 1.0,
            "mfi": 1.0,
            "stoch_k": 1.0,
            "williams_r": 1.0,
            "roc": 1.0,
            "obv": 1.0,
        }
        merged_weights = default_weights.copy()
        if indicator_weights:
            for key, value in indicator_weights.items():
                if key and value is not None:
                    merged_weights[str(key).lower()] = max(0.0, float(value))
        self.indicator_weights = merged_weights

        self.bar_index = 0
        self.last_exit_bar: Dict[str, int] = {}
        self.history_by_symbol: Dict[str, List[Dict[str, Any]]] = {}
        self.df_by_symbol: Dict[str, pd.DataFrame] = {}
        self.trade_state: Dict[str, Dict[str, Any]] = {}

    def on_start(self):
        print(
            "MultiIndicatorDivergenceStrategy started "
            f"(tf={self.timeframe}, indicators={','.join(self.indicators)}, "
            f"bull>={self.required_bullish}/{self.required_bullish_score:.2f}, "
            f"bear>={self.required_bearish}/{self.required_bearish_score:.2f}, "
            f"hidden={self.include_hidden}, regular={self.include_regular}, "
            f"adx>={self.min_adx_for_entry}, regime_filter={self.require_regime_filter})."
        )

    def on_stop(self):
        print("MultiIndicatorDivergenceStrategy stopped.")

    def _as_float(self, value: Any) -> Optional[float]:
        if value is None:
            return None
        try:
            out = float(value)
        except (TypeError, ValueError):
            return None
        if not np.isfinite(out):
            return None
        return out

    def _normalize_timestamp(self, timestamp: Any) -> datetime:
        if isinstance(timestamp, datetime):
            return timestamp if timestamp.tzinfo else timestamp.replace(tzinfo=timezone.utc)
        if hasattr(timestamp, "to_pydatetime"):
            try:
                out = timestamp.to_pydatetime()
                return out if out.tzinfo else out.replace(tzinfo=timezone.utc)
            except Exception:
                pass
        parsed = pd.to_datetime(timestamp, utc=True, errors="coerce")
        if pd.notna(parsed):
            return parsed.to_pydatetime()
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

    def _has_pending_market_or_limit_order(self, symbol: str) -> bool:
        for order in self._iter_open_orders():
            if getattr(order, "symbol", None) != symbol:
                continue
            if self._enumish_to_str(getattr(order, "order_type", "")) in {"MARKET", "LIMIT"}:
                return True
        return False

    def _clamp_leverage(self, leverage: int) -> int:
        lev = max(1, int(leverage))
        return int(max(self.min_leverage, min(self.max_leverage, lev)))

    def _resolve_entry_leverage(self, confirmation_count: int, score: float) -> int:
        count_hi = max(self.required_bullish, self.required_bearish) + 1
        score_hi = max(self.required_bullish_score, self.required_bearish_score) + 1.0
        count_ratio = min(1.0, max(0.0, (confirmation_count - 1) / max(1, count_hi - 1)))
        score_ratio = min(1.0, max(0.0, score / max(self.EPS, score_hi)))
        ratio = 0.5 * count_ratio + 0.5 * score_ratio
        lev = self.min_leverage + ratio * (self.max_leverage - self.min_leverage)
        return self._clamp_leverage(int(round(lev)))

    def _calculate_qty(self, close: float, stop_price: float, leverage: int) -> Optional[float]:
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
        return float(qty)

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
        lev = self._clamp_leverage(leverage if leverage is not None else int(getattr(self.broker, "leverage", 1) or 1))
        order = Order(
            id="",
            symbol=symbol,
            side=side,
            order_type=OrderType.MARKET,
            quantity=float(quantity),
            price=None,
            timestamp=self._normalize_timestamp(timestamp),
            leverage=lev,
        )
        placed = self.submit_order(order)
        if not placed:
            return None
        if self._enumish_to_str(getattr(placed, "status", "")) == "REJECTED":
            return None
        return placed

    def _in_cooldown(self, symbol: str) -> bool:
        last_exit = self.last_exit_bar.get(symbol)
        if last_exit is None:
            return False
        return (self.bar_index - last_exit) < self.cooldown_bars

    def _append_history(self, symbol: str, bar: Dict[str, Any]) -> None:
        if symbol not in self.df_by_symbol:
            self.df_by_symbol[symbol] = pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])
        
        df = self.df_by_symbol[symbol]
        new_row = pd.DataFrame([bar])
        df = pd.concat([df, new_row], ignore_index=True)
        
        # Limit window size to analysis_window_bars + some buffer for lookbacks
        max_window = self.analysis_window_bars + 250
        if len(df) > max_window:
            df = df.iloc[-max_window:].reset_index(drop=True)
        
        self.df_by_symbol[symbol] = df

    def _update_indicators(self, symbol: str) -> Optional[pd.DataFrame]:
        df = self.df_by_symbol.get(symbol)
        if df is None or len(df) < 200:
            return None
            
        # Optimization: Use limited window for calculations.
        # Since we use EWM indicators, we need a bit of history to be accurate.
        
        close = df["close"]
        high = df["high"]
        low = df["low"]
        volume = df["volume"].fillna(0.0)

        df["rsi"] = calculate_rsi(close, 14)

        ema12 = calculate_ema(close, 12)
        ema26 = calculate_ema(close, 26)
        macd_line = ema12 - ema26
        macd_signal = calculate_ema(macd_line, 9)
        df["macd_hist"] = macd_line - macd_signal

        tp = (high + low + close) / 3.0
        tp_sma = tp.rolling(window=20, min_periods=20).mean()
        md = (tp - tp_sma).abs().rolling(window=20, min_periods=20).mean()
        df["cci"] = (tp - tp_sma) / (0.015 * md.replace(0, np.nan))

        raw_mf = tp * volume
        pos_mf = raw_mf.where(tp > tp.shift(1), 0.0)
        neg_mf = raw_mf.where(tp < tp.shift(1), 0.0)
        pos_14 = pos_mf.rolling(window=14, min_periods=14).sum()
        neg_14 = neg_mf.rolling(window=14, min_periods=14).sum()
        mf_ratio = pos_14 / neg_14.replace(0, np.nan)
        df["mfi"] = 100.0 - (100.0 / (1.0 + mf_ratio))

        lowest_14 = low.rolling(window=14, min_periods=14).min()
        highest_14 = high.rolling(window=14, min_periods=14).max()
        range_14 = (highest_14 - lowest_14).replace(0, np.nan)
        df["stoch_k"] = 100.0 * (close - lowest_14) / range_14
        df["williams_r"] = -100.0 * ((highest_14 - close) / range_14)
        df["roc"] = (close / close.shift(12) - 1.0) * 100.0

        direction = np.sign(close.diff().fillna(0.0))
        df["obv"] = (direction * volume).fillna(0.0).cumsum()

        df["atr"] = calculate_atr(high, low, close, self.atr_period)
        df["ema_fast"] = calculate_ema(close, self.regime_fast_ema)
        df["ema_slow"] = calculate_ema(close, self.regime_slow_ema)
        df["adx"] = calculate_adx(high, low, close, 14)
        df["vol_sma"] = volume.rolling(window=self.volume_sma_period, min_periods=self.volume_sma_period).mean()
        
        return df

    def _is_pivot_low(self, arr: np.ndarray, i: int, lb: int) -> bool:
        if i - lb < 0 or i + lb >= len(arr):
            return False
        v = arr[i]
        window = arr[i - lb : i + lb + 1]
        if v > np.min(window):
            return False
        return np.sum(window == v) == 1

    def _is_pivot_high(self, arr: np.ndarray, i: int, lb: int) -> bool:
        if i - lb < 0 or i + lb >= len(arr):
            return False
        v = arr[i]
        window = arr[i - lb : i + lb + 1]
        if v < np.max(window):
            return False
        return np.sum(window == v) == 1

    def _find_pivots(self, arr: np.ndarray, high: bool) -> List[int]:
        pivots: List[int] = []
        lb = self.pivot_lookback
        n = len(arr)
        start_idx = max(lb, n - 300)
        for i in range(start_idx, n - lb):
            ok = self._is_pivot_high(arr, i, lb) if high else self._is_pivot_low(arr, i, lb)
            if ok:
                pivots.append(i)
        return pivots

    def _latest_valid_pair(self, pivots: List[int], n: int) -> Optional[Tuple[int, int]]:
        if len(pivots) < 2:
            return None
        i2 = pivots[-1]
        if (n - 1 - i2) > self.max_pivot_age_bars:
            return None
        for j in range(len(pivots) - 2, -1, -1):
            i1 = pivots[j]
            if (i2 - i1) >= self.min_pivot_separation_bars:
                return i1, i2
        return None

    def _evaluate_divergence_components(
        self,
        price_series: np.ndarray,
        indicator_series: np.ndarray,
        pair: Optional[Tuple[int, int]],
        bullish: bool,
    ) -> Tuple[bool, bool]:
        if pair is None:
            return False, False
        i1, i2 = pair
        if i1 < 0 or i2 >= len(price_series):
            return False, False
        p1 = price_series[i1]
        p2 = price_series[i2]
        s1 = indicator_series[i1]
        s2 = indicator_series[i2]
        if not (np.isfinite(p1) and np.isfinite(p2) and np.isfinite(s1) and np.isfinite(s2)):
            return False, False
        if abs(p1) <= self.EPS:
            return False, False
        price_delta_pct = abs(p2 - p1) / max(self.EPS, abs(p1))
        if price_delta_pct < self.min_price_move_pct:
            return False, False
        if abs(s2 - s1) < self.min_indicator_delta:
            return False, False

        if bullish:
            regular = (p2 < p1) and (s2 > s1)
            hidden = (p2 > p1) and (s2 < s1)
        else:
            regular = (p2 > p1) and (s2 < s1)
            hidden = (p2 < p1) and (s2 > s1)
        return bool(regular), bool(hidden)

    def _divergence_snapshot(
        self, df: pd.DataFrame
    ) -> Tuple[int, int, float, float, float, float, float, float, float, float, float, Dict[str, Dict[str, bool]]]:
        n = len(df)
        low_arr = df["low"].to_numpy(dtype=float)
        high_arr = df["high"].to_numpy(dtype=float)
        close_arr = df["close"].to_numpy(dtype=float)

        low_pivots = self._find_pivots(low_arr, high=False)
        high_pivots = self._find_pivots(high_arr, high=True)
        low_pair = self._latest_valid_pair(low_pivots, n)
        high_pair = self._latest_valid_pair(high_pivots, n)

        indicator_map = {
            "rsi": df["rsi"].to_numpy(dtype=float),
            "macd_hist": df["macd_hist"].to_numpy(dtype=float),
            "cci": df["cci"].to_numpy(dtype=float),
            "mfi": df["mfi"].to_numpy(dtype=float),
            "stoch_k": df["stoch_k"].to_numpy(dtype=float),
            "williams_r": df["williams_r"].to_numpy(dtype=float),
            "roc": df["roc"].to_numpy(dtype=float),
            "obv": df["obv"].to_numpy(dtype=float),
        }

        bullish_count = 0
        bearish_count = 0
        bullish_score = 0.0
        bearish_score = 0.0
        details: Dict[str, Dict[str, bool]] = {}

        for name in self.indicators:
            series = indicator_map.get(name)
            if series is None:
                continue
            weight = max(0.0, float(self.indicator_weights.get(name, 1.0)))

            bull_regular, bull_hidden = self._evaluate_divergence_components(low_arr, series, low_pair, bullish=True)
            bear_regular, bear_hidden = self._evaluate_divergence_components(high_arr, series, high_pair, bullish=False)

            bull_signal = (self.include_regular and bull_regular) or (self.include_hidden and bull_hidden)
            bear_signal = (self.include_regular and bear_regular) or (self.include_hidden and bear_hidden)
            details[name] = {
                "bullish": bool(bull_signal),
                "bearish": bool(bear_signal),
                "bull_regular": bool(bull_regular),
                "bull_hidden": bool(bull_hidden),
                "bear_regular": bool(bear_regular),
                "bear_hidden": bool(bear_hidden),
            }

            if bull_signal:
                bullish_count += 1
                hidden_boost = self.hidden_score_multiplier if (self.include_hidden and bull_hidden) else 1.0
                bullish_score += weight * hidden_boost
            if bear_signal:
                bearish_count += 1
                hidden_boost = self.hidden_score_multiplier if (self.include_hidden and bear_hidden) else 1.0
                bearish_score += weight * hidden_boost

        close = float(close_arr[-1])
        atr = float(df["atr"].iloc[-1]) if pd.notna(df["atr"].iloc[-1]) else 0.0
        adx = float(df["adx"].iloc[-1]) if pd.notna(df["adx"].iloc[-1]) else 0.0
        ema_fast = float(df["ema_fast"].iloc[-1]) if pd.notna(df["ema_fast"].iloc[-1]) else close
        ema_slow = float(df["ema_slow"].iloc[-1]) if pd.notna(df["ema_slow"].iloc[-1]) else close
        volume = float(df["volume"].iloc[-1]) if pd.notna(df["volume"].iloc[-1]) else 0.0
        vol_sma = float(df["vol_sma"].iloc[-1]) if pd.notna(df["vol_sma"].iloc[-1]) else 0.0
        return (
            bullish_count,
            bearish_count,
            bullish_score,
            bearish_score,
            close,
            atr,
            adx,
            ema_fast,
            ema_slow,
            volume,
            vol_sma,
            details,
        )

    def _entry_filters(
        self,
        close: float,
        atr: float,
        adx: float,
        ema_fast: float,
        ema_slow: float,
        volume: float,
        vol_sma: float,
        direction: str,
    ) -> bool:
        atr_ratio = atr / max(self.EPS, abs(close))
        if not (self.atr_ratio_floor <= atr_ratio <= self.atr_ratio_cap):
            return False
        if adx < self.min_adx_for_entry:
            return False
        if self.require_volume_confirmation and vol_sma > self.EPS and volume < vol_sma:
            return False
        if self.require_regime_filter:
            if direction == "long":
                if not (close >= ema_slow and ema_fast >= ema_slow):
                    return False
            else:
                if not (close <= ema_slow and ema_fast <= ema_slow):
                    return False
        return True

    def _open_long(self, symbol: str, timestamp: Any, close: float, atr: float, confirmations: int, score: float) -> None:
        stop_price = close - (self.sl_atr_multiplier * max(atr, close * 0.005))
        lev = self._resolve_entry_leverage(confirmations, score)
        qty = self._calculate_qty(close, stop_price, lev)
        if qty is None:
            return
        placed = self._submit_market_order(symbol, Side.BUY, qty, timestamp, leverage=lev)
        if placed:
            self.trade_state[symbol] = {"direction": "long", "sl_price": stop_price}

    def _open_short(self, symbol: str, timestamp: Any, close: float, atr: float, confirmations: int, score: float) -> None:
        stop_price = close + (self.sl_atr_multiplier * max(atr, close * 0.005))
        lev = self._resolve_entry_leverage(confirmations, score)
        qty = self._calculate_qty(close, stop_price, lev)
        if qty is None:
            return
        placed = self._submit_market_order(symbol, Side.SELL, qty, timestamp, leverage=lev)
        if placed:
            self.trade_state[symbol] = {"direction": "short", "sl_price": stop_price}

    def _manage_long(
        self,
        symbol: str,
        timestamp: Any,
        pos_size: float,
        close: float,
        low: float,
        atr: float,
        bearish_count: int,
        bearish_score: float,
    ) -> None:
        if self._has_pending_market_or_limit_order(symbol):
            return
        state = self.trade_state.get(symbol, {})
        current_sl = self._as_float(state.get("sl_price"))
        if current_sl is None:
            current_sl = close - (self.sl_atr_multiplier * max(atr, close * 0.005))
        trailed_sl = max(current_sl, close - (self.sl_atr_multiplier * max(atr, close * 0.005)))
        state["sl_price"] = trailed_sl
        self.trade_state[symbol] = state

        reversal_exit = (
            bearish_count >= self.required_bearish
            and bearish_score >= self.required_bearish_score
        )
        if low <= trailed_sl or reversal_exit:
            self._submit_market_order(symbol, Side.SELL, abs(pos_size), timestamp)

    def _manage_short(
        self,
        symbol: str,
        timestamp: Any,
        pos_size: float,
        close: float,
        high: float,
        atr: float,
        bullish_count: int,
        bullish_score: float,
    ) -> None:
        if self._has_pending_market_or_limit_order(symbol):
            return
        state = self.trade_state.get(symbol, {})
        current_sl = self._as_float(state.get("sl_price"))
        if current_sl is None:
            current_sl = close + (self.sl_atr_multiplier * max(atr, close * 0.005))
        trailed_sl = min(current_sl, close + (self.sl_atr_multiplier * max(atr, close * 0.005)))
        state["sl_price"] = trailed_sl
        self.trade_state[symbol] = state

        reversal_exit = (
            bullish_count >= self.required_bullish
            and bullish_score >= self.required_bullish_score
        )
        if high >= trailed_sl or reversal_exit:
            self._submit_market_order(symbol, Side.BUY, abs(pos_size), timestamp)

    def on_bar(self, bar: Dict[str, Any]):
        self.bar_index += 1
        symbol = bar.get("symbol")
        timestamp = bar.get("timestamp")
        open_ = self._as_float(bar.get("open"))
        high = self._as_float(bar.get("high"))
        low = self._as_float(bar.get("low"))
        close = self._as_float(bar.get("close"))
        volume = self._as_float(bar.get("volume")) or 0.0
        if not symbol or timestamp is None:
            return
        if any(v is None for v in (open_, high, low, close)):
            return

        clean_bar = {
            "timestamp": self._normalize_timestamp(timestamp),
            "open": float(open_),
            "high": float(high),
            "low": float(low),
            "close": float(close),
            "volume": float(volume),
        }
        self._append_history(symbol, clean_bar)

        frame = self._update_indicators(symbol)
        if frame is None:
            return
        (
            bullish_count,
            bearish_count,
            bullish_score,
            bearish_score,
            last_close,
            atr,
            adx,
            ema_fast,
            ema_slow,
            bar_volume,
            vol_sma,
            _details,
        ) = self._divergence_snapshot(frame)

        if not np.isfinite(last_close) or atr <= self.EPS:
            return

        pos_size = self.get_position_size(symbol)
        if abs(pos_size) <= self.EPS:
            self.trade_state.pop(symbol, None)
            if self._in_cooldown(symbol):
                return
            if self._has_pending_market_or_limit_order(symbol):
                return

            long_signal = (
                bullish_count >= self.required_bullish
                and bullish_score >= self.required_bullish_score
                and self._entry_filters(
                    close=last_close,
                    atr=atr,
                    adx=adx,
                    ema_fast=ema_fast,
                    ema_slow=ema_slow,
                    volume=bar_volume,
                    vol_sma=vol_sma,
                    direction="long",
                )
            )
            if long_signal:
                self._open_long(symbol, timestamp, last_close, atr, bullish_count, bullish_score)
                return

            short_signal = (
                self.enable_shorts
                and bearish_count >= self.required_bearish
                and bearish_score >= self.required_bearish_score
                and self._entry_filters(
                    close=last_close,
                    atr=atr,
                    adx=adx,
                    ema_fast=ema_fast,
                    ema_slow=ema_slow,
                    volume=bar_volume,
                    vol_sma=vol_sma,
                    direction="short",
                )
            )
            if short_signal:
                self._open_short(symbol, timestamp, last_close, atr, bearish_count, bearish_score)
                return

        if pos_size > self.EPS:
            self._manage_long(symbol, timestamp, pos_size, last_close, float(low), atr, bearish_count, bearish_score)
            return
        if pos_size < -self.EPS:
            self._manage_short(symbol, timestamp, pos_size, last_close, float(high), atr, bullish_count, bullish_score)


    def on_fill(self, trade: Trade):
        symbol = trade.symbol
        pos_size = self.get_position_size(symbol)
        if abs(pos_size) <= self.EPS:
            self.trade_state.pop(symbol, None)
            self.last_exit_bar[symbol] = self.bar_index
            return
        state = self.trade_state.get(symbol, {})
        state["direction"] = "long" if pos_size > 0 else "short"
        self.trade_state[symbol] = state
