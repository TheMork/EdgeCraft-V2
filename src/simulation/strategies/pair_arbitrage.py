import math
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, Mapping, Optional

import numpy as np
import pandas as pd

from src.database import QuestDBManager
from src.simulation.models import Order, OrderType, Side, Trade
from src.simulation.strategy import Strategy


@dataclass
class OUParams:
    mu: float
    theta: float
    sigma: float


@dataclass
class Thresholds:
    entry_z: float
    exit_z: float
    stop_loss_z: float


class PairArbitrageStrategy(Strategy):
    EPS = 1e-12
    MAX_HISTORY = 50_000

    def __init__(
        self,
        timeframe: str = "1h",
        variant: str = "v1",
        reference_symbol: str = "BTC/USDT",
        asset_symbol: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        enable_shorts: bool = True,
        min_leverage: int = 1,
        max_leverage: int = 3,
        cooldown_bars: int = 6,
        max_notional_fraction: float = 0.9,
        min_order_qty: float = 1e-6,
        fee_rate: float = 0.001,
        slippage_rate: float = 0.001,
        ols_window: int = 100,
        z_window: int = 144,
        z_entry: float = 2.0,
        kalman_q: float = 1e-9,
        kalman_r: float = 1e-4,
        ou_window: int = 300,
        ou_theta_min: float = 0.005,
        v3_stop_z: float = 4.0,
        v3_trend_hurst_min: float = 0.70,
        v3_mean_revert_hurst_max: float = 0.55,
        v4_min_entry_z: float = 2.4,
        v4_stop_z: float = 4.0,
        v4_mean_exit_z: float = 0.5,
        v4_trend_hurst_min: float = 0.70,
        v4_half_life_mult: float = 2.0,
        v4_min_hold_bars: int = 24,
        v4_max_hold_bars: int = 240,
    ):
        super().__init__()
        self.timeframe = timeframe
        self.variant = (variant or "v1").strip().lower()
        if self.variant not in {"v1", "v2", "v3", "v4"}:
            self.variant = "v1"

        self.reference_symbol = reference_symbol
        self.asset_symbol = asset_symbol
        self.start_date = start_date
        self.end_date = end_date
        self.enable_shorts = enable_shorts
        self.min_leverage = max(1, int(min_leverage))
        self.max_leverage = max(self.min_leverage, int(max_leverage))
        self.cooldown_bars = max(0, int(cooldown_bars))
        self.max_notional_fraction = float(max_notional_fraction)
        self.min_order_qty = float(min_order_qty)
        self.fee_rate = max(0.0, float(fee_rate))
        self.slippage_rate = max(0.0, float(slippage_rate))

        self.ols_window = max(5, int(ols_window))
        self.z_window = max(5, int(z_window))
        self.base_entry_z = float(z_entry)
        self.kalman_q = max(self.EPS, float(kalman_q))
        self.kalman_r = max(self.EPS, float(kalman_r))
        self.ou_window = max(20, int(ou_window))
        self.ou_theta_min = max(0.0, float(ou_theta_min))
        self.v3_stop_z = max(0.1, float(v3_stop_z))
        self.v3_trend_hurst_min = float(v3_trend_hurst_min)
        self.v3_mean_revert_hurst_max = float(v3_mean_revert_hurst_max)

        self.v4_min_entry_z = max(0.1, float(v4_min_entry_z))
        self.v4_stop_z = max(0.1, float(v4_stop_z))
        self.v4_mean_exit_z = max(0.0, float(v4_mean_exit_z))
        self.v4_trend_hurst_min = float(v4_trend_hurst_min)
        self.v4_half_life_mult = max(0.1, float(v4_half_life_mult))
        self.v4_min_hold_bars = max(1, int(v4_min_hold_bars))
        self.v4_max_hold_bars = max(self.v4_min_hold_bars, int(v4_max_hold_bars))

        self.bar_index = 0
        self.last_exit_bar = -10_000
        self.position_open_bar: Optional[int] = None
        self.active_max_hold_bars: Optional[int] = None
        self.pending_action: Optional[Dict[str, Any]] = None

        self.log_y: list[float] = []
        self.log_x: list[float] = []
        self.residuals: list[float] = []
        self.betas: list[float] = []
        self.z_scores: list[float] = []

        self._kalman_state: Optional[list[float]] = None
        self._kalman_cov: Optional[list[list[float]]] = None

        self.reference_prices: Dict[int, float] = {}
        self.reference_last_close: Optional[float] = None

    def on_start(self):
        if self.variant == "v4":
            self.base_entry_z = self.v4_min_entry_z
        if not self.reference_prices:
            self._load_reference_from_db()
        print(
            "PairArbitrageStrategy started "
            f"(variant={self.variant}, tf={self.timeframe}, ref={self.reference_symbol}, "
            f"shorts={self.enable_shorts}, lev={self.min_leverage}-{self.max_leverage})."
        )

    def on_stop(self):
        print("PairArbitrageStrategy stopped.")

    def set_reference_series(self, series: Mapping[Any, Any] | Iterable[tuple[Any, Any]]) -> None:
        parsed: Dict[int, float] = {}
        items = series.items() if isinstance(series, Mapping) else series
        for ts, raw_price in items:
            key = self._to_ts_key(ts)
            price = self._to_positive_float(raw_price)
            if key is None or price is None:
                continue
            parsed[key] = price
        self.reference_prices = parsed

    def _to_float(self, value: Any) -> Optional[float]:
        try:
            out = float(value)
        except (TypeError, ValueError):
            return None
        if not np.isfinite(out):
            return None
        return out

    def _to_positive_float(self, value: Any) -> Optional[float]:
        out = self._to_float(value)
        if out is None or out <= 0:
            return None
        return out

    def _to_datetime(self, timestamp: Any) -> datetime:
        if isinstance(timestamp, datetime):
            return timestamp if timestamp.tzinfo else timestamp.replace(tzinfo=timezone.utc)
        parsed = pd.to_datetime(timestamp, utc=True, errors="coerce")
        if pd.isna(parsed):
            return datetime.now(timezone.utc)
        return parsed.to_pydatetime()

    def _to_ts_key(self, timestamp: Any) -> Optional[int]:
        parsed = pd.to_datetime(timestamp, utc=True, errors="coerce")
        if pd.isna(parsed):
            return None
        return int(parsed.value)

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

    def _enum_to_str(self, value: Any) -> str:
        if value is None:
            return ""
        if hasattr(value, "value"):
            return str(getattr(value, "value")).upper()
        raw = str(value)
        if "." in raw:
            raw = raw.split(".")[-1]
        return raw.upper()

    def _has_pending_market_order(self, symbol: str) -> bool:
        for order in self._iter_open_orders():
            if getattr(order, "symbol", None) != symbol:
                continue
            if self._enum_to_str(getattr(order, "order_type", "")) == "MARKET":
                return True
        return False

    def _load_reference_from_db(self) -> None:
        if not self.start_date or not self.end_date:
            return
        if self.asset_symbol and self.reference_symbol == self.asset_symbol:
            return
        try:
            db = QuestDBManager()
            safe_tf = db.parse_timeframe(self.timeframe) or "1m"
            frame = db.get_ohlcv(self.reference_symbol, self.start_date, self.end_date, timeframe=safe_tf)
            if frame.empty:
                return
            self.reference_prices = {}
            close_col = frame["close"] if "close" in frame.columns else None
            if close_col is None:
                return
            for ts, close in close_col.items():
                key = self._to_ts_key(ts)
                val = self._to_positive_float(close)
                if key is None or val is None:
                    continue
                self.reference_prices[key] = val
        except Exception as exc:
            print(f"PairArbitrage: could not preload reference data: {exc}")

    def _lookup_reference_close(self, symbol: str, timestamp: Any, asset_close: float) -> Optional[float]:
        if symbol == self.reference_symbol:
            return asset_close
        key = self._to_ts_key(timestamp)
        if key is not None:
            raw = self.reference_prices.get(key)
            if raw is not None and raw > 0:
                self.reference_last_close = raw
        return self.reference_last_close

    def _rolling_ols_residual(self, window: int) -> tuple[Optional[float], Optional[float]]:
        if len(self.log_y) < window or len(self.log_x) < window:
            return None, None
        y_slice = np.array(self.log_y[-window:], dtype=float)
        x_slice = np.array(self.log_x[-window:], dtype=float)
        mask = np.isfinite(y_slice) & np.isfinite(x_slice) & (x_slice != 0.0)
        if int(mask.sum()) < int(window * 0.7):
            return None, None
        x_valid = x_slice[mask]
        y_valid = y_slice[mask]
        mean_x = float(np.mean(x_valid))
        mean_y = float(np.mean(y_valid))
        cov = float(np.mean(x_valid * y_valid) - (mean_x * mean_y))
        var_x = float(np.mean(x_valid * x_valid) - (mean_x * mean_x))
        beta = cov / var_x if abs(var_x) > self.EPS else 0.0
        alpha = mean_y - beta * mean_x
        residual = self.log_y[-1] - (alpha + beta * self.log_x[-1])
        return residual, beta

    def _kalman_update(self, y: float, x: float) -> tuple[float, float]:
        if self._kalman_state is None or self._kalman_cov is None:
            self._kalman_state = [y, 0.0]
            self._kalman_cov = [[1.0, 0.0], [0.0, 1.0]]

        s = self._kalman_state
        p = self._kalman_cov
        q = self.kalman_q
        r = self.kalman_r

        x_pred = [s[0], s[1]]
        p_pred = [
            [p[0][0] + q, p[0][1]],
            [p[1][0], p[1][1] + q],
        ]

        y_pred = x_pred[0] + x_pred[1] * x
        error = y - y_pred

        innovation = (
            p_pred[0][0]
            + p_pred[0][1] * x
            + (p_pred[1][0] + p_pred[1][1] * x) * x
            + r
        )
        if abs(innovation) <= self.EPS:
            innovation = self.EPS

        k0 = (p_pred[0][0] + p_pred[0][1] * x) / innovation
        k1 = (p_pred[1][0] + p_pred[1][1] * x) / innovation

        s_new = [x_pred[0] + k0 * error, x_pred[1] + k1 * error]
        p_new = [
            [p_pred[0][0] - (k0 * innovation * k0), p_pred[0][1] - (k0 * innovation * k1)],
            [p_pred[1][0] - (k1 * innovation * k0), p_pred[1][1] - (k1 * innovation * k1)],
        ]

        self._kalman_state = s_new
        self._kalman_cov = p_new
        return error, s_new[1]

    def _latest_zscore(self, values: list[float], window: int) -> Optional[float]:
        if len(values) < window:
            return None
        arr = np.array(values[-window:], dtype=float)
        mask = np.isfinite(arr)
        if int(mask.sum()) < int(window * 0.7):
            return None
        arr = arr[mask]
        std = float(np.std(arr))
        if std <= self.EPS:
            return 0.0
        mean = float(np.mean(arr))
        return (values[-1] - mean) / std

    def _calibrate_ou(self, values: list[float]) -> OUParams:
        if len(values) < 3:
            return OUParams(mu=0.0, theta=0.0, sigma=0.0)
        x_t = np.array(values[:-1], dtype=float)
        x_next = np.array(values[1:], dtype=float)
        mask = np.isfinite(x_t) & np.isfinite(x_next)
        if int(mask.sum()) < 3:
            return OUParams(mu=0.0, theta=0.0, sigma=0.0)
        x_t = x_t[mask]
        x_next = x_next[mask]
        if len(x_t) < 3:
            return OUParams(mu=0.0, theta=0.0, sigma=0.0)

        mean_x = float(np.mean(x_t))
        mean_y = float(np.mean(x_next))
        num = float(np.sum((x_t - mean_x) * (x_next - mean_y)))
        den = float(np.sum((x_t - mean_x) ** 2))
        a = num / den if abs(den) > self.EPS else 0.0
        b = mean_y - a * mean_x

        residuals = x_next - (a * x_t + b)
        sd_eps = float(np.std(residuals))

        if a <= 0.0 or a >= 1.0:
            return OUParams(mu=mean_x, theta=0.001, sigma=sd_eps)

        theta = -math.log(a)
        mu = b / max(self.EPS, 1.0 - a)
        term = max(self.EPS, 1.0 - math.exp(-2.0 * theta))
        sigma = sd_eps * math.sqrt((2.0 * theta) / term)
        return OUParams(mu=mu, theta=theta, sigma=sigma)

    def _calculate_optimal_thresholds(self, params: OUParams) -> Thresholds:
        theta = max(self.EPS, params.theta)
        sigma_eq = params.sigma / math.sqrt(2.0 * theta) if params.sigma > 0 else self.EPS
        total_cost = max(0.0, 2.0 * (self.fee_rate + self.slippage_rate))
        cost_ratio = total_cost / max(self.EPS, abs(sigma_eq))

        entry = 2.0
        if theta < 0.05:
            entry = 2.0 + (0.05 - theta) * 30.0
        if cost_ratio > 0.1:
            entry += (cost_ratio - 0.1) * 2.0
        entry = float(max(1.5, min(4.5, entry)))
        return Thresholds(entry_z=entry, exit_z=0.0, stop_loss_z=entry * 2.0)

    def _calculate_hurst(self, values: list[float]) -> float:
        n = len(values)
        if n < 50:
            return 0.5
        arr = np.array(values, dtype=float)
        arr = arr[np.isfinite(arr)]
        if len(arr) < 50:
            return 0.5
        mean = float(np.mean(arr))
        dev = arr - mean
        cum = np.cumsum(dev)
        r = float(np.max(cum) - np.min(cum))
        s = float(np.std(dev))
        if s <= self.EPS or r <= self.EPS:
            return 0.5
        hurst = math.log(r / s) / max(self.EPS, math.log(float(len(arr))))
        return float(max(0.0, min(1.0, hurst)))

    def _resolve_entry_leverage(self, abs_z: float) -> int:
        lo = max(1.0, self.base_entry_z)
        hi = max(lo + 1.0, 4.5)
        if abs_z <= lo:
            return self.min_leverage
        if abs_z >= hi:
            return self.max_leverage
        ratio = (abs_z - lo) / (hi - lo)
        lev = self.min_leverage + ratio * (self.max_leverage - self.min_leverage)
        return int(round(max(self.min_leverage, min(self.max_leverage, lev))))

    def _calculate_order_qty(self, price: float, leverage: int) -> Optional[float]:
        if price <= self.EPS:
            return None
        lev = max(1, int(leverage))
        try:
            available = float(self.broker.get_available_balance()) if self.broker else float(self.balance)
        except Exception:
            available = float(self.balance)
        notional = max(0.0, available * lev * max(0.0, self.max_notional_fraction))
        qty = notional / price
        if qty <= self.min_order_qty:
            return None
        return float(qty)

    def _submit_market_order(
        self,
        symbol: str,
        side: Side,
        quantity: float,
        timestamp: Any,
        leverage: int,
    ) -> Optional[Order]:
        if quantity <= self.min_order_qty:
            return None
        order = Order(
            id="",
            symbol=symbol,
            side=side,
            order_type=OrderType.MARKET,
            quantity=float(quantity),
            price=None,
            timestamp=self._to_datetime(timestamp),
            leverage=max(1, int(leverage)),
        )
        placed = self.submit_order(order)
        if not placed:
            return None
        if self._enum_to_str(getattr(placed, "status", "")) == "REJECTED":
            return None
        return placed

    def _trim_history(self) -> None:
        if len(self.log_y) <= self.MAX_HISTORY:
            return
        keep = self.MAX_HISTORY
        self.log_y = self.log_y[-keep:]
        self.log_x = self.log_x[-keep:]
        self.residuals = self.residuals[-keep:]
        self.betas = self.betas[-keep:]
        self.z_scores = self.z_scores[-keep:]

    def _schedule_action(
        self,
        action: str,
        signal_z: float,
        max_hold_bars: Optional[int] = None,
    ) -> None:
        self.pending_action = {
            "action": action,
            "signal_z": float(signal_z),
            "max_hold_bars": int(max_hold_bars) if max_hold_bars is not None else None,
        }

    def _execute_pending_action(self, symbol: str, timestamp: Any, exec_price: float) -> None:
        if not self.pending_action:
            return
        action = str(self.pending_action.get("action") or "")
        signal_z = abs(float(self.pending_action.get("signal_z") or 0.0))
        max_hold = self.pending_action.get("max_hold_bars")

        pos_size = self.get_position_size(symbol)
        pending_exists = self._has_pending_market_order(symbol)

        if action == "CLOSE_LONG" and pos_size > self.EPS and not pending_exists:
            self._submit_market_order(symbol, Side.SELL, abs(pos_size), timestamp, leverage=self.min_leverage)
        elif action == "CLOSE_SHORT" and pos_size < -self.EPS and not pending_exists:
            self._submit_market_order(symbol, Side.BUY, abs(pos_size), timestamp, leverage=self.min_leverage)
        elif action == "OPEN_LONG" and abs(pos_size) <= self.EPS and not pending_exists:
            if (self.bar_index - self.last_exit_bar) >= self.cooldown_bars:
                lev = self._resolve_entry_leverage(signal_z)
                qty = self._calculate_order_qty(exec_price, lev)
                if qty is not None:
                    placed = self._submit_market_order(symbol, Side.BUY, qty, timestamp, leverage=lev)
                    if placed:
                        self.position_open_bar = self.bar_index
                        self.active_max_hold_bars = int(max_hold) if max_hold is not None else None
        elif action == "OPEN_SHORT" and abs(pos_size) <= self.EPS and not pending_exists and self.enable_shorts:
            if (self.bar_index - self.last_exit_bar) >= self.cooldown_bars:
                lev = self._resolve_entry_leverage(signal_z)
                qty = self._calculate_order_qty(exec_price, lev)
                if qty is not None:
                    placed = self._submit_market_order(symbol, Side.SELL, qty, timestamp, leverage=lev)
                    if placed:
                        self.position_open_bar = self.bar_index
                        self.active_max_hold_bars = int(max_hold) if max_hold is not None else None

        self.pending_action = None

    def on_bar(self, bar: Dict[str, Any]):
        self.bar_index += 1
        symbol = bar.get("symbol")
        timestamp = bar.get("timestamp")
        close = self._to_positive_float(bar.get("close"))
        open_price = self._to_positive_float(bar.get("open")) or close
        if not symbol or not timestamp or close is None or open_price is None:
            return

        self._execute_pending_action(symbol, timestamp, open_price)

        ref_close = self._lookup_reference_close(symbol, timestamp, close)
        if ref_close is None or ref_close <= 0:
            return

        y = math.log(close)
        x = math.log(ref_close)
        if not (math.isfinite(y) and math.isfinite(x)):
            return

        self.log_y.append(y)
        self.log_x.append(x)

        residual: Optional[float] = None
        beta: Optional[float] = None
        if self.variant == "v1":
            residual, beta = self._rolling_ols_residual(self.ols_window)
        else:
            residual, beta = self._kalman_update(y, x)

        if residual is None or beta is None or not (math.isfinite(residual) and math.isfinite(beta)):
            return

        self.residuals.append(float(residual))
        self.betas.append(float(beta))
        z = self._latest_zscore(self.residuals, self.z_window)
        if z is None or not math.isfinite(z):
            return
        self.z_scores.append(float(z))
        self._trim_history()

        if len(self.z_scores) < 2:
            return

        prev_z = self.z_scores[-2]
        entry_threshold = self.base_entry_z
        skip_entry = False
        close_long = False
        close_short = False
        dynamic_max_hold: Optional[int] = None

        if self.variant in {"v2", "v3", "v4"} and len(self.residuals) > self.ou_window:
            slice_res = self.residuals[-self.ou_window :]
            ou = self._calibrate_ou(slice_res)
            thresholds = self._calculate_optimal_thresholds(ou)
            entry_threshold = max(entry_threshold, thresholds.entry_z)
            if ou.theta < self.ou_theta_min:
                skip_entry = True

            half_life = math.inf
            if ou.theta > self.EPS:
                half_life = math.log(2.0) / ou.theta
            hurst = self._calculate_hurst(slice_res)

            if self.variant == "v3":
                if ou.theta < self.ou_theta_min:
                    skip_entry = True
                elif hurst >= self.v3_trend_hurst_min:
                    skip_entry = True
                elif hurst <= self.v3_mean_revert_hurst_max:
                    skip_entry = False

                close_long = z <= -self.v3_stop_z
                close_short = z >= self.v3_stop_z
                if math.isfinite(half_life):
                    dynamic_max_hold = int(max(8, min(480, round(2.0 * half_life))))
                else:
                    dynamic_max_hold = 240
            elif self.variant == "v4":
                if hurst >= self.v4_trend_hurst_min:
                    skip_entry = True

                if math.isfinite(half_life):
                    dyn = int(round(self.v4_half_life_mult * half_life))
                    dynamic_max_hold = int(max(self.v4_min_hold_bars, min(self.v4_max_hold_bars, dyn)))
                else:
                    dynamic_max_hold = self.v4_max_hold_bars

                mean_exit_long = prev_z < -self.v4_mean_exit_z and z >= -self.v4_mean_exit_z
                mean_exit_short = prev_z > self.v4_mean_exit_z and z <= self.v4_mean_exit_z
                hard_stop_long = z <= -self.v4_stop_z
                hard_stop_short = z >= self.v4_stop_z
                close_long = mean_exit_long or hard_stop_long
                close_short = mean_exit_short or hard_stop_short

        sell_signal = prev_z >= entry_threshold and z < entry_threshold
        buy_signal = prev_z <= -entry_threshold and z > -entry_threshold
        if self.variant in {"v1", "v2", "v3", "v4"}:
            close_long = close_long or sell_signal
            close_short = close_short or buy_signal

        pos_size = self.get_position_size(symbol)
        timed_out = False
        if (
            self.position_open_bar is not None
            and self.active_max_hold_bars is not None
            and self.active_max_hold_bars > 0
            and abs(pos_size) > self.EPS
        ):
            timed_out = (self.bar_index - self.position_open_bar) >= self.active_max_hold_bars

        if abs(pos_size) <= self.EPS:
            if self.pending_action is None and not skip_entry and not self._has_pending_market_order(symbol):
                if self.enable_shorts and sell_signal:
                    self._schedule_action("OPEN_SHORT", signal_z=z, max_hold_bars=dynamic_max_hold)
                elif buy_signal:
                    self._schedule_action("OPEN_LONG", signal_z=z, max_hold_bars=dynamic_max_hold)
            return

        if pos_size > self.EPS and (close_long or timed_out):
            self._schedule_action("CLOSE_LONG", signal_z=z)
            return
        if pos_size < -self.EPS and (close_short or timed_out):
            self._schedule_action("CLOSE_SHORT", signal_z=z)

    def on_fill(self, trade: Trade):
        symbol = trade.symbol
        pos_size = self.get_position_size(symbol)
        if abs(pos_size) <= self.EPS:
            self.last_exit_bar = self.bar_index
            self.position_open_bar = None
            self.active_max_hold_bars = None
            return
        if self.position_open_bar is None:
            self.position_open_bar = self.bar_index
