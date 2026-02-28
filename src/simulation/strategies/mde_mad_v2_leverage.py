from typing import Any, Optional

import numpy as np

from src.simulation.strategies.mde_mad_v2 import MDEMADV2Strategy


class MDEMADV2LeverageStrategy(MDEMADV2Strategy):
    """
    MDE-MAD v2 with explicit target exposure scaling.
    The optimizer still computes a signal weight, then this strategy multiplies
    that signal by `target_leverage_multiplier` before rebalancing.
    """

    def __init__(
        self,
        *args: Any,
        target_leverage_multiplier: float = 2.0,
        max_effective_leverage: Optional[float] = None,
        enable_volatility_targeting: bool = True,
        volatility_lookback_bars: int = 60,
        target_annual_volatility: float = 0.90,
        min_volatility_scale: float = 0.35,
        max_volatility_scale: float = 1.25,
        enable_adaptive_leverage: bool = True,
        flat_outside_regime: bool = True,
        strong_trend_slope_threshold: float = 0.0015,
        strong_trend_distance_threshold: float = 0.02,
        **kwargs: Any,
    ):
        super().__init__(*args, **kwargs)
        self.target_leverage_multiplier = max(1.0, float(target_leverage_multiplier))
        if max_effective_leverage is None:
            self.max_effective_leverage = float(self.max_leverage)
        else:
            self.max_effective_leverage = max(1.0, float(max_effective_leverage))
        self.enable_volatility_targeting = bool(enable_volatility_targeting)
        self.volatility_lookback_bars = max(5, int(volatility_lookback_bars))
        self.target_annual_volatility = max(0.01, float(target_annual_volatility))
        self.min_volatility_scale = max(0.0, float(min_volatility_scale))
        self.max_volatility_scale = max(self.min_volatility_scale, float(max_volatility_scale))
        self.enable_adaptive_leverage = bool(enable_adaptive_leverage)
        self.flat_outside_regime = bool(flat_outside_regime)
        self.strong_trend_slope_threshold = max(1e-6, float(strong_trend_slope_threshold))
        self.strong_trend_distance_threshold = max(1e-6, float(strong_trend_distance_threshold))
        # Keep the effective rebalance trigger close to v2 behavior after scaling.
        self.min_rebalance_weight_delta = self.min_rebalance_weight_delta / self.target_leverage_multiplier

    def on_start(self):
        print(
            f"MDE-MAD-v2 Leverage started "
            f"(x{self.target_leverage_multiplier:.2f}, cap={self.max_effective_leverage:.2f}, "
            f"vol_target={'on' if self.enable_volatility_targeting else 'off'}, "
            f"adaptive={'on' if self.enable_adaptive_leverage else 'off'})."
        )

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

    def _periods_per_year(self) -> float:
        minutes = self._timeframe_to_minutes(self.timeframe)
        if minutes is None or minutes <= 0:
            return 365.0 * 24.0
        return (365.0 * 24.0 * 60.0) / float(minutes)

    def _realized_annualized_volatility(self) -> Optional[float]:
        if len(self.history) < self.volatility_lookback_bars + 1:
            return None
        prices = np.asarray(self.history[-(self.volatility_lookback_bars + 1):], dtype=float)
        returns = self._calculate_log_returns(prices)
        if len(returns) < 2:
            return None
        bar_vol = float(np.std(returns, ddof=1))
        if not np.isfinite(bar_vol):
            return None
        if bar_vol <= self.EPS:
            return 0.0
        return bar_vol * np.sqrt(self._periods_per_year())

    def _volatility_scale(self) -> float:
        if not self.enable_volatility_targeting:
            return 1.0
        realized_vol = self._realized_annualized_volatility()
        if realized_vol is None:
            return 1.0
        if realized_vol <= self.EPS:
            return self.max_volatility_scale
        raw_scale = self.target_annual_volatility / realized_vol
        return float(np.clip(raw_scale, self.min_volatility_scale, self.max_volatility_scale))

    def _adaptive_regime_multiplier(self, signal_target_weight: float) -> float:
        if not self.enable_adaptive_leverage:
            return self.target_leverage_multiplier
        if abs(float(signal_target_weight)) <= self.EPS:
            return 0.0
        if len(self.history) < 2:
            return self.target_leverage_multiplier

        close = float(self.history[-1])
        ema_now = float(self._calculate_ema(self.history, self.trend_filter_period))
        if len(self.history) > 2:
            ema_prev = float(self._calculate_ema(self.history[:-1], self.trend_filter_period))
        else:
            ema_prev = ema_now

        ema_denom = max(abs(ema_now), self.EPS)
        prev_denom = max(abs(ema_prev), self.EPS)
        slope = (ema_now - ema_prev) / prev_denom

        if signal_target_weight > 0.0:
            in_regime = close >= ema_now
            slope_good = slope > 0.0
            distance = max(0.0, (close - ema_now) / ema_denom)
        else:
            in_regime = close <= ema_now
            slope_good = slope < 0.0
            distance = max(0.0, (ema_now - close) / ema_denom)

        if not in_regime:
            return 0.0 if self.flat_outside_regime else 1.0
        if not slope_good:
            return 1.0

        slope_score = min(1.0, abs(slope) / self.strong_trend_slope_threshold)
        distance_score = min(1.0, distance / self.strong_trend_distance_threshold)
        trend_score = 0.5 * (slope_score + distance_score)
        return 1.0 + trend_score * (self.target_leverage_multiplier - 1.0)

    def _scaled_target_weight(self, signal_target_weight: float) -> float:
        if abs(float(signal_target_weight)) <= self.EPS:
            return 0.0
        regime_multiplier = self._adaptive_regime_multiplier(float(signal_target_weight))
        vol_scale = self._volatility_scale()
        effective_multiplier = regime_multiplier * vol_scale
        scaled = float(signal_target_weight) * effective_multiplier
        return float(np.clip(scaled, -self.max_effective_leverage, self.max_effective_leverage))

    def _rebalance(self, symbol: str, target_weight: float, price: float, timestamp: Any):
        # Keep current_weight in signal space for optimizer/turnover logic.
        signal_target = float(target_weight)
        effective_target = self._scaled_target_weight(signal_target)
        previous_signal = float(self.current_weight)

        super()._rebalance(symbol, effective_target, price, timestamp)

        # super() writes current_weight in effective space when order is placed;
        # restore signal-space current weight for the next optimization step.
        if abs(float(self.current_weight) - effective_target) < 1e-12:
            self.current_weight = signal_target
        elif abs(float(self.current_weight) - previous_signal) < 1e-12:
            self.current_weight = previous_signal
