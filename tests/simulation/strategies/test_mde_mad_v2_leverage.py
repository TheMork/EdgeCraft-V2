from datetime import datetime, timezone

from src.simulation.strategies.mde_mad_v2 import MDEMADV2Strategy
from src.simulation.strategies.mde_mad_v2_leverage import MDEMADV2LeverageStrategy


def test_scaled_target_weight_respects_multiplier_and_cap():
    strategy = MDEMADV2LeverageStrategy(
        max_leverage=3,
        target_leverage_multiplier=2.0,
        max_effective_leverage=2.5,
    )

    assert strategy._scaled_target_weight(0.5) == 1.0
    assert strategy._scaled_target_weight(2.0) == 2.5
    assert strategy._scaled_target_weight(-2.0) == -2.5


def test_rebalance_keeps_signal_weight_space(monkeypatch):
    strategy = MDEMADV2LeverageStrategy(
        max_leverage=3,
        target_leverage_multiplier=2.0,
        max_effective_leverage=3.0,
    )
    strategy.current_weight = 0.4
    captured = {}

    def fake_super_rebalance(self, symbol, target_weight, price, timestamp):
        captured["target_weight"] = float(target_weight)
        # Simulate successful order place behavior in base strategy.
        self.current_weight = float(target_weight)

    monkeypatch.setattr(MDEMADV2Strategy, "_rebalance", fake_super_rebalance)

    strategy._rebalance(
        symbol="BTC/USDT",
        target_weight=0.5,
        price=50000.0,
        timestamp=datetime(2024, 1, 1, tzinfo=timezone.utc),
    )

    assert captured["target_weight"] == 1.0
    assert strategy.current_weight == 0.5


def test_volatility_targeting_reduces_scaling_in_high_vol_regime():
    strategy = MDEMADV2LeverageStrategy(
        timeframe="4h",
        max_leverage=5,
        target_leverage_multiplier=2.0,
        max_effective_leverage=5.0,
        enable_adaptive_leverage=False,
        enable_volatility_targeting=True,
        volatility_lookback_bars=6,
        target_annual_volatility=0.3,
        min_volatility_scale=0.4,
        max_volatility_scale=1.5,
    )
    strategy.history = [100.0, 130.0, 80.0, 140.0, 75.0, 145.0, 70.0]

    # High volatility should clip to min_volatility_scale => 2.0 * 0.4 = 0.8x
    scaled = strategy._scaled_target_weight(1.0)
    assert abs(scaled - 0.8) < 1e-12


def test_adaptive_leverage_zeros_longs_when_below_trend():
    strategy = MDEMADV2LeverageStrategy(
        timeframe="4h",
        trend_filter_period=3,
        max_leverage=5,
        target_leverage_multiplier=3.0,
        max_effective_leverage=5.0,
        enable_adaptive_leverage=True,
        flat_outside_regime=True,
        enable_volatility_targeting=False,
    )
    # Clear downtrend, so long exposure should be switched off.
    strategy.history = [200.0, 190.0, 180.0, 170.0, 160.0, 150.0, 140.0, 130.0]
    scaled = strategy._scaled_target_weight(1.0)
    assert scaled == 0.0


def test_adaptive_leverage_uses_boost_in_strong_trend():
    strategy = MDEMADV2LeverageStrategy(
        timeframe="4h",
        trend_filter_period=3,
        max_leverage=5,
        target_leverage_multiplier=3.0,
        max_effective_leverage=5.0,
        enable_adaptive_leverage=True,
        enable_volatility_targeting=False,
        strong_trend_slope_threshold=1e-5,
        strong_trend_distance_threshold=1e-4,
    )
    strategy.history = [100.0, 102.0, 105.0, 109.0, 114.0, 120.0, 127.0, 135.0]

    scaled = strategy._scaled_target_weight(1.0)
    # With tiny thresholds the trend score saturates and reaches full target multiplier.
    assert abs(scaled - 3.0) < 1e-9
