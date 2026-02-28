from datetime import datetime, timezone

from src.simulation.strategies.mde_mad_v2 import MDEMADV2Strategy


class DummyTrade:
    def __init__(self, order_id: str, pnl: float, timestamp: datetime):
        self.order_id = order_id
        self.pnl = pnl
        self.timestamp = timestamp


def test_cost_gate_blocks_low_edge_and_allows_high_edge():
    strategy = MDEMADV2Strategy(
        enable_cost_gate=True,
        fee_rate=0.001,
        slippage_rate=0.001,
        edge_horizon_bars=2,
        min_edge_over_cost_ratio=1.0,
    )

    # Cost = notional * 2 * (fee + slippage) = 1000 * 2 * 0.002 = 4
    # Edge = notional * expected_return * horizon
    assert not strategy._passes_cost_gate(1000.0, expected_return=0.001)  # Edge=2 < 4
    assert strategy._passes_cost_gate(1000.0, expected_return=0.003)  # Edge=6 > 4


def test_cooldown_handles_mixed_timezone_timestamps():
    strategy = MDEMADV2Strategy()
    strategy.cooldown_until = datetime(2024, 1, 1, 1, 0, 0, tzinfo=timezone.utc)

    # Naive timestamp input should not raise and should be normalized internally.
    assert strategy._is_cooldown_active(datetime(2024, 1, 1, 0, 30, 0))
    assert not strategy._is_cooldown_active(datetime(2024, 1, 1, 1, 30, 0))


def test_on_fill_sets_utc_cooldown_timestamp():
    strategy = MDEMADV2Strategy(cooldown_hours=2.0)
    strategy.closing_order_ids.add("order-1")

    trade = DummyTrade(
        order_id="order-1",
        pnl=-10.0,
        timestamp=datetime(2024, 2, 1, 10, 0, 0),  # naive
    )
    strategy.on_fill(trade)

    assert strategy.cooldown_until is not None
    assert strategy.cooldown_until.tzinfo is not None
    assert strategy.cooldown_until == datetime(2024, 2, 1, 12, 0, 0, tzinfo=timezone.utc)
