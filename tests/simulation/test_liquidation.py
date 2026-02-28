import pytest
from src.simulation.liquidation import LiquidationEngine, Position

class TestLiquidationEngine:
    @pytest.fixture
    def engine(self):
        return LiquidationEngine()

    def test_maintenance_margin_tier1(self, engine):
        # Tier 1: 0-50k, 0.4%, amount 0
        # Position: 1 BTC @ 10,000 = 10,000 Notional
        # MM = 10,000 * 0.004 - 0 = 40
        pos = Position("BTCUSDT", 1, 10000)
        mm = engine.calculate_maintenance_margin(pos, 10000)
        assert mm == 40.0

    def test_maintenance_margin_tier2(self, engine):
        # Tier 2: 50k-250k, 0.5%, amount 50
        # Position: 6 BTC @ 10,000 = 60,000 Notional
        # MM = 60,000 * 0.005 - 50 = 300 - 50 = 250
        pos = Position("BTCUSDT", 6, 10000)
        mm = engine.calculate_maintenance_margin(pos, 10000)
        assert mm == 250.0

    def test_unrealized_pnl_long(self, engine):
        # Long: Entry 10000, Current 11000 -> +1000 * Size
        pos = Position("BTCUSDT", 2, 10000)
        pnl = engine.calculate_unrealized_pnl(pos, 11000)
        assert pnl == 2000.0

        # Loss
        pnl = engine.calculate_unrealized_pnl(pos, 9000)
        assert pnl == -2000.0

    def test_unrealized_pnl_short(self, engine):
        # Short: Entry 10000, Current 9000 -> +1000 * |Size|
        pos = Position("BTCUSDT", -2, 10000)
        pnl = engine.calculate_unrealized_pnl(pos, 9000)
        assert pnl == 2000.0

        # Loss: Current 11000 -> -1000 * |Size|
        pnl = engine.calculate_unrealized_pnl(pos, 11000)
        assert pnl == -2000.0

    def test_liquidation_check(self, engine):
        # Wallet: 100
        # Position: 1 BTC @ 10000. MM = 40.
        # Price drops to 9900. PnL = (9900-10000)*1 = -100.
        # Margin Balance = 100 - 100 = 0.
        # 0 < 40 -> Liquidate True
        pos = Position("BTCUSDT", 1, 10000)
        assert engine.check_liquidation(100, [pos], {"BTCUSDT": 9900}) is True

        # Price 9950. PnL = -50.
        # Margin Balance = 50.
        # 50 > 40 -> Liquidate False
        assert engine.check_liquidation(100, [pos], {"BTCUSDT": 9950}) is False
