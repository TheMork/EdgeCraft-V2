from dataclasses import dataclass, field
from typing import List, Dict
from .models import Position

@dataclass
class BinanceTier:
    limit: float
    maintenance_margin_rate: float
    maintenance_amount: float

class LiquidationEngine:
    def __init__(self):
        # Simplified Binance Tiers for BTCUSDT
        # Ensure tiers are sorted by limit
        self.tiers = [
            BinanceTier(limit=50_000, maintenance_margin_rate=0.004, maintenance_amount=0),
            BinanceTier(limit=250_000, maintenance_margin_rate=0.005, maintenance_amount=50),
            BinanceTier(limit=1_000_000, maintenance_margin_rate=0.01, maintenance_amount=1300),
            BinanceTier(limit=5_000_000, maintenance_margin_rate=0.025, maintenance_amount=16300),
            BinanceTier(limit=20_000_000, maintenance_margin_rate=0.05, maintenance_amount=141300),
        ]

    def _get_tier(self, notional_value: float) -> BinanceTier:
        abs_notional = abs(notional_value)
        for tier in self.tiers:
            if abs_notional <= tier.limit:
                return tier
        return self.tiers[-1]

    def calculate_maintenance_margin(self, position: Position, current_price: float) -> float:
        """Calculates the maintenance margin required for a position."""
        notional_value = position.size * current_price
        abs_notional = abs(notional_value)

        tier = self._get_tier(notional_value)

        # Maintenance Margin = Notional Value * Rate - Maintenance Amount
        return (abs_notional * tier.maintenance_margin_rate) - tier.maintenance_amount

    def calculate_unrealized_pnl(self, position: Position, current_price: float) -> float:
        """Calculates unrealized PnL."""
        if position.size == 0:
            return 0.0

        # PnL = (Current Price - Entry Price) * Size
        # If Long (Size > 0): (High - Low) * Pos -> Profit
        # If Short (Size < 0): (Low - High) * Neg -> (Neg) * Neg -> Profit
        return (current_price - position.entry_price) * position.size

    def check_liquidation(self, wallet_balance: float, positions: List[Position], current_prices: Dict[str, float]) -> bool:
        """
        Checks if the account should be liquidated.
        Liquidation occurs when Margin Balance < Maintenance Margin.
        Margin Balance = Wallet Balance + Unrealized PnL
        """
        total_maintenance_margin = 0.0
        total_unrealized_pnl = 0.0

        for position in positions:
            if position.symbol not in current_prices:
                # In a real system, we might use the last known price or mark price
                continue

            price = current_prices[position.symbol]

            mm = self.calculate_maintenance_margin(position, price)
            total_maintenance_margin += mm

            pnl = self.calculate_unrealized_pnl(position, price)
            total_unrealized_pnl += pnl

        margin_balance = wallet_balance + total_unrealized_pnl

        return margin_balance < total_maintenance_margin
