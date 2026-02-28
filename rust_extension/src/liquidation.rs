use pyo3::prelude::*;
use std::collections::HashMap;
use crate::models::Position;

#[derive(Clone)]
struct BinanceTier {
    limit: f64,
    maintenance_margin_rate: f64,
    maintenance_amount: f64,
}

#[pyclass]
pub struct LiquidationEngine {
    tiers: Vec<BinanceTier>,
}

impl LiquidationEngine {
    pub fn new_internal() -> Self {
        LiquidationEngine {
            tiers: vec![
                BinanceTier { limit: 50_000.0, maintenance_margin_rate: 0.004, maintenance_amount: 0.0 },
                BinanceTier { limit: 250_000.0, maintenance_margin_rate: 0.005, maintenance_amount: 50.0 },
                BinanceTier { limit: 1_000_000.0, maintenance_margin_rate: 0.01, maintenance_amount: 1_300.0 },
                BinanceTier { limit: 5_000_000.0, maintenance_margin_rate: 0.025, maintenance_amount: 16_300.0 },
                BinanceTier { limit: 20_000_000.0, maintenance_margin_rate: 0.05, maintenance_amount: 141_300.0 },
            ],
        }
    }

    fn get_tier(&self, notional_value: f64) -> &BinanceTier {
        let abs_notional = notional_value.abs();
        for tier in &self.tiers {
            if abs_notional <= tier.limit {
                return tier;
            }
        }
        self.tiers.last().unwrap()
    }

    pub fn calculate_maintenance_margin_internal(&self, position: &Position, current_price: f64) -> f64 {
        let notional_value = position.size * current_price;
        let tier = self.get_tier(notional_value);

        (notional_value.abs() * tier.maintenance_margin_rate) - tier.maintenance_amount
    }

    pub fn calculate_unrealized_pnl(&self, position: &Position, current_price: f64) -> f64 {
        if position.size == 0.0 {
            return 0.0;
        }
        (current_price - position.entry_price) * position.size
    }

    pub fn check_liquidation_internal(&self, wallet_balance: f64, positions: &Vec<Position>, current_prices: &HashMap<String, f64>) -> bool {
        let mut total_maintenance_margin = 0.0;
        let mut total_unrealized_pnl = 0.0;

        for position in positions {
            if let Some(price) = current_prices.get(&position.symbol) {
                let mm = self.calculate_maintenance_margin_internal(position, *price);
                total_maintenance_margin += mm;

                let pnl = self.calculate_unrealized_pnl(position, *price);
                total_unrealized_pnl += pnl;
            }
        }

        let margin_balance = wallet_balance + total_unrealized_pnl;
        margin_balance < total_maintenance_margin
    }
}

#[pymethods]
impl LiquidationEngine {
    #[new]
    fn new() -> Self {
        Self::new_internal()
    }

    fn calculate_maintenance_margin(&self, position: &Position, current_price: f64) -> f64 {
        self.calculate_maintenance_margin_internal(position, current_price)
    }

    fn check_liquidation(&self, wallet_balance: f64, positions: Vec<Position>, current_prices: HashMap<String, f64>) -> bool {
        self.check_liquidation_internal(wallet_balance, &positions, &current_prices)
    }
}
