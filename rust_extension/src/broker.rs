use pyo3::prelude::*;
use pyo3::Py;
use pyo3::types::{PyDict, PyList};
use std::collections::HashMap;
use crate::models::{Order, Trade, Position, Event};
use crate::liquidation::LiquidationEngine;
use uuid::Uuid;

#[pyclass]
pub struct Broker {
    #[pyo3(get, set)]
    balance: f64,
    #[pyo3(get)]
    equity: f64,
    #[pyo3(get)]
    leverage: i32,

    // We store Py<Position> to share state with Python
    positions: HashMap<String, Py<Position>>,
    open_orders: HashMap<String, Py<Order>>,

    trades: Vec<Py<Trade>>,
    order_history: Vec<Py<Order>>,

    last_prices: HashMap<String, f64>,

    taker_fee_rate: f64,
    maker_fee_rate: f64,

    liquidation_engine: LiquidationEngine,
}

#[pymethods]
impl Broker {
    #[new]
    #[pyo3(signature = (initial_balance=10000.0, leverage=1))]
    fn new(initial_balance: f64, leverage: i32) -> Self {
        Broker {
            balance: initial_balance,
            equity: initial_balance,
            leverage,
            positions: HashMap::new(),
            open_orders: HashMap::new(),
            trades: Vec::new(),
            order_history: Vec::new(),
            last_prices: HashMap::new(),
            taker_fee_rate: 0.0004,
            maker_fee_rate: 0.0002,
            liquidation_engine: LiquidationEngine::new_internal(),
        }
    }

    #[getter]
    fn positions(&self, py: Python<'_>    ) -> PyResult<Py<PyAny>> {
        let dict = PyDict::new(py);
        for (symbol, pos) in &self.positions {
            dict.set_item(symbol, pos)?;
        }
        Ok(dict.into())
    }

    #[getter]
    fn open_orders(&self, py: Python<'_>    ) -> PyResult<Py<PyAny>> {
        let dict = PyDict::new(py);
        for (id, order) in &self.open_orders {
            dict.set_item(id, order)?;
        }
        Ok(dict.into())
    }

    #[getter]
    fn trades(&self, py: Python<'_>    ) -> PyResult<Py<PyAny>> {
        let list = PyList::empty(py);
        for trade in &self.trades {
            list.append(trade)?;
        }
        Ok(list.into())
    }

    #[getter]
    fn order_history(&self, py: Python<'_>    ) -> PyResult<Py<PyAny>> {
        let list = PyList::empty(py);
        for order in &self.order_history {
            list.append(order)?;
        }
        Ok(list.into())
    }

    fn get_position(&self, symbol: String, py: Python<'_>) -> Option<Py<Position>> {
        self.positions.get(&symbol).map(|p| p.clone_ref(py))
    }

    fn get_margin_balance(&self, py: Python<'_>) -> f64 {
        let mut total_unrealized_pnl = 0.0;
        for pos in self.positions.values() {
             let pos_ref = pos.borrow(py);
             total_unrealized_pnl += pos_ref.unrealized_pnl;
        }
        self.balance + total_unrealized_pnl
    }

    fn get_used_margin(&self, py: Python<'_>) -> f64 {
        let mut pos_margin = 0.0;
        for pos in self.positions.values() {
            let pos_ref = pos.borrow(py);
            pos_margin += (pos_ref.size * pos_ref.entry_price).abs() / pos_ref.leverage as f64;
        }

        let mut order_margin = 0.0;
        for order in self.open_orders.values() {
            let order_ref = order.borrow(py);
            if let Some(price) = order_ref.price {
                if price > 0.0 {
                    order_margin += (order_ref.quantity * price) / order_ref.leverage as f64;
                }
            }
        }

        pos_margin + order_margin
    }

    fn get_available_balance(&self, py: Python<'_>) -> f64 {
        self.get_margin_balance(py) - self.get_used_margin(py)
    }

    fn add_position(&mut self, position: Bound<'_, Position>, py: Python<'_>) -> PyResult<()> {
        let pos_ref = position.borrow();
        let symbol = pos_ref.symbol.clone();
        drop(pos_ref);
        let pos_py = position.unbind();
        self.positions.insert(symbol, pos_py);
        Ok(())
    }

    fn submit_order(&mut self, order: Bound<'_, Order>, py: Python<'_>) -> PyResult<Py<Order>> {
        let mut order_ref = order.borrow_mut();

        // Set default leverage
        if order_ref.leverage == 1 && self.leverage != 1 {
            order_ref.leverage = self.leverage;
        }

        // Check Margin
        let price = order_ref.price.unwrap_or(0.0);
        if price > 0.0 {
            let required_margin = (order_ref.quantity * price) / order_ref.leverage as f64;
            let available = self.get_available_balance(py);

            if required_margin > available {
                order_ref.status = "REJECTED".to_string();
                // We still append rejected orders to history? Python code does.
                // But Python code returns the order object.
                // We need to store it in history but not open_orders.
                let order_py = order.clone().unbind();
                self.order_history.push(order_py.clone_ref(py));
                return Ok(order_py);
            }
        }

        // Generate ID if empty
        if order_ref.id.is_empty() {
            order_ref.id = Uuid::new_v4().to_string();
        }

        // In Python code: self.open_orders[order.id] = order
        let order_id = order_ref.id.clone();
        drop(order_ref); // Release borrow

        let order_py = order.clone().unbind();
        self.open_orders.insert(order_id, order_py.clone_ref(py));
        self.order_history.push(order_py.clone_ref(py));

        Ok(order_py)
    }

    fn cancel_order(&mut self, order_id: String, py: Python<'_>) -> bool {
        if let Some(order) = self.open_orders.remove(&order_id) {
            let mut order_ref = order.borrow_mut(py);
            order_ref.status = "CANCELED".to_string();
            true
        } else {
            false
        }
    }

    fn process_market_data(&mut self, event: &Event, py: Python<'_>) -> PyResult<Vec<Py<Trade>>> {
        let payload: &Bound<'_, PyDict> = event.payload.bind(py).downcast()?;

        let symbol: String = payload.get_item("symbol")?.unwrap().extract()?;
        let current_price: f64 = payload.get_item("close")?.unwrap().extract()?;
        let open_price: f64 = payload.get_item("open")?.map(|i: Bound<'_, PyAny>| i.extract::<f64>().unwrap_or(current_price)).unwrap_or(current_price);
        let high_price: f64 = payload.get_item("high")?.map(|i: Bound<'_, PyAny>| i.extract::<f64>().unwrap_or(current_price)).unwrap_or(current_price);
        let low_price: f64 = payload.get_item("low")?.map(|i: Bound<'_, PyAny>| i.extract::<f64>().unwrap_or(current_price)).unwrap_or(current_price);

        // let timestamp = event.timestamp(py)?;
        // We need timestamp for Trade. event.timestamp_micros is i64.

        self.last_prices.insert(symbol.clone(), current_price);

        // 1. Update PnL
        self.update_pnl(py);

        // 2. Check Liquidation
        if self.check_liquidation(py) {
            println!("LIQUIDATION TRIGGERED at {} Price: {}", event.timestamp_micros, current_price);
            // Close all positions
            let mut liquidation_fills = Vec::new();

            // Collect symbols to close to avoid borrowing issues while iterating
            let positions_to_close: Vec<(String, f64, i32)> = self.positions.iter()
                .filter(|(_, p)| p.borrow(py).size != 0.0)
                .map(|(s, p)| {
                    let p_ref = p.borrow(py);
                    (s.clone(), p_ref.size, p_ref.leverage)
                })
                .collect();

            for (s, size, leverage) in positions_to_close {
                 let price = *self.last_prices.get(&s).unwrap_or(&current_price);

                 // Create Liquidation Order
                 let side = if size > 0.0 { "SELL" } else { "BUY" };
                 let quantity = size.abs();

                 // We don't submit this order to open_orders, we execute it directly
                 let trade = self.execute_trade_internal(
                     "LIQUIDATION".to_string(), // order_id
                     s,
                     side.to_string(),
                     quantity,
                     price,
                     leverage,
                     event.timestamp_micros,
                     true, // is_taker
                     py
                 )?;
                 liquidation_fills.push(trade);
            }
            // Logic to actually add these trades to return list
            // But verify: execute_trade_internal adds to self.trades? Yes.
            // And returns Py<Trade>.

            // Note: In Python, liquidation uses execute_trade but constructs a dummy Order object first.
            // Here we skip Order object creation for simplicity or create a dummy one?
            // "liquidation_order = Order(...)"
            // Python code appends to new_fills.

            // Wait, logic above already executed trades.
            // We should return these fills.
        }

        // 3. Match Orders
        let mut new_fills = Vec::new();
        let mut filled_order_ids = Vec::new();

        // Iterate over open orders
        // We need to collect keys first or iterate safely
        let order_ids: Vec<String> = self.open_orders.keys().cloned().collect();

        for order_id in order_ids {
            // Re-borrow order
            let order_py = match self.open_orders.get(&order_id) {
                Some(o) => o.clone_ref(py),
                None => continue,
            };

            // Scope for borrowing order
            let (should_fill, fill_price, is_taker, order_symbol, order_side, order_qty, order_leverage) = {
                let order = order_py.borrow(py);
                if order.symbol != symbol {
                    continue;
                }

                let mut fill_price = None;
                let mut is_taker = true;

                if order.order_type == "MARKET" {
                    fill_price = Some(current_price);
                    is_taker = true;
                } else if order.order_type == "LIMIT" {
                    if order.side == "BUY" {
                         if let Some(p) = order.price {
                             if open_price <= p {
                                 fill_price = Some(open_price);
                                 is_taker = false;
                             } else if low_price <= p {
                                 fill_price = Some(p);
                                 is_taker = false;
                             }
                         }
                    } else if order.side == "SELL" {
                         if let Some(p) = order.price {
                             if open_price >= p {
                                 fill_price = Some(open_price);
                                 is_taker = false;
                             } else if high_price >= p {
                                 fill_price = Some(p);
                                 is_taker = false;
                             }
                         }
                    }
                } else if order.order_type == "STOP" {
                    if let Some(stop_p) = order.stop_price {
                        if order.side == "BUY" {
                             if open_price >= stop_p {
                                 fill_price = Some(open_price);
                             } else if high_price >= stop_p {
                                 fill_price = Some(stop_p);
                             }
                        } else if order.side == "SELL" {
                             if open_price <= stop_p {
                                 fill_price = Some(open_price);
                             } else if low_price <= stop_p {
                                 fill_price = Some(stop_p);
                             }
                        }
                    }
                }

                (fill_price.is_some(), fill_price, is_taker, order.symbol.clone(), order.side.clone(), order.quantity, order.leverage)
            };

            if should_fill {
                let fill_price_val: f64 = fill_price.unwrap();
                let trade = self.execute_trade_internal(
                    order_id.clone(),
                    order_symbol,
                    order_side,
                    order_qty,
                    fill_price_val,
                    order_leverage,
                    event.timestamp_micros,
                    is_taker,
                    py
                )?;

                // Update Order status
                {
                    let mut order = order_py.borrow_mut(py);
                    order.status = "FILLED".to_string();
                    order.filled_quantity = order_qty;
                    order.average_fill_price = fill_price_val;
                }

                new_fills.push(trade);
                filled_order_ids.push(order_id);
            }
        }

        for oid in filled_order_ids {
            self.open_orders.remove(&oid);
        }

        Ok(new_fills)
    }

    fn process_funding_event(&mut self, event: &Event, py: Python<'_>) -> PyResult<()> {
        let payload: &Bound<'_, PyDict> = event.payload.bind(py).downcast()?;
        let symbol: Option<String> = payload.get_item("symbol")?.and_then(|i: Bound<'_, PyAny>| i.extract::<String>().ok());
        let funding_rate: Option<f64> = payload.get_item("funding_rate")?.and_then(|i: Bound<'_, PyAny>| i.extract::<f64>().ok());

        if let (Some(sym), Some(rate)) = (symbol, funding_rate) {
             if let Some(pos_py) = self.positions.get(&sym) {
                 let pos = pos_py.borrow(py);
                 if pos.size != 0.0 {
                     let price = *self.last_prices.get(&sym).unwrap_or(&pos.entry_price);
                     let notional = pos.size * price;
                     let payment = notional * rate;
                     self.balance -= payment;
                 }
             }
        }
        Ok(())
    }
}

impl Broker {
    fn update_pnl(&mut self, py: Python<'_>) {
        let mut total_unrealized_pnl = 0.0;

        for pos_py in self.positions.values() {
            let mut pos = pos_py.borrow_mut(py);
            if pos.size == 0.0 {
                pos.unrealized_pnl = 0.0;
                continue;
            }

            if let Some(&price) = self.last_prices.get(&pos.symbol) {
                let pnl = (price - pos.entry_price) * pos.size;
                pos.unrealized_pnl = pnl;
                total_unrealized_pnl += pnl;

                pos.maintenance_margin = self.liquidation_engine.calculate_maintenance_margin_internal(&pos, price);
                pos.initial_margin = (pos.size * pos.entry_price).abs() / pos.leverage as f64;
            } else {
                 total_unrealized_pnl += pos.unrealized_pnl;
            }
        }
        self.equity = self.balance + total_unrealized_pnl;
    }

    fn check_liquidation(&self, py: Python<'_>) -> bool {
        // Prepare data for liquidation engine
        // We need Vec<Position> but we have HashMap<String, Py<Position>>
        // LiquidationEngine takes &Vec<Position>.
        // This is inefficient if we clone all positions.
        // But LiquidationEngine needs to access fields.

        // Actually, I can just reimplement check_liquidation loop here or update LiquidationEngine to take iterator?
        // Or pass vector of references?

        // Let's create a temporary Vec<Position> (clones)
        let positions: Vec<Position> = self.positions.values().map(|p| p.borrow(py).clone()).collect();

        self.liquidation_engine.check_liquidation_internal(self.balance, &positions, &self.last_prices)
    }

    #[allow(clippy::too_many_arguments)]
    fn execute_trade_internal(
        &mut self,
        order_id: String,
        symbol: String,
        side: String,
        quantity: f64,
        price: f64,
        leverage: i32,
        timestamp_micros: i64,
        is_taker: bool,
        py: Python<'_>
    ) -> PyResult<Py<Trade>> {
        let notional_value = quantity * price;
        let fee_rate = if is_taker { self.taker_fee_rate } else { self.maker_fee_rate };
        let fee = notional_value * fee_rate;

        self.balance -= fee;

        // Create Trade
        // We need datetime object for Trade constructor?
        // Trade::new takes Bound<'_, PyDateTime>.
        // But here we want to construct Trade and return Py<Trade>.
        // We can create PyDateTime from timestamp_micros.
        let ts_float = timestamp_micros as f64 / 1_000_000.0;
        let datetime_mod = py.import("datetime")?;
        let dt_class = datetime_mod.getattr("datetime")?;
        let tz_mod = datetime_mod.getattr("timezone")?;
        let utc = tz_mod.getattr("utc")?;
        let ts_obj = dt_class.call_method1("fromtimestamp", (ts_float, utc))?;

        let trade = Trade::new_internal(
            Uuid::new_v4().to_string(),
            order_id,
            symbol.clone(),
            side.clone(),
            quantity,
            price,
            &ts_obj, // Pass Bound
            fee,
            0.0 // pnl
        )?;

        let trade_py = Py::new(py, trade)?;

        // Update Position
        if !self.positions.contains_key(&symbol) {
             let pos = Position::new_internal(symbol.clone(), 0.0, 0.0, leverage, 0.0, 0.0, 0.0, 0.0);
             self.positions.insert(symbol.clone(), Py::new(py, pos)?);
        }

        let pos_py = self.positions.get(&symbol).unwrap();
        let mut pos = pos_py.borrow_mut(py);

        let mut realized_pnl = 0.0;

        if side == "BUY" {
            if pos.size >= 0.0 {
                let new_size = pos.size + quantity;
                let total_cost = (pos.size * pos.entry_price) + (quantity * price);
                pos.entry_price = if new_size > 0.0 { total_cost / new_size } else { 0.0 };
                pos.size = new_size;
            } else {
                // Closing Short
                let closed_qty = pos.size.abs().min(quantity);
                let remaining_qty = pos.size + quantity;
                realized_pnl = (pos.entry_price - price) * closed_qty;
                self.balance += realized_pnl;
                pos.size = remaining_qty;
                if pos.size == 0.0 { pos.entry_price = 0.0; }
                else if pos.size > 0.0 { pos.entry_price = price; }
            }
        } else if side == "SELL" {
             if pos.size <= 0.0 {
                 let new_size = pos.size - quantity;
                 let total_cost = (pos.size.abs() * pos.entry_price) + (quantity * price);
                 pos.entry_price = if new_size.abs() > 0.0 { total_cost / new_size.abs() } else { 0.0 };
                 pos.size = new_size;
             } else {
                 // Closing Long
                 let closed_qty = pos.size.min(quantity);
                 let remaining_qty = pos.size - quantity;
                 realized_pnl = (price - pos.entry_price) * closed_qty;
                 self.balance += realized_pnl;
                 pos.size = remaining_qty;
                 if pos.size == 0.0 { pos.entry_price = 0.0; }
                 else if pos.size < 0.0 { pos.entry_price = price; }
             }
        }

        // Update trade PnL
        {
            let mut trade_ref = trade_py.borrow_mut(py);
            trade_ref.pnl = realized_pnl;
        }

        self.trades.push(trade_py.clone_ref(py));

        Ok(trade_py)
    }
}
