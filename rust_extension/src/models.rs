use pyo3::prelude::*;
use pyo3::types::PyAny;
use pyo3::Py;
use std::cmp::Ordering;

#[pyclass]
#[derive(Debug)]
pub struct Event {
    pub timestamp_micros: i64,
    #[pyo3(get)]
    pub r#type: i32,
    #[pyo3(get)]
    pub payload: Py<PyAny>,
}

impl Ord for Event {
    fn cmp(&self, other: &Self) -> Ordering {
        other.timestamp_micros.cmp(&self.timestamp_micros)
            .then_with(|| other.r#type.cmp(&self.r#type))
    }
}

impl PartialOrd for Event {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl PartialEq for Event {
    fn eq(&self, other: &Self) -> bool {
        self.timestamp_micros == other.timestamp_micros && self.r#type == other.r#type
    }
}

impl Eq for Event {}

#[pymethods]
impl Event {
    #[new]
    fn new(timestamp: &Bound<'_, PyAny>, r#type: i32, payload: Py<PyAny>) -> PyResult<Self> {
        let ts_float: f64 = timestamp.call_method0("timestamp")?.extract()?;
        let timestamp_micros = (ts_float * 1_000_000.0) as i64;

        Ok(Event {
            timestamp_micros,
            r#type,
            payload,
        })
    }

    #[getter]
    fn timestamp<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let ts_float = self.timestamp_micros as f64 / 1_000_000.0;
        let datetime_mod = py.import("datetime")?;
        let dt_class = datetime_mod.getattr("datetime")?;
        let tz_mod = datetime_mod.getattr("timezone")?;
        let utc = tz_mod.getattr("utc")?;
        dt_class.call_method1("fromtimestamp", (ts_float, utc))
    }

    fn __repr__(&self) -> String {
        format!("Event(timestamp={}, type={}, payload={})", self.timestamp_micros, self.r#type, self.payload)
    }

    fn __lt__(&self, other: &Event) -> bool {
        self.cmp(other) == Ordering::Greater
    }
}

impl Event {
    pub fn clone_ref(&self, py: Python<'_>) -> Self {
        Event {
            timestamp_micros: self.timestamp_micros,
            r#type: self.r#type,
            payload: self.payload.clone_ref(py),
        }
    }
}

#[pyclass]
#[derive(Clone, PartialEq, Debug)]
pub struct Order {
    #[pyo3(get, set)]
    pub id: String,
    #[pyo3(get, set)]
    pub symbol: String,
    #[pyo3(get, set)]
    pub side: String,
    #[pyo3(get, set)]
    pub order_type: String,
    #[pyo3(get, set)]
    pub quantity: f64,
    #[pyo3(get, set)]
    pub price: Option<f64>,
    #[pyo3(get, set)]
    pub stop_price: Option<f64>,
    pub timestamp_micros: i64,
    #[pyo3(get, set)]
    pub status: String,
    #[pyo3(get, set)]
    pub filled_quantity: f64,
    #[pyo3(get, set)]
    pub average_fill_price: f64,
    #[pyo3(get, set)]
    pub leverage: i32,
}

#[pymethods]
impl Order {
    #[new]
    #[pyo3(signature = (id, symbol, side, order_type, quantity, price=None, stop_price=None, timestamp=None, status="NEW".to_string(), filled_quantity=0.0, average_fill_price=0.0, leverage=1))]
    fn new(
        py: Python<'_>,
        id: String,
        symbol: String,
        side: String,
        order_type: String,
        quantity: f64,
        price: Option<f64>,
        stop_price: Option<f64>,
        timestamp: Option<Bound<'_, PyAny>>,
        status: String,
        filled_quantity: f64,
        average_fill_price: f64,
        leverage: i32,
    ) -> PyResult<Self> {
        let timestamp_micros = if let Some(ts) = timestamp {
            let ts_float: f64 = ts.call_method0("timestamp")?.extract()?;
            (ts_float * 1_000_000.0) as i64
        } else {
            let new_event_payload: Py<PyAny> = py.None().into();
            let now_obj = py.import("datetime")?.getattr("datetime")?.call_method1("now", (py.import("datetime")?.getattr("timezone")?.getattr("utc")?,))?;
            let ts_float: f64 = now_obj.call_method0("timestamp")?.extract()?;
            (ts_float * 1_000_000.0) as i64
        };

        Ok(Order {
            id,
            symbol,
            side,
            order_type,
            quantity,
            price,
            stop_price,
            timestamp_micros,
            status,
            filled_quantity,
            average_fill_price,
            leverage,
        })
    }

    #[getter]
    fn timestamp<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let ts_float = self.timestamp_micros as f64 / 1_000_000.0;
        let datetime_mod = py.import("datetime")?;
        let dt_class = datetime_mod.getattr("datetime")?;
        let tz_mod = datetime_mod.getattr("timezone")?;
        let utc = tz_mod.getattr("utc")?;
        dt_class.call_method1("fromtimestamp", (ts_float, utc))
    }

    fn __repr__(&self) -> String {
        format!("Order(id={}, symbol={}, side={}, type={}, quantity={}, status={})",
            self.id, self.symbol, self.side, self.order_type, self.quantity, self.status)
    }
}

#[pyclass]
#[derive(Clone, PartialEq, Debug)]
pub struct Trade {
    #[pyo3(get, set)]
    pub id: String,
    #[pyo3(get, set)]
    pub order_id: String,
    #[pyo3(get, set)]
    pub symbol: String,
    #[pyo3(get, set)]
    pub side: String,
    #[pyo3(get, set)]
    pub quantity: f64,
    #[pyo3(get, set)]
    pub price: f64,
    pub timestamp_micros: i64,
    #[pyo3(get, set)]
    pub fee: f64,
    #[pyo3(get, set)]
    pub pnl: f64,
}

impl Trade {
    pub fn new_internal(
        id: String,
        order_id: String,
        symbol: String,
        side: String,
        quantity: f64,
        price: f64,
        timestamp: &Bound<'_, PyAny>,
        fee: f64,
        pnl: f64,
    ) -> PyResult<Self> {
        let ts_float: f64 = timestamp.call_method0("timestamp")?.extract()?;
        let timestamp_micros = (ts_float * 1_000_000.0) as i64;

        Ok(Trade {
            id,
            order_id,
            symbol,
            side,
            quantity,
            price,
            timestamp_micros,
            fee,
            pnl,
        })
    }
}

#[pymethods]
impl Trade {
    #[new]
    fn new(
        id: String,
        order_id: String,
        symbol: String,
        side: String,
        quantity: f64,
        price: f64,
        timestamp: &Bound<'_, PyAny>,
        fee: f64,
        pnl: f64,
    ) -> PyResult<Self> {
        Self::new_internal(id, order_id, symbol, side, quantity, price, timestamp, fee, pnl)
    }

    #[getter]
    fn timestamp<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let ts_float = self.timestamp_micros as f64 / 1_000_000.0;
        let datetime_mod = py.import("datetime")?;
        let dt_class = datetime_mod.getattr("datetime")?;
        let tz_mod = datetime_mod.getattr("timezone")?;
        let utc = tz_mod.getattr("utc")?;
        dt_class.call_method1("fromtimestamp", (ts_float, utc))
    }

    fn __repr__(&self) -> String {
        format!("Trade(id={}, symbol={}, side={}, quantity={}, price={}, pnl={})",
            self.id, self.symbol, self.side, self.quantity, self.price, self.pnl)
    }
}

#[pyclass]
#[derive(Clone, PartialEq, Debug)]
pub struct Position {
    #[pyo3(get, set)]
    pub symbol: String,
    #[pyo3(get, set)]
    pub size: f64,
    #[pyo3(get, set)]
    pub entry_price: f64,
    #[pyo3(get, set)]
    pub leverage: i32,
    #[pyo3(get, set)]
    pub unrealized_pnl: f64,
    #[pyo3(get, set)]
    pub liquidation_price: f64,
    #[pyo3(get, set)]
    pub initial_margin: f64,
    #[pyo3(get, set)]
    pub maintenance_margin: f64,
}

impl Position {
    #[allow(clippy::too_many_arguments)]
    pub fn new_internal(
        symbol: String,
        size: f64,
        entry_price: f64,
        leverage: i32,
        unrealized_pnl: f64,
        liquidation_price: f64,
        initial_margin: f64,
        maintenance_margin: f64,
    ) -> Self {
        Position {
            symbol,
            size,
            entry_price,
            leverage,
            unrealized_pnl,
            liquidation_price,
            initial_margin,
            maintenance_margin,
        }
    }
}

#[pymethods]
impl Position {
    #[new]
    #[pyo3(signature = (symbol, size, entry_price, leverage=1, unrealized_pnl=0.0, liquidation_price=0.0, initial_margin=0.0, maintenance_margin=0.0))]
    fn new(
        symbol: String,
        size: f64,
        entry_price: f64,
        leverage: i32,
        unrealized_pnl: f64,
        liquidation_price: f64,
        initial_margin: f64,
        maintenance_margin: f64,
    ) -> Self {
        Self::new_internal(symbol, size, entry_price, leverage, unrealized_pnl, liquidation_price, initial_margin, maintenance_margin)
    }

    fn __repr__(&self) -> String {
        format!("Position(symbol={}, size={}, entry_price={}, leverage={}, upnl={})",
            self.symbol, self.size, self.entry_price, self.leverage, self.unrealized_pnl)
    }
}
