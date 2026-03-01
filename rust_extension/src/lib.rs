use pyo3::prelude::*;
use pyo3::types::PyAny;
use pyo3::Py;
use std::collections::{BinaryHeap, HashMap};
use std::sync::Mutex;

mod models;
mod liquidation;
mod broker;

use models::{Event, Order, Trade, Position};
use liquidation::LiquidationEngine;
use broker::Broker;

// EventLoop implementation
struct EventLoopState {
    queue: BinaryHeap<Event>,
    handlers: HashMap<i32, Vec<Py<PyAny>>>,
    current_time: Option<i64>,
    processed_events_count: usize,
}

#[pyclass]
struct EventLoop {
    state: Mutex<EventLoopState>,
    latency_ms: i64,
}

#[pymethods]
impl EventLoop {
    #[new]
    #[pyo3(signature = (latency_ms=200))]
    fn new(latency_ms: i64) -> Self {
        EventLoop {
            state: Mutex::new(EventLoopState {
                queue: BinaryHeap::new(),
                handlers: HashMap::new(),
                current_time: None,
                processed_events_count: 0,
            }),
            latency_ms,
        }
    }

    fn add_event(&self, event: &Event, py: Python<'_>) {
        let event_clone = event.clone_ref(py);
        self.state.lock().unwrap().queue.push(event_clone);
    }

    fn subscribe(&self, event_type: i32, handler: Py<PyAny>) {
        self.state.lock().unwrap().handlers.entry(event_type).or_default().push(handler);
    }

    #[getter]
    fn processed_events_count(&self) -> usize {
        self.state.lock().unwrap().processed_events_count
    }

    #[getter]
    fn current_time<'py>(&self, py: Python<'py>) -> PyResult<Option<Bound<'py, PyAny>>> {
        let state = self.state.lock().unwrap();
        match state.current_time {
            Some(ts_micros) => {
                 let ts_float = ts_micros as f64 / 1_000_000.0;
                 let datetime_mod = py.import("datetime")?;
                 let dt_class = datetime_mod.getattr("datetime")?;
                 let tz_mod = datetime_mod.getattr("timezone")?;
                 let utc = tz_mod.getattr("utc")?;
                 Ok(Some(dt_class.call_method1("fromtimestamp", (ts_float, utc))?))
            },
            None => Ok(None)
        }
    }

    fn run(&self, py: Python<'_>) -> PyResult<()> {
        loop {
            let event = {
                let mut state = self.state.lock().unwrap();
                state.queue.pop()
            };

            let event = match event {
                Some(e) => e,
                None => break,
            };

            {
                let mut state = self.state.lock().unwrap();
                 if let Some(current_time) = state.current_time {
                    if event.timestamp_micros < current_time {
                        return Err(pyo3::exceptions::PyValueError::new_err(format!(
                            "Time travel detected! Event time {} < Current time {}",
                            event.timestamp_micros, current_time
                        )));
                    }
                }
                state.current_time = Some(event.timestamp_micros);
                state.processed_events_count += 1;
            }

            let handlers: Option<Vec<Py<PyAny>>> = {
                let state = self.state.lock().unwrap();
                if let Some(handlers_vec) = state.handlers.get(&event.r#type) {
                    Some(handlers_vec.iter().map(|h: &Py<PyAny>| h.clone_ref(py)).collect())
                } else {
                    None
                }
            };

            if let Some(handlers) = handlers {
                 let py_event: Py<Event> = Py::new(py, event.clone_ref(py))?;
                 for handler in handlers {
                     handler.call1(py, (py_event.clone_ref(py),))?;
                 }
            }
        }
        Ok(())
    }

    #[pyo3(signature = (event, delay_ms=None))]
    fn schedule_delayed(&self, event: &Event, delay_ms: Option<i64>, py: Python<'_>) -> PyResult<()> {
        let delay = delay_ms.unwrap_or(self.latency_ms);
        let delay_micros = delay * 1000;

        let new_timestamp = event.timestamp_micros + delay_micros;

        let new_event = Event {
            timestamp_micros: new_timestamp,
            r#type: event.r#type,
            payload: event.payload.clone_ref(py),
        };

        self.state.lock().unwrap().queue.push(new_event);
        Ok(())
    }
}

#[pymodule]
fn simulation_core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Event>()?;
    m.add_class::<Order>()?;
    m.add_class::<Trade>()?;
    m.add_class::<Position>()?;
    m.add_class::<LiquidationEngine>()?;
    m.add_class::<Broker>()?;
    m.add_class::<EventLoop>()?;
    Ok(())
}
