# Simulation Core (Rust Extension)

This directory contains the high-performance core of the simulation engine, written in Rust and exposed to Python via PyO3.

## Prerequisites

- Rust (latest stable)
- Python 3.7+
- `maturin`

## Building

To build and install the extension in your current Python environment:

```bash
pip install maturin
cd rust_extension
maturin develop --release
```

## Usage

The module is exposed as `simulation_core`. It provides:

- `Event`: A high-performance event object.
- `EventLoop`: A priority-queue based event loop.

See `src/simulation/event_loop.py` for usage examples.
