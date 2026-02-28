# Critical Review of the Architecture

This document identifies potential technical bottlenecks in the proposed Python-based Event-Driven Backtesting Engine and suggests optimizations.

## 1. The Python Global Interpreter Lock (GIL) and Event Loop Overhead

### Bottleneck
The current architecture relies on a Python `while` loop processing `Event` objects one by one. Python's Global Interpreter Lock (GIL) ensures that only one thread executes Python bytecode at a time.
- **Impact:** The event loop is strictly single-threaded. CPU-bound tasks (like matching engine logic or complex signal generation) will block the processing of subsequent market data events.
- **Scale:** Processing 1 year of tick data for multiple symbols (millions of events) in pure Python can be prohibitively slow (hours vs minutes).

### Optimization: Rust Core with PyO3
Move the `EventLoop`, `PriorityQueue`, and `MatchingEngine` to **Rust**.
- **Implementation:** Use `PyO3` to create Python bindings. The Rust core manages the state and event flow efficiently.
- **Benefit:** Rust provides zero-cost abstractions and manual memory management. It can process the event queue order of magnitudes faster than Python. The Python `Strategy` can still be called from Rust when needed, but the high-frequency data processing happens in native code.

## 2. Object Instantiation Overhead (The "Everything is an Object" Problem)

### Bottleneck
In the current design, every market tick (Trade, Quote) is instantiated as a Python `Event` dataclass.
- **Impact:** Python object creation has significant overhead (memory allocation, reference counting). Creating millions of small objects triggers frequent Garbage Collection (GC) cycles, causing "Stop-the-World" pauses that distort performance metrics and slow down the simulation.
- **Memory Pressure:** A `dataclass` is much heavier than a simple C-struct.

### Optimization: Zero-Copy Abstraktionen & Data-Oriented Design
- **Approach:** Instead of creating objects for every tick, pass pointers or indices to a pre-loaded memory buffer.
- **Technology:**
    - **Apache Arrow:** Use Arrow memory format for market data. The engine can read directly from the Arrow buffer without deserializing into Python objects (Zero-Copy).
    - **Rust Structs:** If moving to Rust, use stack-allocated structs or arenas to manage memory, avoiding the allocator overhead.

## 3. I/O Latency for Historical Data Streaming

### Bottleneck
Fetching data from QuestDB or disk row-by-row during the simulation can introduce I/O latency that dominates the execution time.
- **Impact:** The CPU sits idle waiting for the next batch of data from the database.

### Optimization: Async Prefetching & Tokio
- **Approach:** Decouple data loading from the simulation loop.
- **Technology:** Use **Rust `tokio`** (or Python `asyncio` with separate threads) to prefetch data chunks into a ring buffer.
- **Mechanism:** While the Simulation Engine processes Chunk N, the Data Loader asynchronously fetches and parses Chunk N+1. This ensures the CPU is constantly fed with data.

## Summary of Recommendations

| Component | Current Approach | Recommended Optimization | Expected Gain |
|-----------|------------------|--------------------------|---------------|
| **Core Loop** | Python `while` loop | Rust Loop (PyO3) | 10x-100x Speedup |
| **Data Structure** | Python `dataclass` | Arrow / Rust Structs | Reduced GC Pressure |
| **Data Loading** | Synchronous Fetch | Async Prefetch (`tokio`) | I/O Hiding |
