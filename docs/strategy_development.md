# Strategy Development Guide

This guide describes how to implement a new trading strategy in the EdgeCraft Backtesting Station.

## Overview

EdgeCraft uses a drop-in strategy plugin system. A strategy is simply a single Python file placed in the `src/simulation/strategies/` directory that contains a class inheriting from the `Strategy` base class. The backend automatically detects strategies, registers them, and exposes them to the UI.

## Creating a Strategy

Use the `TEMPLATE.py` located in `src/simulation/strategies/` as your starting point.

A valid strategy must satisfy the following requirements:

1. Inherit from `src.simulation.strategy.Strategy`.
2. Define mandatory metadata variables (`NAME`, `DESCRIPTION`, `VERSION`, `SUPPORTED_TIMEFRAMES`).
3. Implement `get_param_schema()` to define parameters dynamically for the UI.
4. Implement `on_start()`, `on_bar()`, and `on_stop()`.

### Metadata Requirements

```python
class MyStrategy(Strategy):
    NAME: str = "My Custom Strategy"
    DESCRIPTION: str = "Trades on simple moving average crossovers."
    VERSION: str = "1.0.0"
    AUTHOR: str = "Your Name" # Optional
    SUPPORTED_TIMEFRAMES: list = ["1h", "4h", "1d"]
```

### Parameter Schema

The `get_param_schema()` class method returns a JSON-schema-like dictionary describing the strategy's configurable parameters. This is used by the frontend to render the configuration form automatically.

```python
    @classmethod
    def get_param_schema(cls) -> Dict[str, Any]:
        return {
            "fast_period": {
                "type": "int",
                "default": 10,
                "min": 2,
                "max": 50,
                "description": "Period for the fast moving average."
            },
            "slow_period": {
                "type": "int",
                "default": 30,
                "min": 10,
                "max": 200,
                "description": "Period for the slow moving average."
            }
        }
```

### Order Execution

Strategies submit orders through the broker simulation interface. They should *never* modify account balance or position state directly.

Available order submission methods on the `Strategy` class:
- `self.buy(symbol, quantity, price, timestamp)`
- `self.sell(symbol, quantity, price, timestamp)`

Use `self.get_position_size(symbol)` to check current position sizes.

## Testing

Ensure that you create unit tests for your strategy in `tests/strategies/`.
