from src.simulation.strategy import Strategy
from typing import Dict, Any

class TemplateStrategy(Strategy):
    """
    Template for creating a new trading strategy.
    Duplicate this file and fill in your custom logic.
    """
    # Jede Strategie MUSS diese Klassenvariablen definieren
    NAME: str = "Template Strategy"
    DESCRIPTION: str = "A template for new strategies."
    VERSION: str = "1.0.0"
    AUTHOR: str = "EdgeCraft"
    SUPPORTED_TIMEFRAMES: list = ["1h", "4h", "1d"]

    def __init__(self, my_param: int = 10, **kwargs):
        super().__init__()
        self.my_param = my_param

    @classmethod
    def get_param_schema(cls) -> Dict[str, Any]:
        """
        Gibt JSON-Schema zurück, das alle __init__-Parameter beschreibt.
        """
        return {
            "my_param": {
                "type": "int",
                "default": 10,
                "min": 1,
                "max": 100,
                "description": "Example parameter"
            }
        }

    def on_start(self):
        """Called when the simulation starts."""
        pass

    def on_bar(self, bar: Dict[str, Any]):
        """Called when a new candle/bar is closed."""
        symbol = bar.get('symbol')
        close = bar.get('close')
        timestamp = bar.get('timestamp')

        # Example logic:
        # if some_condition:
        #    self.buy(symbol, 1.0, close, timestamp)
        pass

    def on_stop(self):
        """Called when simulation ends."""
        pass
