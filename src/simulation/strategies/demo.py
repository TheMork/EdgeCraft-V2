from src.simulation.strategy import Strategy
from typing import Dict, Any

class DemoStrategy(Strategy):
    """
    A simple demo strategy that trades based on candle color.
    """
    def on_start(self):
        print("DemoStrategy: Simulation started.")

    def on_bar(self, bar: Dict[str, Any]):
        symbol = bar.get('symbol', 'BTC/USDT')
        close = bar.get('close')
        open_price = bar.get('open')
        timestamp = bar.get('timestamp')

        if not (close and open_price and timestamp):
            return

        # Use helper to get position size (float)
        current_position_size = self.get_position_size(symbol)

        if close > open_price and current_position_size == 0:
             # Buy 0.1 unit
             quantity = 0.1
             # Check balance? Strategy.buy checks it via Broker.
             # We assume balance is tracked by Broker now.
             # Note: DemoStrategy accessed self.balance directly before.
             # Now self.balance is property delegating to broker.
             if self.balance > (quantity * close):
                self.buy(symbol, quantity, close, timestamp)

        elif close < open_price and current_position_size > 0:
             # Sell all
             self.sell(symbol, current_position_size, close, timestamp)

    def on_event(self, event: Any):
        pass

    def on_stop(self):
        print("DemoStrategy: Simulation stopped.")
