import sys
import pandas as pd
import numpy as np
from datetime import datetime
from typing import List, Dict, Any, Type
from src.database import QuestDBManager
from src.mock_database import MockQuestDBManager
from src.simulation.event_loop import EventLoop, Event, EventType
from src.simulation.strategy import Strategy
from src.simulation.metrics import calculate_metrics
from src.simulation.broker import Broker
from src.simulation.runner import SimulationRunner

class MultiAssetSimulationRunner:
    def __init__(
        self,
        strategy_class: Type[Strategy],
        symbols: List[str],
        start_date: str,
        end_date: str,
        timeframe: str = "1m",
        db_host: str = 'localhost',
        initial_balance: float = 10000.0,
        leverage: int = 1,
        slippage_bps: float = 0.0,
        strategy_params: Dict[str, Any] = None,
        auto_sync_on_missing_data: bool = True,
        allow_trade_backfill: bool = True,
    ):
        self.strategy_class = strategy_class
        self.symbols = symbols
        self.start_date = start_date
        self.end_date = end_date
        self.timeframe = timeframe
        self.auto_sync_on_missing_data = auto_sync_on_missing_data
        self.allow_trade_backfill = allow_trade_backfill
        self.strategy_params = strategy_params or {}

        if db_host == 'mock':
            self.db = MockQuestDBManager(host=db_host)
        else:
            self.db = QuestDBManager(host=db_host)

        self.loop = EventLoop(latency_ms=10) # 10ms simulated latency

        # Single shared broker
        self.broker = Broker(initial_balance=initial_balance, leverage=leverage, slippage_bps=slippage_bps)

        # Instantiate strategy and pass broker
        self.strategy = self.strategy_class(**self.strategy_params)
        self.strategy.set_broker(self.broker)

        # We need an instance of SimulationRunner to reuse `_enrich_ohlcv_with_indicators`.
        # But we must not pass the shared strategy, otherwise it replaces the broker.
        # We'll pass a DummyStrategy instead.
        from src.simulation.strategies.demo import DemoStrategy
        self._dummy_runner = SimulationRunner(
            DemoStrategy(),
            self.symbols[0] if self.symbols else "",
            self.start_date,
            self.end_date,
            timeframe=self.timeframe,
            db_host=db_host,
            initial_balance=initial_balance,
            leverage=leverage
        )

    def load_data(self):
        for symbol in self.symbols:
            print(f"Loading data for {symbol} from {self.start_date} to {self.end_date} (timeframe={self.timeframe})...", file=sys.stderr)
            df_ohlcv = self.db.get_ohlcv(
                symbol,
                self.start_date,
                self.end_date,
                timeframe=self.timeframe,
                allow_trade_backfill=self.allow_trade_backfill,
            )

            # Auto-Sync: If data is empty, optionally try to sync candles automatically
            if df_ohlcv.empty and self.auto_sync_on_missing_data:
                print(f"No local data found for {symbol} ({self.timeframe}). Triggering auto-sync...", file=sys.stderr)
                try:
                    from src.data_manager import SyncManager
                    sm = SyncManager(db_manager=self.db)
                    self.db.create_ohlcv_table(timeframe=self.timeframe)
                    sm.sync_data(
                        symbol,
                        self.start_date,
                        self.end_date,
                        sync_mode="candles",
                        timeframe=self.timeframe
                    )
                    df_ohlcv = self.db.get_ohlcv(
                        symbol,
                        self.start_date,
                        self.end_date,
                        timeframe=self.timeframe,
                        allow_trade_backfill=self.allow_trade_backfill,
                    )
                except Exception as e:
                    print(f"Auto-sync failed for {symbol}: {e}", file=sys.stderr)

            # Reusing the logic from standard runner
            df_ohlcv = self._dummy_runner._enrich_ohlcv_with_indicators(df_ohlcv)
            df_funding = self.db.get_funding_rates(symbol, self.start_date, self.end_date)

            if df_ohlcv.empty:
                print(f"No OHLCV data found for {symbol} even after sync attempt.", file=sys.stderr)
                continue

            print(f"Loaded {len(df_ohlcv)} candles and {len(df_funding)} funding rates for {symbol}. Queuing events...", file=sys.stderr)

            for timestamp, row in df_ohlcv.iterrows():
                ts = timestamp.to_pydatetime() if isinstance(timestamp, pd.Timestamp) else timestamp
                payload = row.to_dict()
                payload['symbol'] = symbol
                payload['timestamp'] = ts
                try:
                    event = Event(ts, int(EventType.MARKET_DATA), payload)
                    self.loop.add_event(event)
                except Exception as e:
                    print(f"Error creating market event at {ts}: {e}", file=sys.stderr)

            if not df_funding.empty:
                for timestamp, row in df_funding.iterrows():
                    ts = timestamp.to_pydatetime() if isinstance(timestamp, pd.Timestamp) else timestamp
                    payload = {
                        'symbol': symbol,
                        'funding_rate': row['funding_rate'],
                        'timestamp': ts
                    }
                    try:
                        event = Event(ts, int(EventType.FUNDING), payload)
                        self.loop.add_event(event)
                    except Exception as e:
                        print(f"Error creating funding event at {ts}: {e}", file=sys.stderr)

    def run(self, on_event_callback=None):
        self.load_data()

        def market_data_handler(event):
            # 1. Process Market Data in Broker (Match Orders, Update PnL/Equity)
            new_trades = self.broker.process_market_data(event)

            # 2. Notify Strategy of Fills
            for trade in new_trades:
                self.strategy.on_fill(trade)

            # 3. Notify Strategy of Bar Close
            self.strategy.on_bar(event.payload)

            # 4. Record Equity
            if 'close' in event.payload:
                # Update equity only at bar close
                # Estimate worst equity could be adapted for multi-asset
                self.strategy.update_equity(event.timestamp)

            if on_event_callback:
                try:
                    on_event_callback(event)
                except Exception as e:
                    print(f"Error in on_event_callback: {e}", file=sys.stderr)

        self.loop.subscribe(int(EventType.MARKET_DATA), market_data_handler)

        def funding_handler(event):
            self.broker.process_funding_event(event)
            if on_event_callback:
                try:
                    on_event_callback(event)
                except Exception as e:
                    print(f"Error in on_event_callback (funding): {e}", file=sys.stderr)

        self.loop.subscribe(int(EventType.FUNDING), funding_handler)

        print("Starting multi-asset simulation...", file=sys.stderr)
        self.strategy.on_start()

        try:
            self.loop.run()
        except Exception as e:
            print(f"Simulation error: {e}", file=sys.stderr)

        self.strategy.on_stop()
        print(f"Multi-asset simulation complete. Processed {self.loop.processed_events_count} events.", file=sys.stderr)

        metrics = calculate_metrics(self.broker.trades, self.strategy.equity_curve)
        return {
            "metrics": metrics,
            "trades": self.broker.trades,
            "equity_curve": self.strategy.equity_curve
        }
