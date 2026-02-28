import sys
import pandas as pd
import numpy as np
from datetime import datetime
from src.database import QuestDBManager
from src.mock_database import MockQuestDBManager
from src.simulation.event_loop import EventLoop, Event, EventType
from src.simulation.strategy import Strategy
from src.simulation.metrics import calculate_metrics
from src.simulation.broker import Broker
from src.simulation.indicators import (
    calculate_sma,
    calculate_ema,
    calculate_rsi,
    calculate_atr,
    calculate_rolling_vwap,
    calculate_adx,
)

class SimulationRunner:
    def __init__(
        self,
        strategy: Strategy,
        symbol: str,
        start_date: str,
        end_date: str,
        timeframe: str = "1m",
        db_host: str = 'localhost',
        initial_balance: float = 10000.0,
        leverage: int = 1,
        auto_sync_on_missing_data: bool = True,
        allow_trade_backfill: bool = True,
    ):
        self.strategy = strategy
        self.symbol = symbol
        self.start_date = start_date
        self.end_date = end_date
        self.timeframe = timeframe
        self.auto_sync_on_missing_data = auto_sync_on_missing_data
        self.allow_trade_backfill = allow_trade_backfill
        if db_host == 'mock':
            self.db = MockQuestDBManager(host=db_host)
        else:
            self.db = QuestDBManager(host=db_host)
        self.loop = EventLoop(latency_ms=10) # 10ms simulated latency

        # Initialize Broker
        self.broker = Broker(initial_balance=initial_balance, leverage=leverage)
        self.strategy.set_broker(self.broker)

    def _as_float(self, value):
        try:
            out = float(value)
        except (TypeError, ValueError):
            return None
        if not np.isfinite(out):
            return None
        return out

    def _estimate_worst_case_equity(self, bar_payload: dict):
        """
        Conservative equity mark using intrabar extremes for the event symbol:
        - long positions are marked to bar low
        - short positions are marked to bar high
        """
        symbol = bar_payload.get("symbol")
        low = self._as_float(bar_payload.get("low"))
        high = self._as_float(bar_payload.get("high"))
        if not symbol or low is None or high is None:
            return None

        try:
            balance = float(getattr(self.broker, "balance", 0.0))
            positions = getattr(self.broker, "positions", {})
        except Exception:
            return None

        if not isinstance(positions, dict):
            try:
                positions = dict(positions)
            except Exception:
                return None

        worst_unrealized = 0.0
        for pos in positions.values():
            try:
                pos_symbol = getattr(pos, "symbol", None)
                size = float(getattr(pos, "size", 0.0))
                entry = float(getattr(pos, "entry_price", 0.0))
            except Exception:
                continue

            if not np.isfinite(size) or abs(size) <= 1e-12:
                continue
            if not np.isfinite(entry):
                continue

            if pos_symbol == symbol:
                mark = low if size > 0 else high
            else:
                # Fallback for non-event symbols to current unrealized mark.
                upnl = self._as_float(getattr(pos, "unrealized_pnl", None))
                if upnl is None:
                    mark = entry
                else:
                    mark = entry + (upnl / size)

            worst_unrealized += (mark - entry) * size

        return balance + worst_unrealized

    def _enrich_ohlcv_with_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Precompute indicator columns once for the full dataset.
        This avoids expensive per-bar DataFrame rebuilding in strategy code.
        """
        if df.empty:
            return df

        enriched = df.copy()
        for c in ['open', 'high', 'low', 'close', 'volume']:
            enriched[c] = pd.to_numeric(enriched[c], errors='coerce')
        enriched = enriched.dropna(subset=['open', 'high', 'low', 'close', 'volume'])
        if enriched.empty:
            return enriched

        close = enriched['close']
        high = enriched['high']
        low = enriched['low']
        volume = enriched['volume']

        enriched['rsi14'] = calculate_rsi(close, 14)
        enriched['rsi14_prev'] = enriched['rsi14'].shift(1)
        enriched['ema50'] = calculate_ema(close, 50)
        enriched['ema20'] = calculate_ema(close, 20)
        enriched['vwap50'] = calculate_rolling_vwap(high, low, close, volume, window=50)
        enriched['atr14'] = calculate_atr(high, low, close, 14)
        enriched['adx14'] = calculate_adx(high, low, close, 14)
        enriched['vol_sma20'] = calculate_sma(volume, 20)
        rsi_prev = enriched['rsi14'].shift(1)
        enriched['has_correction_long_prev10'] = rsi_prev.rolling(window=10, min_periods=1).min() < 50
        enriched['has_correction_short_prev10'] = rsi_prev.rolling(window=10, min_periods=1).max() > 50

        def _resample_ohlcv(rule: str) -> pd.DataFrame:
            resampled = enriched[['open', 'high', 'low', 'close', 'volume']].resample(
                rule,
                label='right',
                closed='right',
            ).agg({
                'open': 'first',
                'high': 'max',
                'low': 'min',
                'close': 'last',
                'volume': 'sum',
            })
            return resampled.dropna(subset=['open', 'high', 'low', 'close', 'volume'])

        df_4h = _resample_ohlcv('4h')
        df_daily = _resample_ohlcv('1D')

        regime_4h = calculate_ema(df_4h['close'], 1100) if not df_4h.empty else pd.Series(dtype=float)
        regime_daily = calculate_sma(df_daily['close'], 200) if not df_daily.empty else pd.Series(dtype=float)

        ready_4h = (
            pd.Series((np.arange(len(df_4h)) + 1) >= 1100, index=df_4h.index)
            if not df_4h.empty
            else pd.Series(dtype=bool)
        )
        ready_daily = (
            pd.Series((np.arange(len(df_daily)) + 1) >= 200, index=df_daily.index)
            if not df_daily.empty
            else pd.Series(dtype=bool)
        )

        enriched['regime_filter_4h'] = regime_4h.reindex(enriched.index, method='ffill')
        enriched['regime_filter_daily'] = regime_daily.reindex(enriched.index, method='ffill')
        enriched['ready_4h'] = ready_4h.reindex(enriched.index, method='ffill').fillna(False).astype(bool)
        enriched['ready_daily'] = ready_daily.reindex(enriched.index, method='ffill').fillna(False).astype(bool)

        return enriched

    def load_data(self):
        print(
            f"Loading data for {self.symbol} from {self.start_date} to {self.end_date} "
            f"(timeframe={self.timeframe})...",
            file=sys.stderr,
        )
        df_ohlcv = self.db.get_ohlcv(
            self.symbol,
            self.start_date,
            self.end_date,
            timeframe=self.timeframe,
            allow_trade_backfill=self.allow_trade_backfill,
        )
        
        # Auto-Sync: If data is empty, optionally try to sync candles automatically
        if df_ohlcv.empty and self.auto_sync_on_missing_data:
            print(f"No local data found for {self.symbol} ({self.timeframe}). Triggering auto-sync...", file=sys.stderr)
            try:
                from src.data_manager import SyncManager
                sm = SyncManager(db_manager=self.db)
                self.db.create_ohlcv_table(timeframe=self.timeframe)
                sm.sync_data(
                    self.symbol, 
                    self.start_date, 
                    self.end_date, 
                    sync_mode="candles", 
                    timeframe=self.timeframe
                )
                df_ohlcv = self.db.get_ohlcv(
                    self.symbol,
                    self.start_date,
                    self.end_date,
                    timeframe=self.timeframe,
                    allow_trade_backfill=self.allow_trade_backfill,
                )
            except Exception as e:
                print(f"Auto-sync failed for {self.symbol}: {e}", file=sys.stderr)

        df_ohlcv = self._enrich_ohlcv_with_indicators(df_ohlcv)
        df_funding = self.db.get_funding_rates(self.symbol, self.start_date, self.end_date)

        if df_ohlcv.empty:
            print(f"No OHLCV data found for {self.symbol} even after sync attempt.", file=sys.stderr)
            return

        print(f"Loaded {len(df_ohlcv)} candles and {len(df_funding)} funding rates. Queuing events...", file=sys.stderr)

        for timestamp, row in df_ohlcv.iterrows():
            # Create a market data event
            # Ensure timestamp is passed as a Python datetime
            ts = timestamp.to_pydatetime() if isinstance(timestamp, pd.Timestamp) else timestamp

            payload = row.to_dict()
            payload['symbol'] = self.symbol
            payload['timestamp'] = ts

            # EventType.MARKET_DATA = 0
            try:
                event = Event(ts, int(EventType.MARKET_DATA), payload)
                self.loop.add_event(event)
            except Exception as e:
                print(f"Error creating market event at {ts}: {e}", file=sys.stderr)

        if not df_funding.empty:
            for timestamp, row in df_funding.iterrows():
                ts = timestamp.to_pydatetime() if isinstance(timestamp, pd.Timestamp) else timestamp

                payload = {
                    'symbol': self.symbol,
                    'funding_rate': row['funding_rate'],
                    'timestamp': ts
                }

                # EventType.FUNDING = 1
                try:
                    event = Event(ts, int(EventType.FUNDING), payload)
                    self.loop.add_event(event)
                except Exception as e:
                    print(f"Error creating funding event at {ts}: {e}", file=sys.stderr)

    def run(self, on_event_callback=None):
        self.load_data()

        # Subscribe strategy methods
        # The Rust EventLoop expects a callable that takes (event,)

        def market_data_handler(event):
            # 1. Process Market Data in Broker (Match Orders, Update PnL/Equity)
            # Returns list of filled trades
            new_trades = self.broker.process_market_data(event)

            # 2. Notify Strategy of Fills
            for trade in new_trades:
                self.strategy.on_fill(trade)

            # 3. Notify Strategy of Bar Close
            self.strategy.on_bar(event.payload)

            # 4. Record Equity
            # Strategy records equity curve (delegating to broker equity)
            if 'close' in event.payload:
                self.strategy.update_equity(event.timestamp)
                if self.strategy.equity_curve:
                    worst_equity = self._estimate_worst_case_equity(event.payload)
                    if worst_equity is not None:
                        self.strategy.equity_curve[-1]["equity_worst"] = worst_equity

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

        # Start
        print("Starting simulation...", file=sys.stderr)
        self.strategy.on_start()

        try:
            self.loop.run()
        except Exception as e:
            print(f"Simulation error: {e}", file=sys.stderr)

        self.strategy.on_stop()
        print(f"Simulation complete. Processed {self.loop.processed_events_count} events.", file=sys.stderr)

        # Use broker trades for metrics
        metrics = calculate_metrics(self.broker.trades, self.strategy.equity_curve)
        return {
            "metrics": metrics,
            "trades": self.broker.trades,
            "equity_curve": self.strategy.equity_curve
        }
