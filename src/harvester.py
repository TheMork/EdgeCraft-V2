import ccxt
import pandas as pd
import time
import sys
from datetime import datetime, timezone
from typing import Generator, Optional

class DataHarvester:
    MAX_CONSECUTIVE_NETWORK_ERRORS = 5
    RETRY_DELAY_SECONDS = 5

    def __init__(self, exchange_id: str = 'binance', sandbox: bool = False):
        self.exchange_id = exchange_id
        self.markets_loaded = False
        self.exchange_restricted = False
        exchange_class = getattr(ccxt, exchange_id)
        # Note: Future optimization should implement weight-based rate limiting
        # by inspecting 'x-mbx-used-weight' headers. For now, we rely on
        # CCXT's built-in enableRateLimit which is robust but conservative.
        self.exchange = exchange_class({
            'enableRateLimit': True,
            'options': {
                'defaultType': 'future', # We are targeting futures as per spec
            }
        })
        if sandbox:
            self.exchange.set_sandbox_mode(True)

        # Ensure we load markets to validate symbols
        try:
            self.exchange.load_markets()
            self.markets_loaded = True
        except Exception as e:
            msg = str(e).lower()
            if "restricted location" in msg or "451" in msg:
                self.exchange_restricted = True
                print(
                    f"Warning: Could not load markets due to regional restriction ({e}). "
                    "Live CCXT calls will be skipped.",
                    file=sys.stderr
                )
            else:
                print(f"Warning: Could not load markets: {e}", file=sys.stderr)

    def _is_restricted_error(self, error: Exception) -> bool:
        msg = str(error).lower()
        return "restricted location" in msg or "451" in msg

    def _handle_network_error(self, error: Exception, retry_count: int) -> Optional[int]:
        if self._is_restricted_error(error):
            self.exchange_restricted = True
            print(
                f"Network access restricted for {self.exchange_id} ({error}). "
                "Stopping live CCXT fetch.",
                file=sys.stderr,
            )
            return None

        next_retry = retry_count + 1
        if next_retry >= self.MAX_CONSECUTIVE_NETWORK_ERRORS:
            print(
                f"Network error persists after {next_retry} attempts: {error}. "
                "Aborting this fetch.",
                file=sys.stderr,
            )
            return None

        print(
            f"Network error: {error}, retrying in {self.RETRY_DELAY_SECONDS}s "
            f"({next_retry}/{self.MAX_CONSECUTIVE_NETWORK_ERRORS})...",
            file=sys.stderr,
        )
        time.sleep(self.RETRY_DELAY_SECONDS)
        return next_retry

    def fetch_historical_ohlcv(self, symbol: str, timeframe: str = '1m', start_date_str: str = '2024-01-01T00:00:00Z') -> Generator[pd.DataFrame, None, None]:
        """
        Fetches historical OHLCV data from start_date until now.
        Yields pandas DataFrames in chunks.
        """
        if self.exchange_restricted:
            print("Skipping OHLCV API fetch: exchange access is region-restricted.", file=sys.stderr)
            return

        # Ensure symbol is in correct format if possible (though users might pass CCXT format)
        if self.markets_loaded and symbol not in self.exchange.markets and symbol.replace('/', '') not in self.exchange.markets:
             print(f"Warning: Symbol {symbol} might be invalid.", file=sys.stderr)

        since = self.exchange.parse8601(start_date_str)
        now = self.exchange.milliseconds()
        limit = 1000 # Binance max is often 1000 or 1500
        consecutive_network_errors = 0

        print(f"Starting fetch for {symbol} from {start_date_str}...", file=sys.stderr)

        while since < now:
            try:
                ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe, since, limit)
                consecutive_network_errors = 0
                if not ohlcv:
                    print(f"No more OHLCV data for {symbol}", file=sys.stderr)
                    break

                df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')

                yield df

                # Update since to the last timestamp + 1ms to get the next batch
                last_ts = ohlcv[-1][0]

                # Safety check against infinite loops
                if last_ts < since:
                     # This implies the exchange returned data BEFORE our requested time, or unordered.
                     # Should not happen with Binance.
                     print(f"Error: Received timestamp {last_ts} < requested {since}", file=sys.stderr)
                     break

                if last_ts == since:
                     # If we only got one candle and it's the same as we asked, and we asked for more...
                     if len(ohlcv) == 1:
                         # We might be at the end
                         break
                     else:
                         print("Loop detected with same timestamp, advancing manually + 1 minute", file=sys.stderr)
                         # This is risky, but better than infinite loop.
                         # Ideally, we add timeframe duration.
                         duration_ms = self.exchange.parse_timeframe(timeframe) * 1000
                         since += duration_ms
                else:
                    since = last_ts + 1

                # Check if we reached 'now' (or close enough)
                if since > now:
                    break

            except ccxt.NetworkError as e:
                retry_value = self._handle_network_error(e, consecutive_network_errors)
                if retry_value is None:
                    break
                consecutive_network_errors = retry_value
            except ccxt.ExchangeError as e:
                if self._is_restricted_error(e):
                    self.exchange_restricted = True
                print(f"Exchange error: {e}", file=sys.stderr)
                break
            except Exception as e:
                print(f"Unexpected error: {e}", file=sys.stderr)
                break

    def fetch_funding_rate_history(self, symbol: str, start_date_str: str = '2024-01-01T00:00:00Z') -> Generator[pd.DataFrame, None, None]:
        """
        Fetches historical Funding Rates.
        Yields pandas DataFrames in chunks.
        """
        if self.exchange_restricted:
            print("Skipping funding API fetch: exchange access is region-restricted.", file=sys.stderr)
            return

        since = self.exchange.parse8601(start_date_str)
        now = self.exchange.milliseconds()
        limit = 1000
        consecutive_network_errors = 0

        print(f"Starting funding rate fetch for {symbol} from {start_date_str}...", file=sys.stderr)

        while since < now:
            try:
                rates = self.exchange.fetch_funding_rate_history(symbol, since, limit)
                consecutive_network_errors = 0

                if not rates:
                    print(f"No more funding data for {symbol}", file=sys.stderr)
                    break

                data = []
                for r in rates:
                    data.append({
                        'timestamp': r['timestamp'],
                        'fundingRate': r['fundingRate'],
                        'symbol': r['symbol']
                    })

                df = pd.DataFrame(data)
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')

                yield df

                last_ts = rates[-1]['timestamp']

                if last_ts < since:
                    break

                if last_ts == since:
                    if len(rates) < limit:
                        break # We are done
                    else:
                        # Defensive increment if we are stuck (shouldn't happen with proper pagination)
                        since += 1
                else:
                    since = last_ts + 1

            except ccxt.NetworkError as e:
                retry_value = self._handle_network_error(e, consecutive_network_errors)
                if retry_value is None:
                    break
                consecutive_network_errors = retry_value
            except Exception as e:
                if self._is_restricted_error(e):
                    self.exchange_restricted = True
                print(f"Error fetching funding rates: {e}", file=sys.stderr)
                break

    def fetch_historical_trades(self, symbol: str, start_date_str: str = '2024-01-01T00:00:00Z') -> Generator[pd.DataFrame, None, None]:
        """
        Fetches historical Public Trades (tick data).
        Yields pandas DataFrames in chunks.
        """
        if self.exchange_restricted:
            print("Skipping trade API fetch: exchange access is region-restricted.", file=sys.stderr)
            return

        since = self.exchange.parse8601(start_date_str)
        now = self.exchange.milliseconds()
        limit = 1000  # Binance limit for trades is often 500 or 1000
        consecutive_network_errors = 0

        print(f"Starting trade fetch for {symbol} from {start_date_str}...", file=sys.stderr)

        while since < now:
            try:
                trades = self.exchange.fetch_trades(symbol, since, limit)
                consecutive_network_errors = 0

                if not trades:
                    print(f"No more trades for {symbol}", file=sys.stderr)
                    break

                data = []
                for t in trades:
                    data.append({
                        'id': str(t['id']),
                        'timestamp': t['timestamp'],
                        'symbol': t['symbol'],
                        'side': t['side'],
                        'price': float(t['price']),
                        'amount': float(t['amount'])
                    })

                df = pd.DataFrame(data)
                if not df.empty:
                    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                    yield df

                    last_ts = trades[-1]['timestamp']
                else:
                    # Should not happen if trades is not empty
                    break

                # Pagination logic
                if last_ts < since:
                    # Should not happen, implies backward time travel
                    print(f"Error: Received timestamp {last_ts} < requested {since}", file=sys.stderr)
                    break

                if last_ts == since:
                    if len(trades) < limit:
                        # Less than limit returned, means we reached end of data for now
                        break
                    else:
                        # Full page returned with same timestamp? Rare but possible.
                        # Force advance to avoid infinite loop.
                        since += 1
                else:
                    # Normal case: advanced time.
                    # Using last_ts + 1 ensures progress and avoids infinite loops.
                    # Duplicate trades (if any) are handled by DB deduplication.
                    since = last_ts + 1

                if since > now:
                    break

            except ccxt.NetworkError as e:
                retry_value = self._handle_network_error(e, consecutive_network_errors)
                if retry_value is None:
                    break
                consecutive_network_errors = retry_value
            except Exception as e:
                if self._is_restricted_error(e):
                    self.exchange_restricted = True
                print(f"Error fetching trades: {e}", file=sys.stderr)
                break
