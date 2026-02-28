import requests
import pandas as pd
from questdb.ingress import Sender, IngressError, TimestampNanos
import sys
import time
import os
from typing import Dict, Optional, Tuple

class QuestDBManager:
    VALID_TIMEFRAMES = ("1m", "3m", "5m", "15m", "30m", "1h", "2h", "4h", "6h", "8h", "12h", "1d", "3d", "1w", "1M")
    OHLCV_TABLE_BY_TIMEFRAME = {
        "1m": "ohlcv",
        "3m": "ohlcv_3m",
        "5m": "ohlcv_5m",
        "15m": "ohlcv_15m",
        "30m": "ohlcv_30m",
        "1h": "ohlcv_1h",
        "2h": "ohlcv_2h",
        "4h": "ohlcv_4h",
        "6h": "ohlcv_6h",
        "8h": "ohlcv_8h",
        "12h": "ohlcv_12h",
        "1d": "ohlcv_1d",
        "3d": "ohlcv_3d",
        "1w": "ohlcv_1w",
        "1M": "ohlcv_1mo",
    }
    TIMEFRAME_ALIASES = {
        "1min": "1m",
        "3min": "3m",
        "5min": "5m",
        "15min": "15m",
        "30min": "30m",
        "1hour": "1h",
        "2hour": "2h",
        "4hour": "4h",
        "6hour": "6h",
        "8hour": "8h",
        "12hour": "12h",
        "1day": "1d",
        "3day": "3d",
        "1week": "1w",
        "1mo": "1M",
        "1mon": "1M",
        "1month": "1M",
    }

    def __init__(
        self,
        host: str = 'localhost',
        ilp_port: int = 9009,
        rest_port: int = 9000,
        request_timeout_seconds: int = 20,
    ):
        self.host = host
        self.ilp_port = ilp_port
        self.rest_port = rest_port
        timeout_raw = os.getenv("EDGECRAFT_DB_REQUEST_TIMEOUT_SECONDS")
        if timeout_raw:
            try:
                request_timeout_seconds = max(2, int(timeout_raw))
            except ValueError:
                pass
        self.request_timeout_seconds = request_timeout_seconds
        self.last_error: Optional[str] = None

    def parse_timeframe(self, timeframe: str) -> Optional[str]:
        raw = (timeframe or "").strip()
        if raw in self.VALID_TIMEFRAMES:
            return raw
        alias = self.TIMEFRAME_ALIASES.get(raw.lower())
        if alias in self.VALID_TIMEFRAMES:
            return alias
        return None

    def _normalize_timeframe(self, timeframe: str) -> str:
        parsed = self.parse_timeframe(timeframe)
        if parsed is not None:
            return parsed
        return "1m"

    def _ohlcv_table_name(self, timeframe: str) -> str:
        safe_timeframe = self._normalize_timeframe(timeframe)
        return self.OHLCV_TABLE_BY_TIMEFRAME[safe_timeframe]

    def create_ohlcv_table(self, timeframe: str = "1m") -> None:
        """Creates the timeframe-specific OHLCV table with proper types and partitioning."""
        table_name = self._ohlcv_table_name(timeframe)
        query = f"""
        CREATE TABLE IF NOT EXISTS {table_name} (
            symbol SYMBOL,
            open DOUBLE,
            high DOUBLE,
            low DOUBLE,
            close DOUBLE,
            volume DOUBLE,
            timestamp TIMESTAMP
        ) TIMESTAMP(timestamp) PARTITION BY MONTH WAL
        DEDUP UPSERT KEYS(timestamp, symbol);
        """
        self._execute_sql(query)

    def create_all_ohlcv_tables(self) -> None:
        for timeframe in self.VALID_TIMEFRAMES:
            self.create_ohlcv_table(timeframe=timeframe)

    def create_funding_table(self) -> None:
        """Creates the funding_rates table."""
        query = """
        CREATE TABLE IF NOT EXISTS funding_rates (
            symbol SYMBOL,
            funding_rate DOUBLE,
            timestamp TIMESTAMP
        ) TIMESTAMP(timestamp) PARTITION BY MONTH WAL
        DEDUP UPSERT KEYS(timestamp, symbol);
        """
        self._execute_sql(query)

    def create_trades_table(self) -> None:
        """Creates the trades table."""
        query = """
        CREATE TABLE IF NOT EXISTS trades (
            id SYMBOL,
            symbol SYMBOL,
            side SYMBOL,
            price DOUBLE,
            amount DOUBLE,
            timestamp TIMESTAMP
        ) TIMESTAMP(timestamp) PARTITION BY DAY WAL
        DEDUP UPSERT KEYS(timestamp, symbol, id);
        """
        self._execute_sql(query)

    def _execute_sql(self, query: str) -> None:
        url = f"http://{self.host}:{self.rest_port}/exec"
        params = {'query': query}
        try:
            response = requests.get(url, params=params, timeout=self.request_timeout_seconds)
            response.raise_for_status()
            self.last_error = None
        except requests.exceptions.RequestException as e:
            self.last_error = str(e)
            print(f"Error executing SQL: {e}", file=sys.stderr)

    def _query_sql(self, query: str) -> Optional[dict]:
        url = f"http://{self.host}:{self.rest_port}/exec"
        params = {'query': query}
        try:
            response = requests.get(url, params=params, timeout=self.request_timeout_seconds)
            response.raise_for_status()
            self.last_error = None
            return response.json()
        except requests.exceptions.RequestException as e:
            self.last_error = str(e)
            print(f"Error querying SQL: {e}", file=sys.stderr)
            return None

    def is_available(self) -> bool:
        result = self._query_sql("SELECT 1")
        return bool(result and 'dataset' in result)

    def _escape_literal(self, value: str) -> str:
        return value.replace("'", "''")

    def _sender_conf(self) -> str:
        return f"tcp::addr={self.host}:{self.ilp_port};"

    def materialize_ohlcv_from_trades(self, symbol: str, start_date: str, end_date: str, timeframe: str = "1m") -> None:
        """
        Aggregates trade ticks into timeframe-specific OHLCV bars.
        """
        safe_symbol = self._escape_literal(symbol)
        safe_start = self._escape_literal(start_date)
        safe_end = self._escape_literal(end_date)
        safe_timeframe = self._normalize_timeframe(timeframe)
        table_name = self._ohlcv_table_name(safe_timeframe)

        self.create_trades_table()
        self.create_ohlcv_table(timeframe=safe_timeframe)
        query = f"""
        INSERT INTO {table_name}
        SELECT
            symbol,
            first(price) AS open,
            max(price) AS high,
            min(price) AS low,
            last(price) AS close,
            sum(amount) AS volume,
            timestamp
        FROM trades
        WHERE symbol = '{safe_symbol}'
          AND timestamp >= '{safe_start}'
          AND timestamp <= '{safe_end}'
        SAMPLE BY {safe_timeframe} ALIGN TO CALENDAR;
        """
        self._execute_sql(query)

    def get_ohlcv(
        self,
        symbol: str,
        start_date: str,
        end_date: str,
        timeframe: str = "1m",
        allow_trade_backfill: bool = True,
    ) -> pd.DataFrame:
        """
        Queries OHLCV data from QuestDB.
        """
        safe_symbol = self._escape_literal(symbol)
        safe_start = self._escape_literal(start_date)
        safe_end = self._escape_literal(end_date)
        safe_timeframe = self._normalize_timeframe(timeframe)
        table_name = self._ohlcv_table_name(safe_timeframe)
        query = (
            f"SELECT * FROM {table_name} "
            f"WHERE symbol = '{safe_symbol}' AND timestamp >= '{safe_start}' AND timestamp <= '{safe_end}' "
            f"ORDER BY timestamp ASC"
        )
        requested_start = pd.to_datetime(start_date, utc=True, errors='coerce')
        requested_end = pd.to_datetime(end_date, utc=True, errors='coerce')
        tolerance_map = {
            "1m": pd.Timedelta(minutes=1),
            "3m": pd.Timedelta(minutes=3),
            "5m": pd.Timedelta(minutes=5),
            "15m": pd.Timedelta(minutes=15),
            "30m": pd.Timedelta(minutes=30),
            "1h": pd.Timedelta(hours=1),
            "2h": pd.Timedelta(hours=2),
            "4h": pd.Timedelta(hours=4),
            "6h": pd.Timedelta(hours=6),
            "8h": pd.Timedelta(hours=8),
            "12h": pd.Timedelta(hours=12),
            "1d": pd.Timedelta(days=1),
            "3d": pd.Timedelta(days=3),
            "1w": pd.Timedelta(days=7),
            "1M": pd.Timedelta(days=31),
        }
        tolerance = tolerance_map.get(safe_timeframe, pd.Timedelta(minutes=1))

        def _to_df(result: Optional[dict]) -> pd.DataFrame:
            if not result or 'dataset' not in result:
                return pd.DataFrame()
            columns = [col['name'] for col in result.get('columns', [])]
            if not columns:
                return pd.DataFrame()
            frame = pd.DataFrame(result['dataset'], columns=columns)
            if 'timestamp' not in frame.columns:
                return pd.DataFrame()
            frame['timestamp'] = pd.to_datetime(frame['timestamp'])
            frame.set_index('timestamp', inplace=True)
            return frame

        def _covers_requested_range(frame: pd.DataFrame) -> bool:
            if frame.empty:
                return False
            if pd.isna(requested_start) or pd.isna(requested_end):
                return True
            existing_start = pd.to_datetime(frame.index.min(), utc=True, errors='coerce')
            existing_end = pd.to_datetime(frame.index.max(), utc=True, errors='coerce')
            if pd.isna(existing_start) or pd.isna(existing_end):
                return False
            return (
                existing_start <= (requested_start + tolerance)
                and existing_end >= (requested_end - tolerance)
            )

        self.last_error = None
        first_query = self._query_sql(query)
        if first_query is None and self.last_error:
            transient_markers = (
                "read timed out",
                "connection refused",
                "failed to establish a new connection",
                "remote end closed connection",
            )
            if any(marker in self.last_error.lower() for marker in transient_markers):
                raise RuntimeError(f"QuestDB unavailable: {self.last_error}")
        df = _to_df(first_query)

        # Backfill timeframe candles from trades on demand.
        if not _covers_requested_range(df):
            if not allow_trade_backfill:
                return pd.DataFrame()
            self.materialize_ohlcv_from_trades(symbol, start_date, end_date, timeframe=safe_timeframe)
            # WAL apply can be delayed for large ranges; wait and re-check.
            for _ in range(30):
                df = _to_df(self._query_sql(query))
                if _covers_requested_range(df):
                    break
                time.sleep(0.2)

            # If persisted OHLCV is still incomplete, aggregate directly from trades
            # so backtest can use the requested range immediately.
            if not _covers_requested_range(df):
                trades_query = f"""
                SELECT
                    symbol,
                    first(price) AS open,
                    max(price) AS high,
                    min(price) AS low,
                    last(price) AS close,
                    sum(amount) AS volume,
                    timestamp
                FROM trades
                WHERE symbol = '{safe_symbol}'
                  AND timestamp >= '{safe_start}'
                  AND timestamp <= '{safe_end}'
                SAMPLE BY {safe_timeframe} ALIGN TO CALENDAR
                ORDER BY timestamp ASC
                """
                df = _to_df(self._query_sql(trades_query))
                if not _covers_requested_range(df):
                    # Avoid misleading simulations on partial ranges.
                    # Caller can prompt user to sync missing periods first.
                    return pd.DataFrame()

        return df

    def get_funding_rates(self, symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
        """
        Queries funding rates from QuestDB.
        """
        query = f"SELECT * FROM funding_rates WHERE symbol = '{symbol}' AND timestamp >= '{start_date}' AND timestamp <= '{end_date}' ORDER BY timestamp ASC"
        data = self._query_sql(query)

        if not data or 'dataset' not in data:
            return pd.DataFrame()

        columns = [col['name'] for col in data['columns']]
        df = pd.DataFrame(data['dataset'], columns=columns)

        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df.set_index('timestamp', inplace=True)

        return df

    def insert_ohlcv(self, symbol: str, df: pd.DataFrame, timeframe: str = "1m") -> None:
        """
        Inserts OHLCV data using ILP.
        df expected to have columns: timestamp (datetime), open, high, low, close, volume
        """
        if df.empty:
            return
        safe_timeframe = self._normalize_timeframe(timeframe)
        table_name = self._ohlcv_table_name(safe_timeframe)
        self.create_ohlcv_table(timeframe=safe_timeframe)

        chunk_size = 10000
        total_rows = len(df)

        try:
            with Sender.from_conf(self._sender_conf()) as sender:
                for start in range(0, total_rows, chunk_size):
                    end = min(start + chunk_size, total_rows)
                    chunk = df.iloc[start:end]

                    for _, row in chunk.iterrows():
                        sender.row(
                            table_name,
                            symbols={'symbol': symbol},
                            columns={
                                'open': float(row['open']),
                                'high': float(row['high']),
                                'low': float(row['low']),
                                'close': float(row['close']),
                                'volume': float(row['volume'])
                            },
                            at=TimestampNanos.from_datetime(row['timestamp'])
                        )
                    sender.flush()
        except IngressError as e:
            print(f"Ingress Error (OHLCV): {e}", file=sys.stderr)
        except Exception as e:
            print(f"Error inserting OHLCV: {e}", file=sys.stderr)

    def insert_funding(self, symbol: str, df: pd.DataFrame) -> None:
        """
        Inserts Funding Rate data using ILP.
        df expected to have columns: timestamp (datetime), fundingRate
        """
        if df.empty:
            return

        try:
            with Sender.from_conf(self._sender_conf()) as sender:
                for _, row in df.iterrows():
                    sender.row(
                        'funding_rates',
                        symbols={'symbol': symbol},
                        columns={'funding_rate': float(row['fundingRate'])},
                        at=TimestampNanos.from_datetime(row['timestamp'])
                    )
                sender.flush()
        except IngressError as e:
            print(f"Ingress Error (Funding): {e}", file=sys.stderr)
        except Exception as e:
            print(f"Error inserting Funding: {e}", file=sys.stderr)

    def insert_trades(self, symbol: str, df: pd.DataFrame) -> None:
        """
        Inserts Trades data using ILP.
        df expected to have columns: timestamp (datetime), id, symbol, side, price, amount
        """
        if df.empty:
            return

        chunk_size = 10000
        total_rows = len(df)

        try:
            with Sender.from_conf(self._sender_conf()) as sender:
                for start in range(0, total_rows, chunk_size):
                    end = min(start + chunk_size, total_rows)
                    chunk = df.iloc[start:end]

                    for _, row in chunk.iterrows():
                        sender.row(
                            'trades',
                            symbols={
                                'symbol': symbol,
                                'id': str(row['id']),
                                'side': str(row['side'])
                            },
                            columns={
                                'price': float(row['price']),
                                'amount': float(row['amount'])
                            },
                            at=TimestampNanos.from_datetime(row['timestamp'])
                        )
                    sender.flush()
        except IngressError as e:
            print(f"Ingress Error (Trades): {e}", file=sys.stderr)
        except Exception as e:
            print(f"Error inserting Trades: {e}", file=sys.stderr)

    def _get_symbol_min_max(
        self,
        table_name: str,
        symbol: str,
    ) -> Tuple[Optional[pd.Timestamp], Optional[pd.Timestamp]]:
        safe_symbol = self._escape_literal(symbol)
        query = f"SELECT min(timestamp), max(timestamp) FROM {table_name} WHERE symbol = '{safe_symbol}'"
        data = self._query_sql(query)

        if not data or 'dataset' not in data or not data['dataset']:
            return None, None

        row = data['dataset'][0]
        # Check if result is empty (QuestDB might return nulls if no rows match)
        if row[0] is None or row[1] is None:
            return None, None

        return pd.to_datetime(row[0]), pd.to_datetime(row[1])

    def get_trade_min_max(self, symbol: str) -> Tuple[Optional[pd.Timestamp], Optional[pd.Timestamp]]:
        """
        Returns the min and max timestamp for trades of a given symbol.
        """
        return self._get_symbol_min_max('trades', symbol)

    def get_ohlcv_min_max(self, symbol: str, timeframe: str = "1m") -> Tuple[Optional[pd.Timestamp], Optional[pd.Timestamp]]:
        """
        Returns the min and max timestamp for OHLCV rows of a given symbol and timeframe.
        """
        table_name = self._ohlcv_table_name(timeframe)
        self.create_ohlcv_table(timeframe=timeframe)
        return self._get_symbol_min_max(table_name, symbol)

    def get_ohlcv_ranges(self, symbol: str) -> Dict[str, Tuple[Optional[pd.Timestamp], Optional[pd.Timestamp]]]:
        ranges: Dict[str, Tuple[Optional[pd.Timestamp], Optional[pd.Timestamp]]] = {}
        for timeframe in self.VALID_TIMEFRAMES:
            ranges[timeframe] = self.get_ohlcv_min_max(symbol, timeframe=timeframe)
        return ranges
