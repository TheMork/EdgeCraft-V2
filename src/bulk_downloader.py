import requests
import zipfile
import io
import pandas as pd
from datetime import datetime, timedelta, timezone
import sys
from typing import Generator, Optional

class BinanceBulkDownloader:
    BASE_URL = "https://data.binance.vision/data/futures/um"

    def __init__(self, sandbox: bool = False):
        # Sandbox doesn't really apply to historical data download usually
        self.sandbox = sandbox

    def _get_monthly_url(self, symbol: str, year: int, month: int) -> str:
        # Standardize symbol: remove /
        s = symbol.replace('/', '').upper()
        # Format month: 01, 02...
        m = f"{month:02d}"
        return f"{self.BASE_URL}/monthly/trades/{s}/{s}-trades-{year}-{m}.zip"

    def _get_daily_url(self, symbol: str, date: datetime) -> str:
        s = symbol.replace('/', '').upper()
        d_str = date.strftime('%Y-%m-%d')
        return f"{self.BASE_URL}/daily/trades/{s}/{s}-trades-{d_str}.zip"

    def download_and_extract(self, url: str) -> Optional[pd.DataFrame]:
        """
        Downloads a ZIP file from the given URL, extracts the CSV,
        and returns a DataFrame.
        Returns None if download fails (e.g., 404).
        """
        print(f"Downloading {url}...", file=sys.stderr)
        try:
            resp = requests.get(url, stream=True, timeout=30)
            if resp.status_code == 404:
                # Expected for recent dates or incomplete months
                return None
            resp.raise_for_status()

            with zipfile.ZipFile(io.BytesIO(resp.content)) as z:
                # Expecting one CSV file inside
                csv_filename = z.namelist()[0]
                with z.open(csv_filename) as f:
                    # Binance Vision CSVs usually don't have headers for trades
                    # Columns: id, price, qty, quote_qty, time, is_buyer_maker
                    # We read without header and assign names manually
                    df = pd.read_csv(
                        f,
                        header=None,
                        names=['id', 'price', 'qty', 'quote_qty', 'time', 'is_buyer_maker'],
                        dtype=str,
                        low_memory=False,
                    )

                    # Heuristic: Check if first row is header (contains 'id')
                    if not df.empty and isinstance(df.iloc[0]['id'], str) and 'id' in str(df.iloc[0]['id']).lower():
                        df = df.iloc[1:]

                    if df.empty:
                        return None

                    # Type conversion
                    df['id'] = df['id'].astype(str)
                    df['price'] = pd.to_numeric(df['price'], errors='coerce')
                    df['amount'] = pd.to_numeric(df['qty'], errors='coerce') # rename qty to amount

                    # Ensure time is numeric before converting to datetime
                    df['timestamp_ms'] = pd.to_numeric(df['time'], errors='coerce')
                    df['timestamp'] = pd.to_datetime(df['timestamp_ms'], unit='ms')

                    # Drop rows with NaN if conversion failed (e.g. bad lines or header leftovers)
                    df.dropna(subset=['price', 'amount', 'timestamp'], inplace=True)

                    # Map is_buyer_maker to side
                    # is_buyer_maker = True -> Maker is Buyer -> Taker is Seller -> SELL
                    # is_buyer_maker = False -> Maker is Seller -> Taker is Buyer -> BUY
                    df['side'] = df['is_buyer_maker'].apply(lambda x: 'SELL' if str(x).lower() == 'true' else 'BUY')

                    # Return only necessary columns
                    return df[['id', 'timestamp', 'side', 'price', 'amount']]

        except Exception as e:
            print(f"Error downloading/processing {url}: {e}", file=sys.stderr)
            return None

    def download_trades(self, symbol: str, start_date_str: str, end_date_str: Optional[str] = None) -> Generator[pd.DataFrame, None, None]:
        """
        Smart download strategy:
        1. Parse dates.
        2. Iterate months. If a full month is requested and available (i.e. not current month), download monthly ZIP.
        3. If partial month or current month, iterate days and download daily ZIPs.
        4. Yield DataFrames.
        """
        # Parse start_date
        try:
            start_date = datetime.fromisoformat(start_date_str.replace('Z', '+00:00'))
        except ValueError:
            # Fallback if isoformat fails or other format
             start_date = pd.to_datetime(start_date_str).to_pydatetime()

        if start_date.tzinfo is None:
             start_date = start_date.replace(tzinfo=timezone.utc)

        if end_date_str:
            try:
                end_date = datetime.fromisoformat(end_date_str.replace('Z', '+00:00'))
            except ValueError:
                end_date = pd.to_datetime(end_date_str).to_pydatetime()

            if end_date.tzinfo is None:
                end_date = end_date.replace(tzinfo=timezone.utc)
        else:
            end_date = datetime.now(timezone.utc)

        current_date = start_date

        while current_date < end_date:
            # Determine if we should try monthly download
            # We can download monthly if:
            # 1. We are at the start of a month (day 1)
            # 2. The entire month is within the requested range (end_date >= end of month)
            # 3. The month is fully completed (i.e., strictly before current month)

            now = datetime.now(timezone.utc)

            # Start of next month
            if current_date.month == 12:
                next_month_start = datetime(current_date.year + 1, 1, 1, tzinfo=timezone.utc)
            else:
                next_month_start = datetime(current_date.year, current_date.month + 1, 1, tzinfo=timezone.utc)

            is_start_of_month = (current_date.day == 1)
            is_full_month_in_range = (next_month_start <= end_date)
            is_past_month = (next_month_start <= now) # Month has fully passed

            if is_start_of_month and is_full_month_in_range and is_past_month:
                # Try monthly download
                url = self._get_monthly_url(symbol, current_date.year, current_date.month)
                df = self.download_and_extract(url)
                if df is not None:
                    df['symbol'] = symbol
                    yield df
                    current_date = next_month_start
                    continue
                else:
                    # If monthly fails (e.g. 404), fall back to daily for this month
                    print(f"Monthly zip not found for {current_date.strftime('%Y-%m')}, falling back to daily...", file=sys.stderr)

            # Daily download
            url = self._get_daily_url(symbol, current_date)
            df = self.download_and_extract(url)
            if df is not None:
                 df['symbol'] = symbol
                 yield df

            current_date += timedelta(days=1)
