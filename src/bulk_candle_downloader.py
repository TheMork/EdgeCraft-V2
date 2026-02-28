import requests
import zipfile
import io
import pandas as pd
import concurrent.futures
from datetime import datetime, timedelta, timezone
import sys
from typing import Generator, Optional

class BinanceCandleDownloader:
    BASE_URL = "https://data.binance.vision/data/futures/um"

    def __init__(self, sandbox: bool = False):
        self.sandbox = sandbox

    def _get_monthly_url(self, symbol: str, timeframe: str, year: int, month: int) -> str:
        s = symbol.replace('/', '').upper()
        m = f"{month:02d}"
        return f"{self.BASE_URL}/monthly/klines/{s}/{timeframe}/{s}-{timeframe}-{year}-{m}.zip"

    def _get_daily_url(self, symbol: str, timeframe: str, date: datetime) -> str:
        s = symbol.replace('/', '').upper()
        d_str = date.strftime('%Y-%m-%d')
        return f"{self.BASE_URL}/daily/klines/{s}/{timeframe}/{s}-{timeframe}-{d_str}.zip"

    def download_and_extract(self, url: str) -> Optional[pd.DataFrame]:
        print(f"Downloading {url}...", file=sys.stderr)
        try:
            resp = requests.get(url, stream=True, timeout=30)
            if resp.status_code == 404:
                return None
            resp.raise_for_status()

            with zipfile.ZipFile(io.BytesIO(resp.content)) as z:
                csv_filename = z.namelist()[0]
                with z.open(csv_filename) as f:
                    # Binance klines CSV columns:
                    # Open time, Open, High, Low, Close, Volume, Close time, Quote asset volume, Number of trades, Taker buy base asset volume, Taker buy quote asset volume, Ignore
                    df = pd.read_csv(
                        f,
                        header=None,
                        names=['open_time', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'quote_volume', 'trades', 'taker_buy_base', 'taker_buy_quote', 'ignore'],
                        dtype=str,
                        low_memory=False,
                    )

                    if not df.empty and isinstance(df.iloc[0]['open_time'], str) and 'time' in str(df.iloc[0]['open_time']).lower():
                        df = df.iloc[1:]

                    if df.empty:
                        return None

                    df['open'] = pd.to_numeric(df['open'], errors='coerce')
                    df['high'] = pd.to_numeric(df['high'], errors='coerce')
                    df['low'] = pd.to_numeric(df['low'], errors='coerce')
                    df['close'] = pd.to_numeric(df['close'], errors='coerce')
                    df['volume'] = pd.to_numeric(df['volume'], errors='coerce')

                    df['timestamp_ms'] = pd.to_numeric(df['open_time'], errors='coerce')
                    df['timestamp'] = pd.to_datetime(df['timestamp_ms'], unit='ms')

                    df.dropna(subset=['open', 'high', 'low', 'close', 'volume', 'timestamp'], inplace=True)

                    return df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
        except Exception as e:
            print(f"Error downloading/processing {url}: {e}", file=sys.stderr)
            return None

    def download_klines(self, symbol: str, timeframe: str, start_date_str: str, end_date_str: Optional[str] = None) -> Generator[pd.DataFrame, None, None]:
        try:
            start_date = datetime.fromisoformat(start_date_str.replace('Z', '+00:00'))
        except ValueError:
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

        urls_to_download = []

        while current_date < end_date:
            now = datetime.now(timezone.utc)

            if current_date.month == 12:
                next_month_start = datetime(current_date.year + 1, 1, 1, tzinfo=timezone.utc)
            else:
                next_month_start = datetime(current_date.year, current_date.month + 1, 1, tzinfo=timezone.utc)

            is_start_of_month = (current_date.day == 1)
            is_full_month_in_range = (next_month_start <= end_date)
            is_past_month = (next_month_start <= now)

            if is_start_of_month and is_full_month_in_range and is_past_month:
                urls_to_download.append((self._get_monthly_url(symbol, timeframe, current_date.year, current_date.month), current_date, next_month_start))
                current_date = next_month_start
            else:
                urls_to_download.append((self._get_daily_url(symbol, timeframe, current_date), current_date, current_date + timedelta(days=1)))
                current_date += timedelta(days=1)

        def _download_task(task_info):
            url, _start, _end = task_info
            return self.download_and_extract(url), task_info

        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
            future_to_url = {executor.submit(_download_task, task_info): task_info for task_info in urls_to_download}

            results = {}
            for future in concurrent.futures.as_completed(future_to_url):
                df, task_info = future.result()
                results[task_info[1]] = (df, task_info)

            sorted_starts = sorted(results.keys())
            for start_ts in sorted_starts:
                df, task_info = results[start_ts]
                url, current_date, next_date = task_info

                if df is None and 'monthly' in url:
                    print(f"Monthly zip not found for {current_date.strftime('%Y-%m')}, falling back to daily...", file=sys.stderr)
                    fallback_curr = current_date
                    while fallback_curr < next_date and fallback_curr < end_date:
                        fallback_url = self._get_daily_url(symbol, timeframe, fallback_curr)
                        fallback_df = self.download_and_extract(fallback_url)
                        if fallback_df is not None:
                            fallback_df['symbol'] = symbol
                            yield fallback_df
                        fallback_curr += timedelta(days=1)
                elif df is not None:
                    df['symbol'] = symbol
                    yield df
