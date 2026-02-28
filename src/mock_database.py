import pandas as pd
import numpy as np
from datetime import datetime, timedelta

class MockQuestDBManager:
    def __init__(self, host: str = 'mock'):
        self.host = host

    def get_ohlcv(self, symbol: str, start_date: str, end_date: str, timeframe: str = "1h", allow_trade_backfill: bool = True) -> pd.DataFrame:
        # Generate synthetic data
        start = pd.to_datetime(start_date)
        end = pd.to_datetime(end_date)

        freq_map = {
            "1m": "1min",
            "3m": "3min",
            "5m": "5min",
            "15m": "15min",
            "30m": "30min",
            "1h": "1h",
            "2h": "2h",
            "4h": "4h",
            "6h": "6h",
            "8h": "8h",
            "12h": "12h",
            "1d": "1d",
            "3d": "3d",
            "1w": "1w",
            "1M": "30d",
        }
        freq = freq_map.get(timeframe, "1h")
        dates = pd.date_range(start, end, freq=freq)
        n = len(dates)

        if n == 0:
            return pd.DataFrame()

        # Sine wave + noise
        t = np.linspace(0, 4*np.pi, n)
        price = 100 + 10 * np.sin(t) + np.random.normal(0, 1, n)

        df = pd.DataFrame({
            'timestamp': dates,
            'open': price,
            'high': price + 1,
            'low': price - 1,
            'close': price + 0.5, # Slightly bullish bias?
            'volume': 1000
        })
        df.set_index('timestamp', inplace=True)
        return df

    def get_funding_rates(self, symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
        start = pd.to_datetime(start_date)
        end = pd.to_datetime(end_date)
        dates = pd.date_range(start, end, freq='8h')

        df = pd.DataFrame({
            'timestamp': dates,
            'funding_rate': 0.0001
        })
        df.set_index('timestamp', inplace=True)
        return df
