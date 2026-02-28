import unittest
from unittest.mock import patch, MagicMock
import pandas as pd
from datetime import datetime, timezone
from src.bulk_candle_downloader import BinanceCandleDownloader
import io
import zipfile

class TestCandleDownloader(unittest.TestCase):
    def setUp(self):
        self.downloader = BinanceCandleDownloader()

    def create_mock_zip(self, content):
        mock_zip = io.BytesIO()
        with zipfile.ZipFile(mock_zip, 'w') as zf:
            zf.writestr('test.csv', content)
        mock_zip.seek(0)
        return mock_zip.read()

    @patch('src.bulk_candle_downloader.requests.get')
    def test_concurrent_download(self, mock_get):
        csv_content = b"1609459200000,100,105,95,100,10,1609459259999,1000,50,5,500,0\n"
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.content = self.create_mock_zip(csv_content)
        mock_get.return_value = mock_response

        # Request 3 days to force multiple concurrent tasks
        start_date = "2024-01-01T00:00:00Z"
        end_date = "2024-01-04T00:00:00Z"

        results = list(self.downloader.download_klines("BTC/USDT", "1m", start_date, end_date))

        self.assertEqual(len(results), 3) # Should yield 3 dataframes
        for df in results:
            self.assertFalse(df.empty)
            self.assertEqual(df.iloc[0]['open'], 100)
            self.assertEqual(df.iloc[0]['high'], 105)
            self.assertEqual(df.iloc[0]['low'], 95)
            self.assertEqual(df.iloc[0]['close'], 100)
            self.assertEqual(df.iloc[0]['volume'], 10)

if __name__ == '__main__':
    unittest.main()
