import unittest
from unittest.mock import patch, MagicMock
import pandas as pd
from datetime import datetime, timezone
from src.bulk_downloader import BinanceBulkDownloader
import io
import zipfile

class TestConcurrentBulkDownloader(unittest.TestCase):
    def setUp(self):
        self.downloader = BinanceBulkDownloader()

    def create_mock_zip(self, content):
        mock_zip = io.BytesIO()
        with zipfile.ZipFile(mock_zip, 'w') as zf:
            zf.writestr('test.csv', content)
        mock_zip.seek(0)
        return mock_zip.read()

    @patch('src.bulk_downloader.requests.get')
    def test_concurrent_download(self, mock_get):
        csv_content = b"id,price,qty,quote_qty,time,is_buyer_maker\n1,100,1,100,1609459200000,true\n"
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.content = self.create_mock_zip(csv_content)
        mock_get.return_value = mock_response

        # Request 3 days to force multiple concurrent tasks
        start_date = "2024-01-01T00:00:00Z"
        end_date = "2024-01-04T00:00:00Z"

        results = list(self.downloader.download_trades("BTC/USDT", start_date, end_date))

        self.assertEqual(len(results), 3) # Should yield 3 dataframes (daily fallback or daily plan)
        for df in results:
            self.assertFalse(df.empty)
            self.assertEqual(df.iloc[0]['price'], 100)
            self.assertEqual(df.iloc[0]['amount'], 1)

if __name__ == '__main__':
    unittest.main()
