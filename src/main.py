import argparse
import sys
from src.database import QuestDBManager
from src.harvester import DataHarvester
from src.bulk_downloader import BinanceBulkDownloader

def main():
    parser = argparse.ArgumentParser(description='Institutional Crypto Backtesting Engine - Data Ingestion')
    parser.add_argument('--symbol', type=str, required=True, help='Trading symbol (e.g., BTC/USDT)')
    parser.add_argument('--timeframe', type=str, default='1m', help='Timeframe for OHLCV (e.g., 1m, 1h)')
    parser.add_argument('--start-date', type=str, default='2024-01-01T00:00:00Z', help='Start date in ISO 8601 format')
    parser.add_argument('--host', type=str, default='localhost', help='QuestDB host')
    parser.add_argument('--fetch-trades', action='store_true', help='Fetch historical trades (tick data)')
    parser.add_argument('--use-bulk', action='store_true', help='Use bulk download (ZIPs) for historical trades')

    args = parser.parse_args()

    print(f"Initializing Ingestion for {args.symbol} from {args.start_date}...", file=sys.stderr)

    # Initialize components
    db_manager = QuestDBManager(host=args.host)
    safe_timeframe = db_manager.parse_timeframe(args.timeframe) or "1m"
    harvester = DataHarvester()
    bulk_downloader = BinanceBulkDownloader()

    # Create tables
    print("Creating tables if not exist...", file=sys.stderr)
    db_manager.create_ohlcv_table(timeframe=safe_timeframe)
    db_manager.create_funding_table()
    db_manager.create_trades_table()

    # Ingest OHLCV
    print(f"Fetching OHLCV data for {args.symbol}...", file=sys.stderr)
    total_ohlcv = 0
    try:
        for df in harvester.fetch_historical_ohlcv(args.symbol, safe_timeframe, args.start_date):
            if not df.empty:
                db_manager.insert_ohlcv(args.symbol, df, timeframe=safe_timeframe)
                total_ohlcv += len(df)
                print(f"Inserted {len(df)} candles. Total: {total_ohlcv}", file=sys.stderr, end='\r')
    except KeyboardInterrupt:
        print("\nStopping OHLCV ingestion...", file=sys.stderr)
    except Exception as e:
        print(f"\nError during OHLCV ingestion: {e}", file=sys.stderr)

    print(f"\nTotal OHLCV inserted: {total_ohlcv}", file=sys.stderr)

    # Ingest Funding Rates
    print(f"Fetching Funding Rates for {args.symbol}...", file=sys.stderr)
    total_funding = 0
    try:
        for df in harvester.fetch_funding_rate_history(args.symbol, args.start_date):
            if not df.empty:
                db_manager.insert_funding(args.symbol, df)
                total_funding += len(df)
                print(f"Inserted {len(df)} funding rates. Total: {total_funding}", file=sys.stderr, end='\r')
    except KeyboardInterrupt:
        print("\nStopping Funding Rate ingestion...", file=sys.stderr)
    except Exception as e:
        print(f"\nError during Funding Rate ingestion: {e}", file=sys.stderr)

    print(f"\nTotal Funding Rates inserted: {total_funding}", file=sys.stderr)

    # Ingest Trades
    if args.fetch_trades:
        print(f"Fetching Historical Trades for {args.symbol}...", file=sys.stderr)
        total_trades = 0
        try:
            if args.use_bulk:
                print("Using Bulk Downloader (ZIPs)...", file=sys.stderr)
                trade_generator = bulk_downloader.download_trades(args.symbol, args.start_date)
            else:
                print("Using API Harvester...", file=sys.stderr)
                trade_generator = harvester.fetch_historical_trades(args.symbol, args.start_date)

            for df in trade_generator:
                if not df.empty:
                    db_manager.insert_trades(args.symbol, df)
                    total_trades += len(df)
                    print(f"Inserted {len(df)} trades. Total: {total_trades}", file=sys.stderr, end='\r')
        except KeyboardInterrupt:
            print("\nStopping Trade ingestion...", file=sys.stderr)
        except Exception as e:
            print(f"\nError during Trade ingestion: {e}", file=sys.stderr)

        print(f"\nTotal Trades inserted: {total_trades}", file=sys.stderr)

    print("Ingestion complete.", file=sys.stderr)

if __name__ == "__main__":
    main()
