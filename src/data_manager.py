import pandas as pd
from datetime import datetime, timezone
from typing import Callable, Iterator, Optional
import os
import sys
import requests
from urllib import parse
from pathlib import Path

from src.database import QuestDBManager
from src.bulk_downloader import BinanceBulkDownloader
from src.harvester import DataHarvester

SyncProgressCallback = Callable[[int, str], None]

class SyncManager:
    BULK_INSERT_CHUNK_SIZE_DEFAULT = 200_000
    VALID_TIMEFRAMES = set(QuestDBManager.VALID_TIMEFRAMES)
    TOP20_FALLBACK = [
        "BTC/USDT", "ETH/USDT", "SOL/USDT", "BNB/USDT", "XRP/USDT",
        "DOGE/USDT", "ADA/USDT", "AVAX/USDT", "TRX/USDT", "LINK/USDT",
        "DOT/USDT", "MATIC/USDT", "SHIB/USDT", "LTC/USDT", "UNI/USDT",
        "ATOM/USDT", "XLM/USDT", "ETC/USDT", "FIL/USDT", "HBAR/USDT",
    ]
    ORDERED_TIMEFRAMES = list(QuestDBManager.VALID_TIMEFRAMES)
    MARKET_CAP_CACHE_FILE = Path("/tmp/edgecraft_top20_market_cap.json")
    STABLE_BASES = {
        "USDT", "USDC", "USDE", "USDS", "DAI", "FDUSD", "TUSD", "USDP", "BUSD", "USDD", "USD1", "PYUSD", "FRAX",
    }
    COIN_ALIASES = {
        "MATIC": "POL",
        "SHIB": "1000SHIB",
    }

    def __init__(self, db_manager: Optional[QuestDBManager] = None):
        self.db = db_manager if db_manager else QuestDBManager()
        self.bulk_downloader = BinanceBulkDownloader()
        self.harvester = DataHarvester()
        chunk_size_raw = os.getenv("EDGECRAFT_BULK_CHUNK_SIZE", str(self.BULK_INSERT_CHUNK_SIZE_DEFAULT))
        try:
            self.bulk_insert_chunk_size = max(10_000, int(chunk_size_raw))
        except ValueError:
            self.bulk_insert_chunk_size = self.BULK_INSERT_CHUNK_SIZE_DEFAULT
            print(
                f"Invalid EDGECRAFT_BULK_CHUNK_SIZE='{chunk_size_raw}', "
                f"using default {self.bulk_insert_chunk_size}.",
                file=sys.stderr,
            )
        # Keep startup lightweight; tables are created lazily in sync/read paths.
        self.db.create_trades_table()
        self.db.create_funding_table()

    def _normalize_symbol(self, symbol: str) -> str:
        s = symbol.strip().upper()
        if ":" in s: s = s.split(":", 1)[0]
        if "/" in s: return s
        if s.endswith("USDT") and len(s) > 4: return f"{s[:-4]}/USDT"
        return s

    def _emit_progress(self, progress_callback: Optional[SyncProgressCallback], progress: int, message: str) -> None:
        if progress_callback:
            try: progress_callback(progress, message)
            except Exception as e: print(f"Progress callback error: {e}", file=sys.stderr)

    def _iter_dataframe_chunks(self, df: pd.DataFrame, chunk_size: int) -> Iterator[pd.DataFrame]:
        if chunk_size <= 0:
            yield df
            return
        total_rows = len(df)
        for start in range(0, total_rows, chunk_size):
            end = min(start + chunk_size, total_rows)
            yield df.iloc[start:end]

    def _load_tradable_usdt_symbols(self) -> set[str]:
        if self.harvester.exchange_restricted:
            return set()
        tradable: set[str] = set()
        try:
            self.harvester.exchange.load_markets()
            for symbol, market in self.harvester.exchange.markets.items():
                if not isinstance(market, dict):
                    continue
                quote = str(market.get("quote", "")).upper()
                is_swap = bool(market.get("swap", False))
                is_active = bool(market.get("active", True))
                if quote != "USDT" or not is_swap or not is_active:
                    continue
                norm = self._normalize_symbol(str(symbol))
                if norm.endswith("/USDT"):
                    tradable.add(norm)
        except Exception as e:
            print(f"Error loading tradable USDT symbols: {e}", file=sys.stderr)
        return tradable

    def _load_cached_market_cap_symbols(self) -> list[str]:
        ttl_raw = os.getenv("EDGECRAFT_TOP20_CACHE_TTL_SEC", "900").strip()
        try:
            ttl_sec = max(0, int(ttl_raw))
        except ValueError:
            ttl_sec = 900
        path = self.MARKET_CAP_CACHE_FILE
        if ttl_sec <= 0 or not path.exists():
            return []
        try:
            age = datetime.now(timezone.utc).timestamp() - path.stat().st_mtime
            if age > ttl_sec:
                return []
            payload = pd.read_json(path, typ="series")
            syms = payload.get("symbols", [])
            if isinstance(syms, list):
                out = []
                for s in syms:
                    n = self._normalize_symbol(str(s))
                    if n not in out:
                        out.append(n)
                return out[:20]
        except Exception:
            return []
        return []

    def _save_cached_market_cap_symbols(self, symbols: list[str]) -> None:
        try:
            s = pd.Series(
                {
                    "saved_at": datetime.now(timezone.utc).isoformat(),
                    "symbols": symbols[:20],
                }
            )
            s.to_json(self.MARKET_CAP_CACHE_FILE)
        except Exception:
            pass

    def get_top_20_symbols_by_market_cap(self) -> list[str]:
        print("Fetching top 20 symbols by market cap (CoinGecko)...", file=sys.stderr)
        cached = self._load_cached_market_cap_symbols()
        if cached:
            print("Using cached CoinGecko top20 symbols.", file=sys.stderr)
            return cached
        params = {
            "vs_currency": "usd",
            "order": "market_cap_desc",
            "per_page": "250",
            "page": "1",
            "sparkline": "false",
        }
        url = "https://api.coingecko.com/api/v3/coins/markets?" + parse.urlencode(params)
        headers = {"User-Agent": "EdgeCraft/1.0"}
        api_key = os.getenv("COINGECKO_API_KEY", "").strip()
        if api_key:
            headers["x-cg-pro-api-key"] = api_key

        tradable = self._load_tradable_usdt_symbols()
        out: list[str] = []
        seen = set()
        try:
            resp = requests.get(url, headers=headers, timeout=30)
            resp.raise_for_status()
            payload = resp.json()

            if isinstance(payload, list):
                for coin in payload:
                    base = str((coin or {}).get("symbol", "")).strip().upper()
                    if not base:
                        continue
                    if base in self.STABLE_BASES:
                        continue
                    candidates = [base]
                    alias = self.COIN_ALIASES.get(base)
                    if alias:
                        candidates.append(alias)
                    selected = None
                    for cand in candidates:
                        symbol = self._normalize_symbol(f"{cand}/USDT")
                        if tradable and symbol not in tradable:
                            continue
                        selected = symbol
                        break
                    if not selected or selected in seen:
                        continue
                    seen.add(selected)
                    out.append(selected)
                    if len(out) >= 20:
                        break
        except Exception as e:
            print(f"Error fetching CoinGecko top 20 by market cap: {e}", file=sys.stderr)

        if len(out) < 20:
            for sym in self.TOP20_FALLBACK:
                raw = self._normalize_symbol(sym)
                base = raw.split("/", 1)[0]
                candidates = [base]
                alias = self.COIN_ALIASES.get(base)
                if alias:
                    candidates.append(alias)
                selected = None
                for cand in candidates:
                    symbol = self._normalize_symbol(f"{cand}/USDT")
                    if tradable and symbol not in tradable:
                        continue
                    selected = symbol
                    break
                if not selected or selected in seen:
                    continue
                seen.add(selected)
                out.append(selected)
                if len(out) >= 20:
                    break

        out = out[:20]
        if out:
            self._save_cached_market_cap_symbols(out)
        return out

    def get_top_20_symbols(self, source: str = "quote_volume") -> list[str]:
        src = (source or "quote_volume").strip().lower()
        if src in {"market_cap", "coingecko", "coingecko_market_cap"}:
            return self.get_top_20_symbols_by_market_cap()

        print("Fetching top 20 symbols by quote volume...", file=sys.stderr)
        if self.harvester.exchange_restricted:
            return self.TOP20_FALLBACK.copy()
        try:
            self.harvester.exchange.load_markets()
            tickers = self.harvester.exchange.fetch_tickers()
            valid_tickers = []
            for symbol, ticker in tickers.items():
                if '/USDT' in symbol and ticker.get('quoteVolume') is not None:
                    valid_tickers.append({"symbol": self._normalize_symbol(symbol), "quoteVolume": ticker["quoteVolume"]})
            sorted_tickers = sorted(valid_tickers, key=lambda x: x['quoteVolume'], reverse=True)
            top_20 = []
            for t in sorted_tickers:
                if t['symbol'] not in top_20: top_20.append(t['symbol'])
                if len(top_20) >= 20: break
            return top_20
        except Exception as e:
            print(f"Error fetching top 20 symbols: {e}", file=sys.stderr)
            return self.TOP20_FALLBACK.copy()

    def sync_top_20(self, start_date_str: str, end_date_str: Optional[str] = None, sync_mode: str = "trades", timeframe: str = "1m", progress_callback: Optional[SyncProgressCallback] = None, symbol_source: str = "quote_volume") -> None:
        symbols = self.get_top_20_symbols(source=symbol_source)
        if not symbols:
            self._emit_progress(progress_callback, 100, "No symbols found for Top 20 sync.")
            return
        total = len(symbols)
        for i, symbol in enumerate(symbols, 1):
            start_p = int(((i - 1) / total) * 100)
            end_p = int((i / total) * 100)
            self._emit_progress(progress_callback, start_p, f"Syncing {symbol} ({i}/{total})")
            def _symbol_progress(p: int, msg: str):
                mapped = start_p + int(((end_p - start_p) * max(0, min(100, int(p)))) / 100)
                self._emit_progress(progress_callback, mapped, f"[{i}/{total}] {symbol}: {msg}")
            try: self.sync_data(symbol, start_date_str, end_date_str, sync_mode=sync_mode, timeframe=timeframe, progress_callback=_symbol_progress)
            except Exception as e: print(f"Failed to sync {symbol}: {e}", file=sys.stderr)
        self._emit_progress(progress_callback, 100, "Top 20 sync complete.")

    def sync_data(self, symbol: str, start_date_str: str, end_date_str: Optional[str] = None, sync_mode: str = "trades", timeframe: str = "1m", progress_callback: Optional[SyncProgressCallback] = None) -> None:
        import fcntl
        lock_path = os.path.join("/tmp", f"sync_{symbol.replace('/', '_')}_{timeframe}.lock")
        lock_file = open(lock_path, "w")
        try:
            fcntl.flock(lock_file, fcntl.LOCK_EX)
            
            symbol = self._normalize_symbol(symbol)
            mode = (sync_mode or "trades").strip().lower()
            tf_raw = (timeframe or "1m").strip()
            
            if mode == "candles_1m": mode, tf_raw = "candles", "1m"
            if mode not in {"trades", "candles", "candles_all"}: raise ValueError(f"Unsupported sync_mode: {mode}")
            
            tf = "all"
            if mode != "candles_all":
                tf = self.db.parse_timeframe(tf_raw)
                if not tf: raise ValueError(f"Unsupported timeframe: {tf_raw}")

            # Check coverage Optimization
            try:
                min_ts, max_ts = self.db.get_ohlcv_min_max(symbol, timeframe=tf)
                if min_ts and max_ts:
                    req_start = pd.to_datetime(start_date_str, utc=True)
                    req_end = pd.to_datetime(end_date_str if end_date_str else datetime.now(timezone.utc), utc=True)
                    if min_ts <= req_start + pd.Timedelta(minutes=5) and max_ts >= req_end - pd.Timedelta(minutes=5):
                        return
            except Exception: pass

            if not self.db.is_available(): raise RuntimeError("QuestDB not reachable")

            self.db.create_trades_table()
            if mode == "candles_all": self.db.create_all_ohlcv_tables()
            else: self.db.create_ohlcv_table(timeframe=tf)

            start_date = pd.to_datetime(start_date_str, utc=True).to_pydatetime()
            end_date = pd.to_datetime(end_date_str, utc=True).to_pydatetime() if end_date_str else datetime.now(timezone.utc)

            if mode == "candles_all":
                self.sync_all_candles(symbol, start_date, end_date, progress_callback=progress_callback)
                return
            if mode == "candles":
                self._sync_candles_only(symbol, start_date, end_date, tf, progress_callback)
                return

            # Trade-based sync
            self._emit_progress(progress_callback, 20, f"Bulk download started for {symbol}...")
            count = 0
            for df in self.bulk_downloader.download_trades(symbol, start_date.isoformat(), end_date.isoformat()):
                if not df.empty:
                    for chunk in self._iter_dataframe_chunks(df, self.bulk_insert_chunk_size):
                        self.db.insert_trades(symbol, chunk)
                        count += len(chunk)
            
            self._emit_progress(progress_callback, 60, f"Bulk complete ({count}). CCXT filling...")
            _, max_ts = self.db.get_trade_min_max(symbol)
            ccxt_start = max_ts.to_pydatetime() if max_ts else start_date
            if ccxt_start.tzinfo is None: ccxt_start = ccxt_start.replace(tzinfo=timezone.utc)

            if ccxt_start < end_date:
                for df in self.harvester.fetch_historical_trades(symbol, ccxt_start.isoformat()):
                    if not df.empty:
                        self.db.insert_trades(symbol, df)
                        if df.iloc[-1]['timestamp'].replace(tzinfo=timezone.utc) >= end_date: break

            self.db.materialize_ohlcv_from_trades(symbol, start_date.isoformat(), end_date.isoformat(), timeframe=tf)
            self._emit_progress(progress_callback, 100, f"Sync complete for {symbol}.")

        finally:
            try:
                fcntl.flock(lock_file, fcntl.LOCK_UN)
                lock_file.close()
            except Exception: pass

    def sync_all_candles(self, symbol: str, start_date: datetime, end_date: datetime, progress_callback: Optional[SyncProgressCallback] = None, timeframes: Optional[list[str]] = None) -> None:
        normalized = []
        for tf in (timeframes or self.ORDERED_TIMEFRAMES):
            parsed = self.db.parse_timeframe(tf)
            if parsed and parsed not in normalized: normalized.append(parsed)
        total = len(normalized)
        for idx, tf in enumerate(normalized, start=1):
            phase_start = 5 + int(((idx - 1) / total) * 95)
            phase_end = 5 + int((idx / total) * 95)
            def _tf_p(p: int, msg: str):
                mapped = phase_start + int(((phase_end - phase_start) * max(0, min(100, int(p)))) / 100)
                self._emit_progress(progress_callback, mapped, f"[{idx}/{total}] {msg}")
            self._sync_candles_only(symbol, start_date, end_date, tf, progress_callback=_tf_p)
        self._emit_progress(progress_callback, 100, f"All timeframes synced for {symbol}.")

    def _sync_candles_only(self, symbol: str, start_date: datetime, end_date: datetime, timeframe: str, progress_callback: Optional[SyncProgressCallback] = None) -> None:
        inserted = 0
        end_ts = pd.Timestamp(end_date).tz_localize(None).tz_localize("UTC") if end_date.tzinfo is None else pd.Timestamp(end_date).tz_convert("UTC")
        try:
            for df in self.harvester.fetch_historical_ohlcv(symbol, timeframe=timeframe, start_date_str=start_date.isoformat()):
                if df.empty: continue
                candles = df.copy()
                candles["timestamp"] = pd.to_datetime(candles["timestamp"], utc=True)
                candles = candles[candles["timestamp"] <= end_ts]
                if candles.empty: continue
                for chunk in self._iter_dataframe_chunks(candles, self.bulk_insert_chunk_size):
                    self.db.insert_ohlcv(symbol, chunk, timeframe=timeframe)
                    inserted += len(chunk)
                self._emit_progress(progress_callback, min(95, 20 + int(inserted / 1000)), f"{timeframe} candles: {inserted}")
                if candles.iloc[-1]["timestamp"] >= end_ts: break
            self._emit_progress(progress_callback, 100, f"{timeframe} complete ({inserted})")
        except Exception as e:
            self._emit_progress(progress_callback, 100, f"Error: {e}")
            raise
