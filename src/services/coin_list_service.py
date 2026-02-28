import pandas as pd
from datetime import datetime, timezone
from pathlib import Path
from urllib import parse
import sys
import os
import requests
from src.harvester import DataHarvester

class CoinListService:
    TOP20_FALLBACK = [
        "BTC/USDT", "ETH/USDT", "SOL/USDT", "BNB/USDT", "XRP/USDT",
        "DOGE/USDT", "ADA/USDT", "AVAX/USDT", "TRX/USDT", "LINK/USDT",
        "DOT/USDT", "MATIC/USDT", "SHIB/USDT", "LTC/USDT", "UNI/USDT",
        "ATOM/USDT", "XLM/USDT", "ETC/USDT", "FIL/USDT", "HBAR/USDT",
    ]
    MARKET_CAP_CACHE_FILE = Path("/tmp/edgecraft_top20_market_cap.json")
    STABLE_BASES = {
        "USDT", "USDC", "USDE", "USDS", "DAI", "FDUSD", "TUSD", "USDP", "BUSD", "USDD", "USD1", "PYUSD", "FRAX",
    }
    COIN_ALIASES = {
        "MATIC": "POL",
        "SHIB": "1000SHIB",
    }

    def __init__(self, harvester: DataHarvester = None):
        self.harvester = harvester if harvester else DataHarvester()

    def _normalize_symbol(self, symbol: str) -> str:
        s = symbol.strip().upper()
        if ":" in s: s = s.split(":", 1)[0]
        if "/" in s: return s
        if s.endswith("USDT") and len(s) > 4: return f"{s[:-4]}/USDT"
        return s

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

    def get_top_n_futures_usdt(self, n: int = 20) -> list[str]:
        # Using get_top_20_symbols but limiting to n, if n <= 20 it's exact, else it only returns 20 max currently
        # A full fetch would require modifying get_top_20_symbols to take N, but top 20 is the requirement
        return self.get_top_20_symbols()[:n]

    def get_available_futures_symbols(self) -> list[str]:
        return list(self._load_tradable_usdt_symbols())
