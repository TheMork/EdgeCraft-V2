"""
data_cache.py – Parquet-basierter OHLCV-Cache für Backtests.

Beim ersten Aufruf werden die Daten von QuestDB direkt per HTTP-Query geladen
(KEIN get_ohlcv()-Wrapper – umgeht den WAL-Backfill-Retry-Loop der hängen kann)
und als Parquet-Datei im Ordner verification/.cache/ gespeichert.
Alle weiteren Aufrufe lesen ausschließlich den lokalen Cache → kein DB-Zugriff.

CLI:
    python verification/data_cache.py --symbols SUI/USDT,BTC/USDT \
        --timeframe 8h --start 2024-01-01T00:00:00 --end 2026-01-01T00:00:00

    python verification/data_cache.py --clear   # Cache leeren
"""
from __future__ import annotations

import argparse
import hashlib
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import requests

# ---------------------------------------------------------------------------
# Pfade
# ---------------------------------------------------------------------------
_THIS_DIR = Path(__file__).resolve().parent
_CACHE_DIR = _THIS_DIR / ".cache"
_PROJECT_ROOT = _THIS_DIR.parent

sys.path.insert(0, str(_PROJECT_ROOT))

# Timeframe → QuestDB-Tabellenname
_TF_TABLE: Dict[str, str] = {
    "1m": "ohlcv", "3m": "ohlcv_3m", "5m": "ohlcv_5m", "15m": "ohlcv_15m",
    "30m": "ohlcv_30m", "1h": "ohlcv_1h", "2h": "ohlcv_2h", "4h": "ohlcv_4h",
    "6h": "ohlcv_6h", "8h": "ohlcv_8h", "12h": "ohlcv_12h", "1d": "ohlcv_1d",
    "3d": "ohlcv_3d", "1w": "ohlcv_1w", "1M": "ohlcv_1mo",
}


# ---------------------------------------------------------------------------
# Interne Hilfsfunktionen
# ---------------------------------------------------------------------------

def _cache_key(symbol: str, timeframe: str, start: str, end: str) -> str:
    raw = f"{symbol}|{timeframe}|{start}|{end}"
    digest = hashlib.md5(raw.encode()).hexdigest()[:12]
    safe_sym = symbol.replace("/", "_").replace(":", "_")
    return f"{safe_sym}_{timeframe}_{digest}"


def _cache_path(symbol: str, timeframe: str, start: str, end: str) -> Path:
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return _CACHE_DIR / f"{_cache_key(symbol, timeframe, start, end)}.parquet"


def _normalize_index(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    idx = pd.to_datetime(df.index, utc=True, errors="coerce")
    df = df.copy()
    df.index = idx
    return df[~df.index.isna()].sort_index()


def _fetch_from_questdb(
    symbol: str,
    timeframe: str,
    start: str,
    end: str,
    host: str = "localhost",
    port: int = 9000,
    timeout: int = 30,
) -> pd.DataFrame:
    """
    Direkter HTTP-Query gegen QuestDB REST API.
    Umgeht get_ohlcv() komplett – kein WAL-Backfill-Loop, kein Hang.
    QuestDB beantwortet solche Queries in Millisekunden.
    """
    table = _TF_TABLE.get(timeframe, "ohlcv_8h")
    safe_sym = symbol.replace("'", "''")
    query = (
        f"SELECT * FROM {table} "
        f"WHERE symbol = '{safe_sym}' "
        f"AND timestamp >= '{start}' AND timestamp <= '{end}' "
        f"ORDER BY timestamp ASC"
    )
    try:
        resp = requests.get(
            f"http://{host}:{port}/exec",
            params={"query": query},
            timeout=timeout,
        )
        resp.raise_for_status()
        data = resp.json()
    except Exception as exc:
        raise RuntimeError(f"QuestDB-Query fehlgeschlagen ({symbol}): {exc}") from exc

    if not data or "dataset" not in data or not data["dataset"]:
        return pd.DataFrame()

    columns = [col["name"] for col in data.get("columns", [])]
    df = pd.DataFrame(data["dataset"], columns=columns)
    if "timestamp" not in df.columns:
        return pd.DataFrame()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    return df.set_index("timestamp").sort_index()


# ---------------------------------------------------------------------------
# Öffentliche API
# ---------------------------------------------------------------------------

def load_or_build_cache(
    symbol: str,
    timeframe: str,
    start: str,
    end: str,
    force_refresh: bool = False,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Gibt ein OHLCV-DataFrame für `symbol` zurück.

    1. Parquet-Cache vorhanden und nicht force_refresh → direkt laden (0 DB-Zugriff).
    2. Sonst: QuestDB per direktem HTTP-Query abfragen, Parquet cachen, zurückgeben.
    """
    path = _cache_path(symbol, timeframe, start, end)

    if not force_refresh and path.exists():
        if verbose:
            print(f"  [cache] HIT  {symbol} ({timeframe}) → {path.name}")
        df = pd.read_parquet(path)
        return _normalize_index(df)

    if verbose:
        print(f"  [cache] MISS {symbol} ({timeframe}) – lade von QuestDB …")

    host = os.environ.get("EDGECRAFT_DB_HOST", "localhost")
    port = int(os.environ.get("EDGECRAFT_DB_REST_PORT", "9000"))
    timeout = int(os.environ.get("EDGECRAFT_CACHE_TIMEOUT", "30"))

    try:
        df = _fetch_from_questdb(symbol, timeframe, start, end, host=host, port=port, timeout=timeout)
    except Exception as exc:
        print(f"  [cache] FEHLER: {exc}", file=sys.stderr)
        return pd.DataFrame()

    df = _normalize_index(df)
    if df.empty:
        if verbose:
            print(f"  [cache] Keine Daten für {symbol}.")
        return df

    cols = ["open", "high", "low", "close", "volume"]
    for col in cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=[c for c in cols if c in df.columns])
    if df.empty:
        return df

    df[cols].to_parquet(path, index=True, compression="snappy")
    if verbose:
        print(f"  [cache] Gespeichert: {path.name} ({len(df)} Bars)")

    return df[cols]


def load_many(
    symbols: List[str],
    timeframe: str,
    start: str,
    end: str,
    force_refresh: bool = False,
    verbose: bool = True,
) -> Tuple[Dict[str, pd.DataFrame], List[str]]:
    """Lädt mehrere Symbole – gibt (bars_by_symbol, missing_symbols) zurück."""
    bars: Dict[str, pd.DataFrame] = {}
    missing: List[str] = []
    for sym in symbols:
        df = load_or_build_cache(sym, timeframe, start, end,
                                  force_refresh=force_refresh, verbose=verbose)
        if df.empty:
            missing.append(sym)
        else:
            bars[sym] = df
    return bars, missing


def clear_cache(symbol: Optional[str] = None, timeframe: Optional[str] = None) -> int:
    """Löscht Cache-Dateien. Gibt Anzahl gelöschter Dateien zurück."""
    if not _CACHE_DIR.exists():
        return 0
    deleted = 0
    for f in _CACHE_DIR.glob("*.parquet"):
        if symbol and symbol.replace("/", "_").replace(":", "_") not in f.name:
            continue
        if timeframe and f"_{timeframe}_" not in f.name:
            continue
        f.unlink()
        deleted += 1
    return deleted


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _main() -> None:
    parser = argparse.ArgumentParser(description="OHLCV Parquet Cache Manager")
    parser.add_argument("--symbols", default="BTC/USDT")
    parser.add_argument("--timeframe", default="8h")
    parser.add_argument("--start", default="2024-01-01T00:00:00")
    parser.add_argument("--end", default="2026-01-01T00:00:00")
    parser.add_argument("--refresh", action="store_true")
    parser.add_argument("--clear", action="store_true")
    args = parser.parse_args()

    if args.clear:
        print(f"Cache geleert: {clear_cache()} Datei(en) gelöscht.")
        return

    syms = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    bars, missing = load_many(syms, args.timeframe, args.start, args.end, force_refresh=args.refresh)

    print(f"\nGeladen: {len(bars)} Symbole | Fehlend: {len(missing)}")
    for sym, df in bars.items():
        print(f"  {sym}: {len(df)} Bars ({df.index[0]} … {df.index[-1]})")
    if missing:
        print(f"  Fehlend: {', '.join(missing)}")


if __name__ == "__main__":
    _main()
