"""Test-Script: Cache-Aufbau und Backtest-Kurzlauf."""
import sys
sys.path.insert(0, '/home/mork/Projekte/EdgeCraft')
import time

from verification.data_cache import load_many, clear_cache

symbols = ["SUI/USDT", "APT/USDT"]
timeframe = "8h"
start = "2024-02-22T00:00:00"
end = "2024-04-01T00:00:00"

print("=== Test 1: Cache-Aufbau (DB-Abfrage) ===")
import os
os.environ["EDGECRAFT_CACHE_TIMEOUT"] = "120"

t0 = time.time()
bars, missing = load_many(symbols, timeframe, start, end, force_refresh=True, verbose=True)
elapsed1 = time.time() - t0
print(f"  Geladen: {len(bars)} Symbole, {len(missing)} fehlend, in {elapsed1:.1f}s")
for sym, df in bars.items():
    print(f"  {sym}: {len(df)} Bars  ({df.index[0]} → {df.index[-1]})")

print("\n=== Test 2: Cache-Lesen (kein DB-Zugriff) ===")
t1 = time.time()
bars2, missing2 = load_many(symbols, timeframe, start, end, verbose=True)
elapsed2 = time.time() - t1
print(f"  Geladen: {len(bars2)} Symbole, in {elapsed2:.1f}s (Speedup: {elapsed1/max(elapsed2,0.001):.0f}×)")
