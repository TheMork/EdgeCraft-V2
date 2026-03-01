# EdgeCraft – Scrum-Projektplan
> **Datenbank:** TimescaleDB (PostgreSQL-Extension) – ersetzt QuestDB für maximale Stabilität bei gleicher Zeitreihen-Performance.

**Ziel:** EdgeCraft zu einer vollständig funktionsfähigen, hochperformanten Backtesting-Station ausbauen – mit automatischem Daten-Download aller gewünschten Coins (Top-20 oder Auswahl), einer professionellen UI mit Parametersweep und allem Komfort, den eine echte Research-Plattform bietet.

---

## Aktueller Ist-Zustand

| Bereich | Status | Problem |
|---|---|---|
| **Datenbank (QuestDB)** | ❌ Wird ersetzt | Wird durch **TimescaleDB** (PostgreSQL) ersetzt – stabiler, echtes Standard-SQL |
| **Daten-Download** | ⚠️ Sequenziell | `BinanceBulkDownloader` lädt pro Symbol nacheinander, kein Multithreading |
| **Broker** | ✅ Rust-Bridge | `simulation_core.Broker` vorhanden, aber Limit-Orders / SL/TP fehlen |
| **Backtesting-Runner** | ✅ Funktional | Einzel-Symbol funktioniert, kein Multi-Asset-Parallel-Betrieb |
| **Parameter-Sweep** | ⚠️ Ad-hoc | Diverse `sweep_*.py` Skripte, kein einheitliches Framework |
| **Frontend-UI** | ❌ Rudimentär | Kein richtiges Backtest-Panel, kein Sweep-Interface, keine Coin-Auswahl |
| **Strategien** | ✅ 15 Strategien | Vorhanden, aber kein dynamisches Discovery-System voll genutzt |
| **Tests** | ⚠️ Lückig | Einzelne Testdateien im Root, kein CI-Durchlauf |

---

## Epics & Sprints

---

### 🔴 EPIC 1 – Migration zu TimescaleDB (PostgreSQL)
**Ziel:** QuestDB vollständig durch TimescaleDB ersetzen. Standard-SQL, natives Connection-Pooling, maximale Stabilität durch PostgreSQL-Reife.

#### Sprint 1.1 – TimescaleDB Setup & Schema (1 Woche)
- [ ] **1.1.1** `docker-compose.yml` anpassen: QuestDB-Service ersetzen durch `timescale/timescaledb:latest-pg16` auf Port 5432
- [ ] **1.1.2** Schema-Init-Skript (`init_timescale.py`) erstellen:
  - Extension aktivieren: `CREATE EXTENSION IF NOT EXISTS timescaledb`
  - Hypertable für jedes OHLCV-Timeframe: `CREATE TABLE ohlcv_1h (...); SELECT create_hypertable('ohlcv_1h', 'timestamp')`
  - Tabellen: `ohlcv_*` (alle Timeframes), `funding_rates`, `trades`
  - UNIQUE-Index auf `(timestamp, symbol)` für UPSERT (entspricht QuestDB DEDUP)
- [ ] **1.1.3** `TimescaleDBManager` (`src/timescale_database.py`) implementieren:
  - Connection via `psycopg2` mit `ThreadedConnectionPool` (echtes Connection-Pooling)
  - Methoden 1:1 zu `QuestDBManager`: `get_ohlcv()`, `insert_ohlcv()`, `get_funding_rates()`, `insert_funding()`, `is_available()`, `get_ohlcv_min_max()`
  - Bulk-Insert via `execute_values` / `COPY`-Statement (statt Zeile für Zeile) für maximale Schreibgeschwindigkeit
  - UPSERT via `INSERT ... ON CONFLICT (timestamp, symbol) DO UPDATE SET ...`
- [ ] **1.1.4** `src/database.py` (`QuestDBManager`) durch `TimescaleDBManager` austauschen – alle Aufrufer (`runner.py`, `data_manager.py`, `batch_service.py`, API-Routes) auf neue Klasse umstellen
- [ ] **1.1.5** Health-Check-Endpunkt `/api/v1/db/health` exposieren (nutzt `is_available()`)
- [ ] **1.1.6** `requirements.txt` aktualisieren: `psycopg2-binary` hinzufügen, `questdb`-Paket entfernen
- [ ] **1.1.7** Unit-Tests mit `pytest-postgresql` oder `testing.postgresql` (`tests/test_timescale_manager.py`)

---

### 🟠 EPIC 2 – Multithreaded & Automatischer Daten-Download
**Ziel:** Alle gewünschten Coins (Top-20 Binance Futures oder freie Auswahl) vollautomatisch und parallel herunterladen.

#### Sprint 2.1 – Concurrent Bulk-Downloader (1–2 Wochen)
> *TimescaleDB (Sprint 1.1) muss abgeschlossen sein.*
- [x] **2.1.1** `BinanceBulkDownloader` für Concurrent-Downloads umbauen:
  - `ThreadPoolExecutor` (I/O-bound wegen HTTP) mit konfigurierter Worker-Anzahl (Standard: 8)
  - Pro (Symbol × Zeitraum) Task parallel – Ergebnisse werden gesammelt und in Batches in TimescaleDB geschrieben
- [x] **2.1.2** `CoinListService` (`src/services/coin_list_service.py`) implementieren:
  - `get_top_n_futures_usdt(n: int)` – ruft Binance Public API ab und liefert Top-N nach 24h-Volumen
  - `get_available_futures_symbols()` – vollständige Liste aller USDT-Perpetuals
- [x] **2.1.3** Download-Jobs-Datenstruktur: `DownloadJob(symbol, start, end, timeframes[])` + Progress-Tracking
- [x] **2.1.4** API-Endpunkt `POST /api/v1/data/sync` mit Body:
  ```json
  { "symbols": ["BTCUSDT", "ETHUSDT"], "start": "2024-01-01", "end": "2024-12-31", "timeframes": ["1h", "4h", "1d"] }
  ```
- [x] **2.1.5** API-Endpunkt `GET /api/v1/data/sync/status` – gibt laufenden Fortschritt zurück (Symbol, %, ETA)
- [x] **2.1.6** WebSocket-Stream `/ws/data-sync` für Echtzeit-Progress im Frontend
- [x] **2.1.7** Ingest in TimescaleDB via Bulk-`execute_values` in Batches (mind. 10.000 Rows pro Batch) für maximale Schreibgeschwindigkeit
- [x] **2.1.8** Tests: Paralleler Download-Mock, Fortschritts-Reporting (`tests/test_bulk_downloader_concurrent.py`)

#### Sprint 2.2 – Candle-Download (direkt, ohne Trades) (1 Woche)
> *Für längere Zeiträume ist der direkte Candle-Download via Binance Vision effizienter als Trades aggregieren.*
- [x] **2.2.1** `BinanceCandleDownloader` (`src/bulk_candle_downloader.py`) implementieren:
  - Lädt OHLCV direkt aus `data.binance.vision/data/futures/um/monthly/klines/`
  - Unterstützt alle Timeframes (1m, 5m, 15m, 1h, 4h, 1d …)
  - Fallback auf Daily-ZIPs wenn monatliche nicht verfügbar
- [x] **2.2.2** `SyncManager` (`src/data_manager.py`) um `sync_mode="candles"` erweitern (nutzt neuen Downloader)
- [x] **2.2.3** Automatische Gap-Detection: prüft welche Zeiträume in TimescaleDB fehlen und lädt nur diese nach
- [x] **2.2.4** Tests: Gap-Detection-Logik, Candle-Parse (`tests/test_candle_downloader.py`)

---

### 🟡 EPIC 3 – Backtesting-Engine vervollständigen
**Ziel:** Realistische Order-Simulation mit Limit-Orders, SL/TP, Margin, Funding.

#### Sprint 3.1 – Broker & Order-Management (1–2 Wochen)
- [ ] **3.1.1** Rust-Broker (`rust_extension`) um Limit-Orders erweitern:
  - Fill-Logik: wenn `low <= limit_price` (buy) oder `high >= limit_price` (sell) innerhalb der Bar
- [ ] **3.1.2** Stop-Loss / Take-Profit als eigenständige Order-Typen (OCO – One-Cancels-Other) im Rust-Broker
- [ ] **3.1.3** Liquidation-Engine (`src/simulation/liquidation.py`) in Broker integrieren – wird pro Bar geprüft
- [ ] **3.1.4** Slippage-Model: konfigurierbar (0 bp bis N bp, oder volumenbasiert)
- [ ] **3.1.5** Taker/Maker-Fees korrekt abziehen (derzeit: bereits vorhanden, prüfen ob korrekt)
- [ ] **3.1.6** Funding-Rate-Verrechnung validieren und Fehler beheben
- [ ] **3.1.7** Tests: Order-Fill-Edge-Cases, Liquidation, Funding (`tests/test_broker_advanced.py`)

#### Sprint 3.2 – Multi-Asset-Backtesting (1 Woche)
- [ ] **3.2.1** `MultiAssetSimulationRunner` – führt mehrere Symbole gleichzeitig mit einem gemeinsamen Broker aus
- [ ] **3.2.2** Portfoliogewichtung & Margin-Sharing über Assets hinweg
- [ ] **3.2.3** API-Endpunkt `POST /api/v1/simulation/multi` für Multi-Asset-Runs
- [ ] **3.2.4** Tests: Portfolio-Equity-Berechnung bei mehreren offenen Positionen

---

### 🟢 EPIC 4 – Parameter-Sweep-Framework
**Ziel:** Einheitliches, wiederverwendbares Framework statt Ad-hoc-Sweep-Skripten. UI-gesteuerter Parametersweep per Klick.

#### Sprint 4.1 – Backend Sweep-Engine (1–2 Wochen)
- [x] **4.1.1** `SweepEngine` (`src/optimization/sweep_engine.py`) als zentrale Klasse:
  - Nimmt `SweepConfig` (Strategie-Klasse, Param-Grid als Dict mit Listen) entgegen
  - Führt Grid-Search mit `ProcessPoolExecutor` durch (CPU-bound → Prozesse, nicht Threads)
  - Schreibt Ergebnisse in eine SQLite-Datenbank (`results/sweeps.db`) für spätere Abfrage
- [x] **4.1.2** `SweepResult`-Datenmodell: alle Parameter + Metriken (Sharpe, Total Return, Max Drawdown, Win-Rate, Trades)
- [x] **4.1.3** Bayesianische Optimierung als Alternative zur Grid-Search einbinden (via `optuna`)
- [x] **4.1.4** API-Endpunkte:
  - `POST /api/v1/sweep/start` – startet einen Sweep-Job (gibt `job_id` zurück)
  - `GET /api/v1/sweep/{job_id}/status` – Fortschritt (N von M Kombinationen fertig)
  - `GET /api/v1/sweep/{job_id}/results` – Ergebnistabelle sortiert nach Sharpe-Ratio
  - `DELETE /api/v1/sweep/{job_id}` – Sweep abbrechen
- [x] **4.1.5** WebSocket `/ws/sweep/{job_id}` für Echtzeit-Fortschritt im Frontend
- [x] **4.1.6** Die bestehenden `sweep_*.py`-Skripte im Root auf die neue `SweepEngine` umstellen (oder als Deprecated markieren)
- [x] **4.1.7** Tests: Grid-Erzeugung, Parallelität, Ergebnis-Serialisierung (`tests/test_sweep_engine.py`)

---

### 🔵 EPIC 5 – Professionelle Backtesting-UI (Frontend)
**Ziel:** Eine UI, die einer echten Backtesting-Station würdig ist – konfigurierbar, übersichtlich, per Klick bedienbar.

#### Sprint 5.1 – Coin-Auswahl & Datenverwaltung (1 Woche)
- [x] **5.1.1** **Data-Manager-Seite** (`/data`):
  - „Top 20 laden"-Button – ruft `CoinListService.get_top_n_futures_usdt(20)` auf
  - Coin-Liste mit Checkboxen zur Einzel-Auswahl
  - Zeitraumauswahl (DatePicker: Start / End) + Timeframe-Multiselect
  - „Download starten"-Button → POST auf `/api/v1/data/sync`
  - Fortschrittsanzeige per WebSocket (Symbol, %, ETA, Gesamt-Balken)
  - Tabelle: bereits verfügbare Daten (Symbol, Timeframe, Von, Bis, Candle-Anzahl)

#### Sprint 5.2 – Backtesting-Konfigurationspage (1–2 Wochen)
- [x] **5.2.1** **Simulation-Seite** (`/simulation`) – komplett überarbeiten:
  - Symbol-Auswahl (Dropdown oder Multi-Select aus verfügbaren Coins)
  - Zeitraum: Start / End (DatePicker)
  - Timeframe-Dropdown
  - Strategie-Dropdown (dynamisch aus `/api/v1/simulation/strategies`)
  - Strategie-Parameter als dynamisches Formular (JSON-Schema-basiert aus Strategy-Klasse)
  - Kapital, Leverage, Fees, Slippage (bps)
  - „Simulation starten"-Button → startet Backtest via WebSocket/API
  - Real-Time-Chart: Candlestick + Buy/Sell-Marker + Equity-Kurve (Lightweight Charts)
  - Metriken-Panel: Sharpe, Total Return, Max Drawdown, Win-Rate, Trade-Anzahl
  - Trade-Tabelle: alle Trades mit Entry/Exit/PnL

#### Sprint 5.3 – Parameter-Sweep-UI (1–2 Wochen)
- [x] **5.3.1** **Sweep-Seite** (`/sweep`):
  - Strategie wählen
  - Pro Parameter: Min / Max / Step (Slider oder Eingabe) → erzeugt Grid
  - Symbol(e) auswählen (Checkboxen, inkl. „Top-10 automatisch")
  - Zeitraum, Timeframe, Kapital, Fees
  - „Sweep starten"-Button
  - Echtzeit-Fortschrittsbalken (N / M Kombinationen, ETA)
  - Ergebnistabelle: sortierbar nach Sharpe, Total Return, Drawdown
  - „Beste Konfiguration übernehmen"-Button → befüllt Simulation-Seite vorab
  - Export: Ergebnisse als CSV herunterladen
- [x] **5.3.2** Heatmap-Visualisierung für 2-Parameter-Sweeps (z. B. Lookback × Risk-Aversion)

#### Sprint 5.4 – UI-Komfortfunktionen (1 Woche)
- [x] **5.4.1** Favoriten-/Watchlist-System: Coin-Sets speichern (z. B. „Top 20", „DeFi-Basket")
- [x] **5.4.2** Backtest-Ergebnis-History: alle vergangenen Runs auflisten, laden, vergleichen
- [x] **5.4.3** Dark Mode (systembasiert), responsive Layout
- [x] **5.4.4** Notifications-System: Toast-Meldungen für abgeschlossene Downloads / Sweeps

---

### ⚪ EPIC 6 – Performance & Code-Qualität
**Ziel:** Sauberer Code, stabile Tests, Rust wo nötig.

#### Sprint 6.1 – Rust-Extensions (parallel zu Epic 3)
- [x] **6.1.1** Rust-Broker Limit/SL/TP fertigstellen (s. Sprint 3.1)
- [x] **6.1.2** Matching-Engine als Rust-Extension für den Sweep (CPU-bound, massiv parallel)
- [x] **6.1.3** OHLCV-Indikator-Berechnung in Rust auslagern (RSI, EMA, ATR) – für Sweep-Performance

#### Sprint 6.2 – Testinfrastruktur (1 Woche)
- [x] **6.2.1** `pytest` als einheitlicher Test-Runner, `conftest.py` mit TimescaleDB-Mock-Fixture (`pytest-postgresql`)
- [x] **6.2.2** GitHub-Actions-Workflow (`.github/workflows/ci.yml`): Tests bei jedem Push auf `main`
- [x] **6.2.3** Coverage-Report (Ziel: >70%)
- [x] **6.2.4** `start_dev.sh` überarbeiten: startet TimescaleDB (Docker), dann FastAPI, dann Next.js in der richtigen Reihenfolge mit Health-Check-Wartelogik

---

### 🟣 EPIC 7 – Strategy-Plugin-System (Drop-in Format)
**Ziel:** Eine Strategie = eine Datei. Einfach in `src/simulation/strategies/` ablegen → automatisch erkannt, im Backend registriert und im Frontend auswählbar. Einheitliches Interface, das alle Strategien einhalten müssen.

#### Sprint 7.1 – Standardisiertes Strategy-Interface & Metadaten (1 Woche)
- [x] **7.1.1** `Strategy`-Basisklasse (`src/simulation/strategy.py`) um Pflicht-Metadaten erweitern:
  ```python
  class Strategy(ABC):
      # Jede Strategie MUSS diese Klassenvariablen definieren
      NAME: str = ""              # Anzeigename in der UI
      DESCRIPTION: str = ""       # Kurzbeschreibung
      VERSION: str = "1.0.0"      # Versionsnummer
      AUTHOR: str = ""            # Optional
      SUPPORTED_TIMEFRAMES: list  # z.B. ["1h", "4h", "1d"]
  ```
- [x] **7.1.2** `get_param_schema()` als Classmethod zur Basisklasse hinzufügen:
  - Gibt JSON-Schema zurück, das alle `__init__`-Parameter beschreibt (Name, Typ, Default, Min, Max, Beschreibung)
  - Wird vom Frontend genutzt, um das Konfigurations-Formular dynamisch zu bauen
  - Beispiel-Output:
    ```json
    {
      "lookback_bars": { "type": "int", "default": 30, "min": 5, "max": 200, "description": "Lookback-Periode" },
      "risk_aversion":  { "type": "float", "default": 2.0, "min": 0.1, "max": 10.0 }
    }
    ```
- [x] **7.1.3** Alle 15 bestehenden Strategien in `src/simulation/strategies/` mit `NAME`, `DESCRIPTION`, `SUPPORTED_TIMEFRAMES` und `get_param_schema()` nachrüsten
- [x] **7.1.4** `strategies/TEMPLATE.py` – leere Vorlage mit allen Pflichtfeldern und Kommentaren erstellen (zum Kopieren für neue Strategien)
- [x] **7.1.5** `docs/strategy_development.md` – Entwicklerdokumentation: Wie schreibe ich eine neue Strategie?

#### Sprint 7.2 – Strategy-Registry (Auto-Discovery) (0.5 Wochen)
- [x] **7.2.1** `StrategyRegistry` (`src/simulation/registry.py`) implementieren:
  - Scannt `src/simulation/strategies/` beim Start via `importlib` (kein manuelles Registrieren nötig)
  - Lädt alle `.py`-Dateien die eine `Strategy`-Subklasse enthalten
  - Ignoriert Dateien die mit `_` beginnen (Private/Templates)
  - Stellt bereit: `get_all()`, `get_by_name(name)`, `list_names()`
- [x] **7.2.2** `GET /api/v1/simulation/strategies` – gibt Liste aller registrierten Strategien zurück (Name, Beschreibung, Version, unterstützte Timeframes, Param-Schema)
- [x] **7.2.3** `SimulationRunner` nutzt Registry statt direktem Import – Strategie wird per Name instanziiert
- [x] **7.2.4** Tests: Registry lädt korrekte Strategien, ignoriert invalide Dateien (`tests/test_strategy_registry.py`)

#### Sprint 7.3 – Test-Migration für Strategien (0.5 Wochen)
- [x] **7.3.1** Alle strategie-spezifischen Tests aus dem Root (`test_runner_sync.py`, `test_cache.py`, etc.) nach `tests/strategies/` verschieben
- [x] **7.3.2** Pro Strategie: Minimal-Testfall der prüft ob sie läuft ohne Fehler (nutzt `MockQuestDBManager`)
- [x] **7.3.3** Integrations-Test: Registry-Discovery + Backtest-Run mit einer Strategie end-to-end

---

### 🧹 EPIC 8 – Projektaufräumen & Struktur
**Ziel:** Nur relevante Dateien im Root. Klare Verzeichnisstruktur. Keine Debug-Skripte, Test-Dateien oder alte Artefakte im Root.

#### Sprint 8.1 – Root-Cleanup (0.5 Wochen)
- [x] **8.1.1** Folgende Dateien aus dem Root **entfernen oder nach `tests/` verschieben**:
  - `check_data.py`, `debug_data_check.py`, `debug_sender.py`, `debug_sender_v2.py` → `tools/debug/`
  - `test_cache.py`, `test_date_query.py`, `test_db_connection.py`, `test_event_loop.py`, `test_ohlcv_aggregation.py`, `test_runner_sync.py`, `test_ws.py` → `tests/`
  - `sweep_classic_mde.py`, `sweep_improved_mde.py`, `sweep_mde_optimized.py`, `sweep_mde_optimized_top20.py`, `sweep_timeframes_mde.py`, `sweep_v4_sl.py` → `tools/sweeps/` (bis SweepEngine fertig, dann deprecated)
  - `init_db.py`, `init_db_dummy_2024.py`, `populate_data.py`, `truncate_ohlcv.py` → in TimescaleDB-Init-Skript (`init_timescale.py`) integrieren, dann löschen
  - `server_BotRunner.ts`, `server_StrategyEngine.ts`, `server_files.txt` → prüfen ob noch gebraucht, ggf. nach `archive/` oder löschen
- [x] **8.1.2** Root darf danach nur noch enthalten: `README.md`, `SCRUM_PLAN.md`, `docker-compose.yml`, `requirements.txt`, `start_dev.sh`, `conftest.py`, `.gitignore`

#### Sprint 8.2 – Verzeichnisstruktur finalisieren (0.5 Wochen)
- [x] **8.2.1** Finale Verzeichnisstruktur herstellen:
  ```
  EdgeCraft/
  ├── src/
  │   ├── api/              (FastAPI-Routes)
  │   ├── simulation/
  │   │   ├── strategies/   (Drop-in Strategie-Dateien)
  │   │   │   └── TEMPLATE.py
  │   │   └── registry.py   (Auto-Discovery)
  │   ├── services/         (CoinListService, SyncManager)
  │   └── optimization/     (SweepEngine)
  ├── tests/
  │   ├── strategies/       (Pro-Strategie Tests)
  │   └── ...               (Sonstige Tests)
  ├── tools/
  │   ├── debug/            (Debug-Skripte)
  │   └── sweeps/           (Legacy Sweep-Skripte)
  ├── frontend/             (Next.js)
  ├── docs/
  │   └── strategy_development.md
  ├── docker-compose.yml
  └── requirements.txt
  ```
- [x] **8.2.2** `README.md` aktualisieren: Projektübersicht, Setup-Anleitung, Link zu `docs/strategy_development.md`
- [x] **8.2.3** `.gitignore` prüfen: `results/`, `*.db`, `__pycache__`, `node_modules`, `.env` einschließen

---

## Sprint-Reihenfolge (Empfehlung)

```
Woche 1:    Sprint 8.1 + 8.2 (Aufräumen)         →  Saubere Basis
Woche 1-2:  Sprint 1.1 (TimescaleDB-Migration)   →  Stabiles Fundament
Woche 2:    Sprint 7.1 + 7.2 (Strategy-Plugin)   →  Drop-in Strategien
Woche 2-3:  Sprint 7.3 (Test-Migration)          →  Testabdeckung
Woche 3-4:  Sprint 2.1 + 2.2 (Download)          →  Daten verfügbar machen
Woche 4-5:  Sprint 3.1 + 3.2 (Broker)            →  Engine zuverlässig
Woche 5-6:  Sprint 4.1 (Sweep-Backend)            →  Optimierung möglich
Woche 6-7:  Sprint 5.1 + 5.2 (UI)               →  UI-MVP
Woche 7-8:  Sprint 5.3 + 5.4 (Sweep-UI)          →  Vollständige Backtest-Station
Woche 8-9:  Sprint 6.1 + 6.2 (Perf+CI)           →  Polishing
```

---

## Definition of Done (DoD)

- [x] Alle Unit-Tests grün (`pytest tests/`)
- [ ] TimescaleDB läuft stabil – kein Absturz oder Timeout bei normaler Last
- [ ] Coin-Download: Top-20 USDT-Perpetuals lassen sich per Klick herunterladen (1h, 4h, 1d, mind. 2 Jahre)
- [ ] Backtest: Vollständig konfigurierbar über UI, inkl. Fees & Slippage
- [ ] Sweep: Grid-Search über ≥3 Parameter läuft parallel, Ergebnisse in sortierter Tabelle
- [x] Strategy-Plugin: Neue `.py`-Datei in `strategies/` ablegen → erscheint automatisch in der UI
- [x] Strategy-Interface: Alle Strategien haben `NAME`, `DESCRIPTION`, `SUPPORTED_TIMEFRAMES`, `get_param_schema()`
- [ ] Strategy-Isolation: Strategy kann Balance nie direkt modifizieren (nur über Broker)
- [ ] Performance: Ein 1-Jahres-1h-Backtest (8760 Bars) läuft in < 5 Sekunden
- [ ] Kein Look-Ahead-Bias: Strategy erhält nur abgeschlossene Bars
- [x] Projektstruktur: Root enthält nur die 7 definierten Kerndateien, alle anderen sind korrekt einsortiert

---

## Technische Architektur-Übersicht

```
┌─────────────────────────────────────────────────────────┐
│                      Next.js Frontend                    │
│  /data (Download)  /simulation  /sweep  /history        │
└──────────────────────┬──────────────────────────────────┘
                       │ HTTP / WebSocket
┌──────────────────────▼──────────────────────────────────┐
│                    FastAPI Backend                       │
│  /api/v1/data   /api/v1/simulation   /api/v1/sweep      │
│  CoinListService  SyncManager  SweepEngine              │
└───────┬──────────────────┬──────────────────────────────┘
        │                  │
┌───────▼────────────┐  ┌──▼───────────────────────────────┐
│   TimescaleDB      │  │  ProcessPoolExecutor (Sweep)      │
│  (PostgreSQL +     │  │  ThreadPoolExecutor (Downloads)  │
│   Hypertables)     │  │  Rust: simulation_core (Broker)  │
│  Port 5432         │  └──────────────────────────────────┘
│  psycopg2 Pool     │
└────────────────────┘
```
