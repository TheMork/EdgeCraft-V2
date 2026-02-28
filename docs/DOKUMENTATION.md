# Dokumentation: Institutional Crypto Backtesting Engine

## Inhaltsverzeichnis

1. [Allgemeine Funktionen, Installation und Möglichkeiten](#1-allgemeine-funktionen-installation-und-möglichkeiten)
2. [Erstellung von Strategien, Backtesting und weiteres](#2-erstellung-von-strategien-backtesting-und-weiteres)
3. [API Dokumentation](#3-api-dokumentation)

---

## 1. Allgemeine Funktionen, Installation und Möglichkeiten

### Überblick
Diese Software ist eine leistungsstarke Backtesting-Engine für institutionellen Kryptohandel, die speziell für Binance Futures entwickelt wurde. Sie zeichnet sich durch eine hybride Architektur aus, die die Flexibilität von Python für die Strategieentwicklung mit der Geschwindigkeit von Rust für die Kernsimulation verbindet.

**Hauptmerkmale:**
*   **Event-Driven Architecture:** Realistische Simulation durch ereignisbasierte Verarbeitung (Marktdaten, Funding Rates, Order Fills).
*   **Rust Core:** Der Matching-Engine, das Risikomanagement und der Order-Status werden in einer hochperformanten Rust-Erweiterung (`simulation_core`) verwaltet.
*   **QuestDB Integration:** Nutzung einer High-Performance Time-Series Datenbank für schnellen Zugriff auf historische Tick- und Kerzendaten.
*   **Web-Interface:** Ein modernes Frontend (Next.js) zur Visualisierung von Backtests und Analysen.

### Installation

Voraussetzungen:
*   Python 3.10+
*   Rust (Cargo)
*   Docker & Docker Compose
*   Node.js (für das Frontend)

**Schritt-für-Schritt Anleitung:**

1.  **Repository klonen:**
    ```bash
    git clone <repo-url>
    cd <repo-name>
    ```

2.  **Datenbank starten (QuestDB):**
    Starten Sie die Datenbank im Hintergrund.
    ```bash
    docker-compose up -d
    ```
    Das Admin-Panel ist unter `http://localhost:9000` erreichbar.

3.  **Rust-Erweiterung kompilieren:**
    Installieren Sie `maturin` und kompilieren Sie den Rust-Kern.
    ```bash
    cd rust_extension
    # Erstellt eine virtuelle Umgebung (empfohlen) oder installiert global
    pip install maturin
    maturin develop --release
    cd ..
    ```

4.  **Backend-Abhängigkeiten installieren:**
    ```bash
    pip install -r requirements.txt
    ```

5.  **Frontend installieren und starten:**
    ```bash
    cd frontend
    npm install
    npm run dev
    ```
    Das Frontend ist nun unter `http://localhost:3000` erreichbar.

6.  **Backend starten:**
    ```bash
    # Im Root-Verzeichnis
    uvicorn src.api.main:app --reload
    ```
    Die API ist unter `http://localhost:8000` erreichbar.

---

## 2. Erstellung von Strategien, Backtesting und weiteres

Die Strategieentwicklung erfolgt in Python. Jede Strategie erbt von der Basisklasse `Strategy`.

### Strategie-Entwicklung

Erstellen Sie eine neue Datei (z.B. `src/simulation/strategies/my_strategy.py`) und implementieren Sie Ihre Logik.

**Basis-Struktur:**

```python
from src.simulation.strategy import Strategy
from src.simulation.models import OrderType
from typing import Dict, Any

class MyStrategy(Strategy):
    def on_start(self):
        """Wird beim Start der Simulation aufgerufen."""
        print("Strategie gestartet.")

    def on_stop(self):
        """Wird am Ende der Simulation aufgerufen."""
        print("Strategie beendet.")

    def on_bar(self, bar: Dict[str, Any]):
        """
        Wird für jede neue Kerze (Bar) aufgerufen.
        bar enthält: 'symbol', 'open', 'high', 'low', 'close', 'volume', 'timestamp'
        """
        symbol = bar['symbol']
        close = bar['close']
        timestamp = bar['timestamp']

        # Zugriff auf Positionsgröße (positiv = Long, negativ = Short)
        position_size = self.get_position_size(symbol)

        # Beispiel Logik:
        if position_size == 0:
            # Kauf (Long)
            # Menge, Preis, Timestamp, OrderType (Standard: MARKET)
            quantity = 0.1
            if self.balance > (quantity * close):
                self.buy(symbol, quantity, close, timestamp)

        elif position_size > 0 and close > 50000:
            # Verkauf (Close Long)
            self.sell(symbol, position_size, close, timestamp)

    def on_fill(self, trade):
        """Optional: Wird aufgerufen, wenn eine Order ausgeführt wurde."""
        print(f"Trade ausgeführt: {trade}")
```

**Wichtige Methoden und Eigenschaften:**

*   `self.buy(symbol, quantity, price, timestamp, type=OrderType.MARKET)`: Sendet eine Kauf-Order.
*   `self.sell(symbol, quantity, price, timestamp, type=OrderType.MARKET)`: Sendet eine Verkaufs-Order.
*   `self.balance`: Aktuelles Guthaben (Cash).
*   `self.positions`: Dictionary der offenen Positionen.
*   `self.get_position_size(symbol)`: Gibt die aktuelle Größe der Position zurück (0.0 wenn keine).
*   `self.broker`: Zugriff auf das unterliegende Broker-Objekt (Rust).

### Backtesting ausführen

Sie haben zwei Möglichkeiten, einen Backtest zu starten:

**1. Über das Web-Interface (Empfohlen für visuelle Analyse):**
*   Öffnen Sie `http://localhost:3000`.
*   Navigieren Sie zur Simulations-Seite.
*   Wählen Sie Symbol, Zeitraum, Startkapital und Hebel.
*   Starten Sie die Simulation. Die Trades und der Equity-Verlauf werden in Echtzeit angezeigt.

**2. Programmatisch (Python Skript):**
Sie können den `SimulationRunner` direkt verwenden.

```python
from src.simulation.runner import SimulationRunner
from src.simulation.strategies.my_strategy import MyStrategy

strategy = MyStrategy()
runner = SimulationRunner(
    strategy=strategy,
    symbol="BTC/USDT",
    start_date="2023-01-01T00:00:00Z",
    end_date="2023-02-01T00:00:00Z",
    initial_balance=10000.0,
    leverage=1
)

# Startet den Backtest
result = runner.run()

print("Metriken:", result["metrics"])
```

---

## 3. API Dokumentation

Das Backend stellt eine REST-API und WebSocket-Endpunkte zur Verfügung.
Base URL: `http://localhost:8000`

### REST Endpunkte

#### Health Check
*   **GET** `/health`
*   **Beschreibung:** Prüft, ob der Server läuft.
*   **Response:** `{"status": "ok"}`

#### Historische Daten
*   **GET** `/api/v1/data/history`
*   **Beschreibung:** Ruft OHLCV-Daten aus QuestDB ab.
*   **Parameter:**
    *   `symbol` (str): Handelspaar (z.B. `BTC/USDT`)
    *   `start_date` (str): ISO-8601 Datum (z.B. `2024-01-01T00:00:00Z`)
    *   `end_date` (str): ISO-8601 Datum
*   **Response:** Liste von OHLCV Objekten.

### WebSocket Endpunkte

#### Simulation Stream
*   **WS** `/api/v1/simulation/ws`
*   **Beschreibung:** Führt eine Simulation in Echtzeit aus und streamt Events.
*   **Query Parameter:**
    *   `symbol` (str)
    *   `start_date` (str)
    *   `end_date` (str)
    *   `initial_balance` (float, optional)
    *   `leverage` (int, optional)
*   **Nachrichtenformat (Server -> Client):**
    Jede Nachricht ist ein JSON-Objekt:
    ```json
    {
      "type": "MARKET_DATA" | "FILL" | "status",
      "timestamp": "ISO-Date",
      "payload": { ... }
    }
    ```
    Am Ende der Simulation wird eine Nachricht mit `type: "status"` und `payload: "simulation_complete"` gesendet, die auch `metrics` und `equity_curve` enthält.
