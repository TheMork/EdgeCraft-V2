# Datenintegration

## Historische Tickdaten (Trades)

Das System unterstützt den Import historischer Tickdaten (Trades) von Binance Futures (UM). Diese Daten können entweder über die API (für aktuelle Daten) oder via Bulk-Download (ZIP-Dateien von Binance Vision) importiert werden.

### Verwendung

Der Import erfolgt über das `src/main.py` Skript.

#### Argumente

- `--fetch-trades`: Aktiviert den Trade-Import.
- `--use-bulk`: Verwendet den Bulk-Downloader (empfohlen für historische Daten > 1 Tag). Lädt monatliche oder tägliche ZIP-Dateien herunter.

#### Beispiele

**Import via API (für kleine Zeiträume oder Live-Tests):**

```bash
python -m src.main --symbol BTC/USDT --fetch-trades --start-date 2024-01-01T00:00:00Z
```

**Import via Bulk-Download (für große Historien):**

```bash
python -m src.main --symbol BTC/USDT --fetch-trades --use-bulk --start-date 2023-01-01T00:00:00Z
```

### Funktionsweise Bulk-Downloader

Der Bulk-Downloader (`src/bulk_downloader.py`) prüft intelligent, welche Dateien benötigt werden:
1.  **Monatliche ZIPs**: Wenn ein voller Monat in der Vergangenheit liegt und angefordert wird.
2.  **Tägliche ZIPs**: Für angebrochene Monate oder den aktuellen Monat (bis gestern).

Die Daten werden automatisch entpackt, geparst und in die `trades` Tabelle der QuestDB eingefügt.

### Datenbankschema (Trades)

| Spalte    | Typ       | Beschreibung                                      |
| :-------- | :-------- | :------------------------------------------------ |
| id        | SYMBOL    | Trade-ID von Binance                              |
| symbol    | SYMBOL    | Handelspaar (z.B. BTC/USDT)                       |
| side      | SYMBOL    | BUY oder SELL (abgeleitet von `is_buyer_maker`)   |
| price     | DOUBLE    | Ausführungspreis                                  |
| amount    | DOUBLE    | Menge                                             |
| timestamp | TIMESTAMP | Zeitpunkt des Trades (UTC)                        |

**Deduplizierung:**
Die Tabelle verwendet `DEDUP UPSERT KEYS(timestamp, symbol, id)`, um Duplikate beim erneuten Import zu verhindern.
