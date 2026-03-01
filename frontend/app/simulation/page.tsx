'use client';

import React, { useState, useRef, useEffect, useCallback } from 'react';
import Chart, { ChartRef } from '@/components/Chart';
import EquityChart, { EquityChartRef } from '@/components/EquityChart';
import MetricsCard from '@/components/ui/MetricsCard';
import { CandlestickData, LineData } from 'lightweight-charts';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Card, CardHeader, CardTitle, CardContent } from '@/components/ui/card';
import api from '@/lib/api';
import { useStore } from '@/lib/store';

// EventType.MARKET_DATA = 0
const MARKET_DATA = 0;
const SYNC_POLL_INTERVAL_MS = 2000;
const SYNC_POLL_REQUEST_TIMEOUT_MS = 10000;
const SYNC_POLL_STALLED_WARNING_MS = 15 * 60 * 1000;
const SYNC_POLL_HARD_TIMEOUT_MS = 12 * 60 * 60 * 1000;

interface SyncStartResponse {
  status: string;
  message: string;
  job_id?: string;
}
type SyncMode = 'trades' | 'candles' | 'candles_1m' | 'candles_all';

interface SyncJobStatusResponse {
  job_id: string;
  status: string;
  progress: number;
  message: string;
}

interface DataCoverageResponse {
  symbol: string;
  timeframe: string;
  available_start: string | null;
  available_end: string | null;
  trades_start: string | null;
  trades_end: string | null;
  ohlcv_start: string | null;
  ohlcv_end: string | null;
  ohlcv_ranges?: Record<string, { start: string | null; end: string | null }>;
}

interface HistoryCandleResponse {
  timestamp: string;
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
}

const formatApiError = (error: any): string => {
  const detail = error?.response?.data?.detail;
  if (typeof detail === 'string' && detail.trim()) {
    return detail;
  }
  if (Array.isArray(detail) && detail.length > 0) {
    const first = detail[0];
    if (typeof first === 'string') return first;
    if (first?.msg) return String(first.msg);
    try {
      return JSON.stringify(first);
    } catch {
      return 'Validation error';
    }
  }
  if (detail && typeof detail === 'object') {
    try {
      return JSON.stringify(detail);
    } catch {
      return 'Request failed';
    }
  }
  const message = error?.message;
  if (typeof message === 'string' && message.trim()) {
    return message;
  }
  return 'Unknown error';
};

export default function SimulationPage() {
  const {
    selectedSymbol,
    selectedStrategy,
    selectedTimeframe,
    startDate,
    setStartDate,
    endDate,
    setEndDate
  } = useStore();

  const [isRunning, setIsRunning] = useState(false);
  const [isSyncing, setIsSyncing] = useState(false);
  const [streamMarketData, setStreamMarketData] = useState(false);
  const [autoDownloadData, setAutoDownloadData] = useState(true);
  const [enableShorts, setEnableShorts] = useState(true);
  const [leverage, setLeverage] = useState(2);
  const [minLeverage, setMinLeverage] = useState(1);
  const [maxLeverage, setMaxLeverage] = useState(3);
  const [syncMessage, setSyncMessage] = useState('');
  const [coverage, setCoverage] = useState<DataCoverageResponse | null>(null);
  const [coverageLoading, setCoverageLoading] = useState(false);
  const [coverageError, setCoverageError] = useState('');
  const [logs, setLogs] = useState<string[]>([]);
  const [metrics, setMetrics] = useState<any>(null);
  const [multiIndicators, setMultiIndicators] = useState('rsi,macd_hist,stoch_k,williams_r,cci,mfi');
  const [multiRequiredBullish, setMultiRequiredBullish] = useState(3);
  const [multiRequiredBearish, setMultiRequiredBearish] = useState(3);
  const [multiBullishScore, setMultiBullishScore] = useState(3);
  const [multiBearishScore, setMultiBearishScore] = useState(3);
  const [multiIncludeRegular, setMultiIncludeRegular] = useState(true);
  const [multiIncludeHidden, setMultiIncludeHidden] = useState(true);
  const [multiHiddenScoreMultiplier, setMultiHiddenScoreMultiplier] = useState(1.15);
  const [multiMinAdx, setMultiMinAdx] = useState(10);
  const [multiRequireRegime, setMultiRequireRegime] = useState(true);
  const [multiRequireVolume, setMultiRequireVolume] = useState(true);
  const [multiPivotLookback, setMultiPivotLookback] = useState(3);
  const [multiPivotSeparation, setMultiPivotSeparation] = useState(5);
  const [multiPivotAge, setMultiPivotAge] = useState(120);
  const [multiMinPriceMovePct, setMultiMinPriceMovePct] = useState(0.001);
  const [multiMinIndicatorDelta, setMultiMinIndicatorDelta] = useState(0);
  const [multiRiskPerTrade, setMultiRiskPerTrade] = useState(0.01);
  const [multiSlAtrMultiplier, setMultiSlAtrMultiplier] = useState(2.5);
  const [multiAnalysisWindowBars, setMultiAnalysisWindowBars] = useState(1200);

  const chartRef = useRef<ChartRef>(null);
  const equityChartRef = useRef<EquityChartRef>(null);
  const wsRef = useRef<WebSocket | null>(null);
  const wsExpectedCloseRef = useRef(false);
  const isMountedRef = useRef(true);

  const log = (msg: string) => {
    setLogs((prev) => [...prev.slice(-49), msg]); // Keep last 50 logs
  };
  const isMultiDivergence = selectedStrategy === 'multi_divergence';

  useEffect(() => {
    return () => {
      isMountedRef.current = false;
      if (wsRef.current) {
        wsExpectedCloseRef.current = true;
        wsRef.current.close();
      }
    };
  }, []);

  const sleep = (ms: number) => new Promise((resolve) => setTimeout(resolve, ms));

  const formatCoverageDate = (value: string | null) => {
    if (!value) return 'n/a';
    const dt = new Date(value);
    if (Number.isNaN(dt.getTime())) return value;
    return dt.toISOString().replace('T', ' ').replace('.000Z', 'Z');
  };

  const fetchCoverage = useCallback(async (symbol: string, timeframe: string) => {
    setCoverageLoading(true);
    try {
      const response = await api.get<DataCoverageResponse>('/api/v1/data/coverage', {
        params: { symbol, timeframe }
      });
      if (!isMountedRef.current) return;
      setCoverage({
        ...response.data,
        ohlcv_ranges: response.data.ohlcv_ranges ?? {},
      });
      setCoverageError('');
    } catch (error: any) {
      if (!isMountedRef.current) return;
      const msg = formatApiError(error);
      setCoverage(null);
      setCoverageError(msg);
    } finally {
      if (isMountedRef.current) {
        setCoverageLoading(false);
      }
    }
  }, []);

  const loadStaticChart = useCallback(async () => {
    try {
      const response = await api.get<HistoryCandleResponse[]>('/api/v1/data/history', {
        params: {
          symbol: selectedSymbol,
          start_date: startDate,
          end_date: endDate,
          timeframe: selectedTimeframe,
        },
        timeout: 20000,
      });
      if (!isMountedRef.current) return;

      const candles: CandlestickData[] = (response.data || [])
        .map((item) => {
          const ts = Math.floor(new Date(item.timestamp).getTime() / 1000);
          const open = Number(item.open);
          const high = Number(item.high);
          const low = Number(item.low);
          const close = Number(item.close);
          if (
            !Number.isFinite(ts)
            || !Number.isFinite(open)
            || !Number.isFinite(high)
            || !Number.isFinite(low)
            || !Number.isFinite(close)
          ) {
            return null;
          }
          return {
            time: ts as any,
            open,
            high,
            low,
            close,
          };
        })
        .filter((x): x is CandlestickData => x !== null);

      chartRef.current?.setData(candles);
      log(`Loaded ${candles.length} candles for static chart.`);
    } catch (error: any) {
      const msg = formatApiError(error);
      log(`Could not load static chart: ${msg}`);
    }
  }, [selectedSymbol, selectedTimeframe, startDate, endDate]);

  useEffect(() => {
    fetchCoverage(selectedSymbol, selectedTimeframe);
  }, [fetchCoverage, selectedSymbol, selectedTimeframe]);

  const pollSyncJob = async (jobId: string, label: string) => {
    const startedAt = Date.now();
    let consecutiveErrors = 0;
    let lastSnapshot = '';
    let lastChangeAt = Date.now();

    while (isMountedRef.current) {
      const elapsedMs = Date.now() - startedAt;
      if (elapsedMs > SYNC_POLL_HARD_TIMEOUT_MS) {
        throw new Error(`${label} is still running after ${Math.floor(elapsedMs / 60000)} minutes.`);
      }

      try {
        const response = await api.get<SyncJobStatusResponse>(`/api/v1/data/sync/jobs/${jobId}`, {
          timeout: SYNC_POLL_REQUEST_TIMEOUT_MS,
        });
        const job = response.data;
        if (!isMountedRef.current) return;
        consecutiveErrors = 0;

        const msg = `${label}: ${job.message} (${job.progress}%)`;
        const snapshot = `${job.status}|${job.progress}|${job.message}`;
        if (snapshot !== lastSnapshot) {
          lastSnapshot = snapshot;
          lastChangeAt = Date.now();
          log(msg);
        }
        setSyncMessage(msg);

        if (job.status === 'completed') {
          return;
        }

        if (job.status === 'failed') {
          throw new Error(job.message || `${label} failed.`);
        }

        const stalledMs = Date.now() - lastChangeAt;
        if (stalledMs > SYNC_POLL_STALLED_WARNING_MS) {
          const stalledMin = Math.floor(stalledMs / 60000);
          setSyncMessage(`${label}: still running, no status change for ${stalledMin} min. Continuing to poll...`);
        }
      } catch (error: any) {
        if (!isMountedRef.current) return;
        const statusCode = error?.response?.status;
        if (statusCode === 404) {
          throw new Error(`${label}: sync job not found. The API may have restarted.`);
        }
        // Non-recoverable backend errors should fail fast.
        if (statusCode && statusCode >= 400 && statusCode < 500) {
          const detail = formatApiError(error);
          throw new Error(`${label} failed: ${detail}`);
        }

        consecutiveErrors += 1;
        const detail = formatApiError(error);
        const retryMsg = `${label}: connection issue (${consecutiveErrors}), retrying... ${detail}`;
        setSyncMessage(retryMsg);
        if (consecutiveErrors === 1 || consecutiveErrors % 5 === 0) {
          log(retryMsg);
        }
      }

      await sleep(SYNC_POLL_INTERVAL_MS);
    }
  };

  const startSimulation = async () => {
    if (isRunning) return;

    setIsRunning(true);
    setLogs([]);
    setMetrics(null);
    chartRef.current?.setData([]); // Clear chart
    equityChartRef.current?.setData([]); // Clear equity chart

    if (autoDownloadData) {
      log('Checking data coverage...');
      let needsSync = false;
      if (!coverage || !coverage.available_start || !coverage.available_end) {
        needsSync = true;
      } else {
        const startReq = new Date(startDate).getTime();
        const endReq = new Date(endDate).getTime();
        const startCov = new Date(coverage.available_start).getTime();
        const endCov = new Date(coverage.available_end).getTime();
        // Allow a small margin (e.g., 1 day) for boundaries if needed, but strictly:
        if (startReq < startCov || endReq > endCov) {
          needsSync = true;
        }
      }

      if (needsSync) {
        log('Data missing for requested range. Auto-downloading...');
        setSyncMessage('Auto-downloading missing data...');
        try {
          const response = await api.post<SyncStartResponse>('/api/v1/data/sync', {
            symbols: [selectedSymbol],
            start_date: startDate,
            end_date: endDate,
            sync_mode: 'candles',
            timeframes: [selectedTimeframe],
          });
          const jobId = response.data.job_id;
          if (jobId) {
            log(`Sync job started: ${jobId}`);
            await pollSyncJob(jobId, 'Auto-download');
            await fetchCoverage(selectedSymbol, selectedTimeframe);
            log('Auto-download complete.');
          } else {
            log('No sync job ID returned. Proceeding...');
          }
        } catch (error: any) {
          console.error('Auto-download failed:', error);
          log('Auto-download failed: ' + formatApiError(error));
          setIsRunning(false);
          setSyncMessage('Simulation aborted due to auto-download failure.');
          return;
        } finally {
          setSyncMessage('');
        }
      } else {
        log('Data coverage is sufficient.');
      }
    }

    const apiUrl = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';
    const wsBaseUrl = apiUrl.replace(/^http/, 'ws');
    const safeLeverage = Math.max(1, Math.min(125, Math.floor(Number(leverage) || 1)));
    const safeMinLeverage = Math.max(1, Math.min(125, Math.floor(Number(minLeverage) || 1)));
    const safeMaxLeverage = Math.max(safeMinLeverage, Math.min(125, Math.floor(Number(maxLeverage) || safeMinLeverage)));
    const params = new URLSearchParams({
      symbol: selectedSymbol,
      start_date: startDate,
      end_date: endDate,
      strategy: selectedStrategy,
      timeframe: selectedTimeframe,
      stream: String(streamMarketData),
      enable_shorts: String(enableShorts),
      leverage: String(safeLeverage),
      min_leverage: String(safeMinLeverage),
      max_leverage: String(safeMaxLeverage),
    });
    if (isMultiDivergence) {
      params.set('multi_indicators', multiIndicators);
      params.set('multi_required_bullish', String(Math.max(1, Math.floor(Number(multiRequiredBullish) || 1))));
      params.set('multi_required_bearish', String(Math.max(1, Math.floor(Number(multiRequiredBearish) || 1))));
      params.set('multi_required_bullish_score', String(Math.max(0.1, Number(multiBullishScore) || 0.1)));
      params.set('multi_required_bearish_score', String(Math.max(0.1, Number(multiBearishScore) || 0.1)));
      params.set('multi_include_regular', String(multiIncludeRegular));
      params.set('multi_include_hidden', String(multiIncludeHidden));
      params.set('multi_hidden_score_multiplier', String(Math.max(1, Number(multiHiddenScoreMultiplier) || 1)));
      params.set('multi_min_adx_for_entry', String(Math.max(0, Number(multiMinAdx) || 0)));
      params.set('multi_require_regime_filter', String(multiRequireRegime));
      params.set('multi_require_volume_confirmation', String(multiRequireVolume));
      params.set('multi_pivot_lookback', String(Math.max(1, Math.floor(Number(multiPivotLookback) || 1))));
      params.set('multi_min_pivot_separation_bars', String(Math.max(1, Math.floor(Number(multiPivotSeparation) || 1))));
      params.set('multi_max_pivot_age_bars', String(Math.max(5, Math.floor(Number(multiPivotAge) || 5))));
      params.set('multi_min_price_move_pct', String(Math.max(0, Number(multiMinPriceMovePct) || 0)));
      params.set('multi_min_indicator_delta', String(Math.max(0, Number(multiMinIndicatorDelta) || 0)));
      params.set('multi_risk_per_trade', String(Math.max(0.0001, Number(multiRiskPerTrade) || 0.0001)));
      params.set('multi_sl_atr_multiplier', String(Math.max(0.1, Number(multiSlAtrMultiplier) || 0.1)));
      params.set('multi_analysis_window_bars', String(Math.max(200, Math.floor(Number(multiAnalysisWindowBars) || 200))));
    }
    const wsUrl = `${wsBaseUrl}/api/v1/simulation/ws?${params.toString()}`;
    log(`Strategy: ${selectedStrategy}`);
    log(`Timeframe: ${selectedTimeframe}`);
    log(`Live Chart: ${streamMarketData ? 'on' : 'off (fast mode)'}`);
    log(`Shorts: ${enableShorts ? 'enabled' : 'disabled'}`);
    log(`Leverage: ${safeLeverage}x (min ${safeMinLeverage}x, max ${safeMaxLeverage}x)`);
    if (isMultiDivergence) {
      log(
        `MultiDiv: indicators=${multiIndicators} | bull=${multiRequiredBullish}/${multiBullishScore} `
        + `bear=${multiRequiredBearish}/${multiBearishScore} | hidden=${multiIncludeHidden} regular=${multiIncludeRegular} `
        + `window=${Math.max(200, Math.floor(Number(multiAnalysisWindowBars) || 200))}`
      );
    }
    log(`Connecting to ${wsUrl}...`);

    const ws = new WebSocket(wsUrl);
    wsRef.current = ws;
    wsExpectedCloseRef.current = false;

    ws.onopen = () => {
      log('Connected to simulation server.');
    };

    ws.onmessage = (event) => {
      try {
        const rawData = String(event.data ?? '');
        const sanitizedData = rawData
          .replace(/\b-?Infinity\b/g, 'null')
          .replace(/\bNaN\b/g, 'null');
        const message = JSON.parse(sanitizedData);

        if (message.type === 'status' && message.payload === 'simulation_complete') {
          log('Simulation complete.');
          if (!streamMarketData) {
            loadStaticChart();
          }
          if (message.metrics) {
              setMetrics(message.metrics);
          }
          if (message.equity_curve && Array.isArray(message.equity_curve)) {
              // Convert equity curve to LineData and keep only valid points.
              const equityData: LineData[] = message.equity_curve
                .map((item: any) => {
                  const timestampSec = Math.floor(new Date(item.timestamp).getTime() / 1000);
                  const equity = Number(item.equity);
                  if (!Number.isFinite(timestampSec) || !Number.isFinite(equity)) {
                    return null;
                  }
                  return {
                    time: timestampSec as any,
                    value: equity
                  };
                })
                .filter((item: LineData | null): item is LineData => item !== null);
              equityChartRef.current?.setData(equityData);
          }
          wsExpectedCloseRef.current = true;
          ws.close();
          return;
        }

        if (message.type === MARKET_DATA) {
          const payload = message.payload;
          // payload has: open, high, low, close, timestamp (string from backend, wait)
          // backend sends timestamp as message.timestamp (iso string)
          // payload has keys: open, high, low, close, volume, timestamp (from db)

          // We need to convert timestamp to seconds
          const timestampSec = Math.floor(new Date(message.timestamp).getTime() / 1000);
          const open = Number(payload.open);
          const high = Number(payload.high);
          const low = Number(payload.low);
          const close = Number(payload.close);
          if (
            !Number.isFinite(timestampSec)
            || !Number.isFinite(open)
            || !Number.isFinite(high)
            || !Number.isFinite(low)
            || !Number.isFinite(close)
          ) {
            return;
          }

          const candle: CandlestickData = {
            time: timestampSec as any, // lightweight-charts expects UTCTimestamp (number)
            open,
            high,
            low,
            close,
          };

          chartRef.current?.update(candle);
        }
      } catch (e) {
        console.error('Error parsing message:', e, event.data);
        log('Received malformed WS payload. Skipping event.');
      }
    };

    ws.onclose = (closeEvent) => {
      if (wsExpectedCloseRef.current || closeEvent.wasClean) {
        log('Disconnected.');
      } else {
        log(`WebSocket closed unexpectedly (code=${closeEvent.code}, reason=${closeEvent.reason || 'n/a'}).`);
      }
      setIsRunning(false);
      wsRef.current = null;
      wsExpectedCloseRef.current = false;
    };

    ws.onerror = (event) => {
      const state = ws.readyState;
      // Browsers expose very little detail here; avoid noisy "{}" console output.
      if (wsExpectedCloseRef.current || state === WebSocket.CLOSING || state === WebSocket.CLOSED) {
        return;
      }
      log(`WebSocket error (state=${state}).`);
      console.error('WebSocket error event:', {
        type: event.type,
        readyState: state,
        url: ws.url,
      });
    };
  };

  const stopSimulation = () => {
    if (wsRef.current) {
      wsExpectedCloseRef.current = true;
      wsRef.current.close();
    }
  };

  const startSingleSync = async (syncMode: SyncMode, label: string) => {
    if (!selectedSymbol || !startDate) return;
    setIsSyncing(true);
    setSyncMessage(`${label} ${selectedSymbol}...`);
    try {
      const response = await api.post<SyncStartResponse>('/api/v1/data/sync', {
        symbol: selectedSymbol,
        start_date: startDate,
        end_date: endDate,
        sync_mode: syncMode,
        timeframe: selectedTimeframe,
      });
      const jobId = response.data.job_id;
      if (jobId) {
        log(`Sync job started: ${jobId}`);
        await pollSyncJob(jobId, `${label} ${selectedSymbol}`);
        await fetchCoverage(selectedSymbol, selectedTimeframe);
      } else {
        setSyncMessage(response.data.message || 'Sync complete.');
        log(`Sync: ${response.data.message || 'Sync complete.'}`);
        await fetchCoverage(selectedSymbol, selectedTimeframe);
      }
    } catch (error: any) {
      console.error('Sync failed:', error);
      const msg = formatApiError(error);
      setSyncMessage('Sync failed: ' + msg);
      log('Sync failed: ' + msg);
    } finally {
      setIsSyncing(false);
    }
  };

  const handleSync = async () => {
    await startSingleSync('candles', `Syncing ${selectedTimeframe} candles for`);
  };

  const handleSyncTrades = async () => {
    await startSingleSync('trades', 'Syncing trades for');
  };

  const handleSyncAllTimeframes = async () => {
    await startSingleSync('candles_all', `Syncing all candle timeframes for`);
  };

  const handleSyncTop20 = async () => {
    setIsSyncing(true);
    setSyncMessage('Syncing Top 20...');
    try {
      const response = await api.post<SyncStartResponse>('/api/v1/data/sync/top20', {
        start_date: startDate,
        end_date: endDate,
        sync_mode: 'candles',
        timeframe: selectedTimeframe,
      });
      const jobId = response.data.job_id;
      if (jobId) {
        log(`Top 20 sync job started: ${jobId}`);
        await pollSyncJob(jobId, 'Top 20 sync');
        await fetchCoverage(selectedSymbol, selectedTimeframe);
      } else {
        setSyncMessage(response.data.message || 'Top 20 sync started.');
        log(`Sync Top 20: ${response.data.message || 'Started.'}`);
        await fetchCoverage(selectedSymbol, selectedTimeframe);
      }
    } catch (error: any) {
      console.error('Sync Top 20 failed:', error);
      const msg = formatApiError(error);
      setSyncMessage('Sync Top 20 failed: ' + msg);
      log('Sync Top 20 failed: ' + msg);
    } finally {
      setIsSyncing(false);
    }
  };

  const handleSyncTop20AllCandles = async () => {
    setIsSyncing(true);
    setSyncMessage('Syncing Top 20 all candle timeframes...');
    try {
      const response = await api.post<SyncStartResponse>('/api/v1/data/sync/top20', {
        start_date: startDate,
        end_date: endDate,
        sync_mode: 'candles_all',
        timeframe: selectedTimeframe,
      });
      const jobId = response.data.job_id;
      if (jobId) {
        log(`Top 20 all-candles sync job started: ${jobId}`);
        await pollSyncJob(jobId, 'Top 20 all-candles sync');
        await fetchCoverage(selectedSymbol, selectedTimeframe);
      } else {
        setSyncMessage(response.data.message || 'Top 20 all-candles sync started.');
        log(`Sync Top 20 all-candles: ${response.data.message || 'Started.'}`);
        await fetchCoverage(selectedSymbol, selectedTimeframe);
      }
    } catch (error: any) {
      console.error('Sync Top 20 all-candles failed:', error);
      const msg = formatApiError(error);
      setSyncMessage('Sync Top 20 all-candles failed: ' + msg);
      log('Sync Top 20 all-candles failed: ' + msg);
    } finally {
      setIsSyncing(false);
    }
  };

  const ohlcvRangesText = coverage
    ? Object.entries(coverage.ohlcv_ranges ?? {})
        .map(([tf, range]) => `${tf}=${formatCoverageDate(range.start)}..${formatCoverageDate(range.end)}`)
        .join(' | ')
    : '';

  return (
    <div className="p-6 space-y-6">
      <Card>
        <CardHeader>
          <CardTitle>Simulation Control</CardTitle>
        </CardHeader>
        <CardContent className="space-y-5">
          <div className="flex flex-col gap-3 lg:flex-row lg:items-center lg:justify-between">
            <div className="space-y-2">
              <Label>Market</Label>
              <div className="flex flex-wrap gap-2 text-xs font-mono">
                <span className="px-2 py-1 rounded border border-border bg-muted/30">{selectedSymbol}</span>
                <span className="px-2 py-1 rounded border border-border bg-muted/30">Strategy: {selectedStrategy}</span>
                <span className="px-2 py-1 rounded border border-border bg-muted/30">Timeframe: {selectedTimeframe}</span>
              </div>
            </div>
            <div className="flex flex-wrap gap-2">
              <Button
                variant="outline"
                onClick={handleSync}
                disabled={isSyncing || isRunning}
              >
                {isSyncing ? '...' : `Sync ${selectedTimeframe} Candles`}
              </Button>
              <Button
                variant="outline"
                onClick={handleSyncTrades}
                disabled={isSyncing || isRunning}
              >
                {isSyncing ? '...' : 'Sync Trades'}
              </Button>
              <Button
                variant="outline"
                onClick={handleSyncAllTimeframes}
                disabled={isSyncing || isRunning}
              >
                {isSyncing ? '...' : 'Sync All TF'}
              </Button>
              <Button
                variant="default"
                onClick={handleSyncTop20}
                disabled={isSyncing || isRunning}
              >
                {isSyncing ? '...' : 'Sync Top 20'}
              </Button>
              <Button
                variant="default"
                onClick={handleSyncTop20AllCandles}
                disabled={isSyncing || isRunning}
              >
                {isSyncing ? '...' : 'Sync Top 20 All TF'}
              </Button>
            </div>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-4 gap-3">
            <div className="space-y-2">
              <Label htmlFor="startDate">Start Date</Label>
              <Input
                id="startDate"
                value={startDate}
                onChange={(e) => setStartDate(e.target.value)}
              />
            </div>
            <div className="space-y-2">
              <Label htmlFor="endDate">End Date</Label>
              <Input
                id="endDate"
                value={endDate}
                onChange={(e) => setEndDate(e.target.value)}
              />
            </div>
            <div className="space-y-2">
              <Label htmlFor="leverage">Leverage</Label>
              <Input
                id="leverage"
                type="number"
                min={1}
                max={125}
                value={leverage}
                onChange={(e) => setLeverage(Number(e.target.value))}
                disabled={isRunning}
              />
            </div>
            <div className="space-y-2">
              <Label htmlFor="minLeverage">Min Leverage</Label>
              <Input
                id="minLeverage"
                type="number"
                min={1}
                max={125}
                value={minLeverage}
                onChange={(e) => setMinLeverage(Number(e.target.value))}
                disabled={isRunning}
              />
            </div>
            <div className="space-y-2">
              <Label htmlFor="maxLeverage">Max Leverage</Label>
              <Input
                id="maxLeverage"
                type="number"
                min={1}
                max={125}
                value={maxLeverage}
                onChange={(e) => setMaxLeverage(Number(e.target.value))}
                disabled={isRunning}
              />
            </div>
            <div className="space-y-2">
              <Label htmlFor="streamMarketData">Live Chart</Label>
              <label className="flex items-center gap-2 h-10 px-3 border border-border rounded bg-muted/20 text-sm">
                <input
                  id="streamMarketData"
                  type="checkbox"
                  checked={streamMarketData}
                  onChange={(e) => setStreamMarketData(e.target.checked)}
                  disabled={isRunning}
                />
                <span>{streamMarketData ? 'Animate candles' : 'Fast mode'}</span>
              </label>
            </div>
            <div className="space-y-2">
              <Label htmlFor="autoDownloadData">Auto-Download Data</Label>
              <label className="flex items-center gap-2 h-10 px-3 border border-border rounded bg-muted/20 text-sm">
                <input
                  id="autoDownloadData"
                  type="checkbox"
                  checked={autoDownloadData}
                  onChange={(e) => setAutoDownloadData(e.target.checked)}
                  disabled={isRunning}
                />
                <span>{autoDownloadData ? 'Enabled' : 'Disabled'}</span>
              </label>
            </div>
            <div className="space-y-2">
              <Label htmlFor="enableShorts">Shorts</Label>
              <label className="flex items-center gap-2 h-10 px-3 border border-border rounded bg-muted/20 text-sm">
                <input
                  id="enableShorts"
                  type="checkbox"
                  checked={enableShorts}
                  onChange={(e) => setEnableShorts(e.target.checked)}
                  disabled={isRunning}
                />
                <span>{enableShorts ? 'Enabled' : 'Disabled'}</span>
              </label>
            </div>
          </div>

          {isMultiDivergence && (
            <div className="rounded border border-border p-3 space-y-3">
              <p className="text-xs uppercase tracking-wide text-muted-foreground">Multi Divergence Settings</p>
              <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-4 gap-3">
                <div className="space-y-2">
                  <Label htmlFor="multiIndicators">Indicators (CSV)</Label>
                  <Input
                    id="multiIndicators"
                    value={multiIndicators}
                    onChange={(e) => setMultiIndicators(e.target.value)}
                    disabled={isRunning}
                  />
                  <p className="text-[11px] text-muted-foreground">
                    Supported: rsi, macd_hist, stoch_k, williams_r, cci, mfi, roc, obv
                  </p>
                </div>
                <div className="space-y-2">
                  <Label htmlFor="multiRequiredBullish">Bull Count</Label>
                  <Input
                    id="multiRequiredBullish"
                    type="number"
                    min={1}
                    max={10}
                    value={multiRequiredBullish}
                    onChange={(e) => setMultiRequiredBullish(Number(e.target.value))}
                    disabled={isRunning}
                  />
                </div>
                <div className="space-y-2">
                  <Label htmlFor="multiBullishScore">Bull Score</Label>
                  <Input
                    id="multiBullishScore"
                    type="number"
                    step="0.1"
                    min={0.1}
                    value={multiBullishScore}
                    onChange={(e) => setMultiBullishScore(Number(e.target.value))}
                    disabled={isRunning}
                  />
                </div>
                <div className="space-y-2">
                  <Label htmlFor="multiRequiredBearish">Bear Count</Label>
                  <Input
                    id="multiRequiredBearish"
                    type="number"
                    min={1}
                    max={10}
                    value={multiRequiredBearish}
                    onChange={(e) => setMultiRequiredBearish(Number(e.target.value))}
                    disabled={isRunning}
                  />
                </div>
                <div className="space-y-2">
                  <Label htmlFor="multiBearishScore">Bear Score</Label>
                  <Input
                    id="multiBearishScore"
                    type="number"
                    step="0.1"
                    min={0.1}
                    value={multiBearishScore}
                    onChange={(e) => setMultiBearishScore(Number(e.target.value))}
                    disabled={isRunning}
                  />
                </div>
                <div className="space-y-2">
                  <Label htmlFor="multiMinAdx">Min ADX</Label>
                  <Input
                    id="multiMinAdx"
                    type="number"
                    step="0.1"
                    min={0}
                    value={multiMinAdx}
                    onChange={(e) => setMultiMinAdx(Number(e.target.value))}
                    disabled={isRunning}
                  />
                </div>
                <div className="space-y-2">
                  <Label htmlFor="multiPivotLookback">Pivot Lookback</Label>
                  <Input
                    id="multiPivotLookback"
                    type="number"
                    min={1}
                    value={multiPivotLookback}
                    onChange={(e) => setMultiPivotLookback(Number(e.target.value))}
                    disabled={isRunning}
                  />
                </div>
                <div className="space-y-2">
                  <Label htmlFor="multiPivotSeparation">Pivot Separation</Label>
                  <Input
                    id="multiPivotSeparation"
                    type="number"
                    min={1}
                    value={multiPivotSeparation}
                    onChange={(e) => setMultiPivotSeparation(Number(e.target.value))}
                    disabled={isRunning}
                  />
                </div>
                <div className="space-y-2">
                  <Label htmlFor="multiPivotAge">Pivot Max Age</Label>
                  <Input
                    id="multiPivotAge"
                    type="number"
                    min={5}
                    value={multiPivotAge}
                    onChange={(e) => setMultiPivotAge(Number(e.target.value))}
                    disabled={isRunning}
                  />
                </div>
                <div className="space-y-2">
                  <Label htmlFor="multiAnalysisWindowBars">Analysis Window Bars</Label>
                  <Input
                    id="multiAnalysisWindowBars"
                    type="number"
                    min={200}
                    value={multiAnalysisWindowBars}
                    onChange={(e) => setMultiAnalysisWindowBars(Number(e.target.value))}
                    disabled={isRunning}
                  />
                </div>
                <div className="space-y-2">
                  <Label htmlFor="multiMinPriceMovePct">Min Price Move %</Label>
                  <Input
                    id="multiMinPriceMovePct"
                    type="number"
                    step="0.0001"
                    min={0}
                    value={multiMinPriceMovePct}
                    onChange={(e) => setMultiMinPriceMovePct(Number(e.target.value))}
                    disabled={isRunning}
                  />
                </div>
                <div className="space-y-2">
                  <Label htmlFor="multiMinIndicatorDelta">Min Indicator Delta</Label>
                  <Input
                    id="multiMinIndicatorDelta"
                    type="number"
                    step="0.01"
                    min={0}
                    value={multiMinIndicatorDelta}
                    onChange={(e) => setMultiMinIndicatorDelta(Number(e.target.value))}
                    disabled={isRunning}
                  />
                </div>
                <div className="space-y-2">
                  <Label htmlFor="multiRiskPerTrade">Risk / Trade</Label>
                  <Input
                    id="multiRiskPerTrade"
                    type="number"
                    step="0.001"
                    min={0.0001}
                    max={0.2}
                    value={multiRiskPerTrade}
                    onChange={(e) => setMultiRiskPerTrade(Number(e.target.value))}
                    disabled={isRunning}
                  />
                </div>
                <div className="space-y-2">
                  <Label htmlFor="multiSlAtrMultiplier">SL ATR Multiplier</Label>
                  <Input
                    id="multiSlAtrMultiplier"
                    type="number"
                    step="0.1"
                    min={0.1}
                    value={multiSlAtrMultiplier}
                    onChange={(e) => setMultiSlAtrMultiplier(Number(e.target.value))}
                    disabled={isRunning}
                  />
                </div>
                <div className="space-y-2">
                  <Label htmlFor="multiHiddenScoreMultiplier">Hidden Score Multiplier</Label>
                  <Input
                    id="multiHiddenScoreMultiplier"
                    type="number"
                    step="0.05"
                    min={1}
                    value={multiHiddenScoreMultiplier}
                    onChange={(e) => setMultiHiddenScoreMultiplier(Number(e.target.value))}
                    disabled={isRunning}
                  />
                </div>
                <div className="space-y-2">
                  <Label htmlFor="multiIncludeRegular">Regular Divergence</Label>
                  <label className="flex items-center gap-2 h-10 px-3 border border-border rounded bg-muted/20 text-sm">
                    <input
                      id="multiIncludeRegular"
                      type="checkbox"
                      checked={multiIncludeRegular}
                      onChange={(e) => setMultiIncludeRegular(e.target.checked)}
                      disabled={isRunning}
                    />
                    <span>{multiIncludeRegular ? 'Enabled' : 'Disabled'}</span>
                  </label>
                </div>
                <div className="space-y-2">
                  <Label htmlFor="multiIncludeHidden">Hidden Divergence</Label>
                  <label className="flex items-center gap-2 h-10 px-3 border border-border rounded bg-muted/20 text-sm">
                    <input
                      id="multiIncludeHidden"
                      type="checkbox"
                      checked={multiIncludeHidden}
                      onChange={(e) => setMultiIncludeHidden(e.target.checked)}
                      disabled={isRunning}
                    />
                    <span>{multiIncludeHidden ? 'Enabled' : 'Disabled'}</span>
                  </label>
                </div>
                <div className="space-y-2">
                  <Label htmlFor="multiRequireRegime">Regime Filter</Label>
                  <label className="flex items-center gap-2 h-10 px-3 border border-border rounded bg-muted/20 text-sm">
                    <input
                      id="multiRequireRegime"
                      type="checkbox"
                      checked={multiRequireRegime}
                      onChange={(e) => setMultiRequireRegime(e.target.checked)}
                      disabled={isRunning}
                    />
                    <span>{multiRequireRegime ? 'Enabled' : 'Disabled'}</span>
                  </label>
                </div>
                <div className="space-y-2">
                  <Label htmlFor="multiRequireVolume">Volume Confirmation</Label>
                  <label className="flex items-center gap-2 h-10 px-3 border border-border rounded bg-muted/20 text-sm">
                    <input
                      id="multiRequireVolume"
                      type="checkbox"
                      checked={multiRequireVolume}
                      onChange={(e) => setMultiRequireVolume(e.target.checked)}
                      disabled={isRunning}
                    />
                    <span>{multiRequireVolume ? 'Enabled' : 'Disabled'}</span>
                  </label>
                </div>
              </div>
            </div>
          )}

          <div className="rounded border border-border p-3 text-xs text-muted-foreground space-y-1">
            {syncMessage && <p className="text-foreground">{syncMessage}</p>}
            {coverageLoading && <p>Loading available backtest range...</p>}
            {!coverageLoading && coverage?.available_start && coverage?.available_end && (
              <p>
                Available ({coverage.timeframe}): {formatCoverageDate(coverage.available_start)} to {formatCoverageDate(coverage.available_end)}
              </p>
            )}
            {!coverageLoading && !coverageError && (!coverage?.available_start || !coverage?.available_end) && (
              <p>No historical data available yet for {selectedSymbol}.</p>
            )}
            {!coverageLoading && coverage && (
              <p>
                Trades: {formatCoverageDate(coverage.trades_start)} to {formatCoverageDate(coverage.trades_end)} | OHLCV: {formatCoverageDate(coverage.ohlcv_start)} to {formatCoverageDate(coverage.ohlcv_end)}
              </p>
            )}
            {!coverageLoading && coverage && ohlcvRangesText && (
              <p className="font-mono break-words">OHLCV by timeframe: {ohlcvRangesText}</p>
            )}
            {!coverageLoading && coverageError && (
              <p className="text-red-400">Could not load coverage: {coverageError}</p>
            )}
          </div>

          <div className="flex flex-wrap gap-2">
            <Button onClick={startSimulation} disabled={isRunning}>
              {isRunning ? 'Running...' : 'Start Simulation'}
            </Button>
            <Button variant="destructive" onClick={stopSimulation} disabled={!isRunning}>
              Stop
            </Button>
          </div>
        </CardContent>
      </Card>

      {metrics && <MetricsCard metrics={metrics} />}

      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          <Card>
            <CardHeader>
              <CardTitle>Price Chart</CardTitle>
            </CardHeader>
            <CardContent>
              <Chart ref={chartRef} />
            </CardContent>
          </Card>

          <Card>
            <CardHeader>
              <CardTitle>Equity Curve</CardTitle>
            </CardHeader>
            <CardContent>
              <EquityChart ref={equityChartRef} />
            </CardContent>
          </Card>
      </div>

      <Card>
        <CardHeader>
          <CardTitle>Logs</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="h-40 overflow-y-auto bg-black/50 p-2 rounded text-xs font-mono">
            {logs.map((l, i) => (
              <div key={i}>{l}</div>
            ))}
          </div>
        </CardContent>
      </Card>
    </div>
  );
}
