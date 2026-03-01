'use client';

import React, { useState, useRef } from 'react';
import EquityChart, { EquityChartRef } from '@/components/EquityChart';
import MetricsCard from '@/components/ui/MetricsCard';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Checkbox } from '@/components/ui/checkbox';
import { Card, CardHeader, CardTitle, CardContent } from '@/components/ui/card';
import api from '@/lib/api';
import { useStore } from '@/lib/store';

const TOP_20_SYMBOLS = [
  "BTC/USDT", "ETH/USDT", "BNB/USDT", "SOL/USDT", "XRP/USDT",
  "ADA/USDT", "AVAX/USDT", "DOGE/USDT", "DOT/USDT", "TRX/USDT",
  "LINK/USDT", "MATIC/USDT", "ICP/USDT", "SHIB/USDT", "LTC/USDT",
  "BCH/USDT", "UNI/USDT", "ATOM/USDT", "ETC/USDT", "XLM/USDT"
];

const ALL_STRATEGIES = [
  "demo", "momentum", "double_rsi_divergence", "multi_divergence",
  "mde_mad_entropy", "mde_mad_classic", "mde_mad_v2", "mde_mad_v2_leverage",
  "mde_mad_v3", "mde_mad_v3_1", "mde_mad_v4",
  "pair_arbitrage_v1", "pair_arbitrage_v2", "pair_arbitrage_v3", "pair_arbitrage_v4"
];

const ALL_TIMEFRAMES = ["1m", "3m", "5m", "15m", "30m", "1h", "2h", "4h", "6h", "8h", "12h", "1d"];

export default function MultiSimulationPage() {
  const {
    startDate,
    setStartDate,
    endDate,
    setEndDate
  } = useStore();

  const [selectedSymbols, setSelectedSymbols] = useState<string[]>([]);
  const [customSymbol, setCustomSymbol] = useState('');
  const [strategyName, setStrategyName] = useState('demo');
  const [timeframe, setTimeframe] = useState('1m');
  const [initialBalance, setInitialBalance] = useState(10000.0);
  const [leverage, setLeverage] = useState(1);
  const [slippageBps, setSlippageBps] = useState(0.0);

  const [isRunning, setIsRunning] = useState(false);
  const [metrics, setMetrics] = useState<any>(null);
  const [errorMsg, setErrorMsg] = useState<string | null>(null);

  const equityChartRef = useRef<EquityChartRef>(null);

  const toggleSymbol = (sym: string) => {
    setSelectedSymbols(prev => prev.includes(sym) ? prev.filter(s => s !== sym) : [...prev, sym]);
  };

  const addCustomSymbol = () => {
    if (customSymbol && !selectedSymbols.includes(customSymbol)) {
      setSelectedSymbols([...selectedSymbols, customSymbol]);
      setCustomSymbol('');
    }
  };

  const selectTop20 = () => setSelectedSymbols(TOP_20_SYMBOLS);
  const clearSymbols = () => setSelectedSymbols([]);

  const startSimulation = async () => {
    if (selectedSymbols.length === 0) {
      alert("Please select at least one symbol.");
      return;
    }

    try {
      setIsRunning(true);
      setMetrics(null);
      setErrorMsg(null);
      if (equityChartRef.current) {
        equityChartRef.current.clear();
      }

      const response = await api.post('/api/v1/simulation/multi', {
        symbols: selectedSymbols,
        strategy_name: strategyName,
        timeframe: timeframe,
        start_date: startDate,
        end_date: endDate,
        initial_balance: initialBalance,
        leverage: leverage,
        slippage_bps: slippageBps
      });

      if (response.data.status === 'success') {
        setMetrics(response.data.metrics);
        if (response.data.equity_curve && equityChartRef.current) {
          const eqData = response.data.equity_curve.map((p: any) => ({
            time: (new Date(p.timestamp).getTime() / 1000) as import('lightweight-charts').UTCTimestamp,
            value: p.equity
          }));
          equityChartRef.current.setData(eqData);
        }
      } else {
        setErrorMsg(response.data.message || "Simulation failed.");
      }
    } catch (e: any) {
      console.error(e);
      setErrorMsg(e.response?.data?.detail || e.message || "Failed to run simulation.");
    } finally {
      setIsRunning(false);
    }
  };

  return (
    <div className="space-y-6 animate-in fade-in duration-300">
      <div className="flex flex-col gap-1">
        <h1 className="text-xl font-bold uppercase tracking-widest text-primary">Portfolio Simulation</h1>
        <p className="text-sm text-muted-foreground font-mono">Run a multi-asset backtest with a shared portfolio balance.</p>
      </div>

      <Card>
        <CardHeader>
          <CardTitle>Configuration</CardTitle>
        </CardHeader>
        <CardContent className="space-y-6">
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">

            {/* Symbols */}
            <div className="space-y-2 border p-3 rounded border-border">
              <div className="flex justify-between items-center mb-2">
                <Label className="font-bold">Symbols</Label>
                <div className="space-x-2">
                  <Button variant="outline" onClick={selectTop20} className="text-xs h-6">Top 20</Button>
                  <Button variant="outline" onClick={clearSymbols} className="text-xs h-6">Clear</Button>
                </div>
              </div>
              <div className="h-40 overflow-y-auto space-y-1 bg-black/20 p-2 rounded text-[10px] font-mono border border-border">
                {selectedSymbols.map(sym => (
                  <div key={sym} className="flex items-center space-x-2">
                    <Checkbox id={`sym-${sym}`} checked onCheckedChange={() => toggleSymbol(sym)} />
                    <label htmlFor={`sym-${sym}`}>{sym}</label>
                  </div>
                ))}
                {selectedSymbols.length === 0 && <p className="text-muted-foreground p-1">No symbols selected</p>}
              </div>
              <div className="flex gap-2 mt-2">
                <Input
                  placeholder="Add symbol..."
                  value={customSymbol}
                  onChange={e => setCustomSymbol(e.target.value)}
                  className="h-8 text-xs bg-black"
                />
                <Button onClick={addCustomSymbol} className="h-8">+</Button>
              </div>
            </div>

            {/* General Settings */}
            <div className="space-y-4 border p-3 rounded border-border">
              <Label className="font-bold mb-2 block">Settings</Label>
              <div className="space-y-2">
                <Label htmlFor="strat">Strategy</Label>
                <select
                  id="strat"
                  className="w-full bg-black border border-border rounded p-2 text-sm text-primary focus:outline-none focus:ring-1 focus:ring-primary"
                  value={strategyName}
                  onChange={e => setStrategyName(e.target.value)}
                >
                  {ALL_STRATEGIES.map(s => <option key={s} value={s}>{s}</option>)}
                </select>
              </div>
              <div className="space-y-2">
                <Label htmlFor="tf">Timeframe</Label>
                <select
                  id="tf"
                  className="w-full bg-black border border-border rounded p-2 text-sm text-primary focus:outline-none focus:ring-1 focus:ring-primary"
                  value={timeframe}
                  onChange={e => setTimeframe(e.target.value)}
                >
                  {ALL_TIMEFRAMES.map(t => <option key={t} value={t}>{t}</option>)}
                </select>
              </div>
              <div className="space-y-2">
                <Label htmlFor="start">Start Date</Label>
                <Input id="start" type="datetime-local" className="bg-black" value={startDate} onChange={e => setStartDate(e.target.value)} />
              </div>
              <div className="space-y-2">
                <Label htmlFor="end">End Date</Label>
                <Input id="end" type="datetime-local" className="bg-black" value={endDate} onChange={e => setEndDate(e.target.value)} />
              </div>
            </div>

            {/* Capital & Risk */}
            <div className="space-y-4 border p-3 rounded border-border">
              <Label className="font-bold mb-2 block">Capital & Risk</Label>
              <div className="space-y-2">
                <Label htmlFor="balance">Initial Balance ($)</Label>
                <Input id="balance" type="number" step="100" className="bg-black" value={initialBalance} onChange={e => setInitialBalance(Number(e.target.value))} />
              </div>
              <div className="space-y-2">
                <Label htmlFor="leverage">Leverage</Label>
                <Input id="leverage" type="number" min="1" max="100" className="bg-black" value={leverage} onChange={e => setLeverage(Number(e.target.value))} />
              </div>
              <div className="space-y-2">
                <Label htmlFor="slippage">Slippage (BPS)</Label>
                <Input id="slippage" type="number" step="0.5" min="0" className="bg-black" value={slippageBps} onChange={e => setSlippageBps(Number(e.target.value))} />
              </div>
            </div>

          </div>

          <div className="flex items-center gap-4">
            <Button onClick={startSimulation} disabled={isRunning} className="w-full md:w-auto">
              {isRunning ? 'Running Simulation...' : 'Start Portfolio Simulation'}
            </Button>
            {errorMsg && (
              <p className="text-xs text-red-500 font-mono">{errorMsg}</p>
            )}
          </div>
        </CardContent>
      </Card>

      {metrics && <MetricsCard metrics={metrics} />}

      <Card>
        <CardHeader>
          <CardTitle>Portfolio Equity Curve</CardTitle>
        </CardHeader>
        <CardContent>
          <EquityChart ref={equityChartRef} />
        </CardContent>
      </Card>

    </div>
  );
}
