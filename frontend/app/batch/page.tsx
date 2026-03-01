'use client';

import React, { useState, useEffect } from 'react';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Card, CardHeader, CardTitle, CardContent } from '@/components/ui/card';
import { Checkbox } from '@/components/ui/checkbox';
import api from '@/lib/api';
import { useStore } from '@/lib/store';

const ALL_STRATEGIES = [
  'demo',
  'momentum',
  'double_rsi_divergence',
  'multi_divergence',
  'pair_arbitrage_v1',
  'pair_arbitrage_v2',
  'pair_arbitrage_v3',
  'pair_arbitrage_v4',
  'mde_mad_entropy',
  'mde_mad_classic',
  'mde_mad_v2',
  'mde_mad_v2_leverage',
  'mde_mad_v3',
  'mde_mad_v3_1',
  'mde_mad_v4',
];

const ALL_TIMEFRAMES = [
  '1m', '5m', '15m', '30m', '1h', '4h', '12h', '1d'
];

const TOP_20_SYMBOLS = [
    "BTC/USDT", "ETH/USDT", "SOL/USDT", "BNB/USDT", "XRP/USDT",
    "DOGE/USDT", "ADA/USDT", "AVAX/USDT", "TRX/USDT", "LINK/USDT",
    "DOT/USDT", "MATIC/USDT", "SHIB/USDT", "LTC/USDT", "UNI/USDT",
    "ATOM/USDT", "XLM/USDT", "ETC/USDT", "FIL/USDT", "HBAR/USDT",
];

interface BatchResult {
  status: string;
  task: {
    symbol: string;
    strategy_name: string;
    timeframe: string;
  };
  metrics?: {
    total_return: number;
    sharpe_ratio: number;
    max_drawdown: number;
    total_trades: number;
    final_equity: number;
  };
  error?: string;
}

interface BatchJobStatus {
  job_id: string;
  status: string;
  message?: string;
  progress: {
    total: number;
    completed: number;
    percent: number;
  };
  error?: string | null;
  results: BatchResult[];
}

interface StrategyAggregate {
  name: string;
  count: number;
  totalReturn: number;
  totalSharpe: number;
  totalDD: number;
  wins: number;
}

export default function BatchPage() {
  const { startDate, setStartDate, endDate, setEndDate } = useStore();
  
  const [selectedSymbols, setSelectedSymbols] = useState<string[]>(['BTC/USDT']);
  const [customSymbol, setCustomSymbol] = useState('');
  const [selectedStrategies, setSelectedStrategies] = useState<string[]>(['momentum']);
  const [selectedTimeframes, setSelectedTimeframes] = useState<string[]>(['1h']);
  const [jobId, setJobId] = useState<string | null>(null);
  const [jobStatus, setJobStatus] = useState<BatchJobStatus | null>(null);
  const [isPolling, setIsPolling] = useState(false);
  const [pollError, setPollError] = useState<string | null>(null);

  const toggleSymbol = (sym: string) => {
    setSelectedSymbols(prev => prev.includes(sym) ? prev.filter(s => s !== sym) : [...prev, sym]);
  };

  const toggleStrategy = (strat: string) => {
    setSelectedStrategies(prev => prev.includes(strat) ? prev.filter(s => s !== strat) : [...prev, strat]);
  };

  const toggleTimeframe = (tf: string) => {
    setSelectedTimeframes(prev => prev.includes(tf) ? prev.filter(t => t !== tf) : [...prev, tf]);
  };

  const addCustomSymbol = () => {
    if (customSymbol && !selectedSymbols.includes(customSymbol)) {
      setSelectedSymbols([...selectedSymbols, customSymbol]);
      setCustomSymbol('');
    }
  };

  const selectTop20 = () => {
    setSelectedSymbols(TOP_20_SYMBOLS);
  };

  const clearSymbols = () => {
    setSelectedSymbols([]);
  };

  const startBatch = async () => {
    if (selectedSymbols.length === 0 || selectedStrategies.length === 0 || selectedTimeframes.length === 0) {
      alert("Please select at least one symbol, strategy, and timeframe.");
      return;
    }

    try {
      setJobStatus(null);
      setPollError(null);
      const response = await api.post('/api/v1/batch/run', {
        symbols: selectedSymbols,
        strategies: selectedStrategies,
        timeframes: selectedTimeframes,
        start_date: startDate,
        end_date: endDate,
        initial_balance: 10000,
        leverage: 1
      });
      setJobId(response.data.job_id);
      setIsPolling(true);
    } catch (e) {
      console.error(e);
      alert("Failed to start batch job.");
    }
  };

  useEffect(() => {
    if (!isPolling || !jobId) return;

    const interval = setInterval(async () => {
      try {
        const response = await api.get<BatchJobStatus>(`/api/v1/batch/status/${jobId}`);
        setJobStatus(response.data);
        if (response.data.status === 'completed' || response.data.status === 'failed') {
          setIsPolling(false);
        }
      } catch (e: unknown) {
        const statusCode =
          typeof e === 'object' && e !== null && 'response' in e
            ? (e as { response?: { status?: number } }).response?.status
            : undefined;
        if (statusCode === 404) {
          setIsPolling(false);
          setPollError('Batch job not found. Backend may have restarted.');
          return;
        }
        console.error(e);
      }
    }, 2000);

    return () => clearInterval(interval);
  }, [isPolling, jobId]);

  // Sorting results
  const sortedResults = [...(jobStatus?.results || [])].sort((a, b) => {
    const valA = a.metrics?.sharpe_ratio || -999;
    const valB = b.metrics?.sharpe_ratio || -999;
    return valB - valA; 
  });

  // Aggregate by Strategy
  const strategySummary = jobStatus?.results.reduce<Record<string, StrategyAggregate>>((acc, res) => {
    if (!res.metrics) return acc;
    const sName = res.task.strategy_name;
    if (!acc[sName]) {
      acc[sName] = { name: sName, count: 0, totalReturn: 0, totalSharpe: 0, totalDD: 0, wins: 0 };
    }
    acc[sName].count += 1;
    acc[sName].totalReturn += res.metrics.total_return;
    acc[sName].totalSharpe += res.metrics.sharpe_ratio;
    acc[sName].totalDD += res.metrics.max_drawdown;
    if (res.metrics.total_return > 0) acc[sName].wins += 1;
    return acc;
  }, {});

  const summaryList = strategySummary ? Object.values(strategySummary).map(s => ({
    ...s,
    avgReturn: s.totalReturn / s.count,
    avgSharpe: s.totalSharpe / s.count,
    avgDD: s.totalDD / s.count,
    winRate: (s.wins / s.count) * 100
  })).sort((a, b) => b.avgSharpe - a.avgSharpe) : [];

  const [activeTab, setActiveTab] = useState<'details' | 'summary'>('details');

  return (
    <div className="p-6 space-y-6">
      <Card>
        <CardHeader>
          <div className="flex justify-between items-center">
            <CardTitle>Batch Backtester (Professional)</CardTitle>
            {jobStatus?.status === 'completed' && (
              <div className="flex bg-muted p-1 rounded">
                <button 
                  onClick={() => setActiveTab('details')}
                  className={`px-3 py-1 text-xs rounded ${activeTab === 'details' ? 'bg-background shadow-sm font-bold' : ''}`}
                >
                  Detail List
                </button>
                <button 
                  onClick={() => setActiveTab('summary')}
                  className={`px-3 py-1 text-xs rounded ${activeTab === 'summary' ? 'bg-background shadow-sm font-bold' : ''}`}
                >
                  Strategy Leaderboard
                </button>
              </div>
            )}
          </div>
        </CardHeader>
        <CardContent className="space-y-6">
          
          {/* Configuration */}
          <div className={`${jobStatus?.status === 'completed' ? 'hidden' : 'grid'} grid-cols-1 md:grid-cols-3 gap-6 animate-in fade-in`}>
            
            {/* Symbols */}
            <div className="space-y-2 border p-3 rounded border-border">
              <div className="flex justify-between items-center mb-2">
                <Label className="font-bold">Symbols</Label>
                <div className="space-x-2">
                  <Button variant="outline" onClick={selectTop20} className="text-xs h-6">Top 20</Button>
                  <Button variant="outline" onClick={clearSymbols} className="text-xs h-6">Clear</Button>
                </div>
              </div>
              <div className="h-40 overflow-y-auto space-y-1 bg-black/20 p-2 rounded text-[10px] font-mono">
                {selectedSymbols.map(sym => (
                  <div key={sym} className="flex items-center space-x-2">
                    <Checkbox id={`sym-${sym}`} checked onCheckedChange={() => toggleSymbol(sym)} />
                    <label htmlFor={`sym-${sym}`}>{sym}</label>
                  </div>
                ))}
                {selectedSymbols.length === 0 && <p className="text-muted-foreground">No symbols selected</p>}
              </div>
              <div className="flex gap-2 mt-2">
                <Input 
                  placeholder="Add symbol..." 
                  value={customSymbol} 
                  onChange={e => setCustomSymbol(e.target.value)}
                  className="h-8 text-xs"
                />
                <Button onClick={addCustomSymbol} className="h-8">+</Button>
              </div>
            </div>

            {/* Strategies */}
            <div className="space-y-2 border p-3 rounded border-border">
              <Label className="font-bold mb-2 block">Strategies</Label>
              <div className="h-48 overflow-y-auto space-y-1 bg-black/20 p-2 rounded text-[10px] font-mono">
                {ALL_STRATEGIES.map(strat => (
                  <div key={strat} className="flex items-center space-x-2">
                    <Checkbox 
                      id={`strat-${strat}`} 
                      checked={selectedStrategies.includes(strat)} 
                      onCheckedChange={() => toggleStrategy(strat)} 
                    />
                    <label htmlFor={`strat-${strat}`}>{strat}</label>
                  </div>
                ))}
              </div>
            </div>

            {/* Timeframes & Dates */}
            <div className="space-y-4 border p-3 rounded border-border">
              <div>
                <Label className="font-bold mb-2 block">Timeframes</Label>
                <div className="flex flex-wrap gap-2">
                  {ALL_TIMEFRAMES.map(tf => (
                    <div key={tf} className="flex items-center space-x-1 text-xs bg-muted/20 px-2 py-1 rounded">
                      <Checkbox 
                        id={`tf-${tf}`} 
                        checked={selectedTimeframes.includes(tf)} 
                        onCheckedChange={() => toggleTimeframe(tf)} 
                      />
                      <label htmlFor={`tf-${tf}`}>{tf}</label>
                    </div>
                  ))}
                </div>
              </div>
              
              <div className="space-y-2">
                <Label htmlFor="start">Start Date</Label>
                <Input id="start" value={startDate} onChange={e => setStartDate(e.target.value)} />
              </div>
              <div className="space-y-2">
                <Label htmlFor="end">End Date</Label>
                <Input id="end" value={endDate} onChange={e => setEndDate(e.target.value)} />
              </div>
            </div>
          </div>

          <div className="flex items-center gap-4">
            <Button onClick={startBatch} disabled={isPolling} className="w-full md:w-auto">
              {isPolling ? 'Running Batch...' : 'Start New Batch Run'}
            </Button>
            {jobStatus && (
              <div className="text-sm font-mono">
                <span className="text-primary">
                  Progress: {jobStatus.progress.completed} / {jobStatus.progress.total} ({jobStatus.progress.percent}%)
                </span>
                {jobStatus.status === 'completed' && <span className="text-green-500 ml-2 font-bold">[COMPLETED]</span>}
                {jobStatus.status === 'failed' && <span className="text-red-500 ml-2 font-bold">[FAILED]</span>}
                {jobStatus.status === 'failed' && jobStatus.error && (
                  <p className="text-red-500 mt-1 text-xs">{jobStatus.error}</p>
                )}
                {jobStatus.message && (
                  <p className="text-muted-foreground mt-1 text-xs">{jobStatus.message}</p>
                )}
              </div>
            )}
            {pollError && (
              <p className="text-xs text-red-500 font-mono">{pollError}</p>
            )}
          </div>

          {/* Results Views */}
          {activeTab === 'summary' && summaryList.length > 0 && (
            <div className="border rounded overflow-hidden animate-in slide-in-from-bottom-2 duration-300">
              <div className="bg-primary/5 p-3 border-b border-border">
                <h3 className="text-sm font-bold uppercase tracking-widest">Aggregated Performance by Strategy</h3>
                <p className="text-[10px] text-muted-foreground">Averages across {selectedSymbols.length} coins and {selectedTimeframes.length} timeframes.</p>
              </div>
              <table className="w-full text-xs text-left">
                <thead className="bg-muted/50 text-muted-foreground font-mono uppercase">
                  <tr>
                    <th className="p-2">Strategy Name</th>
                    <th className="p-2 text-right">Avg Return</th>
                    <th className="p-2 text-right">Avg Sharpe</th>
                    <th className="p-2 text-right">Avg Max DD</th>
                    <th className="p-2 text-right">Profit Factor (Win%)</th>
                    <th className="p-2 text-right">Tests</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-border">
                  {summaryList.map((s, i) => (
                    <tr key={i} className="hover:bg-primary/5 transition-colors">
                      <td className="p-2 font-bold text-primary">{s.name}</td>
                      <td className={`p-2 text-right ${s.avgReturn > 0 ? 'text-green-400' : 'text-red-400'}`}>
                        {(s.avgReturn * 100).toFixed(2)}%
                      </td>
                      <td className="p-2 text-right font-mono font-bold">{s.avgSharpe.toFixed(2)}</td>
                      <td className="p-2 text-right text-red-400">{(s.avgDD * 100).toFixed(2)}%</td>
                      <td className="p-2 text-right">{s.winRate.toFixed(1)}%</td>
                      <td className="p-2 text-right text-muted-foreground">{s.count}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}

          {activeTab === 'details' && sortedResults.length > 0 && (
            <div className="border rounded overflow-hidden animate-in fade-in duration-300">
              <div className="overflow-x-auto">
                <table className="w-full text-[11px] text-left">
                  <thead className="bg-muted/50 text-muted-foreground font-mono uppercase">
                    <tr>
                      <th className="p-2">Symbol</th>
                      <th className="p-2">Strategy</th>
                      <th className="p-2">TF</th>
                      <th className="p-2 text-right">Return</th>
                      <th className="p-2 text-right">Sharpe</th>
                      <th className="p-2 text-right">Max DD</th>
                      <th className="p-2 text-right">Trades</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-border">
                    {sortedResults.map((res, i) => {
                      const m = res.metrics;
                      const isError = !!res.error || !m;
                      const errorText = res.error || 'No metrics returned by backend.';
                      return (
                        <tr key={i} className="hover:bg-muted/10 transition-colors">
                          <td className="p-2 font-bold">{res.task.symbol}</td>
                          <td className="p-2 text-muted-foreground">{res.task.strategy_name}</td>
                          <td className="p-2 font-mono">{res.task.timeframe}</td>
                          {isError ? (
                            <td colSpan={4} className="p-2 text-red-500">{errorText}</td>
                          ) : (
                            <>
                              <td className={`p-2 text-right ${(m?.total_return || 0) > 0 ? 'text-green-400' : 'text-red-400'}`}>
                                {((m?.total_return || 0) * 100).toFixed(1)}%
                              </td>
                              <td className="p-2 text-right font-mono font-bold">{(m?.sharpe_ratio || 0).toFixed(2)}</td>
                              <td className="p-2 text-right text-red-400">{((m?.max_drawdown || 0) * 100).toFixed(1)}%</td>
                              <td className="p-2 text-right">{m?.total_trades}</td>
                            </>
                          )}
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>
            </div>
          )}

        </CardContent>
      </Card>
    </div>
  );
}
