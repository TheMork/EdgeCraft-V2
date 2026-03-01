"use client";

import React, { useState, useEffect } from 'react';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Card, CardHeader, CardTitle, CardContent } from '@/components/ui/card';
import api from '@/lib/api';
import { useStore } from '@/lib/store';

interface SweepJobStatus {
  job_id: string;
  status: string;
  progress: number;
  total_combinations: number;
}

interface SweepResult {
  parameters: Record<string, any>;
  sharpe_ratio: number;
  total_return: number;
  max_drawdown: number;
  win_rate: number;
  total_trades: number;
}

export default function SweepPage() {
  const { selectedSymbol, selectedStrategy, selectedTimeframe, startDate, setStartDate, endDate, setEndDate } = useStore();

  const [method, setMethod] = useState<'grid' | 'bayesian'>('grid');
  const [nTrials, setNTrials] = useState(100);
  const [processes, setProcesses] = useState(4);
  const [initialBalance, setInitialBalance] = useState(10000);
  const [leverage, setLeverage] = useState(1);

  const [paramGridText, setParamGridText] = useState('{\n  "window_size": [10, 20, 30],\n  "threshold": [0.5, 1.0, 1.5]\n}');

  const [jobId, setJobId] = useState<string | null>(null);
  const [jobStatus, setJobStatus] = useState<SweepJobStatus | null>(null);
  const [results, setResults] = useState<SweepResult[]>([]);
  const [isPolling, setIsPolling] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const startSweep = async () => {
    try {
      setError(null);
      setJobStatus(null);
      setResults([]);

      let parsedGrid = {};
      try {
        parsedGrid = JSON.parse(paramGridText);
      } catch (e) {
        setError('Invalid JSON in Parameter Grid. Please check the syntax.');
        return;
      }

      const payload = {
        strategy_name: selectedStrategy,
        symbol: selectedSymbol,
        start_date: startDate,
        end_date: endDate,
        timeframe: selectedTimeframe,
        initial_balance: initialBalance,
        leverage: leverage,
        param_grid: parsedGrid,
        method: method,
        n_trials: nTrials,
        processes: processes
      };

      const response = await api.post('/api/v1/sweep/start', payload);
      setJobId(response.data.job_id);
      setIsPolling(true);
    } catch (e: any) {
      console.error(e);
      setError(e.response?.data?.detail || 'Failed to start sweep job.');
    }
  };

  const fetchResults = async (id: string) => {
    try {
      const res = await api.get(`/api/v1/sweep/${id}/results`);
      setResults(res.data.results || []);
    } catch (e) {
      console.error('Failed to fetch results', e);
    }
  };

  const cancelSweep = async () => {
    if (!jobId) return;
    try {
      await api.delete(`/api/v1/sweep/${jobId}`);
      setIsPolling(false);
      setJobStatus(prev => prev ? { ...prev, status: 'cancelled' } : null);
    } catch (e) {
      console.error('Failed to cancel sweep', e);
    }
  };

  useEffect(() => {
    if (!isPolling || !jobId) return;

    const interval = setInterval(async () => {
      try {
        const response = await api.get<SweepJobStatus>(`/api/v1/sweep/${jobId}/status`);
        setJobStatus(response.data);

        if (response.data.status === 'completed' || response.data.status === 'cancelled' || response.data.status.startsWith('failed')) {
          setIsPolling(false);
          await fetchResults(jobId);
        }
      } catch (e: any) {
        if (e.response?.status === 404) {
          setIsPolling(false);
          setError('Job not found on backend. It may have restarted.');
        } else {
          console.error(e);
        }
      }
    }, 2000);

    return () => clearInterval(interval);
  }, [isPolling, jobId]);

  return (
    <div className="p-6 space-y-6">
      <Card>
        <CardHeader>
          <CardTitle>Parameter Sweep Configuration</CardTitle>
        </CardHeader>
        <CardContent className="space-y-6">

          <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-4 gap-4">
            <div className="space-y-2">
              <Label>Market Context</Label>
              <div className="flex flex-wrap gap-2 text-xs font-mono p-2 border border-border rounded bg-muted/20">
                <span>{selectedSymbol}</span>
                <span>|</span>
                <span>{selectedTimeframe}</span>
                <span>|</span>
                <span>{selectedStrategy}</span>
              </div>
            </div>

            <div className="space-y-2">
              <Label htmlFor="method">Optimization Method</Label>
              <select
                id="method"
                className="w-full h-10 px-3 py-2 bg-black border border-border rounded text-sm text-primary focus:outline-none focus:ring-1 focus:ring-primary"
                value={method}
                onChange={(e) => setMethod(e.target.value as 'grid' | 'bayesian')}
                disabled={isPolling}
              >
                <option value="grid">Grid Search</option>
                <option value="bayesian">Bayesian Optimization</option>
              </select>
            </div>

            <div className="space-y-2">
              <Label htmlFor="startDate">Start Date</Label>
              <Input
                id="startDate"
                value={startDate}
                onChange={(e) => setStartDate(e.target.value)}
                disabled={isPolling}
              />
            </div>

            <div className="space-y-2">
              <Label htmlFor="endDate">End Date</Label>
              <Input
                id="endDate"
                value={endDate}
                onChange={(e) => setEndDate(e.target.value)}
                disabled={isPolling}
              />
            </div>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
            <div className="space-y-2">
              <Label htmlFor="initialBalance">Initial Balance</Label>
              <Input
                id="initialBalance"
                type="number"
                value={initialBalance}
                onChange={(e) => setInitialBalance(Number(e.target.value))}
                disabled={isPolling}
              />
            </div>

            <div className="space-y-2">
              <Label htmlFor="leverage">Leverage</Label>
              <Input
                id="leverage"
                type="number"
                value={leverage}
                onChange={(e) => setLeverage(Number(e.target.value))}
                disabled={isPolling}
              />
            </div>

            {method === 'bayesian' && (
              <div className="space-y-2">
                <Label htmlFor="nTrials">Number of Trials</Label>
                <Input
                  id="nTrials"
                  type="number"
                  value={nTrials}
                  onChange={(e) => setNTrials(Number(e.target.value))}
                  disabled={isPolling}
                />
              </div>
            )}

            {method === 'grid' && (
              <div className="space-y-2">
                <Label htmlFor="processes">Parallel Processes</Label>
                <Input
                  id="processes"
                  type="number"
                  min={1}
                  max={32}
                  value={processes}
                  onChange={(e) => setProcesses(Number(e.target.value))}
                  disabled={isPolling}
                />
              </div>
            )}
          </div>

          <div className="space-y-2">
            <Label htmlFor="paramGrid">Parameter Grid (JSON array mapping)</Label>
            <textarea
              id="paramGrid"
              className="w-full h-40 bg-black/50 border border-border rounded p-3 text-sm font-mono text-primary focus:outline-none focus:ring-1 focus:ring-primary"
              value={paramGridText}
              onChange={(e) => setParamGridText(e.target.value)}
              disabled={isPolling}
              placeholder={'{\n  "param1": [1, 2, 3],\n  "param2": [0.1, 0.5, 0.9]\n}'}
            />
            <p className="text-xs text-muted-foreground">
              Provide a valid JSON object where keys are parameter names from the `{selectedStrategy}` strategy and values are arrays of options. For Bayesian, specify `[min, max]` for numeric boundaries or an array of categories.
            </p>
          </div>

          <div className="flex flex-wrap items-center gap-4">
            <Button onClick={startSweep} disabled={isPolling}>
              {isPolling ? 'Running Sweep...' : 'Start Sweep'}
            </Button>

            {isPolling && (
              <Button variant="destructive" onClick={cancelSweep}>
                Cancel Job
              </Button>
            )}

            {error && <p className="text-red-500 text-sm">{error}</p>}

            {jobStatus && (
              <div className="text-sm font-mono">
                <span className="text-primary">
                  Status: {jobStatus.status} | Progress: {jobStatus.progress} / {jobStatus.total_combinations}
                </span>
                {jobStatus.total_combinations > 0 && (
                  <span className="ml-2">
                    ({((jobStatus.progress / jobStatus.total_combinations) * 100).toFixed(1)}%)
                  </span>
                )}
              </div>
            )}
          </div>

        </CardContent>
      </Card>

      {results.length > 0 && (
        <Card className="animate-in fade-in">
          <CardHeader>
            <CardTitle>Sweep Results (Top Combinations)</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="overflow-x-auto">
              <table className="w-full text-xs text-left">
                <thead className="bg-muted/50 text-muted-foreground font-mono uppercase">
                  <tr>
                    <th className="p-2">Rank</th>
                    <th className="p-2">Parameters</th>
                    <th className="p-2 text-right">Sharpe Ratio</th>
                    <th className="p-2 text-right">Total Return</th>
                    <th className="p-2 text-right">Max Drawdown</th>
                    <th className="p-2 text-right">Win Rate</th>
                    <th className="p-2 text-right">Trades</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-border">
                  {results.map((res, i) => (
                    <tr key={i} className="hover:bg-muted/10 transition-colors">
                      <td className="p-2 font-bold">{i + 1}</td>
                      <td className="p-2 font-mono">
                        {Object.entries(res.parameters).map(([k, v]) => (
                          <div key={k}>{k}: {String(v)}</div>
                        ))}
                      </td>
                      <td className="p-2 text-right font-mono font-bold text-primary">
                        {res.sharpe_ratio.toFixed(2)}
                      </td>
                      <td className={`p-2 text-right ${res.total_return > 0 ? 'text-green-400' : 'text-red-400'}`}>
                        {(res.total_return * 100).toFixed(2)}%
                      </td>
                      <td className="p-2 text-right text-red-400">
                        {(res.max_drawdown * 100).toFixed(2)}%
                      </td>
                      <td className="p-2 text-right">
                        {(res.win_rate * 100).toFixed(1)}%
                      </td>
                      <td className="p-2 text-right text-muted-foreground">
                        {res.total_trades}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </CardContent>
        </Card>
      )}
    </div>
  );
}
