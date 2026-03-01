'use client';

import React from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';

interface Metrics {
  total_return?: number;
  sharpe_ratio?: number;
  sortino_ratio?: number;
  max_drawdown?: number;
  max_drawdown_close?: number;
  max_drawdown_worst?: number;
  final_equity?: number;
  total_trades?: number;
}

interface MetricsCardProps {
  metrics: Metrics;
}

export default function MetricsCard({ metrics }: MetricsCardProps) {
  const formatNumber = (num?: number, decimals: number = 2) => {
    if (num === undefined || num === null) return '-';
    return num.toFixed(decimals);
  };

  const formatPercent = (num?: number) => {
    if (num === undefined || num === null) return '-';
    return (num * 100).toFixed(2) + '%';
  };

  return (
    <Card>
      <CardHeader>
        <CardTitle className="uppercase tracking-widest text-text-primary text-sm font-semibold">Backtest Metrics</CardTitle>
      </CardHeader>
      <CardContent className="grid grid-cols-2 md:grid-cols-3 gap-4">
        <div>
            <div className="text-xs uppercase tracking-wide text-muted-foreground">Total Return</div>
            <div className={`text-2xl font-bold font-mono ${(metrics.total_return || 0) >= 0 ? 'text-bullish' : 'text-bearish'}`}>
                {formatPercent(metrics.total_return)}
            </div>
        </div>
        <div>
            <div className="text-xs uppercase tracking-wide text-muted-foreground">Sharpe Ratio</div>
            <div className="text-2xl font-bold font-mono text-text-primary">{formatNumber(metrics.sharpe_ratio)}</div>
        </div>
        <div>
            <div className="text-xs uppercase tracking-wide text-muted-foreground">Sortino Ratio</div>
            <div className="text-2xl font-bold font-mono text-text-primary">{formatNumber(metrics.sortino_ratio)}</div>
        </div>
        <div>
            <div className="text-xs uppercase tracking-wide text-muted-foreground">Max Drawdown</div>
            <div className="text-2xl font-bold font-mono text-bearish">{formatPercent(metrics.max_drawdown)}</div>
            {(metrics.max_drawdown_close !== undefined || metrics.max_drawdown_worst !== undefined) && (
              <div className="text-[11px] font-mono text-muted-foreground mt-1">
                close: {formatPercent(metrics.max_drawdown_close)} | worst: {formatPercent(metrics.max_drawdown_worst)}
              </div>
            )}
        </div>
        <div>
            <div className="text-xs uppercase tracking-wide text-muted-foreground">Final Equity</div>
            <div className="text-2xl font-bold font-mono text-text-primary">${formatNumber(metrics.final_equity)}</div>
        </div>
        <div>
            <div className="text-xs uppercase tracking-wide text-muted-foreground">Total Trades</div>
            <div className="text-2xl font-bold font-mono text-text-primary">{formatNumber(metrics.total_trades, 0)}</div>
        </div>
      </CardContent>
    </Card>
  );
}
