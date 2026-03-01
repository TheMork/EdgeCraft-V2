'use client';

import React from 'react';
import { Play, TrendingUp, Layers, Activity } from 'lucide-react';
import Link from 'next/link';
import { usePathname } from 'next/navigation';
import { useStore } from '@/lib/store';
import { cn } from '@/lib/utils';

export function Sidebar() {
  const pathname = usePathname();
  const {
    selectedSymbol,
    setSymbol,
    selectedStrategy,
    setStrategy,
    selectedTimeframe,
    setTimeframe
  } = useStore();

  const navItems = [
    { label: 'Data Manager', href: '/data', icon: Activity },
    { label: 'Single Simulation', href: '/simulation', icon: Activity },
    { label: 'Batch Backtester', href: '/batch', icon: Layers },
    { label: 'Portfolio Simulation', href: '/multi', icon: Activity },
    { label: 'Parameter Sweep', href: '/sweep', icon: TrendingUp },
  ];

  return (
    <aside className="w-[260px] border-r border-border bg-surface flex flex-col h-full font-sans shadow-[2px_0_8px_rgba(0,0,0,0.1)]">
      <div className="p-4 border-b border-border">
        <h2 className="text-xs font-bold text-text-muted mb-3 uppercase tracking-wider">Navigation</h2>
        <div className="space-y-1">
          {navItems.map((item) => {
            const Icon = item.icon;
            const isActive = pathname === item.href;
            return (
              <Link 
                key={item.href}
                href={item.href} 
                className={cn(
                  "flex items-center gap-3 px-3 py-2 text-sm rounded-md transition-all duration-200",
                  isActive 
                    ? "bg-primary/10 text-primary font-medium shadow-sm"
                    : "text-text-muted hover:bg-muted/50 hover:text-text-primary"
                )}
              >
                <Icon size={16} className={isActive ? "text-primary" : "text-text-muted"} />
                {item.label}
              </Link>
            );
          })}
        </div>
      </div>

      <div className="p-4 border-b border-border">
        <h2 className="text-xs font-bold text-text-muted mb-2 uppercase tracking-wider">Strategy</h2>
        <select
          className="w-full bg-background border border-border rounded-md p-2 text-sm text-text-primary focus:outline-none focus:ring-1 focus:ring-primary shadow-sm transition-colors cursor-pointer"
          value={selectedStrategy}
          onChange={(e) =>
            setStrategy(
              e.target.value as
                | 'demo'
                | 'momentum'
                | 'double_rsi_divergence'
                | 'multi_divergence'
                | 'pair_arbitrage_v1'
                | 'pair_arbitrage_v2'
                | 'pair_arbitrage_v3'
                | 'pair_arbitrage_v4'
                | 'mde_mad_entropy'
                | 'mde_mad_classic'
                | 'mde_mad_v2'
                | 'mde_mad_v2_leverage'
                | 'mde_mad_v3'
                | 'mde_mad_v3_1'
                | 'mde_mad_v4'
            )
          }
        >
          <option value="momentum">Quantitative Momentum</option>
          <option value="double_rsi_divergence">Double RSI Divergence</option>
          <option value="multi_divergence">Multi Divergence (3x)</option>
          <option value="pair_arbitrage_v1">Pair Arbitrage v1 (OLS)</option>
          <option value="pair_arbitrage_v2">Pair Arbitrage v2 (Kalman+OU)</option>
          <option value="pair_arbitrage_v3">Pair Arbitrage v3 (Regime)</option>
          <option value="pair_arbitrage_v4">Pair Arbitrage v4 (Adaptive)</option>
          <option value="mde_mad_entropy">Mathematical Logic (MDE)</option>
          <option value="mde_mad_classic">MDE Classic (Sweep Winner)</option>
          <option value="mde_mad_v2">MDE v2 (Trend + Turnover)</option>
          <option value="mde_mad_v2_leverage">MDE v2 Leverage</option>
          <option value="mde_mad_v3">MDE v3 (Dynamic Lookback)</option>
          <option value="mde_mad_v3_1">MDE v3.1 (Aggressive + Dynamic)</option>
          <option value="mde_mad_v4">MDE v4 (All-Weather Optimized)</option>
          <option value="demo">Demo Strategy</option>
        </select>
      </div>

      <div className="p-4 flex-1 overflow-y-auto">
        <div className="space-y-4">
          <div>
            <h2 className="text-xs font-bold text-text-muted mb-3 uppercase tracking-wider">Market</h2>
            <div className="space-y-3">
              <div>
                <label className="text-[11px] font-semibold text-text-muted block mb-1 uppercase tracking-wide">Symbol</label>
                <select
                  className="w-full bg-background border border-border rounded-md p-2 text-sm text-text-primary focus:outline-none focus:ring-1 focus:ring-primary shadow-sm transition-colors cursor-pointer"
                  value={selectedSymbol}
                  onChange={(e) => setSymbol(e.target.value)}
                >
                  <option value="BTC/USDT">BTC/USDT</option>
                  <option value="ETH/USDT">ETH/USDT</option>
                  <option value="SOL/USDT">SOL/USDT</option>
                  <option value="BNB/USDT">BNB/USDT</option>
                  <option value="XRP/USDT">XRP/USDT</option>
                </select>
              </div>
              <div>
                <label className="text-[11px] font-semibold text-text-muted block mb-1 uppercase tracking-wide">Timeframe</label>
                <select
                  className="w-full bg-background border border-border rounded-md p-2 text-sm text-text-primary focus:outline-none focus:ring-1 focus:ring-primary shadow-sm transition-colors cursor-pointer"
                  value={selectedTimeframe}
                  onChange={(e) => setTimeframe(e.target.value)}
                >
                  <option value="1m">1m</option>
                  <option value="3m">3m</option>
                  <option value="5m">5m</option>
                  <option value="15m">15m</option>
                  <option value="30m">30m</option>
                  <option value="1h">1h</option>
                  <option value="2h">2h</option>
                  <option value="4h">4h</option>
                  <option value="6h">6h</option>
                  <option value="8h">8h</option>
                  <option value="12h">12h</option>
                  <option value="1d">1d</option>
                  <option value="3d">3d</option>
                  <option value="1w">1w</option>
                  <option value="1M">1M</option>
                </select>
              </div>
            </div>
          </div>
        </div>
      </div>

      <div className="p-4 border-t border-border space-y-2 bg-surface">
        <Link href="/simulation" className="w-full bg-primary text-primary-foreground hover:bg-primary/90 font-medium py-2 px-4 rounded-md flex items-center justify-center gap-2 cursor-pointer transition-all duration-200 shadow-sm text-sm">
          <Play size={16} /> Run Backtest
        </Link>
        <Link href="/sweep" className="w-full bg-background border border-border text-text-primary hover:bg-accent font-medium py-2 px-4 rounded-md flex items-center justify-center gap-2 cursor-pointer transition-all duration-200 shadow-sm text-sm">
          <TrendingUp size={16} /> Optimize
        </Link>
      </div>
    </aside>
  );
}
