'use client';

import React from 'react';
import { useStore } from '@/lib/store';
import { Activity } from 'lucide-react';

export function Header() {
  const { selectedSymbol, selectedStrategy } = useStore();

  return (
    <header className="h-16 border-b border-border bg-surface flex items-center justify-between px-6 shadow-sm">
      <div className="flex items-center gap-3">
        <div className="w-8 h-8 rounded-lg bg-primary/10 flex items-center justify-center border border-primary/20">
          <Activity size={18} className="text-primary" />
        </div>
        <h1 className="text-lg font-bold text-text-primary tracking-tight">
          Crypto Quant Dashboard
        </h1>
      </div>

      <div className="hidden md:flex items-center gap-4 text-sm">
        <div className="flex items-center gap-2">
          <span className="text-text-muted">Strategy:</span>
          <span className="px-2 py-1 bg-background border border-border rounded-md font-mono text-xs">
            {selectedStrategy}
          </span>
        </div>
        <div className="flex items-center gap-2">
          <span className="text-text-muted">Symbol:</span>
          <span className="px-2 py-1 bg-background border border-border rounded-md font-mono text-xs font-semibold text-text-primary">
            {selectedSymbol}
          </span>
        </div>
        <div className="flex items-center gap-2 border-l border-border pl-4">
          <div className="w-2 h-2 rounded-full bg-primary animate-pulse shadow-[0_0_8px_var(--primary)]" />
          <span className="text-xs text-text-muted">System Active</span>
        </div>
      </div>
    </header>
  );
}
