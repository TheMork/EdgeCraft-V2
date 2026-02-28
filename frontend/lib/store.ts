import { create } from 'zustand';

type StrategyName =
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
  | 'mde_mad_v4';

interface AppState {
  selectedSymbol: string;
  selectedStrategy: StrategyName;
  selectedTimeframe: string;
  startDate: string;
  endDate: string;
  setSymbol: (symbol: string) => void;
  setStrategy: (strategy: StrategyName) => void;
  setTimeframe: (timeframe: string) => void;
  setStartDate: (date: string) => void;
  setEndDate: (date: string) => void;
}

export const useStore = create<AppState>((set) => ({
  selectedSymbol: 'BTC/USDT',
  selectedStrategy: 'momentum',
  selectedTimeframe: '1h',
  startDate: '2024-01-01T00:00:00',
  endDate: '2024-01-02T00:00:00',
  setSymbol: (symbol) => set({ selectedSymbol: symbol }),
  setStrategy: (strategy) => set({ selectedStrategy: strategy }),
  setTimeframe: (timeframe) => set({ selectedTimeframe: timeframe }),
  setStartDate: (date) => set({ startDate: date }),
  setEndDate: (date) => set({ endDate: date }),
}));
