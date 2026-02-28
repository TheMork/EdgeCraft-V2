
console.log("BotRunner script loading...");

import { ExchangeClient } from './ExchangeClient';
import { StrategyEngine, StrategySignal, type BotStrategyName } from '../strategies/StrategyEngine';
import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';

console.log("Imports loaded.");

// Fix __dirname in ES modules / CJS Bundle compatibility
let __dirname_local = '';
try {
    // @ts-ignore
    if (typeof import.meta !== 'undefined' && import.meta.url) {
        // @ts-ignore
        const __filename = fileURLToPath(import.meta.url);
        __dirname_local = path.dirname(__filename);
    } else {
        // Fallback for CJS/Bundle
        __dirname_local = __dirname;
    }
} catch (e) {
    console.warn("Could not determine __dirname, using cwd");
    __dirname_local = process.cwd();
}
console.log("Dirname resolved:", __dirname_local);

const CONFIG = {
    // Symbol is now dynamic based on Top 20
    highTF: (process.env.BOT_TIMEFRAME_HIGH || '1d'),
    lowTF: (process.env.BOT_TIMEFRAME_LOW || '4h'),
    riskPercent: parseFloat(process.env.BOT_RISK_PERCENT || '0.01'),
    strategy: (process.env.BOT_STRATEGY || 'ict_trend_mss_fvg').toLowerCase(),
    historyCutoff: process.env.HISTORY_CUTOFF ? (!isNaN(Number(process.env.HISTORY_CUTOFF)) ? Number(process.env.HISTORY_CUTOFF) : new Date(process.env.HISTORY_CUTOFF).getTime()) : 0,
    loopInterval: 30000, // 30 seconds for faster updates
    maxActiveTrades: 20, // Increased to 20 so all monitored coins can be traded concurrently
    requiredQuoteVolume: 0, // Lowered for Testnet (was 10000000)
    enableShorts: process.env.BOT_ENABLE_SHORTS === 'true',
};
console.log("Config loaded:", CONFIG);

interface Trade {
    id: string;
    symbol: string;
    type: 'LONG' | 'SHORT';
    entryPrice: number;
    stopLoss: number;
    originalSL: number; // For R calc
    quantity: number;
    status: 'OPEN' | 'CLOSED';
    entryTime: number;
    exitTime?: number;
    exitPrice?: number;
    pnl?: number;
    pnlPercent?: number;
    highestPrice: number; // For Trailing Long
    lowestPrice: number;  // For Trailing Short
    closeReason?: string;
}

interface BotState {
    activeTrades: Trade[];
    tradeHistory: Trade[];
    historyCutoff: number;
    lastUpdate: number;
    currentStrategy: BotStrategyName;
    monitoredSymbols: string[];
    monitoringQueue: Record<string, { status: string, details: string, lastChecked: number }>;
    performance: {
        totalTrades: number;
        wins: number;
        losses: number;
        breakEvens: number;
        winRate: number;
        totalPnL: number;
    }
    cooldownUntil: Record<string, number>;
    symbolWeights: Record<string, number>;
}

// Path to public/data/bot_state.json so frontend can read it
// Start from project root (assuming we run from root or dist-bot)
// We'll try to resolve to the project root 'public' folder.
const PUBLIC_DATA_DIR = path.resolve(process.cwd(), 'public', 'data');
const STATE_FILE = path.join(PUBLIC_DATA_DIR, 'bot_state.json');

const DIST_DATA_DIR = path.resolve(process.cwd(), 'dist', 'data');
const DIST_STATE_FILE = path.join(DIST_DATA_DIR, 'bot_state.json');

const MDE_REBALANCE_CONFIG = {
    minRebalanceWeightDelta: 0.005,
    minRebalanceNotional: 10,
    enableCostGate: true,
    feeRate: 0.0004,
    slippageRate: 0.0002,
    edgeHorizonBars: 3,
    minEdgeOverCostRatio: 1.25,
    cooldownHours: 1,
    maxAbsTargetWeight: 0.10,
    maxHoldingBars: 20,
    enableAtrStop: true,
    atrPeriod: 14,
    atrStopMultiplier: 3.0,
    maxGrossExposureWeight: 3.0, // 3x account equity gross exposure cap
};

const BEST_MDE_OVERRIDES = {
    lookbackBars: 80,
    trendFilterPeriod: 0,
    riskAversion: 4.5,
    entropyWeight: 0.08,
    turnoverPenalty: 0.09,
    maxLeverage: 3,
    optimizationSteps: 41,
    atrPeriod: 14,
    atrStopMultiplier: 3.0,
    entryWeightThreshold: 0.10,
    trendFilterType: 'EMA' as const,
};

// Ensure directory exists
if (!fs.existsSync(PUBLIC_DATA_DIR)) {
    console.log(`Creating directory: ${PUBLIC_DATA_DIR}`);
    fs.mkdirSync(PUBLIC_DATA_DIR, { recursive: true });
}

class BotRunner {
    private client: ExchangeClient;
    private engine: StrategyEngine;
    private state: BotState;
    private readonly strategyName: BotStrategyName;

    constructor() {
        console.log("BotRunner Constructor: Init ExchangeClient...");
        this.client = new ExchangeClient('binance', true); // Testnet = true
        this.strategyName = this.resolveStrategyName(CONFIG.strategy);
        console.log("BotRunner Constructor: Init StrategyEngine...");
        this.engine = new StrategyEngine(this.strategyName, {
            ...BEST_MDE_OVERRIDES,
            enableShorts: CONFIG.enableShorts,
        });
        console.log("BotRunner Constructor: Loading State...");
        this.state = this.loadState();
        console.log("BotRunner Constructor: Done.");
    }

    private resolveStrategyName(raw: string): BotStrategyName {
        const normalized = raw.trim().toLowerCase();
        if (normalized === 'mde_mad_v2') return 'mde_mad_v2';
        return 'ict_trend_mss_fvg';
    }

    private loadState(): BotState {
        console.log("Loading state from", STATE_FILE);
        if (fs.existsSync(STATE_FILE)) {
            try {
                const data = JSON.parse(fs.readFileSync(STATE_FILE, 'utf-8'));
                // Ensure new fields exist if loading old state
                if (!data.activeTrades) data.activeTrades = [];
                if (!data.tradeHistory) data.tradeHistory = [];
                if (typeof data.historyCutoff !== 'number') data.historyCutoff = 0;

                // Override if ENV is provided and newer
                if (CONFIG.historyCutoff && !isNaN(CONFIG.historyCutoff) && CONFIG.historyCutoff > data.historyCutoff) {
                    data.historyCutoff = CONFIG.historyCutoff;
                    console.log(`[Bot] Applied HISTORY_CUTOFF from ENV: ${new Date(data.historyCutoff).toISOString()}`);
                    // Reset stats because cutoff changed
                    data.performance = { totalTrades: 0, wins: 0, losses: 0, breakEvens: 0, winRate: 0, totalPnL: 0 };
                }

                if (!data.currentStrategy) data.currentStrategy = this.strategyName;
                if (!data.monitoredSymbols) data.monitoredSymbols = [];
                if (!data.monitoringQueue) data.monitoringQueue = {};
                if (!data.cooldownUntil) data.cooldownUntil = {};
                if (!data.symbolWeights) data.symbolWeights = {};
                if (!data.performance) data.performance = { totalTrades: 0, wins: 0, losses: 0, breakEvens: 0, winRate: 0, totalPnL: 0 };

                // Reset persisted data when strategy changed
                const previousStrategy = typeof data.currentStrategy === 'string'
                    ? data.currentStrategy.trim().toLowerCase()
                    : undefined;
                if (previousStrategy && previousStrategy !== this.strategyName) {
                    console.log(`[Bot] Strategy changed from ${previousStrategy} to ${this.strategyName}; clearing persisted trades and stats.`);
                    data.activeTrades = [];
                    data.tradeHistory = [];
                    data.performance = { totalTrades: 0, wins: 0, losses: 0, breakEvens: 0, winRate: 0, totalPnL: 0 };
                    data.monitoringQueue = {};
                    data.cooldownUntil = {};
                    data.symbolWeights = {};
                    data.historyCutoff = 0;
                }

                // Migrate old single activeTrade to array if needed
                // @ts-ignore
                if (data.activeTrade) {
                    // @ts-ignore
                    data.activeTrades.push({ ...data.activeTrade, id: 'legacy-' + Date.now() });
                    // @ts-ignore
                    delete data.activeTrade;
                }
                if (data.historyCutoff > 0) {
                    data.tradeHistory = data.tradeHistory.filter((trade: Trade) => {
                        const ts = trade.exitTime || trade.entryTime || 0;
                        return ts >= data.historyCutoff;
                    });
                }
                console.log("State loaded successfully.");
                data.currentStrategy = this.strategyName;
                return data;
            } catch (e) {
                console.error("Error parsing state file, starting fresh:", e);
            }
        }
        console.log("No valid state file found, creating fresh state.");
        return {
            activeTrades: [],
            tradeHistory: [],
            historyCutoff: 0,
            lastUpdate: Date.now(),
            currentStrategy: this.strategyName,
            monitoredSymbols: [],
            monitoringQueue: {}, // New field for trade planning visibility
            cooldownUntil: {},
            symbolWeights: {},
            performance: { totalTrades: 0, wins: 0, losses: 0, breakEvens: 0, winRate: 0, totalPnL: 0 }
        };
    }

    private saveState() {
        this.updatePerformanceMetrics();
        this.state.lastUpdate = Date.now();
        const stateStr = JSON.stringify(this.state, null, 2);
        fs.writeFileSync(STATE_FILE, stateStr);
        if (fs.existsSync(DIST_DATA_DIR)) {
            try {
                fs.writeFileSync(DIST_STATE_FILE, stateStr);
            } catch (e) {
                console.error("Could not write to dist folder:", e);
            }
        }
    }

    private updatePerformanceMetrics() {
        // Filter history by cutoff so stats also reset
        const history = this.state.tradeHistory.filter(trade => {
            const ts = trade.exitTime || trade.entryTime || 0;
            return ts >= this.state.historyCutoff;
        });

        const BE_THRESHOLD = 0.2; // 0.2%
        const totalTrades = history.length;
        const breakEvens = history.filter(t => Math.abs(t.pnlPercent || 0) <= BE_THRESHOLD).length;
        const nonBESize = totalTrades - breakEvens;

        const wins = history.filter(t => (t.pnlPercent || 0) > BE_THRESHOLD).length;
        const losses = history.filter(t => (t.pnlPercent || 0) < -BE_THRESHOLD).length;

        // Win rate should ideally be based on non-BE trades if the user wants to exclude them
        const winRate = nonBESize > 0 ? (wins / nonBESize) * 100 : 0;
        const totalPnL = history.reduce((sum, t) => sum + (t.pnl || 0), 0);

        this.state.performance = {
            totalTrades,
            wins,
            losses,
            breakEvens,
            winRate,
            totalPnL
        };
    }

    private isCooldownActive(symbol: string): boolean {
        const until = this.state.cooldownUntil[symbol];
        if (!until) return false;
        if (Date.now() >= until) {
            delete this.state.cooldownUntil[symbol];
            return false;
        }
        return true;
    }

    private triggerCooldown(symbol: string): void {
        const ms = MDE_REBALANCE_CONFIG.cooldownHours * 3_600_000;
        this.state.cooldownUntil[symbol] = Date.now() + ms;
        console.log(`[Cooldown] ${symbol} locked for ${MDE_REBALANCE_CONFIG.cooldownHours}h`);
    }

    private getSymbolWeight(symbol: string): number {
        return this.state.symbolWeights[symbol] ?? 0;
    }

    private recordWeight(symbol: string, weight: number): void {
        this.state.symbolWeights[symbol] = weight;
    }

    private passesCostGate(tradeNotional: number, expectedReturn: number): boolean {
        if (!MDE_REBALANCE_CONFIG.enableCostGate) return true;
        const edge = tradeNotional * Math.abs(expectedReturn) * MDE_REBALANCE_CONFIG.edgeHorizonBars;
        const roundTripCost = tradeNotional * 2 * (MDE_REBALANCE_CONFIG.feeRate + MDE_REBALANCE_CONFIG.slippageRate);
        return edge >= roundTripCost * MDE_REBALANCE_CONFIG.minEdgeOverCostRatio;
    }

    private calculateTradePnl(trade: Trade, price: number): number {
        if (!trade) return 0;
        if (trade.type === 'LONG') {
            return (price - trade.entryPrice) * trade.quantity;
        }
        return (trade.entryPrice - price) * trade.quantity;
    }

    private getGrossExposureWeight(): number {
        let gross = 0;
        for (const v of Object.values(this.state.symbolWeights || {})) {
            const w = Number(v);
            if (Number.isFinite(w)) gross += Math.abs(w);
        }
        return gross;
    }

    private timeframeToMs(tf: string): number {
        const match = String(tf || '').trim().toLowerCase().match(/^(\d+)\s*([mhd])$/);
        if (!match) return 8 * 3_600_000;
        const value = parseInt(match[1], 10);
        const unit = match[2];
        if (!Number.isFinite(value) || value <= 0) return 8 * 3_600_000;
        if (unit === 'm') return value * 60_000;
        if (unit === 'h') return value * 3_600_000;
        return value * 86_400_000;
    }

    private calculateAtr(candles: Array<{ high: number; low: number; close: number }>, period: number): number {
        if (!candles.length) return 0;
        if (candles.length < 2) return candles[candles.length - 1].close * 0.01;
        const trValues: number[] = [];
        const start = Math.max(1, candles.length - (period + 1));
        for (let i = start; i < candles.length; i++) {
            const curr = candles[i];
            const prevClose = candles[i - 1].close;
            const tr = Math.max(
                curr.high - curr.low,
                Math.abs(curr.high - prevClose),
                Math.abs(curr.low - prevClose),
            );
            if (Number.isFinite(tr)) trValues.push(tr);
        }
        if (!trValues.length) return candles[candles.length - 1].close * 0.01;
        return trValues.reduce((a, b) => a + b, 0) / trValues.length;
    }

    async start() {
        console.log(`[Bot] Starting Multi-Symbol Paper Trading...`);
        console.log(`[Bot] Strategy: ${this.strategyName}`);

        // Log Mode
        // @ts-ignore
        if (this.client.getPublicMode && this.client.getPublicMode()) {
            console.warn('[Bot] RUNNING IN PUBLIC MODE (No Trading)');
        } else {
            console.log('[Bot] TRADING MODE ACTIVE (Testnet)');
            try {
                // Check Balance
                const bal = await this.client.getBalance();
                console.log('[Bot] Balance:', bal?.total?.USDT || 'Unknown', 'USDT');
            } catch (e) {
                console.error('[Bot] Failed to fetch balance:', e);
            }
        }

        while (true) {
            try {
                await this.tick();
            } catch (e) {
                console.error('[Bot] Error in loop:', e);
            }
            console.log(`[Bot] Sleeping for ${CONFIG.loopInterval / 1000}s...`);
            await new Promise(r => setTimeout(r, CONFIG.loopInterval));
        }
    }

    async tick() {
        console.log(`[Bot] Tick - ${new Date().toISOString()}`);

        // 1. Update Top Symbols
        await this.updateTopSymbols();

        const isMdeMad = this.strategyName === 'mde_mad_v2';

        if (isMdeMad) {
            // ═══ REBALANCING MODE (mde_mad_v2) ═══
            // 0. Global Sync (Close positions not in strategy or sync missing state)
            await this.syncAllPositions();

            // Continuous position adjustment based on target weight
            for (const symbol of this.state.monitoredSymbols) {
                try {
                    await this.rebalanceSymbol(symbol);
                } catch (e) {
                    console.error(`[Bot] Rebalance error ${symbol}:`, e);
                }
                await new Promise(r => setTimeout(r, 1000));
            }
        } else {
            // ═══ LEGACY MODE (ict_trend_mss_fvg) ═══
            // Binary entry/exit with SL management
            for (let i = this.state.activeTrades.length - 1; i >= 0; i--) {
                const trade = this.state.activeTrades[i];
                try {
                    const currentPrice = await this.client.getPrice(trade.symbol);
                    await this.manageTrade(trade, currentPrice);
                } catch (e) {
                    console.error(`[Bot] Error managing trade ${trade.symbol}:`, e);
                }
            }

            if (this.state.activeTrades.length < CONFIG.maxActiveTrades) {
                for (const symbol of this.state.monitoredSymbols) {
                    if (this.state.activeTrades.find(t => t.symbol === symbol)) continue;
                    if (this.state.activeTrades.length >= CONFIG.maxActiveTrades) break;
                    try {
                        await this.scanSymbol(symbol);
                    } catch (e) {
                        console.error(`[Bot] Error scanning ${symbol}:`, e);
                    }
                    await new Promise(r => setTimeout(r, 1000));
                }
            }
        }

        this.saveState();
    }

    /**
     * Global sync: fetch ALL positions from exchange.
     * 1. Close any position NOT in monitoredSymbols (unless it's an old legacy trade we want to keep, but here we want to clean).
     * 2. Sync real entry prices and quantities into state.
     */
    async syncAllPositions() {
        try {
            const positions = await this.client.getPositions();
            const openPositions = positions.filter(p => Math.abs(parseFloat(p.contracts || p.size || 0)) > 0);
            
            // Get balance to calculate weights
            const balanceInfo = await this.client.getBalance();
            const equity = balanceInfo?.total?.USDT || 1000;

            for (const pos of openPositions) {
                // Normalize symbol for comparison (e.g., CCXT format HYPE/USDT:USDT)
                const symbol = pos.symbol;
                const standardSymbol = symbol.includes(':') ? symbol : `${symbol.replace('/', '')}/USDT:USDT`;

                // Check if symbol is being monitored
                const isMonitored = this.state.monitoredSymbols.some(s => s === standardSymbol || s === symbol || standardSymbol.startsWith(s));

                if (!isMonitored) {
                    console.log(`[Sync] Found unmonitored position: ${symbol} (${pos.contracts}). Closing.`);
                    const side: 'buy' | 'sell' = pos.side === 'long' ? 'sell' : 'buy';
                    await this.client.createMarketOrder(symbol, side, Math.abs(pos.contracts), { reduceOnly: true });
                    continue;
                }

                // Sync into state if missing
                let existingTrade = this.state.activeTrades.find(t => t.symbol === symbol || t.symbol === standardSymbol);
                if (!existingTrade) {
                    console.log(`[Sync] Found missing trade in state for monitored symbol: ${symbol}. Syncing.`);
                    existingTrade = {
                        id: `sync-${Date.now()}`,
                        symbol: standardSymbol,
                        type: pos.side === 'long' ? 'LONG' : 'SHORT',
                        entryPrice: pos.entryPrice || 0,
                        stopLoss: 0,
                        originalSL: 0,
                        quantity: Math.abs(pos.contracts),
                        status: 'OPEN',
                        entryTime: Date.now(),
                        highestPrice: pos.entryPrice || 0,
                        lowestPrice: pos.entryPrice || 0
                    };
                    this.state.activeTrades.push(existingTrade);
                } else {
                    // Update existing trade with real exchange data
                    existingTrade.quantity = Math.abs(pos.contracts);
                    if (pos.entryPrice && Math.abs(existingTrade.entryPrice - pos.entryPrice) > 0.01) {
                        existingTrade.entryPrice = pos.entryPrice;
                    }
                }

                // FIX: Update the internal weight tracker to match reality
                const currentQty = existingTrade.type === 'LONG' ? existingTrade.quantity : -existingTrade.quantity;
                const currentPrice = await this.client.getPrice(standardSymbol);
                const currentWeight = (currentQty * currentPrice) / equity;
                this.recordWeight(standardSymbol, currentWeight);
            }

            // Also check for orphaned trades in state that have no position on exchange
            for (let i = this.state.activeTrades.length - 1; i >= 0; i--) {
                const trade = this.state.activeTrades[i];
                const hasPos = openPositions.some(p => p.symbol === trade.symbol || p.symbol.startsWith(trade.symbol.split('/')[0]));
                if (!hasPos) {
                    console.log(`[Sync] Removing orphaned trade from state: ${trade.symbol}`);
                    this.state.activeTrades.splice(i, 1);
                    this.recordWeight(trade.symbol, 0);
                }
            }
        } catch (e) {
            console.error(`[Sync] Error during global sync:`, e);
        }
    }

    /**
     * Rebalancing logic for mde_mad_v2 (matches EdgeCraft Python behavior).
     * On each tick: compute target weight → compare with current position → adjust.
     * No explicit stop loss — the strategy self-corrects via weight changes.
     */
    async rebalanceSymbol(symbol: string) {
        // 1. Fetch candles
        const candlesLow = await this.client.fetchCandles(symbol, CONFIG.lowTF, 300);
        if (candlesLow.length < 201) {
            if (!this.state.monitoringQueue) this.state.monitoringQueue = {};
            this.state.monitoringQueue[symbol] = { status: 'ERROR', details: 'Insufficient data', lastChecked: Date.now() };
            return;
        }

        // 2. Get strategy signal with target weight
        const currentWeight = this.getSymbolWeight(symbol);
        const analysis = this.engine.analyze([], candlesLow, undefined, currentWeight);
        let targetWeight = analysis.signal?.targetWeight ?? 0;
        const expectedReturn = analysis.signal?.expectedReturn ?? 0;
        const rawTargetWeight = targetWeight;
        targetWeight = Math.max(-MDE_REBALANCE_CONFIG.maxAbsTargetWeight, Math.min(MDE_REBALANCE_CONFIG.maxAbsTargetWeight, targetWeight));
        if (Math.abs(rawTargetWeight - targetWeight) > 1e-9) {
            console.log(`[RiskCap] ${symbol}: clamped target weight ${rawTargetWeight.toFixed(3)} -> ${targetWeight.toFixed(3)}`);
        }

        if (!this.state.monitoringQueue) this.state.monitoringQueue = {};

        // Accurate status based on real positions
        let status: string = analysis.status;
        const hasTrade = this.state.activeTrades.some(t => t.symbol === symbol);
        if (hasTrade) {
            status = 'IN_TRADE';
        } else if (status === 'ENTRY' || (status === 'REBALANCE' && Math.abs(targetWeight) > 0.005)) {
            status = 'PENDING';
        }

        this.state.monitoringQueue[symbol] = {
            status,
            details: analysis.details || `w=${targetWeight.toFixed(3)}`,
            lastChecked: Date.now()
        };

        const currentPrice = await this.client.getPrice(symbol);
        const existingTrade = this.state.activeTrades.find(t => t.symbol === symbol);

        if (existingTrade) {
            // Calculate live PnL for UI
            if (existingTrade.type === 'LONG') {
                existingTrade.pnl = (currentPrice - existingTrade.entryPrice) * existingTrade.quantity;
                existingTrade.pnlPercent = ((currentPrice - existingTrade.entryPrice) / existingTrade.entryPrice) * 100;
            } else {
                existingTrade.pnl = (existingTrade.entryPrice - currentPrice) * existingTrade.quantity;
                existingTrade.pnlPercent = ((existingTrade.entryPrice - currentPrice) / existingTrade.entryPrice) * 100;
            }

            // Update SL/TP placeholders if needed or keep at 0 for rebalancing
            existingTrade.highestPrice = Math.max(existingTrade.highestPrice || 0, currentPrice);
            existingTrade.lowestPrice = Math.min(existingTrade.lowestPrice || currentPrice, currentPrice);

            // Time-based exit (best config parity)
            const maxHoldMs = MDE_REBALANCE_CONFIG.maxHoldingBars * this.timeframeToMs(CONFIG.lowTF);
            if (maxHoldMs > 0 && (Date.now() - existingTrade.entryTime) >= maxHoldMs) {
                targetWeight = 0;
                console.log(`[TimeExit] ${symbol}: max holding reached, forcing w=0`);
            }

            // ATR stop exit (best config parity)
            if (MDE_REBALANCE_CONFIG.enableAtrStop) {
                const atr = this.calculateAtr(candlesLow, MDE_REBALANCE_CONFIG.atrPeriod);
                const stopDistance = Math.max(atr * MDE_REBALANCE_CONFIG.atrStopMultiplier, currentPrice * 0.0025);
                const atrStopHit = (
                    (existingTrade.type === 'LONG' && currentPrice <= (existingTrade.entryPrice - stopDistance)) ||
                    (existingTrade.type === 'SHORT' && currentPrice >= (existingTrade.entryPrice + stopDistance))
                );
                if (atrStopHit) {
                    targetWeight = 0;
                    console.log(`[ATRExit] ${symbol}: ATR stop triggered, forcing w=0`);
                }
            }
        }

        const isCooldownActive = this.isCooldownActive(symbol);
        if (isCooldownActive) {
            console.log(`[Cooldown] ${symbol} locked, forcing w=0`);
            targetWeight = 0;
        }

        const existingWeight = this.getSymbolWeight(symbol);
        const grossWeight = this.getGrossExposureWeight();
        const grossExcludingCurrent = Math.max(0, grossWeight - Math.abs(existingWeight));
        const maxAbsForSymbol = Math.max(0, MDE_REBALANCE_CONFIG.maxGrossExposureWeight - grossExcludingCurrent);
        const cappedByPortfolio = Math.max(-maxAbsForSymbol, Math.min(maxAbsForSymbol, targetWeight));
        if (Math.abs(cappedByPortfolio - targetWeight) > 1e-9) {
            console.log(
                `[PortfolioCap] ${symbol}: clamped target weight ${targetWeight.toFixed(3)} -> ${cappedByPortfolio.toFixed(3)} ` +
                `(gross=${grossWeight.toFixed(3)}/${MDE_REBALANCE_CONFIG.maxGrossExposureWeight.toFixed(3)})`
            );
            targetWeight = cappedByPortfolio;
        }

        if (Math.abs(targetWeight - existingWeight) <= MDE_REBALANCE_CONFIG.minRebalanceWeightDelta) {
            return;
        }

        // 3. Get current balance and position
        const balanceInfo = await this.client.getBalance();
        const equity = balanceInfo?.total?.USDT || 1000;

        // 4. Get current position from state (synced by syncAllPositions)
        let currentQty = 0;
        if (existingTrade) {
            currentQty = existingTrade.type === 'LONG' ? existingTrade.quantity : -existingTrade.quantity;
        }
        const currentNotional = currentQty * currentPrice;
        const targetNotional = equity * targetWeight;
        const targetQty = targetWeight === 0 ? 0 : targetNotional / currentPrice;

        // 5. Calculate diff
        const diffNotional = targetNotional - currentNotional;
        const diffQtySigned = targetQty - currentQty;
        const diffQty = Math.abs(diffQtySigned);
        const tradeNotional = diffQty * currentPrice;
        if (diffQty < 1e-9) {
            return;
        }

        const directionChange = existingTrade && ((targetWeight > 0 && existingTrade.type === 'SHORT') || (targetWeight < 0 && existingTrade.type === 'LONG'));
        const isReducing = existingTrade ? (Math.abs(targetQty) < (Math.abs(currentQty) - 1e-9)) : false;
        const isClosing = Math.abs(targetWeight) < MDE_REBALANCE_CONFIG.minRebalanceWeightDelta || Boolean(directionChange) || isReducing;

        if (!isClosing && tradeNotional < MDE_REBALANCE_CONFIG.minRebalanceNotional) {
            return;
        }

        const minRebalanceThreshold = equity * 0.005; // 0.5% of equity minimum change
        if (!isClosing && Math.abs(diffNotional) < minRebalanceThreshold) {
            return;
        }

        const gatePass = this.passesCostGate(tradeNotional, expectedReturn);
        if (!isClosing && !gatePass) {
            return;
        }

        console.log(`[Rebalance] ${symbol}: w=${targetWeight.toFixed(3)} | target=$${targetNotional.toFixed(2)} | current=$${currentNotional.toFixed(2)} | diff=$${diffNotional.toFixed(2)}, gate=${gatePass}`);

        // 6. Execute rebalance
        try {
            if (Math.abs(targetWeight) < MDE_REBALANCE_CONFIG.minRebalanceWeightDelta) {
                if (currentQty !== 0) {
                    const side: 'buy' | 'sell' = currentQty > 0 ? 'sell' : 'buy';
                    const closingPnl = existingTrade ? this.calculateTradePnl(existingTrade, currentPrice) : 0;
                    await this.client.createMarketOrder(symbol, side, Math.abs(currentQty), { reduceOnly: true });
                    if (existingTrade) {
                        existingTrade.status = 'CLOSED';
                        existingTrade.exitTime = Date.now();
                        existingTrade.exitPrice = currentPrice;
                        this.state.tradeHistory.unshift(existingTrade);
                        this.state.activeTrades = this.state.activeTrades.filter(t => t.id !== existingTrade.id);
                        if (closingPnl <= 0) {
                            this.triggerCooldown(symbol);
                        }
                    }
                    console.log(`[Rebalance] ${symbol}: Closed position (w→0)`);
                }
                this.recordWeight(symbol, 0);
            } else if (currentQty === 0) {
                const side: 'buy' | 'sell' = targetWeight > 0 ? 'buy' : 'sell';
                const qty = parseFloat(diffQty.toFixed(3));
                if (qty > 0) {
                    await this.client.createMarketOrder(symbol, side, qty);
                    const trade: Trade = {
                        id: `${Date.now()}-${Math.random().toString(36).substr(2, 9)}`,
                        symbol,
                        type: targetWeight > 0 ? 'LONG' : 'SHORT',
                        entryPrice: currentPrice,
                        stopLoss: 0,
                        originalSL: 0,
                        quantity: qty,
                        status: 'OPEN',
                        entryTime: Date.now(),
                        highestPrice: currentPrice,
                        lowestPrice: currentPrice,
                    };
                    this.state.activeTrades.push(trade);
                    console.log(`[Rebalance] ${symbol}: Opened ${trade.type} qty=${qty} w=${targetWeight.toFixed(3)}`);
                    this.recordWeight(symbol, targetWeight);
                }
            } else if (existingTrade) {
                if (directionChange) {
                    const qtyToClose = existingTrade.quantity;
                    const sideToClose: 'buy' | 'sell' = existingTrade.type === 'LONG' ? 'sell' : 'buy';
                    const closingPnl = this.calculateTradePnl(existingTrade, currentPrice);
                    await this.client.createMarketOrder(symbol, sideToClose, qtyToClose, { reduceOnly: true });
                    existingTrade.status = 'CLOSED';
                    existingTrade.exitTime = Date.now();
                    existingTrade.exitPrice = currentPrice;
                    this.state.tradeHistory.unshift(existingTrade);
                    this.state.activeTrades = this.state.activeTrades.filter(t => t.id !== existingTrade.id);
                    if (closingPnl <= 0) {
                        this.triggerCooldown(symbol);
                    }

                    const sideNew: 'buy' | 'sell' = targetWeight > 0 ? 'buy' : 'sell';
                    const qtyNew = parseFloat(Math.abs(targetQty).toFixed(3));
                    if (qtyNew > 0) {
                        await this.client.createMarketOrder(symbol, sideNew, qtyNew);
                        const newTrade: Trade = {
                            id: `${Date.now()}-${Math.random().toString(36).substr(2, 9)}`,
                            symbol,
                            type: targetWeight > 0 ? 'LONG' : 'SHORT',
                            entryPrice: currentPrice,
                            stopLoss: 0,
                            originalSL: 0,
                            quantity: qtyNew,
                            status: 'OPEN',
                            entryTime: Date.now(),
                            highestPrice: currentPrice,
                            lowestPrice: currentPrice,
                        };
                        this.state.activeTrades.push(newTrade);
                        console.log(`[Rebalance] ${symbol}: Flipped to ${newTrade.type} qty=${qtyNew}`);
                        this.recordWeight(symbol, targetWeight);
                    }
                } else {
                    const isIncrease = Math.abs(targetQty) > Math.abs(currentQty);
                    const side: 'buy' | 'sell' = diffQtySigned > 0 ? 'buy' : 'sell';
                    const qtyRaw = isIncrease ? diffQty : Math.min(diffQty, existingTrade.quantity);
                    const qty = parseFloat(qtyRaw.toFixed(3));
                    if (qty > 0) {
                        if (isIncrease) {
                            await this.client.createMarketOrder(symbol, side, qty);
                            const totalNotional = (existingTrade.entryPrice * existingTrade.quantity) + (currentPrice * qty);
                            existingTrade.quantity += qty;
                            existingTrade.entryPrice = totalNotional / existingTrade.quantity;
                            console.log(`[Rebalance] ${symbol}: Increased by ${qty} → total=${existingTrade.quantity.toFixed(3)}`);
                        } else {
                            await this.client.createMarketOrder(symbol, side, qty, { reduceOnly: true });
                            existingTrade.quantity -= qty;
                            if (existingTrade.quantity <= 0.001) {
                                const closingPnl = this.calculateTradePnl(existingTrade, currentPrice);
                                existingTrade.status = 'CLOSED';
                                existingTrade.exitTime = Date.now();
                                existingTrade.exitPrice = currentPrice;
                                this.state.tradeHistory.unshift(existingTrade);
                                this.state.activeTrades = this.state.activeTrades.filter(t => t.id !== existingTrade.id);
                                if (closingPnl <= 0) {
                                    this.triggerCooldown(symbol);
                                }
                            }
                            console.log(`[Rebalance] ${symbol}: Decreased by ${qty} → remaining=${existingTrade.quantity.toFixed(3)}`);
                        }
                        this.recordWeight(symbol, targetWeight);
                    }
                }
            }
        } catch (e) {
            console.error(`[Rebalance] Order failed for ${symbol}:`, e);
        }
    }

    async updateTopSymbols() {
        const ONE_HOUR = 3600000;
        const now = Date.now();
        // @ts-ignore
        if (this.state.lastTopSymbolsUpdate && (now - this.state.lastTopSymbolsUpdate) < ONE_HOUR && this.state.monitoredSymbols.length > 0) {
            return;
        }

        try {
            console.log("[Bot] Fetching Top Coins by Market Cap from CoinGecko...");
            // ...
            if (matchedSymbols.length > 0) {
                this.state.monitoredSymbols = matchedSymbols;
                // @ts-ignore
                this.state.lastTopSymbolsUpdate = now;
                console.log(`[Bot] Top 20 by Market Cap (on Binance): ${this.state.monitoredSymbols.join(', ')}`);
            } else {
                console.warn("[Bot] No matching symbols found. Keeping previous list.");
            }

        } catch (e) {
            console.error("[Bot] Failed to update top symbols:", e);
        }
    }

    async scanSymbol(symbol: string) {
        // console.log(`[Bot] Scanning ${symbol}...`);

        // 1. Fetch Data
        const candlesHigh = this.strategyName === 'ict_trend_mss_fvg'
            ? await this.client.fetchCandles(symbol, CONFIG.highTF, 100)
            : [];

        // We fetch 300 to ensure we securely cover MDE_MAD_V2 201+ requirements
        const candlesLow = await this.client.fetchCandles(symbol, CONFIG.lowTF, 300);

        const isIct = this.strategyName === 'ict_trend_mss_fvg';
        if (!candlesLow.length || (isIct && !candlesHigh.length)) {
            if (!this.state.monitoringQueue) this.state.monitoringQueue = {};
            this.state.monitoringQueue[symbol] = { status: 'ERROR', details: 'Insufficient Data', lastChecked: Date.now() };
            return;
        }

        const currentPrice = await this.client.getPrice(symbol);

        // 2. Scan for Entry
        const analysis = this.engine.analyze(candlesHigh, candlesLow, undefined, 0);

        // Update Monitoring Queue with status
        if (!this.state.monitoringQueue) this.state.monitoringQueue = {};
        this.state.monitoringQueue[symbol] = {
            status: analysis.status,
            details: analysis.details || 'No setup detected',
            lastChecked: Date.now()
        };

        if (analysis.status !== 'ENTRY' || !analysis.signal) return;

        const signal = analysis.signal;
        console.log(`[Bot] SIGNAL DETECTED on ${symbol}: ${signal.type} @ ${signal.entryPrice}`);

        // Calculate Position Size
        const balanceInfo = await this.client.getBalance();
        const balance = balanceInfo?.total?.USDT || 1000;
        const riskAmount = balance * CONFIG.riskPercent;
        const riskPerUnit = Math.abs(signal.entryPrice - signal.stopLoss);

        if (riskPerUnit === 0) return;

        const quantity = parseFloat((riskAmount / riskPerUnit).toFixed(3)); // Crude rounding

        if (quantity <= 0) return;

        // Check distance
        const distInfo = Math.abs(currentPrice - signal.entryPrice) / signal.entryPrice;

        if (distInfo < 0.005) { // 0.5% tolerance
            await this.executeEntry(symbol, signal, quantity, currentPrice);
            // Clear from monitoring queue if entered? Or keep as entered?
            // Maybe update status to 'IN_TRADE'
            this.state.monitoringQueue[symbol].status = 'IN_TRADE';
            this.state.monitoringQueue[symbol].details = 'Trade Executed';
        }
    }

    async executeEntry(symbol: string, signal: StrategySignal, quantity: number, price: number) {
        try {
            console.log(`[Bot] Executing ${signal.type} on ${symbol} Qty: ${quantity}`);
            // Market Order
            const orderSide: 'buy' | 'sell' = signal.type === 'LONG' ? 'buy' : 'sell';
            await this.client.createMarketOrder(symbol, orderSide, quantity);

            const trade: Trade = {
                id: `${Date.now()}-${Math.random().toString(36).substr(2, 9)}`,
                symbol: symbol,
                type: signal.type,
                entryPrice: price,
                stopLoss: signal.stopLoss,
                originalSL: signal.stopLoss,
                quantity: quantity,
                status: 'OPEN',
                entryTime: Date.now(),
                highestPrice: price,
                lowestPrice: price
            };

            this.state.activeTrades.push(trade);
            console.log(`[Bot] Trade Opened: ${trade.id}`);

        } catch (e) {
            console.error(`[Bot] Entry failed for ${symbol}:`, e);
        }
    }

    async manageTrade(trade: Trade, currentPrice: number) {
        const riskAmt = Math.abs(trade.entryPrice - trade.originalSL);

        // Check SL Hit
        if (trade.type === 'LONG') {
            if (currentPrice <= trade.stopLoss) {
                await this.closeTrade(trade, 'SL Hit', currentPrice);
                return;
            }
            if (currentPrice > trade.highestPrice) trade.highestPrice = currentPrice;

            // Break Even
            if (trade.highestPrice >= trade.entryPrice + riskAmt) {
                if (trade.stopLoss < trade.entryPrice) {
                    trade.stopLoss = trade.entryPrice * 1.002; // Small profit buffer
                    console.log(`[Bot] ${trade.symbol} SL moved to BE`);
                }
            }
            // Trailing
            if (trade.highestPrice > trade.entryPrice + (2 * riskAmt)) {
                const newSL = trade.highestPrice - (1.5 * riskAmt);
                if (newSL > trade.stopLoss) {
                    trade.stopLoss = newSL;
                    console.log(`[Bot] ${trade.symbol} Trailing SL updated`);
                }
            }

        } else {
            // SHORT
            if (currentPrice >= trade.stopLoss) {
                await this.closeTrade(trade, 'SL Hit', currentPrice);
                return;
            }
            if (currentPrice < trade.lowestPrice) trade.lowestPrice = currentPrice;

            // Break Even
            if (trade.lowestPrice <= trade.entryPrice - riskAmt) {
                if (trade.stopLoss > trade.entryPrice) {
                    trade.stopLoss = trade.entryPrice * 0.998;
                    console.log(`[Bot] ${trade.symbol} SL moved to BE`);
                }
            }
            // Trailing
            if (trade.lowestPrice < trade.entryPrice - (2 * riskAmt)) {
                const newSL = trade.lowestPrice + (1.5 * riskAmt);
                if (newSL < trade.stopLoss) {
                    trade.stopLoss = newSL;
                    console.log(`[Bot] ${trade.symbol} Trailing SL updated`);
                }
            }
        }
    }

    async closeTrade(trade: Trade, reason: string, price: number) {
        console.log(`[Bot] Closing ${trade.symbol}: ${reason}`);
        try {
            // Calculate PnL (Estimated before Order)
            let estimatedPnL = 0;
            if (trade.type === 'LONG') {
                estimatedPnL = (price - trade.entryPrice) * trade.quantity;
            } else {
                estimatedPnL = (trade.entryPrice - price) * trade.quantity;
            }

            await this.client.createMarketOrder(trade.symbol, trade.type === 'LONG' ? 'sell' : 'buy', trade.quantity);

            trade.status = 'CLOSED';
            trade.exitTime = Date.now();
            trade.exitPrice = price;
            trade.closeReason = reason;

            // Finalize PnL
            if (trade.type === 'LONG') {
                trade.pnl = (price - trade.entryPrice) * trade.quantity;
                trade.pnlPercent = ((price - trade.entryPrice) / trade.entryPrice) * 100;
            } else {
                trade.pnl = (trade.entryPrice - price) * trade.quantity;
                trade.pnlPercent = ((trade.entryPrice - price) / trade.entryPrice) * 100;
            }

            // Move to history
            this.state.tradeHistory.unshift(trade);
            // Remove from active trades
            this.state.activeTrades = this.state.activeTrades.filter(t => t.id !== trade.id);

            console.log(`[Bot] Trade Closed. PnL: $${trade.pnl?.toFixed(2)} (${trade.pnlPercent?.toFixed(2)}%)`);

        } catch (e) {
            console.error(`[Bot] Close Failed for ${trade.symbol}:`, e);
        }
    }
}

try {
    console.log("Instantiating BotRunner...");
    const bot = new BotRunner();
    console.log("Starting BotRunner...");
    bot.start();
} catch (e) {
    console.error("FATAL ERROR IN BOT STARTUP:", e);
}
