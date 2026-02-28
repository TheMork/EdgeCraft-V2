import { FVGDetector } from './fvgStrategy.calculator';
import { MarketStructureDetector } from './marketStructure';
import type { KLineData } from '../types';

export interface StrategySignal {
    type: 'LONG' | 'SHORT';
    entryPrice: number;
    stopLoss: number;
    takeProfit: number; // Set to Infinity for Trailing
    risk: number;       // Absolute price difference
    reason: string;
    timestamp: number;
    targetWeight?: number;
}

export type StrategyStatus = 'NO_SIGNAL' | 'WATCHING' | 'ENTRY' | 'REBALANCE';

export interface StrategyResult {
    status: StrategyStatus;
    signal?: StrategySignal;
    details?: string;
}

export type BotStrategyName = 'ict_trend_mss_fvg' | 'mde_mad_v2';

interface MdeMadV2Config {
    lookbackBars: number;
    trendFilterPeriod: number;
    riskAversion: number;
    entropyWeight: number;
    turnoverPenalty: number;
    maxLeverage: number;
    enableShorts: boolean;
    optimizationSteps: number;
    atrPeriod: number;
    atrStopMultiplier: number;
    entryWeightThreshold: number;
    trendFilterType: 'EMA' | 'HMA' | 'KAMA';
}

const DEFAULT_MDE_MAD_V2_CONFIG: MdeMadV2Config = {
    lookbackBars: 30,
    trendFilterPeriod: 200,
    riskAversion: 2.0,
    entropyWeight: 0.1,
    turnoverPenalty: 0.05,
    maxLeverage: 10,
    enableShorts: false, // Changed to false
    optimizationSteps: 41,
    atrPeriod: 14,
    atrStopMultiplier: 2.0,
    entryWeightThreshold: 0.10,
    trendFilterType: 'EMA',
};

export class StrategyEngine {
    private static readonly EPS = 1e-9;
    private fvgDetector: FVGDetector;
    private msDetector: MarketStructureDetector;
    private strategyName: BotStrategyName;
    private mdeConfig: MdeMadV2Config;
    private currentWeight: number;

    constructor(strategyName: BotStrategyName = 'ict_trend_mss_fvg', mdeOverrides: Partial<MdeMadV2Config> = {}) {
        this.fvgDetector = new FVGDetector();
        this.msDetector = new MarketStructureDetector();
        this.strategyName = strategyName;
        this.mdeConfig = { ...DEFAULT_MDE_MAD_V2_CONFIG, ...mdeOverrides };
        this.currentWeight = 0;
    }

    getStrategyName(): BotStrategyName {
        return this.strategyName;
    }

    /**
     * Strategy Router
     * @param candlesHigh - Daily Candles (Context)
     * @param candlesLow - 4h Candles (Entry)
     */
    analyze(candlesHigh: KLineData[], candlesLow: KLineData[], trendValue?: number): StrategyResult {
        if (this.strategyName === 'mde_mad_v2') {
            return this.analyzeMdeMadV2(candlesLow, trendValue);
        }
        return this.analyzeIctTrendMssFvg(candlesHigh, candlesLow);
    }

    private analyzeIctTrendMssFvg(candlesHigh: KLineData[], candlesLow: KLineData[]): StrategyResult {
        // Ensure enough data
        if (candlesHigh.length < 50 || candlesLow.length < 200) {
            return { status: 'NO_SIGNAL', details: 'Insufficient Data' };
        }

        const currentCandle = candlesLow[candlesLow.length - 1]; // The "Live" candle
        const currentTime = currentCandle.timestamp;

        // 1. Determine Trend (EMA 50 on Daily)
        const emaHigh = this.calculateEMA(candlesHigh, 50);
        const trendIndex = candlesHigh.length - 2; // Completed Daily candle

        if (trendIndex < 0) return { status: 'NO_SIGNAL', details: 'Insufficient Daily Data' };

        const trendCandle = candlesHigh[trendIndex];
        const trendEMA = emaHigh[trendIndex];

        const htfTrend = trendCandle.close > trendEMA ? 'BULLISH' : 'BEARISH';

        // 2. Detect MSS on 4h
        const lookbackMSS = 50;
        const historySlice = candlesLow.slice(-lookbackMSS); // Last 50 candles
        const swings = this.msDetector.detectSwingPoints(historySlice);
        const recentCandles = candlesLow.slice(-20);
        const recentMax = Math.max(...recentCandles.map(c => c.high));
        const recentMin = Math.min(...recentCandles.map(c => c.low));

        let hasMSS = false;
        if (htfTrend === 'BULLISH') {
            const relevantHighs = swings.filter((s: any) => s.type === 'HIGH' && s.index < (historySlice.length - 10));
            if (relevantHighs.length > 0) {
                const lastHigh = relevantHighs[relevantHighs.length - 1];
                if (recentMax > lastHigh.price) {
                    hasMSS = true;
                }
            }
        } else {
            const relevantLows = swings.filter((s: any) => s.type === 'LOW' && s.index < (historySlice.length - 10));
            if (relevantLows.length > 0) {
                const lastLow = relevantLows[relevantLows.length - 1];
                if (recentMin < lastLow.price) {
                    hasMSS = true;
                }
            }
        }

        if (!hasMSS) {
            return { status: 'NO_SIGNAL', details: `Trend ${htfTrend} but waiting for MSS` };
        }

        // 3. Detect FVG for Entry (in recent candles)
        const recentSlice = candlesLow.slice(-10);
        const fvgs = this.fvgDetector.detectFVGs(recentSlice);

        // Find FVG that matches Trend
        // Bullish Trend -> Bullish FVG (Support)
        const validFvgs = fvgs.filter((f: any) => !f.filled && (
            (htfTrend === 'BULLISH' && f.type === 'BULLISH') ||
            (htfTrend === 'BEARISH' && f.type === 'BEARISH')
        ));

        if (validFvgs.length === 0) {
            return { status: 'WATCHING', details: `Trend ${htfTrend}, MSS Confirmed. Waiting for FVG.` };
        }

        const signalFVG = validFvgs[validFvgs.length - 1]; // Use most recent

        // 4. Check if Price is in Entry Zone
        const currentPrice = currentCandle.close; // Or current real-time price
        let entryPrice = 0;
        let sl = 0;
        let shouldEnter = false;
        let watchingDetails = '';

        if (htfTrend === 'BULLISH') {
            // Buy Condition: Price retraced into FVG or below open of FVG?
            // Usually we buy at the top of the FVG as it retraces down.
            // If current price is <= FVG Top
            if (currentPrice <= signalFVG.top) {
                shouldEnter = true;
                entryPrice = signalFVG.top; // We limit at Top of FVG
                sl = signalFVG.bottom - (currentPrice * 0.005); // 0.5% below FVG
            } else {
                watchingDetails = `Bullish Setup: Waiting for retrace to ${signalFVG.top}`;
            }
        } else {
            // Sell Condition: Price retraced up into FVG
            // If current price >= FVG Bottom
            if (currentPrice >= signalFVG.bottom) {
                shouldEnter = true;
                entryPrice = signalFVG.bottom;
                sl = signalFVG.top + (currentPrice * 0.005);
            } else {
                watchingDetails = `Bearish Setup: Waiting for retrace to ${signalFVG.bottom}`;
            }
        }

        if (shouldEnter) {
            const risk = Math.abs(entryPrice - sl);
            return {
                status: 'ENTRY',
                signal: {
                    type: htfTrend === 'BULLISH' ? 'LONG' : 'SHORT',
                    entryPrice: entryPrice,
                    stopLoss: sl,
                    takeProfit: 0, // Infinity / Trailing
                    risk: risk,
                    reason: `Trend ${htfTrend} + MSS + FVG`,
                    timestamp: currentTime
                },
                details: 'Entry Triggered'
            };
        } else {
            return { status: 'WATCHING', details: watchingDetails };
        }
    }

    private analyzeMdeMadV2(candlesLow: KLineData[], trendValueOverride?: number): StrategyResult {
        const cfg = this.mdeConfig;
        const minRequired = Math.max(cfg.lookbackBars, cfg.trendFilterPeriod) + 1;
        if (candlesLow.length < minRequired) {
            return { status: 'NO_SIGNAL', details: `MDE v2 waiting for ${minRequired} candles` };
        }

        // Optimization: only process what's needed for lookback/trend
        const relevantCount = Math.max(cfg.lookbackBars + 2, trendValueOverride !== undefined ? 0 : cfg.trendFilterPeriod + 1);
        const recentCandles = candlesLow.slice(-relevantCount);
        const closes = recentCandles.map(c => c.close).filter((v) => Number.isFinite(v) && v > 0);

        if (closes.length < 5) {
            return { status: 'NO_SIGNAL', details: 'MDE v2 invalid close series' };
        }

        const close = closes[closes.length - 1];

        // Calculate Trend Line based on type OR use override
        let trendValue = trendValueOverride;
        if (trendValue === undefined) {
            if (cfg.trendFilterType === 'HMA') {
                const hmaSeries = this.calculateHMASeries(closes, cfg.trendFilterPeriod);
                trendValue = hmaSeries[hmaSeries.length - 1];
            } else if (cfg.trendFilterType === 'KAMA') {
                const kamaSeries = this.calculateKAMASeries(closes, cfg.trendFilterPeriod);
                trendValue = kamaSeries[kamaSeries.length - 1];
            } else {
                // Default to EMA
                trendValue = this.calculateEmaValue(closes, cfg.trendFilterPeriod);
            }
        }

        const isUptrend = close >= trendValue;
        const priceWindow = closes.slice(-(cfg.lookbackBars + 1));
        const returns = this.calculateLogReturns(priceWindow);

        if (returns.length < 5) {
            return { status: 'NO_SIGNAL', details: 'MDE v2 waiting for return window' };
        }

        const targetWeight = this.optimizeMdeWeight(returns, isUptrend);
        const absWeight = Math.abs(targetWeight);
        if (absWeight < cfg.entryWeightThreshold) {
            return {
                status: 'REBALANCE',
                signal: {
                    type: targetWeight >= 0 ? 'LONG' : 'SHORT',
                    entryPrice: close,
                    stopLoss: 0,
                    takeProfit: 0,
                    risk: 0,
                    reason: `MDE-MAD v2 rebalance w=${targetWeight.toFixed(3)}`,
                    timestamp: candlesLow[candlesLow.length - 1].timestamp,
                    targetWeight,
