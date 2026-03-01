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
    expectedReturn?: number;
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
    wyckoffPositionCap: number;
    useWyckoffFilter: boolean;
    wyckoffMode: 'full' | 'range_only' | 'range_switch' | 'regime_switch' | 'off' | 'none';
    wyckoffWindow: number;
    wyckoffRangeMaxWidth: number;
    wyckoffMomentumThresh: number;
    wyckoffBreakoutAtr: number;
    wyckoffVolumeSpike: number;
    wyckoffWickRatio: number;
    wyckoffBiasWeight: number;
    useWyckoffV2: boolean;
    wyckoffV2AdxPeriod: number;
    wyckoffV2AdxMax: number;
    wyckoffV2ChopPeriod: number;
    wyckoffV2ChopMin: number;
    wyckoffV2EmaPeriod: number;
    wyckoffV2EmaSlopeMax: number;
    wyckoffV2RangeSignalsMin: number;
    wyckoffV2LongVolumeSpike: number;
    wyckoffV2ShortVolumeSpike: number;
    wyckoffV2LongWickRatio: number;
    wyckoffV2ShortWickRatio: number;
    wyckoffV2SpringScoreMin: number;
    wyckoffV2UpthrustScoreMin: number;
    wyckoffV2LongBiasWeight: number;
    wyckoffV2ShortBiasWeight: number;
    wyckoffV2PersistBars: number;
    wyckoffV2RangeDampen: number;
}

interface WyckoffState {
    long: number;
    short: number;
}

const DEFAULT_MDE_MAD_V2_CONFIG: MdeMadV2Config = {
    lookbackBars: 80,
    trendFilterPeriod: 0,
    riskAversion: 4.5,
    entropyWeight: 0.08,
    turnoverPenalty: 0.09,
    maxLeverage: 3,
    enableShorts: true, // Enabled
    optimizationSteps: 41,
    atrPeriod: 14,
    atrStopMultiplier: 3.0,
    entryWeightThreshold: 0.10,
    trendFilterType: 'EMA',
    wyckoffPositionCap: 0.10,
    useWyckoffFilter: true,
    wyckoffMode: 'range_only',
    wyckoffWindow: 32,
    wyckoffRangeMaxWidth: 0.13,
    wyckoffMomentumThresh: 0.07,
    wyckoffBreakoutAtr: 0.22,
    wyckoffVolumeSpike: 1.60,
    wyckoffWickRatio: 1.20,
    wyckoffBiasWeight: 0.35,
    useWyckoffV2: true,
    wyckoffV2AdxPeriod: 14,
    wyckoffV2AdxMax: 25.0,
    wyckoffV2ChopPeriod: 14,
    wyckoffV2ChopMin: 50.0,
    wyckoffV2EmaPeriod: 20,
    wyckoffV2EmaSlopeMax: 0.005,
    wyckoffV2RangeSignalsMin: 2,
    wyckoffV2LongVolumeSpike: 1.35,
    wyckoffV2ShortVolumeSpike: 1.40,
    wyckoffV2LongWickRatio: 1.00,
    wyckoffV2ShortWickRatio: 1.05,
    wyckoffV2SpringScoreMin: 2.0,
    wyckoffV2UpthrustScoreMin: 2.0,
    wyckoffV2LongBiasWeight: 0.45,
    wyckoffV2ShortBiasWeight: 0.38,
    wyckoffV2PersistBars: 4,
    wyckoffV2RangeDampen: 0.94,
};

export class StrategyEngine {
    private static readonly EPS = 1e-9;
    private fvgDetector: FVGDetector;
    private msDetector: MarketStructureDetector;
    private strategyName: BotStrategyName;
    private mdeConfig: MdeMadV2Config;
    private wyckoffStateBySymbol: Record<string, WyckoffState>;

    constructor(strategyName: BotStrategyName = 'ict_trend_mss_fvg', mdeOverrides: Partial<MdeMadV2Config> = {}) {
        this.fvgDetector = new FVGDetector();
        this.msDetector = new MarketStructureDetector();
        this.strategyName = strategyName;
        this.mdeConfig = { ...DEFAULT_MDE_MAD_V2_CONFIG, ...mdeOverrides };
        this.wyckoffStateBySymbol = {};
    }

    getStrategyName(): BotStrategyName {
        return this.strategyName;
    }

    /**
     * Strategy Router
     * @param candlesHigh - Daily Candles (Context)
     * @param candlesLow - 4h Candles (Entry)
     * @param currentWeight - Optional current weight of the symbol to prevent shared state issues
     */
    analyze(candlesHigh: KLineData[], candlesLow: KLineData[], trendValue?: number, currentWeight: number = 0, symbolKey: string = '__default__'): StrategyResult {
        if (this.strategyName === 'mde_mad_v2') {
            return this.analyzeMdeMadV2(candlesLow, trendValue, currentWeight, symbolKey);
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

    private analyzeMdeMadV2(candlesLow: KLineData[], trendValueOverride?: number, currentWeight: number = 0, symbolKey: string = '__default__'): StrategyResult {
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

        const priceWindow = closes.slice(-(cfg.lookbackBars + 1));
        const returns = this.calculateLogReturns(priceWindow);

        if (returns.length < 5) {
            return { status: 'NO_SIGNAL', details: 'MDE v2 waiting for return window' };
        }

        const expectedReturn = returns.reduce((a, b) => a + b) / returns.length;
        let isUptrend = expectedReturn >= 0;

        // Optional trend filter. Period <= 0 disables the filter.
        if (cfg.trendFilterPeriod > 0) {
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
            isUptrend = close >= (trendValue ?? close);
        }

        const baseTargetWeight = this.optimizeMdeWeight(returns, isUptrend, currentWeight);
        const adjusted = this.applyWyckoffRegime(baseTargetWeight, candlesLow, symbolKey);
        const targetWeight = adjusted.targetWeight;
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
                    reason: `MDE-MAD v2 rebalance w=${targetWeight.toFixed(3)} phase=${adjusted.phaseLabel}`,
                    timestamp: candlesLow[candlesLow.length - 1].timestamp,
                    targetWeight,
                    expectedReturn,
                },
                details: `MDE v2 low conviction (w=${targetWeight.toFixed(3)}, phase=${adjusted.phaseLabel})`,
            };
        }

        const atr = this.calculateAtr(candlesLow, cfg.atrPeriod);
        const stopDistance = Math.max(atr * cfg.atrStopMultiplier, close * 0.0025);

        const isLong = targetWeight > 0;
        const stopLoss = isLong ? close - stopDistance : close + stopDistance;
        const risk = Math.abs(close - stopLoss);
        if (!Number.isFinite(risk) || risk <= StrategyEngine.EPS) {
            return { status: 'NO_SIGNAL', details: 'MDE v2 invalid risk model' };
        }

        const signal: StrategySignal = {
            type: isLong ? 'LONG' : 'SHORT',
            entryPrice: close,
            stopLoss,
            takeProfit: 0,
            risk,
            reason: `MDE-MAD v2 weight=${targetWeight.toFixed(3)} trend=${isUptrend ? 'UP' : 'DOWN'} phase=${adjusted.phaseLabel}`,
            timestamp: candlesLow[candlesLow.length - 1].timestamp,
            targetWeight,
            expectedReturn,
        };

        return {
            status: 'ENTRY',
            signal,
            details: 'MDE v2 entry triggered',
        };
    }

    private applyWyckoffRegime(baseTarget: number, candlesLow: KLineData[], symbolKey: string): { targetWeight: number; phaseLabel: string } {
        const cfg = this.mdeConfig;
        if (!cfg.useWyckoffFilter) return { targetWeight: baseTarget, phaseLabel: 'off' };

        const key = symbolKey || '__default__';
        const prior = this.wyckoffStateBySymbol[key] || { long: 0, short: 0 };
        let state: WyckoffState = { long: prior.long, short: prior.short };

        const window = Math.max(20, Math.floor(cfg.wyckoffWindow));
        const minNeeded = Math.max(
            window + 2,
            cfg.atrPeriod + 2,
            cfg.wyckoffV2AdxPeriod + 2,
            cfg.wyckoffV2ChopPeriod + 2,
            cfg.wyckoffV2EmaPeriod + 2,
        );
        if (candlesLow.length < minNeeded) {
            this.wyckoffStateBySymbol[key] = state;
            return { targetWeight: baseTarget, phaseLabel: 'init' };
        }

        const prevCandles = candlesLow.slice(-(window + 1), -1);
        const last = candlesLow[candlesLow.length - 1];
        if (prevCandles.length < window) {
            this.wyckoffStateBySymbol[key] = state;
            return { targetWeight: baseTarget, phaseLabel: 'init' };
        }

        const chHigh = Math.max(...prevCandles.map((c) => c.high));
        const chLow = Math.min(...prevCandles.map((c) => c.low));
        const chMid = 0.5 * (chHigh + chLow);
        const lastPx = last.close;
        const width = (chHigh - chLow) / Math.max(1e-9, lastPx);
        const firstPrevClose = prevCandles[0]?.close ?? lastPx;
        const mom = (lastPx / Math.max(1e-9, firstPrevClose)) - 1.0;
        const avgVol = prevCandles.reduce((a, b) => a + b.volume, 0) / Math.max(1, prevCandles.length);

        const atr = this.calculateAtr(candlesLow, cfg.atrPeriod);
        const breakout = Math.max(cfg.wyckoffBreakoutAtr * atr, lastPx * 0.001);
        const body = Math.abs(last.close - last.open);
        const upperWick = last.high - Math.max(last.close, last.open);
        const lowerWick = Math.min(last.close, last.open) - last.low;
        const wickRef = cfg.wyckoffWickRatio * (body + 1e-12);
        const volSpike = last.volume >= (cfg.wyckoffVolumeSpike * Math.max(1e-9, avgVol));

        const spring = (
            (last.low < (chLow - breakout)) &&
            (last.close > chLow) &&
            volSpike &&
            (lowerWick > wickRef)
        );
        const upthrust = (
            (last.high > (chHigh + breakout)) &&
            (last.close < chHigh) &&
            volSpike &&
            (upperWick > wickRef)
        );

        const isRange = (width <= cfg.wyckoffRangeMaxWidth) && (Math.abs(mom) <= cfg.wyckoffMomentumThresh);
        const isMarkup = (mom > cfg.wyckoffMomentumThresh) && (lastPx >= chMid);
        const isMarkdown = (mom < -cfg.wyckoffMomentumThresh) && (lastPx <= chMid);

        const mode = String(cfg.wyckoffMode || 'full').trim().toLowerCase();

        if (cfg.useWyckoffV2) {
            const adx = this.computeAdx(candlesLow, cfg.wyckoffV2AdxPeriod);
            const chop = this.computeChoppiness(candlesLow, cfg.wyckoffV2ChopPeriod);
            const emaSlope = this.computeEmaSlope(candlesLow, cfg.wyckoffV2EmaPeriod);

            let rangeSignalCount = 0;
            if (isRange) rangeSignalCount += 1;
            if (adx !== null && adx <= cfg.wyckoffV2AdxMax) rangeSignalCount += 1;
            if (chop !== null && chop >= cfg.wyckoffV2ChopMin) rangeSignalCount += 1;
            if (emaSlope !== null && Math.abs(emaSlope) <= cfg.wyckoffV2EmaSlopeMax) rangeSignalCount += 1;

            const minSignals = Math.min(4, Math.max(1, Math.floor(cfg.wyckoffV2RangeSignalsMin)));
            const isRangeV2 = rangeSignalCount >= minSignals;

            const volRatio = last.volume / Math.max(1e-9, avgVol);
            const reclaim = (last.low < (chLow - breakout)) && (last.close > chLow);
            const reject = (last.high > (chHigh + breakout)) && (last.close < chHigh);
            const springWickOk = lowerWick > (cfg.wyckoffV2LongWickRatio * (body + 1e-12));
            const upthrustWickOk = upperWick > (cfg.wyckoffV2ShortWickRatio * (body + 1e-12));

            const springScore =
                Number(reclaim) +
                Number(springWickOk) +
                Number(volRatio >= cfg.wyckoffV2LongVolumeSpike) +
                Number(last.close >= chMid);
            const upthrustScore =
                Number(reject) +
                Number(upthrustWickOk) +
                Number(volRatio >= cfg.wyckoffV2ShortVolumeSpike) +
                Number(last.close <= chMid);

            const springV2 = isRangeV2 && (springScore >= cfg.wyckoffV2SpringScoreMin);
            const upthrustV2 = isRangeV2 && (upthrustScore >= cfg.wyckoffV2UpthrustScoreMin);

            const persistBars = Math.max(0, Math.floor(cfg.wyckoffV2PersistBars));
            let longPersist = Math.max(0, state.long - 1);
            let shortPersist = Math.max(0, state.short - 1);
            if (springV2) longPersist = Math.max(longPersist, persistBars);
            if (upthrustV2) shortPersist = Math.max(shortPersist, persistBars);
            state = { long: longPersist, short: shortPersist };
            this.wyckoffStateBySymbol[key] = state;

            const longBias = Math.max(0, Math.min(1, cfg.wyckoffV2LongBiasWeight));
            const shortBias = Math.max(0, Math.min(1, cfg.wyckoffV2ShortBiasWeight));
            const dampen = Math.max(0.5, Math.min(1.0, cfg.wyckoffV2RangeDampen));

            let adjusted = baseTarget;
            let phaseLabel = 'neutral_v2';

            if (isRangeV2) {
                phaseLabel = 'range_v2';
                adjusted *= dampen;
                if (springV2) {
                    phaseLabel = 'range_v2_spring';
                    adjusted = Math.max(adjusted, Math.max(0, cfg.wyckoffPositionCap * longBias));
                } else if (upthrustV2) {
                    phaseLabel = 'range_v2_upthrust';
                    adjusted = Math.min(adjusted, -Math.max(0, cfg.wyckoffPositionCap * shortBias));
                } else if (longPersist > 0 && shortPersist === 0) {
                    phaseLabel = 'range_v2_persist_long';
                    adjusted = Math.max(adjusted, Math.max(0, cfg.wyckoffPositionCap * (0.65 * longBias)));
                } else if (shortPersist > 0 && longPersist === 0) {
                    phaseLabel = 'range_v2_persist_short';
                    adjusted = Math.min(adjusted, -Math.max(0, cfg.wyckoffPositionCap * (0.65 * shortBias)));
                }
            } else if (isMarkup) {
                phaseLabel = 'markup_v2';
                if (adjusted < 0) adjusted *= (1.0 - shortBias);
            } else if (isMarkdown) {
                phaseLabel = 'markdown_v2';
                if (adjusted > 0) adjusted *= (1.0 - longBias);
            } else {
                phaseLabel = 'transition_v2';
            }

            if (mode === 'range_only' || mode === 'range_switch' || mode === 'regime_switch') {
                if (phaseLabel !== 'range_v2' && phaseLabel !== 'range_v2_spring' && phaseLabel !== 'range_v2_upthrust') {
                    adjusted = baseTarget;
                }
            } else if (mode === 'off' || mode === 'none') {
                adjusted = baseTarget;
            }

            if (!cfg.enableShorts && adjusted < 0) adjusted = 0;
            return { targetWeight: adjusted, phaseLabel };
        }

        const bias = Math.max(0, Math.min(1, cfg.wyckoffBiasWeight));
        let adjusted = baseTarget;
        let phaseLabel = 'neutral';
        if (isRange) {
            phaseLabel = 'range';
            if (spring) {
                phaseLabel = 'range_spring';
                adjusted = Math.max(adjusted, Math.max(0, cfg.wyckoffPositionCap * bias));
            } else if (upthrust) {
                phaseLabel = 'range_upthrust';
                adjusted = Math.min(adjusted, -Math.max(0, cfg.wyckoffPositionCap * bias));
            } else {
                adjusted *= 0.85;
            }
        } else if (isMarkup) {
            phaseLabel = 'markup';
            if (adjusted < 0) adjusted *= (1.0 - bias);
        } else if (isMarkdown) {
            phaseLabel = 'markdown';
            if (adjusted > 0) adjusted *= (1.0 - bias);
        }

        if (mode === 'range_only' || mode === 'range_switch' || mode === 'regime_switch') {
            if (phaseLabel !== 'range' && phaseLabel !== 'range_spring' && phaseLabel !== 'range_upthrust') {
                adjusted = baseTarget;
            }
        } else if (mode === 'off' || mode === 'none') {
            adjusted = baseTarget;
        }

        if (!cfg.enableShorts && adjusted < 0) adjusted = 0;
        this.wyckoffStateBySymbol[key] = state;
        return { targetWeight: adjusted, phaseLabel };
    }

    private computeAdx(candles: KLineData[], period: number): number | null {
        const p = Math.max(2, Math.floor(period));
        if (candles.length < p + 2) return null;
        const recent = candles.slice(-(p + 2));
        const tr: number[] = [];
        const plusDm: number[] = [];
        const minusDm: number[] = [];

        for (let i = 1; i < recent.length; i++) {
            const curr = recent[i];
            const prev = recent[i - 1];
            const up = curr.high - prev.high;
            const down = prev.low - curr.low;
            plusDm.push((up > down && up > 0) ? up : 0);
            minusDm.push((down > up && down > 0) ? down : 0);
            tr.push(Math.max(
                curr.high - curr.low,
                Math.abs(curr.high - prev.close),
                Math.abs(curr.low - prev.close),
            ));
        }
        if (tr.length < p) return null;

        const atr = tr.slice(-p).reduce((a, b) => a + b, 0) / p;
        if (atr <= StrategyEngine.EPS) return null;
        const plusDi = 100 * (plusDm.slice(-p).reduce((a, b) => a + b, 0) / p) / atr;
        const minusDi = 100 * (minusDm.slice(-p).reduce((a, b) => a + b, 0) / p) / atr;
        const den = plusDi + minusDi;
        if (den <= StrategyEngine.EPS) return 0;
        return 100 * Math.abs(plusDi - minusDi) / den;
    }

    private computeChoppiness(candles: KLineData[], period: number): number | null {
        const p = Math.max(2, Math.floor(period));
        if (candles.length < p + 2) return null;
        const recent = candles.slice(-(p + 1));
        const tr: number[] = [];
        for (let i = 1; i < recent.length; i++) {
            const curr = recent[i];
            const prev = recent[i - 1];
            tr.push(Math.max(
                curr.high - curr.low,
                Math.abs(curr.high - prev.close),
                Math.abs(curr.low - prev.close),
            ));
        }
        if (tr.length < p) return null;
        const sumTr = tr.slice(-p).reduce((a, b) => a + b, 0);
        const hh = Math.max(...recent.slice(-p).map((c) => c.high));
        const ll = Math.min(...recent.slice(-p).map((c) => c.low));
        const span = Math.max(1e-12, hh - ll);
        const ratio = Math.max(1e-12, sumTr / span);
        const denom = Math.max(1e-12, Math.log10(p));
        const chop = 100 * (Math.log10(ratio) / denom);
        return Math.max(0, Math.min(100, chop));
    }

    private computeEmaSlope(candles: KLineData[], period: number): number | null {
        const p = Math.max(2, Math.floor(period));
        if (candles.length < p + 2) return null;
        const closes = candles.map((c) => c.close);
        const alpha = 2 / (p + 1);
        let ema = closes[0];
        let prev = ema;
        for (let i = 1; i < closes.length; i++) {
            prev = ema;
            ema = (alpha * closes[i]) + ((1 - alpha) * ema);
        }
        return (ema - prev) / Math.max(1e-9, Math.abs(prev));
    }

    private calculateEMA(candles: KLineData[], period: number): number[] {
        return this.calculateEmaSeries(candles.map(c => c.close), period);
    }

    private calculateEmaSeries(values: number[], period: number): number[] {
        const k = 2 / (period + 1);
        const emaArray: number[] = new Array(values.length).fill(0);
        if (values.length < period) return emaArray;

        let sum = 0;
        for (let i = 0; i < period; i++) sum += values[i];
        emaArray[period - 1] = sum / period;

        for (let i = period; i < values.length; i++) {
            emaArray[i] = (values[i] * k) + (emaArray[i - 1] * (1 - k));
        }
        return emaArray;
    }

    private calculateEmaValue(values: number[], period: number): number {
        if (!values.length) return 0;
        if (values.length < period) return values[values.length - 1];
        const emaSeries = this.calculateEmaSeries(values, period);
        const last = emaSeries[emaSeries.length - 1];
        return Number.isFinite(last) && last > 0 ? last : values[values.length - 1];
    }

    private calculateLogReturns(prices: number[]): number[] {
        const returns: number[] = [];
        for (let i = 1; i < prices.length; i++) {
            const prev = prices[i - 1];
            const curr = prices[i];
            if (prev > 0 && curr > 0) returns.push(Math.log(curr / prev));
        }
        return returns;
    }

    private calculateMad(returns: number[]): number {
        if (!returns.length) return 0;
        const mean = returns.reduce((a, b) => a + b, 0) / returns.length;
        const absDev = returns.map(r => Math.abs(r - mean));
        return absDev.reduce((a, b) => a + b, 0) / absDev.length;
    }

    private utility(weights: number[], expectedReturn: number, mad: number, currentWeight: number): number[] {
        const cfg = this.mdeConfig;
        return weights.map((w) => {
            const wAbs = Math.abs(w);
            const wCash = Math.max(0, 1 - wAbs);
            const pAssetRaw = Math.max(wAbs, StrategyEngine.EPS);
            const pCashRaw = Math.max(wCash, StrategyEngine.EPS);
            const pSum = pAssetRaw + pCashRaw;
            const pAsset = pAssetRaw / pSum;
            const pCash = pCashRaw / pSum;
            const entropy = -((pAsset * Math.log(pAsset)) + (pCash * Math.log(pCash)));
            const turnover = Math.abs(w - currentWeight);
            return (
                (w * expectedReturn) -
                (cfg.riskAversion * wAbs * mad) +
                (cfg.entropyWeight * entropy) -
                (cfg.turnoverPenalty * turnover)
            );
        });
    }

    private optimizeMdeWeight(returns: number[], isUptrend: boolean, currentWeight: number): number {
        const cfg = this.mdeConfig;
        const expectedReturn = returns.reduce((a, b) => a + b, 0) / returns.length;
        const mad = this.calculateMad(returns);

        if (mad <= StrategyEngine.EPS) {
            if (isUptrend && expectedReturn > 0) return 1;
            if (!isUptrend && cfg.enableShorts && expectedReturn < 0) return -1;
            return 0;
        }

        let lowerBound = 0;
        let upperBound = isUptrend ? cfg.maxLeverage : 0;
        if (!isUptrend && cfg.enableShorts) {
            lowerBound = -cfg.maxLeverage;
            upperBound = 0;
        }
        if (upperBound <= lowerBound) return 0;

        const candidates = new Set<number>();
        const steps = Math.max(9, cfg.optimizationSteps);
        const stepSize = (upperBound - lowerBound) / Math.max(1, steps - 1);
        for (let i = 0; i < steps; i++) candidates.add(lowerBound + (i * stepSize));
        candidates.add(Math.min(upperBound, Math.max(lowerBound, currentWeight)));

        const coarse = Array.from(candidates).sort((a, b) => a - b);
        const coarseUtil = this.utility(coarse, expectedReturn, mad, currentWeight);
        let bestIndex = 0;
        for (let i = 1; i < coarseUtil.length; i++) {
            if (coarseUtil[i] > coarseUtil[bestIndex]) bestIndex = i;
        }
        let bestWeight = coarse[bestIndex];

        const localStep = stepSize;
        const localLower = Math.max(lowerBound, bestWeight - localStep);
        const localUpper = Math.min(upperBound, bestWeight + localStep);
        if (localUpper > localLower) {
            const local: number[] = [];
            for (let i = 0; i < 11; i++) local.push(localLower + ((localUpper - localLower) * i / 10));
            const localUtil = this.utility(local, expectedReturn, mad, currentWeight);
            let bestLocal = 0;
            for (let i = 1; i < localUtil.length; i++) {
                if (localUtil[i] > localUtil[bestLocal]) bestLocal = i;
            }
            bestWeight = local[bestLocal];
        }

        return bestWeight;
    }

    private calculateAtr(candles: KLineData[], period: number): number {
        if (candles.length < 2) return candles[candles.length - 1]?.close * 0.01 || 0;
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

    private calculateWMASeries(values: number[], period: number): number[] {
        const wma: number[] = new Array(values.length).fill(0);
        if (values.length < period) return wma;
        const weightSum = (period * (period + 1)) / 2;
        for (let i = period - 1; i < values.length; i++) {
            let sum = 0;
            for (let j = 0; j < period; j++) {
                sum += values[i - period + 1 + j] * (j + 1);
            }
            wma[i] = sum / weightSum;
        }
        return wma;
    }

    public calculateHMASeries(values: number[], period: number): number[] {
        if (values.length < period) return new Array(values.length).fill(values[values.length - 1] || 0);
        const halfPeriod = Math.floor(period / 2);
        const sqrtPeriod = Math.floor(Math.sqrt(period));

        const wmaHalf = this.calculateWMASeries(values, halfPeriod);
        const wmaFull = this.calculateWMASeries(values, period);

        const diffSeries: number[] = new Array(values.length).fill(0);
        for (let i = 0; i < values.length; i++) {
            diffSeries[i] = 2 * wmaHalf[i] - wmaFull[i];
        }

        return this.calculateWMASeries(diffSeries, sqrtPeriod);
    }

    public calculateKAMASeries(values: number[], period: number): number[] {
        const kama: number[] = new Array(values.length).fill(0);
        if (values.length < period) {
            if (values.length > 0) kama.fill(values[values.length - 1]);
            return kama;
        }

        const fastSC = 2 / (2 + 1);
        const slowSC = 2 / (30 + 1);

        kama[period - 1] = values[period - 1];

        for (let i = period; i < values.length; i++) {
            const change = Math.abs(values[i] - values[i - period]);
            let volatility = 0;
            for (let j = i - period + 1; j <= i; j++) {
                volatility += Math.abs(values[j] - values[j - 1]);
            }

            const er = volatility === 0 ? 0 : change / volatility;
            const sc = Math.pow(er * (fastSC - slowSC) + slowSC, 2);

            kama[i] = kama[i - 1] + sc * (values[i] - kama[i - 1]);
        }
        return kama;
    }

    public calculateEMASeries(values: number[], period: number): number[] {
        const k = 2 / (period + 1);
        const emaArray: number[] = new Array(values.length).fill(0);
        if (values.length < period) return emaArray;

        let sum = 0;
        for (let i = 0; i < period; i++) sum += values[i];
        emaArray[period - 1] = sum / period;

        for (let i = period; i < values.length; i++) {
            emaArray[i] = (values[i] * k) + (emaArray[i - 1] * (1 - k));
        }
        return emaArray;
    }
}
