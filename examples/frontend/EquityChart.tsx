import React, { useEffect, useRef, useMemo } from 'react';
import { createChart, ColorType, IChartApi, ISeriesApi, Time } from 'lightweight-charts';

// Data Interface based on the blueprint
export interface EquityPoint {
    time: string; // ISO 8601 or YYYY-MM-DD
    value: number; // Portfolio Equity
}

interface EquityChartProps {
    data: EquityPoint[];
    height?: number;
    colors?: {
        backgroundColor?: string;
        lineColor?: string;
        textColor?: string;
        areaTopColor?: string;
        areaBottomColor?: string;
    };
}

const DEFAULT_COLORS = {
    backgroundColor: '#161b22', // Dark Mode Surface
    lineColor: '#2962ff',       // Accent Blue
    textColor: '#e6e6e6',       // Primary Text
    areaTopColor: 'rgba(41, 98, 255, 0.56)',
    areaBottomColor: 'rgba(41, 98, 255, 0.04)',
};

export const EquityChart: React.FC<EquityChartProps> = ({
    data,
    height = 400,
    colors = DEFAULT_COLORS
}) => {
    const chartContainerRef = useRef<HTMLDivElement>(null);
    const chartRef = useRef<IChartApi | null>(null);
    const seriesRef = useRef<ISeriesApi<"Area"> | null>(null);

    // Memoize colors to prevent unnecessary re-renders if parent passes a new object
    const finalColors = useMemo(() => ({
        ...DEFAULT_COLORS,
        ...colors,
    }), [colors]);

    useEffect(() => {
        if (!chartContainerRef.current) return;

        // Initialize Chart
        const chart = createChart(chartContainerRef.current, {
            layout: {
                background: { type: ColorType.Solid, color: finalColors.backgroundColor },
                textColor: finalColors.textColor,
            },
            width: chartContainerRef.current.clientWidth,
            height: height,
            grid: {
                vertLines: { color: '#2b2b43' },
                horzLines: { color: '#2b2b43' },
            },
            rightPriceScale: {
                borderColor: '#2b2b43',
            },
            timeScale: {
                borderColor: '#2b2b43',
                timeVisible: true,
                secondsVisible: false,
            },
        });

        chartRef.current = chart;

        // Add Area Series (Equity Curve)
        const newSeries = chart.addAreaSeries({
            lineColor: finalColors.lineColor,
            topColor: finalColors.areaTopColor,
            bottomColor: finalColors.areaBottomColor,
            lineWidth: 2,
        });

        seriesRef.current = newSeries;

        // Initial Data Load
        // Ensure data is sorted by time
        const sortedData = [...data].sort((a, b) => new Date(a.time).getTime() - new Date(b.time).getTime());

        const formattedData = sortedData.map(d => ({
            time: d.time as unknown as Time,
            value: d.value
        }));

        newSeries.setData(formattedData);
        chart.timeScale().fitContent();

        // Handle Resize with Observer
        const handleResize = () => {
            if (chartContainerRef.current) {
                chart.applyOptions({ width: chartContainerRef.current.clientWidth });
            }
        };

        const resizeObserver = new ResizeObserver(() => handleResize());
        resizeObserver.observe(chartContainerRef.current);

        return () => {
            resizeObserver.disconnect();
            chart.remove();
        };
    }, [finalColors, height]); // Re-create chart if colors or height change.

    // Handle Data Updates efficiently without destroying chart
    useEffect(() => {
        if (seriesRef.current && data.length > 0) {
            const sortedData = [...data].sort((a, b) => new Date(a.time).getTime() - new Date(b.time).getTime());
            const formattedData = sortedData.map(d => ({
                time: d.time as unknown as Time,
                value: d.value
            }));
            seriesRef.current.setData(formattedData);
        }
    }, [data]);

    return (
        <div className="relative w-full rounded-lg overflow-hidden border border-gray-800 shadow-xl">
            <div className="absolute top-4 left-4 z-10 pointer-events-none">
                <h3 className="text-sm font-semibold text-gray-400 uppercase tracking-wider">
                    Portfolio Equity
                </h3>
                {data.length > 0 && (
                    <p className="text-2xl font-bold text-white">
                        ${data[data.length - 1].value.toLocaleString()}
                    </p>
                )}
            </div>
            <div ref={chartContainerRef} />
        </div>
    );
};
