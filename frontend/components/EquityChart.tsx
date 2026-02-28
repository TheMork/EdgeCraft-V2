'use client';

import React, { useEffect, useRef, useImperativeHandle, forwardRef } from 'react';
import {
  createChart,
  ColorType,
  IChartApi,
  ISeriesApi,
  LineData,
  AreaSeries,
  BusinessDay,
} from 'lightweight-charts';

interface ChartProps {
  data?: LineData[];
  colors?: {
    backgroundColor?: string;
    lineColor?: string;
    textColor?: string;
    topColor?: string;
    bottomColor?: string;
  };
}

export interface EquityChartRef {
  update: (point: LineData) => void;
  setData: (data: LineData[]) => void;
}

const isBusinessDay = (value: LineData['time']): value is BusinessDay => {
  if (typeof value !== 'object' || value === null) {
    return false;
  }
  return 'year' in value && 'month' in value && 'day' in value;
};

const toUnixSeconds = (time: LineData['time']): number | null => {
  if (typeof time === 'number') {
    if (!Number.isFinite(time)) return null;
    return Math.floor(time);
  }
  if (typeof time === 'string') {
    const ms = Date.parse(time);
    if (!Number.isFinite(ms)) return null;
    return Math.floor(ms / 1000);
  }
  if (isBusinessDay(time)) {
    const ms = Date.UTC(time.year, time.month - 1, time.day);
    if (!Number.isFinite(ms)) return null;
    return Math.floor(ms / 1000);
  }
  return null;
};

const normalizeLineData = (data: LineData[]): LineData[] => {
  const byTime = new Map<number, number>();
  for (const point of data) {
    const ts = toUnixSeconds(point.time);
    const value = Number(point.value);
    if (ts === null || !Number.isFinite(value)) continue;
    // Keep latest value for duplicate timestamps.
    byTime.set(ts, value);
  }

  return Array.from(byTime.entries())
    .sort((a, b) => a[0] - b[0])
    .map(([ts, value]) => ({
      time: ts as LineData['time'],
      value,
    }));
};

const EquityChart = forwardRef<EquityChartRef, ChartProps>((props, ref) => {
  const {
    data = [],
    colors: {
      backgroundColor = '#0a0a0a',
      lineColor = '#00ff41',
      textColor = '#00ff41',
      topColor = 'rgba(0, 255, 65, 0.28)',
      bottomColor = 'rgba(0, 255, 65, 0.0)',
    } = {},
  } = props;

  const chartContainerRef = useRef<HTMLDivElement>(null);
  const chartRef = useRef<IChartApi | null>(null);
  const seriesRef = useRef<ISeriesApi<"Area"> | null>(null);

  useImperativeHandle(ref, () => ({
    update: (point: LineData) => {
      if (seriesRef.current) {
        seriesRef.current.update(point);
      }
    },
    setData: (newData: LineData[]) => {
      if (seriesRef.current) {
        seriesRef.current.setData(normalizeLineData(newData));
      }
    }
  }));

  useEffect(() => {
    if (!chartContainerRef.current) return;

    const handleResize = () => {
      if (chartRef.current && chartContainerRef.current) {
        chartRef.current.applyOptions({ width: chartContainerRef.current.clientWidth });
      }
    };

    const chart = createChart(chartContainerRef.current, {
      layout: {
        background: { type: ColorType.Solid, color: backgroundColor },
        textColor,
      },
      width: chartContainerRef.current.clientWidth,
      height: 300,
      grid: {
          vertLines: { color: '#00441b' },
          horzLines: { color: '#00441b' },
      },
      timeScale: {
        timeVisible: true,
        secondsVisible: false,
      },
    });

    chart.timeScale().fitContent();

    const areaSeries = chart.addSeries(AreaSeries, {
      lineColor,
      topColor,
      bottomColor,
    });

    areaSeries.setData(normalizeLineData(data));

    chartRef.current = chart;
    seriesRef.current = areaSeries;

    window.addEventListener('resize', handleResize);

    return () => {
      window.removeEventListener('resize', handleResize);
      chart.remove();
    };
  }, [backgroundColor, lineColor, textColor, topColor, bottomColor]);

  return <div ref={chartContainerRef} className="w-full h-[300px]" />;
});

EquityChart.displayName = 'EquityChart';

export default EquityChart;
