'use client';

import React, { useEffect, useRef, useImperativeHandle, forwardRef } from 'react';
import { createChart, ColorType, IChartApi, ISeriesApi, CandlestickData, CandlestickSeries } from 'lightweight-charts';

interface ChartProps {
  data?: CandlestickData[];
  colors?: {
    backgroundColor?: string;
    lineColor?: string;
    textColor?: string;
    areaTopColor?: string;
    areaBottomColor?: string;
  };
}

export interface ChartRef {
  update: (candle: CandlestickData) => void;
  setData: (data: CandlestickData[]) => void;
}

const Chart = forwardRef<ChartRef, ChartProps>((props, ref) => {
  const {
    data = [],
    colors: {
      backgroundColor = '#0a0a0a',
      lineColor = '#00ff41',
      textColor = '#00ff41',
      areaTopColor = '#00ff41',
      areaBottomColor = 'rgba(0, 255, 65, 0.28)',
    } = {},
  } = props;

  const chartContainerRef = useRef<HTMLDivElement>(null);
  const chartRef = useRef<IChartApi | null>(null);
  const seriesRef = useRef<ISeriesApi<"Candlestick"> | null>(null);

  useImperativeHandle(ref, () => ({
    update: (candle: CandlestickData) => {
      if (seriesRef.current) {
        seriesRef.current.update(candle);
      }
    },
    setData: (newData: CandlestickData[]) => {
      if (seriesRef.current) {
        seriesRef.current.setData(newData);
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
      height: 400,
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

    const candlestickSeries = chart.addSeries(CandlestickSeries, {
      upColor: '#00ff41',
      downColor: '#ff003c',
      borderVisible: false,
      wickUpColor: '#00ff41',
      wickDownColor: '#ff003c',
    });

    candlestickSeries.setData(data);

    chartRef.current = chart;
    seriesRef.current = candlestickSeries;

    window.addEventListener('resize', handleResize);

    return () => {
      window.removeEventListener('resize', handleResize);
      chart.remove();
    };
  }, [backgroundColor, lineColor, textColor, areaTopColor, areaBottomColor]);

  return <div ref={chartContainerRef} className="w-full h-[400px]" />;
});

Chart.displayName = 'Chart';

export default Chart;
