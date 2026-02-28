import pandas as pd
import numpy as np
import pytest
from src.simulation.indicators import calculate_sma, calculate_ema, calculate_rsi, calculate_atr, calculate_rolling_vwap

def test_sma():
    data = pd.Series([1, 2, 3, 4, 5])
    sma = calculate_sma(data, 3)
    expected = pd.Series([np.nan, np.nan, 2.0, 3.0, 4.0])
    pd.testing.assert_series_equal(sma, expected)

def test_ema():
    data = pd.Series([1, 2, 3, 4, 5])
    ema = calculate_ema(data, 2)
    assert np.isclose(ema.iloc[0], 1.0)
    assert np.isclose(ema.iloc[1], 1.6666666666666665)
    assert np.isclose(ema.iloc[2], 2.5555555555555554)

def test_rsi():
    # Construct a series where RSI is calculable
    data = pd.Series([10, 11, 12, 11, 12, 13, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5])
    # Period 14
    rsi = calculate_rsi(data, 14)
    # Gain at 0 is 0 (fillna).
    assert not np.isnan(rsi.iloc[13])

    # Check bounds on valid values
    valid_rsi = rsi.dropna()
    assert ((valid_rsi >= 0) & (valid_rsi <= 100)).all()

def test_atr():
    high = pd.Series([10, 11, 12, 13, 14])
    low = pd.Series([9, 10, 11, 12, 13])
    close = pd.Series([9.5, 10.5, 11.5, 12.5, 13.5])
    # ATR period 2
    atr = calculate_atr(high, low, close, 2)

    assert np.isclose(atr.iloc[2], 1.375)
    assert np.isclose(atr.iloc[4], 1.46875)

def test_rolling_vwap():
    # Price constant 10, Volume constant 100
    high = pd.Series([10]*5)
    low = pd.Series([10]*5)
    close = pd.Series([10]*5)
    volume = pd.Series([100]*5)

    vwap = calculate_rolling_vwap(high, low, close, volume, 2)
    assert np.isnan(vwap.iloc[0])
    assert np.isclose(vwap.iloc[1], 10.0)
    assert np.isclose(vwap.iloc[4], 10.0)

    # Varying price
    high2 = pd.Series([10, 20])
    low2 = pd.Series([10, 20])
    close2 = pd.Series([10, 20])
    volume2 = pd.Series([100, 100])
    vwap2 = calculate_rolling_vwap(high2, low2, close2, volume2, 2)
    assert np.isclose(vwap2.iloc[1], 15.0)
