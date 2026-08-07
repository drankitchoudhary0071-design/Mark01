"""Technical indicators used across strategy modules (pandas/numpy only)."""

from __future__ import annotations

import numpy as np
import pandas as pd


def true_range(df: pd.DataFrame) -> pd.Series:
    prev_close = df["close"].shift(1)
    return pd.concat(
        [
            df["high"] - df["low"],
            (df["high"] - prev_close).abs(),
            (df["low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)


def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    return true_range(df).rolling(period).mean()


def sma(series: pd.Series, period: int) -> pd.Series:
    return series.rolling(period).mean()


def ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()


def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def bollinger(
    series: pd.Series, period: int = 20, num_std: float = 2.0
) -> tuple[pd.Series, pd.Series, pd.Series]:
    mid = series.rolling(period).mean()
    std = series.rolling(period).std()
    upper = mid + num_std * std
    lower = mid - num_std * std
    return lower, mid, upper


def keltner(
    df: pd.DataFrame, period: int = 20, atr_mult: float = 1.5
) -> tuple[pd.Series, pd.Series, pd.Series]:
    mid = ema(df["close"], period)
    band = atr(df, period) * atr_mult
    return mid - band, mid, mid + band


def adx(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """Return DataFrame with +DI, -DI, ADX columns."""
    up = df["high"].diff()
    down = -df["low"].diff()
    plus_dm = np.where((up > down) & (up > 0), up, 0.0)
    minus_dm = np.where((down > up) & (down > 0), down, 0.0)
    tr = true_range(df)
    atr_s = pd.Series(tr, index=df.index).rolling(period).mean()
    plus_di = 100 * pd.Series(plus_dm, index=df.index).rolling(period).mean() / atr_s
    minus_di = 100 * pd.Series(minus_dm, index=df.index).rolling(period).mean() / atr_s
    dx = (100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan))
    adx_s = dx.rolling(period).mean()
    return pd.DataFrame({"+di": plus_di, "-di": minus_di, "adx": adx_s}, index=df.index)


def donchian(df: pd.DataFrame, period: int = 20) -> tuple[pd.Series, pd.Series]:
    return df["low"].rolling(period).min(), df["high"].rolling(period).max()


def rolling_zscore(series: pd.Series, lookback: int) -> pd.Series:
    mu = series.rolling(lookback).mean()
    sd = series.rolling(lookback).std()
    return (series - mu) / sd.replace(0, np.nan)


def realized_vol(returns: pd.Series, lookback: int) -> pd.Series:
    return returns.rolling(lookback).std() * np.sqrt(lookback)


def macd(
    series: pd.Series,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """MACD line, signal line, histogram (TradingView defaults)."""
    line = ema(series, fast) - ema(series, slow)
    sig = ema(line, signal)
    hist = line - sig
    return line, sig, hist


def stochastic(
    df: pd.DataFrame,
    k_period: int = 14,
    k_smooth: int = 3,
    d_period: int = 3,
) -> tuple[pd.Series, pd.Series]:
    """Stochastic %K / %D (TradingView-style)."""
    lowest = df["low"].rolling(k_period).min()
    highest = df["high"].rolling(k_period).max()
    raw_k = 100 * (df["close"] - lowest) / (highest - lowest).replace(0, np.nan)
    k = raw_k.rolling(k_smooth).mean()
    d = k.rolling(d_period).mean()
    return k, d


def supertrend(
    df: pd.DataFrame, period: int = 10, multiplier: float = 3.0
) -> tuple[pd.Series, pd.Series]:
    """
    Supertrend line + direction (+1 bull / -1 bear).
    Classic TradingView Supertrend (ATR-based).
    """
    atr_s = atr(df, period)
    hl2 = (df["high"] + df["low"]) / 2.0
    basic_upper = hl2 + multiplier * atr_s
    basic_lower = hl2 - multiplier * atr_s

    n = len(df)
    st = np.full(n, np.nan)
    direction = np.ones(n)
    final_upper = np.full(n, np.nan)
    final_lower = np.full(n, np.nan)
    close = df["close"].to_numpy()
    bu = basic_upper.to_numpy()
    bl = basic_lower.to_numpy()

    for i in range(n):
        if np.isnan(bu[i]) or np.isnan(bl[i]):
            continue
        if i == 0 or np.isnan(final_upper[i - 1]):
            final_upper[i] = bu[i]
            final_lower[i] = bl[i]
            direction[i] = 1.0
            st[i] = final_lower[i]
            continue

        final_upper[i] = (
            bu[i]
            if bu[i] < final_upper[i - 1] or close[i - 1] > final_upper[i - 1]
            else final_upper[i - 1]
        )
        final_lower[i] = (
            bl[i]
            if bl[i] > final_lower[i - 1] or close[i - 1] < final_lower[i - 1]
            else final_lower[i - 1]
        )

        if direction[i - 1] > 0:
            if close[i] < final_lower[i]:
                direction[i] = -1.0
                st[i] = final_upper[i]
            else:
                direction[i] = 1.0
                st[i] = final_lower[i]
        else:
            if close[i] > final_upper[i]:
                direction[i] = 1.0
                st[i] = final_lower[i]
            else:
                direction[i] = -1.0
                st[i] = final_upper[i]

    return (
        pd.Series(st, index=df.index),
        pd.Series(direction, index=df.index),
    )


def cci(df: pd.DataFrame, period: int = 20) -> pd.Series:
    tp = (df["high"] + df["low"] + df["close"]) / 3.0
    sma_tp = tp.rolling(period).mean()
    mad = tp.rolling(period).apply(lambda x: np.mean(np.abs(x - x.mean())), raw=True)
    return (tp - sma_tp) / (0.015 * mad.replace(0, np.nan))

