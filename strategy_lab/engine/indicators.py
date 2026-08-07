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
