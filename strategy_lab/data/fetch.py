"""Binance OHLCV fetch with pagination, gap fill, and local CSV cache."""

from __future__ import annotations

import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import requests

# data-api mirror avoids HTTP 451 in some regions
BINANCE_URL = "https://data-api.binance.vision/api/v3/klines"
CACHE_DIR = Path(__file__).resolve().parent.parent / "cache"

INTERVAL_MS = {
    "1m": 60_000,
    "3m": 180_000,
    "5m": 300_000,
    "15m": 900_000,
    "30m": 1_800_000,
    "1h": 3_600_000,
    "4h": 14_400_000,
    "1d": 86_400_000,
}


def _fetch_batch(
    symbol: str,
    interval: str,
    start_ms: int,
    end_ms: int,
    limit: int = 1000,
) -> list:
    params = {
        "symbol": symbol,
        "interval": interval,
        "startTime": start_ms,
        "endTime": end_ms,
        "limit": limit,
    }
    for attempt in range(5):
        try:
            r = requests.get(BINANCE_URL, params=params, timeout=30)
            if r.status_code == 429:
                time.sleep(2**attempt)
                continue
            r.raise_for_status()
            return r.json()
        except requests.RequestException:
            if attempt == 4:
                raise
            time.sleep(2**attempt)
    return []


def fetch_ohlcv(
    symbol: str = "PAXGUSDT",
    interval: str = "1h",
    days: int = 365,
    end: datetime | None = None,
) -> pd.DataFrame:
    """Download OHLCV for ``days`` ending at ``end`` (UTC)."""
    if interval not in INTERVAL_MS:
        raise ValueError(f"Unsupported interval: {interval}")
    if end is None:
        end = datetime.now(timezone.utc)
    start = end - timedelta(days=days)
    start_ms = int(start.timestamp() * 1000)
    end_ms = int(end.timestamp() * 1000)
    step = INTERVAL_MS[interval]

    rows: list = []
    cursor = start_ms
    print(f"Fetching {symbol} {interval} from {start.date()} to {end.date()}...")

    while cursor < end_ms:
        batch = _fetch_batch(symbol, interval, cursor, end_ms)
        if not batch:
            break
        rows.extend(batch)
        last_open = batch[-1][0]
        next_cursor = last_open + step
        if next_cursor <= cursor:
            break
        cursor = next_cursor
        if len(batch) < 1000:
            break
        time.sleep(0.05)

    if not rows:
        raise RuntimeError(f"No data returned for {symbol} {interval}")

    df = pd.DataFrame(
        rows,
        columns=[
            "open_time",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "close_time",
            "quote_volume",
            "trades",
            "taker_buy_base",
            "taker_buy_quote",
            "ignore",
        ],
    )
    df = df[["open_time", "open", "high", "low", "close", "volume"]].copy()
    df["timestamp"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    for col in ("open", "high", "low", "close", "volume"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["open", "high", "low", "close"])
    df = df.drop_duplicates(subset=["timestamp"]).sort_values("timestamp")
    df = df.reset_index(drop=True)
    return df[["timestamp", "open", "high", "low", "close", "volume"]]


def fill_gaps(df: pd.DataFrame, interval: str) -> tuple[pd.DataFrame, dict]:
    """Forward-fill missing bars (flat OHLC, zero volume)."""
    step = pd.Timedelta(milliseconds=INTERVAL_MS[interval])
    original_n = len(df)
    full_idx = pd.date_range(
        df["timestamp"].iloc[0], df["timestamp"].iloc[-1], freq=step, tz="UTC"
    )
    out = df.set_index("timestamp").reindex(full_idx)
    missing = int(out["close"].isna().sum())
    out["volume"] = out["volume"].fillna(0.0)
    out[["open", "high", "low", "close"]] = out[["open", "high", "low", "close"]].ffill()
    out[["open", "high", "low", "close"]] = out[["open", "high", "low", "close"]].bfill()
    flat = out["volume"] == 0.0
    for col in ("open", "high", "low"):
        out.loc[flat, col] = out.loc[flat, "close"]
    out = out.reset_index().rename(columns={"index": "timestamp"})
    report = {
        "interval": interval,
        "original_bars": original_n,
        "filled_bars": len(out),
        "missing_bars_filled": missing,
        "missing_pct": round(100.0 * missing / max(len(out), 1), 4),
        "start": str(out["timestamp"].iloc[0]),
        "end": str(out["timestamp"].iloc[-1]),
    }
    return out, report


def cache_path(symbol: str, interval: str, days: int) -> Path:
    return CACHE_DIR / f"{symbol}_{interval}_{days}d.csv"


def load_or_fetch(
    symbol: str,
    interval: str = "1h",
    days: int = 365,
    use_cache: bool = True,
    refresh: bool = False,
) -> pd.DataFrame:
    """Load CSV cache or fetch from Binance, then gap-fill."""
    path = cache_path(symbol, interval, days)
    if use_cache and path.exists() and not refresh:
        df = pd.read_csv(path, parse_dates=["timestamp"])
        if df["timestamp"].dt.tz is None:
            df["timestamp"] = df["timestamp"].dt.tz_localize("UTC")
        print(f"Loaded cache {path} ({len(df)} bars)")
        return df

    raw = fetch_ohlcv(symbol, interval, days)
    df, report = fill_gaps(raw, interval)
    print(f"Gap fill: {report}")
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    save = df.copy()
    save["timestamp"] = save["timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S%z")
    save.to_csv(path, index=False)
    print(f"Cached → {path}")
    return df


def load_pair(
    symbol_a: str,
    symbol_b: str,
    interval: str = "1h",
    days: int = 365,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fetch two symbols and align on common timestamps (inner join)."""
    a = load_or_fetch(symbol_a, interval, days)
    b = load_or_fetch(symbol_b, interval, days)
    merged = a.merge(b, on="timestamp", suffixes=("_a", "_b"), how="inner")
    df_a = merged[["timestamp", "open_a", "high_a", "low_a", "close_a", "volume_a"]].rename(
        columns={
            "open_a": "open",
            "high_a": "high",
            "low_a": "low",
            "close_a": "close",
            "volume_a": "volume",
        }
    )
    df_b = merged[["timestamp", "open_b", "high_b", "low_b", "close_b", "volume_b"]].rename(
        columns={
            "open_b": "open",
            "high_b": "high",
            "low_b": "low",
            "close_b": "close",
            "volume_b": "volume",
        }
    )
    return df_a.reset_index(drop=True), df_b.reset_index(drop=True)
