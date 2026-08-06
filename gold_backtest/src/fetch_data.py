"""
Fetch 1 year of PAXG/USDT OHLCV from Binance public API.
Handles pagination, rate limits, gap detection, and forward-fill of missing bars.
"""

from __future__ import annotations

import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import requests

# Public market-data mirror (api.binance.com returns HTTP 451 in restricted regions)
BINANCE_URL = "https://data-api.binance.vision/api/v3/klines"
SYMBOL = "PAXGUSDT"
DATA_DIR = Path(__file__).resolve().parent.parent / "data"

INTERVAL_MS = {
    "1h": 60 * 60 * 1000,
    "15m": 15 * 60 * 1000,
    "5m": 5 * 60 * 1000,  # kept for optional re-fetch
}


def _fetch_batch(symbol: str, interval: str, start_ms: int, end_ms: int, limit: int = 1000) -> list:
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
                time.sleep(2 ** attempt)
                continue
            r.raise_for_status()
            return r.json()
        except requests.RequestException:
            if attempt == 4:
                raise
            time.sleep(2 ** attempt)
    return []


def fetch_ohlcv(
    symbol: str = SYMBOL,
    interval: str = "1h",
    days: int = 365,
    end: datetime | None = None,
) -> pd.DataFrame:
    """Download OHLCV candles for `days` ending at `end` (UTC)."""
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
        time.sleep(0.05)  # be polite to Binance

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
    df = df.drop_duplicates(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
    df = df[["timestamp", "open", "high", "low", "close", "volume"]]
    return df


def fill_gaps(df: pd.DataFrame, interval: str) -> tuple[pd.DataFrame, dict]:
    """
    Detect missing bars and forward-fill OHLC (flat) with zero volume.
    Returns cleaned frame + gap report.
    """
    step = pd.Timedelta(milliseconds=INTERVAL_MS[interval])
    full_idx = pd.date_range(df["timestamp"].iloc[0], df["timestamp"].iloc[-1], freq=step, tz="UTC")
    original_n = len(df)
    df = df.set_index("timestamp").reindex(full_idx)
    missing = int(df["close"].isna().sum())

    # Forward-fill price; volume = 0 on synthetic bars
    df["volume"] = df["volume"].fillna(0.0)
    df[["open", "high", "low", "close"]] = df[["open", "high", "low", "close"]].ffill()
    df[["open", "high", "low", "close"]] = df[["open", "high", "low", "close"]].bfill()

    # Synthetic (gap-filled) bars: flat OHLC at last close
    flat_mask = df["volume"] == 0.0
    for col in ("open", "high", "low"):
        df.loc[flat_mask, col] = df.loc[flat_mask, "close"]

    df = df.reset_index().rename(columns={"index": "timestamp"})
    report = {
        "interval": interval,
        "original_bars": original_n,
        "filled_bars": len(df),
        "missing_bars_filled": missing,
        "missing_pct": round(100.0 * missing / max(len(df), 1), 4),
        "start": str(df["timestamp"].iloc[0]),
        "end": str(df["timestamp"].iloc[-1]),
    }
    return df, report


def save_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    out = df.copy()
    out["timestamp"] = out["timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S")
    out.to_csv(path, index=False)
    print(f"Saved {len(out)} rows -> {path}")


def load_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    return df


def fetch_and_save(days: int = 365) -> dict:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    reports = {}
    for interval, fname in (("1h", "paxgusdt_1h.csv"), ("15m", "paxgusdt_15m.csv")):
        raw = fetch_ohlcv(interval=interval, days=days)
        clean, report = fill_gaps(raw, interval)
        # Drop any residual NaNs
        clean = clean.dropna().reset_index(drop=True)
        save_csv(clean, DATA_DIR / fname)
        reports[interval] = report
        print(f"Gap report {interval}: {report}")
    return reports


if __name__ == "__main__":
    fetch_and_save(365)
