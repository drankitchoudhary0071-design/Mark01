"""
data/binance_fetch.py
Fetch OHLCV candles from Binance using python-binance.
Returns a pandas DataFrame with columns:
timestamp, open, high, low, close, volume
"""

from __future__ import annotations

from typing import Optional

import pandas as pd
from binance.client import Client


# Lazily created so importing this module does not hit the network
_client: Optional[Client] = None


def _get_client() -> Client:
    """Public market data does not require API keys."""
    global _client
    if _client is None:
        # ping=False: do not hit the network until get_klines is called
        try:
            _client = Client("", "", requests_params={"timeout": 30}, ping=False)
        except TypeError:
            # Older python-binance without ping= kwarg
            _client = Client("", "")
    return _client


def fetch_ohlcv(
    symbol: str = "PAXGUSDT",
    interval: str = "5m",
    limit: int = 1000,
) -> pd.DataFrame:
    """
    Download klines from Binance and return a clean OHLCV DataFrame.

    Args:
        symbol: Trading pair, e.g. "PAXGUSDT"
        interval: Candle size, e.g. "5m", "1h", "1d"
        limit: Number of candles (max 1000 per request)

    Returns:
        DataFrame sorted by timestamp ascending.
    """
    # Map short interval strings to Binance Client constants when possible
    interval_map = {
        "1m": Client.KLINE_INTERVAL_1MINUTE,
        "3m": Client.KLINE_INTERVAL_3MINUTE,
        "5m": Client.KLINE_INTERVAL_5MINUTE,
        "15m": Client.KLINE_INTERVAL_15MINUTE,
        "30m": Client.KLINE_INTERVAL_30MINUTE,
        "1h": Client.KLINE_INTERVAL_1HOUR,
        "4h": Client.KLINE_INTERVAL_4HOUR,
        "1d": Client.KLINE_INTERVAL_1DAY,
    }
    binance_interval = interval_map.get(interval, interval)

    client = _get_client()
    raw = client.get_klines(symbol=symbol, interval=binance_interval, limit=limit)

    # Binance kline layout:
    # 0 open_time, 1 open, 2 high, 3 low, 4 close, 5 volume, ...
    rows = []
    for candle in raw:
        rows.append(
            {
                "timestamp": pd.to_datetime(candle[0], unit="ms"),
                "open": float(candle[1]),
                "high": float(candle[2]),
                "low": float(candle[3]),
                "close": float(candle[4]),
                "volume": float(candle[5]),
            }
        )

    df = pd.DataFrame(rows)
    df = df.sort_values("timestamp").reset_index(drop=True)
    return df


if __name__ == "__main__":
    # Quick manual smoke test
    data = fetch_ohlcv()
    print(data.tail())
    print(f"Rows: {len(data)}")
