#!/usr/bin/env bash
# Download ~1y XAUUSD M5 bid OHLC from Dukascopy (OANDA-style gold spot proxy).
# Requires: node/npx. No OANDA API key needed.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
CACHE="$ROOT/strategy_lab/cache"
FROM="${1:-2025-08-09}"
TO="${2:-2026-08-09}"
mkdir -p "$ROOT/download" "$CACHE"
cd "$ROOT"
npx --yes dukascopy-node -i xauusd -from "$FROM" -to "$TO" -t m5 -f csv -v
RAW=$(ls -1t download/xauusd-m5-bid-*.csv | head -1)
python3 - <<PY
import pandas as pd
from pathlib import Path
src = Path("$RAW")
out = Path("$CACHE") / "XAUUSD_5m_365d.csv"
df = pd.read_csv(src)
df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
df = df[["timestamp", "open", "high", "low", "close", "volume"]]
df = df.sort_values("timestamp").drop_duplicates("timestamp")
df = df[(df["high"] >= df["low"]) & (df["close"] > 0)]
out.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(out, index=False)
print(f"Cached {len(df)} bars → {out}")
PY
