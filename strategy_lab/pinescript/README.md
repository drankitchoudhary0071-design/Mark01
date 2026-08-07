# Pine Script ports

| Script | Python source | Notes |
|--------|---------------|-------|
| [`RSI_Divergence.pine`](RSI_Divergence.pine) | `strategies/rsi_divergence.py` | Bull/bear pivot RSI divergence, ATR stop/TP |

## RSI Divergence — quick start

1. TradingView → chart **PAXGUSDT** or **BTCUSDT**, timeframe **1H** (matches lab default).
2. Pine Editor → paste `RSI_Divergence.pine` → **Add to chart**.
3. Strategy Tester → set commission **0.1%** (lab uses ~0.10%/side + spread/slip).

### Defaults (same as Python)

| Param | Value |
|-------|-------|
| RSI period | 14 |
| Swing lookback | 5 |
| Max pivot gap | 40 bars |
| Stop | 1.5 × ATR(14) |
| TP | 2.5 × ATR(14) |

Pivot confirmation waits `swingLookback` bars after the extreme (same as the Python window definition), so entries are delayed by that many bars — not lookahead.
