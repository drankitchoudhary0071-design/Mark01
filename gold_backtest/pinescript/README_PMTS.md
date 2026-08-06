# TradingView — PMTS Improved

File: [`PMTS_Improved.pine`](PMTS_Improved.pine)

## Setup

1. Symbol: **BINANCE:PAXGUSDT**
2. Timeframe: **15 minutes**
3. Pine Editor → paste → **Add to chart**
4. Strategy Tester → Commission **0.1%**

## What it implements (improved PMTS)

| Step | Logic |
|------|--------|
| 1H | Displacement candle → consolidation range → structural break |
| 1H | Fib golden zone **61.8%–70.5%** (tight) |
| 15m | BOS + **retest** (no entry on BOS bar) |
| 15m | Engulfing / pin / rejection wick inside zone |
| Filters | Session 08–18 UTC, SMA(100) trend, ATR%ile ≥ 30 |
| Risk | Min stop **0.40%**, **RR 1:2**, setup expires 48h |

## Expectation

Python improved PMTS was **selective and modest** (~33% WR, few trades) — not a high-winrate system. TV results will differ from Python because of `request.security` HTF timing and fill model. Use the purple zone + phase table to verify the 4-step logic visually.
