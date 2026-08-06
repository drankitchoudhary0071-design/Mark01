# TradingView Pine Script — High-WR Gold Momentum

File: [`High_WR_Gold_Momentum.pine`](High_WR_Gold_Momentum.pine)

## Setup (2 minutes)

1. Open [TradingView](https://www.tradingview.com/)
2. Symbol: **PAXGUSDT** (Binance) — or `BINANCE:PAXGUSDT`
3. Timeframe: **15 minutes** (required — LTF logic is 15m)
4. Pine Editor → New → paste the `.pine` file → **Add to chart**
5. Open **Strategy Tester**
6. Recommended properties:
   - Commission: **0.1%**
   - Slippage: **2 ticks** (already in script defaults)
   - Initial capital: 10,000
   - Date range: last ~12 months to mirror the Python backtest

## What it does

| Layer | Condition |
|-------|-----------|
| 1H trend | SMA50 > SMA100, close > SMA100 |
| 1H strength | ADX(14) > 35, +DI > −DI |
| 15m entry | Higher low + close > prior high |
| 15m filter | RSI(14) between 50–70 |
| Session | 08:00–17:00 **UTC** |
| Risk | Stop ≥ 1% of price (≤ 2%), **TP = 2R** |
| Cooldown | 8 bars (~2h) between entries |

Long-only. Matches `gold_backtest/src/strategy_high_wr.py`.

## Expect differences vs Python backtest

TradingView’s ADX/`request.security` fill model won’t match our event engine 1:1. Win rate and trade count can differ by a few points. Use TV to **visually verify** signals; treat Python IS/OOS numbers as the research reference.
