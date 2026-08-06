# Gold PMTS Backtest System

End-to-end algorithmic backtest for **PAXG/USDT** (gold proxy on Binance) implementing the Pure Math Trading System (PMTS), an improved variant, and an independent gold Donchian trend strategy.

## Quick start

```bash
pip install -r gold_backtest/requirements.txt
python -m gold_backtest.src.fetch_data          # download 1y 1H + 15m CSVs
python -m gold_backtest.src.run_pipeline        # full IS/OOS backtests + charts
```

Outputs land in `gold_backtest/results/` (report, metrics JSON, trade CSVs) and `gold_backtest/results/charts/`.

## Pipeline steps

1. **Data** — Binance public klines, gap-filled CSVs (`timestamp,open,high,low,close,volume`)
2. **PMTS** — 1H order blocks → fib golden zone → 15m BOS → candlestick confirmation
3. **Engine** — custom event-driven multi-TF backtester with spread/slippage/commission
4. **Split** — ~10 months in-sample (optimization) / ~2 months pure out-of-sample
5. **Improve** — session, BOS-retest, tight fib, volatility gates; before/after compare
6. **New strategy** — Gold Donchian trend (1H breakout + 15m retest), same evaluation protocol

## Cost model

| Component   | Rate    |
|------------|---------|
| Commission | 0.10%/side |
| Half-spread| 0.02% |
| Slippage   | 0.03% |

Round-trip ≈ **0.30%** — intentional; gold scalps that only work without costs are discarded.

## Honesty policy

Reports include explicit overfitting assessment (IS vs OOS gap). Poor performance is not dressed up.
