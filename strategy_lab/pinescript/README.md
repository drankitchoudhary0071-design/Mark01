# Pine Script ports

| Script | Python source | Chart TF | Notes |
|--------|---------------|----------|-------|
| [`MTF_Daily_4H_RSI.pine`](MTF_Daily_4H_RSI.pine) | `strategies/mtf_daily_4h_rsi.py` | **4H** (+ daily via security) | Best MTF — BTC + gold |
| [`BTC_4H_RSI_MR.pine`](BTC_4H_RSI_MR.pine) | `strategies/btc_4h_rsi_mr.py` | **4H BTCUSDT** | 4h-only RSI MR |
| [`RSI_Divergence.pine`](RSI_Divergence.pine) | `strategies/rsi_divergence.py` | 1H | Lookahead risk in old lab |

## MTF Daily + 4H RSI (recommended for BTC & gold)

```bash
python -m strategy_lab.run_mtf_daily_4h_rsi
```

TV: chart **4H** on `BTCUSDT.P` / `PAXGUSDT.P`, commission 0.1%.
