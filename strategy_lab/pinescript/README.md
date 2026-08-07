# Pine Script ports

| Script | Python source | Chart TF | Notes |
|--------|---------------|----------|-------|
| [`BTC_4H_RSI_MR.pine`](BTC_4H_RSI_MR.pine) | `strategies/btc_4h_rsi_mr.py` | **4H BTCUSDT** | TV-aligned RSI MR (no lookahead) — preferred |
| [`RSI_Divergence.pine`](RSI_Divergence.pine) | `strategies/rsi_divergence.py` | 1H | Lab profit had pivot lookahead; TV often loses |

## BTC 4H RSI Mean-Reversion (recommended)

1. TradingView → **BTCUSDT** (or `BTCUSDT.P` for shorts on spot restrictions) → **4H**
2. Paste `BTC_4H_RSI_MR.pine` → Add to chart
3. Strategy Tester → Commission **0.1%**
4. Properties match lab: RSI 14 / 30–70 cross, stop 1.5 ATR, TP 3.0 ATR (1:2)

```bash
python -m strategy_lab.run_btc_4h_rsi_mr
```
