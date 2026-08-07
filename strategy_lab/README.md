# strategy_lab

Walk-forward / out-of-sample backtests for **10 strategy families** on Binance pairs
(default: **PAXGUSDT** + **BTCUSDT**).

Every discretionary strategy uses the **same shared engine** (fills, costs, risk sizing,
metrics) so results are comparable. Grid, market-making, and pairs use specialized
simulators that still emit the same `Trade` / `BacktestResult` schema.

## Strategies

| # | Module | Family | Signal library |
|---|--------|--------|----------------|
| 1 | `strategies/trend_following.py` | MA crossover + ADX filter | pandas + shared engine |
| 2 | `strategies/mean_reversion.py` | Bollinger + RSI | pandas + shared engine |
| 3 | `strategies/statistical_arbitrage.py` | Pairs residual z-score | pandas/numpy two-leg sim |
| 4 | `strategies/grid_trading.py` | Fixed-interval grid | grid simulator |
| 5 | `strategies/scalping.py` | 5m EMA burst + tight SL/TP | pandas + shared engine |
| 6 | `strategies/regime_momentum.py` | Donchian breakout × vol/ADX regime | pandas + shared engine |
| 7 | `strategies/volatility_squeeze.py` | BB-in-KC squeeze release | pandas + shared engine |
| 8 | `strategies/smc_fvg_ob.py` | Fair Value Gap + Order Block (SMC-lite) | pandas + shared engine |
| 9 | `strategies/market_making.py` | Bid/ask capture (**simulation only**) | MM simulator |
| 10 | `strategies/momentum_vol_filter.py` | Dual momentum × vol percentile gate | pandas + shared engine |
| 11 | `strategies/cointegration_pairs.py` | Engle–Granger cointegration pairs | numpy EG + two-leg sim |
| 12 | `strategies/vwap_twap.py` | VWAP/TWAP execution vs aggressive | execution simulator |
| 13 | `strategies/regime_adaptive.py` | Regime detect → trend/range switch | pandas + shared engine |
| 14 | `strategies/adaptive_mean_reversion.py` | Dynamic mean by ADX regime | pandas + shared engine |
| 15 | `strategies/volume_confirmed_reversal.py` | RSI extreme + volume spike pin | pandas + shared engine |
| 16 | `strategies/rsi_divergence.py` | Price HH/LL vs RSI non-confirm | pandas + shared engine |
| 17 | `strategies/rebalance_arbitrage.py` | Month-end rebalance **proxy** | calendar proxy (applicability flagged) |
| 18 | `strategies/news_event.py` | News/event-driven | **stub — external feed required** |
| 19 | `strategies/mtf_confluence.py` | HTF+LTF confluence | pandas resample + shared engine |
| 20 | `strategies/ensemble_voting.py` | Multi-agent weighted vote (≥3) | ensemble + shared engine |

> **Why not backtrader/vectorbt as the sole runner?**  
> Comparability. Vectorbt is excellent for parameter sweeps on single-asset vectorized
> rules; here the hard requirement is identical cost/risk accounting across families
> that include multi-leg and inventory strategies. Signals stay pandas/numpy; execution
> is centralized in `engine/backtest.py`.

## Cost model (configurable)

| Component | Default |
|-----------|---------|
| Commission | 0.10% / side (taker) |
| Half-spread | 0.02% |
| Slippage | 0.03% |

Round-trip ≈ **0.30%**. Scalps that only “work” with zero costs will fail here on purpose.

## Walk-forward / OOS

- **Default:** rolling walk-forward (~90d train / ~30d test / ~30d step on 1h; scaled for other TFs).
- **Alternative:** `--no-walk-forward --oos-frac 0.25` holdout split.
- Special sims (grid / MM / pairs) use holdout (stateful inventory does not stitch cleanly across WF folds without look-ahead).

## Metrics (every strategy)

- Win rate  
- Expectancy (R and $)  
- Max drawdown  
- Sharpe ratio (daily equity)  
- Profit factor  

Plus IS vs OOS overfitting assessment.

## Curve-fit policy

Defaults in `config.py` are **textbook starting points**, not optimized values.
Parameters that are easy to overfit are flagged via `PARAM_NOTES` /
`curve_fit_warnings()` and printed in every report. **Do not promote a single-run
tuned set into defaults.**

## Quick start

```bash
# from repo root
pip install -r strategy_lab/requirements.txt

# all 10 families on PAXGUSDT + BTCUSDT
python -m strategy_lab.run_all

# custom pair / TF
python -m strategy_lab.run_all --symbols ETHUSDT BTCUSDT --interval 1h --days 180

# subset
python -m strategy_lab.run_all --strategies trend_following mean_reversion volatility_squeeze

# refresh Binance cache
python -m strategy_lab.run_all --refresh-data
```

Outputs → `strategy_lab/results/`:

- `summary_comparison.csv`
- `{strategy}_{symbol}_metrics.json`
- `{strategy}_{symbol}_report.txt`
- `{strategy}_{symbol}_oos_trades.csv`
- `{strategy}_{symbol}_wf_folds.csv` (when walk-forward)

## Configurable knobs

CLI: `--interval`, `--days`, `--risk`, `--commission`, `--half-spread`, `--slippage`,
`--wf-train`, `--wf-test`, `--wf-step`, `--oos-frac`.

Per-strategy: construct the class with overrides, e.g.

```python
from strategy_lab.strategies import TrendFollowingStrategy
s = TrendFollowingStrategy(fast_ma=10, slow_ma=40, adx_threshold=20)
signals = s.generate_signals(df)
print(s.curve_fit_flags)
```

## Project layout

```
strategy_lab/
  config.py              # defaults + curve-fit notes
  data/fetch.py          # Binance OHLCV + cache
  engine/
    backtest.py          # shared event-driven engine
    costs.py             # slippage / spread / commission
    metrics.py           # WR, expectancy, DD, Sharpe, PF
    walk_forward.py      # WF + holdout
    indicators.py        # shared TA
    types.py             # Signal, Trade, BacktestResult
  strategies/            # one module per family
  run_all.py             # CLI
  results/               # reports
  cache/                 # CSV cache (gitignored)
```

## Disclaimer

Research code only. Market-making and SMC modules are simplified heuristics.
Past backtests (especially in-sample) do not imply future performance.
