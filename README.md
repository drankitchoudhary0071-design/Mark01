# Mark01

Contains the Mark01 agent project and a complete **gold (PAXG/USDT) algorithmic backtest system**.

## Gold backtest (`gold_backtest/`)

```bash
pip install -r gold_backtest/requirements.txt
PYTHONPATH=. python3 -m gold_backtest.src.fetch_data
PYTHONPATH=. python3 -m gold_backtest.src.run_pipeline
```

Lower timeframe is **15m** (with 1H setups). See `gold_backtest/README.md` and `gold_backtest/results/full_report.txt`.
