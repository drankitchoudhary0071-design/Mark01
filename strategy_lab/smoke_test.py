"""Offline smoke test with synthetic OHLCV (no network)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from strategy_lab.engine.backtest import run_backtest
from strategy_lab.engine.costs import CostModel
from strategy_lab.engine.metrics import compute_metrics
from strategy_lab.engine.walk_forward import run_holdout, run_walk_forward
from strategy_lab.strategies import STRATEGY_REGISTRY, SPECIAL_SIM
from strategy_lab.strategies.grid_trading import simulate_grid
from strategy_lab.strategies.market_making import simulate_market_making
from strategy_lab.strategies.statistical_arbitrage import simulate_pairs


def make_ohlcv(n: int = 2500, seed: int = 0, start: float = 100.0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rets = rng.normal(0.0001, 0.01, size=n)
    # Add mild regime structure
    rets[n // 3 : 2 * n // 3] += 0.0005
    close = start * np.exp(np.cumsum(rets))
    high = close * (1 + rng.uniform(0.0005, 0.01, size=n))
    low = close * (1 - rng.uniform(0.0005, 0.01, size=n))
    open_ = close * (1 + rng.normal(0, 0.001, size=n))
    vol = rng.uniform(10, 100, size=n)
    ts = pd.date_range("2024-01-01", periods=n, freq="1h", tz="UTC")
    return pd.DataFrame(
        {
            "timestamp": ts,
            "open": open_,
            "high": np.maximum(high, np.maximum(open_, close)),
            "low": np.minimum(low, np.minimum(open_, close)),
            "close": close,
            "volume": vol,
        }
    )


def main() -> int:
    df = make_ohlcv()
    df_b = make_ohlcv(seed=1, start=50_000.0)
    costs = CostModel()
    failures = []

    for name, cls in STRATEGY_REGISTRY.items():
        try:
            if name == "statistical_arbitrage":
                res = simulate_pairs(df, df_b, costs=costs, symbol="A/B")
            elif name == "grid_trading":
                res = simulate_grid(df, costs=costs, symbol="SYN")
            elif name == "market_making":
                res = simulate_market_making(df, costs=costs, symbol="SYN")
            else:
                strat = cls()
                # holdout path
                is_res, oos_res, is_m, oos_m = run_holdout(
                    df,
                    strat.generate_signals,
                    oos_frac=0.25,
                    costs=costs,
                    strategy_name=name,
                    symbol="SYN",
                    params=strat.params,
                    curve_fit_flags=strat.curve_fit_flags,
                )
                # short walk-forward
                wf = run_walk_forward(
                    df,
                    strat.generate_signals,
                    train_bars=800,
                    test_bars=200,
                    step_bars=200,
                    costs=costs,
                    strategy_name=name,
                    symbol="SYN",
                    params=strat.params,
                )
                res = oos_res
                _ = is_m, oos_m, wf
            m = compute_metrics(res)
            assert "win_rate" in m and "expectancy_R" in m
            assert "max_drawdown" in m and "sharpe_ratio" in m and "profit_factor" in m
            print(f"OK  {name:24s} trades={m['total_trades']:4d}  "
                  f"WR={m['win_rate']*100:5.1f}%  ExpR={m['expectancy_R']:+.3f}")
        except Exception as e:
            failures.append((name, e))
            print(f"FAIL {name}: {e}")

    # Direct engine smoke
    from strategy_lab.strategies.trend_following import TrendFollowingStrategy

    sigs = TrendFollowingStrategy().generate_signals(df)
    res = run_backtest(df, sigs, costs=costs, strategy_name="trend_following", symbol="SYN")
    print(f"Engine trades={len(res.trades)} final={res.final_capital:.2f}")

    if failures:
        print(f"\n{len(failures)} failures")
        return 1
    print("\nAll strategy smoke tests passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
