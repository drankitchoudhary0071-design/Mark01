#!/usr/bin/env python3
"""Backtest btc_4h_rsi_mr with TV-like costs + walk-forward / holdout."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from strategy_lab.data.fetch import load_or_fetch
from strategy_lab.engine.costs import CostModel
from strategy_lab.engine.metrics import (
    format_metrics,
    overfitting_assessment,
    trades_to_frame,
)
from strategy_lab.engine.walk_forward import folds_summary_frame, run_holdout, run_walk_forward
from strategy_lab.strategies.btc_4h_rsi_mr import Btc4hRsiMrStrategy, resample_ohlcv_4h

OUT = Path(__file__).resolve().parent / "results"


def main() -> int:
    days = 365
    df1 = load_or_fetch("BTCUSDT", "1h", days)
    df = resample_ohlcv_4h(df1)
    strat = Btc4hRsiMrStrategy()
    # TV Strategy Tester often models 0.1% commission; we add light friction
    costs = CostModel(commission_rate=0.001, half_spread=0.0001, slippage=0.0002)

    print("=" * 72)
    print("btc_4h_rsi_mr — TV-aligned RSI mean-reversion")
    print(f"symbol=BTCUSDT  tf=4h  period={df['timestamp'].iloc[0]} → {df['timestamp'].iloc[-1]}")
    print(f"bars={len(df)}  params={strat.params}")
    print(f"costs RT≈{costs.round_trip_friction()*100:.2f}%")
    print("=" * 72)

    # 4h: ~90d=540 bars train, ~30d=180 bars test
    wf = run_walk_forward(
        df,
        strat.generate_signals,
        train_bars=540,
        test_bars=180,
        step_bars=180,
        costs=costs,
        strategy_name=strat.name,
        symbol="BTCUSDT",
        params=strat.params,
        curve_fit_flags=strat.curve_fit_flags,
    )
    is_r, oos_r, is_m, oos_m = run_holdout(
        df,
        strat.generate_signals,
        oos_frac=0.25,
        costs=costs,
        strategy_name=strat.name,
        symbol="BTCUSDT",
        params=strat.params,
        curve_fit_flags=strat.curve_fit_flags,
    )

    print(format_metrics(wf.combined_oos_metrics, "WALK-FORWARD COMBINED OOS"))
    print(overfitting_assessment(wf.aggregate_is_metrics, wf.combined_oos_metrics))
    print()
    print(format_metrics(oos_m, "HOLDOUT OOS (last 25%)"))
    print(overfitting_assessment(is_m, oos_m))

    OUT.mkdir(parents=True, exist_ok=True)
    trades_to_frame(wf.combined_oos.trades).to_csv(
        OUT / "btc_4h_rsi_mr_wf_oos_trades.csv", index=False
    )
    folds_summary_frame(wf).to_csv(OUT / "btc_4h_rsi_mr_wf_folds.csv", index=False)
    report = "\n\n".join(
        [
            strat.describe(),
            f"Period: {df['timestamp'].iloc[0]} → {df['timestamp'].iloc[-1]} (4h)",
            format_metrics(wf.combined_oos_metrics, "WALK-FORWARD COMBINED OOS"),
            overfitting_assessment(wf.aggregate_is_metrics, wf.combined_oos_metrics),
            format_metrics(oos_m, "HOLDOUT OOS (last 25%)"),
            overfitting_assessment(is_m, oos_m),
        ]
    )
    (OUT / "btc_4h_rsi_mr_report.txt").write_text(report)
    payload = {
        "strategy": strat.name,
        "symbol": "BTCUSDT",
        "timeframe": "4h",
        "period_start": str(df["timestamp"].iloc[0]),
        "period_end": str(df["timestamp"].iloc[-1]),
        "params": strat.params,
        "wf_oos": {k: v for k, v in wf.combined_oos_metrics.items() if k != "params"},
        "holdout_oos": {k: v for k, v in oos_m.items() if k != "params"},
    }
    (OUT / "btc_4h_rsi_mr_metrics.json").write_text(json.dumps(payload, indent=2, default=str))
    print(f"\nWrote reports → {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
