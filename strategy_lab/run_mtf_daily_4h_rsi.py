#!/usr/bin/env python3
"""Run MTF Daily+4H RSI on BTCUSDT and PAXGUSDT with slippage + compounding."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from strategy_lab.data.fetch import load_or_fetch
from strategy_lab.engine.backtest import run_backtest
from strategy_lab.engine.costs import CostModel
from strategy_lab.engine.metrics import (
    compute_metrics,
    format_metrics,
    overfitting_assessment,
    trades_to_frame,
)
from strategy_lab.engine.types import BacktestResult
from strategy_lab.engine.walk_forward import holdout_split, walk_forward_windows
from strategy_lab.strategies.mtf_daily_4h_rsi import (
    MtfDaily4hRsiStrategy,
    prepare_mtf_frame,
    resample_ohlcv,
)

OUT = Path(__file__).resolve().parent / "results"


def run_symbol(sym: str, days: int = 365, risk: float = 0.01) -> dict:
    costs = CostModel(commission_rate=0.001, half_spread=0.0002, slippage=0.0003)
    strat = MtfDaily4hRsiStrategy()
    df_1h = load_or_fetch(sym, "1h", days)
    ltf = resample_ohlcv(df_1h, "4h")

    print("=" * 72)
    print(f"{sym} | MTF 1D trend + 4H RSI | slippage+commission | compounding")
    print(f"Source: {df_1h['timestamp'].iloc[0]} → {df_1h['timestamp'].iloc[-1]}")
    print(f"RT friction≈{costs.round_trip_friction()*100:.2f}%  risk/trade={risk*100:.1f}%")
    print(f"params={strat.params}")
    print("=" * 72)

    # Walk-forward on 4h bars with compounding capital
    n4 = len(ltf)
    wins = walk_forward_windows(n4, 540, 180, 180)
    capital = 10_000.0
    trades = []
    eq_pts = []
    folds = []
    is_metrics = []

    for fi, (tr_s, tr_e, te_s, te_e) in enumerate(wins):
        t0 = ltf["timestamp"].iloc[tr_s]
        t1 = ltf["timestamp"].iloc[te_e - 1]
        chunk = df_1h[
            (df_1h["timestamp"] >= t0 - pd.Timedelta(days=220))
            & (df_1h["timestamp"] <= t1)
        ].reset_index(drop=True)
        prepared = prepare_mtf_frame(chunk)
        train_start = pd.Timestamp(ltf["timestamp"].iloc[tr_s])
        train_end = pd.Timestamp(ltf["timestamp"].iloc[tr_e - 1])
        test_start = pd.Timestamp(ltf["timestamp"].iloc[te_s])
        test_end = pd.Timestamp(ltf["timestamp"].iloc[te_e - 1])
        raw = strat.generate_signals(prepared)
        train_sigs = [
            s for s in raw if train_start <= pd.Timestamp(s.timestamp) <= train_end
        ]
        test_sigs = [
            s for s in raw if test_start <= pd.Timestamp(s.timestamp) <= test_end
        ]
        train_df = ltf.iloc[tr_s:tr_e].reset_index(drop=True)
        test_df = ltf.iloc[te_s:te_e].reset_index(drop=True)
        tr = run_backtest(
            train_df, train_sigs, initial_capital=10_000, risk_per_trade=risk, costs=costs,
            strategy_name=strat.name, symbol=sym, params=strat.params,
            curve_fit_flags=strat.curve_fit_flags,
        )
        te = run_backtest(
            test_df, test_sigs, initial_capital=capital, risk_per_trade=risk, costs=costs,
            strategy_name=strat.name, symbol=sym, params=strat.params,
            curve_fit_flags=strat.curve_fit_flags,
        )
        is_metrics.append(compute_metrics(tr))
        folds.append(
            {
                "fold": fi,
                "test_start": str(test_start),
                "test_end": str(test_end),
                "oos_trades": len(te.trades),
                "oos_ret": te.final_capital / capital - 1 if capital else 0,
                "equity": te.final_capital,
            }
        )
        capital = te.final_capital
        trades.extend(te.trades)
        for t, v in te.equity.items():
            eq_pts.append((pd.Timestamp(t), float(v)))

    idx = pd.DatetimeIndex([t for t, _ in eq_pts])
    eq = pd.Series([v for _, v in eq_pts], index=idx)
    eq = eq[~eq.index.duplicated(keep="last")].sort_index()
    oos = compute_metrics(
        BacktestResult(trades, eq, 10_000.0, capital, strat.name, sym, strat.params, "", strat.curve_fit_flags)
    )

    def _avg(k: str) -> float:
        vals = [m[k] for m in is_metrics]
        return float(sum(vals) / len(vals)) if vals else 0.0

    is_agg = {
        **is_metrics[0],
        "total_trades": int(sum(m["total_trades"] for m in is_metrics)),
        "win_rate": _avg("win_rate"),
        "expectancy_R": _avg("expectancy_R"),
        "profit_factor": _avg("profit_factor"),
        "sharpe_ratio": _avg("sharpe_ratio"),
        "total_return": _avg("total_return"),
        "max_drawdown": _avg("max_drawdown"),
    }

    print(format_metrics(oos, f"{sym} WF OOS (compounded)"))
    print(overfitting_assessment(is_agg, oos))
    for f in folds:
        print(
            f"  fold{f['fold']} {f['test_start'][:10]}→{f['test_end'][:10]} "
            f"n={f['oos_trades']:2d} ret={f['oos_ret']*100:+6.2f}% equity=${f['equity']:,.0f}"
        )

    # Holdout
    prepared_full = prepare_mtf_frame(df_1h)
    a, b = holdout_split(ltf, 0.25)
    a0, a1 = pd.Timestamp(a["timestamp"].iloc[0]), pd.Timestamp(a["timestamp"].iloc[-1])
    b0, b1 = pd.Timestamp(b["timestamp"].iloc[0]), pd.Timestamp(b["timestamp"].iloc[-1])
    raw = strat.generate_signals(prepared_full)
    is_s = [s for s in raw if a0 <= pd.Timestamp(s.timestamp) <= a1]
    oos_s = [s for s in raw if b0 <= pd.Timestamp(s.timestamp) <= b1]
    is_r = run_backtest(a, is_s, costs=costs, risk_per_trade=risk, strategy_name=strat.name, symbol=sym)
    oos_r = run_backtest(b, oos_s, costs=costs, risk_per_trade=risk, strategy_name=strat.name, symbol=sym)
    is_m, ho_m = compute_metrics(is_r), compute_metrics(oos_r)
    print(format_metrics(ho_m, f"{sym} HOLDOUT OOS last 25%"))
    print(overfitting_assessment(is_m, ho_m))

    OUT.mkdir(parents=True, exist_ok=True)
    trades_to_frame(trades).to_csv(OUT / f"mtf_daily_4h_rsi_{sym}_wf_trades.csv", index=False)
    pd.DataFrame(folds).to_csv(OUT / f"mtf_daily_4h_rsi_{sym}_wf_folds.csv", index=False)
    report = "\n\n".join(
        [
            strat.describe(),
            f"Period: {df_1h['timestamp'].iloc[0]} → {df_1h['timestamp'].iloc[-1]}",
            format_metrics(oos, f"{sym} WF OOS compounded"),
            overfitting_assessment(is_agg, oos),
            format_metrics(ho_m, f"{sym} Holdout OOS"),
            overfitting_assessment(is_m, ho_m),
        ]
    )
    (OUT / f"mtf_daily_4h_rsi_{sym}_report.txt").write_text(report)

    return {
        "symbol": sym,
        "wf_oos": oos,
        "holdout_oos": ho_m,
        "folds": folds,
        "final_equity": capital,
    }


def main() -> int:
    rows = []
    for sym in ("BTCUSDT", "PAXGUSDT"):
        r = run_symbol(sym)
        o, h = r["wf_oos"], r["holdout_oos"]
        rows.append(
            {
                "strategy": "mtf_daily_4h_rsi",
                "symbol": sym,
                "wf_trades": o["total_trades"],
                "wf_win_rate": o["win_rate"],
                "wf_expectancy_R": o["expectancy_R"],
                "wf_profit_factor": o["profit_factor"],
                "wf_sharpe": o["sharpe_ratio"],
                "wf_max_dd": o["max_drawdown"],
                "wf_return": o["total_return"],
                "wf_final_equity": o["final_equity"],
                "holdout_trades": h["total_trades"],
                "holdout_win_rate": h["win_rate"],
                "holdout_profit_factor": h["profit_factor"],
                "holdout_return": h["total_return"],
            }
        )
    summary = pd.DataFrame(rows)
    summary.to_csv(OUT / "mtf_daily_4h_rsi_summary.csv", index=False)
    (OUT / "mtf_daily_4h_rsi_metrics.json").write_text(
        json.dumps(rows, indent=2, default=str)
    )
    print("\n" + "=" * 72)
    print("SUMMARY — Gold (PAXG) & BTC")
    print(summary.to_string(index=False))
    print(f"Wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
