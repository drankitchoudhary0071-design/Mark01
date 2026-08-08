#!/usr/bin/env python3
"""Sweep MTF RSI HTF/LTF combos on BTC (and optionally PAXG).

Goal: more trades than 1D+4H while staying OOS-profitable after costs.
"""

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
from strategy_lab.engine.metrics import compute_metrics, format_metrics, overfitting_assessment
from strategy_lab.engine.types import BacktestResult
from strategy_lab.engine.walk_forward import holdout_split, walk_forward_windows
from strategy_lab.strategies.mtf_rsi import MtfRsiStrategy, prepare_mtf_frame, resample_ohlcv

OUT = Path(__file__).resolve().parent / "results"

# (label, htf, ltf, source_interval, rsi_os, rsi_ob)
COMBOS = [
    ("1D+4H_30/70", "1D", "4h", "1h", 30, 70),
    ("1D+1H_30/70", "1D", "1h", "1h", 30, 70),
    ("1D+1H_35/65", "1D", "1h", "1h", 35, 65),
    ("4H+1H_30/70", "4h", "1h", "1h", 30, 70),
    ("4H+1H_35/65", "4h", "1h", "1h", 35, 65),
    ("4H+15m_30/70", "4h", "15m", "15m", 30, 70),
    ("4H+15m_35/65", "4h", "15m", "15m", 35, 65),
    ("1H+15m_30/70", "1h", "15m", "15m", 30, 70),
    ("1H+15m_35/65", "1h", "15m", "15m", 35, 65),
]


def _bars_per_day(ltf: str) -> float:
    return {"15m": 96.0, "30m": 48.0, "1h": 24.0, "4h": 6.0, "1D": 1.0}[ltf]


def _wf_sizes(ltf: str) -> tuple[int, int, int]:
    """~90d train / ~30d test / ~30d step, floored for short series."""
    bpd = _bars_per_day(ltf)
    train = max(int(90 * bpd), 200)
    test = max(int(30 * bpd), 60)
    step = test
    return train, test, step


def _max_hold(ltf: str) -> int:
    """Cap hold ~10 calendar days on LTF bars."""
    return max(int(10 * _bars_per_day(ltf)), 20)


def run_combo(
    sym: str,
    label: str,
    htf: str,
    ltf: str,
    source: str,
    os_lvl: int,
    ob_lvl: int,
    days: int = 365,
    risk: float = 0.01,
) -> dict:
    costs = CostModel(commission_rate=0.001, half_spread=0.0002, slippage=0.0003)
    params = {
        "ltf": ltf,
        "htf": htf,
        "rsi_oversold": os_lvl,
        "rsi_overbought": ob_lvl,
        "stop_atr": 1.5,
        "tp_atr": 3.0,
    }
    strat = MtfRsiStrategy(**params)
    raw = load_or_fetch(sym, source, days)
    # If source is finer than LTF, we resample inside prepare; for 1h source
    # used with 15m LTF we cannot upsample — skip such cases.
    if source == "1h" and ltf == "15m":
        raise ValueError("need 15m source for 15m LTF")

    ltf_bars = resample_ohlcv(raw, {"15m": "15min", "1h": "1h", "4h": "4h", "1D": "1D"}[ltf])
    train_n, test_n, step_n = _wf_sizes(ltf)
    hold = _max_hold(ltf)

    # Warmup for HTF EMA200: need ~200 HTF bars ahead of first signal
    htf_days = {"15m": 200 / 96, "1h": 200 / 24, "4h": 200 / 6, "1D": 200}[htf]
    warmup_td = pd.Timedelta(days=max(htf_days * 1.2, 30))

    print("=" * 72)
    print(f"{sym} | {label} | {htf}→{ltf} RSI {os_lvl}/{ob_lvl}")
    print(f"Source {source}: {raw['timestamp'].iloc[0]} → {raw['timestamp'].iloc[-1]}")
    print(f"LTF bars={len(ltf_bars)} WF train/test/step={train_n}/{test_n}/{step_n}")
    print("=" * 72)

    wins = walk_forward_windows(len(ltf_bars), train_n, test_n, step_n)
    capital = 10_000.0
    trades = []
    eq_pts = []
    folds = []
    is_metrics = []

    if not wins:
        # Short series → holdout only
        prepared = prepare_mtf_frame(raw, ltf=ltf, htf=htf)
        a, b = holdout_split(ltf_bars, 0.25)
        a0, a1 = pd.Timestamp(a["timestamp"].iloc[0]), pd.Timestamp(a["timestamp"].iloc[-1])
        b0, b1 = pd.Timestamp(b["timestamp"].iloc[0]), pd.Timestamp(b["timestamp"].iloc[-1])
        raw_sigs = strat.generate_signals(prepared)
        is_s = [s for s in raw_sigs if a0 <= pd.Timestamp(s.timestamp) <= a1]
        oos_s = [s for s in raw_sigs if b0 <= pd.Timestamp(s.timestamp) <= b1]
        is_r = run_backtest(
            a, is_s, costs=costs, risk_per_trade=risk, max_hold_bars=hold,
            strategy_name=strat.name, symbol=sym, params=strat.params,
        )
        oos_r = run_backtest(
            b, oos_s, costs=costs, risk_per_trade=risk, max_hold_bars=hold,
            strategy_name=strat.name, symbol=sym, params=strat.params,
        )
        oos = compute_metrics(oos_r)
        ho = oos
        print(format_metrics(oos, f"{sym} {label} HOLDOUT-only (no WF)"))
        return {
            "label": label,
            "symbol": sym,
            "htf": htf,
            "ltf": ltf,
            "rsi": f"{os_lvl}/{ob_lvl}",
            "wf_trades": oos["total_trades"],
            "wf_win_rate": oos["win_rate"],
            "wf_expectancy_R": oos["expectancy_R"],
            "wf_profit_factor": oos["profit_factor"],
            "wf_sharpe": oos["sharpe_ratio"],
            "wf_max_dd": oos["max_drawdown"],
            "wf_return": oos["total_return"],
            "wf_final_equity": oos["final_equity"],
            "holdout_trades": ho["total_trades"],
            "holdout_win_rate": ho["win_rate"],
            "holdout_profit_factor": ho["profit_factor"],
            "holdout_return": ho["total_return"],
            "folds": 0,
            "mode": "holdout_only",
        }

    for fi, (tr_s, tr_e, te_s, te_e) in enumerate(wins):
        t0 = ltf_bars["timestamp"].iloc[tr_s]
        t1 = ltf_bars["timestamp"].iloc[te_e - 1]
        chunk = raw[
            (raw["timestamp"] >= t0 - warmup_td) & (raw["timestamp"] <= t1)
        ].reset_index(drop=True)
        prepared = prepare_mtf_frame(
            chunk,
            ltf=ltf,
            htf=htf,
            rsi_period=int(strat.params["rsi_period"]),
            atr_period=int(strat.params["atr_period"]),
            ema_fast=int(strat.params["ema_fast"]),
            ema_slow=int(strat.params["ema_slow"]),
        )
        train_start = pd.Timestamp(ltf_bars["timestamp"].iloc[tr_s])
        train_end = pd.Timestamp(ltf_bars["timestamp"].iloc[tr_e - 1])
        test_start = pd.Timestamp(ltf_bars["timestamp"].iloc[te_s])
        test_end = pd.Timestamp(ltf_bars["timestamp"].iloc[te_e - 1])
        raw_sigs = strat.generate_signals(prepared)
        train_sigs = [
            s for s in raw_sigs if train_start <= pd.Timestamp(s.timestamp) <= train_end
        ]
        test_sigs = [
            s for s in raw_sigs if test_start <= pd.Timestamp(s.timestamp) <= test_end
        ]
        train_df = ltf_bars.iloc[tr_s:tr_e].reset_index(drop=True)
        test_df = ltf_bars.iloc[te_s:te_e].reset_index(drop=True)
        tr = run_backtest(
            train_df, train_sigs, initial_capital=10_000, risk_per_trade=risk,
            costs=costs, max_hold_bars=hold, strategy_name=strat.name, symbol=sym,
            params=strat.params, curve_fit_flags=strat.curve_fit_flags,
        )
        te = run_backtest(
            test_df, test_sigs, initial_capital=capital, risk_per_trade=risk,
            costs=costs, max_hold_bars=hold, strategy_name=strat.name, symbol=sym,
            params=strat.params, curve_fit_flags=strat.curve_fit_flags,
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
        BacktestResult(
            trades, eq, 10_000.0, capital, strat.name, sym, strat.params, "",
            strat.curve_fit_flags,
        )
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

    print(format_metrics(oos, f"{sym} {label} WF OOS"))
    print(overfitting_assessment(is_agg, oos))
    for f in folds:
        print(
            f"  fold{f['fold']} {f['test_start'][:10]}→{f['test_end'][:10]} "
            f"n={f['oos_trades']:3d} ret={f['oos_ret']*100:+6.2f}% equity=${f['equity']:,.0f}"
        )

    # Holdout last 25% on full prepared frame
    prepared_full = prepare_mtf_frame(raw, ltf=ltf, htf=htf)
    a, b = holdout_split(ltf_bars, 0.25)
    a0, a1 = pd.Timestamp(a["timestamp"].iloc[0]), pd.Timestamp(a["timestamp"].iloc[-1])
    b0, b1 = pd.Timestamp(b["timestamp"].iloc[0]), pd.Timestamp(b["timestamp"].iloc[-1])
    all_sigs = strat.generate_signals(prepared_full)
    is_s = [s for s in all_sigs if a0 <= pd.Timestamp(s.timestamp) <= a1]
    oos_s = [s for s in all_sigs if b0 <= pd.Timestamp(s.timestamp) <= b1]
    is_r = run_backtest(
        a, is_s, costs=costs, risk_per_trade=risk, max_hold_bars=hold,
        strategy_name=strat.name, symbol=sym, params=strat.params,
    )
    oos_r = run_backtest(
        b, oos_s, costs=costs, risk_per_trade=risk, max_hold_bars=hold,
        strategy_name=strat.name, symbol=sym, params=strat.params,
    )
    ho = compute_metrics(oos_r)
    print(format_metrics(ho, f"{sym} {label} HOLDOUT last 25%"))

    return {
        "label": label,
        "symbol": sym,
        "htf": htf,
        "ltf": ltf,
        "rsi": f"{os_lvl}/{ob_lvl}",
        "wf_trades": oos["total_trades"],
        "wf_win_rate": oos["win_rate"],
        "wf_expectancy_R": oos["expectancy_R"],
        "wf_profit_factor": oos["profit_factor"],
        "wf_sharpe": oos["sharpe_ratio"],
        "wf_max_dd": oos["max_drawdown"],
        "wf_return": oos["total_return"],
        "wf_final_equity": oos["final_equity"],
        "holdout_trades": ho["total_trades"],
        "holdout_win_rate": ho["win_rate"],
        "holdout_profit_factor": ho["profit_factor"],
        "holdout_return": ho["total_return"],
        "folds": len(folds),
        "mode": "walk_forward",
    }


def main() -> int:
    symbols = ["BTCUSDT"]
    if "--paxg" in sys.argv:
        symbols.append("PAXGUSDT")

    # Prefetch both intervals
    for sym in symbols:
        load_or_fetch(sym, "1h", 365)
        load_or_fetch(sym, "15m", 365)

    rows = []
    for sym in symbols:
        for label, htf, ltf, source, os_lvl, ob_lvl in COMBOS:
            try:
                rows.append(
                    run_combo(sym, label, htf, ltf, source, os_lvl, ob_lvl)
                )
            except Exception as e:
                print(f"FAIL {sym} {label}: {e}")
                rows.append(
                    {
                        "label": label,
                        "symbol": sym,
                        "htf": htf,
                        "ltf": ltf,
                        "rsi": f"{os_lvl}/{ob_lvl}",
                        "error": str(e),
                    }
                )

    OUT.mkdir(parents=True, exist_ok=True)
    summary = pd.DataFrame(rows)
    summary.to_csv(OUT / "mtf_tf_sweep_BTCUSDT.csv", index=False)
    (OUT / "mtf_tf_sweep_BTCUSDT.json").write_text(
        json.dumps(rows, indent=2, default=str)
    )
    print("\n" + "=" * 72)
    print("SWEEP SUMMARY")
    cols = [
        c
        for c in (
            "symbol",
            "label",
            "wf_trades",
            "wf_win_rate",
            "wf_expectancy_R",
            "wf_profit_factor",
            "wf_max_dd",
            "wf_return",
            "holdout_trades",
            "holdout_profit_factor",
            "holdout_return",
        )
        if c in summary.columns
    ]
    print(summary[cols].to_string(index=False))
    print(f"Wrote {OUT / 'mtf_tf_sweep_BTCUSDT.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
