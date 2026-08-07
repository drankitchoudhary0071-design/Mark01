#!/usr/bin/env python3
"""Backtest TradingView multi-indicator MTF variants; print profitable only."""

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
)
from strategy_lab.engine.types import BacktestResult
from strategy_lab.engine.walk_forward import holdout_split, walk_forward_windows
from strategy_lab.strategies.mtf_rsi import resample_ohlcv
from strategy_lab.strategies.tv_mtf_indicators import (
    RULES,
    VARIANT_META,
    TvMtfStrategy,
    prepare_tv_mtf,
)

OUT = Path(__file__).resolve().parent / "results"

# Extra param tweaks on top of each variant (still 1D→4H)
TWEAKS = [
    {},  # defaults stop1.5/tp3
    {"rsi_os": 35, "rsi_ob": 65},
    {"stop_atr": 2.0, "tp_atr": 4.0},
    {"adx_min": 25},
]


def _wf(sym: str, variant: str, tweak: dict, days: int = 365, risk: float = 0.01) -> dict:
    costs = CostModel(commission_rate=0.001, half_spread=0.0002, slippage=0.0003)
    params = {"variant": variant, "ltf": "4h", "htf": "1D", **tweak}
    strat = TvMtfStrategy(**params)
    raw = load_or_fetch(sym, "1h", days)
    ltf = resample_ohlcv(raw, "4h")
    train_n, test_n, step_n = 540, 180, 180
    hold = 60
    warmup = pd.Timedelta(days=240)

    wins = walk_forward_windows(len(ltf), train_n, test_n, step_n)
    capital = 10_000.0
    trades = []
    eq_pts = []
    is_metrics = []
    folds = []

    for fi, (tr_s, tr_e, te_s, te_e) in enumerate(wins):
        t0 = ltf["timestamp"].iloc[tr_s]
        t1 = ltf["timestamp"].iloc[te_e - 1]
        chunk = raw[
            (raw["timestamp"] >= t0 - warmup) & (raw["timestamp"] <= t1)
        ].reset_index(drop=True)
        prepared = prepare_tv_mtf(chunk, ltf="4h", htf="1D", params=strat.params)
        tr0 = pd.Timestamp(ltf["timestamp"].iloc[tr_s])
        tr1 = pd.Timestamp(ltf["timestamp"].iloc[tr_e - 1])
        te0 = pd.Timestamp(ltf["timestamp"].iloc[te_s])
        te1 = pd.Timestamp(ltf["timestamp"].iloc[te_e - 1])
        sigs = strat.generate_signals(prepared)
        train_s = [s for s in sigs if tr0 <= pd.Timestamp(s.timestamp) <= tr1]
        test_s = [s for s in sigs if te0 <= pd.Timestamp(s.timestamp) <= te1]
        tr = run_backtest(
            ltf.iloc[tr_s:tr_e].reset_index(drop=True),
            train_s,
            initial_capital=10_000,
            risk_per_trade=risk,
            costs=costs,
            max_hold_bars=hold,
            strategy_name=strat.name,
            symbol=sym,
            params=strat.params,
        )
        te = run_backtest(
            ltf.iloc[te_s:te_e].reset_index(drop=True),
            test_s,
            initial_capital=capital,
            risk_per_trade=risk,
            costs=costs,
            max_hold_bars=hold,
            strategy_name=strat.name,
            symbol=sym,
            params=strat.params,
        )
        is_metrics.append(compute_metrics(tr))
        folds.append(
            {
                "fold": fi,
                "oos_trades": len(te.trades),
                "oos_ret": te.final_capital / capital - 1 if capital else 0,
            }
        )
        capital = te.final_capital
        trades.extend(te.trades)
        for t, v in te.equity.items():
            eq_pts.append((pd.Timestamp(t), float(v)))

    if not eq_pts:
        oos = {
            "total_trades": 0,
            "win_rate": 0,
            "expectancy_R": 0,
            "profit_factor": 0,
            "sharpe_ratio": 0,
            "max_drawdown": 0,
            "total_return": 0,
            "final_equity": capital,
        }
    else:
        idx = pd.DatetimeIndex([t for t, _ in eq_pts])
        eq = pd.Series([v for _, v in eq_pts], index=idx)
        eq = eq[~eq.index.duplicated(keep="last")].sort_index()
        oos = compute_metrics(
            BacktestResult(
                trades, eq, 10_000.0, capital, strat.name, sym, strat.params, "",
                strat.curve_fit_flags,
            )
        )

    # Holdout
    prepared_full = prepare_tv_mtf(raw, ltf="4h", htf="1D", params=strat.params)
    a, b = holdout_split(ltf, 0.25)
    b0, b1 = pd.Timestamp(b["timestamp"].iloc[0]), pd.Timestamp(b["timestamp"].iloc[-1])
    all_s = strat.generate_signals(prepared_full)
    oos_s = [s for s in all_s if b0 <= pd.Timestamp(s.timestamp) <= b1]
    ho = compute_metrics(
        run_backtest(
            b, oos_s, costs=costs, risk_per_trade=risk, max_hold_bars=hold,
            strategy_name=strat.name, symbol=sym, params=strat.params,
        )
    )

    meta = VARIANT_META[variant]
    tweak_label = ",".join(f"{k}={v}" for k, v in tweak.items()) or "default"
    return {
        "symbol": sym,
        "variant": variant,
        "tweak": tweak_label,
        "indicators": ", ".join(meta["indicators"]),
        "n_indicators": meta["n_ind"],
        "desc": meta["desc"],
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
        "params": strat.params,
        "curve_fit_flags": strat.curve_fit_flags,
        "_oos_metrics": oos,
        "_ho_metrics": ho,
        "_is_metrics": is_metrics,
    }


def _profitable(row: dict) -> bool:
    """WF OOS profitable with PF>1 and at least a few trades."""
    return (
        row.get("wf_trades", 0) >= 8
        and row.get("wf_return", 0) > 0
        and row.get("wf_profit_factor", 0) > 1.0
        and row.get("wf_expectancy_R", 0) > 0
    )


def main() -> int:
    symbols = ["BTCUSDT"]
    if "--paxg" in sys.argv:
        symbols.append("PAXGUSDT")

    rows = []
    for sym in symbols:
        for variant in RULES:
            for tweak in TWEAKS:
                # Skip irrelevant tweaks
                if "adx_min" in tweak and variant not in (
                    "ema_adx_stoch",
                    "adx_rsi_macd_bb",
                ):
                    continue
                if ("rsi_os" in tweak) and variant in (
                    "ema_adx_stoch",
                    "ema_macd_stoch",
                ):
                    continue
                label = f"{variant}|{','.join(f'{k}={v}' for k,v in tweak.items()) or 'default'}"
                print("=" * 72)
                print(f"RUN {sym} {label}")
                try:
                    r = _wf(sym, variant, tweak)
                except Exception as e:
                    print(f"FAIL: {e}")
                    continue
                rows.append(r)
                print(
                    f"  → n={r['wf_trades']} WR={r['wf_win_rate']:.1%} "
                    f"ExpR={r['wf_expectancy_R']:+.3f} PF={r['wf_profit_factor']:.2f} "
                    f"DD={r['wf_max_dd']:.1%} RET={r['wf_return']:+.1%} "
                    f"| HO PF={r['holdout_profit_factor']:.2f} RET={r['holdout_return']:+.1%}"
                )
                if _profitable(r):
                    print(format_metrics(r["_oos_metrics"], f"{sym} {variant} WF OOS ★"))

    OUT.mkdir(parents=True, exist_ok=True)
    # Strip heavy objects for JSON/CSV
    clean = []
    for r in rows:
        c = {k: v for k, v in r.items() if not k.startswith("_")}
        clean.append(c)

    all_df = pd.DataFrame(clean)
    all_df.to_csv(OUT / "tv_mtf_all_runs.csv", index=False)

    prof = [r for r in clean if _profitable(r)]
    # Prefer also holdout non-negative when possible for ranking
    prof_sorted = sorted(
        prof,
        key=lambda r: (
            r["wf_return"],
            r["holdout_return"],
            r["wf_profit_factor"],
        ),
        reverse=True,
    )
    prof_df = pd.DataFrame(prof_sorted)
    prof_df.to_csv(OUT / "tv_mtf_profitable_only.csv", index=False)
    (OUT / "tv_mtf_profitable_only.json").write_text(
        json.dumps(prof_sorted, indent=2, default=str)
    )

    print("\n" + "=" * 72)
    print("★ PROFITABLE ONLY (WF OOS: return>0, PF>1, ExpR>0, trades≥8)")
    print("=" * 72)
    if not prof_sorted:
        print("None profitable under costs.")
    else:
        show = [
            "symbol",
            "variant",
            "tweak",
            "indicators",
            "wf_trades",
            "wf_win_rate",
            "wf_expectancy_R",
            "wf_profit_factor",
            "wf_max_dd",
            "wf_return",
            "holdout_trades",
            "holdout_profit_factor",
            "holdout_return",
        ]
        print(prof_df[[c for c in show if c in prof_df.columns]].to_string(index=False))

        # Write human report of profitable only
        lines = [
            "TradingView multi-indicator MTF — PROFITABLE results only",
            "BTCUSDT | 1D HTF → 4H LTF | ~0.30% RT costs | WF compounded",
            "",
        ]
        for i, r in enumerate(prof_sorted, 1):
            lines += [
                f"{i}. {r['variant']} [{r['tweak']}]",
                f"   Indicators: {r['indicators']}",
                f"   {r['desc']}",
                f"   WF OOS: n={r['wf_trades']} WR={r['wf_win_rate']*100:.1f}% "
                f"ExpR={r['wf_expectancy_R']:+.3f} PF={r['wf_profit_factor']:.2f} "
                f"DD={r['wf_max_dd']*100:.1f}% RET={r['wf_return']*100:+.1f}% "
                f"→ ${r['wf_final_equity']:,.0f}",
                f"   Holdout: n={r['holdout_trades']} PF={r['holdout_profit_factor']:.2f} "
                f"RET={r['holdout_return']*100:+.1f}%",
                "",
            ]
        (OUT / "tv_mtf_profitable_report.txt").write_text("\n".join(lines))
        print(f"\nWrote {OUT / 'tv_mtf_profitable_report.txt'}")

    print(f"All runs: {OUT / 'tv_mtf_all_runs.csv'} ({len(clean)} configs)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
