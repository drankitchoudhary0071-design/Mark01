#!/usr/bin/env python3
"""
Scalping lab: build on first 6 months, test on next 6 months (1m & 5m).

Only prints / saves OOS-profitable configs.
"""

from __future__ import annotations

import json
import sys
from itertools import product
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from strategy_lab.data.fetch import load_or_fetch
from strategy_lab.engine.backtest import run_backtest
from strategy_lab.engine.costs import CostModel
from strategy_lab.engine.metrics import compute_metrics, format_metrics
from strategy_lab.strategies.scalp_ltf import RULES, VARIANT_META, ScalpMtfStrategy

OUT = Path(__file__).resolve().parent / "results"

# Cost models
TAKER = CostModel(commission_rate=0.001, half_spread=0.0002, slippage=0.0003)  # ~0.30% RT
# Maker/limit-style (still pays some friction) — secondary, labeled clearly
MAKER = CostModel(commission_rate=0.0002, half_spread=0.0001, slippage=0.00015)  # ~0.09% RT

# Param grid searched ONLY on train (first 6m)
STOPS = [1.0, 1.5, 2.0, 2.5]
TPS = [1.5, 2.0, 2.5, 3.0, 4.0]
# Keep RR sensible for scalp: TP >= stop * 0.8
MIN_TRAIN_TRADES = 30


def split_6m_6m(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """First half calendar ≈ 6m train, second half ≈ 6m OOS."""
    t0 = pd.Timestamp(df["timestamp"].iloc[0])
    mid = t0 + pd.Timedelta(days=182)
    train = df[df["timestamp"] < mid].reset_index(drop=True)
    test = df[df["timestamp"] >= mid].reset_index(drop=True)
    return train, test


def max_hold_for(interval: str) -> int:
    # Cap hold ~45–60 minutes calendar time
    return {"1m": 45, "5m": 12}[interval]


def vwap_window_for(interval: str) -> int:
    # ~6.5 hours rolling VWAP proxy
    return {"1m": 390, "5m": 78}[interval]


def run_one(
    df: pd.DataFrame,
    variant: str,
    params: dict,
    costs: CostModel,
    risk: float = 0.005,
) -> dict:
    strat = ScalpMtfStrategy(**params)
    hold = int(params["max_hold_bars"])
    sigs = strat.generate_signals(df)
    res = run_backtest(
        df,
        sigs,
        initial_capital=10_000,
        risk_per_trade=risk,
        costs=costs,
        max_hold_bars=hold,
        strategy_name=strat.name,
        symbol="BTCUSDT",
        params=strat.params,
        curve_fit_flags=strat.curve_fit_flags,
    )
    m = compute_metrics(res)
    return {
        "metrics": m,
        "n_signals": len(sigs),
        "final_equity": res.final_capital,
        "strategy": strat.name,
        "params": dict(strat.params),
        "flags": list(strat.curve_fit_flags),
    }


def search_train(
    train: pd.DataFrame,
    interval: str,
    costs: CostModel,
    cost_name: str,
) -> list[dict]:
    """Pick best param set per variant on TRAIN only."""
    hold = max_hold_for(interval)
    vwap_w = vwap_window_for(interval)
    winners: list[dict] = []

    for variant in RULES:
        best = None
        best_score = -1e18
        tested = 0
        for stop, tp in product(STOPS, TPS):
            if tp < stop * 0.9:
                continue
            params = {
                "variant": variant,
                "stop_atr": stop,
                "tp_atr": tp,
                "max_hold_bars": hold,
                "vwap_window": vwap_w,
                "min_atr_pct": 0.0003 if interval == "1m" else 0.0004,
            }
            # Light RSI tweaks for MR variants
            extras = [{}]
            if variant in ("vwap_rsi", "adx_ema_rsi", "bb_stoch"):
                extras = [{}, {"rsi_os": 35, "rsi_ob": 65}]
            if variant == "ema_rsi":
                extras = [{}, {"rsi_long_min": 50, "rsi_short_max": 50}]
            for ex in extras:
                p = {**params, **ex}
                r = run_one(train, variant, p, costs)
                m = r["metrics"]
                tested += 1
                if m["total_trades"] < MIN_TRAIN_TRADES:
                    continue
                if m["expectancy_R"] <= 0 or m["profit_factor"] < 1.05:
                    continue
                # Score: prioritize expectancy then PF then return
                score = (
                    m["expectancy_R"] * 100
                    + m["profit_factor"] * 10
                    + m["total_return"] * 50
                    - abs(m["max_drawdown"]) * 20
                )
                if score > best_score:
                    best_score = score
                    best = {
                        "variant": variant,
                        "interval": interval,
                        "cost_model": cost_name,
                        "train": m,
                        "params": r["params"],
                        "score": score,
                        "meta": VARIANT_META[variant],
                    }
        print(
            f"  train search {interval} {variant} [{cost_name}]: "
            f"tested={tested} best="
            + (
                f"n={best['train']['total_trades']} "
                f"PF={best['train']['profit_factor']:.2f} "
                f"ret={best['train']['total_return']*100:+.1f}%"
                if best
                else "NONE"
            )
        )
        if best:
            winners.append(best)
    return winners


def oos_test(test: pd.DataFrame, cand: dict, costs: CostModel) -> dict:
    r = run_one(test, cand["variant"], cand["params"], costs)
    m = r["metrics"]
    return {
        **cand,
        "oos": m,
        "oos_final_equity": r["final_equity"],
        "oos_profitable": (
            m["total_trades"] >= 15
            and m["total_return"] > 0
            and m["profit_factor"] > 1.0
            and m["expectancy_R"] > 0
        ),
    }


def main() -> int:
    intervals = ["5m"]
    if "--1m" in sys.argv or "--all" in sys.argv:
        intervals.append("1m")
    # Always try 1m if cache exists / fetch works
    if "--5m-only" not in sys.argv:
        try:
            load_or_fetch("BTCUSDT", "1m", 365)
            if "1m" not in intervals:
                intervals.append("1m")
        except Exception as e:
            print(f"1m fetch skipped: {e}")

    cost_modes = [("taker_0.30pct_RT", TAKER), ("makerish_0.09pct_RT", MAKER)]

    all_rows = []
    profitable = []

    for interval in intervals:
        print("=" * 72)
        print(f"Loading BTCUSDT {interval} 365d…")
        raw = load_or_fetch("BTCUSDT", interval, 365)
        train, test = split_6m_6m(raw)
        print(
            f"{interval}: total={len(raw)} train={len(train)} "
            f"({train['timestamp'].iloc[0]}→{train['timestamp'].iloc[-1]}) "
            f"oos={len(test)} ({test['timestamp'].iloc[0]}→{test['timestamp'].iloc[-1]})"
        )

        for cost_name, costs in cost_modes:
            print(f"\n--- Cost: {cost_name} (RT≈{costs.round_trip_friction()*100:.2f}%) ---")
            cands = search_train(train, interval, costs, cost_name)
            for c in cands:
                o = oos_test(test, c, costs)
                row = {
                    "interval": o["interval"],
                    "variant": o["variant"],
                    "cost_model": o["cost_model"],
                    "indicators": ", ".join(o["meta"]["indicators"]),
                    "desc": o["meta"]["desc"],
                    "stop_atr": o["params"]["stop_atr"],
                    "tp_atr": o["params"]["tp_atr"],
                    "train_trades": o["train"]["total_trades"],
                    "train_wr": o["train"]["win_rate"],
                    "train_pf": o["train"]["profit_factor"],
                    "train_expR": o["train"]["expectancy_R"],
                    "train_ret": o["train"]["total_return"],
                    "train_dd": o["train"]["max_drawdown"],
                    "oos_trades": o["oos"]["total_trades"],
                    "oos_wr": o["oos"]["win_rate"],
                    "oos_pf": o["oos"]["profit_factor"],
                    "oos_expR": o["oos"]["expectancy_R"],
                    "oos_ret": o["oos"]["total_return"],
                    "oos_dd": o["oos"]["max_drawdown"],
                    "oos_equity": o["oos_final_equity"],
                    "oos_profitable": o["oos_profitable"],
                    "params": o["params"],
                }
                all_rows.append(row)
                tag = "★ OOS PROFIT" if o["oos_profitable"] else "oos fail"
                print(
                    f"  {tag} {interval} {o['variant']} stop={o['params']['stop_atr']} "
                    f"tp={o['params']['tp_atr']}: "
                    f"OOS n={o['oos']['total_trades']} WR={o['oos']['win_rate']:.1%} "
                    f"PF={o['oos']['profit_factor']:.2f} ExpR={o['oos']['expectancy_R']:+.3f} "
                    f"RET={o['oos']['total_return']*100:+.1f}% DD={o['oos']['max_drawdown']*100:.1f}%"
                )
                if o["oos_profitable"]:
                    profitable.append(row)
                    print(format_metrics(o["oos"], f"OOS {interval} {o['variant']}"))

    OUT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(all_rows).to_csv(OUT / "scalp_6m6m_all.csv", index=False)

    prof_sorted = sorted(
        profitable,
        key=lambda r: (r["oos_ret"], r["oos_pf"], r["oos_expR"]),
        reverse=True,
    )
    pd.DataFrame(prof_sorted).to_csv(OUT / "scalp_6m6m_profitable_oos.csv", index=False)
    (OUT / "scalp_6m6m_profitable_oos.json").write_text(
        json.dumps(prof_sorted, indent=2, default=str)
    )

    lines = [
        "SCALPING 1m/5m — first 6m TRAIN / next 6m OOS — PROFITABLE OOS ONLY",
        "Symbol: BTCUSDT | risk 0.5%/trade | compounding off (flat 10k per split)",
        "",
    ]
    if not prof_sorted:
        lines.append("No OOS-profitable scalp config under tested costs/params.")
        lines.append("Scalping usually dies to commission+slippage on 1m/5m.")
    else:
        for i, r in enumerate(prof_sorted, 1):
            lines += [
                f"{i}. {r['interval']} | {r['variant']} | costs={r['cost_model']}",
                f"   Indicators: {r['indicators']}",
                f"   {r['desc']}",
                f"   Params: stop={r['stop_atr']}×ATR  tp={r['tp_atr']}×ATR",
                f"   TRAIN 6m: n={r['train_trades']} WR={r['train_wr']*100:.1f}% "
                f"PF={r['train_pf']:.2f} ExpR={r['train_expR']:+.3f} "
                f"RET={r['train_ret']*100:+.1f}% DD={r['train_dd']*100:.1f}%",
                f"   OOS   6m: n={r['oos_trades']} WR={r['oos_wr']*100:.1f}% "
                f"PF={r['oos_pf']:.2f} ExpR={r['oos_expR']:+.3f} "
                f"RET={r['oos_ret']*100:+.1f}% DD={r['oos_dd']*100:.1f}% "
                f"→ ${r['oos_equity']:,.0f}",
                "",
            ]
    report = "\n".join(lines)
    (OUT / "scalp_6m6m_profitable_report.txt").write_text(report)

    print("\n" + "=" * 72)
    print(report)
    print(f"\nWrote {OUT / 'scalp_6m6m_profitable_report.txt'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
