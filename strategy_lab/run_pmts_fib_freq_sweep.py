#!/usr/bin/env python3
"""Increase PMTS Fib trade count while keeping WR/DD similar or better.

Baseline (Pine TP=anchor, 1h, buf=3): n≈17 WR≈53% DD≈-3.1% Ret≈+3.4%
Capital $10,000. Costs ≈0.30% RT.
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
from strategy_lab.engine.metrics import compute_metrics
from strategy_lab.strategies.mtf_rsi import resample_ohlcv
from strategy_lab.strategies.pmts_fib_trailing import PmtsFibTrailingStrategy

OUT = Path(__file__).resolve().parent / "results"
CAPITAL = 10_000.0
DAYS = 365
BASE_N = 17
BASE_WR = 0.50
BASE_DD = -0.035


def load_tf(interval: str) -> pd.DataFrame:
    if interval == "4h":
        return resample_ohlcv(load_or_fetch("PAXGUSDT", "1h", DAYS), "4h")
    return load_or_fetch("PAXGUSDT", interval, DAYS)


def run_one(df: pd.DataFrame, interval: str, params: dict, costs: CostModel) -> dict:
    strat = PmtsFibTrailingStrategy(**params)
    sigs = strat.generate_signals(df)
    res = run_backtest(
        df,
        sigs,
        initial_capital=CAPITAL,
        risk_per_trade=0.01,
        costs=costs,
        max_hold_bars=int(params["max_hold_bars"]),
        strategy_name=strat.name,
        symbol="PAXGUSDT",
        params=strat.params,
        curve_fit_flags=strat.curve_fit_flags,
    )
    m = compute_metrics(res)
    wins = sum(1 for t in res.trades if t.pnl > 0)
    losses = sum(1 for t in res.trades if t.pnl < 0)
    out = {
        "interval": interval,
        "swing_len": params.get("swing_len"),
        "entry_level": params.get("entry_level"),
        "sl_buffer_pts": params.get("sl_buffer_pts"),
        "tp_mode": params.get("tp_mode"),
        "rr_multiple": params.get("rr_multiple"),
        "max_hold_bars": params.get("max_hold_bars"),
        "min_range_pts": params.get("min_range_pts", 0),
        "min_planned_rr": params.get("min_planned_rr", 0),
        "flip_soft_exit": params.get("flip_soft_exit", False),
        "htf_bias": params.get("htf_bias", "") or "-",
        "total_trades": m["total_trades"],
        "wins": wins,
        "losses": losses,
        "win_rate": m["win_rate"],
        "expectancy_R": m["expectancy_R"],
        "profit_factor": m["profit_factor"],
        "max_drawdown": m["max_drawdown"],
        "total_return": m["total_return"],
        "final_equity": m["final_equity"],
        "net_pnl": m["final_equity"] - CAPITAL,
        "tag": params.get("_tag", ""),
    }
    return out


def build_configs() -> list[tuple[str, dict]]:
    cfgs: list[tuple[str, dict]] = []

    def add(iv: str, **kw):
        p = {
            "swing_len": 3,
            "entry_level": 0.6,
            "sl_level": 0.7,
            "sl_buffer_pts": 3.0,
            "tp_mode": "anchor",
            "rr_multiple": 3.0,
            "max_hold_bars": {"15m": 96, "1h": 48, "4h": 24}[iv],
            "min_range_pts": 0.0,
            "min_planned_rr": 0.0,
            "flip_soft_exit": False,
            "htf_bias": "",
            "htf_ema": 50,
            "reentry_cooldown": 0,
        }
        p.update(kw)
        cfgs.append((iv, p))

    # Baseline
    add("1h", max_hold_bars=72, sl_buffer_pts=3.0, _tag="BASELINE")

    # 1h: more swings, shorter hold, flip exit, quality filters
    for swing in (2, 3):
        for entry in (0.5, 0.55, 0.6):
            for buf in (3.0, 5.0):
                for hold in (36, 48, 72):
                    for flip in (False, True):
                        for mr, mrr in ((0.0, 0.0), (20.0, 2.0), (25.0, 2.5)):
                            for htf in ("", "4h"):
                                for tpm, rrm in (("anchor", 3.0), ("rr", 5.0)):
                                    if tpm == "rr" and mrr > 0:
                                        continue
                                    add(
                                        "1h",
                                        swing_len=swing,
                                        entry_level=entry,
                                        sl_buffer_pts=buf,
                                        max_hold_bars=hold,
                                        flip_soft_exit=flip,
                                        min_range_pts=mr,
                                        min_planned_rr=mrr,
                                        htf_bias=htf,
                                        tp_mode=tpm,
                                        rr_multiple=rrm,
                                    )

    # 15m: need filters/HTF to protect WR/DD
    for swing in (2, 3):
        for entry in (0.55, 0.6):
            for buf in (4.0, 5.0):
                for hold in (48, 96):
                    for flip in (True, False):
                        for mr, mrr in ((20.0, 2.0), (25.0, 2.5), (30.0, 2.0)):
                            for htf in ("4h", "1D"):
                                for tpm, rrm in (("anchor", 3.0), ("rr", 5.0)):
                                    if tpm == "rr" and mrr > 0:
                                        continue
                                    add(
                                        "15m",
                                        swing_len=swing,
                                        entry_level=entry,
                                        sl_buffer_pts=buf,
                                        max_hold_bars=hold,
                                        flip_soft_exit=flip,
                                        min_range_pts=mr,
                                        min_planned_rr=mrr if tpm == "anchor" else 0.0,
                                        htf_bias=htf,
                                        tp_mode=tpm,
                                        rr_multiple=rrm,
                                    )

    # 4h: slightly more aggressive entry / swing for a few more trades
    for swing in (2, 3):
        for entry in (0.5, 0.55, 0.6):
            for buf in (4.0, 5.0):
                for hold in (12, 18, 24):
                    for flip in (False, True):
                        for htf in ("", "1D"):
                            for tpm, rrm in (("anchor", 3.0), ("rr", 5.0)):
                                add(
                                    "4h",
                                    swing_len=swing,
                                    entry_level=entry,
                                    sl_buffer_pts=buf,
                                    max_hold_bars=hold,
                                    flip_soft_exit=flip,
                                    min_range_pts=0.0,
                                    min_planned_rr=0.0,
                                    htf_bias=htf,
                                    tp_mode=tpm,
                                    rr_multiple=rrm,
                                )

    # Dedup
    seen = set()
    uniq: list[tuple[str, dict]] = []
    for iv, p in cfgs:
        key = (iv, tuple(sorted((k, v) for k, v in p.items() if k != "_tag")))
        if key in seen:
            continue
        seen.add(key)
        uniq.append((iv, p))
    return uniq


def hard_ok(r: dict) -> bool:
    return (
        r["total_trades"] >= BASE_N + 3
        and r["win_rate"] >= BASE_WR
        and r["max_drawdown"] >= BASE_DD
        and r["total_return"] > 0
        and r["profit_factor"] > 1.0
    )


def soft_ok(r: dict) -> bool:
    return (
        r["total_trades"] >= BASE_N
        and r["win_rate"] >= BASE_WR - 0.05
        and r["max_drawdown"] >= BASE_DD - 0.015
        and r["total_return"] > 0
        and r["profit_factor"] > 1.05
    )


def score(r: dict) -> float:
    return (
        r["total_return"] * 120
        + r["win_rate"] * 25
        + min(r["total_trades"], 60) * 0.2
        + r["max_drawdown"] * 60
        + max(r["profit_factor"] - 1, 0) * 8
    )


def fmt(r: dict) -> str:
    tp = "anchor" if r["tp_mode"] == "anchor" else f"{r['rr_multiple']}R"
    return (
        f"{r['interval']:4s} sw={r['swing_len']} ent={r['entry_level']} "
        f"buf={r['sl_buffer_pts']} hold={r['max_hold_bars']} TP={tp} "
        f"flip={r['flip_soft_exit']} minRng={r['min_range_pts']} "
        f"minRR={r['min_planned_rr']} htf={r['htf_bias']} | "
        f"n={r['total_trades']:3d} W/L={r['wins']}/{r['losses']} "
        f"WR={r['win_rate']*100:4.1f}% PF={r['profit_factor']:.2f} "
        f"DD={r['max_drawdown']*100:5.1f}% PnL=${r['net_pnl']:+.0f} "
        f"Ret={r['total_return']*100:+.1f}% Eq=${r['final_equity']:.0f}"
        + (f" [{r['tag']}]" if r.get("tag") else "")
    )


def main() -> int:
    costs = CostModel(commission_rate=0.001, half_spread=0.0002, slippage=0.0003)
    cache = {iv: load_tf(iv) for iv in ("15m", "1h", "4h")}
    cfgs = build_configs()
    print(f"Sweeping {len(cfgs)} configs | PAXG $10k 365d")

    rows: list[dict] = []
    for i, (iv, params) in enumerate(cfgs, 1):
        try:
            r = run_one(cache[iv], iv, params, costs)
        except Exception as e:
            print(f"FAIL {iv} {params.get('_tag')}: {e}")
            continue
        rows.append(r)
        if i % 40 == 0 or r.get("tag") == "BASELINE":
            print(
                f"[{i}/{len(cfgs)}] {iv} n={r['total_trades']} "
                f"WR={r['win_rate']*100:.1f}% DD={r['max_drawdown']*100:.1f}% "
                f"Ret={r['total_return']*100:+.1f}% {r.get('tag','')}"
            )

    OUT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(OUT / "pmts_fib_freq_sweep.csv", index=False)

    hard = sorted([r for r in rows if hard_ok(r)], key=score, reverse=True)
    soft = sorted([r for r in rows if soft_ok(r)], key=score, reverse=True)
    top = sorted(rows, key=score, reverse=True)

    lines = [
        "PMTS Fib — more trades with WR/DD protected | PAXG | $10,000 | 365d",
        "Hard goal: n≥20, WR≥50%, MaxDD≥-3.5%, profitable",
        "Baseline: 1h sw3 ent0.6 buf3 anchor hold72",
        f"Configs: {len(rows)} | Hard hits: {len(hard)} | Soft hits: {len(soft)}",
        "",
        "HARD GOAL TOP",
        "-" * 110,
    ]
    lines += [fmt(r) for r in hard[:20]] if hard else ["(none)"]
    lines += ["", "SOFT GOAL TOP", "-" * 110]
    lines += [fmt(r) for r in soft[:25]]
    lines += ["", "OVERALL TOP 15", "-" * 110]
    lines += [fmt(r) for r in top[:15]]

    pool = hard if hard else soft
    lines += ["", "RECOMMENDED", "-" * 110]
    if pool:
        # Prefer highest trade count among top-scoring half
        top_half = pool[: max(5, len(pool) // 2)] or pool
        best_n = max(top_half, key=lambda x: (x["total_trades"], x["total_return"]))
        best = max(pool, key=score)
        lines.append("Best score:  " + fmt(best))
        lines.append("Most trades: " + fmt(best_n))
        base = next((r for r in rows if r.get("tag") == "BASELINE"), None)
        if base:
            lines.append("Baseline:    " + fmt(base))
            lines.append(
                f"Delta trades {best_n['total_trades'] - base['total_trades']:+d} | "
                f"WR {best_n['win_rate']*100 - base['win_rate']*100:+.1f}pp | "
                f"DD {best_n['max_drawdown']*100 - base['max_drawdown']*100:+.1f}pp | "
                f"Ret {(best_n['total_return'] - base['total_return'])*100:+.1f}pp"
            )
    else:
        lines.append("No soft/hard hits — best profitable with more trades:")
        more = [r for r in rows if r["total_trades"] > BASE_N and r["total_return"] > 0]
        more = sorted(more, key=score, reverse=True)
        lines += [fmt(r) for r in more[:10]]

    text = "\n".join(lines) + "\n"
    (OUT / "pmts_fib_freq_sweep_report.txt").write_text(text)
    (OUT / "pmts_fib_freq_sweep.json").write_text(
        json.dumps({"hard": hard[:25], "soft": soft[:40], "top": top[:25]}, indent=2, default=str)
    )
    print("\n" + text)
    print(f"Wrote {OUT / 'pmts_fib_freq_sweep_report.txt'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
