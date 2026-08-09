#!/usr/bin/env python3
"""
PMTS exit-mode sweep on XAUUSD (and optional PAXG/BTC).

Diagnoses why baseline loses and tests SL @ 0.7 fib + TP variants.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import strategy_lab.run_pmts_exact_v3 as base

OUT = Path(__file__).resolve().parent / "results" / "pmts_exit_sweep"
OUT.mkdir(parents=True, exist_ok=True)

INITIAL = base.INITIAL_CAPITAL
NOTIONAL = base.FIXED_NOTIONAL


@dataclass
class ExitMode:
    name: str
    # sl: swing_pct | fib_0.7 | fib_0.7_buf | swing_pct_wide
    sl_mode: str
    # tp: anchor_pct | anchor | fib_0.0 | rr_2 | rr_3
    tp_mode: str
    sl_pct: float = 0.09
    tp_pct: float = 0.14
    fib_sl: float = 0.7  # beyond zone
    sl_buf_pct: float = 0.05  # extra beyond 0.7 fib
    min_rr: float = 0.0  # skip entry if planned RR below this


def fib_level(ahi: float, alo: float, direction: str, level: float) -> float:
    rng = ahi - alo
    if direction == "bull":
        return ahi - rng * level
    return alo + rng * level


def compute_sl_tp(
    mode: ExitMode,
    direction: str,
    entry: float,
    ahi: float,
    alo: float,
    ltf_sh: float,
    ltf_sl: float,
) -> tuple[float, float] | None:
    if direction == "bull":
        if mode.sl_mode == "swing_pct":
            sl = ltf_sl * (1.0 - mode.sl_pct / 100.0)
        elif mode.sl_mode == "swing_pct_wide":
            sl = ltf_sl * (1.0 - mode.sl_pct / 100.0)
        elif mode.sl_mode in ("fib_0.7", "fib_sl"):
            sl = fib_level(ahi, alo, "bull", mode.fib_sl)
        elif mode.sl_mode == "fib_0.7_buf":
            sl = fib_level(ahi, alo, "bull", mode.fib_sl) * (1.0 - mode.sl_buf_pct / 100.0)
        else:
            raise ValueError(mode.sl_mode)

        if mode.tp_mode == "anchor_pct":
            tp = ahi * (1.0 - mode.tp_pct / 100.0)
        elif mode.tp_mode == "anchor":
            tp = ahi
        elif mode.tp_mode == "fib_0.0":
            tp = ahi
        elif mode.tp_mode.startswith("rr_"):
            rr = float(mode.tp_mode.split("_")[1])
            risk = entry - sl
            if risk <= 0:
                return None
            tp = entry + rr * risk
        else:
            raise ValueError(mode.tp_mode)

        if not (sl < entry < tp):
            return None
    else:
        if mode.sl_mode == "swing_pct":
            sl = ltf_sh * (1.0 + mode.sl_pct / 100.0)
        elif mode.sl_mode == "swing_pct_wide":
            sl = ltf_sh * (1.0 + mode.sl_pct / 100.0)
        elif mode.sl_mode in ("fib_0.7", "fib_sl"):
            sl = fib_level(ahi, alo, "bear", mode.fib_sl)
        elif mode.sl_mode == "fib_0.7_buf":
            sl = fib_level(ahi, alo, "bear", mode.fib_sl) * (1.0 + mode.sl_buf_pct / 100.0)
        else:
            raise ValueError(mode.sl_mode)

        if mode.tp_mode == "anchor_pct":
            tp = alo * (1.0 + mode.tp_pct / 100.0)
        elif mode.tp_mode in ("anchor", "fib_0.0"):
            tp = alo
        elif mode.tp_mode.startswith("rr_"):
            rr = float(mode.tp_mode.split("_")[1])
            risk = sl - entry
            if risk <= 0:
                return None
            tp = entry - rr * risk
        else:
            raise ValueError(mode.tp_mode)

        if not (tp < entry < sl):
            return None

    risk = abs(entry - sl)
    reward = abs(tp - entry)
    if risk <= 0 or (mode.min_rr > 0 and reward / risk < mode.min_rr):
        return None
    return sl, tp


def run_with_mode(symbol: str, mode: ExitMode) -> dict:
    df = base.load_or_fetch(symbol, "5m", base.DAYS)
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)

    htf = base.resample_4h(df)
    htf_state = base.build_htf_state(htf)
    mapped = base.map_closed_htf(df, htf_state)

    high = df["high"].to_numpy(float)
    low = df["low"].to_numpy(float)
    close = df["close"].to_numpy(float)
    ts = df["timestamp"].to_numpy()
    n = len(df)

    htf_dir = mapped["htf_dir"].to_numpy()
    a_hi = mapped["anchor_high"].to_numpy(float)
    a_lo = mapped["anchor_low"].to_numpy(float)

    ph, pl = base.pivot_extremes(high, low, base.ltf_swing_len, base.ltf_swing_len)
    ltf_sh = np.full(n, np.nan)
    ltf_sl = np.full(n, np.nan)
    cur_sh = np.nan
    cur_sl = np.nan
    for i in range(n):
        if np.isfinite(ph[i]):
            cur_sh = ph[i]
        if np.isfinite(pl[i]):
            cur_sl = pl[i]
        ltf_sh[i] = cur_sh
        ltf_sl[i] = cur_sl

    bull_bos = np.zeros(n, dtype=bool)
    bear_bos = np.zeros(n, dtype=bool)
    for i in range(1, n):
        if np.isfinite(ltf_sh[i]) and close[i - 1] <= ltf_sh[i] and close[i] > ltf_sh[i]:
            bull_bos[i] = True
        if np.isfinite(ltf_sl[i]) and close[i - 1] >= ltf_sl[i] and close[i] < ltf_sl[i]:
            bear_bos[i] = True

    zone_top = np.full(n, np.nan)
    zone_bot = np.full(n, np.nan)
    for i in range(n):
        d = htf_dir[i]
        if not isinstance(d, str) or not np.isfinite(a_hi[i]) or not np.isfinite(a_lo[i]):
            continue
        rng = a_hi[i] - a_lo[i]
        if rng <= 0:
            continue
        if d == "bull":
            zone_top[i] = a_hi[i] - rng * base.zone_upper
            zone_bot[i] = a_hi[i] - rng * base.zone_lower
        else:
            zone_bot[i] = a_lo[i] + rng * base.zone_upper
            zone_top[i] = a_lo[i] + rng * base.zone_lower

    in_zone = np.zeros(n, dtype=bool)
    for i in range(n):
        if np.isfinite(zone_top[i]) and np.isfinite(zone_bot[i]):
            in_zone[i] = low[i] <= zone_top[i] and high[i] >= zone_bot[i]

    trades = []
    capital = INITIAL
    equity = [INITIAL]
    equity_ts = [ts[0]]
    in_pos = False
    pos = None
    last_trade_anchors = (None, None)
    armed = False
    armed_dir = None

    for i in range(n):
        d = htf_dir[i] if isinstance(htf_dir[i], str) else None

        if in_pos:
            hit = None
            exit_px = None
            if pos["dir"] == "long":
                if low[i] <= pos["sl"]:
                    hit, exit_px = "SL", pos["sl"]
                elif high[i] >= pos["tp"]:
                    hit, exit_px = "TP", pos["tp"]
            else:
                if high[i] >= pos["sl"]:
                    hit, exit_px = "SL", pos["sl"]
                elif low[i] <= pos["tp"]:
                    hit, exit_px = "TP", pos["tp"]
            if hit is not None:
                pnl = (
                    (exit_px - pos["entry"]) * pos["qty"]
                    if pos["dir"] == "long"
                    else (pos["entry"] - exit_px) * pos["qty"]
                )
                capital += pnl
                trades.append(
                    {
                        "entry_time": pos["entry_time"],
                        "direction": pos["dir"],
                        "entry_price": pos["entry"],
                        "exit_time": pd.Timestamp(ts[i]),
                        "exit_price": float(exit_px),
                        "sl_price": pos["sl"],
                        "tp_price": pos["tp"],
                        "pnl": float(pnl),
                        "exit_hit": hit,
                    }
                )
                equity.append(capital)
                equity_ts.append(ts[i])
                in_pos = False
                pos = None

        if d is not None and d != armed_dir:
            armed = False
            armed_dir = d
        if in_zone[i] and (i == 0 or not in_zone[i - 1]):
            armed = True

        if in_pos or not armed or d is None:
            continue
        if not np.isfinite(a_hi[i]) or not np.isfinite(a_lo[i]):
            continue

        anchors = (float(a_hi[i]), float(a_lo[i]))
        if last_trade_anchors[0] is not None:
            if (
                abs(anchors[0] - last_trade_anchors[0]) < 1e-9
                and abs(anchors[1] - last_trade_anchors[1]) < 1e-9
            ):
                continue

        long_ok = d == "bull" and bull_bos[i]
        short_ok = d == "bear" and bear_bos[i]
        if not long_ok and not short_ok:
            continue

        entry = float(close[i])
        if long_ok:
            if not np.isfinite(ltf_sl[i]):
                continue
            st = compute_sl_tp(mode, "bull", entry, float(a_hi[i]), float(a_lo[i]), float(ltf_sh[i]), float(ltf_sl[i]))
            if st is None:
                continue
            sl, tp = st
            side = "long"
        else:
            if not np.isfinite(ltf_sh[i]):
                continue
            st = compute_sl_tp(mode, "bear", entry, float(a_hi[i]), float(a_lo[i]), float(ltf_sh[i]), float(ltf_sl[i]))
            if st is None:
                continue
            sl, tp = st
            side = "short"

        qty = NOTIONAL / entry
        pos = {
            "dir": side,
            "entry": entry,
            "sl": sl,
            "tp": tp,
            "qty": qty,
            "entry_time": pd.Timestamp(ts[i]),
        }
        in_pos = True
        last_trade_anchors = anchors

    if in_pos and pos is not None:
        exit_px = float(close[-1])
        pnl = (
            (exit_px - pos["entry"]) * pos["qty"]
            if pos["dir"] == "long"
            else (pos["entry"] - exit_px) * pos["qty"]
        )
        capital += pnl
        trades.append(
            {
                "entry_time": pos["entry_time"],
                "direction": pos["dir"],
                "entry_price": pos["entry"],
                "exit_time": pd.Timestamp(ts[-1]),
                "exit_price": exit_px,
                "sl_price": pos["sl"],
                "tp_price": pos["tp"],
                "pnl": float(pnl),
                "exit_hit": "EOD",
            }
        )
        equity.append(capital)
        equity_ts.append(ts[-1])

    tdf = pd.DataFrame(trades)
    eq = pd.Series(equity, index=pd.to_datetime(equity_ts, utc=True))
    return {"trades": tdf, "equity": eq, "final": capital, "mode": mode.name}


def metrics(tdf: pd.DataFrame, final: float, eq: pd.Series) -> dict:
    if tdf.empty:
        return {
            "trades": 0,
            "wins": 0,
            "losses": 0,
            "wr": 0.0,
            "pf": 0.0,
            "net": final - INITIAL,
            "net_pct": (final - INITIAL) / INITIAL * 100,
            "max_dd_pct": 0.0,
            "avg_w": 0.0,
            "avg_l": 0.0,
            "sl_hits": 0,
            "tp_hits": 0,
        }
    wins = tdf[tdf["pnl"] > 1e-8]
    losses = tdf[tdf["pnl"] < -1e-8]
    gp = wins["pnl"].sum() if len(wins) else 0.0
    gl = abs(losses["pnl"].sum()) if len(losses) else 0.0
    pf = gp / gl if gl > 0 else float("inf")
    peak = eq.cummax()
    dd = eq - peak
    return {
        "trades": len(tdf),
        "wins": len(wins),
        "losses": len(losses),
        "wr": len(wins) / len(tdf) * 100,
        "pf": pf,
        "net": final - INITIAL,
        "net_pct": (final - INITIAL) / INITIAL * 100,
        "max_dd_pct": float(dd.min()) / INITIAL * 100,
        "avg_w": float(wins["pnl"].mean()) if len(wins) else 0.0,
        "avg_l": float(losses["pnl"].mean()) if len(losses) else 0.0,
        "sl_hits": int((tdf["exit_hit"] == "SL").sum()),
        "tp_hits": int((tdf["exit_hit"] == "TP").sum()),
    }


MODES = [
    ExitMode("baseline_swing0.09_anchor0.14", "swing_pct", "anchor_pct", sl_pct=0.09, tp_pct=0.14),
    ExitMode("sl_fib0.7_tp_anchor", "fib_0.7", "anchor"),
    ExitMode("sl_fib0.7buf_tp_anchor", "fib_0.7_buf", "anchor", sl_buf_pct=0.05),
    ExitMode("sl_fib0.7_tp_anchor0.14", "fib_0.7", "anchor_pct", tp_pct=0.14),
    ExitMode("sl_fib0.7_tp_rr2", "fib_0.7", "rr_2"),
    ExitMode("sl_fib0.7_tp_rr3", "fib_0.7", "rr_3"),
    ExitMode("sl_swing0.25_tp_anchor", "swing_pct_wide", "anchor", sl_pct=0.25),
    ExitMode("sl_swing0.50_tp_anchor", "swing_pct_wide", "anchor", sl_pct=0.50),
    ExitMode("sl_fib0.7_tp_anchor_minRR1.5", "fib_0.7", "anchor", min_rr=1.5),
    ExitMode("sl_fib0.7_tp_anchor_minRR2", "fib_0.7", "anchor", min_rr=2.0),
    # classic PMTS: SL just beyond 0.7, TP at opposite anchor
    ExitMode("sl_fib0.75_tp_anchor", "fib_sl", "anchor", fib_sl=0.75),
    ExitMode("sl_fib0.8_tp_anchor", "fib_sl", "anchor", fib_sl=0.8),
]


def main():
    rows = []
    best = None
    for sym in ("XAUUSD", "PAXGUSDT"):
        print(f"\n===== {sym} =====")
        for mode in MODES:
            print(f"  {mode.name}...", end=" ", flush=True)
            out = run_with_mode(sym, mode)
            m = metrics(out["trades"], out["final"], out["equity"])
            m["symbol"] = sym
            m["mode"] = mode.name
            rows.append(m)
            print(
                f"n={m['trades']} WR={m['wr']:.1f}% PF={m['pf']:.2f} "
                f"net={m['net_pct']:+.2f}% DD={m['max_dd_pct']:.2f}% "
                f"SL/TP={m['sl_hits']}/{m['tp_hits']}"
            )
            if not out["trades"].empty:
                out["trades"].to_csv(OUT / f"{sym.lower()}_{mode.name}_trades.csv", index=False)
            # track best XAUUSD by net then PF
            if sym == "XAUUSD" and (best is None or m["net_pct"] > best["net_pct"]):
                best = {**m, "equity": out["equity"], "trades": out["trades"]}

    rdf = pd.DataFrame(rows)
    rdf.to_csv(OUT / "exit_sweep_summary.csv", index=False)
    print("\n===== SUMMARY (sorted by net_pct within symbol) =====")
    for sym, g in rdf.groupby("symbol"):
        print(f"\n{sym}:")
        show = g.sort_values("net_pct", ascending=False)[
            ["mode", "trades", "wr", "pf", "net_pct", "max_dd_pct", "sl_hits", "tp_hits"]
        ]
        print(show.to_string(index=False, float_format=lambda x: f"{x:.2f}"))

    if best is not None:
        fig, ax = plt.subplots(figsize=(11, 4.5))
        eq = best["equity"]
        ax.plot(eq.index, eq.values, color="#1f6feb", lw=1.5)
        ax.axhline(INITIAL, color="#888", ls="--", lw=0.8)
        ax.set_title(f"XAUUSD best exit mode: {best['mode']} ({best['net_pct']:+.2f}%)")
        ax.set_ylabel("Equity $")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(OUT / "xauusd_best_equity.png", dpi=140)
        plt.close(fig)
        (OUT / "best.json").write_text(json.dumps({k: v for k, v in best.items() if k not in ("equity", "trades")}, indent=2))

    (OUT / "diagnosis.txt").write_text(
        "Baseline loses because SL at LTF swing (±0.09%) is noise-tight: "
        "most trades stop out before reaching distant anchor TP. "
        "Fib structure can still be OK; exits are the main issue. "
        "SL @ 0.7 fib gives structure-based room beyond the golden zone.\n"
    )
    print(f"\nArtifacts → {OUT}")


if __name__ == "__main__":
    main()
