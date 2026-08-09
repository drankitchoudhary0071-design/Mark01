#!/usr/bin/env python3
"""Sweep partial profit target (pips) for EMA Pocket Scalp — compounding + slippage."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from strategy_lab.data.fetch import load_or_fetch
from strategy_lab.run_ema_pocket_scalp import (
    DAYS,
    INITIAL_CAPITAL,
    aggregate_trades,
    run_backtest,
)

TARGETS = [10, 15, 20, 25, 30, 40, 50, 75, 100, 150, 200]
OUT = Path(__file__).resolve().parent / "results" / "ema_pocket_scalp"
FLAT = Path(__file__).resolve().parent / "results"
OUT.mkdir(parents=True, exist_ok=True)


def sweep_symbol_tf(symbol: str, tf: str, df: pd.DataFrame) -> list[dict]:
    rows = []
    for tp in TARGETS:
        out = run_backtest(symbol, tf, partial_pips=tp, df=df)
        agg = aggregate_trades(out["trades"])
        eq = out["equity"]
        n = len(agg)
        partial_hits = int(out["trades"]["exit_hit"].str.startswith("partial").sum()) if not out["trades"].empty else 0
        rows.append(
            {
                "symbol": symbol,
                "timeframe": tf,
                "partial_pips": tp,
                "trades": n,
                "wins": int((agg.pnl > 0).sum()) if n else 0,
                "wr": float((agg.pnl > 0).mean() * 100) if n else 0.0,
                "pf": (
                    float(agg.loc[agg.pnl > 0, "pnl"].sum() / abs(agg.loc[agg.pnl < 0, "pnl"].sum()))
                    if n and (agg.pnl < 0).any() and (agg.pnl > 0).any()
                    else float("nan")
                ),
                "net": out["final"] - INITIAL_CAPITAL,
                "net_pct": (out["final"] - INITIAL_CAPITAL) / INITIAL_CAPITAL * 100,
                "final_capital": out["final"],
                "max_dd_pct": float((eq - eq.cummax()).min() / INITIAL_CAPITAL * 100) if len(eq) else 0.0,
                "partial_legs": partial_hits,
                "partial_hit_rate": partial_hits / max(len(out["trades"]), 1) * 100,
                "fees": out["total_fees"],
            }
        )
        print(
            f"  {symbol} {tf} target={tp:3d}p → net {rows[-1]['net_pct']:+.3f}% "
            f"WR={rows[-1]['wr']:.1f}% PF={rows[-1]['pf']:.2f} partials={partial_hits}"
        )
    return rows


def plot_sweep(sub: pd.DataFrame, symbol: str, tf: str, path: Path):
    fig, ax1 = plt.subplots(figsize=(10, 4.5))
    ax1.plot(sub.partial_pips, sub.net_pct, "o-", color="#1f6feb", lw=2, label="Net %")
    ax1.axhline(0, color="#888", ls="--", lw=0.8)
    ax1.set_xlabel("Partial target (pips)")
    ax1.set_ylabel("Net return %", color="#1f6feb")
    ax1.tick_params(axis="y", labelcolor="#1f6feb")
    ax1.grid(True, alpha=0.3)

    ax2 = ax1.twinx()
    ax2.plot(sub.partial_pips, sub.wr, "s--", color="#cf222e", lw=1.2, alpha=0.8, label="Win rate %")
    ax2.set_ylabel("Win rate %", color="#cf222e")
    ax2.tick_params(axis="y", labelcolor="#cf222e")

    best = sub.loc[sub.net_pct.idxmax()]
    ax1.axvline(best.partial_pips, color="#2da44e", ls=":", lw=1.2)
    ax1.set_title(f"{symbol} {tf} — target sweep (best={int(best.partial_pips)}p → {best.net_pct:+.2f}%)")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def main():
    print(f"EMA Pocket — partial target sweep {TARGETS} pips (5% compound + slippage)")
    all_rows: list[dict] = []
    best_rows: list[dict] = []

    for sym in ("XAUUSD", "PAXGUSDT"):
        for tf in ("1m", "5m"):
            print(f"\n>>> {sym} {tf}")
            df = load_or_fetch(sym, tf, DAYS)
            rows = sweep_symbol_tf(sym, tf, df)
            all_rows.extend(rows)
            sub = pd.DataFrame(rows)
            stem = f"{sym.lower()}_{tf}"
            plot_sweep(sub, sym, tf, OUT / f"target_sweep_{stem}.png")
            plot_sweep(sub, sym, tf, FLAT / f"ema_pocket_target_sweep_{stem}.png")
            best = sub.loc[sub.net_pct.idxmax()]
            best_rows.append(best.to_dict())

    rdf = pd.DataFrame(all_rows)
    bdf = pd.DataFrame(best_rows)
    rdf.to_csv(OUT / "target_sweep_all.csv", index=False)
    rdf.to_csv(FLAT / "ema_pocket_target_sweep_all.csv", index=False)
    bdf.to_csv(OUT / "target_sweep_best.csv", index=False)
    bdf.to_csv(FLAT / "ema_pocket_target_sweep_best.csv", index=False)

    print("\n" + "=" * 80)
    print("BEST TARGET PER SYMBOL / TIMEFRAME")
    print("=" * 80)
    cols = ["symbol", "timeframe", "partial_pips", "trades", "wr", "pf", "net_pct", "final_capital", "max_dd_pct", "partial_legs"]
    print(bdf[cols].to_string(index=False))
    print(f"\nArtifacts → {OUT}")


if __name__ == "__main__":
    main()
