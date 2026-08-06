"""
Run High Win-Rate Gold Momentum strategy (1:2 RR, target WR >= 50%).
Produces IS/OOS metrics, equity curve, example trades, and comparison vs PMTS.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT.parent))

from gold_backtest.src.fetch_data import DATA_DIR, load_csv
from gold_backtest.src.strategy_high_wr import generate_high_wr_signals, is_tuned_params
from gold_backtest.src.strategy_pmts import PMTSParams, generate_signals, improved_params
from gold_backtest.src.backtest_engine import (
    CostModel,
    run_backtest,
    split_is_oos,
    filter_signals_period,
    filter_df_period,
)
from gold_backtest.src.metrics import (
    compute_metrics,
    format_metrics,
    overfitting_assessment,
    _trade_frame,
)
from gold_backtest.src.visualization import (
    plot_equity,
    plot_example_trade,
    plot_comparison_table_image,
    CHARTS,
    RESULTS,
)


def metrics_row(name: str, period: str, m: dict) -> dict:
    return {
        "Strategy": name,
        "Period": period,
        "Trades": m["total_trades"],
        "Win%": f"{m['win_rate']*100:.1f}",
        "PF": f"{m['profit_factor']:.2f}",
        "Avg R": f"{m['avg_rr_realized']:.2f}",
        "MaxDD%": f"{m['max_drawdown']*100:.1f}",
        "Sharpe": f"{m['sharpe_daily']:.2f}",
        "Return%": f"{m['total_return']*100:.1f}",
    }


def run() -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    CHARTS.mkdir(parents=True, exist_ok=True)

    df_1h = load_csv(DATA_DIR / "paxgusdt_1h.csv")
    df_15m = load_csv(DATA_DIR / "paxgusdt_15m.csv")
    start, is_end, end = split_is_oos(df_15m, 10 / 12)
    costs = CostModel()
    params = is_tuned_params()

    print("=== High-WR Gold Momentum (fixed RR 1:2) ===")
    print(f"Params: {params}")
    print(f"Split IS end: {is_end}")

    sigs = generate_high_wr_signals(df_1h, df_15m, params)
    sigs_is = filter_signals_period(sigs, start, is_end)
    sigs_oos = filter_signals_period(sigs, is_end, end)
    df_is = filter_df_period(df_15m, start, is_end)
    df_oos = filter_df_period(df_15m, is_end, end)

    res_is = run_backtest(df_is, sigs_is, costs=costs, params_note="HWR_IS")
    res_oos = run_backtest(df_oos, sigs_oos, costs=costs, params_note="HWR_OOS")
    m_is = compute_metrics(res_is)
    m_oos = compute_metrics(res_oos)

    report = []
    report.append(format_metrics(m_is, "HIGH-WR MOMENTUM — IN-SAMPLE"))
    report.append("")
    report.append(format_metrics(m_oos, "HIGH-WR MOMENTUM — OUT-OF-SAMPLE"))
    report.append("")
    report.append(overfitting_assessment(m_is, m_oos))
    report.append("")
    report.append(
        "Strategy notes:\n"
        "- Long-only 1H trend (SMA50>SMA100) + ADX>35 + 15m higher-low BOS + RSI 50-70\n"
        "- Fixed RR 1:2; min stop 1.0% so ~0.30% costs do not dominate R\n"
        "- IS-tuned: adx_min=35, min_stop=1%, swing=8, cooldown=8\n"
        "- Caveat: long-only edge leans on bullish gold regime in this sample"
    )

    # Compare vs PMTS improved for context
    imp = improved_params()
    pmts_sigs = generate_signals(df_1h, df_15m, imp)
    pmts_is = run_backtest(df_is, filter_signals_period(pmts_sigs, start, is_end), costs=costs)
    pmts_oos = run_backtest(df_oos, filter_signals_period(pmts_sigs, is_end, end), costs=costs)
    mp_is = compute_metrics(pmts_is)
    mp_oos = compute_metrics(pmts_oos)

    rows = [
        metrics_row("High-WR Momentum 1:2", "IS", m_is),
        metrics_row("High-WR Momentum 1:2", "OOS", m_oos),
        metrics_row("PMTS improved", "IS", mp_is),
        metrics_row("PMTS improved", "OOS", mp_oos),
    ]
    summary = pd.DataFrame(rows)
    report.append("\n=== Comparison ===\n" + summary.to_string(index=False))
    print("\n".join(report))

    (RESULTS / "high_wr_report.txt").write_text("\n".join(report))
    summary.to_csv(RESULTS / "high_wr_summary.csv", index=False)
    _trade_frame(res_is.trades).to_csv(RESULTS / "high_wr_is_trades.csv", index=False)
    _trade_frame(res_oos.trades).to_csv(RESULTS / "high_wr_oos_trades.csv", index=False)
    (RESULTS / "high_wr_metrics.json").write_text(
        json.dumps(
            {"params": params.__dict__, "is": m_is, "oos": m_oos, "split": {"is_end": str(is_end)}},
            indent=2,
            default=str,
        )
    )

    res_full = run_backtest(df_15m, filter_signals_period(sigs, start, end), costs=costs)
    plot_equity(
        {"High-WR 1:2": res_full, "PMTS improved": run_backtest(df_15m, filter_signals_period(pmts_sigs, start, end), costs=costs)},
        "High-WR Momentum vs PMTS (normalized equity)",
        CHARTS / "high_wr_equity.png",
        is_end=is_end,
    )
    plot_comparison_table_image(rows, CHARTS / "high_wr_summary.png", "High-WR 1:2 vs PMTS")

    # Example trades
    trades = res_is.trades
    wins = [t for t in trades if t.pnl > 0]
    losses = [t for t in trades if t.pnl <= 0]
    examples = []
    if wins:
        examples.append(sorted(wins, key=lambda t: -t.pnl)[0])
    if losses:
        examples.append(sorted(losses, key=lambda t: t.pnl)[0])
    for t in trades:
        if t not in examples:
            examples.append(t)
        if len(examples) >= 3:
            break
    sig_by_ts = {s.timestamp: s for s in sigs_is}
    for i, tr in enumerate(examples, 1):
        sig = sig_by_ts.get(tr.entry_time) or (
            min(sigs_is, key=lambda s: abs(s.timestamp - tr.entry_time)) if sigs_is else None
        )
        plot_example_trade(df_15m, tr, sig, CHARTS / f"high_wr_example_trade_{i}.png")

    print(f"\nWrote {RESULTS / 'high_wr_report.txt'}")
    print(f"Charts in {CHARTS}")

    if m_is["win_rate"] < 0.50:
        print("WARNING: IS win rate below 50% target.")
    if m_oos["total_trades"] < 10:
        print("NOTE: OOS trade count is small — treat OOS WR with caution.")


if __name__ == "__main__":
    run()
