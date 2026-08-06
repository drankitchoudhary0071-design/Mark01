"""Performance metrics for backtest results."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .backtest_engine import BacktestResult, Trade


def _trade_frame(trades: list[Trade]) -> pd.DataFrame:
    if not trades:
        return pd.DataFrame(
            columns=[
                "entry_time",
                "exit_time",
                "direction",
                "pnl",
                "pnl_pct",
                "return_R",
                "exit_reason",
                "pattern",
            ]
        )
    return pd.DataFrame(
        [
            {
                "entry_time": t.entry_time,
                "exit_time": t.exit_time,
                "direction": t.direction,
                "entry_price": t.entry_price,
                "exit_price": t.exit_price,
                "stop": t.stop,
                "take_profit": t.take_profit,
                "pnl": t.pnl,
                "pnl_pct": t.pnl_pct,
                "return_R": t.return_R,
                "exit_reason": t.exit_reason,
                "pattern": t.pattern,
            }
            for t in trades
        ]
    )


def compute_metrics(result: BacktestResult, periods_per_year: float = 365.25 * 24 * 12) -> dict:
    """
    periods_per_year default assumes LTF bars if we resample equity —
    we instead compute Sharpe from trade returns and from daily equity.
    """
    trades = result.trades
    tf = _trade_frame(trades)
    initial = result.initial_capital
    final = result.final_capital
    total_return = (final / initial) - 1.0 if initial else 0.0

    n = len(trades)
    if n == 0:
        return {
            "total_trades": 0,
            "win_rate": 0.0,
            "profit_factor": 0.0,
            "avg_rr_realized": 0.0,
            "avg_win_R": 0.0,
            "avg_loss_R": 0.0,
            "max_drawdown": 0.0,
            "sharpe_daily": 0.0,
            "total_return": 0.0,
            "final_equity": final,
            "initial_equity": initial,
            "avg_pnl": 0.0,
            "expectancy_R": 0.0,
            "long_trades": 0,
            "short_trades": 0,
            "stop_exits": 0,
            "tp_exits": 0,
        }

    wins = tf[tf["pnl"] > 0]
    losses = tf[tf["pnl"] <= 0]
    win_rate = len(wins) / n
    gross_profit = wins["pnl"].sum() if len(wins) else 0.0
    gross_loss = abs(losses["pnl"].sum()) if len(losses) else 0.0
    profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else (np.inf if gross_profit > 0 else 0.0)

    # Equity curve drawdown
    eq = result.equity.sort_index()
    # Resample to daily for Sharpe
    daily = eq.resample("1D").last().ffill()
    daily_ret = daily.pct_change().dropna()
    if len(daily_ret) > 1 and daily_ret.std() > 0:
        sharpe = float(np.sqrt(365.25) * daily_ret.mean() / daily_ret.std())
    else:
        sharpe = 0.0

    peak = eq.cummax()
    dd = (eq - peak) / peak
    max_dd = float(dd.min()) if len(dd) else 0.0

    avg_rr = float(tf["return_R"].mean())
    avg_win_R = float(wins["return_R"].mean()) if len(wins) else 0.0
    avg_loss_R = float(losses["return_R"].mean()) if len(losses) else 0.0

    return {
        "total_trades": n,
        "win_rate": float(win_rate),
        "profit_factor": float(profit_factor) if np.isfinite(profit_factor) else 999.0,
        "avg_rr_realized": avg_rr,
        "avg_win_R": avg_win_R,
        "avg_loss_R": avg_loss_R,
        "max_drawdown": max_dd,
        "sharpe_daily": sharpe,
        "total_return": float(total_return),
        "final_equity": float(final),
        "initial_equity": float(initial),
        "avg_pnl": float(tf["pnl"].mean()),
        "expectancy_R": avg_rr,
        "long_trades": int((tf["direction"] == "long").sum()),
        "short_trades": int((tf["direction"] == "short").sum()),
        "stop_exits": int((tf["exit_reason"] == "stop").sum()),
        "tp_exits": int((tf["exit_reason"] == "take_profit").sum()),
    }


def format_metrics(m: dict, title: str = "") -> str:
    lines = []
    if title:
        lines.append(f"=== {title} ===")
    lines.extend(
        [
            f"Total trades     : {m['total_trades']}",
            f"Win rate         : {m['win_rate']*100:.1f}%",
            f"Profit factor    : {m['profit_factor']:.3f}",
            f"Avg realized R   : {m['avg_rr_realized']:.3f}",
            f"Avg win R / loss : {m['avg_win_R']:.3f} / {m['avg_loss_R']:.3f}",
            f"Max drawdown     : {m['max_drawdown']*100:.2f}%",
            f"Sharpe (daily)   : {m['sharpe_daily']:.3f}",
            f"Total return     : {m['total_return']*100:.2f}%",
            f"Final equity     : ${m['final_equity']:,.2f} (from ${m['initial_equity']:,.2f})",
            f"Long/Short       : {m['long_trades']} / {m['short_trades']}",
            f"TP / Stop exits  : {m['tp_exits']} / {m['stop_exits']}",
        ]
    )
    return "\n".join(lines)


def overfitting_assessment(is_m: dict, oos_m: dict) -> str:
    """Honest comparison of in-sample vs out-of-sample."""
    lines = ["=== Overfitting assessment ==="]
    if is_m["total_trades"] < 15:
        lines.append(
            "WARNING: In-sample trade count is low (<15). Statistical confidence is weak; "
            "any 'edge' may be noise."
        )
    if oos_m["total_trades"] < 5:
        lines.append(
            "WARNING: Out-of-sample trade count is very low (<5). OOS results are not reliable."
        )

    is_ret = is_m["total_return"]
    oos_ret = oos_m["total_return"]
    is_wr = is_m["win_rate"]
    oos_wr = oos_m["win_rate"]
    is_pf = is_m["profit_factor"]
    oos_pf = oos_m["profit_factor"]
    is_sh = is_m["sharpe_daily"]
    oos_sh = oos_m["sharpe_daily"]

    lines.append(f"IS  return={is_ret*100:.2f}%  WR={is_wr*100:.1f}%  PF={is_pf:.2f}  Sharpe={is_sh:.2f}")
    lines.append(f"OOS return={oos_ret*100:.2f}%  WR={oos_wr*100:.1f}%  PF={oos_pf:.2f}  Sharpe={oos_sh:.2f}")

    # Heuristics
    suspect = False
    reasons = []
    if is_ret > 0.05 and oos_ret < -0.02:
        suspect = True
        reasons.append("IS profitable but OOS losing — classic overfit signature.")
    if is_wr - oos_wr > 0.20 and is_m["total_trades"] >= 15:
        suspect = True
        reasons.append("Win rate drops >20pp from IS to OOS.")
    if is_pf > 1.3 and oos_pf < 0.9 and oos_m["total_trades"] >= 5:
        suspect = True
        reasons.append("Profit factor collapses below 1.0 out of sample.")
    if is_sh > 0.8 and oos_sh < 0:
        suspect = True
        reasons.append("Sharpe flips from solid positive IS to negative OOS.")
    if is_ret > 0 and oos_ret > 0 and abs(is_ret - oos_ret) < 0.15 and abs(is_wr - oos_wr) < 0.15:
        lines.append(
            "VERDICT: No strong overfitting signal — IS and OOS are directionally consistent "
            "(still treat small samples with caution)."
        )
    elif suspect:
        lines.append("VERDICT: SUSPECT OVERFITTING — " + " ".join(reasons))
    elif is_ret <= 0 and oos_ret <= 0:
        lines.append(
            "VERDICT: Strategy appears weak on both IS and OOS (not overfit — just no edge under "
            "these costs/rules)."
        )
    else:
        lines.append(
            "VERDICT: Mixed / inconclusive. Performance gap exists but does not cleanly match "
            "a textbook overfit pattern; sample size and regime shift may both matter."
        )
        if reasons:
            lines.append("Notes: " + " ".join(reasons))
    return "\n".join(lines)
