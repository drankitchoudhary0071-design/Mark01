"""Performance metrics — identical definition for every strategy."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd

from strategy_lab.engine.types import BacktestResult, Trade


def trades_to_frame(trades: list[Trade]) -> pd.DataFrame:
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
                "strategy",
                "symbol",
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
                "size": t.size,
                "pnl": t.pnl,
                "pnl_pct": t.pnl_pct,
                "return_R": t.return_R,
                "exit_reason": t.exit_reason,
                "pattern": t.pattern,
                "strategy": t.strategy,
                "symbol": t.symbol,
            }
            for t in trades
        ]
    )


def _max_drawdown(equity: pd.Series) -> float:
    if equity is None or len(equity) == 0:
        return 0.0
    eq = equity.sort_index().astype(float)
    peak = eq.cummax()
    dd = (eq - peak) / peak.replace(0, np.nan)
    return float(dd.min()) if len(dd) else 0.0


def _sharpe_daily(equity: pd.Series) -> float:
    if equity is None or len(equity) < 3:
        return 0.0
    eq = equity.sort_index().astype(float)
    daily = eq.resample("1D").last().ffill()
    rets = daily.pct_change().dropna()
    if len(rets) < 2 or rets.std() == 0 or np.isnan(rets.std()):
        return 0.0
    return float(np.sqrt(365.25) * rets.mean() / rets.std())


def compute_metrics(result: BacktestResult) -> dict[str, Any]:
    """
    Key metrics (required by spec):
      win_rate, expectancy (R), max_drawdown, sharpe_ratio, profit_factor
    Plus extras for reports.
    """
    trades = result.trades
    initial = result.initial_capital
    final = result.final_capital
    total_return = (final / initial) - 1.0 if initial else 0.0
    max_dd = _max_drawdown(result.equity)
    sharpe = _sharpe_daily(result.equity)

    n = len(trades)
    empty = {
        "strategy": result.strategy,
        "symbol": result.symbol,
        "total_trades": 0,
        "win_rate": 0.0,
        "expectancy_R": 0.0,
        "expectancy_pnl": 0.0,
        "max_drawdown": max_dd,
        "sharpe_ratio": sharpe,
        "profit_factor": 0.0,
        "total_return": total_return,
        "final_equity": final,
        "initial_equity": initial,
        "avg_win_R": 0.0,
        "avg_loss_R": 0.0,
        "long_trades": 0,
        "short_trades": 0,
        "stop_exits": 0,
        "tp_exits": 0,
        "params": result.params,
        "curve_fit_flags": result.curve_fit_flags,
    }
    if n == 0:
        return empty

    tf = trades_to_frame(trades)
    wins = tf[tf["pnl"] > 0]
    losses = tf[tf["pnl"] <= 0]
    win_rate = len(wins) / n
    gross_profit = float(wins["pnl"].sum()) if len(wins) else 0.0
    gross_loss = float(abs(losses["pnl"].sum())) if len(losses) else 0.0
    if gross_loss > 0:
        profit_factor = gross_profit / gross_loss
    elif gross_profit > 0:
        profit_factor = math.inf
    else:
        profit_factor = 0.0

    expectancy_R = float(tf["return_R"].mean())
    expectancy_pnl = float(tf["pnl"].mean())
    avg_win_R = float(wins["return_R"].mean()) if len(wins) else 0.0
    avg_loss_R = float(losses["return_R"].mean()) if len(losses) else 0.0

    return {
        "strategy": result.strategy,
        "symbol": result.symbol,
        "total_trades": n,
        "win_rate": win_rate,
        "expectancy_R": expectancy_R,
        "expectancy_pnl": expectancy_pnl,
        "max_drawdown": max_dd,
        "sharpe_ratio": sharpe,
        "profit_factor": float(profit_factor) if math.isfinite(profit_factor) else 999.0,
        "total_return": total_return,
        "final_equity": final,
        "initial_equity": initial,
        "avg_win_R": avg_win_R,
        "avg_loss_R": avg_loss_R,
        "long_trades": int((tf["direction"] == "long").sum()),
        "short_trades": int((tf["direction"] == "short").sum()),
        "stop_exits": int((tf["exit_reason"] == "stop").sum()),
        "tp_exits": int((tf["exit_reason"] == "take_profit").sum()),
        "params": result.params,
        "curve_fit_flags": result.curve_fit_flags,
    }


def format_metrics(m: dict[str, Any], title: str = "") -> str:
    lines = []
    if title:
        lines.append(f"=== {title} ===")
    pf = m["profit_factor"]
    pf_s = "inf" if pf >= 998 else f"{pf:.3f}"
    lines.extend(
        [
            f"Strategy         : {m.get('strategy', '')}",
            f"Symbol           : {m.get('symbol', '')}",
            f"Total trades     : {m['total_trades']}",
            f"Win rate         : {m['win_rate'] * 100:.1f}%",
            f"Expectancy (R)   : {m['expectancy_R']:.3f}",
            f"Expectancy (pnl) : ${m['expectancy_pnl']:.2f}",
            f"Max drawdown     : {m['max_drawdown'] * 100:.2f}%",
            f"Sharpe (daily)   : {m['sharpe_ratio']:.3f}",
            f"Profit factor    : {pf_s}",
            f"Total return     : {m['total_return'] * 100:.2f}%",
            f"Final equity     : ${m['final_equity']:,.2f} (from ${m['initial_equity']:,.2f})",
            f"Long / Short     : {m['long_trades']} / {m['short_trades']}",
            f"TP / Stop exits  : {m['tp_exits']} / {m['stop_exits']}",
        ]
    )
    flags = m.get("curve_fit_flags") or []
    if flags:
        lines.append("Curve-fit flags:")
        for f in flags:
            lines.append(f"  - {f}")
    return "\n".join(lines)


def overfitting_assessment(is_m: dict, oos_m: dict) -> str:
    lines = ["=== Overfitting assessment (IS vs OOS) ==="]
    if is_m["total_trades"] < 15:
        lines.append("WARNING: IS trade count < 15 — weak statistical confidence.")
    if oos_m["total_trades"] < 5:
        lines.append("WARNING: OOS trade count < 5 — OOS result unreliable.")

    lines.append(
        f"IS  ret={is_m['total_return']*100:.2f}% WR={is_m['win_rate']*100:.1f}% "
        f"PF={is_m['profit_factor']:.2f} Sharpe={is_m['sharpe_ratio']:.2f} "
        f"ExpR={is_m['expectancy_R']:.3f}"
    )
    lines.append(
        f"OOS ret={oos_m['total_return']*100:.2f}% WR={oos_m['win_rate']*100:.1f}% "
        f"PF={oos_m['profit_factor']:.2f} Sharpe={oos_m['sharpe_ratio']:.2f} "
        f"ExpR={oos_m['expectancy_R']:.3f}"
    )

    suspect = False
    reasons: list[str] = []
    if is_m["total_return"] > 0.05 and oos_m["total_return"] < -0.02:
        suspect = True
        reasons.append("IS profitable but OOS losing.")
    if is_m["win_rate"] - oos_m["win_rate"] > 0.20 and is_m["total_trades"] >= 15:
        suspect = True
        reasons.append("Win rate drops >20pp IS→OOS.")
    if is_m["profit_factor"] > 1.3 and oos_m["profit_factor"] < 0.9 and oos_m["total_trades"] >= 5:
        suspect = True
        reasons.append("Profit factor collapses below 1.0 OOS.")
    if is_m["sharpe_ratio"] > 0.8 and oos_m["sharpe_ratio"] < 0:
        suspect = True
        reasons.append("Sharpe flips positive IS → negative OOS.")

    if (
        is_m["total_return"] > 0
        and oos_m["total_return"] > 0
        and abs(is_m["total_return"] - oos_m["total_return"]) < 0.15
        and abs(is_m["win_rate"] - oos_m["win_rate"]) < 0.15
    ):
        lines.append(
            "VERDICT: No strong overfitting signal — IS/OOS directionally consistent."
        )
    elif suspect:
        lines.append("VERDICT: SUSPECT OVERFITTING — " + " ".join(reasons))
    elif is_m["total_return"] <= 0 and oos_m["total_return"] <= 0:
        lines.append("VERDICT: Weak on both IS and OOS (no edge under these costs).")
    else:
        lines.append("VERDICT: Mixed / inconclusive.")
        if reasons:
            lines.append("Notes: " + " ".join(reasons))
    return "\n".join(lines)
