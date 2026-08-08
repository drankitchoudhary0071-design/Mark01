"""
Shared event-driven backtest engine.

Why a custom engine (not backtrader/vectorbt as the sole runner)?
- All 10 families must be comparable on identical fill/cost/risk semantics.
- Some strategies (grid, market-making, pairs) need multi-leg / inventory state
  that a single vectorbt portfolio template obscures.
- Signal modules stay pure (pandas); this engine owns fills and accounting.

Strategies that are naturally vectorized still *generate* signals with
pandas/numpy; execution always goes through here (or a thin specialized
simulator that reports the same Trade / BacktestResult types).
"""

from __future__ import annotations

from typing import Optional

import pandas as pd

from strategy_lab import config as cfg
from strategy_lab.engine.costs import CostModel, apply_entry_price, apply_exit_price
from strategy_lab.engine.types import BacktestResult, Signal, Trade


def run_backtest(
    df: pd.DataFrame,
    signals: list[Signal],
    *,
    initial_capital: float = cfg.INITIAL_CAPITAL,
    risk_per_trade: float = cfg.RISK_PER_TRADE,
    costs: Optional[CostModel] = None,
    max_hold_bars: int = 24 * 10,
    strategy_name: str = "",
    symbol: str = "",
    params: Optional[dict] = None,
    curve_fit_flags: Optional[list[str]] = None,
    allow_pyramiding: bool = False,
    one_trade_at_a_time: bool = True,
) -> BacktestResult:
    """
    Walk bars chronologically.

    Fill model:
    - Entry at signal bar close ± (half_spread + slippage)
    - Stop / TP checked on subsequent bars via high/low
    - If stop and TP both touched in one bar → assume stop first (conservative)
    - Commission charged on both sides
    """
    costs = costs or CostModel()
    params = params or {}
    curve_fit_flags = curve_fit_flags or []

    if df.empty:
        eq = pd.Series([initial_capital], index=pd.DatetimeIndex([pd.Timestamp.utcnow()]))
        return BacktestResult([], eq, initial_capital, initial_capital, strategy_name, symbol, params, "", curve_fit_flags)

    data = df.reset_index(drop=True)
    capital = float(initial_capital)
    equity_pts: list[tuple[pd.Timestamp, float]] = [
        (pd.Timestamp(data.at[0, "timestamp"]), capital)
    ]
    trades: list[Trade] = []
    open_trade: dict | None = None

    signal_q = sorted(signals, key=lambda s: s.timestamp)
    sig_idx = 0

    for i in range(len(data)):
        ts = pd.Timestamp(data.at[i, "timestamp"])
        h = float(data.at[i, "high"])
        l = float(data.at[i, "low"])
        c = float(data.at[i, "close"])

        if open_trade is not None:
            open_trade["bars"] += 1
            direction = open_trade["direction"]
            stop = open_trade["stop"]
            tp = open_trade["take_profit"]
            exit_px = None
            reason = None
            hold_limit = open_trade.get("max_hold_bars") or max_hold_bars

            if direction == "long":
                if l <= stop:
                    exit_px, reason = stop, "stop"
                elif h >= tp:
                    exit_px, reason = tp, "take_profit"
            else:
                if h >= stop:
                    exit_px, reason = stop, "stop"
                elif l <= tp:
                    exit_px, reason = tp, "take_profit"

            # Optional soft exit price stamped by strategy in meta
            soft = open_trade.get("soft_exits", {})
            if exit_px is None and ts in soft:
                exit_px, reason = soft[ts], "signal_exit"

            if exit_px is None and open_trade["bars"] >= hold_limit:
                exit_px, reason = c, "time_exit"

            if exit_px is not None:
                fill = apply_exit_price(float(exit_px), direction, costs)
                entry = open_trade["entry_fill"]
                size = open_trade["size"]
                if direction == "long":
                    gross = (fill - entry) * size
                else:
                    gross = (entry - fill) * size
                fees = (entry + fill) * size * costs.commission_rate
                pnl = gross - fees
                risk = open_trade["risk_cash"]
                capital += pnl
                trades.append(
                    Trade(
                        entry_time=open_trade["entry_time"],
                        exit_time=ts,
                        direction=direction,
                        entry_price=entry,
                        exit_price=fill,
                        stop=stop,
                        take_profit=tp,
                        size=size,
                        pnl=pnl,
                        pnl_pct=pnl / open_trade["capital_at_entry"]
                        if open_trade["capital_at_entry"]
                        else 0.0,
                        return_R=pnl / risk if risk else 0.0,
                        exit_reason=reason or "exit",
                        pattern=open_trade.get("pattern", ""),
                        symbol=symbol,
                        strategy=strategy_name,
                        meta=open_trade.get("meta", {}),
                    )
                )
                equity_pts.append((ts, capital))
                open_trade = None

        # Entries
        can_enter = open_trade is None if one_trade_at_a_time else True
        if can_enter and not (allow_pyramiding is False and open_trade is not None):
            while sig_idx < len(signal_q) and signal_q[sig_idx].timestamp < ts:
                sig_idx += 1
            if sig_idx < len(signal_q) and signal_q[sig_idx].timestamp == ts:
                sig = signal_q[sig_idx]
                sig_idx += 1
                if open_trade is not None and one_trade_at_a_time:
                    continue
                entry_fill = apply_entry_price(sig.entry, sig.direction, costs)
                if sig.direction == "long":
                    risk_per_unit = entry_fill - sig.stop
                else:
                    risk_per_unit = sig.stop - entry_fill
                if risk_per_unit <= 0:
                    continue
                risk_cash = capital * risk_per_trade
                size = risk_cash / risk_per_unit
                max_size = (capital * cfg.MAX_NOTIONAL_FRAC) / entry_fill if entry_fill else 0.0
                size = min(size, max_size)
                if size <= 0 or capital <= 0:
                    continue
                soft_exits = {}
                if "soft_exits" in sig.meta:
                    soft_exits = sig.meta["soft_exits"]
                open_trade = {
                    "direction": sig.direction,
                    "entry_fill": entry_fill,
                    "entry_time": ts,
                    "stop": sig.stop,
                    "take_profit": sig.take_profit,
                    "size": size,
                    "risk_cash": risk_cash,
                    "capital_at_entry": capital,
                    "bars": 0,
                    "pattern": sig.pattern,
                    "meta": sig.meta,
                    "soft_exits": soft_exits,
                    "max_hold_bars": sig.max_hold_bars,
                }

    # Force close at last bar
    if open_trade is not None:
        ts = pd.Timestamp(data.at[len(data) - 1, "timestamp"])
        c = float(data.at[len(data) - 1, "close"])
        direction = open_trade["direction"]
        fill = apply_exit_price(c, direction, costs)
        entry = open_trade["entry_fill"]
        size = open_trade["size"]
        if direction == "long":
            gross = (fill - entry) * size
        else:
            gross = (entry - fill) * size
        fees = (entry + fill) * size * costs.commission_rate
        pnl = gross - fees
        risk = open_trade["risk_cash"]
        capital += pnl
        trades.append(
            Trade(
                entry_time=open_trade["entry_time"],
                exit_time=ts,
                direction=direction,
                entry_price=entry,
                exit_price=fill,
                stop=open_trade["stop"],
                take_profit=open_trade["take_profit"],
                size=size,
                pnl=pnl,
                pnl_pct=pnl / open_trade["capital_at_entry"]
                if open_trade["capital_at_entry"]
                else 0.0,
                return_R=pnl / risk if risk else 0.0,
                exit_reason="eod",
                pattern=open_trade.get("pattern", ""),
                symbol=symbol,
                strategy=strategy_name,
                meta=open_trade.get("meta", {}),
            )
        )
        equity_pts.append((ts, capital))

    idx = pd.DatetimeIndex([t for t, _ in equity_pts])
    eq = pd.Series([v for _, v in equity_pts], index=idx, name="equity")
    # Deduplicate timestamps keeping last
    eq = eq[~eq.index.duplicated(keep="last")]

    return BacktestResult(
        trades=trades,
        equity=eq,
        initial_capital=initial_capital,
        final_capital=capital,
        strategy=strategy_name,
        symbol=symbol,
        params=params,
        params_note="",
        curve_fit_flags=curve_fit_flags,
    )
