"""
Custom event-driven backtest engine for multi-timeframe strategies.

Why not backtesting.py?
- PMTS requires 1H setup detection + 15m entries; backtesting.py is single-frame.
- We need HTF state machines, setup expiry, and one-trade-per-setup semantics.
- Explicit fill model (spread + slippage) is clearer for gold/crypto spot.

Assumptions for PAXG/USDT (Binance spot-like):
- Commission: 0.10% per side (taker)
- Half-spread: 0.02% (PAXG typically tight but not BTC-tight)
- Slippage: 0.03% adverse on entry/exit
- Round-trip friction ≈ 0.30% before edge
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import pandas as pd

from .strategy_pmts import Signal


@dataclass
class CostModel:
    commission_rate: float = 0.001  # 0.10% per side
    half_spread: float = 0.0002  # 0.02%
    slippage: float = 0.0003  # 0.03%


@dataclass
class Trade:
    entry_time: pd.Timestamp
    exit_time: pd.Timestamp
    direction: str
    entry_price: float
    exit_price: float
    stop: float
    take_profit: float
    size: float
    pnl: float
    pnl_pct: float
    return_R: float
    exit_reason: str
    pattern: str = ""
    meta: dict = field(default_factory=dict)


@dataclass
class BacktestResult:
    trades: list[Trade]
    equity: pd.Series
    equity_times: pd.DatetimeIndex
    initial_capital: float
    final_capital: float
    params_note: str = ""


def apply_entry_price(raw: float, direction: str, costs: CostModel) -> float:
    slip = costs.half_spread + costs.slippage
    if direction == "long":
        return raw * (1 + slip)
    return raw * (1 - slip)


def apply_exit_price(raw: float, direction: str, costs: CostModel) -> float:
    slip = costs.half_spread + costs.slippage
    if direction == "long":
        return raw * (1 - slip)
    return raw * (1 + slip)


def run_backtest(
    df_ltf: pd.DataFrame,
    signals: list[Signal],
    initial_capital: float = 10_000.0,
    risk_per_trade: float = 0.01,
    costs: CostModel | None = None,
    max_hold_bars: int = 4 * 24 * 5,  # ~5 days on 15m
    params_note: str = "",
) -> BacktestResult:
    """
    Event-driven: walk 15m bars, open on signal bar close (next-bar open more realistic —
    we fill at signal close + costs to avoid look-ahead of next open in sparse signals).
    Stops/TPs checked on subsequent bars using high/low (intrabar).
    """
    costs = costs or CostModel()
    if not signals:
        eq = pd.Series([initial_capital], index=df_ltf["timestamp"].iloc[:1])
        return BacktestResult([], eq, eq.index, initial_capital, initial_capital, params_note)

    df = df_ltf.reset_index(drop=True)
    ts_to_i = {pd.Timestamp(t): i for i, t in enumerate(df["timestamp"])}

    capital = initial_capital
    equity_pts: list[tuple[pd.Timestamp, float]] = [(pd.Timestamp(df.at[0, "timestamp"]), capital)]
    trades: list[Trade] = []
    open_trade: dict | None = None
    signal_q = sorted(signals, key=lambda s: s.timestamp)
    sig_idx = 0

    for i in range(len(df)):
        ts = pd.Timestamp(df.at[i, "timestamp"])
        o, h, l, c = (
            float(df.at[i, "open"]),
            float(df.at[i, "high"]),
            float(df.at[i, "low"]),
            float(df.at[i, "close"]),
        )

        # Manage open position
        if open_trade is not None:
            open_trade["bars"] += 1
            direction = open_trade["direction"]
            stop = open_trade["stop"]
            tp = open_trade["take_profit"]
            exit_px = None
            reason = None

            if direction == "long":
                # Conservative: if both hit, assume stop first
                if l <= stop:
                    exit_px, reason = stop, "stop"
                elif h >= tp:
                    exit_px, reason = tp, "take_profit"
            else:
                if h >= stop:
                    exit_px, reason = stop, "stop"
                elif l <= tp:
                    exit_px, reason = tp, "take_profit"

            if exit_px is None and open_trade["bars"] >= max_hold_bars:
                exit_px, reason = c, "time_exit"

            if exit_px is not None:
                fill = apply_exit_price(exit_px, direction, costs)
                entry = open_trade["entry_fill"]
                size = open_trade["size"]
                if direction == "long":
                    gross = (fill - entry) * size
                else:
                    gross = (entry - fill) * size
                # Commission both sides
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
                        pnl_pct=pnl / open_trade["capital_at_entry"],
                        return_R=pnl / risk if risk else 0.0,
                        exit_reason=reason or "exit",
                        pattern=open_trade.get("pattern", ""),
                        meta=open_trade.get("meta", {}),
                    )
                )
                equity_pts.append((ts, capital))
                open_trade = None

        # New entries only if flat
        if open_trade is None:
            while sig_idx < len(signal_q) and signal_q[sig_idx].timestamp < ts:
                sig_idx += 1
            if sig_idx < len(signal_q) and signal_q[sig_idx].timestamp == ts:
                sig = signal_q[sig_idx]
                sig_idx += 1
                entry_fill = apply_entry_price(sig.entry, sig.direction, costs)
                # Adjust stop/tp conceptually still at signal levels; risk from fill
                if sig.direction == "long":
                    risk_per_unit = entry_fill - sig.stop
                else:
                    risk_per_unit = sig.stop - entry_fill
                if risk_per_unit <= 0:
                    continue
                risk_cash = capital * risk_per_trade
                size = risk_cash / risk_per_unit
                # Cap notional at 95% capital
                max_size = (capital * 0.95) / entry_fill
                size = min(size, max_size)
                if size <= 0:
                    continue
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
                }

    # Force close at end
    if open_trade is not None:
        i = len(df) - 1
        ts = pd.Timestamp(df.at[i, "timestamp"])
        c = float(df.at[i, "close"])
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
                pnl_pct=pnl / open_trade["capital_at_entry"],
                return_R=pnl / open_trade["risk_cash"] if open_trade["risk_cash"] else 0.0,
                exit_reason="eod",
                pattern=open_trade.get("pattern", ""),
                meta=open_trade.get("meta", {}),
            )
        )
        equity_pts.append((ts, capital))

    if len(equity_pts) == 1:
        equity_pts.append((pd.Timestamp(df["timestamp"].iloc[-1]), capital))

    times = pd.DatetimeIndex([t for t, _ in equity_pts])
    equity = pd.Series([e for _, e in equity_pts], index=times, name="equity")
    return BacktestResult(trades, equity, times, initial_capital, capital, params_note)


def split_is_oos(
    df: pd.DataFrame,
    is_frac: float = 10 / 12,
) -> tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp]:
    """Return (start, is_end, end) timestamps for ~10mo IS / ~2mo OOS."""
    start = pd.Timestamp(df["timestamp"].iloc[0])
    end = pd.Timestamp(df["timestamp"].iloc[-1])
    span = end - start
    is_end = start + span * is_frac
    return start, is_end, end


def filter_signals_period(
    signals: list[Signal],
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> list[Signal]:
    return [s for s in signals if start <= s.timestamp < end]


def filter_df_period(df: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    m = (df["timestamp"] >= start) & (df["timestamp"] < end)
    return df.loc[m].reset_index(drop=True)
