"""Core types for the shared backtest engine."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Optional

import pandas as pd

Direction = Literal["long", "short"]


@dataclass
class Signal:
    """One entry intent emitted by a strategy module."""

    timestamp: pd.Timestamp
    direction: Direction
    entry: float
    stop: float
    take_profit: float
    pattern: str = ""
    meta: dict[str, Any] = field(default_factory=dict)
    # Optional: strategy-managed exit without fixed TP (engine still needs a TP
    # level for risk sizing; use a wide TP and rely on signal-driven exits via
    # soft_exit_price in meta if needed).
    max_hold_bars: Optional[int] = None


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
    symbol: str = ""
    strategy: str = ""
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass
class BacktestResult:
    trades: list[Trade]
    equity: pd.Series
    initial_capital: float
    final_capital: float
    strategy: str = ""
    symbol: str = ""
    params: dict[str, Any] = field(default_factory=dict)
    params_note: str = ""
    curve_fit_flags: list[str] = field(default_factory=list)
