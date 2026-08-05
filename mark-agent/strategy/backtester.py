"""
strategy/backtester.py
Simple rule-based backtester.
Returns: win_rate, max_drawdown, avg_R, total_trades, sharpe_ratio
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def _rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def _true_range(df: pd.DataFrame) -> pd.Series:
    prev_close = df["close"].shift(1)
    return pd.concat(
        [
            df["high"] - df["low"],
            (df["high"] - prev_close).abs(),
            (df["low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)


def _prepare_indicators(df: pd.DataFrame, strategy: dict[str, Any]) -> pd.DataFrame:
    """Compute indicators referenced by the strategy rules."""
    out = df.copy()
    out["ma20"] = out["close"].rolling(20).mean()
    out["ma50"] = out["close"].rolling(50).mean()
    out["atr14"] = _true_range(out).rolling(14).mean()
    out["rsi14"] = _rsi(out["close"], 14)
    out["roll_high_50"] = out["high"].rolling(50).max()
    out["roll_low_50"] = out["low"].rolling(50).min()
    return out


def _entry_signal(row: pd.Series, prev: pd.Series, strategy: dict[str, Any]) -> bool:
    """Return True if entry rule fires on this bar."""
    entry = strategy.get("entry", {})
    etype = entry.get("type")
    params = entry.get("params", {})
    direction = strategy.get("direction", "long")

    if etype == "ma_crossover":
        # Fast MA crosses above slow (long) or below (short)
        if direction == "long":
            return prev["ma20"] <= prev["ma50"] and row["ma20"] > row["ma50"]
        return prev["ma20"] >= prev["ma50"] and row["ma20"] < row["ma50"]

    if etype == "near_support":
        tol = params.get("tolerance_pct", 0.3) / 100.0
        support = row["roll_low_50"]
        return bool(row["low"] <= support * (1 + tol))

    if etype == "breakout_high":
        # Close breaks prior lookback high
        return bool(row["close"] > prev["roll_high_50"])

    if etype == "rsi_oversold":
        threshold = params.get("threshold", 30)
        return bool(row["rsi14"] < threshold)

    return False


def _exit_signal(row: pd.Series, prev: pd.Series, strategy: dict[str, Any]) -> bool:
    """Soft exit rule (before stop/target)."""
    exit_rule = strategy.get("exit", {})
    etype = exit_rule.get("type")
    params = exit_rule.get("params", {})
    direction = strategy.get("direction", "long")

    if etype == "opposite_crossover":
        if direction == "long":
            return prev["ma20"] >= prev["ma50"] and row["ma20"] < row["ma50"]
        return prev["ma20"] <= prev["ma50"] and row["ma20"] > row["ma50"]

    if etype == "near_resistance":
        tol = params.get("tolerance_pct", 0.3) / 100.0
        resistance = row["roll_high_50"]
        return bool(row["high"] >= resistance * (1 - tol))

    if etype == "rsi_overbought":
        threshold = params.get("threshold", 55)
        return bool(row["rsi14"] > threshold)

    # trailing_atr handled via stop updates in the loop
    return False


def backtest(strategy_dict: dict[str, Any], data: pd.DataFrame) -> dict[str, float]:
    """
    Run a bar-by-bar backtest of a rule-based strategy.

    Returns dict with:
        win_rate, max_drawdown, avg_R, total_trades, sharpe_ratio
    """
    if data is None or len(data) < 60:
        return {
            "win_rate": 0.0,
            "max_drawdown": 0.0,
            "avg_R": 0.0,
            "total_trades": 0,
            "sharpe_ratio": 0.0,
        }

    df = _prepare_indicators(data, strategy_dict)
    direction = strategy_dict.get("direction", "long")
    stop_mult = float(strategy_dict.get("stop_loss", {}).get("multiple", 1.5))
    target_mult = float(strategy_dict.get("target", {}).get("multiple", 2.5))
    use_trailing = strategy_dict.get("exit", {}).get("type") == "trailing_atr"

    trades: list[dict[str, float]] = []
    equity = [1.0]  # start at 1 unit of capital
    position = None  # open trade state

    for i in range(1, len(df)):
        row = df.iloc[i]
        prev = df.iloc[i - 1]
        atr = row["atr14"]
        if pd.isna(atr) or atr <= 0:
            equity.append(equity[-1])
            continue

        # Manage open position
        if position is not None:
            exit_price = None
            outcome_r = None

            if direction == "long":
                # Update trailing stop if configured
                if use_trailing:
                    trail = row["close"] - stop_mult * atr
                    position["stop"] = max(position["stop"], trail)

                if row["low"] <= position["stop"]:
                    exit_price = position["stop"]
                    outcome_r = (exit_price - position["entry"]) / position["risk"]
                elif row["high"] >= position["target"]:
                    exit_price = position["target"]
                    outcome_r = (exit_price - position["entry"]) / position["risk"]
                elif _exit_signal(row, prev, strategy_dict):
                    exit_price = row["close"]
                    outcome_r = (exit_price - position["entry"]) / position["risk"]
            else:  # short
                if use_trailing:
                    trail = row["close"] + stop_mult * atr
                    position["stop"] = min(position["stop"], trail)

                if row["high"] >= position["stop"]:
                    exit_price = position["stop"]
                    outcome_r = (position["entry"] - exit_price) / position["risk"]
                elif row["low"] <= position["target"]:
                    exit_price = position["target"]
                    outcome_r = (position["entry"] - exit_price) / position["risk"]
                elif _exit_signal(row, prev, strategy_dict):
                    exit_price = row["close"]
                    outcome_r = (position["entry"] - exit_price) / position["risk"]

            if exit_price is not None:
                trades.append({"r": float(outcome_r), "pnl": float(outcome_r) * 0.01})
                # Risk 1% of equity per trade
                equity.append(equity[-1] * (1 + float(outcome_r) * 0.01))
                position = None
            else:
                equity.append(equity[-1])
            continue

        # Look for new entry
        if _entry_signal(row, prev, strategy_dict):
            entry = float(row["close"])
            risk = stop_mult * float(atr)
            if risk <= 0:
                equity.append(equity[-1])
                continue
            if direction == "long":
                position = {
                    "entry": entry,
                    "stop": entry - risk,
                    "target": entry + target_mult * float(atr),
                    "risk": risk,
                }
            else:
                position = {
                    "entry": entry,
                    "stop": entry + risk,
                    "target": entry - target_mult * float(atr),
                    "risk": risk,
                }

        equity.append(equity[-1])

    # Force-close any open trade at last close
    if position is not None:
        last = float(df["close"].iloc[-1])
        if direction == "long":
            r = (last - position["entry"]) / position["risk"]
        else:
            r = (position["entry"] - last) / position["risk"]
        trades.append({"r": float(r), "pnl": float(r) * 0.01})
        equity[-1] = equity[-2] * (1 + float(r) * 0.01) if len(equity) > 1 else equity[-1]

    total_trades = len(trades)
    if total_trades == 0:
        return {
            "win_rate": 0.0,
            "max_drawdown": 0.0,
            "avg_R": 0.0,
            "total_trades": 0,
            "sharpe_ratio": 0.0,
        }

    rs = np.array([t["r"] for t in trades], dtype=float)
    wins = np.sum(rs > 0)
    win_rate = float(wins / total_trades)
    avg_R = float(np.mean(rs))

    # Max drawdown from equity curve
    eq = np.array(equity, dtype=float)
    peak = np.maximum.accumulate(eq)
    drawdowns = (peak - eq) / np.where(peak == 0, 1, peak)
    max_drawdown = float(np.max(drawdowns)) if len(drawdowns) else 0.0

    # Sharpe on per-trade R multiples (annualization skipped for simplicity)
    std_r = float(np.std(rs, ddof=1)) if total_trades > 1 else 0.0
    sharpe_ratio = float(avg_R / std_r) if std_r > 0 else 0.0

    return {
        "win_rate": round(win_rate, 4),
        "max_drawdown": round(max_drawdown, 4),
        "avg_R": round(avg_R, 4),
        "total_trades": int(total_trades),
        "sharpe_ratio": round(sharpe_ratio, 4),
    }
