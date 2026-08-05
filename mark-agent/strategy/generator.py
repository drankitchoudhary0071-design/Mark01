"""
strategy/generator.py
Analyze market data and propose ONE new rule-based strategy
as an explainable Python dict (entry, exit, stop loss, target).
Avoids duplicates of strategies already in existing_strategies.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def _atr(df: pd.DataFrame, period: int = 14) -> float:
    """Average True Range — used for volatility sizing of stops/targets."""
    high = df["high"]
    low = df["low"]
    close = df["close"]
    prev_close = close.shift(1)
    tr = pd.concat(
        [
            (high - low),
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return float(tr.rolling(period).mean().iloc[-1])


def _trend_direction(df: pd.DataFrame, fast: int = 20, slow: int = 50) -> str:
    """Simple MA crossover trend label."""
    if len(df) < slow + 1:
        return "sideways"
    ma_fast = df["close"].rolling(fast).mean().iloc[-1]
    ma_slow = df["close"].rolling(slow).mean().iloc[-1]
    if ma_fast > ma_slow * 1.002:
        return "up"
    if ma_fast < ma_slow * 0.998:
        return "down"
    return "sideways"


def _support_resistance(df: pd.DataFrame, lookback: int = 50) -> tuple[float, float]:
    """Recent swing low / high as crude support and resistance."""
    window = df.tail(lookback)
    support = float(window["low"].min())
    resistance = float(window["high"].max())
    return support, resistance


def _strategy_signature(s: dict[str, Any]) -> str:
    """Normalize a strategy into a comparable key (to avoid duplicates)."""
    return "|".join(
        [
            str(s.get("name", "")),
            str(s.get("direction", "")),
            str(s.get("entry", {}).get("type", "")),
            str(s.get("entry", {}).get("params", {})),
            str(s.get("exit", {}).get("type", "")),
            str(s.get("stop_loss", {})),
            str(s.get("target", {})),
        ]
    )


def _build_candidates(df: pd.DataFrame) -> list[dict[str, Any]]:
    """
    Build a small pool of explainable rule-based strategies
    based on volatility, trend, and support/resistance.
    """
    atr = _atr(df)
    last_close = float(df["close"].iloc[-1])
    trend = _trend_direction(df)
    support, resistance = _support_resistance(df)
    atr_pct = (atr / last_close) * 100 if last_close else 0.0

    # Volatility regime drives stop / target multiples
    if atr_pct >= 1.5:
        stop_mult, target_mult = 2.0, 3.0
        vol_label = "high"
    elif atr_pct >= 0.6:
        stop_mult, target_mult = 1.5, 2.5
        vol_label = "medium"
    else:
        stop_mult, target_mult = 1.0, 2.0
        vol_label = "low"

    candidates: list[dict[str, Any]] = []

    # 1) Trend-following MA crossover
    candidates.append(
        {
            "name": f"ma_crossover_{trend}_{vol_label}",
            "direction": "long" if trend != "down" else "short",
            "rationale": (
                f"Trend is {trend}, volatility is {vol_label} "
                f"(ATR≈{atr_pct:.2f}%). Trade MA20/MA50 crossover."
            ),
            "entry": {
                "type": "ma_crossover",
                "params": {"fast": 20, "slow": 50},
            },
            "exit": {
                "type": "opposite_crossover",
                "params": {"fast": 20, "slow": 50},
            },
            "stop_loss": {
                "type": "atr_multiple",
                "multiple": stop_mult,
                "atr_period": 14,
            },
            "target": {
                "type": "atr_multiple",
                "multiple": target_mult,
                "atr_period": 14,
            },
            "meta": {
                "atr": atr,
                "atr_pct": atr_pct,
                "trend": trend,
                "support": support,
                "resistance": resistance,
            },
        }
    )

    # 2) Mean-reversion near support (long bias)
    candidates.append(
        {
            "name": f"support_bounce_{vol_label}",
            "direction": "long",
            "rationale": (
                f"Price near support {support:.2f}; buy dips with "
                f"{stop_mult}x ATR stop and {target_mult}x ATR target."
            ),
            "entry": {
                "type": "near_support",
                "params": {
                    "lookback": 50,
                    "tolerance_pct": 0.3,
                },
            },
            "exit": {
                "type": "near_resistance",
                "params": {"lookback": 50, "tolerance_pct": 0.3},
            },
            "stop_loss": {
                "type": "atr_multiple",
                "multiple": stop_mult,
                "atr_period": 14,
            },
            "target": {
                "type": "atr_multiple",
                "multiple": target_mult,
                "atr_period": 14,
            },
            "meta": {
                "atr": atr,
                "atr_pct": atr_pct,
                "trend": trend,
                "support": support,
                "resistance": resistance,
            },
        }
    )

    # 3) Breakout above resistance
    candidates.append(
        {
            "name": f"resistance_breakout_{vol_label}",
            "direction": "long",
            "rationale": (
                f"Breakout above resistance {resistance:.2f} with "
                f"volatility-scaled stop/target ({vol_label} ATR)."
            ),
            "entry": {
                "type": "breakout_high",
                "params": {"lookback": 50},
            },
            "exit": {
                "type": "trailing_atr",
                "params": {"multiple": stop_mult, "atr_period": 14},
            },
            "stop_loss": {
                "type": "atr_multiple",
                "multiple": stop_mult,
                "atr_period": 14,
            },
            "target": {
                "type": "atr_multiple",
                "multiple": target_mult * 1.2,
                "atr_period": 14,
            },
            "meta": {
                "atr": atr,
                "atr_pct": atr_pct,
                "trend": trend,
                "support": support,
                "resistance": resistance,
            },
        }
    )

    # 4) RSI mean reversion (oversold bounce)
    candidates.append(
        {
            "name": f"rsi_oversold_{vol_label}",
            "direction": "long",
            "rationale": (
                f"Buy when RSI(14) < 30; exit when RSI > 55. "
                f"Stops sized for {vol_label} volatility."
            ),
            "entry": {
                "type": "rsi_oversold",
                "params": {"period": 14, "threshold": 30},
            },
            "exit": {
                "type": "rsi_overbought",
                "params": {"period": 14, "threshold": 55},
            },
            "stop_loss": {
                "type": "atr_multiple",
                "multiple": stop_mult,
                "atr_period": 14,
            },
            "target": {
                "type": "atr_multiple",
                "multiple": target_mult,
                "atr_period": 14,
            },
            "meta": {
                "atr": atr,
                "atr_pct": atr_pct,
                "trend": trend,
                "support": support,
                "resistance": resistance,
            },
        }
    )

    return candidates


def generate_strategy(
    market_data: pd.DataFrame,
    existing_strategies: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """
    Propose ONE new rule-based strategy that is not already in existing_strategies.

    Returns:
        Explainable Python dict with entry, exit, stop_loss, target, and rationale.
    """
    if market_data is None or len(market_data) < 60:
        raise ValueError("Need at least 60 candles of market_data to generate a strategy.")

    existing_strategies = existing_strategies or []
    existing_keys = {_strategy_signature(s) for s in existing_strategies}
    existing_names = {s.get("name") for s in existing_strategies}

    candidates = _build_candidates(market_data)

    for candidate in candidates:
        sig = _strategy_signature(candidate)
        if sig in existing_keys or candidate["name"] in existing_names:
            continue
        return candidate

    # If all templates already exist, invent a unique RSI variant
    atr = _atr(market_data)
    last = float(market_data["close"].iloc[-1])
    atr_pct = (atr / last) * 100 if last else 0.0
    unique_name = f"rsi_custom_{len(existing_strategies) + 1}"
    fallback = {
        "name": unique_name,
        "direction": "long",
        "rationale": (
            f"All standard templates already exist. "
            f"Custom RSI(14)<28 entry with ATR stop (ATR≈{atr_pct:.2f}%)."
        ),
        "entry": {
            "type": "rsi_oversold",
            "params": {"period": 14, "threshold": 28},
        },
        "exit": {
            "type": "rsi_overbought",
            "params": {"period": 14, "threshold": 60},
        },
        "stop_loss": {
            "type": "atr_multiple",
            "multiple": 1.5,
            "atr_period": 14,
        },
        "target": {
            "type": "atr_multiple",
            "multiple": 2.5,
            "atr_period": 14,
        },
        "meta": {"atr": atr, "atr_pct": atr_pct},
    }
    return fallback
