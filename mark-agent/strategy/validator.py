"""
strategy/validator.py
Split data 70/30 train/test, backtest both, and flag OVERFIT
when train win_rate exceeds test win_rate by more than 15%.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from strategy.backtester import backtest


OVERFIT_THRESHOLD = 0.15  # 15 percentage points


def validate(strategy_dict: dict[str, Any], data: pd.DataFrame) -> dict[str, Any]:
    """
    Walk-forward style validation via a simple 70/30 chronological split.

    Returns:
        {
            "train": {...backtest metrics...},
            "test": {...backtest metrics...},
            "flag": "OVERFIT" | "OK",
            "win_rate_gap": float,  # train - test
        }
    """
    if data is None or len(data) < 100:
        return {
            "train": {},
            "test": {},
            "flag": "INSUFFICIENT_DATA",
            "win_rate_gap": 0.0,
            "message": "Need at least 100 candles to validate.",
        }

    split_idx = int(len(data) * 0.70)
    train_data = data.iloc[:split_idx].reset_index(drop=True)
    test_data = data.iloc[split_idx:].reset_index(drop=True)

    train_metrics = backtest(strategy_dict, train_data)
    test_metrics = backtest(strategy_dict, test_data)

    train_wr = float(train_metrics.get("win_rate", 0.0))
    test_wr = float(test_metrics.get("win_rate", 0.0))
    gap = train_wr - test_wr

    flag = "OVERFIT" if gap > OVERFIT_THRESHOLD else "OK"

    return {
        "train": train_metrics,
        "test": test_metrics,
        "flag": flag,
        "win_rate_gap": round(gap, 4),
        "message": (
            f"Train WR={train_wr:.1%}, Test WR={test_wr:.1%}, "
            f"gap={gap:.1%} → {flag}"
        ),
    }
