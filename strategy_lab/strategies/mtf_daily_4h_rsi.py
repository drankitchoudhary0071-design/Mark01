"""
Multi-Timeframe: Daily trend + 4H RSI mean-reversion.

Thin wrapper around ``mtf_rsi`` with fixed 1D+4H defaults for backward
compatibility with ``run_mtf_daily_4h_rsi.py`` and existing Pine ports.
"""

from __future__ import annotations

from typing import Any

from strategy_lab.strategies.mtf_rsi import (
    MtfRsiStrategy,
    prepare_mtf_frame,
    resample_ohlcv,
    to_rule,
)

__all__ = [
    "MtfDaily4hRsiStrategy",
    "prepare_mtf_frame",
    "resample_ohlcv",
    "to_rule",
]


class MtfDaily4hRsiStrategy(MtfRsiStrategy):
    name = "mtf_daily_4h_rsi"
    library = "pandas MTF (1D+4H) + shared engine"

    @classmethod
    def default_params(cls) -> dict[str, Any]:
        return {
            **MtfRsiStrategy.default_params(),
            "ltf": "4h",
            "htf": "1D",
        }

    def __init__(self, **params: Any):
        merged = {**self.default_params(), **params}
        super().__init__(**merged)
        self.name = "mtf_daily_4h_rsi"
        self.curve_fit_flags = list(self.curve_fit_flags) + [
            "[MTF] 1D EMA50/200 regime + 4H RSI cross — different from mtf_confluence / 4h-only RSI-MR.",
        ]
