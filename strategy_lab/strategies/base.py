"""Strategy protocol / base helpers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Optional

import pandas as pd

from strategy_lab.config import curve_fit_warnings
from strategy_lab.engine.types import Signal


class Strategy(ABC):
    """Each family implements ``generate_signals`` + exposes ``name`` / ``params``."""

    name: str = "base"
    # Library note for README / reports
    library: str = "pandas+shared_engine"

    def __init__(self, **params: Any):
        self.params = {**self.default_params(), **params}
        self.curve_fit_flags = curve_fit_warnings(self.params)

    @classmethod
    @abstractmethod
    def default_params(cls) -> dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def generate_signals(self, df: pd.DataFrame) -> list[Signal]:
        raise NotImplementedError

    def describe(self) -> str:
        flags = "\n".join(f"  - {f}" for f in self.curve_fit_flags) or "  (none)"
        return (
            f"{self.name} [{self.library}]\n"
            f"params: {self.params}\n"
            f"curve-fit flags:\n{flags}"
        )


def atr_stop_tp(
    entry: float,
    direction: str,
    atr_val: float,
    stop_atr: float,
    tp_atr: float,
) -> tuple[float, float]:
    """Standard ATR-multiple stop / take-profit."""
    if direction == "long":
        stop = entry - stop_atr * atr_val
        tp = entry + tp_atr * atr_val
    else:
        stop = entry + stop_atr * atr_val
        tp = entry - tp_atr * atr_val
    return stop, tp


def pct_stop_tp(
    entry: float,
    direction: str,
    stop_pct: float,
    tp_pct: float,
) -> tuple[float, float]:
    if direction == "long":
        return entry * (1 - stop_pct), entry * (1 + tp_pct)
    return entry * (1 + stop_pct), entry * (1 - tp_pct)
