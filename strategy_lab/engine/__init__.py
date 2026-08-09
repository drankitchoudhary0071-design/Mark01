"""Shared backtest engine package."""

from strategy_lab.engine.backtest import run_backtest
from strategy_lab.engine.costs import CostModel
from strategy_lab.engine.metrics import compute_metrics, format_metrics, overfitting_assessment
from strategy_lab.engine.types import BacktestResult, Signal, Trade
from strategy_lab.engine.walk_forward import run_holdout, run_walk_forward

__all__ = [
    "run_backtest",
    "CostModel",
    "compute_metrics",
    "format_metrics",
    "overfitting_assessment",
    "BacktestResult",
    "Signal",
    "Trade",
    "run_holdout",
    "run_walk_forward",
]
