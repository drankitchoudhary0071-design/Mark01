"""Walk-forward and holdout out-of-sample validation helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import pandas as pd

from strategy_lab.engine.backtest import run_backtest
from strategy_lab.engine.costs import CostModel
from strategy_lab.engine.metrics import compute_metrics, trades_to_frame
from strategy_lab.engine.types import BacktestResult, Signal, Trade


SignalFn = Callable[[pd.DataFrame], list[Signal]]


@dataclass
class FoldResult:
    fold: int
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    train_metrics: dict
    test_metrics: dict
    test_result: BacktestResult


@dataclass
class WalkForwardResult:
    folds: list[FoldResult]
    combined_oos: BacktestResult
    combined_oos_metrics: dict
    aggregate_is_metrics: dict


def holdout_split(
    df: pd.DataFrame, oos_frac: float = 0.25
) -> tuple[pd.DataFrame, pd.DataFrame]:
    n = len(df)
    cut = int(n * (1.0 - oos_frac))
    cut = max(1, min(n - 1, cut))
    return df.iloc[:cut].reset_index(drop=True), df.iloc[cut:].reset_index(drop=True)


def walk_forward_windows(
    n: int,
    train_bars: int,
    test_bars: int,
    step_bars: int,
) -> list[tuple[int, int, int, int]]:
    """Yield (train_start, train_end, test_start, test_end) integer indices."""
    windows = []
    start = 0
    while True:
        train_start = start
        train_end = train_start + train_bars
        test_start = train_end
        test_end = test_start + test_bars
        if test_end > n:
            break
        windows.append((train_start, train_end, test_start, test_end))
        start += step_bars
    return windows


def run_holdout(
    df: pd.DataFrame,
    signal_fn: SignalFn,
    *,
    oos_frac: float = 0.25,
    initial_capital: float = 10_000.0,
    risk_per_trade: float = 0.01,
    costs: Optional[CostModel] = None,
    strategy_name: str = "",
    symbol: str = "",
    params: Optional[dict] = None,
    curve_fit_flags: Optional[list[str]] = None,
    max_hold_bars: int = 24 * 10,
) -> tuple[BacktestResult, BacktestResult, dict, dict]:
    """Classic IS / OOS split. Signals generated independently on each slice
    (no peeking); indicator warm-up happens within each slice."""
    is_df, oos_df = holdout_split(df, oos_frac)
    is_signals = signal_fn(is_df)
    oos_signals = signal_fn(oos_df)
    common = dict(
        initial_capital=initial_capital,
        risk_per_trade=risk_per_trade,
        costs=costs,
        strategy_name=strategy_name,
        symbol=symbol,
        params=params,
        curve_fit_flags=curve_fit_flags,
        max_hold_bars=max_hold_bars,
    )
    is_res = run_backtest(is_df, is_signals, **common)
    oos_res = run_backtest(oos_df, oos_signals, **common)
    return is_res, oos_res, compute_metrics(is_res), compute_metrics(oos_res)


def run_walk_forward(
    df: pd.DataFrame,
    signal_fn: SignalFn,
    *,
    train_bars: int,
    test_bars: int,
    step_bars: int,
    initial_capital: float = 10_000.0,
    risk_per_trade: float = 0.01,
    costs: Optional[CostModel] = None,
    strategy_name: str = "",
    symbol: str = "",
    params: Optional[dict] = None,
    curve_fit_flags: Optional[list[str]] = None,
    max_hold_bars: int = 24 * 10,
) -> WalkForwardResult:
    """
    Anchored rolling walk-forward:
      for each fold, generate signals on TRAIN only for diagnostics,
      then generate signals on TEST (fresh) and backtest TEST.

    Combined OOS stitches test-fold trades in time order with a single
    capital path starting at initial_capital (fold results also kept).
    """
    n = len(df)
    windows = walk_forward_windows(n, train_bars, test_bars, step_bars)
    folds: list[FoldResult] = []
    all_oos_trades: list[Trade] = []
    equity_pts: list[tuple[pd.Timestamp, float]] = []
    capital = float(initial_capital)

    if not windows:
        # Fall back to holdout if series too short for WF
        is_res, oos_res, is_m, oos_m = run_holdout(
            df,
            signal_fn,
            initial_capital=initial_capital,
            risk_per_trade=risk_per_trade,
            costs=costs,
            strategy_name=strategy_name,
            symbol=symbol,
            params=params,
            curve_fit_flags=curve_fit_flags,
            max_hold_bars=max_hold_bars,
        )
        fold = FoldResult(
            fold=0,
            train_start=pd.Timestamp(df["timestamp"].iloc[0]),
            train_end=pd.Timestamp(is_res.equity.index[-1]) if len(is_res.equity) else pd.Timestamp(df["timestamp"].iloc[0]),
            test_start=pd.Timestamp(oos_res.equity.index[0]) if len(oos_res.equity) else pd.Timestamp(df["timestamp"].iloc[-1]),
            test_end=pd.Timestamp(df["timestamp"].iloc[-1]),
            train_metrics=is_m,
            test_metrics=oos_m,
            test_result=oos_res,
        )
        return WalkForwardResult([fold], oos_res, oos_m, is_m)

    is_metrics_list = []

    for fi, (tr_s, tr_e, te_s, te_e) in enumerate(windows):
        train_df = df.iloc[tr_s:tr_e].reset_index(drop=True)
        test_df = df.iloc[te_s:te_e].reset_index(drop=True)

        # Warm-up: prepend last lookback of train so indicators are valid at
        # test start, but only keep signals whose timestamp is inside test.
        warmup = min(len(train_df), max(100, train_bars // 5))
        warm_df = pd.concat(
            [train_df.iloc[-warmup:], test_df], ignore_index=True
        )
        raw_signals = signal_fn(warm_df)
        test_start_ts = pd.Timestamp(test_df["timestamp"].iloc[0])
        test_end_ts = pd.Timestamp(test_df["timestamp"].iloc[-1])
        test_signals = [
            s
            for s in raw_signals
            if test_start_ts <= pd.Timestamp(s.timestamp) <= test_end_ts
        ]

        train_signals = signal_fn(train_df)
        train_res = run_backtest(
            train_df,
            train_signals,
            initial_capital=initial_capital,
            risk_per_trade=risk_per_trade,
            costs=costs,
            strategy_name=strategy_name,
            symbol=symbol,
            params=params,
            curve_fit_flags=curve_fit_flags,
            max_hold_bars=max_hold_bars,
        )
        test_res = run_backtest(
            test_df,
            test_signals,
            initial_capital=capital,  # continue capital path
            risk_per_trade=risk_per_trade,
            costs=costs,
            strategy_name=strategy_name,
            symbol=symbol,
            params=params,
            curve_fit_flags=curve_fit_flags,
            max_hold_bars=max_hold_bars,
        )
        capital = test_res.final_capital
        all_oos_trades.extend(test_res.trades)
        for t, v in test_res.equity.items():
            equity_pts.append((pd.Timestamp(t), float(v)))

        train_m = compute_metrics(train_res)
        test_m = compute_metrics(test_res)
        is_metrics_list.append(train_m)
        folds.append(
            FoldResult(
                fold=fi,
                train_start=pd.Timestamp(train_df["timestamp"].iloc[0]),
                train_end=pd.Timestamp(train_df["timestamp"].iloc[-1]),
                test_start=test_start_ts,
                test_end=test_end_ts,
                train_metrics=train_m,
                test_metrics=test_m,
                test_result=test_res,
            )
        )

    if equity_pts:
        idx = pd.DatetimeIndex([t for t, _ in equity_pts])
        eq = pd.Series([v for _, v in equity_pts], index=idx, name="equity")
        eq = eq[~eq.index.duplicated(keep="last")]
    else:
        eq = pd.Series(
            [initial_capital],
            index=pd.DatetimeIndex([pd.Timestamp(df["timestamp"].iloc[0])]),
        )

    combined = BacktestResult(
        trades=all_oos_trades,
        equity=eq,
        initial_capital=initial_capital,
        final_capital=capital,
        strategy=strategy_name,
        symbol=symbol,
        params=params or {},
        curve_fit_flags=curve_fit_flags or [],
    )
    combined_m = compute_metrics(combined)

    # Aggregate IS: mean of fold train metrics (simple)
    def _avg_key(key: str) -> float:
        vals = [m[key] for m in is_metrics_list if key in m]
        return float(sum(vals) / len(vals)) if vals else 0.0

    agg_is = {
        "strategy": strategy_name,
        "symbol": symbol,
        "total_trades": int(sum(m["total_trades"] for m in is_metrics_list)),
        "win_rate": _avg_key("win_rate"),
        "expectancy_R": _avg_key("expectancy_R"),
        "expectancy_pnl": _avg_key("expectancy_pnl"),
        "max_drawdown": _avg_key("max_drawdown"),
        "sharpe_ratio": _avg_key("sharpe_ratio"),
        "profit_factor": _avg_key("profit_factor"),
        "total_return": _avg_key("total_return"),
        "final_equity": initial_capital,
        "initial_equity": initial_capital,
        "avg_win_R": _avg_key("avg_win_R"),
        "avg_loss_R": _avg_key("avg_loss_R"),
        "long_trades": int(sum(m["long_trades"] for m in is_metrics_list)),
        "short_trades": int(sum(m["short_trades"] for m in is_metrics_list)),
        "stop_exits": int(sum(m["stop_exits"] for m in is_metrics_list)),
        "tp_exits": int(sum(m["tp_exits"] for m in is_metrics_list)),
        "params": params or {},
        "curve_fit_flags": curve_fit_flags or [],
    }

    return WalkForwardResult(folds, combined, combined_m, agg_is)


def folds_summary_frame(wf: WalkForwardResult) -> pd.DataFrame:
    rows = []
    for f in wf.folds:
        rows.append(
            {
                "fold": f.fold,
                "train_start": f.train_start,
                "train_end": f.train_end,
                "test_start": f.test_start,
                "test_end": f.test_end,
                "is_trades": f.train_metrics["total_trades"],
                "is_wr": f.train_metrics["win_rate"],
                "is_exp_R": f.train_metrics["expectancy_R"],
                "is_pf": f.train_metrics["profit_factor"],
                "oos_trades": f.test_metrics["total_trades"],
                "oos_wr": f.test_metrics["win_rate"],
                "oos_exp_R": f.test_metrics["expectancy_R"],
                "oos_pf": f.test_metrics["profit_factor"],
                "oos_sharpe": f.test_metrics["sharpe_ratio"],
                "oos_dd": f.test_metrics["max_drawdown"],
                "oos_ret": f.test_metrics["total_return"],
            }
        )
    return pd.DataFrame(rows)
