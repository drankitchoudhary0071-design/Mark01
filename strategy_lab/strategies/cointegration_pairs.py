"""
11. Cointegration-based pairs trading (Engle–Granger).

Distinct from ``statistical_arbitrage`` (rolling residual z-score):
here we require a rolling cointegration gate (ADF on OLS residuals)
before allowing mean-reversion entries on the spread.
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np
import pandas as pd

from strategy_lab.engine.costs import CostModel, apply_entry_price, apply_exit_price
from strategy_lab.engine.indicators import rolling_zscore
from strategy_lab.engine.types import BacktestResult, Signal, Trade
from strategy_lab.strategies.base import Strategy


# Approximate ADF critical value for n≈60–120, no constant (conservative)
_ADF_CRIT_5PCT = -2.89


def _ols_beta(y: np.ndarray, x: np.ndarray) -> tuple[float, np.ndarray]:
    """Simple OLS y = a + b x; return (b, residuals)."""
    n = len(y)
    X = np.column_stack([np.ones(n), x])
    try:
        coef, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    except np.linalg.LinAlgError:
        return np.nan, np.full(n, np.nan)
    resid = y - X @ coef
    return float(coef[1]), resid


def _adf_stat(resid: np.ndarray) -> float:
    """ADF(1) test statistic on residuals (no lag selection — textbook stub)."""
    resid = resid[~np.isnan(resid)]
    if len(resid) < 20:
        return np.nan
    dy = np.diff(resid)
    y_lag = resid[:-1]
    # dy = phi * y_lag + e
    var = np.dot(y_lag, y_lag)
    if var <= 0:
        return np.nan
    phi = float(np.dot(y_lag, dy) / var)
    e = dy - phi * y_lag
    s2 = float(np.dot(e, e) / max(len(e) - 1, 1))
    se = np.sqrt(s2 / var) if s2 > 0 else np.nan
    if not se or se <= 0 or np.isnan(se):
        return np.nan
    return phi / se


class CointegrationPairsStrategy(Strategy):
    name = "cointegration_pairs"
    library = "numpy Engle-Granger + two-leg sim"

    @classmethod
    def default_params(cls) -> dict[str, Any]:
        return {
            "coint_lookback": 90,  # CURVE-FIT RISK — window for EG test
            "coint_adf_crit": _ADF_CRIT_5PCT,  # literature approx, not tuned
            "pairs_entry_z": 2.0,
            "pairs_exit_z": 0.5,  # CURVE-FIT RISK
            "stop_z": 3.5,
            "risk_per_trade": 0.01,
            "retest_every": 12,  # re-check cointegration every N bars
        }

    def generate_signals(self, df: pd.DataFrame) -> list[Signal]:
        """Optional single-leg proxy; runner prefers ``simulate_coint_pairs``."""
        return []


def simulate_coint_pairs(
    df_a: pd.DataFrame,
    df_b: pd.DataFrame,
    params: Optional[dict] = None,
    *,
    initial_capital: float = 10_000.0,
    costs: Optional[CostModel] = None,
    strategy_name: str = "cointegration_pairs",
    symbol: str = "PAXGUSDT/BTCUSDT",
) -> BacktestResult:
    params = {**CointegrationPairsStrategy.default_params(), **(params or {})}
    costs = costs or CostModel()
    strat = CointegrationPairsStrategy(**params)

    work = df_a.merge(df_b, on="timestamp", suffixes=("", "_b"))
    lookback = int(params["coint_lookback"])
    n = len(work)
    capital = float(initial_capital)
    equity_pts: list[tuple[pd.Timestamp, float]] = [
        (pd.Timestamp(work["timestamp"].iloc[0]), capital)
    ]
    trades: list[Trade] = []
    pos = None
    coint_ok = False
    beta = np.nan
    last_test_i = -10_000

    for i in range(lookback, n):
        ts = pd.Timestamp(work["timestamp"].iloc[i])
        pa = float(work["close"].iloc[i])
        pb = float(work["close_b"].iloc[i])

        if i - last_test_i >= int(params["retest_every"]):
            y = work["close"].iloc[i - lookback : i].to_numpy(dtype=float)
            x = work["close_b"].iloc[i - lookback : i].to_numpy(dtype=float)
            b, resid = _ols_beta(y, x)
            adf = _adf_stat(resid)
            coint_ok = (not np.isnan(adf)) and adf < float(params["coint_adf_crit"])
            beta = b
            last_test_i = i

        if not coint_ok or np.isnan(beta):
            if pos is not None:
                # Flatten if cointegration breaks
                pass
            else:
                continue

        # Rolling z of residual using current beta
        window = work.iloc[i - lookback : i + 1]
        spread = window["close"] - beta * window["close_b"]
        z_series = rolling_zscore(spread, lookback)
        za = float(z_series.iloc[-1]) if not pd.isna(z_series.iloc[-1]) else np.nan
        if np.isnan(za):
            continue

        if pos is None and coint_ok:
            if za <= -params["pairs_entry_z"] or za >= params["pairs_entry_z"]:
                long_spread = za <= -params["pairs_entry_z"]
                notional = capital * params["risk_per_trade"] * 10
                size_a = notional / pa
                size_b = (notional * abs(float(beta))) / pb
                entry_a = apply_entry_price(pa, "long" if long_spread else "short", costs)
                entry_b = apply_entry_price(pb, "short" if long_spread else "long", costs)
                pos = {
                    "long_spread": long_spread,
                    "size_a": size_a,
                    "size_b": size_b,
                    "entry_a": entry_a,
                    "entry_b": entry_b,
                    "entry_time": ts,
                    "entry_z": za,
                    "capital_at_entry": capital,
                }
        elif pos is not None:
            broken = not coint_ok
            exit_now = abs(za) <= params["pairs_exit_z"] or abs(za) >= params["stop_z"] or broken
            if exit_now or i == n - 1:
                long_spread = pos["long_spread"]
                exit_a = apply_exit_price(pa, "long" if long_spread else "short", costs)
                exit_b = apply_exit_price(pb, "short" if long_spread else "long", costs)
                if long_spread:
                    pnl_a = (exit_a - pos["entry_a"]) * pos["size_a"]
                    pnl_b = (pos["entry_b"] - exit_b) * pos["size_b"]
                else:
                    pnl_a = (pos["entry_a"] - exit_a) * pos["size_a"]
                    pnl_b = (exit_b - pos["entry_b"]) * pos["size_b"]
                fees = (
                    (pos["entry_a"] + exit_a) * pos["size_a"]
                    + (pos["entry_b"] + exit_b) * pos["size_b"]
                ) * costs.commission_rate
                pnl = pnl_a + pnl_b - fees
                risk = pos["capital_at_entry"] * params["risk_per_trade"]
                capital += pnl
                if broken:
                    reason = "coint_break"
                elif abs(za) >= params["stop_z"]:
                    reason = "stop"
                elif i == n - 1:
                    reason = "eod"
                else:
                    reason = "take_profit"
                trades.append(
                    Trade(
                        entry_time=pos["entry_time"],
                        exit_time=ts,
                        direction="long" if long_spread else "short",
                        entry_price=pos["entry_a"],
                        exit_price=exit_a,
                        stop=0.0,
                        take_profit=0.0,
                        size=pos["size_a"],
                        pnl=pnl,
                        pnl_pct=pnl / pos["capital_at_entry"] if pos["capital_at_entry"] else 0.0,
                        return_R=pnl / risk if risk else 0.0,
                        exit_reason=reason,
                        pattern="coint_pairs",
                        symbol=symbol,
                        strategy=strategy_name,
                        meta={"entry_z": pos["entry_z"], "exit_z": za, "beta": beta},
                    )
                )
                equity_pts.append((ts, capital))
                pos = None

    idx = pd.DatetimeIndex([t for t, _ in equity_pts])
    eq = pd.Series([v for _, v in equity_pts], index=idx, name="equity")
    eq = eq[~eq.index.duplicated(keep="last")]
    flags = strat.curve_fit_flags + [
        "[CURVE-FIT RISK] coint_lookback and notional multiplier are research defaults — require WF stability.",
        "[NOTE] ADF critical value is an approximate textbook constant, not fitted to this pair.",
    ]
    return BacktestResult(
        trades=trades,
        equity=eq,
        initial_capital=initial_capital,
        final_capital=capital,
        strategy=strategy_name,
        symbol=symbol,
        params=params,
        curve_fit_flags=flags,
    )
