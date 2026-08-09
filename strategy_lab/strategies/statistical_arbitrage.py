"""
3. Statistical Arbitrage — pairs trading (z-score of spread / residual).

Requires both legs (e.g. PAXGUSDT vs BTCUSDT). The shared single-asset engine
is used on the *spread proxy* via signals on leg A with hedge metadata;
PnL approximation trades leg A only sized on residual risk — honest note
in module docstring: full two-leg accounting is in ``simulate_pairs``.
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np
import pandas as pd

from strategy_lab.engine.costs import CostModel, apply_entry_price, apply_exit_price
from strategy_lab.engine.indicators import rolling_zscore
from strategy_lab.engine.types import BacktestResult, Signal, Trade
from strategy_lab.strategies.base import Strategy


class StatisticalArbitrageStrategy(Strategy):
    name = "statistical_arbitrage"
    library = "pandas+numpy (pairs residual)"

    @classmethod
    def default_params(cls) -> dict[str, Any]:
        return {
            "pairs_lookback": 60,  # CURVE-FIT RISK — see config.PARAM_NOTES
            "pairs_entry_z": 2.0,
            "pairs_exit_z": 0.5,  # CURVE-FIT RISK
            "stop_z": 3.5,  # flatten if residual blows out
            "risk_per_trade": 0.01,
            "hedge_mode": "beta",  # beta | dollar
        }

    def generate_signals(self, df: pd.DataFrame) -> list[Signal]:
        """
        Expects columns: close (leg A), close_b (leg B).
        Emits long/short on leg A when z-score extreme; stop/tp in price space
        approximated from residual sigma.
        """
        if "close_b" not in df.columns:
            return []
        p = self.params
        out = df.copy()
        lookback = int(p["pairs_lookback"])
        # Rolling OLS beta: cov(a,b)/var(b)
        cov = out["close"].rolling(lookback).cov(out["close_b"])
        var_b = out["close_b"].rolling(lookback).var()
        beta = cov / var_b.replace(0, np.nan)
        spread = out["close"] - beta * out["close_b"]
        z = rolling_zscore(spread, lookback)
        out["beta"] = beta
        out["spread"] = spread
        out["z"] = z
        # Price sigma of residual mapped roughly to leg A
        resid_std = spread.rolling(lookback).std()

        signals: list[Signal] = []
        in_pos = None  # track to avoid stacking; engine also enforces
        for i in range(1, len(out)):
            row = out.iloc[i]
            if pd.isna(row["z"]) or pd.isna(row["beta"]) or pd.isna(resid_std.iloc[i]):
                continue
            sigma = float(resid_std.iloc[i])
            if sigma <= 0:
                continue
            entry = float(row["close"])
            zval = float(row["z"])

            if in_pos is None:
                if zval <= -p["pairs_entry_z"]:
                    # spread cheap → long A / short B
                    stop = entry - (p["stop_z"] - abs(zval)) * sigma
                    tp = entry + (abs(zval) - p["pairs_exit_z"]) * sigma
                    signals.append(
                        Signal(
                            timestamp=pd.Timestamp(row["timestamp"]),
                            direction="long",
                            entry=entry,
                            stop=stop,
                            take_profit=max(tp, entry + 0.1 * sigma),
                            pattern="pairs_long_A",
                            meta={"beta": float(row["beta"]), "z": zval, "leg_b": float(row["close_b"])},
                        )
                    )
                    in_pos = "long"
                elif zval >= p["pairs_entry_z"]:
                    stop = entry + (p["stop_z"] - abs(zval)) * sigma
                    tp = entry - (abs(zval) - p["pairs_exit_z"]) * sigma
                    signals.append(
                        Signal(
                            timestamp=pd.Timestamp(row["timestamp"]),
                            direction="short",
                            entry=entry,
                            stop=stop,
                            take_profit=min(tp, entry - 0.1 * sigma),
                            pattern="pairs_short_A",
                            meta={"beta": float(row["beta"]), "z": zval, "leg_b": float(row["close_b"])},
                        )
                    )
                    in_pos = "short"
            else:
                # Reset state machine loosely when |z| collapses (signals themselves
                # are one-shot; engine manages exits via stop/tp).
                if abs(zval) <= p["pairs_exit_z"]:
                    in_pos = None
                elif abs(zval) >= p["stop_z"]:
                    in_pos = None
        return signals


def simulate_pairs(
    df_a: pd.DataFrame,
    df_b: pd.DataFrame,
    params: Optional[dict] = None,
    *,
    initial_capital: float = 10_000.0,
    costs: Optional[CostModel] = None,
    strategy_name: str = "statistical_arbitrage",
    symbol: str = "PAXGUSDT/BTCUSDT",
) -> BacktestResult:
    """
    Two-leg simulation: dollar-neutral-ish positions using rolling beta.
    Reports same BacktestResult type for comparability.
    """
    params = {**StatisticalArbitrageStrategy.default_params(), **(params or {})}
    costs = costs or CostModel()
    strat = StatisticalArbitrageStrategy(**params)

    merged = df_a.merge(df_b, on="timestamp", suffixes=("", "_b"))
    # Rename for signal helper
    work = merged.rename(
        columns={
            "open": "open",
            "high": "high",
            "low": "low",
            "close": "close",
            "close_b": "close_b",
        }
    )
    # Ensure close_b exists (merge suffixes)
    if "close_b" not in work.columns:
        # merge with suffixes _x/_y style
        raise ValueError("simulate_pairs expects aligned frames with close on both")

    lookback = int(params["pairs_lookback"])
    cov = work["close"].rolling(lookback).cov(work["close_b"])
    var_b = work["close_b"].rolling(lookback).var()
    beta = cov / var_b.replace(0, np.nan)
    spread = work["close"] - beta * work["close_b"]
    z = rolling_zscore(spread, lookback)

    capital = float(initial_capital)
    equity_pts: list[tuple[pd.Timestamp, float]] = [
        (pd.Timestamp(work["timestamp"].iloc[0]), capital)
    ]
    trades: list[Trade] = []
    pos = None

    for i in range(len(work)):
        ts = pd.Timestamp(work["timestamp"].iloc[i])
        za = z.iloc[i]
        b = beta.iloc[i]
        if pd.isna(za) or pd.isna(b):
            continue
        pa = float(work["close"].iloc[i])
        pb = float(work["close_b"].iloc[i])

        if pos is None:
            if za <= -params["pairs_entry_z"] or za >= params["pairs_entry_z"]:
                # Long spread (long A short B) if z low
                long_spread = za <= -params["pairs_entry_z"]
                notional = capital * params["risk_per_trade"] * 10  # leverage-ish on residual
                # CURVE-FIT RISK: sizing multiplier 10 is a placeholder — flagged
                size_a = notional / pa
                size_b = (notional * abs(float(b))) / pb
                entry_a = apply_entry_price(pa, "long" if long_spread else "short", costs)
                entry_b = apply_entry_price(pb, "short" if long_spread else "long", costs)
                pos = {
                    "long_spread": long_spread,
                    "size_a": size_a,
                    "size_b": size_b,
                    "entry_a": entry_a,
                    "entry_b": entry_b,
                    "entry_time": ts,
                    "entry_z": float(za),
                    "capital_at_entry": capital,
                }
        else:
            exit_now = abs(za) <= params["pairs_exit_z"] or abs(za) >= params["stop_z"]
            if exit_now or i == len(work) - 1:
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
                reason = "stop" if abs(za) >= params["stop_z"] else (
                    "eod" if i == len(work) - 1 else "take_profit"
                )
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
                        pattern="pairs_two_leg",
                        symbol=symbol,
                        strategy=strategy_name,
                        meta={"entry_z": pos["entry_z"], "exit_z": float(za)},
                    )
                )
                equity_pts.append((ts, capital))
                pos = None

    idx = pd.DatetimeIndex([t for t, _ in equity_pts])
    eq = pd.Series([v for _, v in equity_pts], index=idx, name="equity")
    eq = eq[~eq.index.duplicated(keep="last")]
    flags = strat.curve_fit_flags + [
        "[CURVE-FIT RISK] pairs notional multiplier (risk_per_trade*10) is a placeholder — validate sizing separately."
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
