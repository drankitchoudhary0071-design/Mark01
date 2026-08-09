"""
12. VWAP / TWAP execution simulation (market-impact minimization).

Not a directional alpha strategy: compares slicing a parent order with
TWAP vs VWAP participation against an aggressive lump-sum fill.
Reports the execution PnL gap as trades so it ranks in the shared metrics
table (expect small / cost-dominated outcomes).
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np
import pandas as pd

from strategy_lab.engine.costs import CostModel, apply_entry_price
from strategy_lab.engine.types import BacktestResult, Signal, Trade
from strategy_lab.strategies.base import Strategy


class VwapTwapStrategy(Strategy):
    name = "vwap_twap"
    library = "pandas execution simulator"
    @classmethod
    def default_params(cls) -> dict[str, Any]:
        return {
            "parent_notional_frac": 0.20,  # each parent order ≈ 20% equity
            "slice_bars": 12,  # TWAP horizon (~12h on 1h)
            "vwap_window": 24,
            "impact_coeff": 0.10,  # temporary impact ∝ (child/vol)^0.5 — textbook Almgren-ish
            "schedule_every_bars": 48,  # fire a parent buy then sell alternately
            "side_mode": "alternating",  # buy/sell parents alternate
        }

    def generate_signals(self, df: pd.DataFrame) -> list[Signal]:
        return []


def simulate_vwap_twap(
    df: pd.DataFrame,
    params: Optional[dict] = None,
    *,
    initial_capital: float = 10_000.0,
    costs: Optional[CostModel] = None,
    strategy_name: str = "vwap_twap",
    symbol: str = "",
) -> BacktestResult:
    """
    For each scheduled parent order:
      - benchmark = aggressive mid fill + full adverse costs
      - TWAP = equal slices over ``slice_bars`` with impact
      - VWAP = volume-weighted slices over same horizon
    Record (benchmark_cost - algo_cost) as trade PnL (positive = algo saved money).
    """
    params = {**VwapTwapStrategy.default_params(), **(params or {})}
    costs = costs or CostModel()
    strat = VwapTwapStrategy(**params)

    capital = float(initial_capital)
    equity_pts: list[tuple[pd.Timestamp, float]] = [
        (pd.Timestamp(df["timestamp"].iloc[0]), capital)
    ]
    trades: list[Trade] = []
    slice_n = int(params["slice_bars"])
    every = int(params["schedule_every_bars"])
    impact_c = float(params["impact_coeff"])
    side_buy = True
    i = 0
    n = len(df)

    while i + slice_n < n:
        if i % every != 0:
            i += 1
            continue
        window = df.iloc[i : i + slice_n].reset_index(drop=True)
        ts0 = pd.Timestamp(window["timestamp"].iloc[0])
        ts1 = pd.Timestamp(window["timestamp"].iloc[-1])
        direction = "long" if side_buy else "short"
        notional = capital * float(params["parent_notional_frac"])
        mid0 = float(window["close"].iloc[0])
        if mid0 <= 0 or notional <= 0:
            i += every
            continue

        # Aggressive benchmark
        bench_px = apply_entry_price(mid0, direction, costs)
        bench_size = notional / bench_px
        bench_fee = bench_size * bench_px * costs.commission_rate

        # TWAP slices
        twap_cost = 0.0
        twap_qty = 0.0
        child_notional = notional / slice_n
        for j in range(slice_n):
            px = float(window["close"].iloc[j])
            vol = float(window["volume"].iloc[j]) + 1e-9
            child_qty = child_notional / px
            partic = child_qty / vol
            impact = impact_c * np.sqrt(max(partic, 0.0)) * px
            fill = px + impact if direction == "long" else px - impact
            fill = apply_entry_price(fill, direction, costs)
            fee = child_qty * fill * costs.commission_rate
            twap_cost += child_qty * fill + fee
            twap_qty += child_qty

        # VWAP slices — weight by bar volume
        vols = window["volume"].to_numpy(dtype=float) + 1e-9
        weights = vols / vols.sum()
        vwap_cost = 0.0
        vwap_qty = 0.0
        for j in range(slice_n):
            px = float(window["close"].iloc[j])
            child_n = notional * float(weights[j])
            child_qty = child_n / px
            partic = child_qty / vols[j]
            impact = impact_c * np.sqrt(max(partic, 0.0)) * px
            fill = px + impact if direction == "long" else px - impact
            fill = apply_entry_price(fill, direction, costs)
            fee = child_qty * fill * costs.commission_rate
            vwap_cost += child_qty * fill + fee
            vwap_qty += child_qty

        bench_cost = bench_size * bench_px + bench_fee
        # Savings vs aggressive (positive good for buys = paid less)
        if direction == "long":
            twap_pnl = bench_cost - twap_cost
            vwap_pnl = bench_cost - vwap_cost
        else:
            # For sells, higher proceeds = better
            twap_pnl = twap_cost - bench_cost
            vwap_pnl = vwap_cost - bench_cost

        # Record better of TWAP/VWAP as the strategy outcome this window
        if vwap_pnl >= twap_pnl:
            pnl, algo, fill_px = vwap_pnl, "vwap", (vwap_cost / vwap_qty if vwap_qty else mid0)
        else:
            pnl, algo, fill_px = twap_pnl, "twap", (twap_cost / twap_qty if twap_qty else mid0)

        risk = capital * 0.01
        capital += pnl
        trades.append(
            Trade(
                entry_time=ts0,
                exit_time=ts1,
                direction=direction,
                entry_price=bench_px,
                exit_price=float(fill_px),
                stop=0.0,
                take_profit=0.0,
                size=bench_size,
                pnl=float(pnl),
                pnl_pct=float(pnl) / initial_capital,
                return_R=float(pnl) / risk if risk else 0.0,
                exit_reason="exec_complete",
                pattern=algo,
                symbol=symbol,
                strategy=strategy_name,
                meta={"bench_cost": bench_cost, "twap_pnl": twap_pnl, "vwap_pnl": vwap_pnl},
            )
        )
        equity_pts.append((ts1, capital))
        side_buy = not side_buy
        i += every

    idx = pd.DatetimeIndex([t for t, _ in equity_pts])
    eq = pd.Series([v for _, v in equity_pts], index=idx, name="equity")
    eq = eq[~eq.index.duplicated(keep="last")]
    flags = strat.curve_fit_flags + [
        "[NOTE] Execution sim — PnL is savings vs aggressive fill, not directional alpha.",
        "[CURVE-FIT RISK] impact_coeff is a placeholder Almgren-style coefficient.",
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
