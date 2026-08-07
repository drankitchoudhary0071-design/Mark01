"""
9. Market Making — bid/ask spread capture (simulation only).

This is NOT live market making. We simulate posting at mid ± half_spread,
assuming a fill probability when the bar range crosses our quote, then
paying commission and an inventory penalty. Useful as a research baseline
for whether quoted edge survives costs — not a production MM system.
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np
import pandas as pd

from strategy_lab import config as cfg
from strategy_lab.engine.costs import CostModel
from strategy_lab.engine.types import BacktestResult, Signal, Trade
from strategy_lab.strategies.base import Strategy


class MarketMakingStrategy(Strategy):
    name = "market_making"
    library = "pandas MM simulator (not live)"

    @classmethod
    def default_params(cls) -> dict[str, Any]:
        return {
            "quote_half_spread": cfg.MM_HALF_SPREAD_CAPTURE,  # CURVE-FIT RISK if widened
            "inventory_penalty": cfg.MM_INVENTORY_PENALTY,
            "max_inventory_notional_frac": 0.25,
            "order_notional_frac": 0.02,
            "fill_prob": 0.35,  # CURVE-FIT RISK — synthetic fill model
            "seed": 42,
        }

    def generate_signals(self, df: pd.DataFrame) -> list[Signal]:
        return []


def simulate_market_making(
    df: pd.DataFrame,
    params: Optional[dict] = None,
    *,
    initial_capital: float = 10_000.0,
    costs: Optional[CostModel] = None,
    strategy_name: str = "market_making",
    symbol: str = "",
) -> BacktestResult:
    params = {**MarketMakingStrategy.default_params(), **(params or {})}
    costs = costs or CostModel()
    strat = MarketMakingStrategy(**params)
    rng = np.random.default_rng(int(params["seed"]))

    cash = float(initial_capital)
    inv = 0.0
    avg = 0.0
    trades: list[Trade] = []
    equity_pts: list[tuple[pd.Timestamp, float]] = []
    half = float(params["quote_half_spread"])
    inv_pen = float(params["inventory_penalty"])

    for i in range(len(df)):
        ts = pd.Timestamp(df["timestamp"].iloc[i])
        h = float(df["high"].iloc[i])
        l = float(df["low"].iloc[i])
        c = float(df["close"].iloc[i])
        mid = c
        eq = cash + inv * c
        max_inv_notional = eq * float(params["max_inventory_notional_frac"])
        order_notional = eq * float(params["order_notional_frac"])

        bid = mid * (1 - half)
        ask = mid * (1 + half)

        # Bid fill if low reaches bid
        if l <= bid and rng.random() < float(params["fill_prob"]):
            if inv * c < max_inv_notional:
                fill = bid  # we bought at bid (maker)
                size = order_notional / fill
                fee = size * fill * costs.commission_rate
                # Inventory skew penalty as adverse selection proxy
                skew = inv_pen * fill * size * (1 if inv > 0 else 0.5)
                cash -= size * fill + fee + skew
                new_inv = inv + size
                avg = (avg * inv + fill * size) / new_inv if new_inv else fill
                inv = new_inv
                trades.append(
                    Trade(
                        entry_time=ts,
                        exit_time=ts,
                        direction="long",
                        entry_price=fill,
                        exit_price=fill,
                        stop=0.0,
                        take_profit=0.0,
                        size=size,
                        pnl=-(fee + skew),
                        pnl_pct=-(fee + skew) / initial_capital,
                        return_R=-(fee + skew) / (initial_capital * 0.01),
                        exit_reason="mm_bid_fill",
                        pattern="mm_buy",
                        symbol=symbol,
                        strategy=strategy_name,
                    )
                )

        # Ask fill if high reaches ask
        if h >= ask and rng.random() < float(params["fill_prob"]):
            if inv > 0:
                fill = ask
                size = min(inv, order_notional / fill)
                fee = size * fill * costs.commission_rate
                pnl = size * (fill - avg) - fee
                cash += size * fill - fee
                inv -= size
                if inv <= 1e-12:
                    inv = 0.0
                    avg = 0.0
                trades.append(
                    Trade(
                        entry_time=ts,
                        exit_time=ts,
                        direction="long",
                        entry_price=avg if avg else fill,
                        exit_price=fill,
                        stop=0.0,
                        take_profit=0.0,
                        size=size,
                        pnl=pnl,
                        pnl_pct=pnl / initial_capital,
                        return_R=pnl / (initial_capital * 0.01),
                        exit_reason="mm_ask_fill",
                        pattern="mm_sell",
                        symbol=symbol,
                        strategy=strategy_name,
                    )
                )

        if i % 12 == 0:
            equity_pts.append((ts, cash + inv * c))

    last_ts = pd.Timestamp(df["timestamp"].iloc[-1])
    last_c = float(df["close"].iloc[-1])
    # Flatten residual inventory at adverse exit
    if inv > 0:
        fill = last_c * (1 - costs.adverse_bps)
        fee = inv * fill * costs.commission_rate
        pnl = inv * (fill - avg) - fee
        cash += inv * fill - fee
        trades.append(
            Trade(
                entry_time=last_ts,
                exit_time=last_ts,
                direction="long",
                entry_price=avg,
                exit_price=fill,
                stop=0.0,
                take_profit=0.0,
                size=inv,
                pnl=pnl,
                pnl_pct=pnl / initial_capital,
                return_R=pnl / (initial_capital * 0.01),
                exit_reason="mm_eod_flatten",
                pattern="mm_flatten",
                symbol=symbol,
                strategy=strategy_name,
            )
        )
        inv = 0.0

    final = cash
    equity_pts.append((last_ts, final))
    idx = pd.DatetimeIndex([t for t, _ in equity_pts])
    eq = pd.Series([v for _, v in equity_pts], index=idx, name="equity")
    eq = eq[~eq.index.duplicated(keep="last")]

    flags = strat.curve_fit_flags + [
        "[CURVE-FIT RISK] fill_prob is a synthetic assumption — not calibrated to real queue position.",
        "[SIMULATION ONLY] Market making results are illustrative; do not trade live from this module.",
    ]
    return BacktestResult(
        trades=trades,
        equity=eq,
        initial_capital=initial_capital,
        final_capital=final,
        strategy=strategy_name,
        symbol=symbol,
        params=params,
        curve_fit_flags=flags,
    )
