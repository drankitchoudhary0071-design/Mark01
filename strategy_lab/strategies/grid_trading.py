"""
4. Grid Trading — fixed-interval buy/sell grid (mean-reverting inventory).

Uses a dedicated simulator (not discretionary signal/SL-TP) but emits the
same BacktestResult / Trade schema for cross-strategy comparison.
"""

from __future__ import annotations

from typing import Any, Optional

import pandas as pd

from strategy_lab.engine.costs import CostModel, apply_entry_price, apply_exit_price
from strategy_lab.engine.types import BacktestResult, Signal, Trade
from strategy_lab.strategies.base import Strategy


class GridTradingStrategy(Strategy):
    name = "grid_trading"
    library = "pandas grid simulator"

    @classmethod
    def default_params(cls) -> dict[str, Any]:
        return {
            "grid_spacing_pct": 0.005,  # CURVE-FIT RISK — asset/ATR dependent
            "grid_levels": 5,
            "order_notional_frac": 0.08,  # each level ≈ 8% of capital
            "recenter_pct": 0.05,  # rebuild grid if price drifts 5% from anchor
            "stop_grid_dd": 0.15,  # kill grid if equity DD from start > 15%
        }

    def generate_signals(self, df: pd.DataFrame) -> list[Signal]:
        """Grid does not use discretionary signals; runner calls simulate_grid."""
        return []


def simulate_grid(
    df: pd.DataFrame,
    params: Optional[dict] = None,
    *,
    initial_capital: float = 10_000.0,
    costs: Optional[CostModel] = None,
    strategy_name: str = "grid_trading",
    symbol: str = "",
) -> BacktestResult:
    params = {**GridTradingStrategy.default_params(), **(params or {})}
    costs = costs or CostModel()
    strat = GridTradingStrategy(**params)

    spacing = float(params["grid_spacing_pct"])
    levels = int(params["grid_levels"])
    capital = float(initial_capital)
    peak = capital
    cash = capital
    inv = 0.0  # base asset inventory
    avg_entry = 0.0
    trades: list[Trade] = []
    equity_pts: list[tuple[pd.Timestamp, float]] = []

    anchor = float(df["close"].iloc[0])
    buy_levels = [anchor * (1 - spacing * k) for k in range(1, levels + 1)]
    sell_levels = [anchor * (1 + spacing * k) for k in range(1, levels + 1)]
    filled_buys: set[int] = set()
    filled_sells: set[int] = set()

    def equity(price: float) -> float:
        return cash + inv * price

    for i in range(len(df)):
        ts = pd.Timestamp(df["timestamp"].iloc[i])
        o = float(df["open"].iloc[i])
        h = float(df["high"].iloc[i])
        l = float(df["low"].iloc[i])
        c = float(df["close"].iloc[i])
        eq = equity(c)
        peak = max(peak, eq)
        if (peak - eq) / peak >= params["stop_grid_dd"]:
            # Flatten
            if inv != 0:
                fill = apply_exit_price(c, "long" if inv > 0 else "short", costs)
                pnl = inv * (fill - avg_entry) - abs(inv) * fill * costs.commission_rate
                cash += inv * fill - abs(inv) * fill * costs.commission_rate
                trades.append(
                    Trade(
                        entry_time=ts,
                        exit_time=ts,
                        direction="long" if inv > 0 else "short",
                        entry_price=avg_entry,
                        exit_price=fill,
                        stop=0.0,
                        take_profit=0.0,
                        size=abs(inv),
                        pnl=pnl,
                        pnl_pct=pnl / initial_capital,
                        return_R=pnl / (initial_capital * 0.01),
                        exit_reason="grid_kill",
                        pattern="grid_flatten",
                        symbol=symbol,
                        strategy=strategy_name,
                    )
                )
                inv = 0.0
                avg_entry = 0.0
            capital = cash
            equity_pts.append((ts, cash))
            break

        # Recenter
        if abs(c / anchor - 1.0) >= params["recenter_pct"]:
            anchor = c
            buy_levels = [anchor * (1 - spacing * k) for k in range(1, levels + 1)]
            sell_levels = [anchor * (1 + spacing * k) for k in range(1, levels + 1)]
            filled_buys.clear()
            filled_sells.clear()

        notional = equity(c) * float(params["order_notional_frac"])

        # Buys when bar trades down through a buy level
        for ki, level in enumerate(buy_levels):
            if ki in filled_buys:
                continue
            touched = (l <= level <= h) or (o >= level >= l)
            if not touched:
                continue
            fill = apply_entry_price(level, "long", costs)
            size = notional / fill
            if size * fill > cash:
                continue
            fee = size * fill * costs.commission_rate
            cash -= size * fill + fee
            new_inv = inv + size
            avg_entry = (
                (avg_entry * inv + fill * size) / new_inv if new_inv else fill
            )
            inv = new_inv
            filled_buys.add(ki)
            # paired sell re-arms opposite level
            if ki in filled_sells:
                filled_sells.discard(ki)
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
                    pnl=-fee,
                    pnl_pct=-fee / initial_capital,
                    return_R=-fee / (initial_capital * 0.01),
                    exit_reason="grid_buy",
                    pattern=f"grid_buy_L{ki+1}",
                    symbol=symbol,
                    strategy=strategy_name,
                )
            )

        # Sells when bar trades up through a sell level
        for ki, level in enumerate(sell_levels):
            if ki in filled_sells:
                continue
            touched = (l <= level <= h) or (o <= level <= h)
            if not touched:
                continue
            if inv <= 0:
                continue
            fill = apply_exit_price(level, "long", costs)
            size = min(inv, notional / fill)
            fee = size * fill * costs.commission_rate
            pnl = size * (fill - avg_entry) - fee
            cash += size * fill - fee
            inv -= size
            if inv <= 1e-12:
                inv = 0.0
                avg_entry = 0.0
            filled_sells.add(ki)
            if ki in filled_buys:
                filled_buys.discard(ki)
            trades.append(
                Trade(
                    entry_time=ts,
                    exit_time=ts,
                    direction="long",
                    entry_price=avg_entry if avg_entry else fill,
                    exit_price=fill,
                    stop=0.0,
                    take_profit=0.0,
                    size=size,
                    pnl=pnl,
                    pnl_pct=pnl / initial_capital,
                    return_R=pnl / (initial_capital * 0.01),
                    exit_reason="grid_sell",
                    pattern=f"grid_sell_L{ki+1}",
                    symbol=symbol,
                    strategy=strategy_name,
                )
            )
            equity_pts.append((ts, equity(c)))

        if i % 24 == 0:
            equity_pts.append((ts, equity(c)))

    # Mark to market final
    last_ts = pd.Timestamp(df["timestamp"].iloc[-1])
    last_c = float(df["close"].iloc[-1])
    final = cash + inv * last_c
    equity_pts.append((last_ts, final))
    idx = pd.DatetimeIndex([t for t, _ in equity_pts])
    eq = pd.Series([v for _, v in equity_pts], index=idx, name="equity")
    eq = eq[~eq.index.duplicated(keep="last")]

    return BacktestResult(
        trades=trades,
        equity=eq,
        initial_capital=initial_capital,
        final_capital=final,
        strategy=strategy_name,
        symbol=symbol,
        params=params,
        curve_fit_flags=strat.curve_fit_flags,
    )
