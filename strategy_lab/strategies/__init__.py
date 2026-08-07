"""Strategy registry."""

from __future__ import annotations

from typing import Type

from strategy_lab.strategies.base import Strategy
from strategy_lab.strategies.trend_following import TrendFollowingStrategy
from strategy_lab.strategies.mean_reversion import MeanReversionStrategy
from strategy_lab.strategies.statistical_arbitrage import StatisticalArbitrageStrategy
from strategy_lab.strategies.grid_trading import GridTradingStrategy
from strategy_lab.strategies.scalping import ScalpingStrategy
from strategy_lab.strategies.regime_momentum import RegimeMomentumStrategy
from strategy_lab.strategies.volatility_squeeze import VolatilitySqueezeStrategy
from strategy_lab.strategies.smc_fvg_ob import SmcFvgObStrategy
from strategy_lab.strategies.market_making import MarketMakingStrategy
from strategy_lab.strategies.momentum_vol_filter import MomentumVolFilterStrategy

STRATEGY_REGISTRY: dict[str, Type[Strategy]] = {
    TrendFollowingStrategy.name: TrendFollowingStrategy,
    MeanReversionStrategy.name: MeanReversionStrategy,
    StatisticalArbitrageStrategy.name: StatisticalArbitrageStrategy,
    GridTradingStrategy.name: GridTradingStrategy,
    ScalpingStrategy.name: ScalpingStrategy,
    RegimeMomentumStrategy.name: RegimeMomentumStrategy,
    VolatilitySqueezeStrategy.name: VolatilitySqueezeStrategy,
    SmcFvgObStrategy.name: SmcFvgObStrategy,
    MarketMakingStrategy.name: MarketMakingStrategy,
    MomentumVolFilterStrategy.name: MomentumVolFilterStrategy,
}

# Families that need specialized simulators (not plain signal→engine)
SPECIAL_SIM = {
    "statistical_arbitrage",
    "grid_trading",
    "market_making",
}

__all__ = [
    "STRATEGY_REGISTRY",
    "SPECIAL_SIM",
    "TrendFollowingStrategy",
    "MeanReversionStrategy",
    "StatisticalArbitrageStrategy",
    "GridTradingStrategy",
    "ScalpingStrategy",
    "RegimeMomentumStrategy",
    "VolatilitySqueezeStrategy",
    "SmcFvgObStrategy",
    "MarketMakingStrategy",
    "MomentumVolFilterStrategy",
]
