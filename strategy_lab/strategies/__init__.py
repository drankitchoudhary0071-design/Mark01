"""Strategy registry (20 families)."""

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
from strategy_lab.strategies.cointegration_pairs import CointegrationPairsStrategy
from strategy_lab.strategies.vwap_twap import VwapTwapStrategy
from strategy_lab.strategies.regime_adaptive import RegimeAdaptiveStrategy
from strategy_lab.strategies.adaptive_mean_reversion import AdaptiveMeanReversionStrategy
from strategy_lab.strategies.volume_confirmed_reversal import VolumeConfirmedReversalStrategy
from strategy_lab.strategies.rsi_divergence import RsiDivergenceStrategy
from strategy_lab.strategies.rebalance_arbitrage import RebalanceArbitrageStrategy
from strategy_lab.strategies.news_event import NewsEventStrategy
from strategy_lab.strategies.mtf_confluence import MultiTimeframeConfluenceStrategy
from strategy_lab.strategies.ensemble_voting import EnsembleVotingStrategy
from strategy_lab.strategies.btc_4h_rsi_mr import Btc4hRsiMrStrategy
from strategy_lab.strategies.mtf_daily_4h_rsi import MtfDaily4hRsiStrategy
from strategy_lab.strategies.mtf_rsi import MtfRsiStrategy
from strategy_lab.strategies.tv_mtf_indicators import TvMtfStrategy
from strategy_lab.strategies.scalp_ltf import ScalpMtfStrategy
from strategy_lab.strategies.pmts_fib_trailing import PmtsFibTrailingStrategy
from strategy_lab.strategies.pmts_4h_5m_bos import Pmts4h5mBosStrategy

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
    CointegrationPairsStrategy.name: CointegrationPairsStrategy,
    VwapTwapStrategy.name: VwapTwapStrategy,
    RegimeAdaptiveStrategy.name: RegimeAdaptiveStrategy,
    AdaptiveMeanReversionStrategy.name: AdaptiveMeanReversionStrategy,
    VolumeConfirmedReversalStrategy.name: VolumeConfirmedReversalStrategy,
    RsiDivergenceStrategy.name: RsiDivergenceStrategy,
    RebalanceArbitrageStrategy.name: RebalanceArbitrageStrategy,
    NewsEventStrategy.name: NewsEventStrategy,
    MultiTimeframeConfluenceStrategy.name: MultiTimeframeConfluenceStrategy,
    EnsembleVotingStrategy.name: EnsembleVotingStrategy,
    Btc4hRsiMrStrategy.name: Btc4hRsiMrStrategy,
    MtfDaily4hRsiStrategy.name: MtfDaily4hRsiStrategy,
    MtfRsiStrategy.name: MtfRsiStrategy,
    "tv_mtf": TvMtfStrategy,
    "scalp_ltf": ScalpMtfStrategy,
    PmtsFibTrailingStrategy.name: PmtsFibTrailingStrategy,
    Pmts4h5mBosStrategy.name: Pmts4h5mBosStrategy,
}

# Families that need specialized simulators (not plain signal→engine)
SPECIAL_SIM = {
    "statistical_arbitrage",
    "cointegration_pairs",
    "grid_trading",
    "market_making",
    "vwap_twap",
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
    "CointegrationPairsStrategy",
    "VwapTwapStrategy",
    "RegimeAdaptiveStrategy",
    "AdaptiveMeanReversionStrategy",
    "VolumeConfirmedReversalStrategy",
    "RsiDivergenceStrategy",
    "RebalanceArbitrageStrategy",
    "NewsEventStrategy",
    "MultiTimeframeConfluenceStrategy",
    "EnsembleVotingStrategy",
    "Btc4hRsiMrStrategy",
    "MtfDaily4hRsiStrategy",
    "MtfRsiStrategy",
    "TvMtfStrategy",
    "ScalpMtfStrategy",
    "PmtsFibTrailingStrategy",
    "Pmts4h5mBosStrategy",
]
