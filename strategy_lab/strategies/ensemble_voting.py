"""
20. Multi-agent Ensemble Voting.

Combines signals from ≥3 prior families into a weighted vote. Entry only
when net vote reaches ``min_votes``. Stops/TPs use ATR multiples (shared
with the engine), not agent-specific levels — keeps risk model comparable.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from strategy_lab.engine import indicators as ind
from strategy_lab.engine.types import Signal
from strategy_lab.strategies.base import Strategy, atr_stop_tp
from strategy_lab.strategies.trend_following import TrendFollowingStrategy
from strategy_lab.strategies.mean_reversion import MeanReversionStrategy
from strategy_lab.strategies.regime_momentum import RegimeMomentumStrategy
from strategy_lab.strategies.volatility_squeeze import VolatilitySqueezeStrategy


class EnsembleVotingStrategy(Strategy):
    name = "ensemble_voting"
    library = "multi-agent vote + shared engine"

    @classmethod
    def default_params(cls) -> dict[str, Any]:
        return {
            # Equal weights by default — do NOT optimize on one backtest
            "weight_trend_following": 1.0,
            "weight_mean_reversion": 1.0,
            "weight_regime_momentum": 1.0,
            "weight_volatility_squeeze": 1.0,
            "min_votes": 2.0,  # need net |vote| >= 2 with unit weights
            "vote_window_bars": 3,  # align signals within ±N bars
            "atr_period": 14,
            "stop_atr": 1.5,
            "tp_atr": 2.5,
            "cooldown_bars": 6,
        }

    def __init__(self, **params: Any):
        super().__init__(**params)
        self.curve_fit_flags = list(self.curve_fit_flags) + [
            "[CURVE-FIT RISK] Do not optimize agent weights on a single IS window — "
            "defaults are equal weights.",
        ]

    def generate_signals(self, df: pd.DataFrame) -> list[Signal]:
        p = self.params
        agents: list[tuple[str, Strategy, float]] = [
            ("trend_following", TrendFollowingStrategy(), float(p["weight_trend_following"])),
            ("mean_reversion", MeanReversionStrategy(), float(p["weight_mean_reversion"])),
            ("regime_momentum", RegimeMomentumStrategy(), float(p["weight_regime_momentum"])),
            ("volatility_squeeze", VolatilitySqueezeStrategy(), float(p["weight_volatility_squeeze"])),
        ]
        # Collect signed votes at signal timestamps
        raw_votes: list[tuple[pd.Timestamp, float]] = []
        for _name, agent, w in agents:
            if w == 0:
                continue
            for sig in agent.generate_signals(df):
                ts = pd.Timestamp(sig.timestamp)
                signed = w if sig.direction == "long" else -w
                raw_votes.append((ts, signed))

        if not raw_votes:
            return []

        out = df.copy()
        out["atr"] = ind.atr(out, int(p["atr_period"]))
        out["timestamp"] = pd.to_datetime(out["timestamp"], utc=True)
        ts_index = pd.DatetimeIndex(out["timestamp"])

        # Aggregate agent votes onto bars; within ±vote_window, sum contributions
        win = int(p["vote_window_bars"])
        bar_vote = np.zeros(len(out), dtype=float)
        for ts, v in raw_votes:
            idx = int(ts_index.searchsorted(ts, side="left"))
            if idx >= len(out):
                idx = len(out) - 1
            # If timestamp falls between bars, snap to nearest
            if idx > 0 and abs((ts_index[idx] - ts).total_seconds()) > abs(
                (ts_index[idx - 1] - ts).total_seconds()
            ):
                idx -= 1
            lo = max(0, idx - win)
            hi = min(len(out), idx + win + 1)
            bar_vote[lo:hi] += v

        signals: list[Signal] = []
        last_i = -10_000
        cooldown = int(p["cooldown_bars"])
        min_votes = float(p["min_votes"])

        for i, net in enumerate(bar_vote):
            if i - last_i < cooldown:
                continue
            row = out.iloc[i]
            if pd.isna(row["atr"]) or row["atr"] <= 0:
                continue
            if net >= min_votes:
                direction = "long"
            elif net <= -min_votes:
                direction = "short"
            else:
                continue
            entry = float(row["close"])
            stop, tp = atr_stop_tp(
                entry, direction, float(row["atr"]), p["stop_atr"], p["tp_atr"]
            )
            signals.append(
                Signal(
                    timestamp=pd.Timestamp(row["timestamp"]),
                    direction=direction,  # type: ignore[arg-type]
                    entry=entry,
                    stop=stop,
                    take_profit=tp,
                    pattern="ensemble_vote",
                    meta={"net_vote": float(net)},
                )
            )
            last_i = i
        return signals
