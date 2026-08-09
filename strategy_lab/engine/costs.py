"""Shared cost model and fill helpers."""

from __future__ import annotations

from dataclasses import dataclass

from strategy_lab import config as cfg


@dataclass
class CostModel:
    commission_rate: float = cfg.COMMISSION_RATE
    half_spread: float = cfg.HALF_SPREAD
    slippage: float = cfg.SLIPPAGE

    @property
    def adverse_bps(self) -> float:
        """Adverse price move fraction on one side (spread + slip)."""
        return self.half_spread + self.slippage

    def round_trip_friction(self) -> float:
        """Approx fraction cost for a round-trip taker trade."""
        return 2.0 * (self.commission_rate + self.adverse_bps)


def apply_entry_price(raw: float, direction: str, costs: CostModel) -> float:
    slip = costs.adverse_bps
    if direction == "long":
        return raw * (1 + slip)
    return raw * (1 - slip)


def apply_exit_price(raw: float, direction: str, costs: CostModel) -> float:
    slip = costs.adverse_bps
    if direction == "long":
        return raw * (1 - slip)
    return raw * (1 + slip)
