"""
Global defaults and parameter metadata.

CURVE-FIT POLICY
----------------
Any parameter that was (or looks like it was) tuned on a single historical
window is marked `curve_fit_risk=True` in PARAM_NOTES. Defaults below are
**literature / textbook starting points**, not optimized values. Treat them
as hypotheses to walk-forward validate — do not promote a tuned set to
defaults without documenting the search space and OOS degradation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


# ---------------------------------------------------------------------------
# Universe
# ---------------------------------------------------------------------------
DEFAULT_SYMBOLS: tuple[str, ...] = ("PAXGUSDT", "BTCUSDT")
DEFAULT_INTERVAL: str = "1h"
DEFAULT_DAYS: int = 365
# Scalping uses a tighter timeframe; pairs trading needs both legs.
SCALP_INTERVAL: str = "5m"
SCALP_DAYS: int = 60

# ---------------------------------------------------------------------------
# Capital / risk
# ---------------------------------------------------------------------------
INITIAL_CAPITAL: float = 10_000.0
RISK_PER_TRADE: float = 0.01  # 1% of equity risked per discretionary trade
MAX_NOTIONAL_FRAC: float = 0.95

# ---------------------------------------------------------------------------
# Cost model (Binance spot-like, intentionally conservative)
# Round-trip ≈ 2*(commission + half_spread + slippage) ≈ 0.30%
# ---------------------------------------------------------------------------
COMMISSION_RATE: float = 0.001  # 0.10% per side (taker)
HALF_SPREAD: float = 0.0002  # 0.02%
SLIPPAGE: float = 0.0003  # 0.03%

# Market-making simulation uses a tighter effective half-spread capture
# but still pays commission; inventory risk is modelled separately.
MM_HALF_SPREAD_CAPTURE: float = 0.00015  # CURVE-FIT RISK if raised without OOS proof
MM_INVENTORY_PENALTY: float = 0.00005

# ---------------------------------------------------------------------------
# Walk-forward
# ---------------------------------------------------------------------------
WF_TRAIN_BARS: int = 24 * 90  # ~90 days on 1h
WF_TEST_BARS: int = 24 * 30  # ~30 days on 1h
WF_STEP_BARS: int = 24 * 30  # roll forward by ~1 month
# For 5m scalp data, scale separately in runner.

# Simple IS/OOS holdout fraction when not using rolling WF
HOLDOUT_OOS_FRAC: float = 0.25


@dataclass(frozen=True)
class ParamNote:
    name: str
    default: Any
    rationale: str
    curve_fit_risk: bool = False
    note: str = ""


# Document defaults so reviewers can spot suspicious numbers.
PARAM_NOTES: list[ParamNote] = [
    ParamNote("fast_ma", 20, "Common short MA; textbook, not optimized", False),
    ParamNote("slow_ma", 50, "Common medium MA; textbook, not optimized", False),
    ParamNote("adx_period", 14, "Wilder default", False),
    ParamNote("adx_threshold", 25, "Textbook 'trending' ADX level", False,
              "Some families (regime_momentum) use 20 for event density — still not optimized."),
    ParamNote("bb_period", 20, "Bollinger default", False),
    ParamNote("bb_std", 2.0, "Bollinger default", False),
    ParamNote("rsi_period", 14, "Wilder default", False),
    ParamNote("rsi_oversold", 30, "Textbook oversold", False),
    ParamNote("rsi_overbought", 70, "Textbook overbought", False),
    ParamNote("pairs_lookback", 60, "Rolling hedge / z-score window", True,
              "Window length is often overfit; require WF stability."),
    ParamNote("pairs_entry_z", 2.0, "Common pairs entry threshold", False),
    ParamNote("pairs_exit_z", 0.5, "Common mean-reversion exit", True,
              "Exit z often tuned; flag if changed from 0.5 without WF."),
    ParamNote("grid_spacing_pct", 0.005, "0.5% grid — asset dependent", True,
              "Spacing must match ATR regime; do not hardcode from one run."),
    ParamNote("scalp_atr_stop", 0.8, "Tight ATR stop for scalp", True,
              "Tight stops are highly curve-fit prone under costs."),
    ParamNote("scalp_rr", 1.2, "Modest RR for scalp", True),
    ParamNote("squeeze_bb_kc_mult", 1.5, "TTM Squeeze-style KC mult", False),
    ParamNote("fvg_min_gap_atr", 0.25, "Min FVG size in ATR", True,
              "SMC thresholds are notoriously curve-fit; keep conservative."),
    ParamNote("ob_lookback", 20, "Order-block scan window", True),
    ParamNote("vol_regime_lookback", 48, "~2 days on 1h", False),
    ParamNote("mom_lookback", 24, "~1 day momentum on 1h", False),
]


@dataclass
class RunConfig:
    symbols: tuple[str, ...] = DEFAULT_SYMBOLS
    interval: str = DEFAULT_INTERVAL
    days: int = DEFAULT_DAYS
    initial_capital: float = INITIAL_CAPITAL
    risk_per_trade: float = RISK_PER_TRADE
    commission_rate: float = COMMISSION_RATE
    half_spread: float = HALF_SPREAD
    slippage: float = SLIPPAGE
    wf_train_bars: int = WF_TRAIN_BARS
    wf_test_bars: int = WF_TEST_BARS
    wf_step_bars: int = WF_STEP_BARS
    holdout_oos_frac: float = HOLDOUT_OOS_FRAC
    use_walk_forward: bool = True
    strategies: tuple[str, ...] = ()  # empty = all
    extra: dict[str, Any] = field(default_factory=dict)


def curve_fit_warnings(params: dict[str, Any]) -> list[str]:
    """Flag parameters documented as high curve-fit risk.

    Strategy-specific defaults (e.g. scalp ``rsi_period=7``) are *not* treated
    as PARAM DRIFT — only names marked ``curve_fit_risk=True`` in PARAM_NOTES
    are always flagged so reports stay honest without false alarms.
    """
    warnings: list[str] = []
    by_name = {p.name: p for p in PARAM_NOTES}
    for key, val in params.items():
        note = by_name.get(key)
        if note is None:
            continue
        if note.curve_fit_risk:
            warnings.append(
                f"[CURVE-FIT RISK] `{key}={val}` — {note.note or note.rationale}"
            )
    return warnings
