"""
brain/router.py
Route voice command text to the right action using simple keyword matching.
(LLM-based routing can replace this later.)
"""

from __future__ import annotations

from typing import Any, Callable

import pandas as pd

from data.binance_fetch import fetch_ohlcv
from strategy.backtester import backtest
from strategy.generator import generate_strategy
from strategy.validator import validate
from voice.speaker import speak


# Shared session state so commands can chain (fetch → generate → backtest → validate)
class AgentState:
    def __init__(self) -> None:
        self.market_data: pd.DataFrame | None = None
        self.strategies: list[dict[str, Any]] = []
        self.last_strategy: dict[str, Any] | None = None
        self.last_backtest: dict[str, Any] | None = None
        self.last_validation: dict[str, Any] | None = None


state = AgentState()


def _hindi_metrics(metrics: dict[str, Any]) -> str:
    """Build a short Hindi summary of backtest numbers."""
    return (
        f"जीत दर {metrics.get('win_rate', 0) * 100:.1f} प्रतिशत, "
        f"कुल ट्रेड {metrics.get('total_trades', 0)}, "
        f"औसत आर {metrics.get('avg_R', 0):.2f}, "
        f"मैक्स ड्रॉडाउन {metrics.get('max_drawdown', 0) * 100:.1f} प्रतिशत, "
        f"शार्प {metrics.get('sharpe_ratio', 0):.2f}"
    )


def action_fetch(_command: str) -> str:
    """Fetch OHLCV data from Binance."""
    try:
        df = fetch_ohlcv(symbol="PAXGUSDT", interval="5m", limit=1000)
    except Exception as exc:  # network / geo / API errors
        msg = f"डेटा नहीं मिल सका। त्रुटि: {exc}"
        speak(msg)
        return msg
    state.market_data = df
    msg = f"डेटा तैयार है। {len(df)} कैंडल्स लाई गईं। पीएएक्सजी यूएसडीटी पाँच मिनट।"
    speak(msg)
    return msg


def action_generate(_command: str) -> str:
    """Generate one new strategy from current market data."""
    if state.market_data is None:
        # Auto-fetch if user skipped the fetch step
        state.market_data = fetch_ohlcv()

    strategy = generate_strategy(state.market_data, state.strategies)
    state.strategies.append(strategy)
    state.last_strategy = strategy

    name = strategy.get("name", "नामहीन")
    rationale = strategy.get("rationale", "")
    msg = f"नई रणनीति बनी: {name}। {rationale}"
    speak(msg)
    return msg


def action_backtest(_command: str) -> str:
    """Backtest the last generated strategy."""
    if state.market_data is None:
        state.market_data = fetch_ohlcv()
    if state.last_strategy is None:
        state.last_strategy = generate_strategy(state.market_data, state.strategies)
        state.strategies.append(state.last_strategy)

    metrics = backtest(state.last_strategy, state.market_data)
    state.last_backtest = metrics
    msg = "बैकटेस्ट पूरा। " + _hindi_metrics(metrics)
    speak(msg)
    return msg


def action_validate(_command: str) -> str:
    """Validate last strategy with 70/30 train/test split."""
    if state.market_data is None:
        state.market_data = fetch_ohlcv()
    if state.last_strategy is None:
        state.last_strategy = generate_strategy(state.market_data, state.strategies)
        state.strategies.append(state.last_strategy)

    result = validate(state.last_strategy, state.market_data)
    state.last_validation = result

    flag = result.get("flag", "OK")
    if flag == "OVERFIT":
        msg = (
            "सत्यापन पूरा। चेतावनी: ओवरफिट। "
            + result.get("message", "")
        )
    elif flag == "INSUFFICIENT_DATA":
        msg = "सत्यापन नहीं हो सका। डेटा कम है।"
    else:
        msg = "सत्यापन ठीक है। " + result.get("message", "")

    speak(msg)
    return msg


def action_help(_command: str) -> str:
    msg = (
        "आप कह सकते हैं: डेटा लाओ, रणनीति बनाओ, बैकटेस्ट करो, "
        "या सत्यापन करो। बंद करने के लिए बंद कहें।"
    )
    speak(msg)
    return msg


def action_quit(_command: str) -> str:
    msg = "ठीक है, बंद कर रहा हूँ।"
    speak(msg)
    return msg


# Keyword → handler map (Hindi + English)
# Checked in order; first match wins.
# More specific intents (validate / backtest / quit) come before broad words
# like "strategy" or "data" so "validate strategy" routes correctly.
ROUTES: list[tuple[list[str], Callable[[str], str]]] = [
    (
        ["quit", "exit", "stop", "बंद", "बाय", "goodbye"],
        action_quit,
    ),
    (
        ["validate", "validation", "सत्यापन", "ओवरफिट", "overfit", "जांच"],
        action_validate,
    ),
    (
        ["backtest", "बैकटेस्ट", "बैक टेस्ट", "back test", "परीक्षण"],
        action_backtest,
    ),
    (
        [
            "fetch",
            "ohlcv",
            "candle",
            "डेटा",
            "डाटा",
            "डेटा लाओ",
            "बाजार",
            "market data",
            "market",
            "fetch data",
        ],
        action_fetch,
    ),
    (
        [
            "generate",
            "strategy",
            "नयी रणनीति",
            "नई रणनीति",
            "रणनीति",
            "बनाओ",
            "तैयार",
            "create",
        ],
        action_generate,
    ),
    (
        ["help", "मदद", "हेल्प", "क्या कर"],
        action_help,
    ),
]


def route(command_text: str) -> str:
    """
    Match voice command text to an action via simple keywords.
    Speaks the result in Hindi and returns the same message string.
    """
    if not command_text or not command_text.strip():
        msg = "कुछ सुनाई नहीं दिया। कृपया दोबारा बोलें।"
        speak(msg)
        return msg

    text = command_text.strip().lower()

    for keywords, handler in ROUTES:
        for kw in keywords:
            if kw.lower() in text:
                return handler(command_text)

    msg = (
        f"समझ नहीं आया: {command_text}। "
        "डेटा, रणनीति, बैकटेस्ट या सत्यापन कहें।"
    )
    speak(msg)
    return msg
