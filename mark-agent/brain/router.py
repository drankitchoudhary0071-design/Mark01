"""
brain/router.py
Route voice commands via Anthropic Claude, with keyword matching as fallback.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any, Callable, Optional

import pandas as pd
from dotenv import load_dotenv

from data.binance_fetch import fetch_ohlcv
from strategy.backtester import backtest
from strategy.generator import generate_strategy
from strategy.validator import validate
from voice.speaker import speak

# Load ANTHROPIC_API_KEY from mark-agent/.env (never hardcode secrets)
_ENV_PATH = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(_ENV_PATH)
load_dotenv()  # also allow process / cwd env

# Claude model + rough Sonnet-class pricing (USD per million tokens)
CLAUDE_MODEL = "claude-sonnet-5"
INPUT_COST_PER_MTOK = 3.00
OUTPUT_COST_PER_MTOK = 15.00

# Valid actions Claude may choose
VALID_ACTIONS = {
    "fetch_data",
    "generate_strategy",
    "backtest",
    "validate",
    "general_answer",
}


# Shared session state so commands can chain (fetch → generate → backtest → validate)
class AgentState:
    def __init__(self) -> None:
        self.market_data: Optional[pd.DataFrame] = None
        self.strategies: list[dict[str, Any]] = []
        self.last_strategy: Optional[dict[str, Any]] = None
        self.last_backtest: Optional[dict[str, Any]] = None
        self.last_validation: Optional[dict[str, Any]] = None


state = AgentState()

# Running totals for this process
_usage_totals = {"input_tokens": 0, "output_tokens": 0, "cost_usd": 0.0}


def _hindi_metrics(metrics: dict[str, Any]) -> str:
    """Build a short Hindi summary of backtest numbers."""
    return (
        f"जीत दर {metrics.get('win_rate', 0) * 100:.1f} प्रतिशत, "
        f"कुल ट्रेड {metrics.get('total_trades', 0)}, "
        f"औसत आर {metrics.get('avg_R', 0):.2f}, "
        f"मैक्स ड्रॉडाउन {metrics.get('max_drawdown', 0) * 100:.1f} प्रतिशत, "
        f"शार्प {metrics.get('sharpe_ratio', 0):.2f}"
    )


def _summarize_market_context(market_context: Optional[dict[str, Any]]) -> str:
    """Turn market_context into a short string for the Claude prompt."""
    if not market_context:
        # Fall back to live session state
        ctx: dict[str, Any] = {
            "has_market_data": state.market_data is not None,
            "candles": len(state.market_data) if state.market_data is not None else 0,
            "strategy_count": len(state.strategies),
            "last_strategy": (state.last_strategy or {}).get("name"),
            "last_backtest": state.last_backtest,
            "last_validation_flag": (state.last_validation or {}).get("flag"),
        }
        if state.market_data is not None and len(state.market_data) > 0:
            last = state.market_data.iloc[-1]
            ctx["last_close"] = float(last["close"])
            ctx["last_timestamp"] = str(last["timestamp"])
        return json.dumps(ctx, ensure_ascii=False, default=str)

    # Shallow-copy and trim any raw DataFrame
    safe = dict(market_context)
    data = safe.pop("market_data", None)
    if data is not None and hasattr(data, "__len__"):
        safe["candles"] = len(data)
        try:
            safe["last_close"] = float(data["close"].iloc[-1])
        except Exception:
            pass
    return json.dumps(safe, ensure_ascii=False, default=str)


def _log_token_usage(input_tokens: int, output_tokens: int) -> dict[str, Any]:
    """Print per-call token usage + estimated cost; update running totals."""
    cost = (input_tokens / 1_000_000) * INPUT_COST_PER_MTOK + (
        output_tokens / 1_000_000
    ) * OUTPUT_COST_PER_MTOK
    _usage_totals["input_tokens"] += input_tokens
    _usage_totals["output_tokens"] += output_tokens
    _usage_totals["cost_usd"] += cost

    print(
        f"📊 Claude tokens — input: {input_tokens}, output: {output_tokens}, "
        f"est. cost: ${cost:.6f} "
        f"(session total: ${_usage_totals['cost_usd']:.6f})"
    )
    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "cost_usd": round(cost, 6),
        "session_cost_usd": round(_usage_totals["cost_usd"], 6),
    }


def _parse_claude_json(text: str) -> dict[str, Any]:
    """Extract a JSON object from Claude's reply (allows markdown fences)."""
    text = text.strip()
    # Prefer fenced ```json ... ``` block if present
    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fence:
        return json.loads(fence.group(1))
    # Or first {...} blob
    brace = re.search(r"\{.*\}", text, re.DOTALL)
    if brace:
        return json.loads(brace.group(0))
    raise ValueError(f"No JSON object found in Claude reply: {text[:200]}")


def _ask_claude(voice_text: str, market_context: Optional[dict[str, Any]]) -> dict[str, Any]:
    """
    Call Anthropic Claude to choose an action.
    Returns parsed dict with at least {"action": str, "reply": str}.
    Raises on missing key / API / parse errors (caller falls back to keywords).
    """
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError(
            "ANTHROPIC_API_KEY not set. Add it to mark-agent/.env or the environment."
        )

    # Import here so keyword fallback still works if anthropic is not installed
    from anthropic import Anthropic

    client = Anthropic(api_key=api_key)
    context_str = _summarize_market_context(market_context)

    system = (
        "You are the brain of mark-agent, a Hindi/English voice trading assistant.\n"
        "Given the user's voice command and market context, choose exactly ONE action.\n\n"
        "Allowed actions:\n"
        "- fetch_data: download OHLCV candles from Binance\n"
        "- generate_strategy: invent one new rule-based strategy\n"
        "- backtest: backtest the latest strategy\n"
        "- validate: 70/30 train/test validation (overfit check)\n"
        "- general_answer: answer a question or chit-chat; put spoken Hindi reply in 'reply'\n\n"
        "Respond with ONLY valid JSON (no extra prose):\n"
        '{"action": "<one of the actions above>", "reply": "<short Hindi text to speak '
        'for general_answer; empty string for tool actions>"}\n'
    )
    user_msg = (
        f"Voice command: {voice_text}\n"
        f"Market context: {context_str}\n"
    )

    message = client.messages.create(
        model=CLAUDE_MODEL,
        max_tokens=300,
        system=system,
        messages=[{"role": "user", "content": user_msg}],
    )

    input_tokens = int(getattr(message.usage, "input_tokens", 0) or 0)
    output_tokens = int(getattr(message.usage, "output_tokens", 0) or 0)
    _log_token_usage(input_tokens, output_tokens)

    # Concatenate text blocks from the response
    parts = []
    for block in message.content:
        if getattr(block, "type", None) == "text":
            parts.append(block.text)
        elif isinstance(block, dict) and block.get("type") == "text":
            parts.append(block.get("text", ""))
    raw = "\n".join(parts).strip()
    parsed = _parse_claude_json(raw)

    action = str(parsed.get("action", "")).strip().lower()
    if action not in VALID_ACTIONS:
        raise ValueError(f"Unknown action from Claude: {action!r}")
    parsed["action"] = action
    parsed["reply"] = str(parsed.get("reply", "") or "")
    return parsed


# ---------------------------------------------------------------------------
# Action handlers (execute + speak Hindi result)
# ---------------------------------------------------------------------------

def action_fetch(_command: str = "") -> str:
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


def action_generate(_command: str = "") -> str:
    """Generate one new strategy from current market data."""
    if state.market_data is None:
        try:
            state.market_data = fetch_ohlcv()
        except Exception as exc:
            msg = f"रणनीति नहीं बन सकी। पहले डेटा चाहिए। त्रुटि: {exc}"
            speak(msg)
            return msg

    strategy = generate_strategy(state.market_data, state.strategies)
    state.strategies.append(strategy)
    state.last_strategy = strategy

    name = strategy.get("name", "नामहीन")
    rationale = strategy.get("rationale", "")
    msg = f"नई रणनीति बनी: {name}। {rationale}"
    speak(msg)
    return msg


def action_backtest(_command: str = "") -> str:
    """Backtest the last generated strategy."""
    try:
        if state.market_data is None:
            state.market_data = fetch_ohlcv()
        if state.last_strategy is None:
            state.last_strategy = generate_strategy(state.market_data, state.strategies)
            state.strategies.append(state.last_strategy)
    except Exception as exc:
        msg = f"बैकटेस्ट नहीं हो सका। त्रुटि: {exc}"
        speak(msg)
        return msg

    metrics = backtest(state.last_strategy, state.market_data)
    state.last_backtest = metrics
    msg = "बैकटेस्ट पूरा। " + _hindi_metrics(metrics)
    speak(msg)
    return msg


def action_validate(_command: str = "") -> str:
    """Validate last strategy with 70/30 train/test split."""
    try:
        if state.market_data is None:
            state.market_data = fetch_ohlcv()
        if state.last_strategy is None:
            state.last_strategy = generate_strategy(state.market_data, state.strategies)
            state.strategies.append(state.last_strategy)
    except Exception as exc:
        msg = f"सत्यापन नहीं हो सका। त्रुटि: {exc}"
        speak(msg)
        return msg

    result = validate(state.last_strategy, state.market_data)
    state.last_validation = result

    flag = result.get("flag", "OK")
    if flag == "OVERFIT":
        msg = "सत्यापन पूरा। चेतावनी: ओवरफिट। " + result.get("message", "")
    elif flag == "INSUFFICIENT_DATA":
        msg = "सत्यापन नहीं हो सका। डेटा कम है।"
    else:
        msg = "सत्यापन ठीक है। " + result.get("message", "")

    speak(msg)
    return msg


def action_general_answer(reply: str, _command: str = "") -> str:
    """Speak Claude's general reply (or a short default)."""
    msg = reply.strip() if reply and reply.strip() else (
        "मैं ट्रेडिंग डेटा, रणनीति, बैकटेस्ट और सत्यापन में मदद कर सकता हूँ।"
    )
    speak(msg)
    return msg


def action_help(_command: str = "") -> str:
    msg = (
        "आप कह सकते हैं: डेटा लाओ, रणनीति बनाओ, बैकटेस्ट करो, "
        "या सत्यापन करो। बंद करने के लिए बंद कहें।"
    )
    speak(msg)
    return msg


def action_quit(_command: str = "") -> str:
    msg = "ठीक है, बंद कर रहा हूँ।"
    speak(msg)
    return msg


# Map Claude action names → callables
CLAUDE_ACTION_MAP: dict[str, Callable[..., str]] = {
    "fetch_data": action_fetch,
    "generate_strategy": action_generate,
    "backtest": action_backtest,
    "validate": action_validate,
}


# Keyword → handler map (Hindi + English) — fallback if Claude fails
# Checked in order; first match wins.
ROUTES: list[tuple[list[str], Callable[..., str]]] = [
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


def route_keyword(command_text: str) -> str:
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


# Keep old name so existing imports keep working as keyword fallback
def route(command_text: str) -> str:
    """Alias for keyword routing (used as Claude fallback)."""
    return route_keyword(command_text)


def route_command(
    voice_text: str,
    market_context: Optional[dict[str, Any]] = None,
) -> str:
    """
    Route a voice command with Claude (claude-sonnet-5).

    1. Send voice_text + market_context to Claude
    2. Claude picks: fetch_data / generate_strategy / backtest / validate / general_answer
    3. Parse JSON and run the matching function
    4. Log input/output tokens and estimated cost after every call
    5. Return the final Hindi result (also spoken)

    Falls back to keyword routing if the API key is missing or the call fails.
    """
    if not voice_text or not voice_text.strip():
        msg = "कुछ सुनाई नहीं दिया। कृपया दोबारा बोलें।"
        speak(msg)
        return msg

    # Quick local quit check — no need to spend tokens
    lowered = voice_text.strip().lower()
    if any(k in lowered for k in ("बंद", "quit", "exit", "goodbye", "बाय")):
        return action_quit(voice_text)

    try:
        decision = _ask_claude(voice_text, market_context)
        action = decision["action"]
        reply = decision.get("reply", "")
        print(f"🧠 Claude chose action: {action}")

        if action == "general_answer":
            return action_general_answer(reply, voice_text)

        handler = CLAUDE_ACTION_MAP.get(action)
        if handler is None:
            raise ValueError(f"No handler for action {action}")
        return handler(voice_text)

    except Exception as exc:
        print(f"⚠ Claude routing failed ({exc}); falling back to keywords.")
        return route_keyword(voice_text)
