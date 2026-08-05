"""
main.py — entry point for mark-agent

Loop: listen → route (Claude, keyword fallback) → execute → speak
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure project root is on sys.path when run as `python main.py`
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from brain.router import route_command, state
from voice.listener import Listener
from voice.speaker import speak


def _market_context() -> dict:
    """Build a small context dict for Claude from current session state."""
    ctx = {
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
    return ctx


def main() -> None:
    """Run the listen → route → execute → speak loop until quit."""
    listener = Listener(language="hi-IN")

    speak("मार्क एजेंट तैयार है। आदेश बोलिए।")
    print("=" * 50)
    print("mark-agent started (Claude routing + keyword fallback)")
    print("Say: डेटा लाओ / रणनीति बनाओ / बैकटेस्ट / सत्यापन / बंद")
    print("=" * 50)

    while True:
        # 1) Listen (speech → text)
        command = listener.listen()
        if not command:
            continue

        # 2 + 3 + 4) Claude route → execute matched fn → speak result
        result = route_command(command, market_context=_market_context())

        # Quit detection (handler already spoke goodbye)
        lowered = command.lower()
        if any(k in lowered for k in ("बंद", "quit", "exit", "goodbye", "बाय")):
            print(f"Exiting after: {result}")
            break


if __name__ == "__main__":
    main()
