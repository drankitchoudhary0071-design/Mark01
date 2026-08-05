"""
main.py — entry point for mark-agent

Loop: listen → route → execute → speak
Voice commands are handled by brain/router.py (keyword matching for now).
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure project root is on sys.path when run as `python main.py`
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from brain.router import route
from voice.listener import Listener
from voice.speaker import speak


def main() -> None:
    """Run the listen → route → execute → speak loop until quit."""
    listener = Listener(language="hi-IN")

    speak("मार्क एजेंट तैयार है। आदेश बोलिए।")
    print("=" * 50)
    print("mark-agent started")
    print("Say: डेटा लाओ / रणनीति बनाओ / बैकटेस्ट / सत्यापन / बंद")
    print("=" * 50)

    while True:
        # 1) Listen (speech → text)
        command = listener.listen()
        if not command:
            continue

        # 2 + 3) Route and execute (handler also speaks the result)
        result = route(command)

        # 4) Quit keywords return from action_quit
        lowered = command.lower()
        if any(k in lowered for k in ("बंद", "quit", "exit", "goodbye", "बाय")):
            print(f"Exiting after: {result}")
            break


if __name__ == "__main__":
    main()
