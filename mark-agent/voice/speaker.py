"""
voice/speaker.py
Text-to-speech using pyttsx3 with Hindi output support.
"""

from __future__ import annotations

from typing import Optional

import pyttsx3


class Speaker:
    """Simple pyttsx3 wrapper that prefers a Hindi-capable voice when available."""

    def __init__(self, rate: int = 160, volume: float = 1.0):
        self.engine = pyttsx3.init()
        self.engine.setProperty("rate", rate)
        self.engine.setProperty("volume", volume)
        self._prefer_hindi_voice()

    def _prefer_hindi_voice(self) -> None:
        """
        Try to pick a Hindi / Indian voice from the OS voice list.
        On Mac this often matches names containing 'hi', 'hindi', or 'india'.
        Falls back to the system default if none found.
        """
        voices = self.engine.getProperty("voices")
        for voice in voices:
            name = (voice.name or "").lower()
            langs = getattr(voice, "languages", None) or []
            lang = " ".join(
                x.decode("utf-8", errors="ignore") if isinstance(x, bytes) else str(x)
                for x in langs
            ).lower()
            vid = (voice.id or "").lower()
            blob = f"{name} {lang} {vid}"
            if any(key in blob for key in ("hindi", "hi-in", "hi_in", "india", "lekha")):
                self.engine.setProperty("voice", voice.id)
                print(f"🔊 Using voice: {voice.name}")
                return
        print("🔊 No Hindi voice found; using system default.")

    def speak(self, text: str) -> None:
        """Speak the given text aloud (blocks until finished)."""
        if not text:
            return
        print(f"💬 Speaking: {text}")
        self.engine.say(text)
        self.engine.runAndWait()


# Module-level singleton for simple imports
_speaker: Optional[Speaker] = None


def speak(text: str) -> None:
    """Speak text in Hindi (or best available voice)."""
    global _speaker
    if _speaker is None:
        _speaker = Speaker()
    _speaker.speak(text)
