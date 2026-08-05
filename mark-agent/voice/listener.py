"""
voice/listener.py
Continuously listens via microphone and converts speech to text.
Supports Hindi (hi-IN) via SpeechRecognition / Google Speech API.
"""

import speech_recognition as sr


class Listener:
    """Wraps SpeechRecognition for continuous mic listening."""

    def __init__(self, language: str = "hi-IN"):
        # Default language is Hindi; change to "en-IN" or "en-US" if needed
        self.language = language
        self.recognizer = sr.Recognizer()
        # Slight pause adjustment helps in noisy rooms
        self.recognizer.pause_threshold = 0.8
        self.recognizer.energy_threshold = 300

    def listen(self, timeout: int = 5, phrase_time_limit: int = 10) -> str:
        """
        Capture audio from the default microphone and return recognized text.
        Returns empty string on failure so the main loop can keep running.
        """
        with sr.Microphone() as source:
            # Quick ambient noise calibration each turn
            self.recognizer.adjust_for_ambient_noise(source, duration=0.5)
            print("🎤 Listening...")
            try:
                audio = self.recognizer.listen(
                    source,
                    timeout=timeout,
                    phrase_time_limit=phrase_time_limit,
                )
            except sr.WaitTimeoutError:
                print("⏱ No speech detected (timeout).")
                return ""

        try:
            text = self.recognizer.recognize_google(audio, language=self.language)
            print(f"🗣 Heard: {text}")
            return text.strip()
        except sr.UnknownValueError:
            print("❓ Could not understand audio.")
            return ""
        except sr.RequestError as exc:
            print(f"⚠ Speech API error: {exc}")
            return ""


def listen_once(language: str = "hi-IN") -> str:
    """Convenience helper used by main.py."""
    return Listener(language=language).listen()
