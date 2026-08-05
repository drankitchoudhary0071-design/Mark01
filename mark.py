#!/usr/bin/env python3
"""
Mark — Personal AI Agent powered by Gemini 2.5 Flash.

Capabilities:
  - Run local Mac Terminal commands (safe subprocess)
  - Write, read, and execute Python code locally
  - Search the internet (coding & trading strategy topics)
  - Send instant WhatsApp messages (pywhatkit)
  - Speak Hindi confirmations (gTTS)

Set GEMINI_API_KEY in your environment before running.
"""

from __future__ import annotations

import os
import platform
import re
import subprocess
import sys
import tempfile
import textwrap
import traceback
from pathlib import Path
from typing import Any

from google import genai
from google.genai import types

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MODEL_ID = "gemini-2.5-flash"
WORKSPACE_DIR = Path.home() / "MarkWorkspace"
WORKSPACE_DIR.mkdir(parents=True, exist_ok=True)

BLOCKED_TERMINAL_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"\brm\s+(-[^\s]*\s+)*(-[^\s]*\s+)*[/~]", re.I),
    re.compile(r"\brm\s+-rf\s+[/~]", re.I),
    re.compile(r"\bmkfs\b", re.I),
    re.compile(r"\bdd\s+if=", re.I),
    re.compile(r":\(\)\s*\{", re.I),
    re.compile(r"\bshutdown\b", re.I),
    re.compile(r"\breboot\b", re.I),
    re.compile(r"\bsudo\s+rm\b", re.I),
    re.compile(r"\bchmod\s+-R\s+777\s+/", re.I),
    re.compile(r"\b>\s*/dev/sd", re.I),
)

SYSTEM_INSTRUCTION = textwrap.dedent(
    """
    You are Mark, a personal AI assistant for terminal work, Python coding,
    trading strategy research, and quick messaging.

    Personality:
    - Professional, concise, and action-oriented.
    - Prefer executing tools over long explanations.
    - After completing important actions, call speak_hindi_confirmation with a
      short Hindi sentence summarizing what you did.

    Tool usage guidelines:
    - Use run_terminal_command for shell/terminal tasks on the user's Mac.
    - Use write_python_file, read_python_file, and execute_python_code for
      local Python automation.
    - Use search_internet for coding help, documentation, or trading strategy
      research.
    - Use send_whatsapp_message only when the user explicitly asks to WhatsApp
      someone; phone numbers must include country code (e.g. +91...).
    - Use speak_hindi_confirmation for brief spoken confirmations in Hindi.

    Safety:
    - Never run destructive commands.
    - Confirm before sending WhatsApp messages if the recipient is ambiguous.
    """
).strip()


def _result(ok: bool, message: str, **extra: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {"ok": ok, "message": message}
    payload.update(extra)
    return payload


# ---------------------------------------------------------------------------
# Tool implementations (exposed to Gemini via automatic function calling)
# ---------------------------------------------------------------------------


def run_terminal_command(command: str, timeout_seconds: int = 120) -> dict[str, Any]:
    """Run a local Mac Terminal shell command safely.

    Args:
        command: Shell command to execute (zsh on macOS, bash elsewhere).
        timeout_seconds: Maximum seconds to wait before killing the process.

    Returns:
        A dict with ok, message, stdout, stderr, and returncode.
    """
    command = command.strip()
    if not command:
        return _result(False, "Empty command.")

    for pattern in BLOCKED_TERMINAL_PATTERNS:
        if pattern.search(command):
            return _result(False, f"Blocked for safety: {command}")

    shell = "/bin/zsh" if platform.system() == "Darwin" else "/bin/bash"
    try:
        completed = subprocess.run(
            [shell, "-lc", command],
            capture_output=True,
            text=True,
            timeout=max(5, min(timeout_seconds, 600)),
            cwd=str(WORKSPACE_DIR),
        )
        return _result(
            completed.returncode == 0,
            "Command finished.",
            command=command,
            returncode=completed.returncode,
            stdout=completed.stdout[-8000:],
            stderr=completed.stderr[-8000:],
        )
    except subprocess.TimeoutExpired:
        return _result(False, f"Command timed out after {timeout_seconds}s.", command=command)
    except Exception as exc:
        return _result(False, f"Command failed: {exc}", command=command)


def write_python_file(relative_path: str, content: str) -> dict[str, Any]:
    """Write Python source code to a file under ~/MarkWorkspace.

    Args:
        relative_path: File path relative to MarkWorkspace (e.g. scripts/bot.py).
        content: Full Python source code to write.

    Returns:
        A dict with ok, message, and absolute path.
    """
    try:
        target = (WORKSPACE_DIR / relative_path).resolve()
        if WORKSPACE_DIR not in target.parents and target != WORKSPACE_DIR:
            return _result(False, "Path must stay inside ~/MarkWorkspace.")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
        return _result(True, "File written.", path=str(target), bytes=len(content.encode()))
    except Exception as exc:
        return _result(False, f"Write failed: {exc}")


def read_python_file(relative_path: str) -> dict[str, Any]:
    """Read a Python or text file from ~/MarkWorkspace.

    Args:
        relative_path: File path relative to MarkWorkspace.

    Returns:
        A dict with ok, message, path, and content.
    """
    try:
        target = (WORKSPACE_DIR / relative_path).resolve()
        if WORKSPACE_DIR not in target.parents and target != WORKSPACE_DIR:
            return _result(False, "Path must stay inside ~/MarkWorkspace.")
        if not target.is_file():
            return _result(False, f"File not found: {relative_path}")
        content = target.read_text(encoding="utf-8")
        return _result(True, "File read.", path=str(target), content=content[:12000])
    except Exception as exc:
        return _result(False, f"Read failed: {exc}")


def execute_python_code(code: str, timeout_seconds: int = 60) -> dict[str, Any]:
    """Execute Python code locally in a subprocess and return stdout/stderr.

    Args:
        code: Python source code to run.
        timeout_seconds: Maximum runtime in seconds.

    Returns:
        A dict with ok, message, stdout, stderr, and returncode.
    """
    if not code.strip():
        return _result(False, "Empty code.")

    with tempfile.NamedTemporaryFile(
        mode="w",
        suffix=".py",
        delete=False,
        encoding="utf-8",
        dir=str(WORKSPACE_DIR),
    ) as handle:
        handle.write(code)
        script_path = handle.name

    try:
        completed = subprocess.run(
            [sys.executable, script_path],
            capture_output=True,
            text=True,
            timeout=max(5, min(timeout_seconds, 300)),
            cwd=str(WORKSPACE_DIR),
        )
        return _result(
            completed.returncode == 0,
            "Python execution finished.",
            returncode=completed.returncode,
            stdout=completed.stdout[-8000:],
            stderr=completed.stderr[-8000:],
            script_path=script_path,
        )
    except subprocess.TimeoutExpired:
        return _result(False, f"Python code timed out after {timeout_seconds}s.")
    except Exception as exc:
        return _result(False, f"Execution failed: {exc}")
    finally:
        try:
            os.unlink(script_path)
        except OSError:
            pass


def search_internet(query: str, max_results: int = 5) -> dict[str, Any]:
    """Search the internet for coding or trading strategy information.

    Args:
        query: Search query (e.g. 'Python asyncio tutorial', 'RSI trading strategy').
        max_results: Number of results to return (1-10).

    Returns:
        A dict with ok, message, query, and results list.
    """
    query = query.strip()
    if not query:
        return _result(False, "Empty search query.")

    max_results = max(1, min(max_results, 10))

    try:
        from duckduckgo_search import DDGS

        with DDGS() as ddgs:
            hits = list(ddgs.text(query, max_results=max_results))

        results = [
            {
                "title": item.get("title", ""),
                "url": item.get("href", item.get("link", "")),
                "snippet": item.get("body", item.get("snippet", "")),
            }
            for item in hits
        ]
        return _result(
            True,
            f"Found {len(results)} result(s).",
            query=query,
            results=results,
        )
    except Exception as exc:
        return _result(False, f"Search failed: {exc}", query=query)


def send_whatsapp_message(phone_number: str, message: str, wait_time: int = 15) -> dict[str, Any]:
    """Send an instant WhatsApp message using pywhatkit (opens WhatsApp Web).

    Args:
        phone_number: Recipient number with country code, e.g. +919876543210.
        message: Text message to send.
        wait_time: Seconds to wait for WhatsApp Web to load (default 15).

    Returns:
        A dict with ok and message.
    """
    phone_number = phone_number.strip()
    message = message.strip()
    if not phone_number.startswith("+"):
        return _result(False, "phone_number must include country code, e.g. +91XXXXXXXXXX")
    if not message:
        return _result(False, "Message cannot be empty.")

    try:
        import pywhatkit

        pywhatkit.sendwhatmsg_instantly(
            phone_no=phone_number,
            message=message,
            wait_time=max(10, min(wait_time, 60)),
            tab_close=True,
            close_time=3,
        )
        return _result(True, f"WhatsApp message sent to {phone_number}.", phone_number=phone_number)
    except Exception as exc:
        return _result(False, f"WhatsApp send failed: {exc}", phone_number=phone_number)


def speak_hindi_confirmation(text: str) -> dict[str, Any]:
    """Speak a short confirmation aloud in Hindi using gTTS.

    Args:
        text: Hindi text to speak (Devanagari or romanized Hindi).

    Returns:
        A dict with ok, message, and audio_path.
    """
    text = text.strip()
    if not text:
        return _result(False, "Empty confirmation text.")

    try:
        from gtts import gTTS

        audio_path = WORKSPACE_DIR / "mark_hindi_confirmation.mp3"
        tts = gTTS(text=text, lang="hi")
        tts.save(str(audio_path))

        if platform.system() == "Darwin":
            subprocess.run(["afplay", str(audio_path)], check=False)
        elif platform.system() == "Linux":
            for player in (["mpg123"], ["ffplay", "-nodisp", "-autoexit"], ["aplay"]):
                try:
                    subprocess.run([*player, str(audio_path)], check=False, timeout=120)
                    break
                except FileNotFoundError:
                    continue

        return _result(True, "Hindi confirmation spoken.", text=text, audio_path=str(audio_path))
    except Exception as exc:
        return _result(False, f"TTS failed: {exc}")


MARK_TOOLS = [
    run_terminal_command,
    write_python_file,
    read_python_file,
    execute_python_code,
    search_internet,
    send_whatsapp_message,
    speak_hindi_confirmation,
]


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------


class MarkAgent:
    """Gemini-powered agent with local tool access."""

    def __init__(self, api_key: str) -> None:
        self.client = genai.Client(api_key=api_key)
        self.config = types.GenerateContentConfig(
            tools=MARK_TOOLS,
            system_instruction=SYSTEM_INSTRUCTION,
            automatic_function_calling=types.AutomaticFunctionCallingConfig(
                maximum_remote_calls=12,
            ),
        )
        self.chat = self.client.chats.create(model=MODEL_ID, config=self.config)

    def ask(self, user_message: str) -> str:
        """Send a message to Mark and return the final text response."""
        response = self.chat.send_message(user_message)
        return (response.text or "").strip() or "(Mark completed the task with no text reply.)"


def _require_api_key() -> str:
    api_key = os.environ.get("GEMINI_API_KEY", "").strip()
    if not api_key:
        raise EnvironmentError(
            "GEMINI_API_KEY environment variable is not set.\n"
            "Export it first:  export GEMINI_API_KEY='your-key-here'"
        )
    return api_key


def _print_banner() -> None:
    print("=" * 60)
    print("  Mark — Personal AI Agent (Gemini 2.5 Flash)")
    print("=" * 60)
    print(f"  Workspace : {WORKSPACE_DIR}")
    print(f"  Platform  : {platform.system()} ({platform.machine()})")
    print("  Commands  : terminal, Python, web search, WhatsApp, Hindi TTS")
    print("  Type 'exit' or 'quit' to stop.")
    print("=" * 60)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    _print_banner()

    try:
        mark = MarkAgent(api_key=_require_api_key())
    except EnvironmentError as exc:
        print(f"\nError: {exc}", file=sys.stderr)
        sys.exit(1)
    except Exception as exc:
        print(f"\nFailed to initialize Mark: {exc}", file=sys.stderr)
        traceback.print_exc()
        sys.exit(1)

    print("\nMark is ready. Enter your terminal / trading / coding commands:\n")

    while True:
        try:
            user_input = input("You> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nMark signing off. Goodbye!")
            break

        if not user_input:
            continue

        if user_input.lower() in {"exit", "quit", "bye", "q"}:
            try:
                farewell = mark.ask("Say a brief goodbye in Hindi and English.")
                print(f"\nMark> {farewell}")
            except Exception:
                print("\nMark> Goodbye!")
            break

        try:
            reply = mark.ask(user_input)
            print(f"\nMark> {reply}\n")
        except Exception as exc:
            print(f"\nMark> Error: {exc}\n", file=sys.stderr)
            traceback.print_exc()
