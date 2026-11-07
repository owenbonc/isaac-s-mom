"""The one door to Argon: a per-run call budget, an `argon_call` event per call, and a hold when it is unreachable.

Building the Gemini generator checks the key, so a run without one stops before any episode. Only server, rate-limit
and transport errors count as "unreachable"; any other 4xx (bad key, wrong model, malformed request) raises.
"""

import os
import time
from collections.abc import Callable
from pathlib import Path

from isaacs_mom.contracts import EventType
from isaacs_mom.orchestrator.events import EventLog

MODEL = "gemini-4-argon"
DEFAULT_BUDGET = 800  # a hug-and-place demo is ~7-9 calls: ~190 a round plus the B1 baseline
TIMEOUT_S = 60
RATE_LIMITED = 429
ENV_PATH = Path(__file__).resolve().parents[2] / ".env"

Part = str | bytes  # text, or a JPEG from the head camera
Purpose = str  # "act" or "diagnose"
Generate = Callable[[Purpose, list[Part]], str]


class ArgonUnavailable(Exception):
    """The request failed on the network, at the server, or was rate limited: the robot holds."""


class MissingKey(RuntimeError):
    """GEMINI_API_KEY is in neither the environment nor .env."""


def env_value(name: str, env_path: Path | None = None) -> str | None:
    if os.environ.get(name):
        return os.environ[name]
    env_path = env_path or ENV_PATH
    if not env_path.exists():
        return None
    for line in env_path.read_text().splitlines():
        key, sep, value = line.strip().removeprefix("export ").partition("=")
        if not sep or key.strip() != name:
            continue
        value = value.strip()
        quote = value[:1]
        if quote in ("'", '"') and quote in value[1:]:
            value = value[1 : value.index(quote, 1)]
        else:
            value = value.lstrip("'\"").split(" #", 1)[0].strip()
        return value or None
    return None


def gemini_generate(model: str = MODEL) -> Generate:
    key = env_value("GEMINI_API_KEY")
    if not key:
        raise MissingKey("GEMINI_API_KEY is not set in the environment or .env (or run with --teacher stub)")
    import httpx2  # loaded only for a live run: CI and stub runs never import the SDK or its transport
    from google import genai
    from google.genai import errors, types

    client = genai.Client(api_key=key, http_options=types.HttpOptions(timeout=TIMEOUT_S * 1000))

    def generate(purpose: Purpose, parts: list[Part]) -> str:
        try:
            response = client.models.generate_content(
                model=model,
                contents=[
                    types.Part.from_bytes(data=p, mime_type="image/jpeg") if isinstance(p, bytes) else p for p in parts
                ],
                config=types.GenerateContentConfig(response_mime_type="application/json"),
            )
        except errors.ClientError as e:
            if e.code != RATE_LIMITED:
                raise
            raise ArgonUnavailable(f"ClientError: {e}") from e
        except (errors.ServerError, errors.UnknownApiResponseError, httpx2.TransportError) as e:
            raise ArgonUnavailable(f"{type(e).__name__}: {e}") from e
        return response.text or ""

    return generate


def stub_generate(purpose: Purpose, parts: list[Part]) -> str:
    """An offline stand-in: diagnoses nothing and ends every demo at once, so the loop runs without a key."""
    if purpose == "diagnose":
        return '{"hypothesis": "stub teacher: not diagnosed", "evidence": "", "confidence": 0.0, "lesson": ""}'
    return '{"done": true, "reason": "stub teacher: offline stand-in"}'


class Argon:
    """Counts every call against the budget (SPEC §2 rule 5) and logs each one, failed or not."""

    def __init__(self, events: EventLog, budget: int | None = None, generate: Generate | None = None) -> None:
        self.events = events
        self.budget = budget if budget is not None else int(env_value("ARGON_CALL_BUDGET") or DEFAULT_BUDGET)
        self.generate = generate or gemini_generate()
        self.calls = 0
        self.failed = 0

    @property
    def exhausted(self) -> bool:
        return self.calls >= self.budget

    def ask(self, purpose: Purpose, parts: list[Part]) -> str | None:
        """Argon's answer, or None when it is over budget, unreachable or silent."""
        if self.exhausted:
            self.error(purpose, f"Argon call budget of {self.budget} used up")
            return None
        self.calls += 1
        started = time.perf_counter()
        try:
            text, error = self.generate(purpose, parts), None
        except ArgonUnavailable as e:
            text, error = None, str(e)
        if text == "":
            text, error = None, "empty answer"
        self.failed += error is not None
        self._emit(
            "argon_call",
            purpose=purpose,
            model=MODEL,
            call=self.calls,
            budget=self.budget,
            ok=error is None,
            error=error,
            latency_ms=round((time.perf_counter() - started) * 1000, 1),
        )
        return text

    def summary(self) -> str:
        """For the end of a run (SPEC §3.3: the run reports when Argon was unavailable or over budget)."""
        line = f"argon: {self.calls}/{self.budget} calls, {self.failed} unavailable or empty"
        return line + (", budget used up" if self.exhausted else "")

    def error(self, purpose: Purpose, error: str, **data: object) -> None:
        self._emit("error", source="argon", purpose=purpose, error=error, **data)

    def _emit(self, type_: EventType, **data: object) -> None:
        self.events.emit(type_, **data)
