"""The one door to Argon: a per-run call budget, an `argon_call` event per call, and a hold when it is unreachable.

The key and the SDK load only on the first call, so importing never needs either. A missing key is a setup error and
raises; only network and API errors count as "unreachable".
"""

import os
import time
from collections.abc import Callable
from pathlib import Path

from isaacs_mom.contracts import EventType
from isaacs_mom.orchestrator.events import EventLog

MODEL = "gemini-4-argon"
DEFAULT_BUDGET = 800  # Owen: a hug-and-place demo is ~7-9 calls, ~190 a round plus the B1 baseline
TIMEOUT_S = 60
ENV_PATH = Path(__file__).resolve().parents[2] / ".env"

Part = str | bytes  # text, or a JPEG from the head camera
Generate = Callable[[list[Part]], str]


class ArgonUnavailable(Exception):
    """The request failed on the network or at the API: the robot holds and re-observes."""


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
        if value[:1] in ("'", '"') and value[0] in value[1:]:
            value = value[1 : value.index(value[0], 1)]
        else:
            value = value.split(" #", 1)[0].strip()
        return value or None
    return None


def gemini_generate(model: str = MODEL) -> Generate:
    clients: list[object] = []  # one client kept alive for the run: a temporary one is closed before it sends

    def generate(parts: list[Part]) -> str:
        key = env_value("GEMINI_API_KEY")
        if not key:
            raise MissingKey("GEMINI_API_KEY is not set in the environment or .env")
        import httpx2  # imported on first call: CI and FakeSim runs never load the SDK or its transport
        from google import genai
        from google.genai import errors, types

        if not clients:
            clients.append(genai.Client(api_key=key, http_options=types.HttpOptions(timeout=TIMEOUT_S * 1000)))
        client: genai.Client = clients[0]  # type: ignore[assignment]  # the list only ever holds this client
        try:
            response = client.models.generate_content(
                model=model,
                contents=[
                    types.Part.from_bytes(data=p, mime_type="image/jpeg") if isinstance(p, bytes) else p for p in parts
                ],
                config=types.GenerateContentConfig(response_mime_type="application/json"),
            )
        except (errors.APIError, httpx2.HTTPError) as e:
            raise ArgonUnavailable(f"{type(e).__name__}: {e}") from e
        return response.text or ""

    return generate


class Argon:
    """Counts every call against the budget (SPEC §2 rule 5) and logs each one, failed or not."""

    def __init__(self, events: EventLog, budget: int | None = None, generate: Generate | None = None) -> None:
        self.events = events
        self.budget = budget if budget is not None else int(env_value("ARGON_CALL_BUDGET") or DEFAULT_BUDGET)
        self.generate = generate or gemini_generate()
        self.calls = 0

    def ask(self, purpose: str, parts: list[Part]) -> str | None:
        """Argon's answer, or None when it is over budget, unreachable or silent."""
        if self.calls >= self.budget:
            self.error(purpose, f"Argon call budget of {self.budget} used up")
            return None
        self.calls += 1
        started = time.perf_counter()
        try:
            text, error = self.generate(parts), None
        except ArgonUnavailable as e:
            text, error = None, str(e)
        if text == "":
            text, error = None, "empty answer"
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

    def error(self, purpose: str, error: str, **data: object) -> None:
        self._emit("error", source="argon", purpose=purpose, error=error, **data)

    def _emit(self, type_: EventType, **data: object) -> None:
        self.events.emit(type_, **data)
