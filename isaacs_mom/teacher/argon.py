"""The one door to Argon: a per-run call budget, an `argon_call` event per call, and no crash when it is unreachable.

The key is read from the environment or `.env` only when the first call is made, so importing never needs it.
"""

import os
import time
from collections.abc import Callable
from pathlib import Path

from isaacs_mom.contracts import EventType
from isaacs_mom.orchestrator.events import EventLog

MODEL = "gemini-4-argon"
DEFAULT_BUDGET = 200
ENV_PATH = Path(__file__).resolve().parents[2] / ".env"

Part = str | bytes  # text, or a JPEG from the head camera
Generate = Callable[[list[Part]], str]


def env_value(name: str, env_path: Path = ENV_PATH) -> str | None:
    if os.environ.get(name):
        return os.environ[name]
    if not env_path.exists():
        return None
    for line in env_path.read_text().splitlines():
        key, sep, value = line.partition("=")
        if sep and key.strip() == name and value.strip():
            return value.strip().strip("'\"")
    return None


def gemini_generate(model: str = MODEL) -> Generate:
    clients: list[object] = []  # one client kept alive for the run: a temporary one is closed before it sends

    def generate(parts: list[Part]) -> str:
        key = env_value("GEMINI_API_KEY")
        if not key:
            raise RuntimeError("GEMINI_API_KEY is not set in the environment or .env")
        from google import genai  # imported on first call: CI and FakeSim runs never load the SDK
        from google.genai import types

        if not clients:
            clients.append(genai.Client(api_key=key))
        client: genai.Client = clients[0]  # type: ignore[assignment]

        contents: list[types.Part | str] = [
            types.Part.from_bytes(data=p, mime_type="image/jpeg") if isinstance(p, bytes) else p for p in parts
        ]
        response = client.models.generate_content(
            model=model,
            contents=contents,  # type: ignore[arg-type]
            config=types.GenerateContentConfig(response_mime_type="application/json"),
        )
        return response.text or ""

    return generate


class Argon:
    """Counts every call against the budget (SPEC §2 rule 5) and returns None instead of raising."""

    def __init__(
        self,
        events: EventLog | None = None,
        budget: int | None = None,
        generate: Generate | None = None,
    ) -> None:
        self.events = events
        self.budget = budget if budget is not None else int(env_value("ARGON_CALL_BUDGET") or DEFAULT_BUDGET)
        self.generate = generate or gemini_generate()
        self.calls = 0

    @property
    def exhausted(self) -> bool:
        return self.calls >= self.budget

    def ask(self, purpose: str, parts: list[Part]) -> str | None:
        if self.exhausted:
            self._emit("error", purpose=purpose, error=f"Argon call budget of {self.budget} used up")
            return None
        self.calls += 1
        started = time.perf_counter()
        try:
            text, error = self.generate(parts), None
        except Exception as e:  # noqa: BLE001  # network, auth and quota errors all mean "unreachable" to the robot
            text, error = None, f"{type(e).__name__}: {e}"
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

    def _emit(self, type_: EventType, **data: object) -> None:
        if self.events:
            self.events.emit(type_, **data)
