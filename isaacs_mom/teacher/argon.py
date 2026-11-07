"""The one door to Argon: a per-run call budget, an `argon_call` event per call, and a hold when it is unreachable.

Building the Gemini generator checks the key and the model with one metadata request, so a bad key or a wrong model
stops the run before any episode. Only server, rate-limit and transport errors count as "unreachable", and those are
retried after a wait (the server's Retry-After when it gives one); any other 4xx raises.
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
RETRY_WAITS_S = (5.0, 15.0, 45.0)  # covers a minute-long rate limit before a step is given up
MAX_WAIT_S = 120.0
ENV_PATH = Path(__file__).resolve().parents[2] / ".env"

Part = str | bytes  # text, or a JPEG from the head camera
Purpose = str  # "act" or "diagnose"
Generate = Callable[[Purpose, list[Part]], str]


class ArgonUnavailable(Exception):
    """The request failed on the network, at the server, or was rate limited: the robot holds."""

    def __init__(self, message: str, retry_after_s: float | None = None) -> None:
        super().__init__(message)
        self.retry_after_s = retry_after_s


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


def retry_after_s(error: Exception) -> float | None:
    """The server's own wait: a Retry-After header, or a google.rpc.RetryInfo `retryDelay` like "37s"."""
    headers = getattr(getattr(error, "response", None), "headers", None)
    header = headers.get("retry-after") if hasattr(headers, "get") else None  # type: ignore[union-attr]
    try:
        if header:
            return float(header)
    except (TypeError, ValueError):
        pass
    body = getattr(error, "details", None)
    details = body.get("error", {}).get("details", []) if isinstance(body, dict) else []
    for detail in details if isinstance(details, list) else []:
        delay = detail.get("retryDelay") if isinstance(detail, dict) else None
        if isinstance(delay, str) and delay.endswith("s"):
            try:
                return float(delay[:-1])
            except ValueError:
                return None
    return None


def gemini_generate(model: str = MODEL) -> Generate:
    key = env_value("GEMINI_API_KEY")
    if not key:
        raise MissingKey("GEMINI_API_KEY is not set in the environment or .env (or run with --teacher stub)")
    import httpx2  # loaded only for a live run: CI and stub runs never import the SDK or its transport
    from google import genai
    from google.genai import errors, types

    client = genai.Client(api_key=key, http_options=types.HttpOptions(timeout=TIMEOUT_S * 1000))
    unreachable = (errors.ServerError, errors.UnknownApiResponseError, httpx2.TransportError)
    try:
        client.models.get(model=model)  # a bad key (401/403) or a wrong model (404) raises ClientError here
    except unreachable as e:
        raise ArgonUnavailable(f"Argon not reachable at startup: {type(e).__name__}: {e}") from e

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
            raise ArgonUnavailable(f"ClientError: {e}", retry_after_s(e)) from e
        except unreachable as e:
            raise ArgonUnavailable(f"{type(e).__name__}: {e}", retry_after_s(e)) from e
        return response.text or ""

    return generate


def stub_generate(purpose: Purpose, parts: list[Part]) -> str:
    """An offline stand-in: diagnoses nothing and ends every demo at once, so the loop runs without a key."""
    if purpose == "diagnose":
        return '{"hypothesis": "stub teacher: not diagnosed", "evidence": "", "confidence": 0.0, "lesson": ""}'
    return '{"done": true, "reason": "stub teacher: offline stand-in"}'


class Argon:
    """Counts every call against the budget (SPEC §2 rule 5) and logs each one, failed or not."""

    def __init__(
        self,
        events: EventLog,
        budget: int | None = None,
        generate: Generate | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.events = events
        self.sleep = sleep
        self.budget = budget if budget is not None else int(env_value("ARGON_CALL_BUDGET") or DEFAULT_BUDGET)
        self.generate = generate or gemini_generate()
        self.calls = 0
        self.failed = 0

    @property
    def exhausted(self) -> bool:
        return self.calls >= self.budget

    def ask(self, purpose: Purpose, parts: list[Part]) -> str | None:
        """Argon's answer, or None when it is over budget, or still unreachable or silent after every retry."""
        for retry, wait in enumerate((*RETRY_WAITS_S, None)):
            if self.exhausted:
                self.error(purpose, f"Argon call budget of {self.budget} used up")
                return None
            text, server_wait = self._call(purpose, parts, retry)
            if text is not None:
                return text
            if wait is not None:
                self.sleep(min(server_wait or wait, MAX_WAIT_S))
        return None

    def _call(self, purpose: Purpose, parts: list[Part], retry: int) -> tuple[str | None, float | None]:
        self.calls += 1
        started = time.perf_counter()
        server_wait = None
        try:
            text, error = self.generate(purpose, parts), None
        except ArgonUnavailable as e:
            text, error, server_wait = None, str(e), e.retry_after_s
        if text == "":
            text, error = None, "empty answer"
        self.failed += error is not None
        self._emit(
            "argon_call",
            purpose=purpose,
            model=MODEL,
            call=self.calls,
            budget=self.budget,
            retry=retry,
            ok=error is None,
            error=error,
            latency_ms=round((time.perf_counter() - started) * 1000, 1),
        )
        return text, server_wait

    def summary(self) -> str:
        """For the end of a run (SPEC §3.3: the run reports when Argon was unavailable or over budget)."""
        line = f"argon: {self.calls}/{self.budget} calls, {self.failed} unavailable or empty"
        return line + (", budget used up" if self.exhausted else "")

    def error(self, purpose: Purpose, error: str, **data: object) -> None:
        self._emit("error", source="argon", purpose=purpose, error=error, **data)

    def _emit(self, type_: EventType, **data: object) -> None:
        self.events.emit(type_, **data)
