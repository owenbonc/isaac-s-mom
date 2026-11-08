"""The one door to Argon: a per-run call budget and an `argon_call` event for every call, failed or not.

Server, timeout, rate-limit and transport errors raise `ArgonUnavailable`, which the teacher holds on; any other 4xx
(bad key, wrong model, malformed request) raises as it is and stops the run. `check` makes one real call at startup,
so a bad key or a wrong model stops the run before any episode.
"""

import json
import math
import os
import re
import time
from collections.abc import Callable
from pathlib import Path
from typing import Literal

from isaacs_mom.contracts import EventType
from isaacs_mom.orchestrator.events import EventLog

MODEL = "gemini-4-argon"
DEFAULT_BUDGET = 800  # a hug-and-place demo is ~7-9 calls: ~190 a round plus the B1 baseline
TIMEOUT_S = 60
RETRYABLE_CODES = (408, 429)  # request timeout and rate limit: transient, unlike the other 4xx
RETRY_WAITS_S = (10.0, 50.0)  # between the 3 attempts of a step: a minute-long rate limit is ridden out
MAX_WAIT_S = 120.0
ENV_PATH = Path(__file__).resolve().parents[2] / ".env"

Part = str | bytes  # text, or a JPEG from the head camera
Purpose = Literal["act", "diagnose", "startup", "task"]
Generate = Callable[[Purpose, list[Part]], str]


class ArgonUnavailable(Exception):
    """The request timed out, failed on the network or at the server, or was rate limited: the robot holds."""

    def __init__(self, message: str, retry_after_s: float | None = None) -> None:
        super().__init__(message)
        self.retry_after_s = retry_after_s


class MissingKey(RuntimeError):
    """GEMINI_API_KEY is in neither the environment nor .env."""


def env_value(name: str, env_path: Path | None = None) -> str | None:
    """The environment first, then the last definition in .env (quotes, `export`, a BOM and `# comments` handled)."""
    if os.environ.get(name):
        return os.environ[name]
    env_path = env_path or ENV_PATH
    if not env_path.exists():
        return None
    found: str | None = None
    for line in env_path.read_text(encoding="utf-8-sig").splitlines():
        key, sep, value = line.strip().removeprefix("export ").partition("=")
        if not sep or key.strip() != name:
            continue
        value = value.strip()
        quote = value[:1]
        if quote in ("'", '"') and quote in value[1:]:
            value = value[1 : value.index(quote, 1)]
        else:
            value = re.split(r"\s#", value.lstrip("'\""), maxsplit=1)[0].strip()
        found = value or None
    return found


def hide_key(text: str) -> str:
    """The text with the Argon key shown as `[key]`: an SDK error can quote it, and events reach the dashboard."""
    key = env_value("GEMINI_API_KEY")
    return text.replace(key, "[key]") if key else text


def _seconds(text: object) -> float | None:
    """A usable wait: finite and not negative (time.sleep raises on a negative one, and NaN slips past min)."""
    try:
        value = float(text)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) and value >= 0 else None


def retry_after_s(error: Exception) -> float | None:
    """The server's own wait: a Retry-After header, or a google.rpc.RetryInfo `retryDelay` like "37s"."""
    headers = getattr(getattr(error, "response", None), "headers", None)
    header = headers.get("retry-after") if hasattr(headers, "get") else None  # type: ignore[union-attr]
    if header and (wait := _seconds(header)) is not None:
        return wait
    body = getattr(error, "details", None)
    details = body.get("error", {}).get("details", []) if isinstance(body, dict) else []
    for detail in details if isinstance(details, list) else []:
        delay = detail.get("retryDelay") if isinstance(detail, dict) else None
        if isinstance(delay, str) and delay.endswith("s"):
            return _seconds(delay[:-1])
    return None


def gemini_generate(model: str = MODEL) -> Generate:
    key = env_value("GEMINI_API_KEY")
    if not key:
        raise MissingKey("GEMINI_API_KEY is not set in the environment or .env (or run with --teacher stub)")
    import httpx2  # loaded only for a live run: CI and stub runs never import the SDK or its transport
    from google import genai
    from google.genai import errors, types

    client = genai.Client(api_key=key, http_options=types.HttpOptions(timeout=TIMEOUT_S * 1000))
    jpeg = "image/jpeg"

    def generate(purpose: Purpose, parts: list[Part]) -> str:
        try:
            response = client.models.generate_content(
                model=model,
                contents=[types.Part.from_bytes(data=p, mime_type=jpeg) if isinstance(p, bytes) else p for p in parts],
                config=types.GenerateContentConfig(response_mime_type="application/json"),
            )
        except errors.ClientError as e:
            if e.code not in RETRYABLE_CODES:
                raise
            raise ArgonUnavailable(f"ClientError: {e}", retry_after_s(e)) from e
        except httpx2.TooManyRedirects:
            raise  # a redirect loop is a setup fault, not a passing outage
        except (errors.ServerError, errors.UnknownApiResponseError, httpx2.RequestError) as e:
            raise ArgonUnavailable(f"{type(e).__name__}: {e}", retry_after_s(e)) from e
        return response.text or ""

    return generate


def stub_generate(purpose: Purpose, parts: list[Part]) -> str:
    """An offline stand-in: diagnoses nothing and ends every demo at once, so the loop runs without a key. A round's
    diagnosis is one mode naming every failed trial, so the round teaches as the per-episode lessons would."""
    if purpose != "diagnose":
        return '{"done": true, "reason": "stub teacher: offline stand-in"}'
    nothing = {"hypothesis": "stub teacher: not diagnosed", "evidence": "", "confidence": 0.0, "lesson": ""}
    failed = re.findall(r"trial (ep-\S+) \(failed\)", str(parts[0]))
    return json.dumps({"modes": [{**nothing, "explains": failed}]} if failed else nothing)


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
        self.budget = budget if budget is not None else int(env_value("ARGON_CALL_BUDGET") or DEFAULT_BUDGET)
        self.generate = generate or gemini_generate()
        self.sleep = sleep
        self.calls = 0
        self.failed = 0
        self.server_wait_s: float | None = None
        self.last_error: str | None = None

    @property
    def exhausted(self) -> bool:
        return self.calls >= self.budget

    def check(self) -> None:
        """One real call at startup, counted and logged: a bad key or a wrong model raises before any episode."""
        if self.ask("startup", ['Reply with exactly this JSON object: {"done": true}']) is None:
            raise ArgonUnavailable(f"Argon did not answer the startup check: {self.last_error}")

    def ask(self, purpose: Purpose, parts: list[Part]) -> str | None:
        """One attempt: Argon's answer, or None when it is over budget, unreachable or silent."""
        if self.exhausted:
            self.error(purpose, f"Argon call budget of {self.budget} used up")
            return None
        self.calls += 1
        started = time.perf_counter()
        self.server_wait_s = None
        try:
            text, error = self.generate(purpose, parts), None
        except ArgonUnavailable as e:
            text, error, self.server_wait_s = None, str(e), e.retry_after_s
        except Exception as e:
            self._log_call(purpose, started, f"{type(e).__name__}: {e}")  # a fatal 4xx is still a call (rule 5)
            raise
        if text == "":
            text, error = None, "empty answer"
        self._log_call(purpose, started, error)
        return text

    def _log_call(self, purpose: Purpose, started: float, error: str | None) -> None:
        error = None if error is None else hide_key(error)
        self.failed += error is not None
        self.last_error = error
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

    def wait(self, attempt: int) -> None:
        """Before the next attempt: the server's Retry-After when the last call gave one, else the schedule."""
        default = RETRY_WAITS_S[min(attempt, len(RETRY_WAITS_S) - 1)]
        wait = self.server_wait_s if self.server_wait_s is not None else default
        self.sleep(min(max(0.0, wait), MAX_WAIT_S))

    def summary(self) -> str:
        """For the end of a run (SPEC §3.3: calls used out of the budget, unavailable answers, budget run out)."""
        line = f"argon: {self.calls}/{self.budget} calls, {self.failed} unavailable or empty"
        return line + (", budget used up" if self.exhausted else "")

    def error(self, purpose: Purpose, error: str, **data: object) -> None:
        self._emit("error", source="argon", purpose=purpose, error=error, **data)

    def _emit(self, type_: EventType, **data: object) -> None:
        self.events.emit(type_, **data)
