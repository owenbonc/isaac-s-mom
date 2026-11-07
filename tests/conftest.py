import pytest


@pytest.fixture(autouse=True)
def no_live_gemini(monkeypatch: pytest.MonkeyPatch) -> None:
    """Every test runs offline: building a real Gemini client fails, so a stray live call can't spend the budget."""
    from google import genai

    def refuse(*_: object, **__: object) -> None:
        raise RuntimeError("a test tried to build a live Gemini client; inject a fake generate instead")

    monkeypatch.setattr(genai, "Client", refuse)
