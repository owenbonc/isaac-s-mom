import pytest


@pytest.fixture(autouse=True)
def no_live_gemini(monkeypatch: pytest.MonkeyPatch) -> None:
    """Every test runs offline: building a real Gemini client or sending a request fails, so a stray live call can't
    spend the budget, however the client was made."""
    from google import genai
    from google.genai import models

    def refuse(*_: object, **__: object) -> None:
        raise RuntimeError("a test reached a live Gemini client or request; inject a fake generate instead")

    monkeypatch.setattr(genai, "Client", refuse)
    for method in ("generate_content", "generate_content_stream", "get"):
        monkeypatch.setattr(models.Models, method, refuse)
