import sys
from pathlib import Path

import httpx
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from shared.utilities import llm_provider


def test_default_model_env_override(monkeypatch):
    monkeypatch.setenv("LOCAL_AI_MODEL", "qwen3.5:9b")
    import importlib
    importlib.reload(llm_provider)
    assert llm_provider.DEFAULT_MODEL == "qwen3.5:9b"


def test_completion_defaults_to_end_stop():
    comp = llm_provider.Completion()
    assert comp.stop == "end"
    assert comp.text == ""


def test_context_window_scales_with_prompt_length_not_just_max_tokens():
    small_prompt = llm_provider._context_window_for(max_tokens=2000, prompt_chars=100)
    large_prompt = llm_provider._context_window_for(max_tokens=2000, prompt_chars=200_000)
    assert large_prompt > small_prompt
    assert large_prompt in llm_provider._CONTEXT_TIERS


# --- Gemini provider (DECISIONS.md D-022) -----------------------------------


def test_gemini_schema_inlines_refs_and_strips_unsupported_keywords():
    schema = {
        "$defs": {
            "Point": {
                "type": "object",
                "title": "Point",
                "additionalProperties": False,
                "properties": {"content": {"type": "string"}},
                "required": ["content"],
            }
        },
        "type": "object",
        "title": "Card",
        "additionalProperties": False,
        "properties": {"points": {"type": "array", "items": {"$ref": "#/$defs/Point"}}},
    }
    result = llm_provider._gemini_schema(schema)
    assert "$defs" not in result
    assert "title" not in result
    assert "additionalProperties" not in result
    point_schema = result["properties"]["points"]["items"]
    assert point_schema["properties"]["content"]["type"] == "string"
    assert "$ref" not in str(point_schema)


def test_gemini_schema_flat_schema_passes_through_unchanged_in_substance():
    schema = {"type": "object", "properties": {"x": {"type": "string"}}, "required": ["x"]}
    result = llm_provider._gemini_schema(schema)
    assert result["properties"]["x"]["type"] == "string"


def test_stream_json_dispatches_to_ollama_by_default(monkeypatch):
    """Confirms the ollama code path is taken (not gemini), without a real
    server: a fake AsyncClient makes the ollama branch fail fast and
    deterministically instead of hitting the network."""
    monkeypatch.setattr(llm_provider, "AI_PROVIDER", "ollama")
    called = {"gemini": False}

    async def fake_gemini(**kwargs):
        called["gemini"] = True
        yield {"type": "done", "completion": llm_provider.Completion()}

    monkeypatch.setattr(llm_provider, "_stream_json_gemini", fake_gemini)

    class _FakeAsyncClient:
        def __init__(self, *args, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return False

        def stream(self, *args, **kwargs):
            raise httpx.ConnectError("no server")

    monkeypatch.setattr(llm_provider.httpx, "AsyncClient", _FakeAsyncClient)

    import asyncio

    async def drive():
        try:
            async for _ in llm_provider.stream_json(
                system="s", user="u", schema={"type": "object"},
            ):
                pass
        except llm_provider.LLMProviderError:
            pass

    asyncio.run(drive())
    assert called["gemini"] is False


def test_stream_json_dispatches_to_gemini_when_configured(monkeypatch):
    monkeypatch.setattr(llm_provider, "AI_PROVIDER", "gemini")
    called = {"gemini": False}

    async def fake_gemini(**kwargs):
        called["gemini"] = True
        yield {"type": "done", "completion": llm_provider.Completion()}

    monkeypatch.setattr(llm_provider, "_stream_json_gemini", fake_gemini)

    import asyncio

    async def drive():
        async for _ in llm_provider.stream_json(system="s", user="u", schema={"type": "object"}):
            pass

    asyncio.run(drive())
    assert called["gemini"] is True


def test_gemini_provider_raises_clear_error_when_api_key_missing(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)

    import asyncio

    async def drive():
        events = []
        async for event in llm_provider._stream_json_gemini(
            system="s", user="u", schema={"type": "object"}, model=None,
            max_tokens=100, temperature=0.3,
        ):
            events.append(event)
        return events

    with pytest.raises(llm_provider.LLMProviderError, match="GEMINI_API_KEY"):
        asyncio.run(drive())


def test_gemini_sends_api_key_as_header_never_as_url_query_param(monkeypatch):
    """Regression test for a real secret exposure caught in manual testing:
    a query-param API key ends up in the request URL, which appears in
    httpx's own exception messages (and any logging/traceback of the
    request) — so the key must travel as a header instead."""
    monkeypatch.setenv("GEMINI_API_KEY", "secret-value-must-not-leak")
    captured = {}

    class _FakeResponse:
        status_code = 200

        def json(self):
            return {"candidates": [{"content": {"parts": [{"text": "{}"}]}, "finishReason": "STOP"}]}

        def raise_for_status(self):
            pass

    class _FakeAsyncClient:
        def __init__(self, *a, **k):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return False

        async def post(self, url, **kwargs):
            captured["url"] = url
            captured["params"] = kwargs.get("params")
            captured["headers"] = kwargs.get("headers")
            return _FakeResponse()

    monkeypatch.setattr(llm_provider.httpx, "AsyncClient", _FakeAsyncClient)

    import asyncio

    async def drive():
        async for _ in llm_provider._stream_json_gemini(
            system="s", user="u", schema={"type": "object"}, model=None,
            max_tokens=100, temperature=0.3,
        ):
            pass

    asyncio.run(drive())
    assert "secret-value-must-not-leak" not in captured["url"]
    assert not captured["params"]
    assert captured["headers"]["x-goog-api-key"] == "secret-value-must-not-leak"


def test_gemini_http_error_message_never_includes_the_request_url_or_key(monkeypatch):
    """The original bug: catching httpx.HTTPStatusError without `from None`
    let Python's automatic exception chaining print the ORIGINAL exception's
    message (which httpx builds from the full request URL) even though the
    re-raised LLMProviderError's own message was already clean."""
    monkeypatch.setenv("GEMINI_API_KEY", "secret-value-must-not-leak")

    class _FakeResponse:
        status_code = 503

        def raise_for_status(self):
            request = httpx.Request("POST", "https://example.invalid/?key=secret-value-must-not-leak")
            raise httpx.HTTPStatusError("error", request=request, response=self)

    class _FakeAsyncClient:
        def __init__(self, *a, **k):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return False

        async def post(self, url, **kwargs):
            return _FakeResponse()

    monkeypatch.setattr(llm_provider.httpx, "AsyncClient", _FakeAsyncClient)

    import asyncio
    import traceback

    async def drive():
        async for _ in llm_provider._stream_json_gemini(
            system="s", user="u", schema={"type": "object"}, model=None,
            max_tokens=100, temperature=0.3,
        ):
            pass

    try:
        asyncio.run(drive())
        raise AssertionError("expected LLMProviderError")
    except llm_provider.LLMProviderError as error:
        # Check the error's own message AND the full printed traceback (the
        # real leak vector — Python's default chaining prints the original
        # exception's text too, unless the code uses `raise ... from None`).
        full_output = "".join(traceback.format_exception(type(error), error, error.__traceback__))
        assert "secret-value-must-not-leak" not in str(error)
        assert "secret-value-must-not-leak" not in full_output
        assert error.__cause__ is None
        # `raise ... from None` sets __suppress_context__, not __context__ to
        # None (Python still keeps __context__ internally) — the printed
        # traceback (full_output, checked above) is what actually matters,
        # and is what a real terminal/log would show.
        assert error.__suppress_context__ is True


def test_health_check_for_gemini_checks_api_key_configured(monkeypatch):
    monkeypatch.setattr(llm_provider, "AI_PROVIDER", "gemini")
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)

    import asyncio
    assert asyncio.run(llm_provider.health_check()) is False

    monkeypatch.setenv("GEMINI_API_KEY", "fake-key")
    assert asyncio.run(llm_provider.health_check()) is True


def test_gemini_thinking_is_off_only_when_asked(monkeypatch):
    """Gemini 2.5 counts thinking against maxOutputTokens; think=False must
    reach it as a zero thinking budget, or a long answer gets cut off."""
    import asyncio
    monkeypatch.setenv("GEMINI_API_KEY", "k")
    sent = []

    class _FakeResponse:
        status_code = 200

        def json(self):
            return {"candidates": [{"content": {"parts": [{"text": "{}"}]}, "finishReason": "STOP"}]}

        def raise_for_status(self):
            pass

    class _FakeAsyncClient:
        def __init__(self, *a, **k):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return False

        async def post(self, url, **kwargs):
            sent.append(kwargs["json"]["generationConfig"])
            return _FakeResponse()

    monkeypatch.setattr(llm_provider.httpx, "AsyncClient", _FakeAsyncClient)
    monkeypatch.setattr(llm_provider, "AI_PROVIDER", "gemini")
    for think in (False, True):
        asyncio.run(llm_provider.complete_json(system="s", user="u", schema={"type": "object"},
                                               think=think))
    assert sent[0]["thinkingConfig"] == {"thinkingBudget": 0}
    assert "thinkingConfig" not in sent[1]
