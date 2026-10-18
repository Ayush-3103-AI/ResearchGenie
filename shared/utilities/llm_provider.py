"""LLM provider layer shared by every pipeline service.

Every service asks for JSON-schema-constrained completions through this module
instead of talking to a model SDK directly, so the underlying model can be
swapped by changing configuration, not service code (see DECISIONS.md D-003).

The default and recommended provider is a local Ollama model (LOCAL_AI_MODEL,
default "qwen3.5:9b") reached over Ollama's native /api/chat endpoint. An
optional cloud provider (Gemini) was added in DECISIONS.md D-022, at explicit
user request, as an opt-in alternative for users without local hardware
capable of running the model well — it is never the default, and Ollama
remains the recommended, privacy-preserving choice presented first in setup.

Provider selection: AI_PROVIDER env var, "ollama" (default) or "gemini".
Gemini requires GEMINI_API_KEY (never logged; see researchgenie/config.py for
how the CLI stores it locally).
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import AsyncIterator

import httpx

OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434")
DEFAULT_MODEL = os.getenv("LOCAL_AI_MODEL", "qwen3.5:9b")

AI_PROVIDER = os.getenv("AI_PROVIDER", "ollama")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
GEMINI_API_BASE = "https://generativelanguage.googleapis.com/v1beta"


class LLMProviderError(Exception):
    """A user-facing error from the model layer."""


_CONTEXT_TIERS = (4096, 8192, 16384, 24576, 32768)


def _context_window_for(max_tokens: int, prompt_chars: int = 0) -> int:
    """Smallest context tier that comfortably fits max_tokens of output plus the
    actual prompt. Tiered rather than exact so repeated calls at similar sizes
    reuse the same loaded context instead of forcing a model reload each time.

    prompt_chars should be len(system) + len(user); ~4 chars/token is a safe
    rough estimate. A caller with a large input (e.g. multi-document context)
    must pass this — sizing from max_tokens alone underestimates badly and
    causes slow context-window shifting during prefill (see DECISIONS.md D-009)."""
    needed = max_tokens + max(prompt_chars // 4, max_tokens)
    for tier in _CONTEXT_TIERS:
        if tier >= needed:
            return tier
    return _CONTEXT_TIERS[-1]


@dataclass
class Completion:
    text: str = ""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    stop: str = "end"  # "end" | "length" | "refusal"


_GEMINI_ALLOWED_SCHEMA_KEYS = {
    "type", "format", "description", "nullable", "enum",
    "maxItems", "minItems", "properties", "required", "items",
}


def _gemini_schema(schema: dict) -> dict:
    """Adapt a standard JSON Schema (as every service already builds, either
    by hand or via response_model.model_json_schema()) for Gemini's
    structured-output support, which — unlike Ollama — does not understand
    $ref/$defs and only accepts a narrow subset of schema keywords.

    Untested against the live Gemini API in this environment (no API key
    available here) — validated only at the unit level (schema shape in,
    schema shape out) and via mocked HTTP requests. See DECISIONS.md D-022.
    """
    defs = schema.get("$defs", {})

    def resolve(node: dict) -> dict:
        """Resolve one JSON-schema OBJECT node (its keys are schema
        keywords: type/properties/items/etc — never arbitrary field names).
        "properties" and "items" hold nested schema nodes with different
        shapes (a name->schema map, and a single schema, respectively) and
        must be recursed into specially rather than treated as more
        keyword/value pairs."""
        if "$ref" in node:
            ref_name = node["$ref"].rsplit("/", 1)[-1]
            return resolve(defs.get(ref_name, {}))
        result: dict = {}
        for key, value in node.items():
            if key not in _GEMINI_ALLOWED_SCHEMA_KEYS:
                continue  # e.g. additionalProperties, title — unsupported by Gemini
            if key == "properties" and isinstance(value, dict):
                result[key] = {name: resolve(sub) for name, sub in value.items()}
            elif key == "items":
                result[key] = resolve(value) if isinstance(value, dict) else value
            else:
                result[key] = value
        return result

    return resolve(schema)


async def _stream_json_gemini(*, system: str, user: str, schema: dict,
                              model: str | None, max_tokens: int,
                              temperature: float) -> AsyncIterator[dict]:
    """Gemini's generateContent endpoint does not stream partial structured
    JSON usefully, so this yields the complete response as a single delta,
    then "done" — callers already only look at the final Completion for
    structured calls, so this degrades gracefully rather than needing a
    separate code path per provider."""
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise LLMProviderError(
            "GEMINI_API_KEY is not set. Run `researchgenie config` to add your Gemini API key."
        )
    model = model or GEMINI_MODEL
    payload = {
        "contents": [{"role": "user", "parts": [{"text": user}]}],
        "systemInstruction": {"parts": [{"text": system}]},
        "generationConfig": {
            "temperature": temperature,
            "maxOutputTokens": max_tokens,
            "responseMimeType": "application/json",
            "responseSchema": _gemini_schema(schema),
        },
    }
    comp = Completion()
    try:
        async with httpx.AsyncClient(timeout=120) as http:
            resp = await http.post(
                f"{GEMINI_API_BASE}/models/{model}:generateContent",
                # The API key travels as a header, never a URL query parameter:
                # httpx (and anything that logs/prints a request or response,
                # including default exception tracebacks) includes the full
                # URL, so a query-string key would leak into any such output.
                headers={"x-goog-api-key": api_key}, json=payload,
            )
            if resp.status_code == 401 or resp.status_code == 403:
                raise LLMProviderError(
                    "Gemini rejected the configured API key. Run `researchgenie config` "
                    "to re-enter it."
                ) from None
            resp.raise_for_status()
            data = resp.json()
    except httpx.ConnectError:
        raise LLMProviderError(
            "Could not reach the Gemini API — check your internet connection."
        ) from None
    except httpx.HTTPStatusError as e:
        # `from None` suppresses Python's automatic exception chaining: the
        # original httpx.HTTPStatusError's own message includes the request
        # URL, which — even with the key no longer embedded in the URL
        # itself — is unnecessary detail to surface, and this keeps the
        # precedent of never propagating raw request/response internals.
        raise LLMProviderError(f"Gemini returned an error ({e.response.status_code}).") from None

    candidates = data.get("candidates") or []
    usage = data.get("usageMetadata", {})
    comp.prompt_tokens = usage.get("promptTokenCount", 0)
    comp.completion_tokens = usage.get("candidatesTokenCount", 0)
    if not candidates:
        comp.stop = "refusal"
        yield {"type": "done", "completion": comp}
        return
    finish_reason = candidates[0].get("finishReason", "")
    if finish_reason == "MAX_TOKENS":
        comp.stop = "length"
    parts = candidates[0].get("content", {}).get("parts", [])
    text = "".join(part.get("text", "") for part in parts)
    comp.text = text
    if not text.strip():
        comp.stop = "refusal"
    yield {"type": "delta", "text": text}
    yield {"type": "done", "completion": comp}


async def stream_json(*, system: str, user: str, schema: dict,
                      model: str | None = None,
                      max_tokens: int = 8000,
                      temperature: float = 0.3,
                      think: bool = True) -> AsyncIterator[dict]:
    """Yield {"type": "delta", "text": str} as output streams, then a single
    {"type": "done", "completion": Completion}. Raises LLMProviderError on failure.

    think: whether the model may reason before answering. Reasoning tokens count
    against max_tokens, so a simple/deterministic task should pass think=False —
    otherwise a tight max_tokens budget can be exhausted by reasoning before any
    actual JSON content is produced, silently yielding an empty (refusal-looking)
    completion. Ignored by the Gemini provider (no equivalent setting)."""
    if AI_PROVIDER == "gemini":
        async for event in _stream_json_gemini(
            system=system, user=user, schema=schema, model=model,
            max_tokens=max_tokens, temperature=temperature,
        ):
            yield event
        return
    model = model or DEFAULT_MODEL
    payload = {
        "model": model,
        "messages": [{"role": "system", "content": system},
                     {"role": "user", "content": user}],
        "format": schema,
        "think": think,
        "options": {
            "temperature": temperature,
            "num_predict": max_tokens,
            # Ollama's server-side default context window (commonly 4096) is far
            # smaller than this model's 256K support and silently truncates
            # generation once prompt+output exceed it, regardless of num_predict.
            # Scale with max_tokens instead of a large fixed value: on an 8GB GPU,
            # a bigger context reserves proportionally more VRAM for KV cache,
            # and pushing past what fits forces slow CPU-offloaded inference even
            # for small calls. Round up to the nearest power-of-two-ish tier so
            # most calls reuse an already-loaded context size instead of forcing
            # a reload every time num_ctx changes.
            "num_ctx": _context_window_for(max_tokens, len(system) + len(user)),
        },
        "stream": True,
    }
    comp = Completion()
    parts: list[str] = []
    try:
        async with httpx.AsyncClient(timeout=None) as http:
            async with http.stream("POST", f"{OLLAMA_HOST}/api/chat", json=payload) as resp:
                if resp.status_code == 404:
                    raise LLMProviderError(
                        f"Model '{model}' is not installed in Ollama. "
                        f"Run: ollama pull {model}")
                resp.raise_for_status()
                async for line in resp.aiter_lines():
                    if not line.strip():
                        continue
                    chunk = json.loads(line)
                    delta = (chunk.get("message") or {}).get("content", "")
                    if delta:
                        parts.append(delta)
                        yield {"type": "delta", "text": delta}
                    if chunk.get("done"):
                        comp.prompt_tokens = chunk.get("prompt_eval_count", 0)
                        comp.completion_tokens = chunk.get("eval_count", 0)
                        if chunk.get("done_reason") == "length":
                            comp.stop = "length"
    except httpx.ConnectError:
        raise LLMProviderError(
            "Could not reach the local Ollama server at "
            f"{OLLAMA_HOST}. Start it with: ollama serve")
    except httpx.HTTPStatusError as e:
        raise LLMProviderError(f"Ollama returned an error ({e.response.status_code}). "
                               "Check `ollama list` and the model name.")

    comp.text = "".join(parts)
    if not comp.text.strip():
        comp.stop = "refusal"
    yield {"type": "done", "completion": comp}


async def complete_json(*, system: str, user: str, schema: dict,
                        model: str | None = None,
                        max_tokens: int = 8000,
                        temperature: float = 0.3,
                        think: bool = True) -> Completion:
    """A structured-JSON completion, non-streaming from the caller's point of view."""
    comp = Completion()
    async for event in stream_json(system=system, user=user, schema=schema,
                                   model=model, max_tokens=max_tokens,
                                   temperature=temperature, think=think):
        if event["type"] == "done":
            comp = event["completion"]
    return comp


async def complete_tools(*, system: str, user: str, tools: list[dict],
                         model: str | None = None, max_tokens: int = 800,
                         temperature: float = 0.2) -> list[dict] | None:
    """One chat turn that may call tools (Ollama's /api/chat `tools`, #16).
    Returns the calls as [{"name": str | None, "arguments": Any}], unchecked —
    the caller validates them. None means this provider can't call tools here
    (Gemini, or an Ollama model without tool support), so the caller falls back.
    Raises LLMProviderError when the model can't be reached at all."""
    if AI_PROVIDER == "gemini":
        return None  # ponytail: Gemini function calling not wired; the domain fallback covers it
    model = model or DEFAULT_MODEL
    payload = {
        "model": model,
        "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
        "tools": tools, "stream": False, "think": False,
        "options": {"temperature": temperature, "num_predict": max_tokens,
                    "num_ctx": _context_window_for(max_tokens, len(system) + len(user) + len(json.dumps(tools)))},
    }
    try:
        async with httpx.AsyncClient(timeout=120) as http:
            resp = await http.post(f"{OLLAMA_HOST}/api/chat", json=payload)
    except httpx.ConnectError:
        raise LLMProviderError("Could not reach the local Ollama server at "
                               f"{OLLAMA_HOST}. Start it with: ollama serve") from None
    except httpx.HTTPError as error:
        raise LLMProviderError(f"Ollama tool call failed ({type(error).__name__}).") from None
    if resp.status_code == 400 and "does not support tools" in resp.text:
        return None
    if resp.status_code == 404:
        raise LLMProviderError(f"Model '{model}' is not installed in Ollama. Run: ollama pull {model}")
    if resp.status_code >= 400:
        raise LLMProviderError(f"Ollama returned an error ({resp.status_code}).")
    calls = ((resp.json().get("message") or {}).get("tool_calls")) or []
    return [{"name": (c.get("function") or {}).get("name"),
             "arguments": (c.get("function") or {}).get("arguments")}
            for c in calls if isinstance(c, dict)]


async def health_check(model: str | None = None) -> bool:
    """True if the configured provider is ready: for Gemini, that an API key
    is configured (a real reachability check costs a token-billed request,
    so this only checks configuration, not a live call); for Ollama, that
    the server is running and the model is installed."""
    if AI_PROVIDER == "gemini":
        return bool(os.getenv("GEMINI_API_KEY"))
    model = model or DEFAULT_MODEL
    try:
        async with httpx.AsyncClient(timeout=5) as http:
            resp = await http.get(f"{OLLAMA_HOST}/api/tags")
            resp.raise_for_status()
            names = {m["name"] for m in resp.json().get("models", [])}
            return model in names or f"{model}:latest" in names
    except httpx.HTTPError:
        return False
