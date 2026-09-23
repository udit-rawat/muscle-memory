"""Thin provider router: primary (Groq) with fallback (Gemini), both via OpenAI-compatible APIs.

Deliberately not LiteLLM: two providers that speak the same wire format don't need a
100-provider abstraction, and a smaller dependency surface matters for software that
would sit next to banking systems.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Generic, TypeVar

import instructor
from openai import APIConnectionError, APIStatusError, OpenAI, RateLimitError
from pydantic import BaseModel, SecretStr

from mm.config import Settings, get_settings

log = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)

# Errors worth failing over on. Anything else (bad request, auth) is a bug and should surface.
_FAILOVER_ERRORS = (RateLimitError, APIConnectionError)


@dataclass(frozen=True)
class Provider:
    name: str
    model: str
    client: instructor.Instructor
    extra: dict[str, object] = field(default_factory=dict)  # provider-specific request params


@dataclass(frozen=True)
class LLMCall(Generic[T]):
    value: T
    provider: str
    model: str
    input_tokens: int
    output_tokens: int


class NoProviderConfigured(RuntimeError):
    pass


class LLMRouter:
    def __init__(self, providers: list[Provider]) -> None:
        if not providers:
            raise NoProviderConfigured("Set GROQ_API_KEY and/or GEMINI_API_KEY in .env")
        self.providers = providers

    @classmethod
    def from_settings(cls, settings: Settings | None = None) -> LLMRouter:
        s = settings or get_settings()
        providers: list[Provider] = []
        if key := _real_key(s.groq_api_key):
            # Low reasoning effort keeps gpt-oss well inside Groq's free-tier tokens-per-minute.
            providers.append(Provider("groq", s.mm_primary_model, _client(s.mm_primary_base_url, key),
                                      {"reasoning_effort": "low"}))
        if key := _real_key(s.gemini_api_key):
            providers.append(Provider("gemini", s.mm_fallback_model, _client(s.mm_fallback_base_url, key)))
        return cls(providers)

    def structured(
        self,
        response_model: type[T],
        messages: list[dict[str, object]],
        max_retries: int = 2,
    ) -> LLMCall[T]:
        """Validated `response_model` from the first provider that answers.

        On a validation error Instructor re-asks the same provider with the error, up to `max_retries`.
        """
        last_exc: Exception | None = None
        for p in self.providers:
            try:
                result, completion = p.client.chat.completions.create_with_completion(
                    model=p.model,
                    response_model=response_model,
                    messages=messages,  # type: ignore[arg-type]
                    max_retries=max_retries,
                    temperature=0,
                    **p.extra,  # type: ignore[arg-type]
                )
                usage = getattr(completion, "usage", None)
                return LLMCall(result, p.name, p.model,
                               getattr(usage, "prompt_tokens", 0) or 0, getattr(usage, "completion_tokens", 0) or 0)
            except _FAILOVER_ERRORS as exc:
                log.warning("provider %s unavailable (%s); failing over", p.name, type(exc).__name__)
                last_exc = exc
            except APIStatusError as exc:
                if exc.status_code >= 500:
                    log.warning("provider %s returned %s; failing over", p.name, exc.status_code)
                    last_exc = exc
                    continue
                raise
        assert last_exc is not None
        raise last_exc


def _real_key(secret: SecretStr | None) -> str | None:
    """Treat unset keys and the .env.example placeholders as 'not configured'."""
    if secret is None:
        return None
    value = secret.get_secret_value().strip()
    return None if not value or value.startswith("your-") else value


def _client(base_url: str, api_key: str) -> instructor.Instructor:
    # JSON mode is the most portable across Groq and Gemini's OpenAI-compat layers.
    # max_retries covers 429s with the server's retry-after before we fail over to the next provider.
    client = OpenAI(base_url=base_url, api_key=api_key, max_retries=3, timeout=60)
    return instructor.from_openai(client, mode=instructor.Mode.JSON)
