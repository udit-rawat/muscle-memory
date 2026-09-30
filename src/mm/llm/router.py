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


class LLMUnavailable(RuntimeError):
    """No provider produced a valid answer. `errors` holds one line per provider, for the run log."""

    def __init__(self, errors: list[str]) -> None:
        super().__init__("no LLM provider produced a valid answer: " + " | ".join(errors))
        self.errors = errors


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
            client = _client(s.mm_fallback_base_url, key)
            # A comma-separated list: each model is tried in turn (free-tier models are often "high demand").
            for model in [m.strip() for m in s.mm_fallback_model.split(",") if m.strip()]:
                providers.append(Provider("gemini", model, client))
        return cls(providers)

    def structured(
        self,
        response_model: type[T],
        messages: list[dict[str, object]],
        max_retries: int = 2,
    ) -> LLMCall[T]:
        """Validated `response_model` from the first provider that answers.

        On a validation error Instructor re-asks the same provider with the error, up to `max_retries`.
        Instructor wraps every failure in its own exception, so the decision is made on the root cause:
        availability problems (rate limit, connection, timeout, 5xx) and exhausted validation retries fail
        over to the next provider; configuration errors (401/403/400/404) are raised at once, because the
        next provider would only hide a broken setup. Raises LLMUnavailable when every provider failed.
        """
        errors: list[str] = []
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
            except Exception as exc:  # noqa: BLE001 — classified below, re-raised when not recoverable
                cause = _root_cause(exc)
                if isinstance(cause, APIStatusError) and not isinstance(cause, RateLimitError) \
                        and cause.status_code < 500:
                    raise cause from exc
                errors.append(f"{p.name}: {type(cause).__name__}: {str(cause)[:160]}")
                log.warning("provider %s failed (%s); trying the next one", p.name, type(cause).__name__)
        raise LLMUnavailable(errors)


def _root_cause(exc: BaseException) -> BaseException:
    """Walk past wrapper exceptions (Instructor's retry exception) to the error that actually happened."""
    seen = {id(exc)}
    while True:
        nxt = exc.__cause__ or exc.__context__
        if nxt is None or id(nxt) in seen or isinstance(exc, (APIConnectionError, APIStatusError)):
            return exc
        seen.add(id(nxt))
        exc = nxt


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
