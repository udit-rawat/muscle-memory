"""Fixes found by probing the fallback model: credential fields only take secret placeholders; the fallback
can be a list of models."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
from pydantic import BaseModel, ValidationError

from mm.agent.decision import for_screen

Screen = for_screen({"e1", "e2", "e3"}, {"member_id"}, {"MOCKBANK_USERNAME", "MOCKBANK_PASSWORD"},
                    credential_refs={"e1", "e2"})


@pytest.mark.parametrize("value", ["admin", "operator1", "{{member_id}}", "x{{secret:MOCKBANK_PASSWORD}}"])
def test_a_credential_field_never_takes_a_typed_or_guessed_value(value: str) -> None:
    with pytest.raises(ValidationError, match="credential"):
        Screen(thought="t", action="fill", ref="e1", value=value)


def test_a_credential_field_takes_a_secret_placeholder() -> None:
    assert Screen(thought="t", action="fill", ref="e2", value="{{secret:MOCKBANK_PASSWORD}}").ref == "e2"


def test_ordinary_fields_still_take_values() -> None:
    assert Screen(thought="t", action="fill", ref="e3", value="{{member_id}}").ref == "e3"


def test_the_surface_marks_credential_fields() -> None:
    from mm.surface.web import WebSurface
    s = WebSurface(headless=True)
    try:
        s.page.set_content('<label>Operator ID <input name="u"></label><label>Password <input type="password" '
                           'name="p"></label><label>Member # <input name="m"></label>'
                           '<input type="text" title="PIN" name="x"><input type="text" title="Nickname" name="n">')
        marked = {e.name: e.credential for e in s.observe().elements}
    finally:
        s.close()
    assert marked == {"Operator ID": True, "Password": True, "Member #": False, "PIN": True, "Nickname": False}


def test_the_fallback_can_be_a_list_of_models() -> None:
    from pydantic import SecretStr

    from mm.config import Settings
    from mm.llm.router import LLMRouter
    s = Settings(groq_api_key=SecretStr("gsk_x"), gemini_api_key=SecretStr("AIza_x"),
                 mm_fallback_model="gemini-3.8-flash, gemini-3.1-flash-lite")
    router = LLMRouter.from_settings(s)
    assert [(p.name, p.model) for p in router.providers] == [
        ("groq", "openai/gpt-oss-120b"), ("gemini", "gemini-3.8-flash"), ("gemini", "gemini-3.1-flash-lite")]


def test_failover_walks_the_whole_list() -> None:
    from mm.llm.router import LLMRouter, Provider, _client

    class Ping(BaseModel):
        ok: bool

    def stub(value: Any) -> Any:
        completions = SimpleNamespace(create_with_completion=lambda **_: (value, SimpleNamespace(usage=None)))
        return Provider("gemini", "gemini-3.1-flash-lite", SimpleNamespace(chat=SimpleNamespace(  # type: ignore[arg-type]
            completions=completions)))

    dead = [Provider("groq", "x", _client("http://127.0.0.1:9/v1", "k")),
            Provider("gemini", "gemini-3.8-flash", _client("http://127.0.0.1:9/v1", "k"))]
    call = LLMRouter([*dead, stub(Ping(ok=True))]).structured(Ping, [{"role": "user", "content": "x"}])
    assert call.model == "gemini-3.1-flash-lite"
