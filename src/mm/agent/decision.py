"""The one thing the LLM produces per turn: a typed next action.

Validation runs against the live observation, so an invented ref or a missing value is rejected
and Instructor re-asks with the error message. The screen context is bound to a per-turn subclass
(`for_screen`) rather than passed as Instructor `context`: Instructor renders messages as Jinja
templates when `context` is given, and page text is untrusted input that must never be templated.
"""

from __future__ import annotations

import re
from typing import Any, ClassVar, Literal

from pydantic import BaseModel, Field, model_validator

AgentAction = Literal["click", "fill", "select", "extract", "navigate", "done", "fail", "request_human"]
_SNAKE = re.compile(r"^[a-z][a-z0-9_]*$")


class Decision(BaseModel):
    screen: ClassVar[dict[str, Any]] = {}
    thought: str = Field(description="One short sentence: why this action moves toward the goal.")
    action: AgentAction
    ref: str | None = Field(None, description="Element ref from the current screen, e.g. 'e12'.")
    value: str | None = Field(
        None, description="fill: text or placeholder like {{member_id}} / {{secret:NAME}}; select: option label; "
                          "navigate: url.")
    output_name: str | None = Field(None, description="extract: snake_case name for the value read.")
    summary: str | None = Field(None, description="done/fail/request_human: what happened and why.")

    @model_validator(mode="after")
    def _check(self) -> Decision:
        ctx = type(self).screen
        refs: set[str] = ctx.get("refs", set())
        inputs: set[str] = ctx.get("inputs", set())
        secrets: set[str] = ctx.get("secrets", set())

        if self.action in ("click", "fill", "select", "extract"):
            if not self.ref:
                raise ValueError(f"{self.action} requires 'ref'")
            if refs and self.ref not in refs:
                raise ValueError(f"ref {self.ref!r} is not on the current screen; choose one of the listed refs")
        if self.action in ("fill", "select", "navigate") and not self.value:
            raise ValueError(f"{self.action} requires 'value'")
        if self.action == "extract" and not (self.output_name and _SNAKE.match(self.output_name)):
            raise ValueError("extract requires a snake_case 'output_name'")
        if self.action in ("done", "fail", "request_human") and not self.summary:
            raise ValueError(f"{self.action} requires 'summary'")
        for m in re.finditer(r"\{\{\s*(secret:)?([A-Za-z_][A-Za-z0-9_]*)\s*\}\}", self.value or ""):
            known = secrets if m.group(1) else inputs
            if m.group(2) not in known:
                kind = "secret" if m.group(1) else "input"
                raise ValueError(f"unknown {kind} placeholder {m.group(0)}; available: {sorted(known)}")
        return self


def for_screen(refs: set[str], inputs: set[str], secrets: set[str]) -> type[Decision]:
    """A Decision type whose validator knows what is on the current screen."""

    class ScreenDecision(Decision):
        screen: ClassVar[dict[str, Any]] = {"refs": refs, "inputs": inputs, "secrets": secrets}

    ScreenDecision.__name__ = ScreenDecision.__qualname__ = "Decision"
    return ScreenDecision
