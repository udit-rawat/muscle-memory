"""The seam between "how we perceive/act on a surface" and "the recorded flow".

Artifacts only ever contain surface-neutral `Target`s (role, accessible name, label, text,
frame path). Each Surface implementation (web, legacy web, desktop) decides how to resolve
them. Nothing above this module imports Playwright.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Protocol

from pydantic import BaseModel, Field


class ActionType(StrEnum):
    NAVIGATE = "navigate"
    CLICK = "click"
    FILL = "fill"
    SELECT = "select"
    EXTRACT = "extract"


class ElementRef(BaseModel):
    """One actionable/readable node in an observation, addressable by a short ref the LLM can cite."""

    ref: str
    role: str
    name: str = ""
    value: str | None = None
    frame_path: list[str] = Field(default_factory=list)


class Observation(BaseModel):
    url: str
    title: str
    elements: list[ElementRef]
    text_excerpt: str = ""
    screenshot_png: bytes | None = Field(default=None, repr=False, exclude=True)


class ActResult(BaseModel):
    ok: bool
    detail: str = ""


class Surface(Protocol):
    """A live, controllable UI session. One instance == one session (cookies, window, process)."""

    def observe(self, with_screenshot: bool = False) -> Observation: ...

    def act(self, action: ActionType, ref: str, value: str | None = None) -> ActResult: ...

    def close(self) -> None: ...
