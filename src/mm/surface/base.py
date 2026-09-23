"""The seam between "how we perceive/act on a surface" and "the recorded flow".

Artifacts only ever contain surface-neutral `Target`s (role, accessible name, title, table
anchors, frame path). Each Surface implementation (web, legacy web, desktop) decides how to
resolve them. Nothing above this module imports Playwright.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Annotated, Literal, Protocol

from pydantic import BaseModel, Field


class ActionType(StrEnum):
    NAVIGATE = "navigate"
    CLICK = "click"
    FILL = "fill"
    SELECT = "select"
    EXTRACT = "extract"


# --- targeting ---------------------------------------------------------------------------------
# Ordered from most semantic (survives markup changes, maps onto desktop accessibility APIs) to
# most structural (last resort; a hit on one of these is logged as a drift signal).


class RoleStrategy(BaseModel):
    by: Literal["role"] = "role"
    role: str
    name: str


class TitleStrategy(BaseModel):
    by: Literal["title"] = "title"
    title: str


class TextStrategy(BaseModel):
    by: Literal["text"] = "text"
    text: str


class TableCellStrategy(BaseModel):
    """A cell addressed by meaning, not position: "the Balance column of the row whose key is Share Savings".

    Used for reads, where the element's own text is the *value* and so cannot identify it.
    """

    by: Literal["table_cell"] = "table_cell"
    row_key: str
    column_header: str | None = None
    column_index: int | None = None  # 1-based; used when the table has no header row


class AttrStrategy(BaseModel):
    """Form-control `name` attribute. Legacy apps rarely have ids but their form posts need names."""

    by: Literal["attr"] = "attr"
    tag: str
    name: str


class CssStrategy(BaseModel):
    by: Literal["css"] = "css"
    selector: str


Strategy = Annotated[
    RoleStrategy | TitleStrategy | TextStrategy | TableCellStrategy | AttrStrategy | CssStrategy,
    Field(discriminator="by"),
]

STRUCTURAL = {"attr", "css"}


class Target(BaseModel):
    frame_path: list[str] = Field(default_factory=list, description="Frame names from the top document.")
    strategies: list[Strategy] = Field(min_length=1, description="Tried in order; each must match exactly one.")
    description: str = Field("", description="Human-readable, e.g. 'textbox \"Member #\" in frame main'.")


class Checkpoint(BaseModel):
    """A condition asserted after a step, so we never assume a click worked."""

    kind: Literal["url_matches", "text_visible", "target_present"]
    frame_path: list[str] = Field(default_factory=list)
    pattern: str | None = None  # regex for url_matches, literal text for text_visible
    target: Target | None = None  # for target_present


# --- observation (discovery only) ---------------------------------------------------------------


class ElementRef(BaseModel):
    """One actionable/readable node in an observation, addressable by a short ref the LLM can cite."""

    ref: str
    role: str
    name: str = ""
    value: str | None = None
    frame_path: list[str] = Field(default_factory=list)


class FrameText(BaseModel):
    frame_path: list[str]
    url: str
    text: str


class Observation(BaseModel):
    url: str
    title: str
    elements: list[ElementRef]
    frames: list[FrameText] = Field(default_factory=list)
    screenshot_png: bytes | None = Field(default=None, repr=False, exclude=True)


class ActResult(BaseModel):
    ok: bool
    detail: str = ""
    target: Target | None = None  # discovery: verified, replayable target for the element acted on
    strategy_index: int | None = None  # replay: which strategy matched (>0 is a drift signal)
    attempts: list[str] = Field(default_factory=list)  # replay: per-strategy outcome when resolving
    extracted: str | None = None
    frame_urls: dict[str, str] = Field(default_factory=dict)  # "main" -> url, after the action settled


class ResolveResult(BaseModel):
    found: bool
    strategy_index: int | None = None
    attempts: list[str] = Field(default_factory=list)  # per-strategy outcome, for debugging


class Surface(Protocol):
    """A live, controllable UI session. One instance == one session (cookies, window, process)."""

    # discovery: act on refs from the latest observation, return a verified replayable target
    def observe(self, with_screenshot: bool = False) -> Observation: ...
    def act(self, action: ActionType, ref: str | None, value: str | None = None) -> ActResult: ...

    # replay: act on recorded targets, no refs, no model
    def navigate(self, url: str) -> None: ...
    def resolve(self, target: Target, timeout_ms: int) -> ResolveResult: ...
    def perform(
        self, action: ActionType, target: Target | None, value: str | None, timeout_ms: int
    ) -> ActResult: ...
    def check(self, checkpoint: Checkpoint, timeout_ms: int) -> tuple[bool, str]: ...

    def screenshot(self, path: str) -> None: ...
    def close(self) -> None: ...
