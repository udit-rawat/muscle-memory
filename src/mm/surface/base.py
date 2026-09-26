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

STRUCTURAL = {"attr", "css"}  # positional/markup-bound: fine to find a button, never trusted for a read


class Target(BaseModel):
    frame_path: list[str] = Field(default_factory=list, description="Frame names from the top document.")
    strategies: list[Strategy] = Field(min_length=1, description="Tried in order; each must match exactly one.")
    description: str = Field("", description="Human-readable, e.g. 'textbox \"Member #\" in frame main'.")


class Checkpoint(BaseModel):
    """A condition on the current UI state. Used as a post-step checkpoint (so we never assume a click
    worked), as a success condition, and as the trigger of a detector.

    kind:
      url_matches    regex searched in the frame's URL
      text_visible   literal text contained in the frame's visible text
      text_matches   regex searched in the frame's visible text (the matched line is reported)
      target_present a Target resolves to exactly one element
    """

    kind: Literal["url_matches", "text_visible", "text_matches", "target_present"]
    frame_path: list[str] = Field(default_factory=list, description="Frame names from the top document.")
    any_frame: bool = Field(False, description="Hold if the condition holds in any frame (frame_path ignored).")
    pattern: str | None = None
    target: Target | None = None  # for target_present


# --- observation (discovery only) ---------------------------------------------------------------


class ElementRef(BaseModel):
    """One actionable/readable node in an observation, addressable by a short ref the LLM can cite."""

    ref: str
    role: str
    name: str = ""
    value: str | None = None
    options: list[str] | None = None  # for dropdowns: the option labels
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
    truncated: int = Field(0, description="Elements left out because the observation hit its size cap.")
    screenshot_png: bytes | None = Field(default=None, repr=False, exclude=True)


ActError = Literal[
    "not_found",          # no strategy resolved to exactly one element: nothing was dispatched
    "untargetable",       # discovery: no replayable locator could be built, so the action was not performed
    "action_failed",      # the element was found and the action attempted; its effect is unknown
    "navigation_failed",  # the page could not be loaded (app down, DNS, TLS, refused)
    "policy_blocked",     # refused by the safety policy (allowlist, action type, unapproved irreversible step)
    "control_not_held",   # automation does not hold the session's control lease (a human does)
]


class ActResult(BaseModel):
    ok: bool
    error: ActError | None = None
    dispatched: bool | None = None  # set when an error happened after the action was already sent to the app
    detail: str = ""
    target: Target | None = None  # discovery: verified, replayable target for the element acted on
    strategy_index: int | None = None  # replay: which strategy matched (>0 is a drift signal)
    attempts: list[str] = Field(default_factory=list)  # replay: per-strategy outcome when resolving
    extracted: str | None = None
    options: list[str] | None = None  # discovery, select: the options offered at record time
    frame_urls: dict[str, str] = Field(default_factory=dict)  # "main" -> url, after the action settled


class ResolveResult(BaseModel):
    found: bool
    strategy_index: int | None = None
    attempts: list[str] = Field(default_factory=list)  # per-strategy outcome, for debugging


def session_closed(exc: BaseException) -> bool:
    """True when an error means the live session itself is gone (window closed, browser exited)."""
    text = f"{type(exc).__name__}: {exc}"
    return "TargetClosedError" in text or "has been closed" in text or "Browser closed" in text


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
    def blocking_overlays(self) -> list[str]: ...

    def screenshot(self, path: str) -> None: ...
    def idle(self, ms: int) -> None: ...  # let the session's event loop run (e.g. while a human works)
    def drain_blocked_requests(self) -> list[str]: ...  # requests the network policy refused since last call
    def close(self) -> None: ...
