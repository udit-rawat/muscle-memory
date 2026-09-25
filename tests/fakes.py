"""A scripted, browser-free Surface for testing replay control flow deterministically."""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass, field

from mm.surface.base import ActionType, ActResult, Checkpoint, Observation, ResolveResult, Target


@dataclass
class FakeState:
    url: str = "http://app/login"
    texts: set[str] = field(default_factory=set)  # visible text fragments
    present: set[str] = field(default_factory=set)  # target descriptions that resolve
    overlays: list[str] = field(default_factory=list)
    performed: list[str] = field(default_factory=list)


Effect = Callable[[FakeState], ActResult | None]


class FakeSurface:
    """`effects` maps a target description (or navigate url) to what performing it does to the state."""

    def __init__(self, state: FakeState, effects: dict[str, Effect]) -> None:
        self.state, self.effects = state, effects

    def perform(self, action: ActionType, target: Target | None, value: str | None, timeout_ms: int) -> ActResult:
        key = target.description if target else (value or "")
        if target is not None and key not in self.state.present:
            return ActResult(ok=False, error="not_found", detail="target not found", attempts=[f"{key}: 0 matches"])
        self.state.performed.append(key)
        effect = self.effects.get(key)
        result = effect(self.state) if effect else None
        return result or ActResult(ok=True, strategy_index=0 if target else None)

    def check(self, cp: Checkpoint, timeout_ms: int) -> tuple[bool, str]:
        if cp.kind == "url_matches":
            return bool(re.search(cp.pattern or "", self.state.url)), self.state.url
        if cp.kind == "text_visible":
            hit = next((t for t in self.state.texts if (cp.pattern or "") in t), None)
            return hit is not None, hit or ""
        if cp.kind == "text_matches":
            hit = next((t for t in self.state.texts if re.search(cp.pattern or "", t)), None)
            return hit is not None, hit or ""
        present = cp.target is not None and cp.target.description in self.state.present
        return present, "present" if present else "absent"

    def blocking_overlays(self) -> list[str]:
        return list(self.state.overlays)

    def navigate(self, url: str) -> None:
        self.state.url = url

    def resolve(self, target: Target, timeout_ms: int) -> ResolveResult:
        return ResolveResult(found=target.description in self.state.present, strategy_index=0)

    def observe(self, with_screenshot: bool = False) -> Observation:
        return Observation(url=self.state.url, title="", elements=[])

    def act(self, action: ActionType, ref: str | None, value: str | None = None) -> ActResult:
        self.state.performed.append(f"{action}:{ref}")
        return ActResult(ok=True, frame_urls={"": self.state.url})

    def frame_urls(self) -> dict[str, str]:
        return {"": self.state.url}

    def screenshot(self, path: str) -> None:
        with open(path, "wb") as fh:
            fh.write(b"")

    def close(self) -> None:
        pass
