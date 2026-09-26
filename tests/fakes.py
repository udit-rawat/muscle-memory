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

    def idle(self, ms: int) -> None:
        pass

    def drain_blocked_requests(self) -> list[str]:
        return []

    def screenshot(self, path: str) -> None:
        with open(path, "wb") as fh:
            fh.write(b"")

    def close(self) -> None:
        pass


# --- shared test helpers --------------------------------------------------------------------------

from mm.artifact.approval import Approval  # noqa: E402
from mm.artifact.schema import Capability  # noqa: E402
from mm.policy.model import NetworkPolicy, Policy  # noqa: E402

# For browser-free tests: any origin, no risk rules (the artifact's own risk levels still apply).
PERMISSIVE = Policy(network=NetworkPolicy(allow_origins=["*"]))


def approval_for(cap: Capability, by: str = "test-reviewer") -> Approval:
    """A reviewer's sign-off covering every irreversible step of `cap`, exactly as `mm approve` computes it."""
    from mm.artifact.approval import content_digest
    return Approval(capability_id=cap.id, version=cap.version, tenant=cap.app.tenant, sha256=content_digest(cap),
                    approved_by=by,
                    approved_at="2026-09-26T00:00:00+00:00",
                    irreversible_steps=[s.id for s in cap.steps if s.risk == "irreversible"])
