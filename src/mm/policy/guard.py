"""GuardedSurface: the single place where safety is enforced on UI actions.

It wraps any Surface. The discovery agent and the replay executor only ever hold a GuardedSurface, so
neither can act outside policy, act while a human holds the session, or perform an irreversible
action that was not explicitly authorised for that one action. Network-level enforcement (every
request any frame makes) is installed in the browser via `request_policy()`.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager

from mm.handoff.lease import ControlLease, ControlNotHeld
from mm.policy.model import Policy, Risk, stricter
from mm.surface.base import (
    ActionType,
    ActResult,
    Checkpoint,
    Observation,
    ResolveResult,
    RoleStrategy,
    Surface,
    Target,
    TextStrategy,
)
from mm.surface.web import RequestPolicy


class PolicyViolation(RuntimeError):
    pass


def request_policy(policy: Policy) -> RequestPolicy:
    """The browser-side check: allowlist for every request, and commit requests only inside an authorised window."""
    def check(method: str, url: str, irreversible_window: bool) -> tuple[bool, str]:
        allowed, why = policy.url_allowed(url)
        if not allowed:
            return False, why
        if policy.is_irreversible_request(method, url) and not irreversible_window:
            return False, "irreversible request outside an authorised step"
        return True, ""
    return check


class GuardedSurface:
    def __init__(self, inner: Surface, policy: Policy, lease: ControlLease) -> None:
        self.inner, self.policy, self.lease = inner, policy, lease
        self.epoch = lease.state.epoch
        self._authorized = False
        self._names: dict[str, str] = {}  # ref -> accessible name, from the latest observation

    # --- control -------------------------------------------------------------------------------

    def resync(self) -> None:
        """Called by automation after a human hands control back *and* the screen has been re-verified."""
        self.epoch = self.lease.state.epoch

    @contextmanager
    def irreversible_authorized(self) -> Iterator[None]:
        """Authorise irreversible effects for exactly the actions performed inside this block."""
        self._authorized = True
        window = hasattr(self.inner, "irreversible_window")
        if window:
            self.inner.irreversible_window = True  # type: ignore[attr-defined]
        try:
            yield
        finally:
            self._authorized = False
            if window:
                self.inner.irreversible_window = False  # type: ignore[attr-defined]

    def risk_of(self, action: ActionType, target: Target | None = None, ref: str | None = None) -> Risk:
        names = [self._names.get(ref, "")] if ref else []
        if target is not None:
            names += [s.name for s in target.strategies if isinstance(s, RoleStrategy)]
            names += [s.text for s in target.strategies if isinstance(s, TextStrategy)]
        risk: Risk = "safe"
        for name in names:
            risk = stricter(risk, self.policy.classify_control(action, name))
        return risk

    # --- guarded actions -----------------------------------------------------------------------

    def act(self, action: ActionType, ref: str | None, value: str | None = None) -> ActResult:
        if (refusal := self._refuse(action, value, self.risk_of(action, ref=ref))) is not None:
            return refusal
        return self._blocked_requests_to_failure(self.inner.act(action, ref, value))

    def perform(self, action: ActionType, target: Target | None, value: str | None, timeout_ms: int) -> ActResult:
        if (refusal := self._refuse(action, value, self.risk_of(action, target=target))) is not None:
            return refusal
        return self._blocked_requests_to_failure(self.inner.perform(action, target, value, timeout_ms))

    def navigate(self, url: str) -> None:
        self.lease.require_agent(self.epoch)
        allowed, why = self.policy.url_allowed(url)
        if not allowed:
            raise PolicyViolation(why)
        self.inner.navigate(url)

    def _refuse(self, action: ActionType, value: str | None, risk: Risk) -> ActResult | None:
        try:
            self.lease.require_agent(self.epoch)
        except ControlNotHeld as exc:
            return ActResult(ok=False, error="control_not_held", detail=str(exc))
        if not self.policy.action_allowed(action):
            return ActResult(ok=False, error="policy_blocked", detail=f"action {action} is not allowed by policy")
        if action is ActionType.NAVIGATE and value:
            allowed, why = self.policy.url_allowed(value)
            if not allowed:
                return ActResult(ok=False, error="policy_blocked", detail=why)
        if risk == "irreversible" and not self._authorized:
            return ActResult(ok=False, error="policy_blocked",
                             detail="irreversible action without authorisation (approval required)")
        return None

    def _blocked_requests_to_failure(self, res: ActResult) -> ActResult:
        blocked = self.inner.drain_blocked_requests()
        if blocked and res.ok:
            # The action ran but something it triggered was refused at the network layer: never report that as ok.
            return res.model_copy(update={"ok": False, "error": "policy_blocked", "dispatched": True,
                                          "detail": "request blocked by policy: " + "; ".join(blocked)})
        return res

    # --- pass-through (reads never change application state) ------------------------------------

    def observe(self, with_screenshot: bool = False) -> Observation:
        obs = self.inner.observe(with_screenshot)
        self._names = {e.ref: e.name for e in obs.elements}
        return obs

    def resolve(self, target: Target, timeout_ms: int) -> ResolveResult:
        return self.inner.resolve(target, timeout_ms)

    def check(self, checkpoint: Checkpoint, timeout_ms: int) -> tuple[bool, str]:
        return self.inner.check(checkpoint, timeout_ms)

    def blocking_overlays(self) -> list[str]:
        return self.inner.blocking_overlays()

    def screenshot(self, path: str) -> None:
        self.inner.screenshot(path)

    def idle(self, ms: int) -> None:
        self.inner.idle(ms)

    def drain_blocked_requests(self) -> list[str]:
        return self.inner.drain_blocked_requests()

    def frame_urls(self) -> dict[str, str]:
        urls = getattr(self.inner, "frame_urls", None)
        return dict(urls()) if callable(urls) else {}

    def close(self) -> None:
        self.inner.close()
