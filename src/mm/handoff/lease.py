"""Who controls the live session right now.

One lease per live session. Automation may act only while it holds the lease *at the epoch it
acquired*: every transfer bumps the epoch, so an automated action that was already in flight when a
human took over, or that resumes without re-verifying the screen, is refused rather than racing the
human. The lease is the mechanism, not a convention: GuardedSurface checks it before every action.
"""

from __future__ import annotations

import threading
from collections.abc import Callable
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from enum import StrEnum


class Owner(StrEnum):
    AGENT = "AGENT"  # automation (discovery agent or replay executor)
    HUMAN = "HUMAN"  # an operator working in the same browser session
    NONE = "NONE"  # the run is over; nobody may act


class ControlNotHeld(RuntimeError):
    pass


@dataclass(frozen=True)
class LeaseState:
    owner: Owner
    epoch: int
    holder: str
    since: str
    reason: str = ""


class ControlLease:
    def __init__(self, on_change: Callable[[LeaseState], None] | None = None) -> None:
        self._lock = threading.Lock()
        self._state = LeaseState(Owner.AGENT, 1, "automation", _now(), "run started")
        self._on_change = on_change

    @property
    def state(self) -> LeaseState:
        with self._lock:
            return self._state

    def require_agent(self, epoch: int) -> None:
        s = self.state
        if s.owner is not Owner.AGENT or s.epoch != epoch:
            raise ControlNotHeld(f"automation holds epoch {epoch}, but the lease is {s.owner} at epoch {s.epoch}")

    def to_human(self, holder: str, reason: str) -> LeaseState:
        return self._transfer({Owner.AGENT}, Owner.HUMAN, holder, reason)

    def to_agent(self, reason: str) -> LeaseState:
        return self._transfer({Owner.HUMAN, Owner.AGENT}, Owner.AGENT, "automation", reason)

    def end(self, reason: str) -> LeaseState:
        return self._transfer(set(Owner), Owner.NONE, "nobody", reason)

    def _transfer(self, allowed_from: set[Owner], to: Owner, holder: str, reason: str) -> LeaseState:
        with self._lock:
            if self._state.owner not in allowed_from:
                raise ControlNotHeld(f"cannot hand control to {to} from {self._state.owner}")
            self._state = replace(self._state, owner=to, epoch=self._state.epoch + 1, holder=holder,
                                  since=_now(), reason=reason)
            state = self._state
        if self._on_change:
            self._on_change(state)
        return state


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")
