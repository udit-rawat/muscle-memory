"""Intervention requests: how a run asks a human for help, and how the answer comes back.

Lifecycle:
  open --claim--> claimed --resume--> resolved     (human took control of the live session, then gave it back)
  open --approve/reject--> approved | rejected     (a decision only; control never changes hands)
  open|claimed --abort--> aborted                  (the human stops the run)
  open|claimed --(timeout)--> expired

The run that raised the request blocks in `wait()`, which keeps the browser's event loop turning, so
the human's clicks in the live window are captured while they work.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field

from mm.evidence.recorder import RunRecorder
from mm.handoff.lease import ControlLease, LeaseState, Owner
from mm.redact import mask_pii
from mm.surface.base import session_closed


class Kind(StrEnum):
    UNRECOVERABLE_STATE = "unrecoverable_state"  # replay hit something no detector handles
    STUCK = "stuck"  # the discovery agent cannot make progress or asked for help
    APPROVAL_REQUIRED = "approval_required"  # an irreversible action needs a human's go-ahead


class Status(StrEnum):
    OPEN = "open"
    CLAIMED = "claimed"
    RESOLVED = "resolved"
    APPROVED = "approved"
    REJECTED = "rejected"
    ABORTED = "aborted"
    EXPIRED = "expired"
    SESSION_CLOSED = "session_closed"  # the live browser session ended while the request was open


TERMINAL = {Status.RESOLVED, Status.APPROVED, Status.REJECTED, Status.ABORTED, Status.EXPIRED, Status.SESSION_CLOSED}


class HumanAction(BaseModel):
    at: str
    kind: str  # click | change
    tag: str = ""
    name: str = ""
    frame: str = ""
    detail: str = ""  # never a typed value: only its length, or a chosen option's label


class Intervention(BaseModel):
    id: str
    run_id: str
    kind: Kind
    subject: str  # capability id (replay) or the goal (discovery)
    step_id: str | None = None
    reason: str
    expected: str | None = None
    observed: str | None = None
    screenshot: str | None = None
    page_excerpt: str = ""
    suggested: list[str] = Field(default_factory=list)
    created_at: str
    status: Status = Status.OPEN
    claimed_by: str | None = None
    resolved_by: str | None = None
    resolved_at: str | None = None
    note: str = ""
    human_actions: list[HumanAction] = Field(default_factory=list)


class InterventionSummary(BaseModel):
    id: str
    kind: Kind
    status: Status
    step_id: str | None
    claimed_by: str | None
    resolved_by: str | None
    human_actions: int


class HandoffError(RuntimeError):
    pass


class HandoffController:
    """Owns the lease and the interventions of one run. The operator console calls into it from its own
    thread; the run calls `open` and `wait` from the browser thread."""

    def __init__(self, recorder: RunRecorder) -> None:
        self.rec = recorder
        self.lease = ControlLease(on_change=self._lease_changed)
        self._lock = threading.Lock()
        self._items: dict[str, Intervention] = {}
        self.operator_tick: Callable[[Intervention], None] | None = None  # a simulated operator, if any

    # --- raised by the run ---------------------------------------------------------------------

    def open(self, kind: Kind, subject: str, reason: str, **fields: Any) -> Intervention:
        clean = {k: self._clean(v) if k in ("expected", "observed", "page_excerpt") and isinstance(v, str) else v
                 for k, v in fields.items()}
        item = Intervention(id=f"iv-{uuid4().hex[:8]}", run_id=self.rec.run_id, kind=kind, subject=subject,
                            reason=self._clean(reason), created_at=_now(), **clean)
        with self._lock:
            self._items[item.id] = item
        self._persist(item, "intervention_opened")
        return item

    def wait(self, item_id: str, idle: Callable[[int], None], timeout_s: float) -> Intervention:
        """Block until the intervention reaches a terminal state, pumping the browser meanwhile."""
        deadline = time.monotonic() + timeout_s
        while True:
            item = self.get(item_id)
            if item.status in TERMINAL:
                return item
            if time.monotonic() >= deadline:
                return self._finish(item_id, Status.EXPIRED, "nobody", "no operator response before the timeout")
            if self.operator_tick is not None:
                self.operator_tick(item)
            try:
                idle(200)
            except Exception as exc:  # noqa: BLE001 — classified: a closed session ends the request, anything else is a bug
                if not session_closed(exc):
                    raise
                if self.lease.state.owner is Owner.HUMAN:
                    self.lease.to_agent("the browser session was closed")
                return self._finish(item_id, Status.SESSION_CLOSED, "session", "the browser session was closed")

    # --- called by the operator (console) ------------------------------------------------------

    def claim(self, item_id: str, by: str) -> Intervention:
        item = self._require(item_id, {Status.OPEN}, "claim")
        if item.kind is Kind.APPROVAL_REQUIRED:
            raise HandoffError("an approval request is decided with approve/reject; control stays with automation")
        self.lease.to_human(by, f"{item.kind}: {item.reason}")
        return self._update(item_id, "intervention_claimed", status=Status.CLAIMED, claimed_by=by)

    def resume(self, item_id: str, by: str, note: str = "") -> Intervention:
        self._require(item_id, {Status.CLAIMED}, "resume")
        self.lease.to_agent(f"handed back by {by}")
        return self._finish(item_id, Status.RESOLVED, by, note)

    def approve(self, item_id: str, by: str, note: str = "") -> Intervention:
        self._require(item_id, {Status.OPEN}, "approve", kind=Kind.APPROVAL_REQUIRED)
        return self._finish(item_id, Status.APPROVED, by, note)

    def reject(self, item_id: str, by: str, note: str = "") -> Intervention:
        self._require(item_id, {Status.OPEN}, "reject", kind=Kind.APPROVAL_REQUIRED)
        return self._finish(item_id, Status.REJECTED, by, note)

    def abort(self, item_id: str, by: str, note: str = "") -> Intervention:
        item = self._require(item_id, {Status.OPEN, Status.CLAIMED}, "abort")
        if item.status is Status.CLAIMED:
            self.lease.to_agent(f"aborted by {by}")  # the run is about to stop; it needs control back to clean up
        return self._finish(item_id, Status.ABORTED, by, note)

    # --- human activity captured in the live session -------------------------------------------

    def record_human_action(self, payload: dict[str, Any]) -> None:
        """Called for every click/change in the browser. Kept only while a human holds the lease."""
        if self.lease.state.owner is not Owner.HUMAN:
            return
        action = HumanAction(at=_now(), kind=str(payload.get("kind", "")), tag=str(payload.get("tag", "")),
                             name=self._clean(str(payload.get("name", "")))[:80],
                             frame=str(payload.get("frame", "")), detail=self._clean(str(payload.get("detail", ""))))
        with self._lock:
            claimed = [i for i in self._items.values() if i.status is Status.CLAIMED]
            if not claimed:
                return
            claimed[-1].human_actions.append(action)
            item = claimed[-1]
        self.rec.event("human_action", intervention=item.id, **action.model_dump())
        self._persist(item, None)

    # --- queries -------------------------------------------------------------------------------

    def get(self, item_id: str) -> Intervention:
        with self._lock:
            if item_id not in self._items:
                raise HandoffError(f"no intervention {item_id}")
            return self._items[item_id].model_copy(deep=True)

    def all(self) -> list[Intervention]:
        with self._lock:
            return [i.model_copy(deep=True) for i in self._items.values()]

    def summaries(self) -> list[InterventionSummary]:
        return [InterventionSummary(id=i.id, kind=i.kind, status=i.status, step_id=i.step_id, claimed_by=i.claimed_by,
                                    resolved_by=i.resolved_by, human_actions=len(i.human_actions)) for i in self.all()]

    # --- internals -----------------------------------------------------------------------------

    def _require(self, item_id: str, states: set[Status], verb: str, kind: Kind | None = None) -> Intervention:
        item = self.get(item_id)
        if item.status not in states:
            raise HandoffError(f"cannot {verb} {item_id}: it is {item.status}")
        if kind is not None and item.kind is not kind:
            raise HandoffError(f"cannot {verb} {item_id}: it is a {item.kind} request")
        return item

    def _finish(self, item_id: str, status: Status, by: str, note: str) -> Intervention:
        return self._update(item_id, f"intervention_{status}", status=status, resolved_by=by, resolved_at=_now(),
                            note=note)

    def _update(self, item_id: str, event: str, **changes: Any) -> Intervention:
        with self._lock:
            item = self._items[item_id].model_copy(update=changes)
            self._items[item_id] = item
        self._persist(item, event)
        return item

    def _persist(self, item: Intervention, event: str | None) -> None:
        self.rec.write_text(f"interventions/{item.id}.json", item.model_dump_json(indent=2))
        if event:
            self.rec.event(event, intervention=item.id, kind=item.kind, status=item.status, step_id=item.step_id,
                           by=item.resolved_by or item.claimed_by)

    def _clean(self, text: str) -> str:
        return self.rec.scrub(mask_pii(text))

    def _lease_changed(self, state: LeaseState) -> None:
        self.rec.event("lease", owner=state.owner, epoch=state.epoch, holder=state.holder, reason=state.reason)


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")
