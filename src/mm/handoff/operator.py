"""A simulated operator, for demos and tests where no human is at the keyboard.

It behaves like a person at the console: it calls the console's real HTTP API to take control, approve,
reject, resume or abort, and it clicks in the *same live browser session* with real DOM events, so the
capture script records its actions exactly as it would a human's. Nothing else is simulated: the
lease, the console, the capture and the executor's re-verification are the production code paths.
It is always labelled as simulated in the evidence (`by: simulated-operator`).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import httpx
from playwright.sync_api import Page

from mm.handoff.intervention import Intervention, Kind, Status

WHO = "simulated-operator"


@dataclass
class SimulatedOperator:
    console_url: str
    page: Page
    clicks: list[str] = field(default_factory=list)  # button/link names to click once in control
    decision: str = "reject"  # for approval requests: approve | reject | abort. Never approves unless told to.
    token: str = ""  # the console's per-run token
    abort: bool = False  # for takeover requests: abort instead of fixing
    _done: set[str] = field(default_factory=set)

    def tick(self, item: Intervention) -> None:
        if item.id in self._done:
            return
        if item.kind is Kind.APPROVAL_REQUIRED:
            self._call(item.id, self.decision, note=f"{self.decision}d after reviewing the screen")
        elif self.abort:
            self._call(item.id, "abort", note="stopping the run")
        elif item.status is Status.OPEN:
            self._call(item.id, "claim")
            return  # the next tick acts, once the lease is ours
        elif item.status is Status.CLAIMED:
            for name in self.clicks:
                self._click(name)
            self._call(item.id, "resume", note=f"clicked {', '.join(self.clicks) or 'nothing'}")
        self._done.add(item.id)

    def _call(self, item_id: str, action: str, note: str = "") -> None:
        r = httpx.post(f"{self.console_url}/api/interventions/{item_id}/{action}", json={"by": WHO, "note": note},
                       headers={"X-Operator-Token": self.token}, timeout=10)
        r.raise_for_status()

    def _click(self, name: str) -> None:
        for frame in self.page.frames:
            if frame.is_detached():
                continue
            loc = frame.get_by_role("button", name=name, exact=True)
            if loc.count() == 0:
                loc = frame.get_by_role("link", name=name, exact=True)
            if loc.count() == 1:
                loc.click()
                self.page.wait_for_timeout(500)
                return
        raise RuntimeError(f"simulated operator could not find {name!r} on the page")
