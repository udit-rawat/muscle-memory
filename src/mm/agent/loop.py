"""Discovery: an LLM-driven observe → decide → act loop against a live surface.

This is the only place a model makes decisions. Its output is a list of RecordedSteps, each
holding a verified, replayable Target (built by the surface, not by the model) and the value as a
template. The compiler turns that into a Capability; the transcript stays in the evidence folder.

Every action goes through GuardedSurface, so the agent cannot leave the allowlist or act while a
human holds the session. An irreversible click pauses for a human approval. When the agent is stuck
or asks for help, a human can take over the same live session and hand it back.
"""

from __future__ import annotations

import contextlib
import time
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Literal

from mm.agent.decision import for_screen
from mm.agent.prompts import SYSTEM, user_message
from mm.evidence.recorder import RunRecorder
from mm.handoff.intervention import HandoffController, Intervention, Kind, Status
from mm.handoff.lease import ControlLease
from mm.llm.router import LLMRouter, LLMUnavailable
from mm.policy.guard import GuardedSurface
from mm.policy.model import Policy
from mm.redact import mask_pii, mask_value
from mm.surface.base import ActionType, Observation, Surface, Target
from mm.values import SecretStore, TemplateError, parameterize, referenced, render

Status_ = Literal["success", "failed", "stuck", "needs_human", "max_steps", "timeout", "llm_error", "error", "aborted"]
_MAX_HANDOFFS = 3


@dataclass
class RecordedStep:
    action: ActionType
    intent: str
    target: Target | None
    value_template: str | None
    output: str | None
    frame_urls_before: dict[str, str]
    frame_urls_after: dict[str, str]
    interruption: bool = False  # dismissed an unexpected popup: becomes a detector, not a step
    options: list[str] | None = None  # select: the dropdown's options when recorded


@dataclass
class DiscoveryResult:
    run_id: str
    status: Status_
    summary: str
    goal: str
    entry_url: str
    inputs: dict[str, str]
    outputs: dict[str, str] = field(default_factory=dict)
    steps: list[RecordedStep] = field(default_factory=list)
    provider: str = ""
    model: str = ""
    human_took_control: bool = False  # an operator performed actions the compiler cannot turn into steps
    approvals: list[str] = field(default_factory=list)  # "<control> approved by <who>"


class _Discovery:
    def __init__(self, goal: str, entry_url: str, inputs: Mapping[str, str], required_outputs: list[str],
                 surface: GuardedSurface, router: LLMRouter, secrets: SecretStore, recorder: RunRecorder,
                 handoff: HandoffController | None, handoff_timeout_s: float) -> None:
        self.goal, self.inputs, self.required = goal, inputs, required_outputs
        self.surface, self.router, self.secrets, self.rec = surface, router, secrets, recorder
        self.handoff, self.handoff_timeout_s = handoff, handoff_timeout_s
        self.result = DiscoveryResult(recorder.run_id, "max_steps", "", goal, entry_url, dict(inputs))
        self.history: list[str] = []
        self.recent: list[tuple[str, str, str]] = []
        self.failures = 0
        self.entered: set[str] = set()  # inputs the agent has actually typed/selected
        self.handoffs = 0

    # --- the loop ------------------------------------------------------------------------------

    def run(self, max_steps: int, timeout_s: float) -> DiscoveryResult:
        r = self.result
        try:
            self.surface.navigate(r.entry_url)
        except Exception as exc:  # noqa: BLE001 — an unreachable or disallowed entry ends discovery with a status
            r.status, r.summary = "error", f"could not open {r.entry_url}: {exc}"
            return self._end()
        started = time.monotonic()
        try:
            for n in range(1, max_steps + 1):
                if time.monotonic() - started > timeout_s:
                    r.status, r.summary = "timeout", f"no result after {timeout_s:.0f}s"
                    break
                if not self._turn(n):
                    break
        except Exception as exc:  # noqa: BLE001 — a crashed browser or bug ends discovery with a status
            r.status, r.summary = "error", f"{type(exc).__name__}: {str(exc)[:300]}"
            self.rec.event("internal_error", error=r.summary)
        return self._end()

    def _turn(self, n: int) -> bool:
        """One observe → decide → act cycle. False when discovery should stop."""
        r = self.result
        obs = self.surface.observe()
        # Reading or acting underneath a modal would record a flow in a state replay never sees.
        overlay = "; ".join(self.surface.blocking_overlays())
        # One string, sanitised once, is both sent to the model and kept as evidence: the log cannot
        # show something other than what the model received.
        prompt = self.rec.scrub(mask_pii(self._prompt(obs, overlay)))
        self.rec.write_text(f"prompts/{n:02d}.txt", prompt)
        try:
            call = self.router.structured(
                for_screen({e.ref for e in obs.elements}, set(self.inputs), set(self.secrets.names),
                           set(r.outputs), overlay, {e.ref for e in obs.elements if e.credential}),
                [{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}],
            )
        except LLMUnavailable as exc:
            r.status, r.summary = "llm_error", str(exc)
            self.rec.event("llm_unavailable", step=n, errors=exc.errors)
            return False
        d = call.value
        r.provider, r.model = call.provider, call.model
        element = next((e for e in obs.elements if e.ref == d.ref), None)
        if element is not None and element.role == "cell":
            self.rec.taint(element.name)  # a cell's name is its content: mask it before it is logged
        described = f'{element.role} "{element.name}"' if element else ""
        self.rec.event("decision", step=n, provider=call.provider, model=call.model,
                       tokens={"in": call.input_tokens, "out": call.output_tokens}, thought=d.thought,
                       action=d.action, ref=d.ref, element=described, value=d.value, output_name=d.output_name,
                       interruption=d.interruption, summary=d.summary)

        if d.action == "done":
            missing = [o for o in self.required if o not in r.outputs]
            unused = [k for k in self.inputs if k not in self.entered]
            if not (missing or unused):
                r.status, r.summary = "success", d.summary or ""
                return False
            self.history.append(f"{n}. done REJECTED: missing outputs {missing}, inputs never entered {unused}")
            return self._no_progress(n)
        if d.action == "fail":
            r.status, r.summary = "failed", d.summary or ""
            return False
        if d.action == "request_human":
            return self._take_over(n, Kind.STUCK, f"the agent asked for help: {d.summary}")

        action = ActionType(d.action)
        template = parameterize(d.value, self.inputs) if d.value else None
        try:
            concrete = render(template, self.inputs, self.secrets) if template else None
        except TemplateError as exc:
            self.history.append(f"{n}. {d.action} {described} -> REJECTED: {exc}")
            return self._no_progress(n)

        before = self.surface.frame_urls()
        if self.surface.risk_of(action, ref=d.ref) == "irreversible":
            verdict = self._approval(n, described, obs)
            if verdict is None:
                return False  # aborted
            if not verdict:
                why = "refused: irreversible actions need an operator's approval and no operator is available" \
                    if self.handoff is None else "NOT APPROVED by the operator"
                self.history.append(f"{n}. {d.action} {described} -> {why}; do not retry it")
                return self._no_progress(n)
            with self.surface.irreversible_authorized():
                res = self.surface.act(action, d.ref, concrete)
        else:
            res = self.surface.act(action, d.ref, concrete)
        self.rec.taint(res.extracted)
        self.rec.event("act", step=n, ok=res.ok, error=res.error, detail=res.detail,
                       target=res.target.model_dump(mode="json") if res.target else None,
                       extracted=mask_value(res.extracted) if res.extracted else None, frame_urls=res.frame_urls)

        shown = f" {template!r}" if template else ""
        if not res.ok:
            self.history.append(f"{n}. {d.action} {described}{shown} -> FAILED: {res.detail}")
            return self._no_progress(n)
        self.failures = 0
        if action is ActionType.EXTRACT and d.output_name:
            r.outputs[d.output_name] = res.extracted or ""
        if template:
            self.entered |= referenced(template)[0]
        r.steps.append(RecordedStep(action, d.thought, res.target, template, d.output_name, before, res.frame_urls,
                                    d.interruption, res.options))
        stored = f" (stored as {d.output_name})" if action is ActionType.EXTRACT else ""
        self.history.append(f"{n}. {d.action} {described}{shown} -> ok{stored}")
        self.recent = (self.recent + [(d.action, described, template or "")])[-3:]
        if len(self.recent) == 3 and len(set(self.recent)) == 1:
            return self._stuck(n)
        return True

    # --- getting unstuck -------------------------------------------------------------------------

    def _no_progress(self, n: int) -> bool:
        self.failures += 1
        return self._stuck(n) if self.failures >= 3 else True

    def _stuck(self, n: int) -> bool:
        return self._take_over(n, Kind.STUCK, f"no progress after step {n}: {self.history[-1]}")

    def _take_over(self, n: int, kind: Kind, reason: str) -> bool:
        """Ask a human to take the live session; continue afterwards if they hand it back."""
        r = self.result
        if self.handoff is None or self.handoffs >= _MAX_HANDOFFS:
            r.status = "needs_human" if kind is Kind.STUCK and "asked for help" in reason else "stuck"
            r.summary = reason
            return False
        self.handoffs += 1
        item = self._open(kind, reason, n)
        final = self.handoff.wait(item.id, idle=self.surface.idle, timeout_s=self.handoff_timeout_s)
        if final.status is not Status.RESOLVED:
            r.status, r.summary = "aborted", f"handoff {final.status}: {final.note or reason}"
            return False
        # A human held the session: whatever they did (captured or not: keyboard, URL bar...) is not in the
        # recorded steps, so the result must not compile as if it were complete.
        r.human_took_control = True
        did = "; ".join(f"{a.kind} {a.name!r}" for a in final.human_actions) or "nothing"
        self.history.append(f"{n}. HUMAN OPERATOR took control and did: {did}. Control is back with you.")
        self.surface.resync()
        self.failures, self.recent = 0, []
        return True

    def _approval(self, n: int, described: str, obs: Observation) -> bool | None:
        """True approved, False rejected (or nobody to ask), None aborted."""
        if self.handoff is None:
            self.rec.event("irreversible_refused", step=n, element=described, reason="no operator available")
            return False
        item = self._open(Kind.APPROVAL_REQUIRED, f"the agent wants to click {described}, which is irreversible", n,
                          obs)  # the observation the pending action's ref belongs to: never re-observe here
        final = self.handoff.wait(item.id, idle=self.surface.idle, timeout_s=self.handoff_timeout_s)
        if final.status is Status.APPROVED:
            self.result.approvals.append(f"{described} approved by {final.resolved_by}")
            return True
        if final.status is Status.ABORTED:
            self.result.status, self.result.summary = "aborted", f"operator aborted at {described}"
            return None
        return False

    def _open(self, kind: Kind, reason: str, n: int, obs: Observation | None = None) -> Intervention:
        assert self.handoff is not None
        shot = self.rec.screenshot_path(f"intervention-{n:02d}")
        with contextlib.suppress(Exception):
            self.surface.screenshot(str(shot))
        frames = (obs or self.surface.observe()).frames
        excerpt = "\n".join(f"[{'/'.join(f.frame_path) or 'top'}] {f.text[:300]}" for f in frames)
        return self.handoff.open(kind, subject=self.goal, reason=reason, step_id=f"turn-{n}",
                                 screenshot=str(shot) if shot.exists() else None,
                                 page_excerpt=self.rec.scrub(mask_pii(excerpt))[:1200],
                                 suggested=["Approve or Reject"] if kind is Kind.APPROVAL_REQUIRED else
                                 ["Take control, do the next step(s) in the browser window, then Resume",
                                  "Or Abort"])

    # --- helpers -------------------------------------------------------------------------------

    def _prompt(self, obs: Observation, overlay: str) -> str:
        r = self.result
        prompt = user_message(self.goal, self.inputs, self.secrets.names, list(r.outputs), self.history, obs)
        notes = []
        if self.required:
            notes.append(f"OUTPUTS required: {', '.join(self.required)}")
        if unused := [k for k in self.inputs if k not in self.entered]:
            notes.append("INPUTS not yet entered: " + ", ".join("{{" + k + "}}" for k in unused))
        if self.required and not [o for o in self.required if o not in r.outputs] and not unused:
            notes.append("STATUS: every required output is extracted and every input entered. "
                         "If the screen confirms the goal is achieved, respond with done.")
        if overlay:
            notes.append(f"BLOCKING OVERLAY on screen: {mask_pii(overlay)}. Dismiss it first "
                         "(click, interruption=true).")
        if notes:
            prompt = prompt.replace("\n\nHISTORY", "\n" + "\n".join(notes) + "\n\nHISTORY", 1)
        return prompt

    def _end(self) -> DiscoveryResult:
        r = self.result
        shot = self.rec.screenshot_path("final")
        with contextlib.suppress(Exception):  # evidence is best-effort once the run has ended
            self.surface.screenshot(str(shot))
        self.rec.event("discovery_end", status=r.status, summary=r.summary, steps=len(r.steps),
                       outputs={k: mask_value(v) for k, v in r.outputs.items()}, approvals=r.approvals,
                       human_took_control=r.human_took_control)
        return r


def discover(
    *,
    goal: str,
    entry_url: str,
    inputs: Mapping[str, str],
    required_outputs: list[str],
    surface: Surface,
    router: LLMRouter,
    secrets: SecretStore,
    recorder: RunRecorder,
    policy: Policy,
    handoff: HandoffController | None = None,
    handoff_timeout_s: float = 600,
    max_steps: int = 25,
    timeout_s: float = 600,
) -> DiscoveryResult:
    recorder.event("discovery_start", goal=goal, entry_url=entry_url, inputs=dict(inputs),
                   required_outputs=required_outputs, secrets=secrets.names, escalation=handoff is not None)
    lease = handoff.lease if handoff else ControlLease()
    guarded = GuardedSurface(surface, policy, lease, on_event=recorder.event)
    try:
        return _Discovery(goal, entry_url, inputs, required_outputs, guarded, router, secrets, recorder, handoff,
                          handoff_timeout_s).run(max_steps, timeout_s)
    finally:
        lease.end("discovery finished")
