"""Deterministic replay: execute a Capability with caller-supplied inputs. No model in the loop.

Same artifact + same inputs + same app state -> same steps, same outputs. For each step:

  1. perform the action on the step's target (every strategy must match exactly one element);
  2. wait until the step's checkpoints hold, checking the capability's detectors on every poll,
     so an exceptional state is recognised as what it is the moment it appears:
       business_outcome -> stop, return the code to the caller (a legitimate answer, not an error)
       recoverable      -> run the detector's bounded, deterministic handler, then continue / redo / restart
       hard_failure     -> stop with a debuggable failure
  3. before moving on, make sure nothing unrecognised (e.g. an unknown modal) is blocking the UI.

If a step fails outright (target missing, click intercepted), detectors get the first say too:
"no Select link" after a search is MEMBER_NOT_FOUND, not TARGET_NOT_FOUND.

Every action goes through GuardedSurface (allowlist, action types, control lease). An irreversible step
runs only under a valid approval that names it, and only inside an authorised window (the browser also
blocks commit requests outside that window). If a human operator is available, an unrecoverable
state becomes an intervention request instead of a failure: the operator takes control of this same
live session, fixes it, hands control back, and replay re-verifies the screen before carrying on.

A non-safe step (mutating/irreversible) is performed at most once. Once its action has been
dispatched, whether or not it reported success, no recovery may redo it or restart the flow past it:
that is UNSAFE_TO_REPEAT, because repeating a commit is worse than stopping.
"""

from __future__ import annotations

import re
import time
from collections import Counter
from collections.abc import Callable, Mapping
from decimal import Decimal, InvalidOperation
from typing import Any, Literal
from urllib.parse import urljoin

from mm.artifact.approval import Approval
from mm.artifact.schema import Capability, Detector, OutputSpec, Step
from mm.evidence.recorder import RunRecorder
from mm.handoff.intervention import HandoffController, Kind, Status
from mm.handoff.lease import ControlLease
from mm.policy.guard import GuardedSurface
from mm.policy.model import Policy, stricter
from mm.redact import mask_value
from mm.replay.result import (
    FailureKind,
    Recovery,
    ReplayBusinessOutcome,
    ReplayFailure,
    ReplayResult,
    ReplaySuccess,
    StepTrace,
)
from mm.surface.base import ActionType, Surface
from mm.values import SecretStore, render

_POLL_S = 0.15
_MAX_STEP_ATTEMPTS = 3
_MAX_ESCALATIONS = 3
# Failures a human operator can plausibly fix in the live session. Business outcomes never escalate
# (they are answers), nor do input errors, policy refusals or an unreachable app.
_ESCALATABLE = {FailureKind.UNEXPECTED_STATE, FailureKind.TARGET_NOT_FOUND, FailureKind.CHECKPOINT_FAILED,
                FailureKind.ACTION_FAILED, FailureKind.RECOVERY_EXHAUSTED}
_RECOVERY_ACTION: dict[str, Literal["handled", "retried_step", "restarted"]] = {
    "continue": "handled", "retry_step": "retried_step", "restart": "restarted"}


class OutputParseError(ValueError):
    pass


class _Stop(Exception):  # noqa: N818 — control flow, carries the final result
    def __init__(self, result: ReplayResult) -> None:
        self.result = result


class _Restart(Exception):  # noqa: N818
    pass


def validate_inputs(cap: Capability, params: Mapping[str, str]) -> list[str]:
    problems: list[str] = []
    if missing := sorted(cap.inputs.keys() - params.keys()):
        problems.append(f"missing inputs: {missing}")
    if unknown := sorted(params.keys() - cap.inputs.keys()):
        problems.append(f"unknown inputs: {unknown}")
    for name, spec in cap.inputs.items():
        value = params.get(name)
        if value is None:
            continue
        if spec.pattern and not re.fullmatch(spec.pattern, value):
            problems.append(f"input {name!r} does not match {spec.pattern}")
        if spec.enum and value not in spec.enum:
            problems.append(f"input {name!r} must be one of {spec.enum}")
        if spec.type == "decimal":
            try:
                Decimal(value)
            except InvalidOperation:
                problems.append(f"input {name!r} is not a decimal")
    return problems


def replay(
    cap: Capability,
    params: Mapping[str, str],
    *,
    base_url: str,
    surface_factory: Callable[[RunRecorder], Surface],
    secrets: SecretStore,
    recorder: RunRecorder,
    policy: Policy,
    approval: Approval | None = None,
    approval_problem: str = "capability has no approval",
    handoff: HandoffController | None = None,
    escalation_timeout_s: float = 600,
) -> ReplayResult:
    """`approval`: a valid sign-off for this exact artifact (see artifact/approval.py), required for any
    irreversible step. `handoff`: if given, unrecoverable states are escalated to a human operator."""
    base: dict[str, Any] = {"capability_id": cap.id, "capability_version": cap.version, "run_id": recorder.run_id,
                            "approved_by": approval.approved_by if approval else None}
    recorder.event("replay_start", capability=cap.id, version=cap.version, status=cap.status, base_url=base_url,
                   inputs={k: (mask_value(v) if cap.inputs[k].sensitive else v)
                           for k, v in params.items() if k in cap.inputs})

    result: ReplayResult
    if problems := validate_inputs(cap, params):
        result = ReplayFailure(**base, kind=FailureKind.INPUT_INVALID, step_id=None, message="; ".join(problems))
        recorder.event("replay_end", **_loggable(result))
        return result
    if cap.status == "deprecated":
        result = ReplayFailure(**base, kind=FailureKind.POLICY_BLOCKED, step_id=None,
                               message=f"{cap.id} {cap.version} is deprecated")
        recorder.event("replay_end", **_loggable(result))
        return result

    lease = handoff.lease if handoff else ControlLease()
    surface = GuardedSurface(surface_factory(recorder), policy, lease)
    try:
        result = _Replay(cap, params, base_url, surface, secrets, recorder, base, approval, approval_problem,
                         handoff, escalation_timeout_s).run()
    finally:
        lease.end("replay finished")
        surface.close()
    if isinstance(result, ReplayFailure) and recorder.trace_path.exists():  # tracing is opt-in
        result.trace = str(recorder.trace_path)
    recorder.event("replay_end", **_loggable(result))
    return result


class _Replay:
    def __init__(self, cap: Capability, params: Mapping[str, str], base_url: str, surface: GuardedSurface,
                 secrets: SecretStore, recorder: RunRecorder, base: dict[str, Any], approval: Approval | None,
                 approval_problem: str, handoff: HandoffController | None, escalation_timeout_s: float) -> None:
        self.cap, self.params, self.base_url = cap, params, base_url
        self.surface, self.secrets, self.rec, self.base = surface, secrets, recorder, base
        self.approval, self.approval_problem = approval, approval_problem
        self.handoff, self.escalation_timeout_s = handoff, escalation_timeout_s
        self.escalations = 0
        self.dispatched_non_safe: set[str] = set()
        self.outputs: dict[str, str] = {}
        self.traces: list[StepTrace] = []
        self.drift: list[str] = []
        self.recoveries: list[Recovery] = []
        self.fired: Counter[str] = Counter()
        self.committed = False  # a non-safe step has run: restarting could repeat a side effect

    # --- top level -----------------------------------------------------------------------------

    def run(self) -> ReplayResult:
        step_id: str | None = None
        try:
            i = 0
            while i < len(self.cap.steps):
                step = self.cap.steps[i]
                step_id = step.id
                try:
                    self._step(step)
                    i += 1
                except _Restart:
                    self.rec.event("restart", from_step=step_id)
                    self.outputs.clear()
                    i = 0
                except _Stop as stop:
                    if not self._can_escalate(stop.result):
                        raise
                    if self._escalate(step, stop.result) == "next":
                        i += 1
            return self._finish()
        except _Stop as stop:
            return stop.result
        except Exception as exc:  # noqa: BLE001 — the caller gets a result, never a traceback
            self.rec.event("internal_error", step_id=step_id, error=f"{type(exc).__name__}: {exc}")
            return self._failure(FailureKind.INTERNAL_ERROR, step_id, f"{type(exc).__name__}: {str(exc)[:300]}",
                                 committed=self.committed)

    def _finish(self) -> ReplayResult:
        if missing := sorted(self.cap.outputs.keys() - self.outputs.keys()):
            return self._failure(FailureKind.OUTPUT_MISSING, None, f"outputs never extracted: {missing}")
        for cp in self.cap.success:
            ok, observed = self.surface.check(cp, 5_000)
            if not ok:
                return self._failure(FailureKind.SUCCESS_CHECK_FAILED, None, "success condition not met",
                                     expected=f"{cp.kind} {cp.pattern}", observed=observed)
        return ReplaySuccess(**self.base, steps=self.traces, recoveries=self.recoveries,
                             interventions=self._interventions(), outputs=self.outputs, drift=self.drift)

    # --- one step ------------------------------------------------------------------------------

    def _step(self, step: Step) -> None:
        for attempt in range(1, _MAX_STEP_ATTEMPTS + 1):
            t0 = time.monotonic()
            value = render(step.value, self.params, self.secrets) if step.value else None
            if step.action is ActionType.NAVIGATE and value:
                value = urljoin(self.base_url.rstrip("/") + "/", value.lstrip("/"))
            # The artifact's risk, or the current policy's reading of the control, whichever is stricter.
            risk = stricter(step.risk, self.surface.risk_of(step.action, target=step.target))
            if risk == "irreversible":
                self._require_approval(step)
                with self.surface.irreversible_authorized():
                    res = self.surface.perform(step.action, step.target, value, step.timeout_ms)
            else:
                res = self.surface.perform(step.action, step.target, value, step.timeout_ms)
            by = (step.target.strategies[res.strategy_index].by
                  if step.target and res.strategy_index is not None else None)
            self.traces.append(StepTrace(step_id=step.id, ok=res.ok, strategy=by, strategy_index=res.strategy_index,
                                         duration_ms=int((time.monotonic() - t0) * 1000)))
            self.rec.event("step", step_id=step.id, attempt=attempt, action=step.action, risk=step.risk, ok=res.ok,
                           strategy=by, strategy_index=res.strategy_index, detail=res.detail, attempts=res.attempts,
                           extracted=mask_value(res.extracted) if res.extracted else None)

            dispatched = res.ok or res.error == "action_failed" or res.dispatched is True
            if dispatched and risk != "safe":
                self.committed = True  # it may have taken effect even if it reported an error
                self.dispatched_non_safe.add(step.id)

            if not res.ok:
                desc = step.target.description if step.target else step.value
                if res.error == "policy_blocked":
                    raise _Stop(self._failure(FailureKind.POLICY_BLOCKED, step.id, f"{step.action} on {desc} refused",
                                              expected="an action inside the safety policy", observed=res.detail))
                if res.error == "control_not_held":
                    raise _Stop(self._failure(FailureKind.CONTROL_LOST, step.id, "automation does not hold the session",
                                              observed=res.detail))
                if res.error == "navigation_failed":
                    raise _Stop(self._failure(FailureKind.APP_UNREACHABLE, step.id, f"could not load {value}",
                                              expected="the application to respond", observed=res.detail))
                # The page may be in a known exceptional state; detectors get the first say.
                if self._react(step) == "none":
                    self._fail_if_unrecognised_overlay(step)
                    if res.error == "not_found":
                        raise _Stop(self._failure(FailureKind.TARGET_NOT_FOUND, step.id, f"could not find {desc}",
                                                  expected=_strategies(step), observed="; ".join(res.attempts)))
                    raise _Stop(self._failure(FailureKind.ACTION_FAILED, step.id, f"{step.action} on {desc} failed",
                                              expected=step.intent, observed=res.detail))
                self._refuse_repeat(step, dispatched, "redo the step after a recovery")
                continue  # a recoverable condition was handled and nothing was dispatched; redo the action

            if res.strategy_index:
                self.drift.append(f"{step.id}: matched by fallback strategy #{res.strategy_index} ({by})")
            if step.action is ActionType.EXTRACT and step.output:
                self.rec.taint(res.extracted)  # a value read from the app never appears unmasked in the log
                spec = self.cap.outputs[step.output]
                try:
                    self.outputs[step.output] = _parse(res.extracted or "", spec)
                    self.rec.taint(self.outputs[step.output])
                except OutputParseError as exc:
                    raise _Stop(self._failure(FailureKind.OUTPUT_UNPARSEABLE, step.id, str(exc),
                                              expected=spec.parse, observed=mask_value(res.extracted))) from exc

            if self._await(step) == "retry":
                self._refuse_repeat(step, True, "retry the step")
                continue
            return
        raise _Stop(self._failure(FailureKind.RECOVERY_EXHAUSTED, step.id,
                                  f"step still not complete after {_MAX_STEP_ATTEMPTS} attempts"))

    def _await(self, step: Step) -> Literal["ok", "retry"]:
        """Poll until the step's checkpoints hold, letting detectors react to whatever shows up meanwhile."""
        deadline = time.monotonic() + step.timeout_ms / 1000
        while True:
            reaction = self._react(step)
            if reaction == "retry":
                return "retry"
            if reaction == "handled":
                deadline = time.monotonic() + step.timeout_ms / 1000  # the handler may have navigated
                continue
            pending = [cp for cp in step.expect if not self.surface.check(cp, 0)[0]]
            if not pending:
                self._fail_if_unrecognised_overlay(step)
                return "ok"
            if time.monotonic() >= deadline:
                cp = pending[0]
                _, observed = self.surface.check(cp, 0)
                self._fail_if_unrecognised_overlay(step)
                where = "any frame" if cp.any_frame else f"frame {'/'.join(cp.frame_path) or 'top'}"
                what = cp.pattern or (cp.target.description if cp.target else "")
                raise _Stop(self._failure(
                    FailureKind.CHECKPOINT_FAILED, step.id, f"after {step.id}, the UI never reached the expected state",
                    expected=f"{cp.kind} {what} in {where}", observed=observed))
            time.sleep(_POLL_S)

    # --- detectors -----------------------------------------------------------------------------

    def _react(self, step: Step) -> Literal["none", "handled", "retry"]:
        """Check detectors once. Stops the run for outcomes/failures; returns what a recovery asks for."""
        hit = self._first_detector(step)
        if hit is None:
            return "none"
        det, observed = hit
        self.rec.event("detector_fired", detector=det.id, detector_class=det.class_, code=det.code,
                       step_id=step.id, observed=observed)
        if det.class_ == "business_outcome":
            shot = self._screenshot(f"outcome-{step.id}")
            raise _Stop(ReplayBusinessOutcome(**self.base, steps=self.traces, recoveries=self.recoveries,
                                              interventions=self._interventions(),
                                              code=det.code or "", message=observed or det.description,
                                              detector_id=det.id, step_id=step.id, screenshot=shot))
        if det.class_ == "hard_failure":
            raise _Stop(self._failure(FailureKind.APP_ERROR, step.id, det.description or det.id, code=det.code,
                                      expected="no error state", observed=observed))
        return self._recover(det, step)

    def _recover(self, det: Detector, step: Step) -> Literal["handled", "retry"]:
        self.fired[det.id] += 1
        if self.fired[det.id] > det.max_times:
            raise _Stop(self._failure(FailureKind.RECOVERY_EXHAUSTED, step.id,
                                      f"{det.id} recurred more than {det.max_times} time(s)",
                                      expected=f"at most {det.max_times} recoveries", observed=det.description))
        if det.then == "restart" and self.committed:
            raise _Stop(self._failure(FailureKind.UNSAFE_TO_REPEAT, step.id,
                                      f"{det.id} requires restarting the flow, but a non-safe step may already "
                                      "have taken effect", expected="restart only before any state-changing step",
                                      observed=det.description, committed=True))
        for h in det.handle:
            value = render(h.value, self.params, self.secrets) if h.value else None
            res = self.surface.perform(h.action, h.target, value, 5_000)
            if not res.ok:
                raise _Stop(self._failure(FailureKind.RECOVERY_EXHAUSTED, step.id,
                                          f"handler for {det.id} failed: {res.detail}",
                                          expected=h.target.description if h.target else h.value,
                                          observed="; ".join(res.attempts) or res.detail))
        action = _RECOVERY_ACTION[det.then]
        self.recoveries.append(Recovery(detector_id=det.id, step_id=step.id, action=action, detail=det.description))
        self.rec.event("recovery", detector=det.id, step_id=step.id, action=action)
        if det.then == "restart":
            raise _Restart()
        return "retry" if det.then == "retry_step" else "handled"

    def _first_detector(self, step: Step) -> tuple[Detector, str] | None:
        for det in self.cap.detectors:
            if det.after_steps is not None and step.id not in det.after_steps:
                continue
            observed = ""
            for cond in det.when:
                held, seen = self.surface.check(cond, 0)
                if not held:
                    break
                if cond.kind in ("text_visible", "text_matches") and not observed:
                    observed = seen
            else:
                return det, observed
        return None

    # --- approval and escalation ---------------------------------------------------------------

    def _require_approval(self, step: Step) -> None:
        if self.approval is not None and step.id in self.approval.irreversible_steps:
            self.rec.event("irreversible_authorized", step_id=step.id, approved_by=self.approval.approved_by)
            return
        why = self.approval_problem if self.approval is None else f"the approval does not cover {step.id}"
        raise _Stop(self._failure(FailureKind.POLICY_BLOCKED, step.id,
                                  f"{step.id} is irreversible and needs an approval: {why}",
                                  expected=f"a valid approval of {self.cap.id} {self.cap.version} naming {step.id}",
                                  observed=why, committed=self.committed))

    def _can_escalate(self, result: ReplayResult) -> bool:
        return (self.handoff is not None and isinstance(result, ReplayFailure) and result.kind in _ESCALATABLE
                and self.escalations < _MAX_ESCALATIONS)

    def _escalate(self, step: Step, failure: ReplayResult) -> Literal["next", "retry"]:
        """Hand the live session to a human, wait, then re-verify before automation continues."""
        assert self.handoff is not None and isinstance(failure, ReplayFailure)
        self.escalations += 1
        item = self.handoff.open(
            Kind.UNRECOVERABLE_STATE, subject=f"{self.cap.id} {self.cap.version}", reason=failure.message,
            step_id=step.id, expected=failure.expected, observed=failure.observed, screenshot=failure.screenshot,
            suggested=[f"Bring the application to the state after: {step.intent}", "Then hand control back (Resume)",
                       "Or Abort if the run should stop"])
        final = self.handoff.wait(item.id, idle=self.surface.idle, timeout_s=self.escalation_timeout_s)
        if final.status is Status.ABORTED:
            raise _Stop(self._failure(FailureKind.ESCALATION_ABORTED, step.id, f"operator aborted: {final.note}",
                                      observed=failure.message))
        if final.status is not Status.RESOLVED:
            raise _Stop(self._failure(FailureKind.ESCALATION_TIMEOUT, step.id, "no operator took over in time",
                                      observed=failure.message))
        # Control is back. Never assume what the human did: look at the screen again first.
        if overlays := self.surface.blocking_overlays():
            raise _Stop(self._failure(FailureKind.UNEXPECTED_STATE, step.id, "still blocked after the handoff",
                                      observed="; ".join(overlays)))
        done = bool(step.expect) and all(self.surface.check(cp, 2_000)[0] for cp in step.expect)
        if not done and step.id in self.dispatched_non_safe:
            raise _Stop(self._failure(FailureKind.UNSAFE_TO_REPEAT, step.id,
                                      f"after the handoff {step.id} is not complete, and it may already have "
                                      "taken effect: not repeating it", committed=True))
        self.surface.resync()  # automation takes the session back only after re-verifying it
        self.rec.event("handoff_resumed", intervention=item.id, step_id=step.id,
                       continue_from="next step" if done else "same step")
        return "next" if done else "retry"

    def _interventions(self) -> list[Any]:
        return list(self.handoff.summaries()) if self.handoff else []

    def _refuse_repeat(self, step: Step, dispatched: bool, what: str) -> None:
        if dispatched and step.risk != "safe":
            raise _Stop(self._failure(
                FailureKind.UNSAFE_TO_REPEAT, step.id,
                f"recovery wants to {what}, but {step.id} is {step.risk} and was already dispatched",
                expected=f"{step.id} performed at most once", observed="a recoverable condition after dispatch",
                committed=True))

    def _fail_if_unrecognised_overlay(self, step: Step) -> None:
        """Nothing we don't understand may be covering the UI when we move on."""
        overlays = self.surface.blocking_overlays()
        if overlays:
            raise _Stop(self._failure(FailureKind.UNEXPECTED_STATE, step.id,
                                      "an unrecognised overlay is blocking the application",
                                      expected="no blocking overlay (or one a detector recognises)",
                                      observed="; ".join(overlays)))

    # --- helpers -------------------------------------------------------------------------------

    def _failure(self, kind: FailureKind, step_id: str | None, message: str, *, code: str | None = None,
                 expected: str | None = None, observed: str | None = None,
                 committed: bool | None = None) -> ReplayFailure:
        shot = self._screenshot(f"failure-{step_id or 'run'}")
        return ReplayFailure(**self.base, steps=self.traces, recoveries=self.recoveries,
                             interventions=self._interventions(), kind=kind, code=code,
                             step_id=step_id, message=message, expected=expected, observed=observed,
                             screenshot=shot, may_have_committed=self.committed if committed is None else committed)

    def _screenshot(self, name: str) -> str | None:
        path = self.rec.screenshot_path(name)
        try:
            self.surface.screenshot(str(path))
        except Exception:  # noqa: BLE001 — a dead browser must not turn a failure report into a crash
            return None
        return str(path)


def _parse(raw: str, spec: OutputSpec) -> str:
    text = raw.strip()
    if spec.parse == "currency":
        cleaned = text.replace("$", "").replace(",", "").strip()
        negative = cleaned.startswith("(") and cleaned.endswith(")")
        try:
            amount = Decimal(cleaned.strip("()"))
        except InvalidOperation as exc:
            raise OutputParseError(f"not a currency amount: {mask_value(text)!r}") from exc
        return str(-amount if negative else amount)
    return text


def _strategies(step: Step) -> str:
    if step.target is None:
        return ""
    return " | ".join(f"{s.by}: " + s.model_dump_json(exclude={"by"}) for s in step.target.strategies)


def _loggable(result: ReplayResult) -> dict[str, object]:
    data = result.model_dump(mode="json")
    if isinstance(result, ReplaySuccess):
        data["outputs"] = {k: mask_value(v) for k, v in result.outputs.items()}
    return data
