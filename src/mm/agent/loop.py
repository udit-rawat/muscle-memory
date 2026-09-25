"""Discovery: an LLM-driven observe → decide → act loop against a live surface.

This is the only place a model makes decisions. Its output is a list of RecordedSteps, each
holding a verified, replayable Target (built by the surface, not by the model) and the value as a
template. The compiler turns that into a Capability; the transcript stays in the evidence folder.
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
from mm.llm.router import LLMRouter, LLMUnavailable
from mm.redact import mask_value
from mm.surface.base import ActionType, Surface, Target
from mm.values import SecretStore, TemplateError, parameterize, referenced, render

Status = Literal["success", "failed", "stuck", "needs_human", "max_steps", "timeout", "llm_error", "error"]


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
    status: Status
    summary: str
    goal: str
    entry_url: str
    inputs: dict[str, str]
    outputs: dict[str, str] = field(default_factory=dict)
    steps: list[RecordedStep] = field(default_factory=list)
    provider: str = ""
    model: str = ""


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
    max_steps: int = 25,
    timeout_s: float = 600,
) -> DiscoveryResult:
    result = DiscoveryResult(recorder.run_id, "max_steps", "", goal, entry_url, dict(inputs))
    recorder.event("discovery_start", goal=goal, entry_url=entry_url, inputs=dict(inputs),
                   required_outputs=required_outputs, secrets=secrets.names)
    try:
        surface.navigate(entry_url)
    except Exception as exc:  # noqa: BLE001 — an unreachable app ends discovery with a status, not a traceback
        result.status, result.summary = "error", f"could not open {entry_url}: {exc}"
        recorder.event("discovery_end", status=result.status, summary=result.summary, steps=0)
        return result
    history: list[str] = []
    recent: list[tuple[str, str, str]] = []
    consecutive_failures = 0
    entered: set[str] = set()  # inputs the agent has actually typed/selected
    started = time.monotonic()

    for n in range(1, max_steps + 1):
        if time.monotonic() - started > timeout_s:
            result.status, result.summary = "timeout", f"no result after {timeout_s:.0f}s"
            break
        obs = surface.observe()
        # Reading or acting underneath a modal would record a flow in a state replay never sees.
        overlay = "; ".join(surface.blocking_overlays())
        prompt = user_message(goal, inputs, secrets.names, list(result.outputs), history, obs)
        notes = []
        if required_outputs:
            notes.append(f"OUTPUTS required: {', '.join(required_outputs)}")
        if unused := [k for k in inputs if k not in entered]:
            notes.append("INPUTS not yet entered: " + ", ".join("{{" + k + "}}" for k in unused))
        if required_outputs and not [o for o in required_outputs if o not in result.outputs] and not unused:
            notes.append("STATUS: every required output is extracted and every input entered. "
                         "If the screen confirms the goal is achieved, respond with done.")
        if overlay:
            notes.append(f"BLOCKING OVERLAY on screen: {overlay}. Dismiss it first (click, interruption=true).")
        if notes:
            prompt = prompt.replace("\n\nHISTORY", "\n" + "\n".join(notes) + "\n\nHISTORY", 1)
        recorder.write_text(f"prompts/{n:02d}.txt", prompt)

        try:
            call = router.structured(
                for_screen({e.ref for e in obs.elements}, set(inputs), set(secrets.names), set(result.outputs),
                           overlay),
                [{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}],
            )
        except LLMUnavailable as exc:
            result.status, result.summary = "llm_error", str(exc)
            recorder.event("llm_unavailable", step=n, errors=exc.errors)
            break
        d = call.value
        result.provider, result.model = call.provider, call.model
        element = next((e for e in obs.elements if e.ref == d.ref), None)
        if element is not None and element.role == "cell":
            recorder.taint(element.name)  # a cell's name is its content: mask it before it is logged
        described = f'{element.role} "{element.name}"' if element else ""
        recorder.event("decision", step=n, provider=call.provider, model=call.model,
                       tokens={"in": call.input_tokens, "out": call.output_tokens},
                       thought=d.thought, action=d.action, ref=d.ref, element=described, value=d.value,
                       output_name=d.output_name, interruption=d.interruption, summary=d.summary)

        if d.action == "done":
            missing = [o for o in required_outputs if o not in result.outputs]
            unused = [k for k in inputs if k not in entered]
            if missing or unused:
                history.append(f"{n}. done REJECTED: missing outputs {missing}, inputs never entered {unused}")
                consecutive_failures += 1  # a rejected "done" is not progress
                if consecutive_failures >= 3:
                    result.status, result.summary = "stuck", f"no progress after step {n}: {history[-1]}"
                    break
                continue
            result.status, result.summary = "success", d.summary or ""
            break
        if d.action in ("fail", "request_human"):
            result.status = "failed" if d.action == "fail" else "needs_human"
            result.summary = d.summary or ""
            break

        action = ActionType(d.action)
        template = parameterize(d.value, inputs) if d.value else None
        try:
            concrete = render(template, inputs, secrets) if template else None
        except TemplateError as exc:
            history.append(f"{n}. {d.action} {described} -> REJECTED: {exc}")
            consecutive_failures += 1
            continue
        before = _frame_urls(surface)
        res = surface.act(action, d.ref, concrete)
        recorder.taint(res.extracted)
        recorder.event("act", step=n, ok=res.ok, detail=res.detail,
                       target=res.target.model_dump(mode="json") if res.target else None,
                       extracted=mask_value(res.extracted) if res.extracted else None,
                       frame_urls=res.frame_urls)

        shown_value = f" {template!r}" if template else ""
        if res.ok:
            consecutive_failures = 0
            if action is ActionType.EXTRACT and d.output_name:
                result.outputs[d.output_name] = res.extracted or ""
            if template:
                entered |= referenced(template)[0]
            result.steps.append(RecordedStep(action, d.thought, res.target, template, d.output_name,
                                             before, res.frame_urls, d.interruption, res.options))
            stored = f" (stored as {d.output_name})" if action is ActionType.EXTRACT else ""
            history.append(f"{n}. {d.action} {described}{shown_value} -> ok{stored}")
        else:
            consecutive_failures += 1
            history.append(f"{n}. {d.action} {described}{shown_value} -> FAILED: {res.detail}")

        recent = (recent + [(d.action, described, template or "")])[-3:]
        repeating = len(recent) == 3 and len(set(recent)) == 1
        if consecutive_failures >= 3 or repeating:
            result.status, result.summary = "stuck", f"no progress after step {n}: {history[-1]}"
            break

    shot = recorder.screenshot_path("final")
    with contextlib.suppress(Exception):  # evidence is best-effort once the run has ended
        surface.screenshot(str(shot))
    recorder.event("discovery_end", status=result.status, summary=result.summary, steps=len(result.steps),
                   outputs=_masked(result.outputs), screenshot=str(shot))
    return result


def _frame_urls(surface: Surface) -> dict[str, str]:
    urls = getattr(surface, "frame_urls", None)
    return dict(urls()) if callable(urls) else {}


def _masked(outputs: Mapping[str, str]) -> dict[str, str]:
    return {k: mask_value(v) for k, v in outputs.items()}
