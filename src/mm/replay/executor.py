"""Deterministic replay: execute a Capability with caller-supplied inputs. No model in the loop.

Same artifact + same inputs + same app state -> same steps, same outputs. Every step's target
must resolve to exactly one element, every declared checkpoint must hold, and every declared
output must be extracted and parsed, otherwise the run stops with a structured failure.
"""

from __future__ import annotations

import re
import time
from collections.abc import Callable, Mapping
from decimal import Decimal, InvalidOperation
from typing import Any
from urllib.parse import urljoin

from mm.artifact.schema import Capability, OutputSpec
from mm.evidence.recorder import RunRecorder
from mm.redact import mask_value
from mm.replay.result import FailureKind, ReplayFailure, ReplayResult, ReplaySuccess, StepTrace
from mm.surface.base import ActionType, Surface
from mm.values import SecretStore, render


class OutputParseError(ValueError):
    pass


def validate_inputs(cap: Capability, params: Mapping[str, str]) -> list[str]:
    problems: list[str] = []
    if missing := sorted(cap.inputs.keys() - params.keys()):
        problems.append(f"missing inputs: {missing}")
    if unknown := sorted(params.keys() - cap.inputs.keys()):
        problems.append(f"unknown inputs: {unknown}")
    for name, spec in cap.inputs.items():
        value = params.get(name)
        if value is not None and spec.pattern and not re.fullmatch(spec.pattern, value):
            problems.append(f"input {name!r} does not match {spec.pattern}")
    return problems


def replay(
    cap: Capability,
    params: Mapping[str, str],
    *,
    base_url: str,
    surface_factory: Callable[[RunRecorder], Surface],
    secrets: SecretStore,
    recorder: RunRecorder,
) -> ReplayResult:
    base: dict[str, Any] = {"capability_id": cap.id, "capability_version": cap.version, "run_id": recorder.run_id}
    recorder.event("replay_start", capability=cap.id, version=cap.version, status=cap.status, base_url=base_url,
                   inputs={k: (mask_value(v) if cap.inputs[k].sensitive else v)
                           for k, v in params.items() if k in cap.inputs})

    if problems := validate_inputs(cap, params):
        result: ReplayResult = ReplayFailure(**base, kind=FailureKind.INPUT_INVALID, step_id=None,
                                             message="; ".join(problems))
        recorder.event("replay_end", **_loggable(result))
        return result

    surface = surface_factory(recorder)
    try:
        result = _run(cap, params, base_url, surface, secrets, recorder, base)
    finally:
        surface.close()
    if isinstance(result, ReplayFailure):
        result.trace = str(recorder.trace_path)
    recorder.event("replay_end", **_loggable(result))
    return result


def _run(
    cap: Capability, params: Mapping[str, str], base_url: str, surface: Surface, secrets: SecretStore,
    recorder: RunRecorder, base: dict[str, Any],
) -> ReplayResult:
    outputs: dict[str, str] = {}
    traces: list[StepTrace] = []
    drift: list[str] = []

    def fail(kind: FailureKind, step_id: str | None, message: str,
             expected: str | None = None, observed: str | None = None) -> ReplayFailure:
        shot = recorder.screenshot_path(f"failure-{step_id or 'run'}")
        surface.screenshot(str(shot))
        return ReplayFailure(**base, steps=traces, kind=kind, step_id=step_id, message=message,
                             expected=expected, observed=observed, screenshot=str(shot))

    for step in cap.steps:
        t0 = time.monotonic()
        value = render(step.value, params, secrets) if step.value else None
        if step.action is ActionType.NAVIGATE and value:
            value = urljoin(base_url.rstrip("/") + "/", value.lstrip("/"))
        res = surface.perform(step.action, step.target, value, step.timeout_ms)
        by = step.target.strategies[res.strategy_index].by if step.target and res.strategy_index is not None else None
        traces.append(StepTrace(step_id=step.id, ok=res.ok, strategy=by, strategy_index=res.strategy_index,
                                duration_ms=int((time.monotonic() - t0) * 1000)))
        recorder.event("step", step_id=step.id, action=step.action, ok=res.ok, strategy=by,
                       strategy_index=res.strategy_index, detail=res.detail, attempts=res.attempts,
                       extracted=mask_value(res.extracted) if res.extracted else None)

        if not res.ok:
            desc = step.target.description if step.target else step.value
            if res.detail == "target not found":
                return fail(FailureKind.TARGET_NOT_FOUND, step.id, f"could not find {desc}",
                            expected=_strategies(step), observed="; ".join(res.attempts))
            return fail(FailureKind.ACTION_FAILED, step.id, f"{step.action} on {desc} failed",
                        expected=step.intent, observed=res.detail)
        if res.strategy_index:
            drift.append(f"{step.id}: matched by fallback strategy #{res.strategy_index} ({by})")

        if step.action is ActionType.EXTRACT and step.output:
            try:
                outputs[step.output] = _parse(res.extracted or "", cap.outputs[step.output])
            except OutputParseError as exc:
                return fail(FailureKind.OUTPUT_UNPARSEABLE, step.id, str(exc),
                            expected=cap.outputs[step.output].parse, observed=mask_value(res.extracted))

        for cp in step.expect:
            ok, observed = surface.check(cp, step.timeout_ms)
            if not ok:
                return fail(FailureKind.CHECKPOINT_FAILED, step.id, f"after {step.id}, the UI is not where expected",
                            expected=f"{cp.kind} {cp.pattern} in frame {'/'.join(cp.frame_path) or 'top'}",
                            observed=observed)

    if missing := sorted(cap.outputs.keys() - outputs.keys()):
        return fail(FailureKind.OUTPUT_MISSING, None, f"outputs never extracted: {missing}")
    for cp in cap.success:
        ok, observed = surface.check(cp, 5_000)
        if not ok:
            return fail(FailureKind.SUCCESS_CHECK_FAILED, None, "success condition not met",
                        expected=f"{cp.kind} {cp.pattern}", observed=observed)
    return ReplaySuccess(**base, steps=traces, outputs=outputs, drift=drift)


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


def _strategies(step: object) -> str:
    target = getattr(step, "target", None)
    if target is None:
        return ""
    return " | ".join(s.model_dump_json(exclude={"by"}) + f" by {s.by}" for s in target.strategies)


def _loggable(result: ReplayResult) -> dict[str, object]:
    data = result.model_dump(mode="json")
    if isinstance(result, ReplaySuccess):
        data["outputs"] = {k: mask_value(v) for k, v in result.outputs.items()}
    return data
