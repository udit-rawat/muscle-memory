"""Compile a successful discovery run into a Capability.

What the compiler decides (not the model):
- inputs/outputs and their types, inferred from the concrete values seen during discovery;
- values become templates, so no concrete input value is stored in the artifact;
- per-step checkpoints, from which frame URL changed when the step ran;
- the success condition, from where the flow ended.
"""

from __future__ import annotations

import re
from datetime import UTC, datetime
from urllib.parse import urlparse

from mm.agent.loop import DiscoveryResult, RecordedStep
from mm.artifact.schema import AppRef, Capability, InputSpec, OutputSpec, Provenance, Step
from mm.surface.base import ActionType, Checkpoint
from mm.values import parameterize, referenced

_CURRENCY = re.compile(r"^-?\$?\s?[\d,]+\.\d{2}$")


class CompileError(ValueError):
    pass


def compile_run(run: DiscoveryResult, cap_id: str, summary: str | None = None) -> Capability:
    entry_path = _relative(run.entry_url)
    if run.status != "success":
        raise CompileError(f"only successful runs compile into capabilities (run status: {run.status})")

    steps = [Step(id="s01_open_app", intent="Open the application entry page.",
                  action=ActionType.NAVIGATE, value=parameterize(entry_path, run.inputs))]
    for i, rec in enumerate(run.steps, start=2):
        steps.append(Step(
            id=f"s{i:02d}_{_slug(rec)}",
            intent=rec.intent,
            action=rec.action,
            target=rec.target,
            value=rec.value_template,
            output=rec.output,
            expect=_checkpoints(rec),
        ))

    secrets: set[str] = set()
    used_inputs: set[str] = set()
    for s in steps:
        if s.value:
            i_names, s_names = referenced(s.value)
            used_inputs |= i_names
            secrets |= s_names

    extracted = {s.output for s in steps if s.output}
    outputs = {name: _output_spec(run.outputs.get(name, "")) for name in sorted(extracted)}
    inputs = {name: _input_spec(value) for name, value in run.inputs.items() if name in used_inputs}

    return Capability(
        id=cap_id,
        summary=summary or parameterize(run.goal, run.inputs),
        app=AppRef(name=cap_id.split(".")[0], entry_path=entry_path),
        inputs=inputs,
        outputs=outputs,
        secrets=sorted(secrets),
        steps=steps,
        success=_success(run.steps),
        provenance=Provenance(
            discovery_run_id=run.run_id,
            recorded_at=datetime.now(UTC),
            provider=run.provider,
            model=run.model,
            goal_template=parameterize(run.goal, run.inputs),
        ),
    )


def _checkpoints(rec: RecordedStep) -> list[Checkpoint]:
    """A step that navigated a frame must land on the same page (path only; query values vary per input)."""
    changed = {k: v for k, v in rec.frame_urls_after.items()
               if _path(rec.frame_urls_before.get(k, "")) != _path(v)}
    if "" in changed:  # the whole document changed: asserting the top frame is enough
        changed = {"": changed[""]}
    return [
        Checkpoint(kind="url_matches", frame_path=[p for p in key.split("/") if p],
                   pattern=re.escape(_path(url)) + r"(\?|$)")
        for key, url in sorted(changed.items())
    ]


def _success(steps: list[RecordedStep]) -> list[Checkpoint]:
    last = next((s for s in reversed(steps) if s.target is not None), None)
    if last is None or last.target is None:
        return []
    key = "/".join(last.target.frame_path)
    url = last.frame_urls_after.get(key)
    if not url:
        return []
    pattern = re.escape(_path(url)) + r"(\?|$)"
    return [Checkpoint(kind="url_matches", frame_path=last.target.frame_path, pattern=pattern)]


def _input_spec(example: str) -> InputSpec:
    if example.isdigit():
        return InputSpec(type="string", pattern=r"^[0-9]+$", description="Numeric identifier.")
    return InputSpec(type="string")


def _output_spec(example: str) -> OutputSpec:
    if _CURRENCY.match(example.strip()):
        return OutputSpec(type="decimal", parse="currency")
    return OutputSpec(type="string")


def _slug(rec: RecordedStep) -> str:
    name = ""
    if rec.target is not None:
        name = rec.target.description.split('"')[1] if '"' in rec.target.description else ""
    if rec.action is ActionType.EXTRACT and rec.output:
        name = rec.output
    words = re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")[:30].strip("_")
    return f"{rec.action.value}_{words}" if words else rec.action.value


def _path(url: str) -> str:
    return urlparse(url).path if url else ""


def _relative(url: str) -> str:
    parts = urlparse(url)
    return parts.path + (f"?{parts.query}" if parts.query else "") or "/"
