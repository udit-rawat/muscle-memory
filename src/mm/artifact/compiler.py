"""Compile a successful discovery run into a Capability.

What the compiler decides (not the model):
- inputs/outputs and their types, inferred from the concrete values seen during discovery
  (a dropdown's options become the input's enum);
- values become templates, so no concrete input value is stored in the artifact;
- per-step checkpoints: the frame URL a step navigated to, or else "the next step's target is there",
  so replay never acts before the UI is ready and never assumes a click worked;
- each step's risk level;
- detectors: the app's detector pack, plus any popup the agent had to dismiss during discovery,
  which becomes a recoverable detector rather than a step (it will not be there on every run);
- the success condition, from where the flow ended.
"""

from __future__ import annotations

import re
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import urlparse

from mm.agent.loop import DiscoveryResult, RecordedStep
from mm.artifact.packs import load_pack
from mm.artifact.schema import (
    AppRef,
    Capability,
    Detector,
    HandlerAction,
    InputSpec,
    OutputSpec,
    Provenance,
    Risk,
    Step,
)
from mm.surface.base import ActionType, Checkpoint, Target
from mm.values import parameterize, referenced

_CURRENCY = re.compile(r"^-?\$?\s?[\d,]+\.\d{2}$")
_DECIMAL = re.compile(r"^\d+\.\d{1,2}$")
_PLACEHOLDER = re.compile(r"^\{\{\s*([A-Za-z_][A-Za-z0-9_]*)\s*\}\}$")
# Controls whose activation commits something. Phase 3 moves this rule into the policy file.
_IRREVERSIBLE = re.compile(
    r"(?i)\b(confirm|post|transfer|delete|close account|approve|disburse|submit payment)\b")


class CompileError(ValueError):
    pass


def compile_run(
    run: DiscoveryResult, cap_id: str, summary: str | None = None, pack: str | None = None,
    packs_dir: Path = Path("packs"),
) -> Capability:
    if run.status != "success":
        raise CompileError(f"only successful runs compile into capabilities (run status: {run.status})")
    entry_path = _relative(run.entry_url)
    flow = [r for r in run.steps if not r.interruption]

    steps = [Step(id="s01_open_app", intent="Open the application entry page.",
                  action=ActionType.NAVIGATE, value=parameterize(entry_path, run.inputs))]
    for i, rec in enumerate(flow):
        nxt = flow[i + 1] if i + 1 < len(flow) else None
        steps.append(Step(
            id=f"s{i + 2:02d}_{_slug(rec)}",
            intent=rec.intent,
            action=rec.action,
            target=rec.target,
            value=rec.value_template,
            output=rec.output,
            expect=_checkpoints(rec, nxt),
            risk=_risk(rec),
        ))

    pack_name = pack if pack is not None else cap_id.split(".")[0]
    loaded = load_pack(pack_name, packs_dir) if pack_name else None
    detectors = list(loaded.detectors) if loaded else []
    detectors += _learned_detectors(run, detectors)

    used_inputs: set[str] = set()
    secrets: set[str] = set()
    for s in steps:
        if s.value:
            i_names, s_names = referenced(s.value)
            used_inputs |= i_names
            secrets |= s_names

    extracted = {s.output for s in steps if s.output}
    outputs = {name: _output_spec(run.outputs.get(name, "")) for name in sorted(extracted)}
    inputs = {name: _input_spec(value, _enum_for(name, flow)) for name, value in run.inputs.items()
              if name in used_inputs}

    return Capability(
        id=cap_id,
        summary=summary or parameterize(run.goal, run.inputs),
        app=AppRef(name=cap_id.split(".")[0], entry_path=entry_path),
        inputs=inputs,
        outputs=outputs,
        outcomes=sorted({d.code for d in detectors if d.class_ == "business_outcome" and d.code}),
        secrets=sorted(secrets),
        steps=steps,
        detectors=detectors,
        success=_success(flow),
        provenance=Provenance(
            discovery_run_id=run.run_id,
            recorded_at=datetime.now(UTC),
            provider=run.provider,
            model=run.model,
            goal_template=parameterize(run.goal, run.inputs),
        ),
    )


def _checkpoints(rec: RecordedStep, nxt: RecordedStep | None) -> list[Checkpoint]:
    """A step that navigated a frame must land on the same page (path only; query values vary per input).
    A step that didn't navigate must at least make the next step's control appear."""
    changed = {k: v for k, v in rec.frame_urls_after.items()
               if _path(rec.frame_urls_before.get(k, "")) != _path(v)}
    if "" in changed:  # the whole document changed: asserting the top frame is enough
        changed = {"": changed[""]}
    checks = [
        Checkpoint(kind="url_matches", frame_path=[p for p in key.split("/") if p],
                   pattern=re.escape(_path(url)) + r"(\?|$)")
        for key, url in sorted(changed.items())
    ]
    if not checks and nxt is not None and nxt.target is not None:
        checks.append(Checkpoint(kind="target_present", frame_path=nxt.target.frame_path, target=nxt.target))
    return checks


def _risk(rec: RecordedStep) -> Risk:
    if rec.action is ActionType.CLICK and rec.target is not None and _IRREVERSIBLE.search(rec.target.description):
        return "irreversible"
    return "safe"


def _learned_detectors(run: DiscoveryResult, existing: list[Detector]) -> list[Detector]:
    """Popups the agent dismissed during discovery become recoverable detectors (unless already known)."""
    known = {_first_strategy(h.target) for d in existing for h in d.handle if h.target is not None}
    learned: list[Detector] = []
    for rec in run.steps:
        if not rec.interruption or rec.target is None or _first_strategy(rec.target) in known:
            continue
        known.add(_first_strategy(rec.target))
        learned.append(Detector(
            id=f"learned_{_slug(rec)}", class_="recoverable",
            description=f"Seen during discovery and dismissed by the agent: {rec.intent}",
            when=[Checkpoint(kind="target_present", frame_path=rec.target.frame_path, target=rec.target)],
            handle=[HandlerAction(action=ActionType.CLICK, target=rec.target)],
            then="continue", max_times=2, source=f"discovery:{run.run_id}",
        ))
    return learned


def _first_strategy(target: Target) -> str:
    return "/".join(target.frame_path) + "|" + target.strategies[0].model_dump_json()


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


def _enum_for(name: str, steps: list[RecordedStep]) -> list[str] | None:
    """If an input is the whole value of a dropdown selection, its allowed values are the options."""
    for rec in steps:
        m = _PLACEHOLDER.match(rec.value_template or "")
        if rec.action is ActionType.SELECT and m and m.group(1) == name and rec.options:
            return [o for o in rec.options if o and not o.startswith("--")]
    return None


def _input_spec(example: str, enum: list[str] | None) -> InputSpec:
    if enum:
        return InputSpec(type="string", enum=enum, description="One of the dropdown's options.")
    if example.isdigit():
        return InputSpec(type="string", pattern=r"^[0-9]+$", description="Numeric identifier.")
    if _DECIMAL.match(example):
        return InputSpec(type="decimal", pattern=r"^[0-9]+(\.[0-9]{1,2})?$",
                         description="Amount with up to two decimals.")
    return InputSpec(type="string")


def _output_spec(example: str) -> OutputSpec:
    if _CURRENCY.match(example.strip()):
        return OutputSpec(type="decimal", parse="currency")
    return OutputSpec(type="string")


def _slug(rec: RecordedStep) -> str:
    name = ""
    if rec.target is not None and '"' in rec.target.description:
        name = rec.target.description.split('"')[1]
    if rec.action is ActionType.EXTRACT and rec.output:
        name = rec.output
    words = re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")[:30].strip("_")
    return f"{rec.action.value}_{words}" if words else rec.action.value


def _path(url: str) -> str:
    return urlparse(url).path if url else ""


def _relative(url: str) -> str:
    parts = urlparse(url)
    return parts.path + (f"?{parts.query}" if parts.query else "") or "/"
