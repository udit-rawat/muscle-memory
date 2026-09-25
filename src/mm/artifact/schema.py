"""Capability artifact: a typed, versioned, reviewable description of a recorded flow.

A capability is a *contract* (inputs, outputs, status, version) plus a *procedure* (steps with
surface-neutral targets and checkpoints) plus *detectors* (the runtime states the procedure knows
how to recognise, and what each one means). It is decoupled from the model transcript: nothing in
here needs an LLM to execute. The Pydantic models are the single source of truth; the JSON Schema
an agent or reviewer reads is generated from them (`Capability.model_json_schema()`).
"""

from __future__ import annotations

import re
from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field, model_validator

from mm.surface.base import STRUCTURAL, ActionType, Checkpoint, Target
from mm.values import referenced

SCHEMA_VERSION = "0.3"
_ID = re.compile(r"^[a-z0-9_]+(\.[a-z0-9_]+)+$")
_SEMVER = re.compile(r"^\d+\.\d+\.\d+$")
_CODE = re.compile(r"^[A-Z][A-Z0-9_]*$")

Risk = Literal["safe", "mutating", "irreversible"]


class InputSpec(BaseModel):
    type: Literal["string", "integer", "decimal"] = "string"
    description: str = ""
    pattern: str | None = Field(None, description="Regex the value must fully match.")
    enum: list[str] | None = Field(None, description="Allowed values (e.g. the options of a dropdown).")
    sensitive: bool = False


class OutputSpec(BaseModel):
    type: Literal["string", "decimal"] = "string"
    parse: Literal["text", "currency"] = "text"
    description: str = ""
    sensitive: bool = Field(True, description="Returned to the caller but masked in logs/evidence.")


class Step(BaseModel):
    id: str
    intent: str = Field(description="What this step is for, in plain words (for reviewers).")
    action: ActionType
    target: Target | None = None
    value: str | None = Field(None, description="Template: literal text, {{input}} or {{secret:NAME}}.")
    output: str | None = Field(None, description="For extract: which declared output receives the text.")
    expect: list[Checkpoint] = Field(default_factory=list, description="Must hold after the step.")
    risk: Risk = Field("safe", description="safe: read/navigate/idempotent; mutating: changes state "
                                           "reversibly; irreversible: commits (post, confirm, transfer).")
    timeout_ms: int = 10_000

    @model_validator(mode="after")
    def _shape(self) -> Step:
        if self.action is ActionType.NAVIGATE and not self.value:
            raise ValueError(f"step {self.id}: navigate needs a url value")
        if self.action is not ActionType.NAVIGATE and self.target is None:
            raise ValueError(f"step {self.id}: {self.action} needs a target")
        if self.action is ActionType.EXTRACT and not self.output:
            raise ValueError(f"step {self.id}: extract needs an output name")
        structural = [st.by for st in self.target.strategies if st.by in STRUCTURAL] if self.target else []
        if self.action is ActionType.EXTRACT and structural:
            raise ValueError(f"step {self.id}: reads may not use structural locators {structural}; a positional "
                             "match can return a value from the wrong row")
        return self


class HandlerAction(BaseModel):
    """One deterministic action a recoverable detector performs (never model-decided)."""

    action: Literal[ActionType.CLICK, ActionType.FILL, ActionType.SELECT, ActionType.NAVIGATE]
    target: Target | None = None
    value: str | None = None


class Detector(BaseModel):
    """A runtime state the capability recognises, and what it means.

    class_:
      business_outcome  a legitimate answer the caller must branch on (e.g. MEMBER_NOT_FOUND); not a crash
      recoverable       handled inside replay with a bounded, deterministic handler (e.g. dismiss a notice)
      hard_failure      stop and surface a debuggable error (e.g. the app returned HTTP 500)

    Detectors are checked while waiting for each step's checkpoints, and again whenever a step fails,
    so an exceptional state is recognised for what it is instead of surfacing as "element not found".
    """

    id: str
    class_: Literal["business_outcome", "recoverable", "hard_failure"] = Field(alias="class")
    description: str = ""
    when: list[Checkpoint] = Field(min_length=1, description="All must hold for the detector to fire.")
    code: str | None = Field(None, description="Result code returned to the caller, e.g. MEMBER_NOT_FOUND.")
    after_steps: list[str] | None = Field(None, description="Only active after these steps; null = every step.")
    handle: list[HandlerAction] = Field(default_factory=list, description="recoverable: actions to run.")
    then: Literal["continue", "retry_step", "restart"] = Field(
        "continue", description="recoverable: after handling, keep waiting on the current step, redo it, or "
                                "restart the flow from the first step. Redo and restart are refused once a "
                                "non-safe step may have taken effect (UNSAFE_TO_REPEAT).")
    max_times: int = Field(1, ge=1, le=5, description="recoverable: bound per run; exceeding it is a failure.")
    source: str = Field("", description="Where this detector came from: a pack, a discovery run, or a reviewer.")

    model_config = {"populate_by_name": True}

    @model_validator(mode="after")
    def _shape(self) -> Detector:
        if self.class_ in ("business_outcome", "hard_failure") and not (self.code and _CODE.match(self.code)):
            raise ValueError(f"detector {self.id}: {self.class_} needs an UPPER_SNAKE code")
        if self.class_ != "recoverable" and self.handle:
            raise ValueError(f"detector {self.id}: only recoverable detectors have handlers")
        if self.class_ == "recoverable" and not self.handle and self.then == "continue":
            raise ValueError(f"detector {self.id}: recoverable with no handler must retry_step or restart")
        return self


class AppRef(BaseModel):
    """Which application this runs against. Deliberately no host: the base URL is bound per invocation,
    because the same vendor app runs at a different address for every tenant."""

    name: str
    entry_path: str = Field(description="Path relative to the tenant's base URL, e.g. /login")


class Provenance(BaseModel):
    discovery_run_id: str
    recorded_at: datetime
    provider: str
    model: str
    goal_template: str = Field(description="The discovery goal with input values replaced by placeholders.")


class Capability(BaseModel):
    schema_version: str = SCHEMA_VERSION
    id: str = Field(description="Dotted name, e.g. corebank.member.get_savings_balance")
    version: str = "0.1.0"
    status: Literal["draft", "approved", "deprecated"] = "draft"
    summary: str
    app: AppRef
    inputs: dict[str, InputSpec] = Field(default_factory=dict)
    outputs: dict[str, OutputSpec] = Field(default_factory=dict)
    outcomes: list[str] = Field(default_factory=list, description="Business outcome codes a caller may receive.")
    secrets: list[str] = Field(default_factory=list, description="Names of secrets the flow needs.")
    steps: list[Step]
    detectors: list[Detector] = Field(default_factory=list, description="Checked in order; first match wins.")
    success: list[Checkpoint] = Field(default_factory=list)
    provenance: Provenance | None = None

    @model_validator(mode="after")
    def _consistent(self) -> Capability:
        """A reviewer should be able to trust the header: every reference is declared and vice versa."""
        if not _ID.match(self.id):
            raise ValueError(f"id {self.id!r} must be dotted lowercase, e.g. app.area.verb_noun")
        if not _SEMVER.match(self.version):
            raise ValueError(f"version {self.version!r} must be semver MAJOR.MINOR.PATCH")
        step_ids = [s.id for s in self.steps]
        if len(set(step_ids)) != len(step_ids):
            raise ValueError("step ids must be unique")
        used_inputs: set[str] = set()
        used_secrets: set[str] = set()
        templates = [s.value for s in self.steps] + [h.value for d in self.detectors for h in d.handle]
        for t in templates:
            if t:
                i, s = referenced(t)
                used_inputs |= i
                used_secrets |= s
        for step in self.steps:
            if step.output and step.output not in self.outputs:
                raise ValueError(f"step {step.id} extracts undeclared output {step.output!r}")
        if missing := used_inputs - self.inputs.keys():
            raise ValueError(f"steps use undeclared inputs: {sorted(missing)}")
        if missing := used_secrets - set(self.secrets):
            raise ValueError(f"steps use undeclared secrets: {sorted(missing)}")
        produced = {s.output for s in self.steps if s.output}
        if unproduced := self.outputs.keys() - produced:
            raise ValueError(f"declared outputs never extracted: {sorted(unproduced)}")

        detector_ids = [d.id for d in self.detectors]
        if len(set(detector_ids)) != len(detector_ids):
            raise ValueError("detector ids must be unique")
        for d in self.detectors:
            if d.after_steps and (unknown := set(d.after_steps) - set(step_ids)):
                raise ValueError(f"detector {d.id} scoped to unknown steps {sorted(unknown)}")
        codes = {d.code for d in self.detectors if d.class_ == "business_outcome" and d.code}
        if codes != set(self.outcomes):
            raise ValueError(f"outcomes {sorted(self.outcomes)} must list exactly the business codes {sorted(codes)}")
        return self
