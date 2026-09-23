"""Capability artifact: a typed, versioned, reviewable description of a recorded flow.

A capability is a *contract* (inputs, outputs, status, version) plus a *procedure* (steps with
surface-neutral targets and checkpoints). It is decoupled from the model transcript: nothing in
here needs an LLM to execute. The Pydantic models are the single source of truth; the JSON
Schema an agent reads is generated from them (`Capability.model_json_schema()`).
"""

from __future__ import annotations

import re
from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field, model_validator

from mm.surface.base import ActionType, Checkpoint, Target
from mm.values import referenced

SCHEMA_VERSION = "0.1"
_ID = re.compile(r"^[a-z0-9_]+(\.[a-z0-9_]+)+$")
_SEMVER = re.compile(r"^\d+\.\d+\.\d+$")


class InputSpec(BaseModel):
    type: Literal["string", "integer", "decimal"] = "string"
    description: str = ""
    pattern: str | None = Field(None, description="Regex the value must fully match.")
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
    expect: list[Checkpoint] = Field(default_factory=list, description="Asserted after the step.")
    timeout_ms: int = 10_000

    @model_validator(mode="after")
    def _shape(self) -> Step:
        if self.action is ActionType.NAVIGATE and not self.value:
            raise ValueError(f"step {self.id}: navigate needs a url value")
        if self.action is not ActionType.NAVIGATE and self.target is None:
            raise ValueError(f"step {self.id}: {self.action} needs a target")
        if self.action is ActionType.EXTRACT and not self.output:
            raise ValueError(f"step {self.id}: extract needs an output name")
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
    secrets: list[str] = Field(default_factory=list, description="Names of secrets the flow needs.")
    steps: list[Step]
    success: list[Checkpoint] = Field(default_factory=list)
    provenance: Provenance | None = None

    @model_validator(mode="after")
    def _consistent(self) -> Capability:
        """A reviewer should be able to trust the header: every reference is declared and vice versa."""
        if not _ID.match(self.id):
            raise ValueError(f"id {self.id!r} must be dotted lowercase, e.g. app.area.verb_noun")
        if not _SEMVER.match(self.version):
            raise ValueError(f"version {self.version!r} must be semver MAJOR.MINOR.PATCH")
        used_inputs: set[str] = set()
        used_secrets: set[str] = set()
        for step in self.steps:
            if step.value:
                i, s = referenced(step.value)
                used_inputs |= i
                used_secrets |= s
            if step.output and step.output not in self.outputs:
                raise ValueError(f"step {step.id} extracts undeclared output {step.output!r}")
        if missing := used_inputs - self.inputs.keys():
            raise ValueError(f"steps use undeclared inputs: {sorted(missing)}")
        if missing := used_secrets - set(self.secrets):
            raise ValueError(f"steps use undeclared secrets: {sorted(missing)}")
        produced = {s.output for s in self.steps if s.output}
        if unproduced := self.outputs.keys() - produced:
            raise ValueError(f"declared outputs never extracted: {sorted(unproduced)}")
        return self
