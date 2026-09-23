"""The replay result contract returned to the calling agent.

Phase 1 distinguishes success from failure. Business outcomes (e.g. MEMBER_NOT_FOUND) and
escalations are added to this union as their own variants, never as failure sub-kinds, so a
caller can branch on `status` alone.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Annotated, Literal

from pydantic import BaseModel, Field


class FailureKind(StrEnum):
    INPUT_INVALID = "INPUT_INVALID"  # rejected before touching the UI
    TARGET_NOT_FOUND = "TARGET_NOT_FOUND"  # no strategy matched exactly one element in time
    ACTION_FAILED = "ACTION_FAILED"  # element found but the click/fill was refused
    CHECKPOINT_FAILED = "CHECKPOINT_FAILED"  # the step ran but the UI isn't where it should be
    OUTPUT_MISSING = "OUTPUT_MISSING"
    OUTPUT_UNPARSEABLE = "OUTPUT_UNPARSEABLE"
    SUCCESS_CHECK_FAILED = "SUCCESS_CHECK_FAILED"


class StepTrace(BaseModel):
    step_id: str
    ok: bool
    strategy: str | None = None  # which locator strategy matched, e.g. "role"
    strategy_index: int | None = None
    duration_ms: int


class _Base(BaseModel):
    capability_id: str
    capability_version: str
    run_id: str
    steps: list[StepTrace] = Field(default_factory=list)


class ReplaySuccess(_Base):
    status: Literal["success"] = "success"
    outputs: dict[str, str]
    drift: list[str] = Field(default_factory=list, description="Steps that only matched via a fallback strategy.")


class ReplayFailure(_Base):
    status: Literal["failure"] = "failure"
    kind: FailureKind
    step_id: str | None
    message: str
    expected: str | None = None
    observed: str | None = None
    screenshot: str | None = None
    trace: str | None = None


ReplayResult = Annotated[ReplaySuccess | ReplayFailure, Field(discriminator="status")]
