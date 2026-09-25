"""The replay result contract returned to the calling agent.

A caller branches on `status` alone:

  success           outputs are present and the success condition held
  business_outcome  a legitimate answer from the application (e.g. MEMBER_NOT_FOUND); `code` is one of
                    the capability's declared `outcomes`. Not an error: nothing is wrong with the system.
  failure           the capability could not complete; `kind` says why and `step_id`/`expected`/`observed`
                    plus a screenshot say where, for whoever debugs it.

Recoverable conditions (a dismissed notice, a re-login, a retried slow step) never change the status;
they are listed in `recoveries` so they stay visible without failing the run.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Annotated, Literal

from pydantic import BaseModel, Field


class FailureKind(StrEnum):
    INPUT_INVALID = "INPUT_INVALID"  # rejected before touching the UI
    TARGET_NOT_FOUND = "TARGET_NOT_FOUND"  # no strategy matched exactly one element in time
    ACTION_FAILED = "ACTION_FAILED"  # element found but the click/fill was refused
    CHECKPOINT_FAILED = "CHECKPOINT_FAILED"  # the step ran but the UI never reached the expected state
    UNEXPECTED_STATE = "UNEXPECTED_STATE"  # something unrecognised (e.g. an unknown dialog) blocks the UI
    APP_ERROR = "APP_ERROR"  # a hard_failure detector fired (the detector's code is in `code`)
    RECOVERY_EXHAUSTED = "RECOVERY_EXHAUSTED"  # a recoverable condition kept coming back past its bound
    RESTART_UNSAFE = "RESTART_UNSAFE"  # a restart was needed after a non-safe step had already run
    OUTPUT_MISSING = "OUTPUT_MISSING"
    OUTPUT_UNPARSEABLE = "OUTPUT_UNPARSEABLE"
    SUCCESS_CHECK_FAILED = "SUCCESS_CHECK_FAILED"


class StepTrace(BaseModel):
    step_id: str
    ok: bool
    strategy: str | None = None  # which locator strategy matched, e.g. "role"
    strategy_index: int | None = None
    duration_ms: int


class Recovery(BaseModel):
    detector_id: str
    step_id: str
    action: Literal["handled", "retried_step", "restarted"]
    detail: str = ""


class _Base(BaseModel):
    capability_id: str
    capability_version: str
    run_id: str
    steps: list[StepTrace] = Field(default_factory=list)
    recoveries: list[Recovery] = Field(default_factory=list)


class ReplaySuccess(_Base):
    status: Literal["success"] = "success"
    outputs: dict[str, str]
    drift: list[str] = Field(default_factory=list, description="Steps that only matched via a fallback strategy.")


class ReplayBusinessOutcome(_Base):
    status: Literal["business_outcome"] = "business_outcome"
    code: str
    message: str  # what the application said, e.g. "No member matches the search criteria."
    detector_id: str
    step_id: str | None
    screenshot: str | None = None


class ReplayFailure(_Base):
    status: Literal["failure"] = "failure"
    kind: FailureKind
    code: str | None = None  # for APP_ERROR: the hard_failure detector's code
    step_id: str | None
    message: str
    expected: str | None = None
    observed: str | None = None
    screenshot: str | None = None
    trace: str | None = None


ReplayResult = Annotated[ReplaySuccess | ReplayBusinessOutcome | ReplayFailure, Field(discriminator="status")]
