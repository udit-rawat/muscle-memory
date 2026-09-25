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

from mm.handoff.intervention import InterventionSummary


class FailureKind(StrEnum):
    INPUT_INVALID = "INPUT_INVALID"  # rejected before touching the UI
    TARGET_NOT_FOUND = "TARGET_NOT_FOUND"  # no strategy matched exactly one element in time
    ACTION_FAILED = "ACTION_FAILED"  # element found but the click/fill was refused
    CHECKPOINT_FAILED = "CHECKPOINT_FAILED"  # the step ran but the UI never reached the expected state
    UNEXPECTED_STATE = "UNEXPECTED_STATE"  # something unrecognised (e.g. an unknown dialog) blocks the UI
    APP_ERROR = "APP_ERROR"  # a hard_failure detector fired (the detector's code is in `code`)
    RECOVERY_EXHAUSTED = "RECOVERY_EXHAUSTED"  # a recoverable condition kept coming back past its bound
    UNSAFE_TO_REPEAT = "UNSAFE_TO_REPEAT"  # recovering would re-run a step that may already have committed
    APP_UNREACHABLE = "APP_UNREACHABLE"  # the application could not be loaded at all
    INTERNAL_ERROR = "INTERNAL_ERROR"  # a bug in this system, reported as a result instead of a traceback
    POLICY_BLOCKED = "POLICY_BLOCKED"  # outside the allowlist, or irreversible without a valid approval
    CONTROL_LOST = "CONTROL_LOST"  # automation tried to act without holding the session's control lease
    ESCALATION_ABORTED = "ESCALATION_ABORTED"  # the operator chose to stop the run
    ESCALATION_TIMEOUT = "ESCALATION_TIMEOUT"  # nobody took over before the escalation timed out
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
    approved_by: str | None = None  # who signed off this artifact version, if anyone
    steps: list[StepTrace] = Field(default_factory=list)
    recoveries: list[Recovery] = Field(default_factory=list)
    interventions: list[InterventionSummary] = Field(default_factory=list)


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
    may_have_committed: bool = Field(False, description="A non-safe step was dispatched before the failure: "
                                                        "check the application before retrying the capability.")


ReplayResult = Annotated[ReplaySuccess | ReplayBusinessOutcome | ReplayFailure, Field(discriminator="status")]
