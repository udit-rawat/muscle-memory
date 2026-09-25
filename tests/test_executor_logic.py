"""Replay control flow against a scripted surface: the edge cases that are hard to trigger live."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from mm.artifact.schema import Capability
from mm.evidence.recorder import RunRecorder
from mm.replay.executor import replay
from mm.replay.result import FailureKind, ReplayBusinessOutcome, ReplayFailure, ReplayResult, ReplaySuccess
from mm.surface.base import ActResult
from mm.values import SecretStore
from tests.fakes import Effect, FakeState, FakeSurface

SECRETS = SecretStore({})


def _t(name: str) -> dict[str, Any]:
    return {"strategies": [{"by": "role", "role": "button", "name": name}], "description": name}


def _cap(steps: list[dict[str, Any]], detectors: list[dict[str, Any]] | None = None,
         outcomes: list[str] | None = None) -> Capability:
    return Capability.model_validate({
        "id": "app.test.flow", "summary": "test", "app": {"name": "app", "entry_path": "/login"},
        "steps": steps, "detectors": detectors or [], "outcomes": outcomes or [],
    })


def _run(cap: Capability, state: FakeState, effects: dict[str, Effect], tmp_path: Path) -> ReplayResult:
    rec = RunRecorder(tmp_path, "replay", SECRETS)
    try:
        return replay(cap, {}, base_url="http://app", secrets=SECRETS, recorder=rec,
                      surface_factory=lambda _: FakeSurface(state, effects))
    finally:
        rec.close()


def _goto(url: str, *present: str, texts: tuple[str, ...] = ()) -> Effect:
    def effect(s: FakeState) -> ActResult:
        s.url, s.present, s.texts = url, set(present), set(texts)
        return ActResult(ok=True, strategy_index=0)
    return effect


SEARCH = {"id": "s1", "intent": "search", "action": "click", "target": _t("Search"),
          "expect": [{"kind": "target_present", "target": _t("Select")}]}
SELECT = {"id": "s2", "intent": "open", "action": "click", "target": _t("Select"),
          "expect": [{"kind": "url_matches", "pattern": "/detail"}]}
NOT_FOUND = {"id": "member_not_found", "class": "business_outcome", "code": "MEMBER_NOT_FOUND",
             "when": [{"kind": "text_visible", "any_frame": True, "pattern": "No member matches"}]}


def test_business_outcome_is_recognised_while_waiting_for_a_checkpoint(tmp_path: Path) -> None:
    cap = _cap([SEARCH, SELECT], [NOT_FOUND], ["MEMBER_NOT_FOUND"])
    state = FakeState(present={"Search"})
    result = _run(cap, state, {"Search": _goto("/search", texts=("No member matches the criteria.",))}, tmp_path)
    assert isinstance(result, ReplayBusinessOutcome)
    assert (result.code, result.step_id) == ("MEMBER_NOT_FOUND", "s1")
    assert result.message == "No member matches the criteria."


def test_detectors_get_the_first_say_when_a_step_fails(tmp_path: Path) -> None:
    # The Select link is missing *because* nothing matched: that is an outcome, not TARGET_NOT_FOUND.
    # Scoped to s2 so it can only fire through the failed-step path, not while waiting after s1.
    cap = _cap([{**SEARCH, "expect": []}, SELECT], [{**NOT_FOUND, "after_steps": ["s2"]}], ["MEMBER_NOT_FOUND"])
    state = FakeState(present={"Search"})
    result = _run(cap, state, {"Search": _goto("/search", texts=("No member matches",))}, tmp_path)
    assert isinstance(result, ReplayBusinessOutcome) and result.step_id == "s2"
    assert "Select" not in state.performed  # it never clicked anything; the missing link *was* the answer


def test_without_a_matching_detector_a_missing_target_is_a_hard_failure(tmp_path: Path) -> None:
    cap = _cap([{**SEARCH, "expect": []}, SELECT])
    result = _run(cap, FakeState(present={"Search"}), {"Search": _goto("/search")}, tmp_path)
    assert isinstance(result, ReplayFailure) and result.kind is FailureKind.TARGET_NOT_FOUND
    assert result.step_id == "s2" and result.expected and "role" in result.expected


def test_recoverable_detector_handles_then_continues(tmp_path: Path) -> None:
    notice = {"id": "notice", "class": "recoverable",
              "when": [{"kind": "target_present", "target": _t("Acknowledge")}],
              "handle": [{"action": "click", "target": _t("Acknowledge")}], "then": "continue"}
    cap = _cap([SEARCH, SELECT], [notice])
    state = FakeState(present={"Search"})
    effects = {"Search": _goto("/search", "Select", "Acknowledge"),
               "Acknowledge": lambda s: s.present.discard("Acknowledge") or None,
               "Select": _goto("/detail")}
    result = _run(cap, state, effects, tmp_path)
    assert isinstance(result, ReplaySuccess)
    assert [(r.detector_id, r.action) for r in result.recoveries] == [("notice", "handled")]


def test_recovery_is_bounded(tmp_path: Path) -> None:
    sticky = {"id": "sticky", "class": "recoverable", "max_times": 2,
              "when": [{"kind": "target_present", "target": _t("Acknowledge")}],
              "handle": [{"action": "click", "target": _t("Acknowledge")}]}
    cap = _cap([SEARCH, SELECT], [sticky])
    state = FakeState(present={"Search"})
    result = _run(cap, state, {"Search": _goto("/search", "Select", "Acknowledge")}, tmp_path)  # never goes away
    assert isinstance(result, ReplayFailure) and result.kind is FailureKind.RECOVERY_EXHAUSTED


EXPIRED = {"id": "session_expired", "class": "recoverable", "then": "restart",
           "when": [{"kind": "text_visible", "any_frame": True, "pattern": "session has expired"}]}


def _expiring_flow(confirm_risk: str) -> tuple[Capability, FakeState, dict[str, Effect]]:
    steps = [
        {"id": "s1", "intent": "open", "action": "navigate", "value": "/login"},
        {"id": "s2", "intent": "go", "action": "click", "target": _t("Go"),
         "expect": [{"kind": "target_present", "target": _t("Confirm")}]},
        {"id": "s3", "intent": "commit", "action": "click", "target": _t("Confirm"), "risk": confirm_risk,
         "expect": [{"kind": "url_matches", "pattern": "/done"}]},
    ]
    state = FakeState(present={"Go"})
    expired_once: list[bool] = []

    def confirm(s: FakeState) -> ActResult:
        if not expired_once:  # the session expires exactly when Confirm is pressed the first time
            expired_once.append(True)
            s.url, s.texts, s.present = "/login?expired=1", {"Your session has expired."}, {"Go"}
        else:
            s.url, s.texts = "/done", set()
        return ActResult(ok=True, strategy_index=0)

    effects = {"http://app/login": _goto("/login", "Go"), "Go": _goto("/form", "Confirm"), "Confirm": confirm}
    return _cap(steps, [EXPIRED]), state, effects


def test_session_expiry_before_any_commit_restarts_the_flow(tmp_path: Path) -> None:
    cap, state, effects = _expiring_flow(confirm_risk="safe")
    result = _run(cap, state, effects, tmp_path)
    assert isinstance(result, ReplaySuccess)
    assert [r.action for r in result.recoveries] == ["restarted"]


def test_restart_is_refused_after_an_irreversible_step(tmp_path: Path) -> None:
    # Restarting would press Confirm a second time: never silently repeat a side effect.
    cap, state, effects = _expiring_flow(confirm_risk="irreversible")
    result = _run(cap, state, effects, tmp_path)
    assert isinstance(result, ReplayFailure) and result.kind is FailureKind.RESTART_UNSAFE
    assert state.performed.count("Confirm") == 1


def test_unrecognised_overlay_blocks_progress(tmp_path: Path) -> None:
    cap = _cap([SEARCH, SELECT])
    state = FakeState(present={"Search"})

    def search(s: FakeState) -> ActResult:
        s.present, s.overlays = {"Select"}, ["overlay in frame main: 'Quick Survey'"]
        return ActResult(ok=True, strategy_index=0)

    result = _run(cap, state, {"Search": search}, tmp_path)
    assert isinstance(result, ReplayFailure) and result.kind is FailureKind.UNEXPECTED_STATE
    assert result.step_id == "s1" and "Quick Survey" in (result.observed or "")


def test_hard_failure_detector_stops_with_its_code(tmp_path: Path) -> None:
    boom = {"id": "app_error", "class": "hard_failure", "code": "APP_ERROR",
            "when": [{"kind": "text_matches", "any_frame": True, "pattern": "HTTP 500"}]}
    cap = _cap([SEARCH, SELECT], [boom])
    result = _run(cap, FakeState(present={"Search"}), {"Search": _goto("/x", texts=("HTTP 500 - Internal",))}, tmp_path)
    assert isinstance(result, ReplayFailure) and (result.kind, result.code) == (FailureKind.APP_ERROR, "APP_ERROR")


def test_detector_scope_limits_where_it_applies(tmp_path: Path) -> None:
    scoped = {**NOT_FOUND, "after_steps": ["s2"]}
    cap = _cap([SEARCH, SELECT], [scoped], ["MEMBER_NOT_FOUND"])
    state = FakeState(present={"Search"})
    result = _run(cap, state, {"Search": _goto("/search", "Select", texts=("No member matches",)),
                               "Select": _goto("/detail", texts=("No member matches",))}, tmp_path)
    # Not active after s1; after s2 the text is still on screen, so it fires there.
    assert isinstance(result, ReplayBusinessOutcome) and result.step_id == "s2"
