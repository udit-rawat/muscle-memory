"""Deterministic replay of a real recorded artifact against the mock bank. No LLM involved."""

from pathlib import Path

import pytest

from mm.artifact import store
from mm.evidence.recorder import RunRecorder
from mm.replay.executor import replay
from mm.replay.result import FailureKind, ReplayFailure, ReplaySuccess
from mm.surface.web import WebSurface
from mm.values import SecretStore

FIXTURE = Path(__file__).parent / "fixtures" / "get_savings_balance.yaml"
SECRETS = SecretStore({"MOCKBANK_USERNAME": "operator1", "MOCKBANK_PASSWORD": "change-me-local-only"})


def _replay(bank_url: str, tmp_path: Path, member_id: str) -> ReplaySuccess | ReplayFailure:
    recorder = RunRecorder(tmp_path, "replay", SECRETS)
    try:
        return replay(store.load(FIXTURE), {"member_id": member_id}, base_url=bank_url, secrets=SECRETS,
                      recorder=recorder, surface_factory=lambda rec: WebSurface(headless=True))
    finally:
        recorder.close()


@pytest.mark.parametrize("member_id, balance", [("10234", "2450.17"), ("10871", "15320.00")])
def test_replay_returns_parsed_output_for_any_member(bank_url: str, tmp_path: Path, member_id: str,
                                                     balance: str) -> None:
    result = _replay(bank_url, tmp_path, member_id)
    assert isinstance(result, ReplaySuccess), result
    assert result.outputs == {"savings_balance": balance}
    assert result.drift == []  # every step matched on its primary (semantic) strategy


def test_bad_input_fails_before_touching_the_ui(bank_url: str, tmp_path: Path) -> None:
    result = _replay(bank_url, tmp_path, "12ab")
    assert isinstance(result, ReplayFailure) and result.kind is FailureKind.INPUT_INVALID
    assert not list(tmp_path.glob("*/screenshots/*.png"))


def test_unknown_member_stops_with_debuggable_failure(bank_url: str, tmp_path: Path) -> None:
    # Phase 1 behaviour: a hard failure with the step, the strategies tried and a screenshot.
    # (Phase 2 turns this into a MEMBER_NOT_FOUND business outcome via detectors.)
    result = _replay(bank_url, tmp_path, "99999")
    assert isinstance(result, ReplayFailure) and result.kind is FailureKind.TARGET_NOT_FOUND
    assert result.step_id == "s08_click_select" and result.screenshot and Path(result.screenshot).exists()


def test_secrets_never_reach_the_event_log(bank_url: str, tmp_path: Path) -> None:
    _replay(bank_url, tmp_path, "10234")
    log = next(tmp_path.glob("*/events.jsonl")).read_text()
    assert "change-me-local-only" not in log and "2,450.17" not in log
