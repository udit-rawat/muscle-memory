"""Deterministic replay of real recorded artifacts against the mock bank, under each injected fault.
No LLM involved. Every row pins the result class the caller receives."""

from __future__ import annotations

from pathlib import Path

import pytest

from mm.artifact import store
from mm.evidence.recorder import RunRecorder
from mm.policy.model import Policy
from mm.replay.executor import replay
from mm.replay.result import FailureKind, ReplayBusinessOutcome, ReplayFailure, ReplayResult, ReplaySuccess
from mm.surface.web import WebSurface
from mm.values import SecretStore
from mock_bank import app as bank
from tests.fakes import approval_for

CAPS = Path(__file__).resolve().parent.parent / "capabilities"
POLICY = Policy.load(CAPS.parent / "config" / "policy.yaml")
SECRETS = SecretStore({"MOCKBANK_USERNAME": "operator1", "MOCKBANK_PASSWORD": "change-me-local-only"})
OPEN = {"account_type": "Holiday Club", "deposit": "25.00", "nickname": "Fund"}


def _replay(bank_url: str, tmp_path: Path, cap_id: str, **inputs: str) -> ReplayResult:
    recorder = RunRecorder(tmp_path, "replay", SECRETS)
    try:
        cap = store.load(store.latest_path(cap_id, CAPS))
        return replay(cap, inputs, base_url=bank_url, secrets=SECRETS, policy=POLICY, approval=approval_for(cap),
                      recorder=recorder, surface_factory=lambda rec: WebSurface(headless=True))
    finally:
        recorder.close()


def _balance(bank_url: str, tmp_path: Path, member_id: str) -> ReplayResult:
    return _replay(bank_url, tmp_path, "corebank.member.get_savings_balance", member_id=member_id)


@pytest.mark.parametrize("member_id, balance", [("10234", "2450.17"), ("10871", "15320.00")])
def test_success_for_any_member(bank_url: str, tmp_path: Path, member_id: str, balance: str) -> None:
    result = _balance(bank_url, tmp_path, member_id)
    assert isinstance(result, ReplaySuccess), result
    assert result.outputs == {"savings_balance": balance} and result.drift == [] and result.recoveries == []


@pytest.mark.parametrize("member_id, fault, code", [
    ("99999", None, "MEMBER_NOT_FOUND"),
    ("12011", None, "PERMISSION_DENIED"),  # restricted member
    ("10234", "permission", "PERMISSION_DENIED"),
])
def test_business_outcomes(bank_url: str, tmp_path: Path, member_id: str, fault: str | None, code: str) -> None:
    if fault:
        bank.FAULTS.add(fault)
    result = _balance(bank_url, tmp_path, member_id)
    assert isinstance(result, ReplayBusinessOutcome), result
    assert result.code == code and result.screenshot and Path(result.screenshot).exists()


@pytest.mark.parametrize("fault, detector, action", [
    ("notice", "system_notice", "handled"),
    ("session_expired", "session_expired", "restarted"),
    ("survey", "learned_click_maybe_later", "handled"),  # learned live, during discovery with the popup on
])
def test_recoverable_conditions_succeed_and_are_reported(bank_url: str, tmp_path: Path, fault: str, detector: str,
                                                         action: str) -> None:
    bank.FAULTS.add(fault)
    result = _balance(bank_url, tmp_path, "10234")
    assert isinstance(result, ReplaySuccess), result
    assert result.outputs == {"savings_balance": "2450.17"}
    assert [(r.detector_id, r.action) for r in result.recoveries] == [(detector, action)]


def test_slow_application_is_absorbed_by_checkpoint_waits(bank_url: str, tmp_path: Path) -> None:
    bank.FAULTS.add("slow")
    result = _balance(bank_url, tmp_path, "10234")
    assert isinstance(result, ReplaySuccess) and result.outputs == {"savings_balance": "2450.17"}


@pytest.mark.parametrize("member_id, fault, kind", [
    ("10234", "error500", FailureKind.APP_ERROR),
    ("12ab", None, FailureKind.INPUT_INVALID),
])
def test_failures(bank_url: str, tmp_path: Path, member_id: str, fault: str | None, kind: FailureKind) -> None:
    if fault:
        bank.FAULTS.add(fault)
    result = _balance(bank_url, tmp_path, member_id)
    assert isinstance(result, ReplayFailure), result
    assert result.kind is kind


def test_open_sub_account_commits_and_returns_confirmation(bank_url: str, tmp_path: Path) -> None:
    result = _replay(bank_url, tmp_path, "corebank.member.open_sub_account", member_id="10871", **OPEN)
    assert isinstance(result, ReplaySuccess), result
    assert result.outputs["confirmation_number"].startswith("CNF") and result.outputs["new_suffix"].startswith("S")
    assert len(bank.data.MEMBERS["10871"].accounts) == 3


@pytest.mark.parametrize("deposit, message", [
    ("1.00", "at least $5.00"),
    ("300000", "supervisor override required"),
])
def test_open_sub_account_validation_is_a_business_outcome(bank_url: str, tmp_path: Path, deposit: str,
                                                            message: str) -> None:
    inputs = {**OPEN, "deposit": deposit}
    result = _replay(bank_url, tmp_path, "corebank.member.open_sub_account", member_id="10871", **inputs)
    assert isinstance(result, ReplayBusinessOutcome), result
    assert result.code == "VALIDATION_REJECTED" and message in result.message
    assert len(bank.data.MEMBERS["10871"].accounts) == 2  # nothing was committed


def test_open_sub_account_rejects_values_outside_the_dropdown(bank_url: str, tmp_path: Path) -> None:
    result = _replay(bank_url, tmp_path, "corebank.member.open_sub_account", member_id="10871",
                     **{**OPEN, "account_type": "Checking"})
    assert isinstance(result, ReplayFailure) and result.kind is FailureKind.INPUT_INVALID


def test_secrets_and_outputs_never_reach_the_event_log(bank_url: str, tmp_path: Path) -> None:
    _balance(bank_url, tmp_path, "10234")
    log = next(tmp_path.glob("*/events.jsonl")).read_text()
    assert "change-me-local-only" not in log and "2,450.17" not in log and "2450.17" not in log


def test_popup_unknown_to_a_capability_blocks_it(bank_url: str, tmp_path: Path) -> None:
    # open_sub_account was recorded without the survey, so for it the popup is unknown: never carry on under it.
    bank.FAULTS.add("survey")
    result = _replay(bank_url, tmp_path, "corebank.member.open_sub_account", member_id="10871", **OPEN)
    assert isinstance(result, ReplayFailure) and result.kind is FailureKind.UNEXPECTED_STATE
    assert "Quick Survey" in (result.observed or "")
    assert len(bank.data.MEMBERS["10871"].accounts) == 2
