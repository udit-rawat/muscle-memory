"""A closed browser session is reported as such, and one capability is reused across tenants via profiles."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from mm.artifact.schema import Capability
from mm.evidence.recorder import RunRecorder
from mm.handoff.intervention import HandoffController, Status
from mm.replay.executor import replay
from mm.replay.result import FailureKind, ReplayFailure
from mm.surface.base import ActResult
from mm.values import SecretStore
from tests.fakes import PERMISSIVE, FakeState, FakeSurface, approval_for

CLOSED = "Target page, context or browser has been closed"


def _cap() -> Capability:
    t = {"strategies": [{"by": "role", "role": "button", "name": "Search"}], "description": "Search"}
    return Capability.model_validate({
        "id": "app.t.flow", "summary": "s", "app": {"name": "a", "entry_path": "/"},
        "steps": [{"id": "s1", "intent": "search", "action": "click", "target": t,
                   "expect": [{"kind": "url_matches", "pattern": "/results"}]}]})


def _survey(s: FakeState) -> ActResult:
    s.url, s.overlays = "/results", ["overlay in frame main: 'Quick Survey'"]
    return ActResult(ok=True, strategy_index=0)


class _WindowClosedDuringHandoff(FakeSurface):
    def idle(self, ms: int) -> None:
        raise RuntimeError(f"Page.wait_for_timeout: {CLOSED}")


class _WindowClosedMidStep(FakeSurface):
    def perform(self, *args: Any, **kwargs: Any) -> ActResult:
        raise RuntimeError(f"Locator.click: {CLOSED}")


def test_closing_the_browser_during_a_handoff_is_reported_as_such(tmp_path: Path) -> None:
    rec = RunRecorder(tmp_path, "replay", SecretStore({}))
    c = HandoffController(rec)
    cap = _cap()
    result = replay(cap, {}, base_url="http://a", secrets=SecretStore({}), recorder=rec, policy=PERMISSIVE,
                    approval=approval_for(cap), handoff=c,
                    surface_factory=lambda _: _WindowClosedDuringHandoff(FakeState(present={"Search"}),
                                                                         {"Search": _survey}))
    rec.close()
    assert isinstance(result, ReplayFailure) and result.kind is FailureKind.SESSION_CLOSED
    assert c.all()[0].status is Status.SESSION_CLOSED


def test_closing_the_browser_mid_step_is_reported_as_such(tmp_path: Path) -> None:
    rec = RunRecorder(tmp_path, "replay", SecretStore({}))
    cap = _cap()
    result = replay(cap, {}, base_url="http://a", secrets=SecretStore({}), recorder=rec, policy=PERMISSIVE,
                    approval=approval_for(cap),
                    surface_factory=lambda _: _WindowClosedMidStep(FakeState(present={"Search"}), {}))
    rec.close()
    assert isinstance(result, ReplayFailure) and result.kind is FailureKind.SESSION_CLOSED


def test_slow_mo_reaches_the_browser() -> None:
    from mm.cli import web_surface_factory
    from mm.policy.model import Policy
    from mm.surface.web import WebSurface
    seen: dict[str, Any] = {}
    original = WebSurface.__init__

    def spy(self: WebSurface, *args: Any, **kwargs: Any) -> None:
        seen.update(kwargs)
        original(self, *args, **kwargs)
    WebSurface.__init__ = spy  # type: ignore[method-assign]
    try:
        rec = RunRecorder(Path(__import__("tempfile").mkdtemp()), "t", SecretStore({}))
        web_surface_factory(True, False, Policy.load().bind("http://127.0.0.1:1"), slow_mo_ms=250)(rec).close()
    finally:
        WebSurface.__init__ = original  # type: ignore[method-assign]
    assert seen.get("slow_mo_ms") == 250


# --- tenant reuse: one recorded capability, specialised per tenant, never re-recorded --------------

import pytest  # noqa: E402

from mm.artifact import store  # noqa: E402
from mm.replay.result import ReplaySuccess  # noqa: E402
from mm.surface.web import WebSurface  # noqa: E402
from mock_bank import app as bank  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
BALANCE = store.latest_path("corebank.member.get_savings_balance", ROOT / "capabilities")
OPEN = store.latest_path("corebank.member.open_sub_account", ROOT / "capabilities")
SECRETS = SecretStore({"MOCKBANK_USERNAME": "operator1", "MOCKBANK_PASSWORD": "change-me-local-only"})


def test_specialise_maps_the_tenant_vocabulary_everywhere() -> None:
    from mm.artifact.tenancy import load_tenant, specialise
    base = store.load(BALANCE)
    cap = specialise(base, load_tenant("tenant_b", ROOT / "tenants"))
    dumped = cap.model_dump_json(by_alias=True)
    for renamed in ('"Find Member"', '"Account Holder ID"', '"Go"', '"Primary Savings"'):
        assert renamed in dumped
    assert '"name":"Member Search"' not in dumped.replace(" ", "") and '"row_key":"Share Savings"' not in dumped
    assert cap.app.tenant == "tenant_b" and base.app.tenant is None
    assert [s.id for s in cap.steps] == [s.id for s in base.steps]  # same flow, same contract
    assert cap.inputs == base.inputs and cap.outputs == base.outputs and cap.outcomes == base.outcomes


def test_a_tenant_profile_for_another_app_is_refused() -> None:
    from mm.artifact.tenancy import TenantMismatch, TenantProfile, specialise
    with pytest.raises(TenantMismatch):
        specialise(store.load(BALANCE), TenantProfile(tenant="x", app="loans"))


@pytest.fixture
def tenant_b_bank(bank_url: str) -> Any:
    bank.TENANT = "tenant_b"
    yield bank_url
    bank.TENANT = "tenant_a"


def _replay(bank_url: str, tmp_path: Path, artifact: Path, inputs: dict[str, str], tenant: str | None = None,
            approval: Any = None) -> Any:
    from mm.artifact.tenancy import load_tenant, specialise
    from mm.policy.model import Policy
    cap = store.load(artifact)
    if tenant:
        cap = specialise(cap, load_tenant(tenant, ROOT / "tenants"))
    rec = RunRecorder(tmp_path, "replay", SECRETS)
    try:
        return replay(cap, inputs, base_url=bank_url, secrets=SECRETS, recorder=rec,
                      policy=Policy.load().bind(bank_url), approval=approval,
                      surface_factory=lambda _: WebSurface(headless=True))
    finally:
        rec.close()


def test_base_capability_on_tenant_b_fails_cleanly_not_wrongly(tenant_b_bank: str, tmp_path: Path) -> None:
    # Tenant B renamed "Member Search" and swapped the menu order: the recorded CSS path now points at a
    # *different* link ("Dashboard"). A positional locator may find an element, but the action must not
    # proceed on an element whose on-screen name is not the one the step expects.
    result = _replay(tenant_b_bank, tmp_path, BALANCE, {"member_id": "10871"})
    assert isinstance(result, ReplayFailure) and result.kind is FailureKind.TARGET_NOT_FOUND
    assert result.step_id == "s05_click_member_search"
    assert "Dashboard" in (result.observed or "")  # the refused candidate is named in the evidence


def test_tenant_profile_makes_the_same_capability_work_on_tenant_b(tenant_b_bank: str, tmp_path: Path) -> None:
    result = _replay(tenant_b_bank, tmp_path, BALANCE, {"member_id": "10871"}, tenant="tenant_b")
    assert isinstance(result, ReplaySuccess), result
    assert result.outputs == {"savings_balance": "15320.00"} and result.drift == []


def test_approvals_are_per_tenant(tenant_b_bank: str, tmp_path: Path) -> None:
    from mm.artifact.tenancy import load_tenant, specialise
    inputs = {"member_id": "10871", "account_type": "Holiday Club", "deposit": "25.00", "nickname": "Fund"}
    base_approval = approval_for(store.load(OPEN))
    blocked = _replay(tenant_b_bank, tmp_path, OPEN, inputs, tenant="tenant_b", approval=base_approval)
    assert isinstance(blocked, ReplayFailure) and blocked.kind is FailureKind.POLICY_BLOCKED
    tenant_cap = specialise(store.load(OPEN), load_tenant("tenant_b", ROOT / "tenants"))
    ok = _replay(tenant_b_bank, tmp_path, OPEN, inputs, tenant="tenant_b", approval=approval_for(tenant_cap))
    assert isinstance(ok, ReplaySuccess), ok


def test_cli_tenant_replay_and_per_tenant_approval(tenant_b_bank: str, tmp_path: Path) -> None:
    import json
    import os
    import subprocess
    import sys
    artifact = tmp_path / OPEN.name
    artifact.write_bytes(OPEN.read_bytes())
    env = {**os.environ, "MM_RUNS_DIR": str(tmp_path / "runs")}

    def mm(*args: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run([sys.executable, "-m", "mm.cli", *args], capture_output=True, text=True, env=env,
                              cwd=ROOT)

    balance = json.loads(mm("replay", str(BALANCE), "-p", "member_id=10871", "--base-url", tenant_b_bank,
                            "--tenant", "tenant_b", "--headless").stdout)
    assert balance["status"] == "success" and balance["outputs"] == {"savings_balance": "15320.00"}
    params = ["-p", "member_id=10871", "-p", "account_type=Holiday Club", "-p", "deposit=25", "-p", "nickname=Fund"]
    assert mm("approve", str(artifact), "--by", "reviewer").returncode == 0  # the base approval...
    blocked = json.loads(mm("replay", str(artifact), *params, "--base-url", tenant_b_bank, "--tenant", "tenant_b",
                            "--headless").stdout)
    assert blocked["kind"] == "POLICY_BLOCKED"  # ...does not cover tenant_b
    assert mm("approve", str(artifact), "--by", "reviewer", "--tenant", "tenant_b").returncode == 0
    assert (tmp_path / f"{artifact.stem}.tenant_b.approval.yaml").exists()
    ok = json.loads(mm("replay", str(artifact), *params, "--base-url", tenant_b_bank, "--tenant", "tenant_b",
                       "--headless").stdout)
    assert ok["status"] == "success" and ok["approved_by"] == "reviewer"
