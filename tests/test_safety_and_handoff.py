"""Safety guardrails (policy, guard, network enforcement, approvals, redaction) and the human
handoff (lease, interventions, console, live takeover of the same session)."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from mm.agent.loop import discover
from mm.agent.prompts import user_message
from mm.artifact import store
from mm.artifact.approval import approve, load_valid
from mm.cli import web_surface_factory
from mm.evidence.recorder import RunRecorder
from mm.handoff.console import Console, create_app
from mm.handoff.intervention import HandoffController, HandoffError, Kind, Status
from mm.handoff.lease import ControlLease, ControlNotHeld, Owner
from mm.handoff.operator import SimulatedOperator
from mm.llm.router import LLMCall
from mm.policy.guard import GuardedSurface
from mm.policy.model import Policy
from mm.redact import mask_pii
from mm.replay.executor import replay
from mm.replay.result import FailureKind, ReplayBusinessOutcome, ReplayFailure, ReplayResult, ReplaySuccess
from mm.surface.base import ActionType, ActResult, ElementRef, FrameText, Observation, RoleStrategy, Target
from mm.values import SecretStore
from mock_bank import app as bank
from tests.fakes import PERMISSIVE, FakeState, FakeSurface, approval_for

ROOT = Path(__file__).resolve().parent.parent
POLICY = Policy.load(ROOT / "config" / "policy.yaml")
SECRETS = SecretStore({"MOCKBANK_USERNAME": "operator1", "MOCKBANK_PASSWORD": "change-me-local-only"})
BALANCE = store.latest_path("corebank.member.get_savings_balance", ROOT / "capabilities")
OPEN = store.latest_path("corebank.member.open_sub_account", ROOT / "capabilities")
OPEN_INPUTS = {"member_id": "10871", "account_type": "Holiday Club", "deposit": "25.00", "nickname": "Fund"}


def _accounts(member: str = "10871") -> int:
    return len(bank.data.MEMBERS[member].accounts)


def _replay(bank_url: str, tmp_path: Path, artifact: Path, inputs: dict[str, str], *, approved: bool = True,
            handoff: bool = False, operator: dict[str, Any] | None = None, timeout_s: float = 60,
            policy: Policy = POLICY) -> tuple[ReplayResult, RunRecorder]:
    """Replay the way the CLI does (same surface factory, bound policy, real console when handing off)."""
    policy = policy.bind(bank_url) if "{base_url}" in policy.network.allow_origins else policy
    rec = RunRecorder(tmp_path, "replay", SECRETS)
    cap = store.load(artifact)
    controller = HandoffController(rec) if handoff else None
    sim = None
    port = _free_port()
    if controller is not None and operator is not None:
        sim = SimulatedOperator(f"http://127.0.0.1:{port}", page=None, **operator)  # type: ignore[arg-type]
        controller.operator_tick = sim.tick
    factory = web_surface_factory(True, False, policy, controller, sim, SECRETS)
    try:
        if controller is None:
            return replay(cap, inputs, base_url=bank_url, secrets=SECRETS, recorder=rec, policy=policy,
                          approval=approval_for(cap) if approved else None, surface_factory=factory), rec
        with Console(controller, port) as console:
            if sim is not None:
                sim.token = console.token
            return replay(cap, inputs, base_url=bank_url, secrets=SECRETS, recorder=rec, policy=policy,
                          approval=approval_for(cap) if approved else None, handoff=controller,
                          escalation_timeout_s=timeout_s, surface_factory=factory), rec
    finally:
        rec.close()


def _events(rec: RunRecorder) -> list[dict[str, Any]]:
    return [json.loads(line) for line in (rec.dir / "events.jsonl").read_text().splitlines()]


def _free_port() -> int:
    import socket
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


# --- policy --------------------------------------------------------------------------------------

def test_policy_allowlist_and_risk_rules() -> None:
    policy = POLICY.bind("http://127.0.0.1:8600")
    assert policy.url_allowed("http://127.0.0.1:8600/core/main.jsp")[0]
    assert not policy.url_allowed("https://evil.example/core/main.jsp")[0]
    assert not policy.url_allowed("http://127.0.0.1:8600/__control/faults")[0]
    assert not policy.url_allowed("http://127.0.0.1:8600/logout")[0]
    assert POLICY.classify_control(ActionType.CLICK, "Confirm") == "irreversible"
    assert POLICY.classify_control(ActionType.CLICK, "Continue") == "mutating"  # a form submit
    assert POLICY.classify_control(ActionType.EXTRACT, "Confirm") == "safe"  # reading a label commits nothing
    assert POLICY.is_irreversible_request("POST", "http://h/core/opensub_confirm.jsp")
    assert not POLICY.is_irreversible_request("GET", "http://h/core/opensub_confirm.jsp")


# --- guard (browser-free) ------------------------------------------------------------------------

def _target(name: str) -> Target:
    return Target(strategies=[RoleStrategy(role="button", name=name)], description=name)


def test_guard_refuses_to_act_while_a_human_holds_the_session() -> None:
    lease = ControlLease()
    state = FakeState(present={"Go"})
    guard = GuardedSurface(FakeSurface(state, {}), PERMISSIVE, lease)
    lease.to_human("alice", "test")
    res = guard.perform(ActionType.CLICK, _target("Go"), None, 1000)
    assert (res.ok, res.error) == (False, "control_not_held") and state.performed == []
    lease.to_agent("back")
    # Control is back, but automation has not re-verified yet: its old epoch is still refused.
    assert guard.perform(ActionType.CLICK, _target("Go"), None, 1000).error == "control_not_held"
    guard.resync()
    assert guard.perform(ActionType.CLICK, _target("Go"), None, 1000).ok


def test_guard_enforces_action_types_and_irreversible_authorisation() -> None:
    policy = POLICY.model_copy(update={"actions": {"allowed": [ActionType.CLICK]}})
    state = FakeState(present={"Go", "Confirm"})
    guard = GuardedSurface(FakeSurface(state, {}), policy, ControlLease())
    assert guard.perform(ActionType.FILL, _target("Go"), "x", 1000).error == "policy_blocked"
    assert guard.perform(ActionType.CLICK, _target("Confirm"), None, 1000).error == "policy_blocked"
    assert "Confirm" not in state.performed
    with guard.irreversible_authorized():
        assert guard.perform(ActionType.CLICK, _target("Confirm"), None, 1000).ok
    assert guard.perform(ActionType.CLICK, _target("Confirm"), None, 1000).error == "policy_blocked"  # one-shot


# --- network enforcement (live browser) ----------------------------------------------------------

def test_every_request_is_filtered_by_policy_in_the_browser(bank_url: str, tmp_path: Path) -> None:
    rec = RunRecorder(tmp_path, "t", SECRETS)
    policy = POLICY.bind(bank_url)
    surface = web_surface_factory(True, False, policy)(rec)
    guard = GuardedSurface(surface, policy, ControlLease())
    try:
        guard.navigate(f"{bank_url}/login")
        # A script on the page (or a compromised one) calling the control plane never reaches the server.
        blocked = surface.page.evaluate(f"fetch('{bank_url}/__control/faults').then(() => 'sent', () => 'blocked')")
        assert blocked == "blocked"
        assert any("/__control/faults" in b for b in surface.drain_blocked_requests())
        # Navigating to a denied path is refused before the browser is touched.
        res = guard.perform(ActionType.NAVIGATE, None, f"{bank_url}/logout", 1000)
        assert res.error == "policy_blocked" and res.dispatched is None
        with pytest.raises(Exception, match="not on the allowlist"):
            guard.navigate("https://example.com/")
    finally:
        surface.close()
        rec.close()


def test_a_click_whose_request_is_blocked_is_not_reported_as_ok(bank_url: str, tmp_path: Path) -> None:
    rec = RunRecorder(tmp_path, "t", SECRETS)
    policy = POLICY.bind(bank_url)
    surface = web_surface_factory(True, False, policy)(rec)
    guard = GuardedSurface(surface, policy, ControlLease())
    try:
        guard.navigate(f"{bank_url}/login")
        surface.page.set_content(f'<a href="{bank_url}/logout">Sign Off</a>')
        res = guard.perform(ActionType.CLICK, Target(strategies=[RoleStrategy(role="link", name="Sign Off")]), None,
                            2000)
        assert (res.ok, res.error, res.dispatched) == (False, "policy_blocked", True)
    finally:
        surface.close()
        rec.close()


# --- approvals -----------------------------------------------------------------------------------

def test_unapproved_irreversible_step_is_blocked_and_nothing_commits(bank_url: str, tmp_path: Path) -> None:
    result, _ = _replay(bank_url, tmp_path, OPEN, OPEN_INPUTS, approved=False)
    assert isinstance(result, ReplayFailure) and result.kind is FailureKind.POLICY_BLOCKED
    assert result.step_id and "confirm" in result.step_id and _accounts() == 2


def test_approved_capability_commits_and_reports_the_approver(bank_url: str, tmp_path: Path) -> None:
    result, rec = _replay(bank_url, tmp_path, OPEN, OPEN_INPUTS)
    assert isinstance(result, ReplaySuccess) and result.approved_by == "test-reviewer" and _accounts() == 3
    assert any(e["type"] == "irreversible_authorized" for e in _events(rec))


def test_approval_is_bound_to_the_artifact_bytes(tmp_path: Path) -> None:
    copy = tmp_path / OPEN.name
    copy.write_bytes(OPEN.read_bytes())
    assert load_valid(copy, store.load(copy)) == (None, "capability has no approval")
    approve(copy, store.load(copy), by="alice", note="reviewed")
    approval, _ = load_valid(copy, store.load(copy))
    assert approval is not None and approval.approved_by == "alice"
    assert approval.irreversible_steps == [s.id for s in store.load(copy).steps if s.risk == "irreversible"]
    copy.write_text(copy.read_text().replace("timeout_ms: 10000", "timeout_ms: 10001", 1))
    verdict = load_valid(copy, store.load(copy))
    assert verdict[0] is None and "changed" in verdict[1]  # content changed


def test_commit_requests_are_blocked_even_if_the_control_is_misclassified(bank_url: str, tmp_path: Path) -> None:
    # Defence in depth: with a policy that does not recognise "Confirm" as irreversible, the artifact's own risk
    # level is removed as well; the network layer still refuses the commit POST outside an authorised window.
    lax = POLICY.bind(bank_url).model_copy(deep=True)
    lax.risk["irreversible"].control_names = r"(?!)"
    copy = tmp_path / OPEN.name
    copy.write_text(OPEN.read_text().replace("risk: irreversible", "risk: safe"))
    result, _ = _replay(bank_url, tmp_path, copy, OPEN_INPUTS, policy=lax)
    assert isinstance(result, ReplayFailure) and result.kind is FailureKind.POLICY_BLOCKED
    assert "irreversible request outside an authorised step" in (result.observed or "")
    # The commit was blocked in the browser and never reached the server: nothing committed. The result still
    # says may_have_committed, correctly: the "Continue" submit (a mutating step) was dispatched before it.
    assert result.may_have_committed is True and _accounts() == 2


# --- redaction -----------------------------------------------------------------------------------

def test_mask_pii_patterns() -> None:
    text = "Balance $2,450.17 SSN 123-45-6789 acct 4111111111111111 mail a.b@x.io tel 555-123-4567 member 10234"
    masked = mask_pii(text)
    for raw in ("2,450.17", "123-45-6789", "4111111111111111", "a.b@x.io", "555-123-4567"):
        assert raw not in masked
    assert "member 10234" in masked  # identifiers the task needs are not treated as PII


def test_the_model_never_sees_regulated_values() -> None:
    obs = Observation(url="u", title="t", frames=[FrameText(frame_path=["main"], url="u",
                                                            text="Share Savings $2,450.17\nSSN 123-45-6789")],
                      elements=[ElementRef(ref="e1", role="cell", name="$2,450.17", frame_path=["main"])])
    prompt = user_message("read the balance", {}, [], [], [], obs)
    assert "2,450.17" not in prompt and "123-45-6789" not in prompt
    assert "[e1] cell" in prompt  # it can still point at the value to extract it


def test_screenshots_mask_pii_and_password_fields(bank_url: str, tmp_path: Path, monkeypatch: Any) -> None:
    rec = RunRecorder(tmp_path, "t", SECRETS)
    surface = web_surface_factory(True, False, POLICY.bind(bank_url))(rec)
    seen: dict[str, Any] = {}
    try:
        surface.navigate(f"{bank_url}/login")
        original = surface.page.screenshot
        monkeypatch.setattr(surface.page, "screenshot", lambda **kw: seen.update(kw) or original(**kw))
        surface.screenshot(str(tmp_path / "s.png"))
        assert seen["mask"] and any("password" in str(m) for m in seen["mask"])
        assert any(re.search("get_by_text|internal:text", str(m)) for m in seen["mask"])
    finally:
        surface.close()
        rec.close()


# --- lease and intervention state machine --------------------------------------------------------

def test_lease_transfers_bump_the_epoch() -> None:
    lease = ControlLease()
    assert (lease.state.owner, lease.state.epoch) == (Owner.AGENT, 1)
    lease.to_human("alice", "stuck")
    with pytest.raises(ControlNotHeld):
        lease.require_agent(1)
    with pytest.raises(ControlNotHeld):
        lease.to_human("bob", "second claimant")  # one holder at a time
    assert lease.to_agent("done").epoch == 3
    assert lease.end("over").owner is Owner.NONE


def test_intervention_state_machine(tmp_path: Path) -> None:
    rec = RunRecorder(tmp_path, "t", SECRETS)
    c = HandoffController(rec)
    approval = c.open(Kind.APPROVAL_REQUIRED, "cap", "irreversible click")
    with pytest.raises(HandoffError):
        c.claim(approval.id, "alice")  # approvals never transfer control
    assert c.approve(approval.id, "alice").status is Status.APPROVED
    with pytest.raises(HandoffError):
        c.reject(approval.id, "alice")  # already decided
    item = c.open(Kind.UNRECOVERABLE_STATE, "cap", "unknown dialog")
    c.record_human_action({"kind": "click", "name": "ignored"})  # agent holds the lease: not a human action
    c.claim(item.id, "alice")
    c.record_human_action({"kind": "change", "tag": "input", "name": "Nickname", "detail": "entered 4 characters"})
    resolved = c.resume(item.id, "alice", note="dismissed it")
    assert resolved.status is Status.RESOLVED and [a.name for a in resolved.human_actions] == ["Nickname"]
    assert c.lease.state.owner is Owner.AGENT and c.lease.state.epoch == 3
    rec.close()
    saved = json.loads((rec.dir / "interventions" / f"{item.id}.json").read_text())
    assert saved["status"] == "resolved" and saved["resolved_by"] == "alice"


def test_console_api_drives_the_lease(tmp_path: Path) -> None:
    rec = RunRecorder(tmp_path, "t", SECRETS)
    c = HandoffController(rec)
    item = c.open(Kind.UNRECOVERABLE_STATE, "cap", "unknown dialog", step_id="s09")
    client = TestClient(create_app(c, token="t0k", own_origin="http://testserver"),
                        headers={"X-Operator-Token": "t0k"})
    assert client.get("/").status_code == 200 and item.id in client.get("/").text
    assert client.post(f"/api/interventions/{item.id}/resume", json={"by": "x"}).status_code == 409  # not claimed
    assert client.post(f"/api/interventions/{item.id}/claim", json={"by": "alice"}).json()["status"] == "claimed"
    assert client.get("/api/state").json()["lease"]["owner"] == "HUMAN"
    assert client.post(f"/api/interventions/{item.id}/resume", json={"by": "alice"}).json()["status"] == "resolved"
    assert client.get("/api/state").json()["lease"]["owner"] == "AGENT"
    rec.close()


# --- live handoff during replay ------------------------------------------------------------------

def test_unknown_popup_is_handed_to_a_human_who_fixes_it_in_the_same_session(bank_url: str, tmp_path: Path) -> None:
    bank.FAULTS.add("survey")
    result, rec = _replay(bank_url, tmp_path, OPEN, OPEN_INPUTS, handoff=True, operator={"clicks": ["Maybe Later"]})
    assert isinstance(result, ReplaySuccess), result
    assert [(i.kind, i.status, i.claimed_by, i.human_actions) for i in result.interventions] == \
        [(Kind.UNRECOVERABLE_STATE, Status.RESOLVED, "simulated-operator", 1)]
    events = _events(rec)
    leases = [(e["owner"], e["epoch"]) for e in events if e["type"] == "lease"]
    assert leases[:2] == [("HUMAN", 2), ("AGENT", 3)]  # handed over, then handed back
    human = [e for e in events if e["type"] == "human_action"]
    assert human and human[0]["name"] == "Maybe Later"
    assert next(e for e in events if e["type"] == "handoff_resumed")["continue_from"] == "next step"
    assert _accounts() == 3


def test_operator_abort_stops_the_run_before_any_commit(bank_url: str, tmp_path: Path) -> None:
    bank.FAULTS.add("survey")
    result, _ = _replay(bank_url, tmp_path, OPEN, OPEN_INPUTS, handoff=True, operator={"abort": True})
    assert isinstance(result, ReplayFailure) and result.kind is FailureKind.ESCALATION_ABORTED
    assert result.interventions[0].status is Status.ABORTED and _accounts() == 2


def test_nobody_answering_is_a_timeout_not_a_hang(bank_url: str, tmp_path: Path) -> None:
    bank.FAULTS.add("survey")
    result, _ = _replay(bank_url, tmp_path, OPEN, OPEN_INPUTS, handoff=True, timeout_s=1)
    assert isinstance(result, ReplayFailure) and result.kind is FailureKind.ESCALATION_TIMEOUT
    assert result.interventions[0].status is Status.EXPIRED and _accounts() == 2


def test_business_outcomes_are_answers_and_never_escalate(bank_url: str, tmp_path: Path) -> None:
    result, _ = _replay(bank_url, tmp_path, BALANCE, {"member_id": "99999"}, handoff=True,
                        operator={"clicks": []})
    assert isinstance(result, ReplayBusinessOutcome) and result.interventions == []


def test_human_actions_never_capture_typed_values(bank_url: str, tmp_path: Path) -> None:
    rec = RunRecorder(tmp_path, "t", SECRETS)
    c = HandoffController(rec)
    surface = web_surface_factory(True, False, POLICY.bind(bank_url), c)(rec)
    try:
        surface.navigate(f"{bank_url}/login")
        item = c.open(Kind.STUCK, "goal", "help")
        c.claim(item.id, "alice")
        surface.page.fill("input[name=username]", "secret-typed-text")
        surface.page.press("input[name=username]", "Tab")  # blur fires 'change'
        surface.idle(300)
        actions = c.get(item.id).human_actions
        assert actions and actions[0].detail == "entered 17 characters"
        assert "secret-typed-text" not in (rec.dir / "events.jsonl").read_text()
    finally:
        surface.close()
        rec.close()


# --- discovery: approval before an irreversible click, and takeover when stuck --------------------

class _Script:
    """A router that returns scripted decisions (the model is not what is under test here)."""

    def __init__(self, decisions: list[dict[str, Any]]) -> None:
        self.decisions = decisions

    def structured(self, model: Any, messages: Any, max_retries: int = 2) -> Any:
        return LLMCall(model(**self.decisions.pop(0)), "scripted", "m", 0, 0)


class _ConfirmPage(FakeSurface):
    """One screen with an irreversible "Confirm" button, as the discovery agent would observe it."""

    def observe(self, with_screenshot: bool = False) -> Observation:
        return Observation(url="http://a/review", title="Review",
                           elements=[ElementRef(ref="e1", role="button", name="Confirm")])

    def act(self, action: ActionType, ref: str | None, value: str | None = None) -> ActResult:
        self.state.performed.append(f"{action}:{ref}:{getattr(self, 'window', None)}")
        return ActResult(ok=True, target=_target("Confirm"), frame_urls={"": "http://a/done"})


def _discover_confirm(tmp_path: Path, decision: str) -> tuple[Any, FakeState, HandoffController]:
    rec = RunRecorder(tmp_path, "discover", SECRETS)
    c = HandoffController(rec)
    c.operator_tick = lambda item: getattr(c, decision)(item.id, "alice")  # an explicit decision, never a default
    state = FakeState()
    decisions = [{"thought": "commit", "action": "click", "ref": "e1"}]
    decisions += [{"thought": "done", "action": "done", "summary": "ok"}] if decision == "approve" else \
        [{"thought": "give up", "action": "fail", "summary": "not approved"}]
    result = discover(goal="open an account", entry_url="http://a/login", inputs={}, required_outputs=[],
                      surface=_ConfirmPage(state, {}), router=_Script(decisions), secrets=SECRETS,  # type: ignore[arg-type]
                      recorder=rec, policy=POLICY.model_copy(update={"network": PERMISSIVE.network}), handoff=c)
    rec.close()
    return result, state, c


def test_discovery_pauses_for_approval_before_an_irreversible_click(tmp_path: Path) -> None:
    result, state, c = _discover_confirm(tmp_path, "approve")
    assert result.status == "success" and result.approvals == ['button "Confirm" approved by alice']
    assert [a for a in state.performed if a.startswith("click")] == ["click:e1:None"]
    assert c.all()[0].kind is Kind.APPROVAL_REQUIRED and c.all()[0].status is Status.APPROVED


def test_discovery_does_not_click_what_the_operator_rejected(tmp_path: Path) -> None:
    result, state, _ = _discover_confirm(tmp_path, "reject")
    assert result.status == "failed" and not [a for a in state.performed if a.startswith("click")]


def test_a_stuck_agent_hands_the_session_to_a_human_and_carries_on(tmp_path: Path) -> None:
    rec = RunRecorder(tmp_path, "discover", SECRETS)
    c = HandoffController(rec)

    def operator(item: Any) -> None:
        if item.status is Status.OPEN:
            c.claim(item.id, "alice")
            c.record_human_action({"kind": "click", "tag": "button", "name": "Unlock"})
            c.resume(item.id, "alice")

    c.operator_tick = operator
    router = _Script([{"thought": "stuck", "action": "request_human", "summary": "account locked"},
                      {"thought": "done", "action": "done", "summary": "ok"}])
    result = discover(goal="g", entry_url="http://a/login", inputs={}, required_outputs=[],
                      surface=FakeSurface(FakeState(), {}), router=router, secrets=SECRETS,  # type: ignore[arg-type]
                      recorder=rec, policy=PERMISSIVE, handoff=c)
    rec.close()
    assert result.status == "success" and result.human_took_control
    # ...and a run a human had to finish is not compiled into a capability that would silently miss their steps.
    from mm.artifact.compiler import CompileError, compile_run
    with pytest.raises(CompileError, match="human took control"):
        compile_run(result, "app.t.x", pack="none", policy=PERMISSIVE)
