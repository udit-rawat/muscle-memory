"""Regression tests for the Phase 3 audit (P1-P17, plus the test-bias gaps T*). Each was written to fail on
the pre-fix code."""

from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

import httpx
import pytest

from mm.agent.loop import discover
from mm.artifact import store
from mm.evidence.recorder import RunRecorder
from mm.handoff.intervention import HandoffController, Kind, Status
from mm.handoff.lease import ControlLease, ControlNotHeld
from mm.llm.router import LLMCall
from mm.policy.guard import GuardedSurface
from mm.policy.model import Policy
from mm.replay.executor import replay
from mm.replay.result import FailureKind, ReplayBusinessOutcome, ReplayFailure, ReplayResult, ReplaySuccess
from mm.surface.base import (
    ActionType,
    ActResult,
    AttrStrategy,
    ElementRef,
    FrameText,
    Observation,
    RoleStrategy,
    TableCellStrategy,
    Target,
)
from mm.surface.web import WebSurface
from mm.values import SecretStore
from mock_bank import app as bank
from tests.fakes import PERMISSIVE, FakeState, FakeSurface

ROOT = Path(__file__).resolve().parent.parent
RAW_POLICY = ROOT / "config" / "policy.yaml"
SECRETS = SecretStore({"MOCKBANK_USERNAME": "operator1", "MOCKBANK_PASSWORD": "change-me-local-only"})
OPEN = store.latest_path("corebank.member.open_sub_account", ROOT / "capabilities")
BALANCE = store.latest_path("corebank.member.get_savings_balance", ROOT / "capabilities")
OPEN_INPUTS = {"member_id": "10871", "account_type": "Holiday Club", "deposit": "25.00", "nickname": "Fund"}


def _policy(base_url: str) -> Policy:
    return Policy.load(RAW_POLICY).bind(base_url)


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def _t(name: str) -> Target:
    return Target(strategies=[RoleStrategy(role="button", name=name)], description=name)


class _Script:
    def __init__(self, decisions: list[dict[str, Any]]) -> None:
        self.decisions, self.sent = decisions, []  # type: ignore[var-annotated]

    def structured(self, model: Any, messages: Any, max_retries: int = 2) -> Any:
        self.sent.append(messages[1]["content"])
        return LLMCall(model(**self.decisions.pop(0)), "scripted", "m", 0, 0)


def _approved(cap_path: Path) -> Any:
    from mm.artifact.approval import approve
    tmp = Path(tempfile.mkdtemp()) / cap_path.name
    tmp.write_bytes(cap_path.read_bytes())
    return approve(tmp, store.load(tmp), by="reviewer")


def _live(bank_url: str, tmp_path: Path, artifact: Path, inputs: dict[str, str], approval: Any = None) -> ReplayResult:
    from mm.cli import web_surface_factory
    rec = RunRecorder(tmp_path, "replay", SECRETS)
    policy = _policy(bank_url)
    try:
        return replay(store.load(artifact), inputs, base_url=bank_url, secrets=SECRETS, recorder=rec, policy=policy,
                      approval=approval, surface_factory=web_surface_factory(True, False, policy, secrets=SECRETS))
    finally:
        rec.close()


# --- P1: the prompt the model receives is the prompt we log, and holds no regulated value ---------

class _BalancePage(FakeSurface):
    def observe(self, with_screenshot: bool = False) -> Observation:
        return Observation(url="http://a/d", title="Detail",
                           frames=[FrameText(frame_path=["main"], url="u", text="Share Savings $2,450.17")],
                           elements=[ElementRef(ref="e1", role="cell", name="$2,450.17", frame_path=["main"])])

    def act(self, action: ActionType, ref: str | None, value: str | None = None) -> ActResult:
        cell = TableCellStrategy(row_key="Share Savings", column_header="Balance")
        return ActResult(ok=True, extracted="$2,450.17", frame_urls={"": "http://a/d"},
                         target=Target(strategies=[cell]))


def test_p1_sent_prompt_equals_logged_prompt_and_has_no_pii(tmp_path: Path) -> None:
    rec = RunRecorder(tmp_path, "discover", SecretStore({}))
    router = _Script([{"thought": "read", "action": "extract", "ref": "e1", "output_name": "savings_balance"},
                      {"thought": "done", "action": "done", "summary": "balance is $2,450.17"}])
    discover(goal="read the balance", entry_url="http://a/login", inputs={}, required_outputs=["savings_balance"],
             surface=_BalancePage(FakeState(), {}), router=router, secrets=SecretStore({}),  # type: ignore[arg-type]
             recorder=rec, policy=PERMISSIVE, max_steps=3)
    rec.close()
    for i, sent in enumerate(router.sent, start=1):
        assert "2,450.17" not in sent, f"turn {i} sent a raw balance to the model"
        assert sent == (rec.dir / "prompts" / f"{i:02d}.txt").read_text(), "evidence differs from what was sent"


# --- P2: an approval only authorises the exact capability it was given for -----------------------

def test_p2_approval_for_another_capability_does_not_authorise(bank_url: str, tmp_path: Path) -> None:
    other = _approved(BALANCE).model_copy(update={"irreversible_steps": [s.id for s in store.load(OPEN).steps
                                                                         if s.risk == "irreversible"]})
    result = _live(bank_url, tmp_path, OPEN, OPEN_INPUTS, approval=other)
    assert isinstance(result, ReplayFailure) and result.kind is FailureKind.POLICY_BLOCKED
    assert len(bank.data.MEMBERS["10871"].accounts) == 2


def test_p2_approval_for_modified_content_does_not_authorise(bank_url: str, tmp_path: Path) -> None:
    approval = _approved(OPEN)
    changed = tmp_path / OPEN.name
    changed.write_text(OPEN.read_text().replace("timeout_ms: 10000", "timeout_ms: 10001", 1))
    result = _live(bank_url, tmp_path, changed, OPEN_INPUTS, approval=approval)
    assert isinstance(result, ReplayFailure) and result.kind is FailureKind.POLICY_BLOCKED


def test_t2_cli_approve_then_replay_then_tamper(bank_url: str, tmp_path: Path) -> None:
    """The real path end to end: `mm approve` writes the sidecar, `mm replay` honours it, a one-byte edit breaks it."""
    artifact = tmp_path / OPEN.name
    artifact.write_bytes(OPEN.read_bytes())
    env = {**os.environ, "MM_RUNS_DIR": str(tmp_path / "runs"), "MOCKBANK_PASSWORD": "change-me-local-only",
           "MOCKBANK_USERNAME": "operator1"}
    params = [a for k, v in OPEN_INPUTS.items() for a in ("-p", f"{k}={v}")]

    def mm(*args: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run([sys.executable, "-m", "mm.cli", *args], capture_output=True, text=True, env=env,
                              cwd=ROOT)

    assert json.loads(mm("replay", str(artifact), *params, "--base-url", bank_url, "--headless").stdout)["kind"] \
        == "POLICY_BLOCKED"
    assert mm("approve", str(artifact), "--by", "reviewer").returncode == 0
    ok = json.loads(mm("replay", str(artifact), *params, "--base-url", bank_url, "--headless").stdout)
    assert ok["status"] == "success" and ok["approved_by"] == "reviewer"
    artifact.write_text(artifact.read_text().replace("timeout_ms: 10000", "timeout_ms: 10001", 1))
    assert json.loads(mm("replay", str(artifact), *params, "--base-url", bank_url, "--headless").stdout)["kind"] \
        == "POLICY_BLOCKED"


# --- P3: the allowlist is the tenant's origin, never the console or other local services ---------

def test_p3_policy_is_bound_to_the_tenant_origin() -> None:
    policy = Policy.load(RAW_POLICY).bind("http://127.0.0.1:8600", deny_origins=["http://127.0.0.1:8700"])
    assert policy.url_allowed("http://127.0.0.1:8600/core/main.jsp")[0]
    for url in ("http://127.0.0.1:8700/i/iv-1/approve", "http://127.0.0.1:6379/", "http://localhost:9200/"):
        assert not policy.url_allowed(url)[0], url
    with pytest.raises(ValueError, match="console"):
        Policy.load(RAW_POLICY).bind("http://127.0.0.1:8700", deny_origins=["http://127.0.0.1:8700"])


# --- P4: the console needs this run's token and refuses cross-site posts --------------------------

def test_p4_console_requires_the_run_token_and_same_origin(tmp_path: Path) -> None:
    from mm.handoff.console import Console
    c = HandoffController(RunRecorder(tmp_path, "t", SECRETS))
    item = c.open(Kind.APPROVAL_REQUIRED, "cap", "click Confirm")
    with Console(c, _free_port()) as console:
        base = console.base
        assert httpx.post(f"{base}/i/{item.id}/approve", data={"by": "x"}).status_code == 403  # no token
        assert httpx.get(f"{base}/api/state").status_code == 403
        forged = httpx.post(f"{base}/i/{item.id}/approve", data={"by": "x", "token": console.token},
                            headers={"Origin": "http://evil.example"})
        assert forged.status_code == 403  # a cross-site form cannot act even if it guessed the token
        assert c.get(item.id).status is Status.OPEN
        ok = httpx.post(f"{base}/api/interventions/{item.id}/approve", json={"by": "alice"},
                        headers={"X-Operator-Token": console.token})
        assert ok.status_code == 200 and c.get(item.id).status is Status.APPROVED


# --- P5: any takeover counts as human control, captured or not ------------------------------------

def test_p5_takeover_without_captured_actions_still_blocks_compilation(tmp_path: Path) -> None:
    from mm.artifact.compiler import CompileError, compile_run
    rec = RunRecorder(tmp_path, "discover", SECRETS)
    c = HandoffController(rec)

    def keyboard_only_operator(item: Any) -> None:  # e.g. pressed Enter or used the URL bar: nothing captured
        if item.status is Status.OPEN:
            c.claim(item.id, "alice")
            c.resume(item.id, "alice")
    c.operator_tick = keyboard_only_operator
    router = _Script([{"thought": "stuck", "action": "request_human", "summary": "help"},
                      {"thought": "ok", "action": "done", "summary": "ok"}])
    result = discover(goal="g", entry_url="http://a/login", inputs={}, required_outputs=[],
                      surface=FakeSurface(FakeState(), {}), router=router, secrets=SECRETS,  # type: ignore[arg-type]
                      recorder=rec, policy=PERMISSIVE, handoff=c)
    rec.close()
    assert result.human_took_control
    with pytest.raises(CompileError):
        compile_run(result, "app.t.x", pack="none", policy=PERMISSIVE)


# --- P6: a request blocked in the background is logged, not blamed on the next action -------------

def test_p6_background_block_is_not_attributed_to_the_next_action(bank_url: str, tmp_path: Path) -> None:
    from mm.cli import web_surface_factory
    rec = RunRecorder(tmp_path, "t", SECRETS)
    policy = _policy(bank_url)
    surface = web_surface_factory(True, False, policy)(rec)
    guard = GuardedSurface(surface, policy, ControlLease(), on_event=rec.event)
    try:
        guard.navigate(f"{bank_url}/login")
        surface.page.evaluate(f"fetch('{bank_url}/__control/faults').catch(() => 0)")
        surface.idle(300)
        res = guard.perform(ActionType.FILL, Target(strategies=[AttrStrategy(tag="input", name="username")]),
                            "operator1", 2000)
        assert res.ok, res.detail
    finally:
        surface.close()
        rec.close()
    assert "background_requests_blocked" in (rec.dir / "events.jsonl").read_text()


# --- P7: a console that cannot serve is an error, not a silent hang -------------------------------

def test_p7_console_on_a_busy_port_fails_fast(tmp_path: Path) -> None:
    from mm.handoff.console import Console, ConsoleUnavailable
    c = HandoffController(RunRecorder(tmp_path, "t", SECRETS))
    squatter = socket.socket()
    squatter.bind(("127.0.0.1", 0))
    squatter.listen()
    try:
        with pytest.raises(ConsoleUnavailable), Console(c, squatter.getsockname()[1]):
            pass
    finally:
        squatter.close()


# --- P8: discovery reports errors as a status ------------------------------------------------------

def test_p8_discovery_error_is_a_status_not_a_traceback(tmp_path: Path) -> None:
    class Crash(FakeSurface):
        def observe(self, with_screenshot: bool = False) -> Observation:
            raise RuntimeError("Target page, context or browser has been closed")
    rec = RunRecorder(tmp_path, "d", SECRETS)
    result = discover(goal="g", entry_url="http://a/login", inputs={}, required_outputs=[],
                      surface=Crash(FakeState(), {}), router=_Script([]), secrets=SECRETS,  # type: ignore[arg-type]
                      recorder=rec, policy=PERMISSIVE)
    rec.close()
    assert result.status == "error" and "browser has been closed" in result.summary


# --- P9: the simulated operator never approves unless told to --------------------------------------

def test_p9_simulated_operator_rejects_by_default() -> None:
    from mm.handoff.operator import SimulatedOperator
    assert SimulatedOperator("http://x", page=None).decision == "reject"  # type: ignore[arg-type]


# --- P10: form submits are "mutating": no approval needed, but never repeated ----------------------

def test_p10_three_risk_levels() -> None:
    policy = Policy.load(RAW_POLICY)
    assert policy.classify_control(ActionType.CLICK, "Continue") == "mutating"
    assert policy.classify_control(ActionType.CLICK, "Confirm") == "irreversible"
    assert policy.classify_control(ActionType.CLICK, "Search") == "safe"
    assert policy.classify_control(ActionType.CLICK, "Add Share Account") == "safe"  # just opens a form


def test_p10_expiry_right_after_a_submit_is_not_replayed_blindly(bank_url: str, tmp_path: Path) -> None:
    bank.FAULTS.add("session_expired_on_submit")
    result = _live(bank_url, tmp_path, OPEN, OPEN_INPUTS, approval=_approved(OPEN))
    assert isinstance(result, ReplayFailure) and result.kind is FailureKind.UNSAFE_TO_REPEAT
    assert result.may_have_committed and len(bank.data.MEMBERS["10871"].accounts) == 2


def test_p10_expiry_before_any_submit_still_restarts(bank_url: str, tmp_path: Path) -> None:
    bank.FAULTS.add("session_expired_on_detail")
    result = _live(bank_url, tmp_path, OPEN, OPEN_INPUTS, approval=_approved(OPEN))
    assert isinstance(result, ReplaySuccess), result
    assert [r.action for r in result.recoveries] == ["restarted"]


# --- P11: evidence screenshots hide credentials too -----------------------------------------------

def test_p11_screenshots_mask_the_operator_identity(bank_url: str, tmp_path: Path) -> None:
    surface = WebSurface(headless=True, mask_texts=SECRETS.values())
    try:
        surface.navigate(f"{bank_url}/login")
        surface.page.fill("input[name=username]", "operator1")
        masks = surface.screenshot_masks()
        assert any("operator1" in str(m) for m in masks), "banner/login text with the operator id is not masked"
        assert any("nth=0" in str(m) or "input" in str(m) for m in masks), "the typed operator id is not masked"
    finally:
        surface.close()


def _black_fraction(surface: WebSurface, png: bytes) -> float:
    """Share of (near-)black pixels in a PNG, decoded by the browser's own canvas (no image library needed)."""
    import base64
    page = surface.page.context.new_page()
    try:
        return float(page.evaluate("""async (src) => {
            const img = new Image(); img.src = src; await img.decode();
            const c = document.createElement('canvas'); c.width = img.width; c.height = img.height;
            const ctx = c.getContext('2d'); ctx.drawImage(img, 0, 0);
            const d = ctx.getImageData(0, 0, c.width, c.height).data; let black = 0;
            for (let i = 0; i < d.length; i += 4) if (d[i] < 20 && d[i + 1] < 20 && d[i + 2] < 20) black++;
            return black / (d.length / 4);
        }""", "data:image/png;base64," + base64.b64encode(png).decode()))
    finally:
        page.close()


def test_t4_masked_screenshots_hide_different_balances_identically(bank_url: str, tmp_path: Path) -> None:
    """Pixel-level: masked, the balance cell is (almost) entirely black for every member; unmasked it is not."""
    surface = WebSurface(headless=True, mask_texts=SECRETS.values())
    try:
        surface.navigate(f"{bank_url}/login")
        p = surface.page
        p.fill("input[name=username]", "operator1")
        p.fill("input[name=password]", "change-me-local-only")
        p.click("input[type=submit]")
        surface.idle(500)
        shots = {}
        for member in ("10234", "10871"):
            p.frame(name="main").goto(f"{bank_url}/core/mbrdtl.jsp?m={member}")
            surface.idle(300)
            cell = p.frame(name="main").locator("xpath=//tr[td[normalize-space(.)='Share Savings']]/td[4]")
            box = cell.bounding_box()
            assert box is not None
            shots[member] = (_black_fraction(surface, surface.masked_png(clip=box)),
                             _black_fraction(surface, p.screenshot(clip=box)))
        for member, (masked, unmasked) in shots.items():
            assert masked > 0.95, f"member {member}: only {masked:.0%} of the balance cell is masked"
            assert unmasked < 0.5, f"member {member}: the unmasked cell should show the digits"
    finally:
        surface.close()


# --- P12: intervention records are masked like everything else ------------------------------------

def test_p12_intervention_fields_are_pii_masked(tmp_path: Path) -> None:
    rec = RunRecorder(tmp_path, "t", SECRETS)
    c = HandoffController(rec)
    item = c.open(Kind.UNRECOVERABLE_STATE, "cap", "balance $2,450.17 did not load", observed="SSN 123-45-6789 shown")
    rec.close()
    saved = (rec.dir / "interventions" / f"{item.id}.json").read_text()
    assert "2,450.17" not in saved and "123-45-6789" not in saved


# --- P13: control only returns to automation from a human ------------------------------------------

def test_p13_lease_returns_to_agent_only_from_human() -> None:
    lease = ControlLease()
    with pytest.raises(ControlNotHeld):
        lease.to_agent("nobody handed it over")
    assert lease.state.epoch == 1


# --- P14: the CLI works from any directory -----------------------------------------------------------

def test_p14_cli_runs_from_another_directory(tmp_path: Path) -> None:
    out = subprocess.run([sys.executable, "-m", "mm.cli", "replay", str(BALANCE), "-p", "member_id=12ab",
                          "--headless"], capture_output=True, text=True, cwd=tmp_path,
                         env={**os.environ, "MM_RUNS_DIR": str(tmp_path / "runs")})
    assert json.loads(out.stdout)["kind"] == "INPUT_INVALID", out.stderr[-500:]


# --- P16: after a handoff, detectors look at the screen before automation continues -----------------

def test_p16_business_outcome_after_a_handoff_is_recognised(tmp_path: Path) -> None:
    from mm.artifact.schema import Capability
    from tests.fakes import approval_for
    cap = Capability.model_validate({
        "id": "app.t.flow", "summary": "s", "app": {"name": "a", "entry_path": "/"},
        "outcomes": ["MEMBER_NOT_FOUND"],
        "steps": [{"id": "s1", "intent": "search", "action": "click",
                   "target": {"strategies": [{"by": "role", "role": "button", "name": "Search"}],
                              "description": "Search"},
                   "expect": [{"kind": "url_matches", "pattern": "/results"}]}],
        "detectors": [{"id": "member_not_found", "class": "business_outcome", "code": "MEMBER_NOT_FOUND",
                       "after_steps": [], "when": [{"kind": "text_visible", "any_frame": True,
                                                    "pattern": "No member matches"}]}],
    })
    state = FakeState(present={"Search"})

    def search(s: FakeState) -> ActResult:
        s.url, s.overlays = "/results", ["overlay in frame main: 'Quick Survey'"]
        return ActResult(ok=True, strategy_index=0)

    rec = RunRecorder(tmp_path, "replay", SECRETS)
    c = HandoffController(rec)

    def operator(item: Any) -> None:  # dismisses the popup; underneath, the search found nobody
        if item.status is Status.OPEN:
            c.claim(item.id, "alice")
            state.overlays, state.texts = [], {"No member matches the search criteria."}
            c.resume(item.id, "alice")
    c.operator_tick = operator
    # after_steps=[] keeps the detector inactive during normal waiting, so only the post-handoff check can see it
    cap.detectors[0].after_steps = None
    result = replay(cap, {}, base_url="http://a", secrets=SECRETS, recorder=rec, policy=PERMISSIVE,
                    approval=approval_for(cap), handoff=c,
                    surface_factory=lambda _: _OverlayFirst(state, {"Search": search}))
    rec.close()
    assert isinstance(result, ReplayBusinessOutcome) and result.code == "MEMBER_NOT_FOUND"


class _OverlayFirst(FakeSurface):
    """Detectors see nothing until the overlay is gone (it covers the page), like a real modal."""

    def check(self, cp: Any, timeout_ms: int) -> tuple[bool, str]:
        if self.state.overlays and cp.kind != "url_matches":
            return False, "covered by overlay"
        return super().check(cp, timeout_ms)


# --- P17 / T5 ---------------------------------------------------------------------------------------

def test_p17_no_operator_is_not_described_as_an_operator_rejection(tmp_path: Path) -> None:
    class ConfirmPage(FakeSurface):
        def observe(self, with_screenshot: bool = False) -> Observation:
            return Observation(url="http://a/r", title="Review", elements=[ElementRef(ref="e1", role="button",
                                                                                      name="Confirm")])
    rec = RunRecorder(tmp_path, "d", SECRETS)
    router = _Script([{"thought": "commit", "action": "click", "ref": "e1"},
                      {"thought": "stop", "action": "fail", "summary": "cannot commit"}])
    discover(goal="g", entry_url="http://a/login", inputs={}, required_outputs=[], surface=ConfirmPage(FakeState(), {}),
             router=router, secrets=SECRETS, recorder=rec,  # type: ignore[arg-type]
             policy=Policy.load(RAW_POLICY).model_copy(update={"network": PERMISSIVE.network}))
    rec.close()
    assert "by the operator" not in router.sent[1] and "no operator" in router.sent[1].lower()


def test_t5_escalate_without_simulation_forces_a_visible_browser(tmp_path: Path) -> None:
    from mm.cli import _handoff
    from mm.config import get_settings
    _, _, headless = _handoff(RunRecorder(tmp_path, "t", SECRETS), get_settings(), True, None, True)
    assert headless is False
