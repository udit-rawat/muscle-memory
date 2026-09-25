"""Regression tests for the Phase 1-2 audit. Each was written to fail on the pre-fix code.

C = critical, H = high, M = medium, L = low, B = a test-bias gap in the original suite.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import zipfile
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from mm.artifact import store
from mm.evidence.recorder import RunRecorder
from mm.policy.model import Policy
from mm.replay.executor import replay
from mm.replay.result import ReplayBusinessOutcome, ReplayFailure, ReplayResult, ReplaySuccess
from mm.surface.base import ActResult
from mm.surface.web import WebSurface
from mm.values import SecretStore
from mock_bank import app as bank
from tests.fakes import PERMISSIVE, FakeState, FakeSurface, approval_for

ROOT = Path(__file__).resolve().parent.parent
BALANCE = "corebank.member.get_savings_balance"
OPEN = "corebank.member.open_sub_account"
PASSWORD = "change-me-local-only"
SECRETS = SecretStore({"MOCKBANK_USERNAME": "operator1", "MOCKBANK_PASSWORD": PASSWORD})
POLICY = Policy.load(ROOT / "config" / "policy.yaml")
OPEN_INPUTS = {"account_type": "Holiday Club", "deposit": "25.00", "nickname": "Fund"}


def _latest(cap_id: str) -> Path:
    return store.latest_path(cap_id, ROOT / "capabilities")


def _replay(bank_url: str, tmp_path: Path, cap_id: str, **inputs: str) -> ReplayResult:
    rec = RunRecorder(tmp_path, "replay", SECRETS)
    try:
        cap = store.load(_latest(cap_id))
        return replay(cap, inputs, base_url=bank_url, secrets=SECRETS, recorder=rec, policy=POLICY,
                      approval=approval_for(cap), surface_factory=lambda _: WebSurface(headless=True))
    finally:
        rec.close()


def _cli_replay(bank_url: str, runs: Path, cap_id: str, **inputs: str) -> tuple[dict[str, Any], Path]:
    """Run the real CLI (production defaults) and return its JSON result and run directory."""
    params = [a for k, v in inputs.items() for a in ("-p", f"{k}={v}")]
    env = {**os.environ, "MM_RUNS_DIR": str(runs), "MOCKBANK_PASSWORD": PASSWORD, "MOCKBANK_USERNAME": "operator1"}
    out = subprocess.run([sys.executable, "-m", "mm.cli", "replay", str(_latest(cap_id)), *params,
                          "--base-url", bank_url, "--headless"], capture_output=True, text=True, env=env, cwd=ROOT)
    run_dir = sorted(runs.glob("replay-*"), key=lambda p: p.stat().st_mtime)[-1]
    return json.loads(out.stdout), run_dir


def _all_bytes(run_dir: Path) -> bytes:
    """Every byte a run persisted, including the contents of any archive (traces are zips)."""
    blob = b""
    for f in run_dir.rglob("*"):
        if f.is_file():
            blob += f.read_bytes()
            if zipfile.is_zipfile(f):
                with zipfile.ZipFile(f) as z:
                    blob += b"".join(z.read(n) for n in z.namelist())
    return blob


# --- C1: a read must never return a value from the wrong place ---------------------------------

def test_c1_member_without_savings_is_not_a_success(bank_url: str, tmp_path: Path) -> None:
    result = _replay(bank_url, tmp_path, BALANCE, member_id="13100")
    assert not isinstance(result, ReplaySuccess), f"returned {getattr(result, 'outputs', None)} for a member " \
                                                  "with no savings account"
    assert isinstance(result, ReplayFailure) and result.kind == "TARGET_NOT_FOUND"


def test_c1_schema_rejects_structural_locators_for_reads() -> None:
    from pydantic import ValidationError

    from mm.artifact.schema import Capability
    data = store.load(_latest(BALANCE)).model_dump(mode="json", by_alias=True)
    extract = next(s for s in data["steps"] if s["action"] == "extract")
    extract["target"]["strategies"].append({"by": "css", "selector": "td"})
    with pytest.raises(ValidationError, match="structural"):
        Capability.model_validate(data)


# --- C2: nothing persisted by a production run contains a secret or a raw output ----------------

@pytest.mark.parametrize("member_id, status", [("10234", "success"), ("99999", "business_outcome"),
                                               ("13100", "failure")])
def test_c2_production_runs_persist_no_secret_or_raw_value(bank_url: str, tmp_path: Path, member_id: str,
                                                           status: str) -> None:
    result, run_dir = _cli_replay(bank_url, tmp_path, BALANCE, member_id=member_id)
    assert result["status"] == status
    assert not (run_dir / "trace.zip").exists(), "tracing must be opt-in"
    blob = _all_bytes(run_dir)
    for leaked in (PASSWORD.encode(), b"2,450.17", b"2450.17", b"777.77"):
        assert leaked not in blob, f"{leaked!r} persisted in {run_dir}"


# --- C3: a step that may already have committed is never performed again ------------------------

def _t(name: str) -> dict[str, Any]:
    return {"strategies": [{"by": "role", "role": "button", "name": name}], "description": name}


def _run_fake(cap_data: dict[str, Any], state: FakeState, effects: dict[str, Any], tmp_path: Path) -> ReplayResult:
    from mm.artifact.schema import Capability
    cap = Capability.model_validate({"id": "app.t.flow", "summary": "s", "app": {"name": "a", "entry_path": "/"},
                                     **cap_data})
    rec = RunRecorder(tmp_path, "replay", SecretStore({}))
    try:
        return replay(cap, {}, base_url="http://a", secrets=SecretStore({}), recorder=rec, policy=PERMISSIVE,
                      approval=approval_for(cap),
                      surface_factory=lambda _: FakeSurface(state, effects))
    finally:
        rec.close()


CONFIRM = {"id": "s1", "intent": "commit", "action": "click", "target": _t("Confirm"), "risk": "irreversible",
           "expect": [{"kind": "url_matches", "pattern": "/done"}]}


def test_c3_retry_never_repeats_an_irreversible_step(tmp_path: Path) -> None:
    flaky = {"id": "flaky", "class": "recoverable", "then": "retry_step", "max_times": 2,
             "when": [{"kind": "text_visible", "any_frame": True, "pattern": "Please wait"}]}
    state = FakeState(present={"Confirm"})

    def confirm(s: FakeState) -> ActResult:
        s.texts, s.url = {"Please wait"}, "/pending"
        return ActResult(ok=True, strategy_index=0)

    result = _run_fake({"steps": [CONFIRM], "detectors": [flaky]}, state, {"Confirm": confirm}, tmp_path)
    assert state.performed.count("Confirm") == 1
    assert isinstance(result, ReplayFailure) and result.kind == "UNSAFE_TO_REPEAT"


def test_c3_a_dispatched_but_failed_irreversible_click_is_not_redone(tmp_path: Path) -> None:
    notice = {"id": "notice", "class": "recoverable", "when": [{"kind": "target_present", "target": _t("OK")}],
              "handle": [{"action": "click", "target": _t("OK")}]}
    state = FakeState(present={"Confirm"})

    def confirm(s: FakeState) -> ActResult:  # the click went out, then the UI errored (outcome unknown)
        s.present = {"Confirm", "OK"}
        return ActResult(ok=False, detail="click: element detached after dispatch", error="action_failed")

    result = _run_fake({"steps": [CONFIRM], "detectors": [notice]}, state,
                       {"Confirm": confirm, "OK": lambda s: s.present.discard("OK")}, tmp_path)
    assert state.performed.count("Confirm") == 1
    assert isinstance(result, ReplayFailure) and result.kind == "UNSAFE_TO_REPEAT"


# --- H4: provider failover, and clean discovery termination when no provider answers -----------

def _stub_provider(name: str, value: Any = None, exc: Exception | None = None) -> Any:
    from mm.llm.router import Provider

    def create_with_completion(**_: Any) -> Any:
        if exc is not None:
            raise exc
        return value, SimpleNamespace(usage=None)
    completions = SimpleNamespace(create_with_completion=create_with_completion)
    client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    return Provider(name, "stub-model", client)  # type: ignore[arg-type]


def _dead_provider() -> Any:
    from mm.llm.router import Provider, _client
    return Provider("dead", "x", _client("http://127.0.0.1:9/v1", "k"))


class _Ping(SimpleNamespace):
    pass


def test_h4_router_fails_over_when_the_first_provider_is_unreachable() -> None:
    from pydantic import BaseModel

    from mm.llm.router import LLMRouter

    class Ping(BaseModel):
        ok: bool
    router = LLMRouter([_dead_provider(), _stub_provider("backup", Ping(ok=True))])
    call = router.structured(Ping, [{"role": "user", "content": "x"}])
    assert call.provider == "backup"


def test_h4_router_does_not_fail_over_on_an_auth_error() -> None:
    import httpx
    import openai
    from instructor.core.exceptions import InstructorRetryException
    from pydantic import BaseModel

    from mm.llm.router import LLMRouter

    class Ping(BaseModel):
        ok: bool
    auth = openai.AuthenticationError("bad key", response=httpx.Response(401, request=httpx.Request("POST", "http://x")),
                                      body=None)
    wrapped = InstructorRetryException("failed", n_attempts=1, total_usage=0)
    wrapped.__cause__ = auth
    backup = _stub_provider("backup", Ping(ok=True))
    with pytest.raises(Exception):  # noqa: B017 — a misconfigured key must surface, not be papered over
        LLMRouter([_stub_provider("primary", exc=wrapped), backup]).structured(Ping, [{"role": "user", "content": "x"}])


def test_h4_discovery_ends_cleanly_when_no_provider_answers(tmp_path: Path) -> None:
    from mm.agent.loop import discover
    from mm.llm.router import LLMRouter
    rec = RunRecorder(tmp_path, "discover", SecretStore({}))
    result = discover(goal="g", entry_url="http://a/login", inputs={}, required_outputs=[],
                      surface=FakeSurface(FakeState(), {}), router=LLMRouter([_dead_provider()]),
                      secrets=SecretStore({}), recorder=rec, policy=PERMISSIVE, max_steps=3)
    rec.close()
    assert result.status == "llm_error"


# --- H5: an unreachable application is a structured failure, not a traceback --------------------

def test_h5_unreachable_app_is_a_structured_failure(tmp_path: Path) -> None:
    import socket
    with socket.socket() as sock:  # a port nothing listens on: connection refused, like an app that is down
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    result = _replay(f"http://127.0.0.1:{port}", tmp_path, BALANCE, member_id="10234")
    assert isinstance(result, ReplayFailure) and result.kind == "APP_UNREACHABLE"


# --- H6: the scrubber catches secrets however they are encoded, and values read during a run ----

@pytest.mark.parametrize("secret", ['pa"ss\\word', "xy"])
def test_h6_scrubber_catches_escaped_and_short_secrets(tmp_path: Path, secret: str) -> None:
    rec = RunRecorder(tmp_path, "x", SecretStore({"S": secret}))
    rec.event("t", value=f"typed {secret} here")
    rec.close()
    assert "typed [REDACTED] here" in next(tmp_path.glob("*/events.jsonl")).read_text()


def test_h6_values_read_during_discovery_are_masked_in_the_log(tmp_path: Path) -> None:
    rec = RunRecorder(tmp_path, "x", SecretStore({}))
    rec.taint("CNF264987")
    rec.event("decision", element='cell "CNF264987"', summary="Extracted CNF264987")
    rec.close()
    assert "CNF264987" not in next(tmp_path.glob("*/events.jsonl")).read_text()


# --- M7: a published version is immutable -------------------------------------------------------

def test_m7_saving_over_an_existing_version_is_refused(tmp_path: Path) -> None:
    cap = store.load(_latest(BALANCE))
    store.save(cap, tmp_path)
    with pytest.raises(store.VersionExists):
        store.save(cap, tmp_path)
    assert store.next_version(cap.id, tmp_path) == "0.{}.0".format(int(cap.version.split(".")[1]) + 1)


# --- M8: navigation typed mid-flow does not bake a host into the artifact ----------------------

def _nav_run(url: str) -> Any:
    from mm.agent.loop import DiscoveryResult, RecordedStep
    from mm.surface.base import ActionType
    u = {"": "http://10.1.2.3:8600/core/x.jsp"}
    return DiscoveryResult("r", "success", "", "g", "http://10.1.2.3:8600/login", {},
                           steps=[RecordedStep(ActionType.NAVIGATE, "go", None, url, None, u, u)])


def test_m8_same_origin_navigation_is_stored_relative(tmp_path: Path) -> None:
    from mm.artifact.compiler import compile_run
    cap = compile_run(_nav_run("http://10.1.2.3:8600/core/mbrsrch.jsp?x=1"), "corebank.t.x", pack="none")
    assert cap.steps[-1].value == "/core/mbrsrch.jsp?x=1"


def test_m8_cross_origin_navigation_does_not_compile() -> None:
    from mm.artifact.compiler import CompileError, compile_run
    with pytest.raises(CompileError, match="origin"):
        compile_run(_nav_run("https://elsewhere.example/steal"), "corebank.t.x", pack="none")


# --- M9 / M11 / L5: surface guarantees, on a real browser -------------------------------------

@pytest.fixture
def page_surface() -> Any:
    s = WebSurface(headless=True)
    yield s
    s.close()


def test_m9_an_element_that_cannot_be_targeted_is_not_acted_on(page_surface: WebSurface, monkeypatch: Any) -> None:
    from mm.surface.base import ActionType
    page_surface.page.set_content('<button onclick="window.clicked=1">Go</button>')
    ref = page_surface.observe().elements[0].ref
    monkeypatch.setattr(page_surface, "_is_unique_match", lambda *a: False)
    res = page_surface.act(ActionType.CLICK, ref)
    assert not res.ok and page_surface.page.evaluate("window.clicked") is None


def test_m11_a_fill_the_page_did_not_keep_is_a_failed_action(page_surface: WebSurface) -> None:
    from mm.surface.base import ActionType, AttrStrategy, Target
    page_surface.page.set_content('<input name="mbr" maxlength="3">')
    target = Target(strategies=[AttrStrategy(tag="input", name="mbr")])
    res = page_surface.perform(ActionType.FILL, target, "12345", 2000)
    assert not res.ok and "not retained" in res.detail


def test_l5_truncated_observations_say_so(page_surface: WebSurface) -> None:
    page_surface.page.set_content("".join(f'<a href="#{i}">link {i}</a><br>' for i in range(300)))
    obs = page_surface.observe()
    assert obs.truncated > 0


# --- M10: a missing detector pack is an error, not a capability with no detectors --------------

def test_m10_missing_default_pack_is_an_error(tmp_path: Path) -> None:
    from mm.artifact.compiler import CompileError, compile_run
    with pytest.raises(CompileError, match="pack"):
        compile_run(_nav_run("/x"), "nopack.t.x", packs_dir=tmp_path)


# --- M12 / L2: discovery observability and stuck detection --------------------------------------

class _ScriptedRouter:
    def __init__(self, decisions: list[dict[str, Any]]) -> None:
        self.decisions = decisions

    def structured(self, model: Any, messages: Any, max_retries: int = 2) -> Any:
        from mm.llm.router import LLMCall
        d = self.decisions.pop(0) if len(self.decisions) > 1 else self.decisions[0]
        return LLMCall(model(**d), "scripted", "m", 0, 0)


def test_m12_interruption_flag_is_logged(tmp_path: Path) -> None:
    from mm.agent.loop import discover
    rec = RunRecorder(tmp_path, "discover", SecretStore({}))
    router = _ScriptedRouter([{"thought": "t", "action": "click", "ref": "e1", "interruption": True},
                              {"thought": "t", "action": "done", "summary": "ok"}])
    discover(goal="g", entry_url="http://a", inputs={}, required_outputs=[], surface=FakeSurface(FakeState(), {}),
             router=router, secrets=SecretStore({}), recorder=rec, policy=PERMISSIVE,  # type: ignore[arg-type]
             max_steps=5)
    rec.close()
    events = [json.loads(line) for line in next(tmp_path.glob("*/events.jsonl")).read_text().splitlines()]
    assert any(e["type"] == "decision" and e.get("interruption") is True for e in events)


def test_l2_repeated_rejected_done_counts_as_stuck(tmp_path: Path) -> None:
    from mm.agent.loop import discover
    rec = RunRecorder(tmp_path, "discover", SecretStore({}))
    router = _ScriptedRouter([{"thought": "t", "action": "done", "summary": "done"}])
    result = discover(goal="g", entry_url="http://a", inputs={}, required_outputs=["balance"],
                      surface=FakeSurface(FakeState(), {}), router=router,  # type: ignore[arg-type]
                      secrets=SecretStore({}), recorder=rec, policy=PERMISSIVE, max_steps=20)
    rec.close()
    assert result.status == "stuck"


# --- L1 / L4 ------------------------------------------------------------------------------------

def test_l1_url_checkpoints_are_anchored() -> None:
    import re
    cap = store.load(_latest(BALANCE))
    pattern = next(c.pattern for s in cap.steps for c in s.expect if c.kind == "url_matches" and c.pattern)
    assert not re.search(pattern, "http://evil.example/redirect?to=http://x" + pattern.strip("^$()?|\\"))
    assert not re.search(pattern, "http://127.0.0.1:8600/evil/core/main.jsp")


def test_l4_failure_does_not_point_at_a_trace_that_was_never_recorded(bank_url: str, tmp_path: Path) -> None:
    result = _replay(bank_url, tmp_path, BALANCE, member_id="13100")
    assert isinstance(result, ReplayFailure)
    assert result.trace is None or Path(result.trace).exists()


# --- B: gaps in the original suite ------------------------------------------------------------

def test_b_session_expiry_mid_flow_restarts_a_read_only_flow(bank_url: str, tmp_path: Path) -> None:
    bank.FAULTS.add("session_expired_on_detail")
    result = _replay(bank_url, tmp_path, BALANCE, member_id="10234")
    assert isinstance(result, ReplaySuccess), result
    assert [(r.detector_id, r.action) for r in result.recoveries] == [("session_expired", "restarted")]


def test_b_session_expiry_on_confirm_is_never_replayed_blindly(bank_url: str, tmp_path: Path) -> None:
    bank.FAULTS.add("session_expired_on_confirm")
    result = _replay(bank_url, tmp_path, OPEN, member_id="10871", **OPEN_INPUTS)
    assert isinstance(result, ReplayFailure) and result.kind == "UNSAFE_TO_REPEAT"
    assert len(bank.data.MEMBERS["10871"].accounts) == 2  # nothing committed, and nothing pressed twice


def test_b_slow_fault_actually_slows_the_run(bank_url: str, tmp_path: Path) -> None:
    import time
    t0 = time.monotonic()
    bank.FAULTS.add("slow")
    result = _replay(bank_url, tmp_path, BALANCE, member_id="10234")
    assert isinstance(result, ReplaySuccess) and time.monotonic() - t0 > 8


@pytest.mark.parametrize("cap_id", [BALANCE, OPEN])
def test_b_no_concrete_values_in_any_committed_capability(cap_id: str) -> None:
    text = _latest(cap_id).read_text()
    for leaked in ("10234", "2,450.17", "2450.17", PASSWORD, "operator1", "Holiday Fund", "CNF"):
        assert leaked not in text


def test_b_business_outcome_is_not_masked_by_structural_fallback(bank_url: str, tmp_path: Path) -> None:
    result = _replay(bank_url, tmp_path, BALANCE, member_id="99999")
    assert isinstance(result, ReplayBusinessOutcome) and result.code == "MEMBER_NOT_FOUND"
