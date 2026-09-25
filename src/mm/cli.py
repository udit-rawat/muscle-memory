"""`mm` command-line entrypoint."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

import typer
from pydantic import BaseModel
from rich.console import Console

from mm.config import Settings, get_settings
from mm.evidence.recorder import RunRecorder
from mm.handoff.intervention import HandoffController
from mm.handoff.operator import SimulatedOperator
from mm.policy.guard import request_policy
from mm.policy.model import Policy
from mm.surface.web import WebSurface

app = typer.Typer(no_args_is_help=True, add_completion=False, help="muscle-memory: record once, replay many.")
console = Console()
err_console = Console(stderr=True)  # replay's stdout is reserved for the JSON result


@app.command()
def mockbank(
    port: int = typer.Option(8600, help="Port to serve on."),
    tenant: str = typer.Option("tenant_a", help="tenant_a | tenant_b"),
    faults: str = typer.Option(
        "", help="Comma-separated: slow,notice,survey,permission,error500,session_expired[_on_detail|_on_confirm]"),
) -> None:
    """Serve the mock legacy core-banking app."""
    import os

    import uvicorn

    os.environ["MOCKBANK_TENANT"] = tenant
    os.environ["MOCKBANK_FAULTS"] = faults
    s = get_settings()
    os.environ.setdefault("MOCKBANK_PASSWORD", s.mockbank_password.get_secret_value())
    os.environ.setdefault("MOCKBANK_USERNAME", s.mockbank_username)
    uvicorn.run("mock_bank.app:app", host="127.0.0.1", port=port, log_level="warning")


class _Ping(BaseModel):
    ok: bool


@app.command()
def doctor(ping: bool = typer.Option(False, help="Make one tiny structured call per provider.")) -> None:
    """Check configuration: providers, browser, target."""
    from mm.llm.router import LLMRouter, NoProviderConfigured

    s = get_settings()
    try:
        router = LLMRouter.from_settings(s)
    except NoProviderConfigured as exc:
        console.print(f"[red]LLM:[/] {exc}")
        raise typer.Exit(1) from exc
    for p in router.providers:
        line = f"[green]LLM provider[/] {p.name} → {p.model}"
        if ping:
            try:
                p.client.chat.completions.create(
                    model=p.model,
                    response_model=_Ping,
                    messages=[{"role": "user", "content": 'Reply with JSON {"ok": true}.'}],
                    max_retries=1,
                )
                line += "  [green]reachable[/]"
            except Exception as exc:  # noqa: BLE001 — diagnostic command, report and continue
                line += f"  [red]{type(exc).__name__}: {str(exc)[:120]}[/]"
        console.print(line)
    console.print(f"Target: {s.mockbank_url} (tenant={s.mockbank_tenant})")


TRACE_WARNING = ("--trace records a Playwright trace: it stores typed values (including passwords) and full page "
                 "snapshots, and cannot be redacted. Use it for local debugging only; never commit it.")
POLICY = Path("config/policy.yaml")


def web_surface_factory(headless: bool, trace: bool, policy: Policy,
                        handoff: HandoffController | None = None,
                        operator: SimulatedOperator | None = None) -> Callable[[RunRecorder], WebSurface]:
    """How the CLI builds browser sessions: the network policy is enforced on every request, and human actions
    are captured when a handoff is possible. Tests use this too, so they run with production defaults."""
    def make(rec: RunRecorder) -> WebSurface:
        surface = WebSurface(headless=headless, trace_path=rec.trace_path if trace else None,
                             request_policy=request_policy(policy),
                             on_human_action=handoff.record_human_action if handoff else None)
        if operator is not None:
            operator.page = surface.page
        return surface
    return make


def _handoff(rec: RunRecorder, s: Settings, escalate: bool, simulate: str | None, headless: bool
             ) -> tuple[HandoffController | None, SimulatedOperator | None, bool]:
    """Returns (controller, simulated operator, headless). A real operator needs to see the browser window."""
    if not escalate and simulate is None:
        return None, None, headless
    controller = HandoffController(rec)
    operator = None
    if simulate is not None:
        clicks = [c.split(":", 1)[1] for c in simulate.split(",") if c.startswith("click:")]
        decision = next((c for c in simulate.split(",") if c in ("approve", "reject", "abort")), "approve")
        operator = SimulatedOperator(f"http://127.0.0.1:{s.mm_operator_port}", page=None,  # type: ignore[arg-type]
                                     clicks=clicks, decision=decision, abort=decision == "abort" and not clicks)
        controller.operator_tick = operator.tick
    elif headless:
        err_console.print("[yellow]--escalate needs a visible browser for the operator; running headed.[/]")
        headless = False
    return controller, operator, headless


def _kv(pairs: list[str]) -> dict[str, str]:
    out: dict[str, str] = {}
    for pair in pairs:
        key, sep, value = pair.partition("=")
        if not sep or not key:
            raise typer.BadParameter(f"expected name=value, got {pair!r}")
        out[key.strip()] = value.strip()
    return out


ESCALATE_HELP = "On an unrecoverable state (replay), when stuck or before an irreversible click (discovery), " \
                "raise an intervention; the operator console runs on MM_OPERATOR_PORT."
SIMULATE_HELP = "Stand-in for a human at the console (demos/tests), e.g. 'click:Maybe Later' or 'approve'. " \
                "Uses the real console API and real clicks in the live session."


@app.command()
def discover(
    goal: str = typer.Argument(..., help="What to accomplish, in natural language."),
    name: str = typer.Option(..., "--name", help="Capability id, e.g. corebank.member.get_savings_balance"),
    url: str = typer.Option("", help="Entry URL (default: mock bank sign-on page)."),
    param: list[str] = typer.Option([], "--param", "-p", help="Task input as name=value; becomes a typed input."),
    output: list[str] = typer.Option([], "--output", "-o", help="Output name the run must extract."),
    max_steps: int = typer.Option(0, help="Step budget (default from settings)."),
    pack: str = typer.Option(None, help="Detector pack (default: the capability id's app prefix; 'none' for no pack)."),
    headless: bool = typer.Option(None, "--headless/--headed", help="Override MM_HEADLESS."),
    trace: bool = typer.Option(False, "--trace", help="Record a Playwright trace (unredacted; local debugging only)."),
    escalate: bool = typer.Option(False, "--escalate", help=ESCALATE_HELP),
    simulate_operator: str = typer.Option(None, "--simulate-operator", help=SIMULATE_HELP),
    handoff_timeout: float = typer.Option(600, help="Seconds to wait for an operator."),
) -> None:
    """Run the LLM agent on a goal; on success, compile and save a capability artifact."""
    from mm.agent.loop import discover as run_discovery
    from mm.artifact import store
    from mm.artifact.compiler import compile_run
    from mm.handoff.console import Console as OperatorConsole
    from mm.llm.router import LLMRouter
    from mm.values import SecretStore

    s = get_settings()
    policy = Policy.load(POLICY)
    secrets = SecretStore.from_settings(s)
    router = LLMRouter.from_settings(s)
    recorder = RunRecorder(s.mm_runs_dir, "discover", secrets)
    handoff, operator, show = _handoff(recorder, s, escalate, simulate_operator,
                                       s.mm_headless if headless is None else headless)
    if trace:
        console.print(f"[yellow]warning:[/] {TRACE_WARNING}")
    surface = web_surface_factory(show, trace, policy, handoff, operator)(recorder)
    console.print(f"[bold]discovery[/] {recorder.run_id} → {recorder.dir}")
    try:
        with OperatorConsole(handoff, s.mm_operator_port) if handoff else _nothing() as oc:
            if oc is not None:
                console.print(f"operator console: {oc.url}")
            result = run_discovery(
                goal=goal, entry_url=url or f"{s.mockbank_url}/login", inputs=_kv(param), required_outputs=output,
                surface=surface, router=router, secrets=secrets, recorder=recorder, policy=policy, handoff=handoff,
                handoff_timeout_s=handoff_timeout, max_steps=max_steps or s.mm_max_steps,
            )
    finally:
        surface.close()
    colour = "green" if result.status == "success" else "red"
    console.print(f"[{colour}]{result.status}[/] after {len(result.steps)} recorded steps: {result.summary}")
    if result.status == "success":
        cap = compile_run(result, name, pack=pack, version=store.next_version(name), policy=policy)
        path = store.save(cap)
        (recorder.dir / "artifact.yaml").write_text(path.read_text(encoding="utf-8"), encoding="utf-8")
        recorder.event("artifact_saved", path=str(path), capability=cap.id, version=cap.version)
        console.print(f"[green]capability saved[/] {path}")
    recorder.close()
    raise typer.Exit(0 if result.status == "success" else 1)


@app.command()
def replay(
    artifact: Path = typer.Argument(..., exists=True, dir_okay=False, help="Path to a capability YAML."),
    param: list[str] = typer.Option([], "--param", "-p", help="Input as name=value."),
    base_url: str = typer.Option("", help="Tenant base URL to bind (default: MOCKBANK_URL)."),
    headless: bool = typer.Option(None, "--headless/--headed", help="Override MM_HEADLESS."),
    trace: bool = typer.Option(False, "--trace", help="Record a Playwright trace (unredacted; local debugging only)."),
    escalate: bool = typer.Option(False, "--escalate", help=ESCALATE_HELP),
    simulate_operator: str = typer.Option(None, "--simulate-operator", help=SIMULATE_HELP),
    handoff_timeout: float = typer.Option(600, help="Seconds to wait for an operator."),
) -> None:
    """Replay a capability deterministically (no LLM). Prints the structured result as JSON.

    Exit codes: 0 success, 3 business outcome (a legitimate answer, e.g. MEMBER_NOT_FOUND), 2 failure.
    """
    from mm.artifact import store
    from mm.artifact.approval import load_valid
    from mm.handoff.console import Console as OperatorConsole
    from mm.replay.executor import replay as run_replay
    from mm.values import SecretStore

    s = get_settings()
    policy = Policy.load(POLICY)
    cap = store.load(artifact)
    approval, approval_problem = load_valid(artifact)
    secrets = SecretStore.from_settings(s)
    recorder = RunRecorder(s.mm_runs_dir, "replay", secrets)
    handoff, operator, show = _handoff(recorder, s, escalate, simulate_operator,
                                       s.mm_headless if headless is None else headless)
    if trace:
        err_console.print(f"[yellow]warning:[/] {TRACE_WARNING}")
    with OperatorConsole(handoff, s.mm_operator_port) if handoff else _nothing() as oc:
        if oc is not None:
            err_console.print(f"operator console: {oc.url}")
        result = run_replay(
            cap, _kv(param), base_url=base_url or s.mockbank_url, secrets=secrets, recorder=recorder, policy=policy,
            approval=approval, approval_problem=approval_problem, handoff=handoff,
            escalation_timeout_s=handoff_timeout,
            surface_factory=web_surface_factory(show, trace, policy, handoff, operator),
        )
    recorder.close()
    print(json.dumps(result.model_dump(mode="json"), indent=2))
    raise typer.Exit({"success": 0, "business_outcome": 3}.get(result.status, 2))


@app.command()
def approve(
    artifact: Path = typer.Argument(..., exists=True, dir_okay=False, help="Path to a capability YAML."),
    by: str = typer.Option(..., "--by", help="Who is signing off (recorded in the approval and every run)."),
    note: str = typer.Option("", help="Why it is approved, e.g. the review ticket."),
) -> None:
    """Sign off a capability version so its irreversible steps may replay. Bound to the file's exact bytes."""
    from mm.artifact import store
    from mm.artifact.approval import approval_path
    from mm.artifact.approval import approve as sign

    cap = store.load(artifact)
    irreversible = [s for s in cap.steps if s.risk == "irreversible"]
    console.print(f"{cap.id} {cap.version}: {len(cap.steps)} steps, irreversible: "
                  + (", ".join(f"{s.id} ({s.intent})" for s in irreversible) or "none"))
    approval = sign(artifact, cap, by, note)
    console.print(f"[green]approved[/] by {approval.approved_by} → {approval_path(artifact)} "
                  f"(sha256 {approval.sha256[:12]}…)")


@app.command()
def schema() -> None:
    """Print the capability JSON Schema: the contract a calling agent or reviewer reads."""
    from mm.artifact.schema import Capability

    print(json.dumps(Capability.model_json_schema(by_alias=True), indent=2))


class _nothing:  # noqa: N801 — a no-op context manager standing in for the console
    def __enter__(self) -> None:
        return None

    def __exit__(self, *_: object) -> None:
        return None


if __name__ == "__main__":
    app()
