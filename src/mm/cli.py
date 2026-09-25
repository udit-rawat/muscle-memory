"""`mm` command-line entrypoint."""

from __future__ import annotations

import json
from pathlib import Path

import typer
from pydantic import BaseModel
from rich.console import Console

from mm.config import get_settings

app = typer.Typer(no_args_is_help=True, add_completion=False, help="muscle-memory: record once, replay many.")
console = Console()


@app.command()
def mockbank(
    port: int = typer.Option(8600, help="Port to serve on."),
    tenant: str = typer.Option("tenant_a", help="tenant_a | tenant_b"),
    faults: str = typer.Option("", help="Comma-separated: slow,notice,survey,session_expired,permission,error500"),
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


def _kv(pairs: list[str]) -> dict[str, str]:
    out: dict[str, str] = {}
    for pair in pairs:
        key, sep, value = pair.partition("=")
        if not sep or not key:
            raise typer.BadParameter(f"expected name=value, got {pair!r}")
        out[key.strip()] = value.strip()
    return out


@app.command()
def discover(
    goal: str = typer.Argument(..., help="What to accomplish, in natural language."),
    name: str = typer.Option(..., "--name", help="Capability id, e.g. corebank.member.get_savings_balance"),
    url: str = typer.Option("", help="Entry URL (default: mock bank sign-on page)."),
    param: list[str] = typer.Option([], "--param", "-p", help="Task input as name=value; becomes a typed input."),
    output: list[str] = typer.Option([], "--output", "-o", help="Output name the run must extract."),
    max_steps: int = typer.Option(0, help="Step budget (default from settings)."),
    pack: str = typer.Option(None, help="Detector pack to include (default: the capability id's app prefix)."),
    headless: bool = typer.Option(None, "--headless/--headed", help="Override MM_HEADLESS."),
) -> None:
    """Run the LLM agent on a goal; on success, compile and save a capability artifact."""
    from mm.agent.loop import discover as run_discovery
    from mm.artifact import store
    from mm.artifact.compiler import compile_run
    from mm.evidence.recorder import RunRecorder
    from mm.llm.router import LLMRouter
    from mm.surface.web import WebSurface
    from mm.values import SecretStore

    s = get_settings()
    secrets = SecretStore.from_settings(s)
    router = LLMRouter.from_settings(s)
    recorder = RunRecorder(s.mm_runs_dir, "discover", secrets)
    surface = WebSurface(headless=s.mm_headless if headless is None else headless, trace_path=recorder.trace_path)
    console.print(f"[bold]discovery[/] {recorder.run_id} → {recorder.dir}")
    try:
        result = run_discovery(
            goal=goal, entry_url=url or f"{s.mockbank_url}/login", inputs=_kv(param), required_outputs=output,
            surface=surface, router=router, secrets=secrets, recorder=recorder,
            max_steps=max_steps or s.mm_max_steps,
        )
    finally:
        surface.close()
    colour = "green" if result.status == "success" else "red"
    console.print(f"[{colour}]{result.status}[/] after {len(result.steps)} recorded steps: {result.summary}")
    if result.status == "success":
        cap = compile_run(result, name, pack=pack)
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
) -> None:
    """Replay a capability deterministically (no LLM). Prints the structured result as JSON.

    Exit codes: 0 success, 3 business outcome (a legitimate answer, e.g. MEMBER_NOT_FOUND), 2 failure.
    """
    from mm.artifact import store
    from mm.evidence.recorder import RunRecorder
    from mm.replay.executor import replay as run_replay
    from mm.surface.web import WebSurface
    from mm.values import SecretStore

    s = get_settings()
    cap = store.load(artifact)
    secrets = SecretStore.from_settings(s)
    recorder = RunRecorder(s.mm_runs_dir, "replay", secrets)
    show = s.mm_headless if headless is None else headless
    result = run_replay(
        cap, _kv(param), base_url=base_url or s.mockbank_url, secrets=secrets, recorder=recorder,
        surface_factory=lambda rec: WebSurface(headless=show, trace_path=rec.trace_path),
    )
    recorder.close()
    print(json.dumps(result.model_dump(mode="json"), indent=2))
    raise typer.Exit({"success": 0, "business_outcome": 3}.get(result.status, 2))


if __name__ == "__main__":
    app()
