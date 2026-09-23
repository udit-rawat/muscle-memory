"""`mm` command-line entrypoint."""

from __future__ import annotations

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
    faults: str = typer.Option("", help="Comma-separated: slow,notice,session_expired,permission,error500"),
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


if __name__ == "__main__":
    app()
