"""Minimal operator console: the human side of the handoff.

A deliberately bare web page (the brief puts a full co-browsing console out of scope). The operator
sees the intervention queue with each request's reason, step, expected/observed state and a masked
screenshot, and who holds the session right now. They act in the *same* browser window the automation
was driving (it runs headed), then press Resume. The JSON API under /api is what a real console, a
chat integration or a pager would call.

Access: every request must carry this run's random token (printed once, in the console URL), and
state-changing requests from another origin are refused, so neither another local process nor a web page
can approve or take over. The operator's name is still self-declared: identity would come from SSO in
production (see REPORT.md, Safety).
"""

from __future__ import annotations

import contextlib
import html
import secrets
import threading
import time
from collections.abc import Awaitable, Callable
from pathlib import Path

import httpx
import uvicorn
from fastapi import FastAPI, Form, HTTPException, Request, Response
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, RedirectResponse
from pydantic import BaseModel

from mm.handoff.intervention import HandoffController, HandoffError, Intervention, Status


class OperatorCall(BaseModel):
    by: str = "operator"
    note: str = ""


class ConsoleUnavailable(RuntimeError):
    pass


def create_app(controller: HandoffController, token: str, own_origin: str) -> FastAPI:
    app = FastAPI(title="muscle-memory operator console", docs_url=None, redoc_url=None)

    @app.middleware("http")
    async def guard(request: Request, call_next: Callable[[Request], Awaitable[Response]]) -> Response:
        origin = request.headers.get("origin")
        if request.method != "GET" and origin is not None and origin != own_origin:
            return JSONResponse({"detail": "cross-origin request refused"}, status_code=403)
        presented = request.headers.get("x-operator-token") or request.query_params.get("token")
        if presented is None and request.method == "POST" and "form" in request.headers.get("content-type", ""):
            presented = (await request.form()).get("token")  # type: ignore[assignment]
        if not isinstance(presented, str) or not secrets.compare_digest(presented, token):
            return JSONResponse({"detail": "missing or wrong operator token"}, status_code=403)
        return await call_next(request)

    q = f"?token={token}"
    def run_action(item_id: str, action: str, by: str, note: str) -> Intervention:
        try:
            if action == "claim":
                return controller.claim(item_id, by)
            if action == "resume":
                return controller.resume(item_id, by, note)
            if action == "approve":
                return controller.approve(item_id, by, note)
            if action == "reject":
                return controller.reject(item_id, by, note)
            if action == "abort":
                return controller.abort(item_id, by, note)
        except HandoffError as exc:
            raise HTTPException(409, str(exc)) from exc
        raise HTTPException(404, f"unknown action {action}")

    # --- JSON API ------------------------------------------------------------------------------

    @app.get("/api/state")
    def state() -> dict[str, object]:
        lease = controller.lease.state
        return {"run_id": controller.rec.run_id,
                "lease": {"owner": lease.owner, "epoch": lease.epoch, "holder": lease.holder, "since": lease.since,
                          "reason": lease.reason},
                "interventions": [i.model_dump(mode="json") for i in controller.all()]}

    @app.post("/api/interventions/{item_id}/{action}")
    def api_action(item_id: str, action: str, call: OperatorCall) -> dict[str, object]:
        return run_action(item_id, action, call.by, call.note).model_dump(mode="json")

    # --- HTML ----------------------------------------------------------------------------------

    @app.get("/", response_class=HTMLResponse)
    def index() -> str:
        lease = controller.lease.state
        rows = "".join(
            f"<tr><td><a href='/i/{i.id}{q}'>{i.id}</a></td><td>{i.kind}</td><td><b>{i.status}</b></td>"
            f"<td>{html.escape(i.step_id or '')}</td><td>{html.escape(i.reason[:120])}</td></tr>"
            for i in reversed(controller.all()))
        return _page("Operator console", f"""
            <p class=lease>Session control: <b>{lease.owner}</b> ({html.escape(lease.holder)}, epoch {lease.epoch})
            &mdash; {html.escape(lease.reason)}</p>
            <table><tr><th>id</th><th>kind</th><th>status</th><th>step</th><th>reason</th></tr>
            {rows or "<tr><td colspan=5>No interventions yet.</td></tr>"}</table>""", q, refresh=True)

    @app.get("/i/{item_id}", response_class=HTMLResponse)
    def detail(item_id: str) -> str:
        try:
            i = controller.get(item_id)
        except HandoffError as exc:
            raise HTTPException(404, str(exc)) from exc
        buttons = {Status.OPEN: ["approve", "reject", "abort"] if i.kind == "approval_required" else ["claim", "abort"],
                   Status.CLAIMED: ["resume", "abort"]}.get(i.status, [])
        labels = {"claim": "Take control", "resume": "Resume (hand back)", "approve": "Approve", "reject": "Reject",
                  "abort": "Abort run"}
        forms = "".join(f"<form method=post action='/i/{i.id}/{a}'><input type=hidden name=token value='{token}'>"
                        f"<input name=by value=operator><input name=note placeholder=note>"
                        f"<button>{labels[a]}</button></form>" for a in buttons)
        acts = "".join(f"<li>{a.at} {a.kind} {html.escape(a.tag)} {html.escape(a.name)} "
                       f"{html.escape(a.detail)}</li>" for a in i.human_actions)
        shot = f"<img src='/i/{i.id}/screenshot{q}' alt='masked screenshot'>" if i.screenshot else ""
        return _page(f"{i.kind}: {i.id}", f"""
            <p><b>{i.status}</b> &middot; {html.escape(i.subject)} &middot; step {html.escape(i.step_id or '-')}</p>
            <p><b>Why:</b> {html.escape(i.reason)}</p>
            <p><b>Expected:</b> {html.escape(i.expected or '-')}<br>
               <b>Observed:</b> {html.escape(i.observed or '-')}</p>
            <p><b>Suggested:</b> {html.escape(' / '.join(i.suggested))}</p>
            {forms}<h3>Your actions in the session</h3><ul>{acts or '<li>none yet</li>'}</ul>
            <pre>{html.escape(i.page_excerpt)}</pre>{shot}""", q, refresh=i.status in (Status.OPEN, Status.CLAIMED))

    @app.get("/i/{item_id}/screenshot")
    def screenshot(item_id: str) -> FileResponse:
        i = controller.get(item_id)
        if not i.screenshot or not Path(i.screenshot).exists():
            raise HTTPException(404, "no screenshot")
        return FileResponse(i.screenshot, media_type="image/png")

    @app.post("/i/{item_id}/{action}")
    def form_action(item_id: str, action: str, by: str = Form("operator"), note: str = Form("")) -> RedirectResponse:
        run_action(item_id, action, by, note)
        return RedirectResponse(f"/i/{item_id}{q}", status_code=303)

    return app


class Console:
    """The console served from a background thread for the duration of one run."""

    def __init__(self, controller: HandoffController, port: int) -> None:
        self.controller = controller
        self.token = secrets.token_urlsafe(16)
        self.base = f"http://127.0.0.1:{port}"
        self.url = f"{self.base}/?token={self.token}"
        config = uvicorn.Config(create_app(controller, self.token, self.base), host="127.0.0.1", port=port,
                                log_level="warning")
        self._server = uvicorn.Server(config)
        self._thread = threading.Thread(target=self._serve, daemon=True)

    def _serve(self) -> None:
        # A failed bind makes uvicorn exit its thread; __enter__ detects that and raises ConsoleUnavailable.
        with contextlib.suppress(SystemExit, OSError):
            self._server.run()

    def __enter__(self) -> Console:
        self._thread.start()
        deadline = time.monotonic() + 10
        while not self._server.started and self._thread.is_alive() and time.monotonic() < deadline:
            time.sleep(0.05)
        # Trust nothing about the port: the console must answer, with this run's id, before anyone is sent to it.
        try:
            state = httpx.get(f"{self.base}/api/state", headers={"X-Operator-Token": self.token}, timeout=3).json()
            ours = state.get("run_id") == self.controller.rec.run_id
        except (httpx.HTTPError, ValueError):
            ours = False
        if not (self._server.started and ours):
            self.__exit__()
            raise ConsoleUnavailable(f"the operator console could not serve on {self.base} "
                                     "(port in use?); set MM_OPERATOR_PORT to a free port")
        return self

    def __exit__(self, *_: object) -> None:
        self._server.should_exit = True
        self._thread.join(timeout=5)


def _page(title: str, body: str, q: str, refresh: bool = False) -> str:
    meta = "<meta http-equiv=refresh content=2>" if refresh else ""
    return f"""<!doctype html><html><head><title>{html.escape(title)}</title>{meta}<style>
      body {{ font: 14px system-ui, sans-serif; margin: 24px; max-width: 1000px; }}
      table {{ border-collapse: collapse; width: 100%; }} td, th {{ border: 1px solid #ccc; padding: 4px 8px; }}
      form {{ display: inline-block; margin: 4px 8px 4px 0; }} img {{ max-width: 100%; border: 1px solid #999; }}
      .lease {{ padding: 8px; background: #f3f3f3; }} pre {{ white-space: pre-wrap; background: #fafafa; }}
    </style></head><body><h2>{html.escape(title)}</h2><p><a href="/{q}">queue</a></p>{body}</body></html>"""
