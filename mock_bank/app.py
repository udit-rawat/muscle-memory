"""Mock legacy core-banking app: the stand-in for a vendor back-office system with no API.

Deliberately "legacy": server-rendered, a <frameset> shell, table layouts, no ids or test ids,
form posts with full page reloads. Runtime faults can be injected so replay has real
exceptional states to detect: slow loads, a System Notice interstitial (known to the detector
pack), a survey pop-up (deliberately unknown to it), session expiry, permission denial, HTTP 500.

Run it:  uv run uvicorn mock_bank.app:app --port 8600
Faults:  MOCKBANK_FAULTS=notice,slow  or  POST /__control/faults {"faults": ["notice"]}
"""

from __future__ import annotations

import os
import random
import time
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse, RedirectResponse, Response
from fastapi.templating import Jinja2Templates
from starlette.middleware.sessions import SessionMiddleware

from mock_bank import data, tenants
from mock_bank.data import Account

TENANT = os.getenv("MOCKBANK_TENANT", "tenant_a")
USERNAME = os.getenv("MOCKBANK_USERNAME", "operator1")
PASSWORD = os.getenv("MOCKBANK_PASSWORD", "change-me-local-only")
SESSION_TTL_S = int(os.getenv("MOCKBANK_SESSION_TTL", "900"))

# session_expired fires on the next in-app request; the _on_* variants fire mid-flow on one page.
EXPIRY_FAULTS = {"session_expired": None, "session_expired_on_detail": "/core/mbrdtl.jsp",
                 "session_expired_on_confirm": "/core/opensub_confirm.jsp"}
VALID_FAULTS = {"slow", "notice", "survey", "permission", "error500", *EXPIRY_FAULTS}
FAULTS: set[str] = {f for f in os.getenv("MOCKBANK_FAULTS", "").split(",") if f in VALID_FAULTS}

app = FastAPI(title="CoreOne (mock)", docs_url=None, redoc_url=None)
app.add_middleware(
    SessionMiddleware, secret_key=os.getenv("MOCKBANK_SESSION_SECRET", "dev-only-session-secret")
)
templates = Jinja2Templates(directory=str(Path(__file__).parent / "templates"))


# --- helpers ---------------------------------------------------------------------------------

def _ctx(request: Request, **extra: Any) -> dict[str, Any]:
    t = tenants.get(TENANT)
    return {"request": request, "t": t, "L": t["labels"], **extra}


def _render(request: Request, name: str, status_code: int = 200, **extra: Any) -> HTMLResponse:
    return templates.TemplateResponse(request, name, _ctx(request, **extra), status_code=status_code)


def _authed(request: Request) -> bool:
    last = request.session.get("last_seen")
    if not request.session.get("user") or last is None:
        return False
    fired = [f for f, path in EXPIRY_FAULTS.items() if f in FAULTS and path in (None, request.url.path)]
    if fired or time.time() - last > SESSION_TTL_S:
        FAULTS.difference_update(fired)  # one-shot: expires the current session once
        request.session.clear()
        return False
    request.session["last_seen"] = time.time()
    return True


def _guard(request: Request) -> Response | None:
    """Common pre-checks for every in-app page: auth, injected slowness, injected 500."""
    if not _authed(request):
        return RedirectResponse("/login?expired=1", status_code=303)
    if "slow" in FAULTS:
        time.sleep(random.uniform(2.0, 4.0))
    if "error500" in FAULTS:
        return _render(request, "error.html", status_code=500)
    return None


# --- auth ------------------------------------------------------------------------------------

@app.get("/", include_in_schema=False)
def root() -> RedirectResponse:
    return RedirectResponse("/login", status_code=303)


@app.get("/login", response_class=HTMLResponse)
def login_form(request: Request, expired: int = 0) -> HTMLResponse:
    return _render(request, "login.html", expired=bool(expired), error=None)


@app.post("/login", response_class=HTMLResponse, response_model=None)
def login(request: Request, username: str = Form(""), password: str = Form("")) -> Response:
    if username != USERNAME or password != PASSWORD:
        return _render(request, "login.html", expired=False, error="Invalid operator ID or password.")
    request.session.update(user=username, last_seen=time.time())
    return RedirectResponse("/core/main.jsp", status_code=303)


@app.get("/logout")
def logout(request: Request) -> RedirectResponse:
    request.session.clear()
    return RedirectResponse("/login", status_code=303)


# --- frameset shell --------------------------------------------------------------------------

@app.get("/core/main.jsp", response_class=HTMLResponse, response_model=None)
def main_frameset(request: Request) -> Response:
    if not _authed(request):
        return RedirectResponse("/login?expired=1", status_code=303)
    return _render(request, "frameset.html")


@app.get("/core/banner.jsp", response_class=HTMLResponse, response_model=None)
def banner(request: Request) -> Response:
    return _guard(request) or _render(request, "banner.html", user=request.session.get("user"))


@app.get("/core/nav.jsp", response_class=HTMLResponse, response_model=None)
def nav(request: Request) -> Response:
    return _guard(request) or _render(request, "nav.html")


@app.get("/core/welcome.jsp", response_class=HTMLResponse, response_model=None)
def welcome(request: Request) -> Response:
    return _guard(request) or _render(request, "welcome.html")


# --- member search → detail ------------------------------------------------------------------

@app.get("/core/mbrsrch.jsp", response_class=HTMLResponse, response_model=None)
def member_search_form(request: Request) -> Response:
    return _guard(request) or _render(request, "search.html", results=None, error=None, q={})


@app.post("/core/mbrsrch.jsp", response_class=HTMLResponse, response_model=None)
def member_search(request: Request, mbr_no: str = Form(""), lname: str = Form("")) -> Response:
    if (g := _guard(request)) is not None:
        return g
    q = {"mbr_no": mbr_no, "lname": lname}
    if not mbr_no.strip() and not lname.strip():
        return _render(request, "search.html", results=None, q=q,
                       error="At least one search criterion is required.")
    if mbr_no.strip() and not mbr_no.strip().isdigit():
        return _render(request, "search.html", results=None, q=q,
                       error="Member number must be numeric.")
    return _render(request, "search.html", results=data.find(mbr_no, lname), error=None, q=q)


@app.get("/core/mbrdtl.jsp", response_class=HTMLResponse, response_model=None)
def member_detail(request: Request, m: str = "", ack: int = 0) -> Response:
    if (g := _guard(request)) is not None:
        return g
    member = data.MEMBERS.get(m)
    if member is None:
        return _render(request, "message.html", status_code=404,
                       heading="Member Inquiry", message="Record not found.")
    if member.restricted or "permission" in FAULTS:
        return _render(request, "message.html", status_code=403, heading="Member Inquiry",
                       message="You are not authorized to view this member record. (SEC-041)")
    show_notice = "notice" in FAULTS and not ack
    show_survey = "survey" in FAULTS and not ack
    return _render(request, "detail.html", member=member, show_notice=show_notice, show_survey=show_survey)


# --- open sub-account: form → review → confirm (irreversible) --------------------------------

ACCOUNT_TYPES = {"savings": "Share Savings", "certificate": "Share Certificate", "club": "Holiday Club"}


@app.get("/core/opensub.jsp", response_class=HTMLResponse, response_model=None)
def open_sub_form(request: Request, m: str = "") -> Response:
    if (g := _guard(request)) is not None:
        return g
    member = data.MEMBERS.get(m)
    if member is None:
        return _render(request, "message.html", status_code=404,
                       heading="Open Sub-Account", message="Record not found.")
    return _render(request, "opensub.html", member=member, types=ACCOUNT_TYPES, error=None, f={})


@app.post("/core/opensub.jsp", response_class=HTMLResponse, response_model=None)
def open_sub_review(
    request: Request, m: str = Form(""), acct_type: str = Form(""),
    deposit: str = Form(""), nickname: str = Form(""),
) -> Response:
    if (g := _guard(request)) is not None:
        return g
    member = data.MEMBERS.get(m)
    if member is None:
        return _render(request, "message.html", status_code=404,
                       heading="Open Sub-Account", message="Record not found.")
    f = {"acct_type": acct_type, "deposit": deposit, "nickname": nickname}
    error = None
    try:
        amount = Decimal(deposit.replace(",", "").replace("$", ""))
    except InvalidOperation:
        amount = Decimal("-1")
    if acct_type not in ACCOUNT_TYPES:
        error = "Select an account type."
    elif amount < Decimal("5.00"):
        error = "Opening deposit must be at least $5.00."
    elif amount > Decimal("250000"):
        error = "Opening deposit exceeds teller limit; supervisor override required."
    if error:
        return _render(request, "opensub.html", member=member, types=ACCOUNT_TYPES, error=error, f=f)
    request.session["pending_open"] = {"m": m, **f, "amount": str(amount)}
    return _render(request, "review.html", member=member, p=request.session["pending_open"],
                   type_name=ACCOUNT_TYPES[acct_type])


@app.post("/core/opensub_confirm.jsp", response_class=HTMLResponse, response_model=None)
def open_sub_confirm(request: Request) -> Response:
    if (g := _guard(request)) is not None:
        return g
    p = request.session.pop("pending_open", None)
    if not p or p["m"] not in data.MEMBERS:
        return _render(request, "message.html", status_code=409, heading="Open Sub-Account",
                       message="No pending request. The transaction may have already been posted.")
    member = data.MEMBERS[p["m"]]
    suffix = f"S{len(member.accounts) + 20:02d}"
    member.accounts.append(Account(suffix, p["acct_type"], p["nickname"] or ACCOUNT_TYPES[p["acct_type"]],
                                   Decimal(p["amount"])))
    conf = f"CNF{random.randint(100000, 999999)}"
    return _render(request, "confirm.html", member=member, suffix=suffix, conf=conf,
                   amount=p["amount"], type_name=ACCOUNT_TYPES[p["acct_type"]])


# --- test-harness control plane (outside the agent's allowlist) ------------------------------

@app.get("/__control/faults")
def get_faults() -> dict[str, list[str]]:
    return {"faults": sorted(FAULTS), "valid": sorted(VALID_FAULTS)}


@app.post("/__control/faults")
async def set_faults(request: Request) -> dict[str, list[str]]:
    body = await request.json()
    FAULTS.clear()
    FAULTS.update(f for f in body.get("faults", []) if f in VALID_FAULTS)
    return {"faults": sorted(FAULTS)}


@app.post("/__control/reset")
def reset() -> dict[str, str]:
    FAULTS.clear()
    data.reset()
    return {"status": "reset"}
