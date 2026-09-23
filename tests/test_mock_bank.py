"""The mock bank is the test target, so its business states and faults must behave as documented."""

import pytest
from fastapi.testclient import TestClient

from mock_bank import app as bank


@pytest.fixture
def client() -> TestClient:
    bank.FAULTS.clear()
    bank.data.reset()
    c = TestClient(bank.app)
    r = c.post("/login", data={"username": bank.USERNAME, "password": bank.PASSWORD}, follow_redirects=False)
    assert r.status_code == 303
    return c


def test_unauthenticated_redirects_to_login() -> None:
    r = TestClient(bank.app).get("/core/mbrsrch.jsp", follow_redirects=False)
    assert r.status_code == 303 and "/login" in r.headers["location"]


def test_search_found_and_not_found(client: TestClient) -> None:
    assert "mbrdtl.jsp?m=10234" in client.post("/core/mbrsrch.jsp", data={"mbr_no": "10234"}).text
    assert "No member matches" in client.post("/core/mbrsrch.jsp", data={"mbr_no": "99999"}).text


def test_search_validation_error(client: TestClient) -> None:
    assert "must be numeric" in client.post("/core/mbrsrch.jsp", data={"mbr_no": "12a"}).text


def test_detail_shows_savings_balance(client: TestClient) -> None:
    assert "$2,450.17" in client.get("/core/mbrdtl.jsp?m=10234").text


def test_restricted_member_is_permission_denied(client: TestClient) -> None:
    r = client.get("/core/mbrdtl.jsp?m=12011")
    assert r.status_code == 403 and "not authorized" in r.text


def test_notice_interstitial_until_acknowledged(client: TestClient) -> None:
    bank.FAULTS.add("notice")
    assert "System Notice" in client.get("/core/mbrdtl.jsp?m=10234").text
    assert "System Notice" not in client.get("/core/mbrdtl.jsp?m=10234&ack=1").text


def test_session_expiry_is_one_shot(client: TestClient) -> None:
    bank.FAULTS.add("session_expired")
    r = client.get("/core/welcome.jsp", follow_redirects=False)
    assert "expired=1" in r.headers["location"]
    assert "session_expired" not in bank.FAULTS


def test_open_sub_account_review_then_confirm(client: TestClient) -> None:
    form = {"m": "10234", "acct_type": "club", "deposit": "25", "nickname": "Holiday"}
    assert "Opening deposit must be" in client.post("/core/opensub.jsp", data={**form, "deposit": "1"}).text
    assert "Please verify" in client.post("/core/opensub.jsp", data=form).text
    assert "Confirmation #" in client.post("/core/opensub_confirm.jsp").text
    # Confirm is not idempotent: a second post finds nothing pending.
    assert "No pending request" in client.post("/core/opensub_confirm.jsp").text
