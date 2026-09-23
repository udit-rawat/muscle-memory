"""Shared fixtures: an in-process mock bank on a free port, so browser tests need no running services."""

from __future__ import annotations

import socket
import threading
import time
from collections.abc import Iterator

import pytest
import uvicorn

from mock_bank import app as bank


@pytest.fixture(scope="session")
def bank_url() -> Iterator[str]:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    server = uvicorn.Server(uvicorn.Config(bank.app, host="127.0.0.1", port=port, log_level="warning"))
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    while not server.started:
        time.sleep(0.05)
    yield f"http://127.0.0.1:{port}"
    server.should_exit = True
    thread.join(timeout=5)


@pytest.fixture(autouse=True)
def clean_bank() -> None:
    bank.FAULTS.clear()
    bank.data.reset()
