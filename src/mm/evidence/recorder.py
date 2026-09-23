"""Per-run evidence: an append-only JSONL event log, screenshots, and a Playwright trace.

Every event passes through a scrubber that removes known secret values, so even a bug that puts
a resolved password into an event cannot persist it.
"""

from __future__ import annotations

import json
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

from mm.values import SecretStore

REDACTED = "[REDACTED]"


class RunRecorder:
    def __init__(self, runs_dir: Path, kind: str, secrets: SecretStore) -> None:
        stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        self.run_id = f"{kind}-{stamp}-{uuid4().hex[:6]}"
        self.dir = runs_dir / self.run_id
        (self.dir / "screenshots").mkdir(parents=True, exist_ok=True)
        self._log = (self.dir / "events.jsonl").open("a", encoding="utf-8")
        self._secret_values = [v for v in secrets.values() if len(v) >= 4]
        self._t0 = time.monotonic()

    @property
    def trace_path(self) -> Path:
        return self.dir / "trace.zip"

    def event(self, type_: str, **data: Any) -> None:
        record = {"ts": datetime.now(UTC).isoformat(timespec="milliseconds"),
                  "t": round(time.monotonic() - self._t0, 3), "type": type_, **data}
        line = json.dumps(record, default=str, ensure_ascii=False)
        self._log.write(self._scrub(line) + "\n")
        self._log.flush()

    def screenshot_path(self, name: str) -> Path:
        return self.dir / "screenshots" / f"{name}.png"

    def write_text(self, name: str, text: str) -> Path:
        path = self.dir / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self._scrub(text), encoding="utf-8")
        return path

    def close(self) -> None:
        self._log.close()

    def _scrub(self, text: str) -> str:
        for v in self._secret_values:
            text = text.replace(v, REDACTED)
        return text
