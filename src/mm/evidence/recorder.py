"""Per-run evidence: an append-only JSONL event log, screenshots, and (opt-in) a Playwright trace.

Everything written goes through one scrubber:
- secrets are replaced by [REDACTED], in raw form *and* in the escaped form they take inside JSON;
- tainted values (anything read from the application during the run, e.g. an extracted balance) are
  replaced by a shape-preserving mask, wherever they reappear (element labels, model summaries...).

The Playwright trace cannot be scrubbed (it stores typed values and full page snapshots), which is
why it is off by default and never part of committed evidence.
"""

from __future__ import annotations

import json
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

from mm.redact import mask_value
from mm.values import SecretStore

REDACTED = "[REDACTED]"
_MIN_TAINT = 3  # shorter values ("S", "OK") would mangle ordinary words


class RunRecorder:
    def __init__(self, runs_dir: Path, kind: str, secrets: SecretStore) -> None:
        stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        self.run_id = f"{kind}-{stamp}-{uuid4().hex[:6]}"
        self.dir = runs_dir / self.run_id
        (self.dir / "screenshots").mkdir(parents=True, exist_ok=True)
        self._log = (self.dir / "events.jsonl").open("a", encoding="utf-8")
        self._replacements: dict[str, str] = {}
        for value in secrets.values():
            self._add(value, REDACTED)
        self._t0 = time.monotonic()

    @property
    def trace_path(self) -> Path:
        return self.dir / "trace.zip"

    def taint(self, value: str | None) -> None:
        """Mark a value read from the application: it will be masked wherever it appears from now on."""
        if value and len(value.strip()) >= _MIN_TAINT:
            self._add(value.strip(), mask_value(value.strip()))

    def event(self, type_: str, **data: Any) -> None:
        record = {"ts": datetime.now(UTC).isoformat(timespec="milliseconds"),
                  "t": round(time.monotonic() - self._t0, 3), "type": type_, **data}
        line = json.dumps(record, default=str, ensure_ascii=False)
        self._log.write(self.scrub(line) + "\n")
        self._log.flush()

    def screenshot_path(self, name: str) -> Path:
        return self.dir / "screenshots" / f"{name}.png"

    def write_text(self, name: str, text: str) -> Path:
        path = self.dir / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.scrub(text), encoding="utf-8")
        return path

    def close(self) -> None:
        self._log.close()

    def scrub(self, text: str) -> str:
        # Longest first, so a secret containing a shorter tainted value is replaced whole.
        for raw in sorted(self._replacements, key=len, reverse=True):
            text = text.replace(raw, self._replacements[raw])
        return text

    def _add(self, value: str, replacement: str) -> None:
        if not value:
            return
        escaped = json.dumps(value, ensure_ascii=False)[1:-1]  # how the value looks inside a JSON string
        self._replacements[value] = replacement
        self._replacements[escaped] = replacement
