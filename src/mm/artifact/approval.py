"""Approval: a reviewer's sign-off on the exact bytes of one capability version.

Stored next to the artifact as `<version>.approval.yaml`, never inside it, so the artifact stays
immutable. It records the sha256 of the artifact file: change one byte of the capability and the
approval no longer applies. Irreversible steps replay only under a valid approval that names them.
"""

from __future__ import annotations

import hashlib
from datetime import UTC, datetime
from pathlib import Path

import yaml
from pydantic import BaseModel

from mm.artifact.schema import Capability


class Approval(BaseModel):
    capability_id: str
    version: str
    sha256: str
    approved_by: str
    approved_at: str
    irreversible_steps: list[str]
    note: str = ""


def approval_path(artifact: Path) -> Path:
    return artifact.with_name(f"{artifact.stem}.approval.yaml")


def digest(artifact: Path) -> str:
    return hashlib.sha256(artifact.read_bytes()).hexdigest()


def approve(artifact: Path, cap: Capability, by: str, note: str = "") -> Approval:
    approval = Approval(capability_id=cap.id, version=cap.version, sha256=digest(artifact), approved_by=by,
                        approved_at=datetime.now(UTC).isoformat(timespec="seconds"),
                        irreversible_steps=[s.id for s in cap.steps if s.risk == "irreversible"], note=note)
    approval_path(artifact).write_text(yaml.safe_dump(approval.model_dump(), sort_keys=False), encoding="utf-8")
    return approval


def load_valid(artifact: Path) -> tuple[Approval | None, str]:
    """The approval if it exists and still matches the artifact's bytes; otherwise None and why."""
    path = approval_path(artifact)
    if not path.exists():
        return None, "capability has no approval"
    approval = Approval.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))
    if approval.sha256 != digest(artifact):
        return None, "approval does not match the artifact (it changed after it was approved)"
    return approval, ""
