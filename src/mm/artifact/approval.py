"""Approval: a reviewer's sign-off on the exact content of one capability version.

Stored next to the artifact as `<version>.approval.yaml`, never inside it, so the artifact stays
immutable. It records the capability id, version and a sha256 of the capability's canonical content, and
replay re-checks all three against the capability it is about to run: an approval for another capability,
another version or modified content authorises nothing.

What it is not: a signature. It proves the reviewed content is what runs (integrity against drift and
accidental edits); it does not prove *who* approved it, since anyone who can write the file can write an
approval. In production the sign-off would be issued by a service with authenticated reviewers (SSO) and
signed; see REPORT.md, Safety.
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
    tenant: str | None = None  # a tenant-specialised capability is approved separately
    sha256: str  # of the capability's canonical content (see content_digest)
    approved_by: str
    approved_at: str
    irreversible_steps: list[str]
    note: str = ""


def content_digest(cap: Capability) -> str:
    """sha256 of the capability's canonical JSON: any change to what it does changes the digest."""
    canonical = cap.model_dump_json(by_alias=True, exclude_none=True)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def approval_path(artifact: Path, tenant: str | None = None) -> Path:
    suffix = f".{tenant}" if tenant else ""
    return artifact.with_name(f"{artifact.stem}{suffix}.approval.yaml")


def approve(artifact: Path, cap: Capability, by: str, note: str = "") -> Approval:
    """Sign off `cap` (the base capability, or its specialisation for one tenant) as stored at `artifact`."""
    approval = Approval(capability_id=cap.id, version=cap.version, tenant=cap.app.tenant, sha256=content_digest(cap),
                        approved_by=by, approved_at=datetime.now(UTC).isoformat(timespec="seconds"),
                        irreversible_steps=[s.id for s in cap.steps if s.risk == "irreversible"], note=note)
    approval_path(artifact, cap.app.tenant).write_text(yaml.safe_dump(approval.model_dump(), sort_keys=False),
                                                       encoding="utf-8")
    return approval


def mismatch(approval: Approval, cap: Capability) -> str | None:
    """Why this approval does not apply to this capability, or None if it does."""
    if (approval.capability_id, approval.version, approval.tenant) != (cap.id, cap.version, cap.app.tenant):
        return (f"approval is for {approval.capability_id} {approval.version} (tenant {approval.tenant or 'base'}), "
                f"not {cap.id} {cap.version} (tenant {cap.app.tenant or 'base'})")
    if approval.sha256 != content_digest(cap):
        return "approval does not match the capability's content (it changed after it was approved)"
    return None


def load_valid(artifact: Path, cap: Capability) -> tuple[Approval | None, str]:
    """The approval for `cap` (as loaded from `artifact`, possibly specialised for a tenant) if it exists and
    still matches; otherwise None and why."""
    path = approval_path(artifact, cap.app.tenant)
    if not path.exists():
        who = f"for tenant {cap.app.tenant}" if cap.app.tenant else ""
        return None, f"capability has no approval {who}".strip()
    approval = Approval.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))
    if (why := mismatch(approval, cap)) is not None:
        return None, why
    return approval, ""
