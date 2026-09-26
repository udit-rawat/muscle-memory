"""Tenant profiles: reuse one recorded capability across institutions running the same vendor product.

Hundreds of tenants run the same core-banking product, each configured, branded and versioned a little
differently. Re-recording every capability per tenant does not scale, so a capability is recorded once
(on any tenant) and *specialised* per tenant at load time by a small, reviewable profile:

  labels  exact renames of what an operator reads on screen (menu items, field labels, row names). They
          apply to every capability of the app: one profile per tenant, not one per capability.
  steps   per-capability, per-step patches for structural differences a rename cannot express.

Specialising only rewrites how things are *found*. The flow, the contract (inputs, outputs, outcomes)
and the detectors' meaning are unchanged. The tenant is recorded in the result (`app.tenant`), so its
content, and therefore its approval, is distinct from the base capability's.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field

from mm.artifact.schema import Capability
from mm.config import PROJECT_ROOT

# Fields that hold on-screen text (as opposed to ids, patterns of URLs, or values), wherever they occur.
_TEXT_FIELDS = {"name", "text", "title", "row_key", "column_header", "description", "pattern"}


class TenantMismatch(ValueError):
    pass


class TenantProfile(BaseModel):
    tenant: str
    app: str
    product_version: str = ""
    labels: dict[str, str] = Field(default_factory=dict)
    steps: dict[str, dict[str, dict[str, Any]]] = Field(default_factory=dict)  # capability id -> step id -> patch


def load_tenant(name: str, tenants_dir: Path = PROJECT_ROOT / "tenants") -> TenantProfile:
    path = tenants_dir / f"{name}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"no tenant profile {name!r} in {tenants_dir}/")
    return TenantProfile.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))


def specialise(cap: Capability, profile: TenantProfile) -> Capability:
    if profile.app != cap.app.name:
        raise TenantMismatch(f"tenant {profile.tenant} profiles app {profile.app!r}, not {cap.app.name!r}")
    data = cap.model_dump(mode="json", by_alias=True, exclude_none=True)
    for step_key in ("steps", "detectors", "success"):
        data[step_key] = _rename(data.get(step_key, []), profile.labels)
    patches = profile.steps.get(cap.id, {})
    if unknown := set(patches) - {s["id"] for s in data["steps"]}:
        raise TenantMismatch(f"tenant {profile.tenant} patches unknown steps of {cap.id}: {sorted(unknown)}")
    data["steps"] = [{**s, **patches.get(s["id"], {})} for s in data["steps"]]
    data["app"]["tenant"] = profile.tenant
    return Capability.model_validate(data)


def _rename(node: Any, labels: dict[str, str], key: str | None = None) -> Any:
    """Replace exact on-screen texts (whole values, or quoted inside a human-readable description)."""
    if isinstance(node, dict):
        return {k: _rename(v, labels, k) for k, v in node.items()}
    if isinstance(node, list):
        return [_rename(v, labels, key) for v in node]
    if isinstance(node, str) and key in _TEXT_FIELDS:
        if node in labels:
            return labels[node]
        if key == "description":  # e.g. 'link "Member Search" in frame nav'
            for old, new in labels.items():
                node = node.replace(f'"{old}"', f'"{new}"')
    return node
