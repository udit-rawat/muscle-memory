"""Artifacts live as YAML files: `capabilities/<id>/<version>.yaml`, so they are reviewable in a diff."""

from __future__ import annotations

from pathlib import Path

import yaml

from mm.artifact.schema import Capability


def save(cap: Capability, root: Path = Path("capabilities")) -> Path:
    path = root / cap.id / f"{cap.version}.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    data = cap.model_dump(mode="json", exclude_none=True)
    path.write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=True, width=110), encoding="utf-8")
    return path


def load(path: Path) -> Capability:
    return Capability.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))
