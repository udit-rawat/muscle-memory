"""Artifacts live as YAML files: `capabilities/<id>/<version>.yaml`, so they are reviewable in a diff.

A saved version is immutable: re-recording a capability produces a new version, never an in-place edit,
so a caller pinned to a version always gets the behaviour that was reviewed.
"""

from __future__ import annotations

from pathlib import Path

import yaml

from mm.artifact.schema import Capability


class VersionExists(FileExistsError):
    pass


def save(cap: Capability, root: Path = Path("capabilities")) -> Path:
    path = root / cap.id / f"{cap.version}.yaml"
    if path.exists():
        raise VersionExists(f"{cap.id} {cap.version} already exists; versions are immutable, save a new one")
    path.parent.mkdir(parents=True, exist_ok=True)
    data = cap.model_dump(mode="json", exclude_none=True, by_alias=True)
    path.write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=True, width=110), encoding="utf-8")
    return path


def load(path: Path) -> Capability:
    return Capability.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))


def versions(cap_id: str, root: Path = Path("capabilities")) -> list[tuple[int, int, int]]:
    out = []
    for f in (root / cap_id).glob("*.yaml"):
        parts = f.stem.split(".")
        if len(parts) == 3 and all(p.isdigit() for p in parts):
            out.append((int(parts[0]), int(parts[1]), int(parts[2])))
    return sorted(out)


def next_version(cap_id: str, root: Path = Path("capabilities")) -> str:
    """A re-recording is a minor bump over the highest existing version (0.1.0 if there is none)."""
    existing = versions(cap_id, root)
    if not existing:
        return "0.1.0"
    major, minor, _ = existing[-1]
    return f"{major}.{minor + 1}.0"


def latest_path(cap_id: str, root: Path = Path("capabilities")) -> Path:
    existing = versions(cap_id, root)
    if not existing:
        raise FileNotFoundError(f"no versions of {cap_id} under {root}")
    return root / cap_id / (".".join(map(str, existing[-1])) + ".yaml")
