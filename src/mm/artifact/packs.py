"""Detector packs: per-application knowledge of runtime states, shared across its capabilities."""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel

from mm.artifact.schema import Detector
from mm.config import PROJECT_ROOT


class DetectorPack(BaseModel):
    pack: str
    version: str
    applies_to: str = ""
    detectors: list[Detector]


def load_pack(name: str, packs_dir: Path = PROJECT_ROOT / "packs") -> DetectorPack | None:
    path = packs_dir / f"{name}.yaml"
    if not path.exists():
        return None
    pack = DetectorPack.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))
    source = f"pack:{pack.pack}@{pack.version}"
    pack.detectors = [d.model_copy(update={"source": d.source or source}) for d in pack.detectors]
    return pack
