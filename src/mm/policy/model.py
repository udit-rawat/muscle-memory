"""The safety policy: an explicit allowlist of where automation may go and what it may do, plus the
rules that classify an action's risk. Loaded from config/policy.yaml; enforced by GuardedSurface
(UI actions) and by the browser's request filter (every network request)."""

from __future__ import annotations

import fnmatch
import re
from functools import cached_property
from pathlib import Path
from typing import Literal
from urllib.parse import urlparse

import yaml
from pydantic import BaseModel, Field

from mm.surface.base import ActionType

Risk = Literal["safe", "mutating", "irreversible"]
_ORDER: dict[Risk, int] = {"safe": 0, "mutating": 1, "irreversible": 2}


class RequestRule(BaseModel):
    method: str = "*"
    path: str


class IrreversibleRules(BaseModel):
    control_names: str = r"(?!)"  # matches nothing
    requests: list[RequestRule] = Field(default_factory=list)


class NetworkPolicy(BaseModel):
    allow_origins: list[str]
    deny_paths: list[str] = Field(default_factory=list)


class Policy(BaseModel):
    version: int = 1
    network: NetworkPolicy
    actions: dict[str, list[ActionType]] = Field(default_factory=lambda: {"allowed": list(ActionType)})
    risk: dict[str, IrreversibleRules] = Field(default_factory=dict)

    @classmethod
    def load(cls, path: Path = Path("config/policy.yaml")) -> Policy:
        return cls.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))

    # --- where ---------------------------------------------------------------------------------

    def url_allowed(self, url: str) -> tuple[bool, str]:
        parts = urlparse(url)
        if parts.scheme in ("about", "data", "blob") or url == "about:blank":
            return True, ""
        origin = f"{parts.scheme}://{parts.netloc}"
        if not any(fnmatch.fnmatchcase(origin, pat) for pat in self.network.allow_origins):
            return False, f"origin {origin} is not on the allowlist"
        for pattern in self.network.deny_paths:
            if re.search(pattern, parts.path):
                return False, f"path {parts.path} is denied by policy"
        return True, ""

    # --- what ----------------------------------------------------------------------------------

    def action_allowed(self, action: ActionType) -> bool:
        return action in self.actions.get("allowed", [])

    @cached_property
    def _irreversible(self) -> IrreversibleRules:
        return self.risk.get("irreversible", IrreversibleRules())

    def classify_control(self, action: ActionType, accessible_name: str) -> Risk:
        """Risk of acting on a control, judged from what a human operator would read on it."""
        if action is ActionType.CLICK and re.search(self._irreversible.control_names, accessible_name):
            return "irreversible"
        return "safe"

    def is_irreversible_request(self, method: str, url: str) -> bool:
        path = urlparse(url).path
        return any(re.search(r.path, path) and r.method in ("*", method.upper()) for r in self._irreversible.requests)


def stricter(a: Risk, b: Risk) -> Risk:
    return a if _ORDER[a] >= _ORDER[b] else b
