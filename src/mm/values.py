"""Value templates used in artifacts and agent actions.

    {{member_id}}                  -> a typed input supplied by the caller at invocation time
    {{secret:MOCKBANK_PASSWORD}}   -> resolved from the secret store at the last moment

Artifacts and logs only ever contain the template, never the resolved value. The LLM is told the
*names* of secrets it may use, never their values.
"""

from __future__ import annotations

import re
from collections.abc import Mapping

from mm.config import Settings, get_settings

_TOKEN = re.compile(r"\{\{\s*(secret:)?([A-Za-z_][A-Za-z0-9_]*)\s*\}\}")


class TemplateError(ValueError):
    pass


class SecretStore:
    """Named secrets. Production would back this with a vault; here it reads settings/.env."""

    def __init__(self, secrets: Mapping[str, str]) -> None:
        self._secrets = dict(secrets)

    @classmethod
    def from_settings(cls, settings: Settings | None = None) -> SecretStore:
        s = settings or get_settings()
        return cls({
            "MOCKBANK_USERNAME": s.mockbank_username,
            "MOCKBANK_PASSWORD": s.mockbank_password.get_secret_value(),
        })

    @property
    def names(self) -> list[str]:
        return sorted(self._secrets)

    def get(self, name: str) -> str:
        if name not in self._secrets:
            raise TemplateError(f"unknown secret {name!r}")
        return self._secrets[name]

    def values(self) -> list[str]:
        return [v for v in self._secrets.values() if v]


def render(template: str, params: Mapping[str, str], secrets: SecretStore) -> str:
    def sub(m: re.Match[str]) -> str:
        is_secret, name = bool(m.group(1)), m.group(2)
        if is_secret:
            return secrets.get(name)
        if name not in params:
            raise TemplateError(f"missing input {name!r}")
        return params[name]

    return _TOKEN.sub(sub, template)


def referenced(template: str) -> tuple[set[str], set[str]]:
    """(input names, secret names) used by a template."""
    inputs: set[str] = set()
    secrets: set[str] = set()
    for m in _TOKEN.finditer(template):
        (secrets if m.group(1) else inputs).add(m.group(2))
    return inputs, secrets


def parameterize(literal: str, params: Mapping[str, str]) -> str:
    """Safety net for the compiler: replace any literal input value with its placeholder.

    Longest values first so '10234' inside '102345' can't be partially replaced by a shorter one.
    """
    out = literal
    for name, value in sorted(params.items(), key=lambda kv: -len(kv[1])):
        if value:
            out = out.replace(value, "{{" + name + "}}")
    return out
