"""Masking for values that must be visible in shape but not in content (balances, names, ids)."""

from __future__ import annotations


def mask_value(value: str | None, keep_last: int = 2) -> str:
    """'$2,450.17' -> '$*,***.17'; 'Avery Testperson' -> '***** ********on'. Keeps length and punctuation."""
    if not value:
        return ""
    head, tail = value[:-keep_last] if len(value) > keep_last else "", value[-keep_last:]
    if len(value) <= keep_last:
        return "*" * len(value)
    return "".join("*" if c.isalnum() else c for c in head) + tail
