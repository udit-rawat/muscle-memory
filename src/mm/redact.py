"""Masking for values that must be visible in shape but not in content (balances, names, ids)."""

from __future__ import annotations

import re


def mask_value(value: str | None, keep_last: int = 2) -> str:
    """'$2,450.17' -> '$*,***.17'; 'Avery Testperson' -> '***** ********on'. Keeps length and punctuation."""
    if not value:
        return ""
    head, tail = value[:-keep_last] if len(value) > keep_last else "", value[-keep_last:]
    if len(value) <= keep_last:
        return "*" * len(value)
    return "".join("*" if c.isalnum() else c for c in head) + tail


# Regulated values that should never reach the model, a screenshot or a log in the clear.
# Deliberately conservative patterns; names and addresses are not caught (see REPORT: Safety, limits).
PII_PATTERNS = [
    re.compile(r"\$\s?-?\(?[\d,]+\.\d{2}\)?"),  # currency amounts
    re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),  # SSN
    re.compile(r"\b\d{8,19}\b"),  # account / card numbers
    re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b"),  # email
    re.compile(r"\b\(?\d{3}\)?[-.\s]\d{3}[-.\s]\d{4}\b"),  # phone
]
PII_ANY = re.compile("|".join(p.pattern for p in PII_PATTERNS))


def mask_pii(text: str) -> str:
    """Mask every PII-looking span in free text, keeping its shape ('$2,450.17' -> '$*,***.17')."""
    return PII_ANY.sub(lambda m: mask_value(m.group(0)), text)
