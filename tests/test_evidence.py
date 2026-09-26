"""Committed evidence must never contain a secret or a raw regulated value (checked on every byte)."""

from __future__ import annotations

import re
import zipfile
from pathlib import Path

import pytest

from mock_bank import data

EVIDENCE = Path(__file__).resolve().parent.parent / "evidence"
SECRETS = [b"change-me-local-only", b"operator1"]
UNMASKED_AMOUNT = re.compile(rb"\$\s?\d[\d,]*\.\d{2}")  # "$2,450.17"; the masked form is "$*,***.17"


def _seeded_values() -> list[bytes]:
    values = []
    for member in data._SEED.values():
        for acct in member.accounts:
            values += [f"{acct.balance:,.2f}".encode(), f"{acct.balance:.2f}".encode()]
    return [v for v in values if len(v) > 4]  # "5.00" alone is too generic to be meaningful


def _files() -> list[Path]:
    return [p for p in EVIDENCE.rglob("*") if p.is_file()]


@pytest.mark.skipif(not EVIDENCE.exists(), reason="no evidence generated yet")
def test_no_secret_or_raw_value_in_evidence() -> None:
    offenders = []
    for path in _files():
        blobs = [path.read_bytes()]
        if zipfile.is_zipfile(path):
            with zipfile.ZipFile(path) as z:
                blobs += [z.read(n) for n in z.namelist()]
        for blob in blobs:
            hits = [s for s in SECRETS + _seeded_values() if s in blob]
            if path.suffix != ".png" and (m := UNMASKED_AMOUNT.search(blob)):
                hits.append(m.group(0))
            if hits:
                offenders.append(f"{path.relative_to(EVIDENCE)}: {hits[:3]}")
    assert not offenders, "\n".join(offenders)


@pytest.mark.skipif(not EVIDENCE.exists(), reason="no evidence generated yet")
def test_evidence_has_no_traces() -> None:
    assert not [p for p in _files() if p.name == "trace.zip"]


@pytest.mark.skipif(not EVIDENCE.exists(), reason="no evidence generated yet")
def test_evidence_holds_no_machine_specific_paths() -> None:
    offenders = [str(p.relative_to(EVIDENCE)) for p in _files() if p.suffix != ".png"
                 and re.search(rb"/(Users|home|private|var/folders|tmp)/", p.read_bytes())]
    assert not offenders, offenders
