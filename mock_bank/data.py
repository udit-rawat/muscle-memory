"""Synthetic, obviously fake members. No real PII ever touches this project."""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from decimal import Decimal


@dataclass
class Account:
    suffix: str
    kind: str  # "savings" | "checking" | "certificate"
    nickname: str
    balance: Decimal


@dataclass
class Member:
    member_no: str
    first: str
    last: str
    city: str
    ssn_last4: str
    restricted: bool = False  # viewing requires elevated permission
    accounts: list[Account] = field(default_factory=list)


_SEED: dict[str, Member] = {
    m.member_no: m
    for m in [
        Member("10234", "Avery", "Testperson", "Springfield", "0001", accounts=[
            Account("S00", "savings", "Share Savings", Decimal("2450.17")),
            Account("S10", "checking", "Everyday Checking", Decimal("812.40")),
        ]),
        Member("10871", "Jordan", "Sampleton", "Riverton", "0002", accounts=[
            Account("S00", "savings", "Share Savings", Decimal("15320.00")),
            Account("C01", "certificate", "12mo Certificate", Decimal("5000.00")),
        ]),
        Member("11502", "Casey", "Mockwell", "Lakeview", "0003", accounts=[
            Account("S00", "savings", "Share Savings", Decimal("5.00")),
        ]),
        Member("12011", "Riley", "Fakerson", "Hill Valley", "0004", restricted=True, accounts=[
            Account("S00", "savings", "Share Savings", Decimal("98000.55")),
        ]),
        # No savings account at all: a read of "the savings balance" has no right answer here.
        Member("13100", "Morgan", "Nosavings", "Riverton", "0005", accounts=[
            Account("S10", "checking", "Everyday Checking", Decimal("777.77")),
        ]),
    ]
}

MEMBERS: dict[str, Member] = {}


def reset() -> None:
    MEMBERS.clear()
    MEMBERS.update(copy.deepcopy(_SEED))


def find(member_no: str = "", last_name: str = "") -> list[Member]:
    if member_no:
        m = MEMBERS.get(member_no.strip())
        return [m] if m else []
    if last_name:
        q = last_name.strip().lower()
        return [m for m in MEMBERS.values() if m.last.lower().startswith(q)]
    return []


reset()
