"""Two tenants running the "same vendor product", configured differently (labels, branding, nav order)."""

from __future__ import annotations

from typing import Any

TENANTS: dict[str, dict[str, Any]] = {
    "tenant_a": {
        "brand": "Pinecrest Federal Credit Union",
        "product": "CoreOne 7.4.2",
        "color": "#003366",
        "labels": {
            "nav_search": "Member Search",
            "nav_home": "Home",
            "member_no": "Member #",
            "last_name": "Last Name",
            "search_btn": "Search",
            "savings": "Share Savings",
            "open_sub": "Open Sub-Account",
        },
        "nav_order": ["nav_home", "nav_search"],
    },
    "tenant_b": {
        "brand": "Harborview Community CU",
        "product": "CoreOne 7.5.0",
        "color": "#5a2d0c",
        "labels": {
            "nav_search": "Find Member",
            "nav_home": "Dashboard",
            "member_no": "Account Holder ID",
            "last_name": "Surname",
            "search_btn": "Go",
            "savings": "Primary Savings",
            "open_sub": "Add Share Account",
        },
        "nav_order": ["nav_search", "nav_home"],
    },
}


def get(tenant: str) -> dict[str, Any]:
    return TENANTS.get(tenant, TENANTS["tenant_a"])
