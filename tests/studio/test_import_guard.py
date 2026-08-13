"""ADR 0095 INV-1 — studio is a sidecar, not a composite / root export."""
from __future__ import annotations

from apeGmsh import apeGmsh as Session
from apeGmsh import __all__ as root_all


def test_studio_not_a_composite_and_not_reexported() -> None:
    assert "studio" not in root_all
    names = [row[0] for row in Session._COMPOSITES]
    assert "studio" not in names
