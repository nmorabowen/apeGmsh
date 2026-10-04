"""Shared FEM stub for the live zeroLength spring tests (#1337)."""
from __future__ import annotations

from tests.opensees.fixtures.fem_stub import (
    FEMStub,
    _ElementGroupView,
    _ElementsStub,
    _NodesStub,
)


def coincident_spring() -> FEMStub:
    """One zeroLength spring: nodes 1 and 2 both at the origin.

    Same PGs as ``make_two_node_beam`` (``"Cols"`` the element,
    ``"Base"`` node 1, ``"Top"`` node 2). The spring tests used to borrow
    the beam fixture under ``ndm=2``, where the ends coincided only
    because node 2's z = 1 was dropped.
    """
    return FEMStub(
        nodes=_NodesStub(
            ids=[1, 2],
            coords=[(0.0, 0.0, 0.0), (0.0, 0.0, 0.0)],
            node_pgs={"Base": [1], "Top": [2]},
        ),
        elements=_ElementsStub(elem_pgs={
            "Cols": _ElementGroupView(ids=(1,), connectivity=((1, 2),)),
        }),
    )
