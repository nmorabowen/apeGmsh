"""Pin the per-kind topology mapping.

PR 1 collapsed two sources of truth (the dialog's hand-written
``_KIND_TO_TOPOLOGY`` dict and each diagram subclass's composite-reader
calls) into one: a class-level ``topology`` attribute on each Diagram.
The dialog now derives its dict from that attribute. These tests guard
the eight existing kinds so a future rename can't silently re-open the
drift.
"""
from __future__ import annotations

from apeGmsh.viewers.diagrams._contour import ContourDiagram
from apeGmsh.viewers.diagrams._deformed_shape import DeformedShapeDiagram
from apeGmsh.viewers.diagrams._fiber_section import FiberSectionDiagram
from apeGmsh.viewers.diagrams._gauss_marker import GaussPointDiagram
from apeGmsh.viewers.diagrams._layer_stack import LayerStackDiagram
from apeGmsh.viewers.diagrams._line_force import LineForceDiagram
from apeGmsh.viewers.diagrams._spring_force import SpringForceDiagram
from apeGmsh.viewers.diagrams._vector_glyph import VectorGlyphDiagram


_EXPECTED: dict[str, str] = {
    "contour":        "nodes",
    "deformed_shape": "nodes",
    "vector_glyph":   "nodes",
    "line_force":     "line_stations",
    "fiber_section":  "fibers",
    "layer_stack":    "layers",
    "gauss_marker":   "gauss",
    "spring_force":   "springs",
}


def test_each_diagram_class_declares_expected_topology() -> None:
    cases = [
        (ContourDiagram,       "contour",        "nodes"),
        (DeformedShapeDiagram, "deformed_shape", "nodes"),
        (VectorGlyphDiagram,   "vector_glyph",   "nodes"),
        (LineForceDiagram,     "line_force",     "line_stations"),
        (FiberSectionDiagram,  "fiber_section",  "fibers"),
        (LayerStackDiagram,    "layer_stack",    "layers"),
        (GaussPointDiagram,    "gauss_marker",   "gauss"),
        (SpringForceDiagram,   "spring_force",   "springs"),
    ]
    for cls, kind, topology in cases:
        assert cls.kind == kind, f"{cls.__name__} kind drift"
        assert cls.topology == topology, (
            f"{cls.__name__} topology drift: got {cls.topology!r}, "
            f"expected {topology!r}"
        )


def test_dialog_kind_to_topology_matches_subclass_attrs() -> None:
    """The dialog dict must match the subclass attrs exactly.

    Importing the dialog requires qtpy at module load — skip if Qt
    isn't available in this environment.
    """
    import pytest
    pytest.importorskip("qtpy")
    from apeGmsh.viewers.ui._add_diagram_dialog import _KIND_TO_TOPOLOGY
    assert _KIND_TO_TOPOLOGY == _EXPECTED
