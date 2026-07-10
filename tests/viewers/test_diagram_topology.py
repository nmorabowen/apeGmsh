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
from apeGmsh.viewers.diagrams._fiber_section import FiberSectionDiagram
from apeGmsh.viewers.diagrams._gauss_marker import GaussPointDiagram
from apeGmsh.viewers.diagrams._layer_stack import LayerStackDiagram
from apeGmsh.viewers.diagrams._line_force import LineForceDiagram
from apeGmsh.viewers.diagrams._principal_glyph import PrincipalDirectionDiagram
from apeGmsh.viewers.diagrams._sand import SandDiagram
from apeGmsh.viewers.diagrams._spring_force import SpringForceDiagram
from apeGmsh.viewers.diagrams._vector_glyph import VectorGlyphDiagram


_EXPECTED: dict[str, str] = {
    "contour":        "nodes",
    "vector_glyph":   "nodes",
    "line_force":     "line_stations",
    "fiber_section":  "fibers",
    "layer_stack":    "layers",
    "gauss_marker":   "gauss",
    "spring_force":   "springs",
    # Loads is special: the "topology" string is virtual — data
    # comes from ``fem.nodes.loads`` rather than a Results composite.
    "loads":          "loads",
    # Reactions read from results.nodes (recorded), so topology="nodes".
    "reactions":      "nodes",
    # Sand interpolates a nodal component to interior grains.
    "sand":           "nodes",
    # Principal-direction arrows read the gauss stress/strain tensor.
    "principal_glyph": "gauss",
}


def test_each_diagram_class_declares_expected_topology() -> None:
    cases = [
        (ContourDiagram,       "contour",        "nodes"),
        (VectorGlyphDiagram,   "vector_glyph",   "nodes"),
        (LineForceDiagram,     "line_force",     "line_stations"),
        (FiberSectionDiagram,  "fiber_section",  "fibers"),
        (LayerStackDiagram,    "layer_stack",    "layers"),
        (GaussPointDiagram,    "gauss_marker",   "gauss"),
        (SpringForceDiagram,   "spring_force",   "springs"),
        (SandDiagram,          "sand",           "nodes"),
        (PrincipalDirectionDiagram, "principal_glyph", "gauss"),
    ]
    for cls, kind, topology in cases:
        assert cls.kind == kind, f"{cls.__name__} kind drift"
        assert cls.topology == topology, (
            f"{cls.__name__} topology drift: got {cls.topology!r}, "
            f"expected {topology!r}"
        )


def test_registry_data_topology_matches_subclass_attrs() -> None:
    """The registry's per-kind data_topology must match the subclass
    attrs exactly (the dialog's old hand-written ``_KIND_TO_TOPOLOGY``
    dict is registry-derived since ADR 0058 S0). ``section_cut``
    registers ``data_topology=None`` (no Results composite to
    enumerate) and is therefore absent — same as the old dict, which
    skipped classes without a usable topology attribute.
    """
    from apeGmsh.viewers.diagrams._kinds import all_kinds
    derived = {
        k.kind_id: k.data_topology
        for k in all_kinds() if k.data_topology is not None
    }
    assert derived == _EXPECTED
