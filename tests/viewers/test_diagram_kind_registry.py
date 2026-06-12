"""Diagram-kind registry guard (ADR 0058 S0).

The declarative registry (``apeGmsh.viewers.diagrams._kinds``) replaced
four hand-maintained per-kind tables (dialog ``_KINDS``, catalog
``_KIND_DEFINITIONS``, session ``_KIND_TO_STYLE``, preset
``KIND_TO_STYLE_CLASS``) that had already drifted: labels disagreed,
and the session/preset maps silently lacked ``loads`` / ``reactions``,
so those layers never survived a save/restore. This guard keeps the
registry the single source of truth:

* every concrete ``Diagram`` subclass in the package is registered —
  a new diagram that forgets its ``@register_diagram_kind`` fails here,
  not silently in the Add dialog;
* registration is internally consistent (kind_id mirrors the class
  attr, style/labels/orders don't collide);
* every registered kind round-trips through the session spec codec —
  the regression that motivated the consolidation.
"""
from __future__ import annotations

import inspect

import apeGmsh.viewers.diagrams as diagrams_pkg
from apeGmsh.viewers.diagrams._base import Diagram, DiagramSpec
from apeGmsh.viewers.diagrams._kinds import all_kinds, kind_def
from apeGmsh.viewers.diagrams._selectors import SlabSelector
from apeGmsh.viewers.diagrams._session import (
    deserialize_spec, serialize_spec,
)
from apeGmsh.viewers.diagrams._styles import DiagramStyle


def _package_diagram_classes() -> list[type]:
    """Concrete Diagram subclasses exported by the diagrams package."""
    out = []
    for name in dir(diagrams_pkg):
        obj = getattr(diagrams_pkg, name)
        if (
            inspect.isclass(obj)
            and issubclass(obj, Diagram)
            and obj is not Diagram
            and not inspect.isabstract(obj)
        ):
            out.append(obj)
    return sorted(out, key=lambda c: c.__name__)


def test_every_package_diagram_class_is_registered() -> None:
    registered = {k.diagram_class for k in all_kinds()}
    missing = [
        cls.__name__ for cls in _package_diagram_classes()
        if cls not in registered
    ]
    assert not missing, (
        f"{missing} are Diagram subclasses without a "
        "@register_diagram_kind decoration — they would be invisible "
        "to the Add dialog, the kind catalog, and session restore."
    )


def test_registry_is_internally_consistent() -> None:
    kinds = all_kinds()
    assert kinds, "registry is empty — package import didn't populate it"

    # kind_id mirrors the class-level attr (read from it at decoration).
    for k in kinds:
        assert k.kind_id == k.diagram_class.kind
        assert issubclass(k.diagram_class, Diagram)
        assert issubclass(k.style_class, DiagramStyle)
        assert k.label.strip()

    # No label / order collisions (both drive UI listings).
    labels = [k.label for k in kinds]
    assert len(labels) == len(set(labels)), f"duplicate labels: {labels}"
    orders = [k.order for k in kinds]
    assert len(orders) == len(set(orders)), f"duplicate orders: {orders}"

    # data_topology=None is reserved for kinds with no Results
    # composite to enumerate; everything else mirrors the class attr.
    for k in kinds:
        if k.data_topology is not None:
            assert k.data_topology == k.diagram_class.topology


def test_known_kind_ids_are_present() -> None:
    # Self-check against vacuous discovery: pin the shipped family.
    ids = {k.kind_id for k in all_kinds()}
    assert {
        "contour", "deformed_shape", "line_force", "fiber_section",
        "layer_stack", "vector_glyph", "gauss_marker", "spring_force",
        "loads", "reactions", "section_cut",
    } <= ids


def test_every_kind_round_trips_through_session_codec() -> None:
    """The session discriminator must cover every registered kind.

    This is the bug that motivated S0: the hand-maintained
    ``_KIND_TO_STYLE`` lacked ``loads`` and ``reactions``, so those
    diagrams silently vanished from restored sessions.
    """
    for k in all_kinds():
        if k.kind_id == "section_cut":
            # SectionCutStyle requires a loaded SectionCutDef — its
            # creation flow never builds a default style. Covered by
            # the section-cut session tests.
            continue
        spec = DiagramSpec(
            kind=k.kind_id,
            selector=SlabSelector(component="displacement_z"),
            style=k.make_default_style("displacement_z"),
            stage_id=None,
        )
        back = deserialize_spec(serialize_spec(spec))
        assert back.kind == k.kind_id
        assert type(back.style) is k.style_class


def test_default_styles_construct() -> None:
    for k in all_kinds():
        if k.kind_id == "section_cut":
            continue  # requires a loaded SectionCutDef (see above)
        style = k.make_default_style("bending_moment_y")
        assert isinstance(style, k.style_class)


def test_kind_def_lookup() -> None:
    assert kind_def("contour") is not None
    assert kind_def("contour").diagram_class.__name__ == "ContourDiagram"
    assert kind_def("not_a_kind") is None
