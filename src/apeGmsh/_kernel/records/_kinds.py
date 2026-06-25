"""
Kind constants — linter-friendly enumerations of record ``kind`` values.

These classes expose each valid string literal that can appear on a
constraint or load record's ``kind`` field. Using them instead of bare
string literals at call sites gives autocomplete, typo protection,
and a single point of definition if the wire values ever need to
change.

The values are plain ``str`` (not ``enum.Enum``) so equality against
raw strings — e.g. ``rec.kind == "rigid_beam"`` in user notebooks —
continues to work unchanged.

Lives in :mod:`apeGmsh.mesh.records` because constraint and load
records are defined there; the sub-composites on
:class:`~apeGmsh.mesh.FEMData.NodeComposite` re-expose these classes
as ``fem.nodes.constraints.Kind`` / ``fem.nodes.loads.Kind`` for
convenient lookup.
"""

from __future__ import annotations

from typing import ClassVar


class ConstraintKind:
    """String constants for constraint record ``kind`` values.

    Exposed as ``Kind`` on each constraint sub-composite so the user
    gets autocomplete right where they need it::

        K = fem.nodes.constraints.Kind
        for c in fem.nodes.constraints.pairs():
            if c.kind == K.RIGID_BEAM:
                ops.rigidLink("beam", c.master_node, c.slave_node)

    The constants are typed as ``ClassVar[str]`` so Pylance/mypy
    recognise them as static attributes (not instance fields).
    """
    EQUAL_DOF:          ClassVar[str] = "equal_dof"
    EQUAL_DOF_MIXED:    ClassVar[str] = "equal_dof_mixed"
    RIGID_BEAM:         ClassVar[str] = "rigid_beam"
    RIGID_BEAM_STIFF:   ClassVar[str] = "rigid_beam_stiff"
    RIGID_ROD:          ClassVar[str] = "rigid_rod"
    RIGID_DIAPHRAGM:    ClassVar[str] = "rigid_diaphragm"
    RIGID_BODY:         ClassVar[str] = "rigid_body"
    KINEMATIC_COUPLING: ClassVar[str] = "kinematic_coupling"
    PENALTY:            ClassVar[str] = "penalty"
    NODE_TO_SURFACE:    ClassVar[str] = "node_to_surface"
    NODE_TO_SURFACE_SPRING: ClassVar[str] = "node_to_surface_spring"
    TIE:                ClassVar[str] = "tie"
    DISTRIBUTING:       ClassVar[str] = "distributing"
    EMBEDDED:           ClassVar[str] = "embedded"
    TIED_CONTACT:       ClassVar[str] = "tied_contact"
    MORTAR:             ClassVar[str] = "mortar"

    # Classification for rendering / routing.
    NODE_PAIR_KINDS: ClassVar[frozenset[str]] = frozenset({
        "equal_dof", "equal_dof_mixed", "rigid_beam", "rigid_beam_stiff",
        "rigid_rod", "rigid_diaphragm", "rigid_body", "kinematic_coupling",
        "penalty", "node_to_surface",
    })
    SURFACE_KINDS: ClassVar[frozenset[str]] = frozenset({
        "tie", "distributing", "embedded", "tied_contact", "mortar",
    })


class LoadKind:
    """String constants for load record ``kind`` values.

    Exposed as ``Kind`` on each load sub-composite::

        K = fem.nodes.loads.Kind
    """
    NODAL:   ClassVar[str] = "nodal"
    ELEMENT: ClassVar[str] = "element"


__all__ = ["ConstraintKind", "LoadKind"]
