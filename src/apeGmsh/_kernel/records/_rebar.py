"""Resolved structural-rebar-element records (ADR 0067 P5.2 / B1).

A :class:`RebarElementRecord` is the *auto-emit* intent captured when a
cage is placed with ``g.rebar.place(..., emit_elements=True)``: it names
the bar's physical group plus everything the bridge needs to realise the
bar's OWN structural element (``CorotTruss`` for ``"truss"``,
``dispBeamColumn`` for ``"beam"``) along the bar — separate from the
``LadrunoEmbeddedRebar`` coupling, which carries no axial stiffness.

Unlike :class:`~apeGmsh._kernel.records._constraints.ReinforceTieRecord`,
this record needs **no mesh inverse-map**: the PG label, element kind,
material name, and area are all known at authoring (``place``) time. The
bridge fans it across the PG's line cells at build (``emit_rebar_elements``),
resolving the material by **name** (Option B, like the reinforce bond).

Solver-agnostic — no OpenSees imports here.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RebarElementRecord:
    """One auto-emitted structural rebar element spec.

    Attributes
    ----------
    pg
        The bar's physical-group label (the curve PG ``place`` created);
        the bridge fans the spec across this PG's 2-node line cells.
    element
        ``"truss"`` → ``CorotTruss``; ``"beam"`` → ``dispBeamColumn``
        (beam emission is B1b — the emit pass raises ``NotImplementedError``
        for ``"beam"`` until then).
    material
        The bar's uniaxial-material **name** (resolved to a tag at emit
        via the bridge name-alias map, Option B).
    area
        Cross-sectional area ``π·d_b²/4``. Must be > 0.
    role
        The bar role (``"longitudinal"`` / ``"tie"`` / …) — diagnostics only.
    connectivity
        The bar's resolved 2-node line cells ``((i, j), …)`` as node tags,
        extracted from the live mesh at ``get_fem_data`` time (the dim-1
        cells are dropped from a dim-3 ``FEMData``, so they are carried
        here). The bridge emits one structural element per pair.
    """

    pg: str
    element: str
    material: str
    area: float
    role: str = ""
    connectivity: "tuple[tuple[int, int], ...]" = ()
