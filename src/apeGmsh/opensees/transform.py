"""
Orientation primitives + ``geomTransf`` typed primitives.

This module re-exports the orientation classes from
:mod:`apeGmsh.opensees._orientation` (ADR 0010) and ships the typed
``geomTransf`` primitives (Linear, PDelta, Corotational) that
replace the legacy ``add_geom_transf`` flow.

OpenSees command shape::

    geomTransf <Linear|PDelta|Corotational> tag <vecxzX vecxzY vecxzZ> \
        <-jntOffset dXi dYi dZi dXj dYj dZj>

``jnt_offset=`` (all three types) adds the rigid joint offsets, in global
coordinates (2-D: ``-jntOffset dXi dYi dXj dYj``): a constant pair, or an
:class:`~apeGmsh.opensees.joint_offset.InsertionPoint` rule the bridge
resolves per element (see :mod:`apeGmsh.opensees.joint_offset`). A 3-D
``Corotational`` refuses one: ``CorotCrdTransf3d`` zeroes the offsets
with a warning ("rigid joint zones not implemented yet").

Members: ETABS puts a frame object's end offsets at its two ends. The mesh
alone cannot tell a member meshed into several elements from a continuous
girder of several members, so an offset that is not the same transverse
shift at both ends (an end length, an axial component, unequal ends)
refuses where two of a transform's elements meet turning less than 30 deg
(a straight or faceted member) until elements carry their source curve
(#2551). One-element members, corners of 30 deg or more, nodes where three
or more of its elements meet, and an equal transverse shift on every
element are unaffected.

Loads on an offset beam: a ``Plain.from_model`` line load (global,
``beamUniform`` / ``line_load``) covers the full node-to-node length, as
ETABS/CSI load a frame object: the share over each rigid end zone goes
to its joint as a force plus its moment, the rest to the flexible span.
``Plain.beam_uniform`` is OpenSees' own ``eleLoad -beamUniform`` and,
like it, loads only the flexible element between the offset ends.

In 3D, ``vecxz`` is required: a vector in the local x-z plane that
fixes the section's strong axis. In 2D, ``vecxz`` is omitted and
local axes are derived from the element's node order: the build
pipeline emits the bare ``geomTransf <Type> tag`` form, dropping an
explicit ``vecxz`` along global Z and refusing any other vector (Tcl's
2-D ``geomTransf`` rejects trailing args).

Two ways to pin orientation at construction:

* **Explicit ``vecxz``** — one 3-tuple applied to every element that
  references this transform. Suitable for prismatic frames where all
  members share the same orientation (e.g. all columns + all
  beams in a regular Cartesian frame).

* **Orientation (``orientation=``)** — the bridge derives a
  per-element ``vecxz`` from the orientation triad and the element's
  tangent at build time. Curved members emit one ``geomTransf`` line
  per distinct vecxz (per ADR 0010). Suitable for ring beams, dome
  ribs, arches.

Exactly one of ``orientation`` / ``vecxz`` must be supplied (or
neither, for 2D models — the bridge will reject "neither" in 3D at
build time).

Phase 1D scope
--------------
The typed primitives accept BOTH ``orientation`` and explicit
``vecxz`` at construction. ``_emit`` only handles the ``vecxz`` case
for now; orientation-derived per-element vecxz fan-out lands when the
build pipeline materializes it (Phase 4). When ``orientation`` is set
and ``vecxz`` is ``None``, ``_emit`` raises
:class:`NotImplementedError` with a clear message.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, TypeAlias

from ._orientation import (
    AlongBeam,
    Cartesian,
    Cylindrical,
    Spherical,
    resolve_vecxz,
)

from ._internal.types import GeomTransf, Primitive
from .joint_offset import (
    InsertionPoint,
    JntOffset,
    flatten_jnt_offset,
    insertion_point_offsets,
)

if TYPE_CHECKING:
    from .emitter.base import Emitter


__all__ = [
    # Orientation classes (ADR 0010)
    "AlongBeam",
    "Cartesian",
    "Cylindrical",
    "Spherical",
    "resolve_vecxz",
    # GeomTransf primitives (Phase 1D)
    "Orientation",
    "Linear",
    "PDelta",
    "Corotational",
    # Joint offsets (-jntOffset)
    "JntOffset",
    "InsertionPoint",
    "insertion_point_offsets",
]


# Union of orientation classes for typed kwargs. The orientation
# classes are not part of the typed-primitive hierarchy (they're
# standalone helpers); we accept any of them or None.
Orientation: TypeAlias = Cartesian | Cylindrical | Spherical | AlongBeam


# ---------------------------------------------------------------------------
# Shared validation helper
# ---------------------------------------------------------------------------

def _check_transform_args(
    type_name: str,
    orientation: Orientation | None,
    vecxz: tuple[float, float, float] | None,
    roll_deg: float,
    jnt_offset: "JntOffset | InsertionPoint | None" = None,
) -> None:
    """Validate transform construction arguments.

    Two rules:

    1. **orientation XOR vecxz** — supply at most one. "Neither" is
       permitted (the 2D-model construction path; the bridge rejects
       it at build time when the model is 3D). Both at once is
       ambiguous.

    2. **roll_deg only with orientation** — ``roll_deg`` rotates the
       per-element ``vecxz`` about the element's tangent (which varies
       per element). With an explicit ``vecxz=``, the tangent is not
       known at construction time and applying the same Rodrigues
       rotation to a constant vector would silently produce a wrong
       result. The orientation= path is the supported way to roll an
       asymmetric section about each beam's tangent.
    """
    if orientation is not None and vecxz is not None:
        raise ValueError(
            f"{type_name}: supply either orientation= or vecxz=, not both."
        )
    if vecxz is not None and roll_deg != 0.0:
        raise ValueError(
            f"{type_name}: roll_deg= has no defined meaning with an "
            "explicit vecxz= (the rotation axis is the element's "
            "tangent, which varies per element; vecxz= is a single "
            "constant). Use orientation= to roll asymmetric sections "
            "about each beam's tangent, or set roll_deg=0 if vecxz= "
            "is the correct constant direction."
        )
    flat: tuple[float, ...] = ()
    if jnt_offset is not None and not isinstance(jnt_offset, InsertionPoint):
        flat = flatten_jnt_offset(jnt_offset, type_name)
    # Only a 3-component pair is visibly 3-D here: a 2-D model may carry
    # vecxz=(0, 0, 1) (the bridge accepts global Z), so a vecxz or an
    # orientation alone defers the 3-D refusal to the build, which knows
    # ndm (``_check_jnt_offset``).
    if type_name == "Corotational" and len(flat) == 6:
        raise ValueError(
            "Corotational: jnt_offset= is not supported in 3-D. OpenSees' "
            "CorotCrdTransf3d (stock and Ladruno) prints \"rigid joint zones "
            "not implemented yet\" and zeroes the offsets, so the element "
            "would solve without them. Use Linear or PDelta, or model the "
            "offsets with explicit nodes and rigidLink beam."
        )


def _jnt_kwargs(
    jnt_offset: "JntOffset | InsertionPoint | None", where: str,
) -> dict[str, tuple[float, ...]]:
    """``{"jnt_offset": flat}`` for a constant non-zero pair, else ``{}``.

    An all-zero pair emits no ``-jntOffset`` (the deck of an offset-free
    transform). An :class:`InsertionPoint` rule is resolved per element
    by the bridge fan-out; a primitive emitted on its own cannot.
    """
    if jnt_offset is None:
        return {}
    if isinstance(jnt_offset, InsertionPoint):
        raise NotImplementedError(
            f"{where}: an InsertionPoint jnt_offset is resolved per element "
            "by the bridge build pipeline; emit through apeSees, or pass a "
            "constant ((dXi, dYi, dZi), (dXj, dYj, dZj)) pair."
        )
    flat = flatten_jnt_offset(jnt_offset, where)
    return {"jnt_offset": flat} if any(v != 0.0 for v in flat) else {}


# ---------------------------------------------------------------------------
# Linear — small-displacement linearized transform
# ---------------------------------------------------------------------------

@dataclass(frozen=True, kw_only=True, slots=True)
class Linear(GeomTransf):
    """``geomTransf Linear`` — small-displacement linearized transform.

    Either ``orientation`` (the bridge derives per-element ``vecxz``
    at build time via the orientation rule from ADR 0010) or
    ``vecxz`` (one vector applied to all elements that reference this
    transform) must be supplied for 3D models. Supplying neither is
    permitted at construction (2D-only path); the bridge will raise
    if used in a 3D model.

    Parameters
    ----------
    orientation
        Orientation field used to derive ``vecxz`` per element at
        build time. Mutually exclusive with ``vecxz``.
    vecxz
        Explicit local-x-z vector (one 3-tuple). Mutually exclusive
        with ``orientation``. If the same prismatic frame produces
        inconsistent vecxz signs in the deck (a tangent-direction
        artifact of how gmsh ordered the element nodes), flip the
        offending PG with ``g.mesh.editing.reverse(dim_tags="MyPG")``
        before constructing the bridge.
    roll_deg
        Rotation about the element tangent applied AFTER the
        orientation rule produces a candidate ``vecxz``. Useful for
        asymmetric sections (channels, angles) on curved members.
        **Only valid with** ``orientation=``: the rotation axis is the
        element's per-element tangent, which only the orientation path
        knows. Combined with an explicit ``vecxz=`` it raises
        :class:`ValueError`.
    jnt_offset
        Rigid joint offsets, ``-jntOffset``: a constant global pair
        ``((dXi, dYi, dZi), (dXj, dYj, dZj))`` (2-D: 2-tuples) from each
        node to its element end, or an
        :class:`~apeGmsh.opensees.joint_offset.InsertionPoint` rule (ETABS
        cardinal point, joint and end-length offsets) resolved per
        element at build, one ``geomTransf`` per distinct result. An
        all-zero pair emits no flag.
    """

    orientation: Orientation | None = None
    vecxz: tuple[float, float, float] | None = None
    roll_deg: float = 0.0
    jnt_offset: JntOffset | InsertionPoint | None = None

    def __post_init__(self) -> None:
        _check_transform_args(
            "Linear", self.orientation, self.vecxz, self.roll_deg, self.jnt_offset)

    def _emit(self, emitter: "Emitter", tag: int) -> None:
        # Phase 1D scope: ``_emit`` requires an explicit ``vecxz``.
        # When ``orientation`` is set, the bridge build pipeline
        # (Phase 4) is responsible for substituting one or more
        # concrete ``vecxz`` vectors prior to ``_emit`` — see ADR 0010
        # for the one-line-per-distinct-vecxz behavior on curved
        # members.
        if self.vecxz is None:
            raise NotImplementedError(
                "Linear._emit: orientation-derived per-element vecxz "
                "fan-out is the bridge build pipeline's responsibility "
                "(Phase 4). For Phase 1D unit tests, construct with "
                "explicit vecxz=."
            )
        emitter.geomTransf(
            "Linear", tag, *self.vecxz,
            **_jnt_kwargs(self.jnt_offset, "Linear._emit"))

    def dependencies(self) -> tuple[Primitive, ...]:
        return ()


# ---------------------------------------------------------------------------
# PDelta — adds P-Δ effects
# ---------------------------------------------------------------------------

@dataclass(frozen=True, kw_only=True, slots=True)
class PDelta(GeomTransf):
    """``geomTransf PDelta`` — linear transform plus P-Δ effects.

    See :class:`Linear` for parameter semantics — PDelta shares the
    same construction shape, differing only in the OpenSees command
    type token emitted.
    """

    orientation: Orientation | None = None
    vecxz: tuple[float, float, float] | None = None
    roll_deg: float = 0.0
    jnt_offset: JntOffset | InsertionPoint | None = None

    def __post_init__(self) -> None:
        _check_transform_args(
            "PDelta", self.orientation, self.vecxz, self.roll_deg, self.jnt_offset)

    def _emit(self, emitter: "Emitter", tag: int) -> None:
        if self.vecxz is None:
            raise NotImplementedError(
                "PDelta._emit: orientation-derived per-element vecxz "
                "fan-out is the bridge build pipeline's responsibility "
                "(Phase 4). For Phase 1D unit tests, construct with "
                "explicit vecxz=."
            )
        emitter.geomTransf(
            "PDelta", tag, *self.vecxz,
            **_jnt_kwargs(self.jnt_offset, "PDelta._emit"))

    def dependencies(self) -> tuple[Primitive, ...]:
        return ()


# ---------------------------------------------------------------------------
# Corotational — finite-displacement rotations
# ---------------------------------------------------------------------------

@dataclass(frozen=True, kw_only=True, slots=True)
class Corotational(GeomTransf):
    """``geomTransf Corotational`` — finite-displacement transform.

    See :class:`Linear` for parameter semantics — Corotational shares
    the same construction shape, differing only in the OpenSees
    command type token emitted.
    """

    orientation: Orientation | None = None
    vecxz: tuple[float, float, float] | None = None
    roll_deg: float = 0.0
    jnt_offset: JntOffset | InsertionPoint | None = None

    def __post_init__(self) -> None:
        _check_transform_args(
            "Corotational", self.orientation, self.vecxz, self.roll_deg,
            self.jnt_offset)

    def _emit(self, emitter: "Emitter", tag: int) -> None:
        if self.vecxz is None:
            raise NotImplementedError(
                "Corotational._emit: orientation-derived per-element "
                "vecxz fan-out is the bridge build pipeline's "
                "responsibility (Phase 4). For Phase 1D unit tests, "
                "construct with explicit vecxz=."
            )
        emitter.geomTransf(
            "Corotational", tag, *self.vecxz,
            **_jnt_kwargs(self.jnt_offset, "Corotational._emit"))

    def dependencies(self) -> tuple[Primitive, ...]:
        return ()
