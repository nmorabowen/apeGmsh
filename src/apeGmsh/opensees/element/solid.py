"""
Typed primitives for OpenSees solid (continuum) ``element`` commands.

Phase 2δ ships six classes:

* :class:`FourNodeTetrahedron` — 4-node linear tet
* :class:`TenNodeTetrahedron`  — 10-node quadratic tet
* :class:`stdBrick`            — 8-node linear brick (``Brick`` / ``stdBrick``)
* :class:`FourNodeQuad`        — 4-node 2D plane quad (Tcl token ``quad``)
* :class:`Tri31`               — 3-node 2D plane tri  (Tcl token ``tri31``)
* :class:`SixNodeTri`          — 6-node quadratic 2D plane tri (Tcl token ``tri6n``)

Each class composes one :class:`NDMaterial` and is fan-out-driven by
the bridge: the emitter is given the per-element node tags via
:func:`set_element_nodes` before each ``_emit`` call, and the typed
class reads them with :func:`current_element_nodes`. The material
tag is resolved through :func:`resolve_tag` against the bridge-attached
resolver.

The OpenSees Tcl signatures these classes emit:

* ``element FourNodeTetrahedron $tag $i $j $k $l $matTag <$b1 $b2 $b3>``
* ``element TenNodeTetrahedron  $tag $i ... $r $matTag <$b1 $b2 $b3>``
* ``element stdBrick            $tag $i $j $k $l $m $n $o $p $matTag <$b1 $b2 $b3>``
* ``element quad                $tag $i $j $k $l $thick $type $matTag <$pressure $rho $b1 $b2>``
* ``element tri31               $tag $i $j $k $thick $type $matTag <$pressure $rho $b1 $b2>``
* ``element tri6n               $tag $i $j $k $l $n5 $n6 $thick $type $matTag <$pressure $rho $b1 $b2>``
* ``element BezierTri6          $tag $i $j $k $l $n5 $n6 $thick $type $matTag [-bbar] [-cMass] [-pressure $p] [-rho $r] [-bodyForce $b1 $b2]`` (Ladruno fork)
* ``element BezierTet10         $tag $n1 ... $n10 $matTag [-bbar] [-cMass] [-rho $r] [-bodyForce $b1 $b2 $b3] [-pressure $p]`` (Ladruno fork)

Note that for the 2D elements the **Python class name and the OpenSees
type token differ**: ``FourNodeQuad`` emits ``"quad"``, ``Tri31``
emits ``"tri31"``, and ``SixNodeTri`` emits ``"tri6n"``. The Python
class name is the user-facing identity (matches the OpenSees C++ class
name); the lowercase token is what the OpenSees Tcl parser expects.

Priority-2 classes (``bbarBrick``, ``SSPbrick``, ``SSPquad``) are
deferred — their parameters / penalty parameters need an OpenSees-
expert sign-off before being locked into the typed surface.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import TYPE_CHECKING

from .._internal.tag_resolution import current_element_nodes, resolve_tag
from .._internal.types import Element, NDMaterial, Primitive

if TYPE_CHECKING:
    from ..emitter.base import Emitter


__all__ = [
    "BezierBBarPlaneStressWarning",
    "BezierTet10",
    "BezierTri6",
    "FourNodeQuad",
    "FourNodeTetrahedron",
    "SixNodeTri",
    "TenNodeTetrahedron",
    "Tri31",
    "stdBrick",
]


_PLANE_TYPES: tuple[str, ...] = ("PlaneStrain", "PlaneStress")

# BezierTri6's fork factory (OPS_BezierTri6.cpp:97-101) validates ONLY the
# canonical pair and errors on anything else — it does NOT accept the
# ``*2D`` spellings ``SixNodeTri`` tolerates. Give BezierTri6 its own
# 2-value validator so a ``PlaneStress2D`` string fails at construction.
_PLANE_TYPES_BEZIER_TRI6: tuple[str, ...] = ("PlaneStrain", "PlaneStress")


class BezierBBarPlaneStressWarning(UserWarning):
    """``BezierTri6(bbar=True)`` was requested with ``PlaneStress``.

    The fork element (D5, ``OPS_BezierTri6.cpp``) **warns and disables**
    B-bar under plane stress (the volumetric/deviatoric split is degenerate
    there). apeGmsh mirrors that behavior: the ``-bbar`` flag is dropped
    from the emitted command and the run proceeds. Subclass of
    :class:`UserWarning` so it can be silenced or promoted to an error
    per-call.
    """

# SixNodeTri's upstream parser accepts the ``*2D``-suffixed variants in
# addition to the canonical pair (SixNodeTri.cpp:144). Tri31 and
# FourNodeQuad lock the typed surface to only ``PlaneStrain``/
# ``PlaneStress``; SixNodeTri intentionally diverges per user request
# (some advanced NDMaterials register only the ``*2D`` spelling).
_PLANE_TYPES_SIXNODETRI: tuple[str, ...] = (
    "PlaneStrain", "PlaneStress", "PlaneStrain2D", "PlaneStress2D",
)


# ---------------------------------------------------------------------------
# 3-D solid elements (tetrahedra + brick)
# ---------------------------------------------------------------------------

@dataclass(frozen=True, kw_only=True, slots=True)
class FourNodeTetrahedron(Element):
    """``element FourNodeTetrahedron`` — 4-node linear tetrahedron.

    Tcl signature::

        element FourNodeTetrahedron $tag $i $j $k $l $matTag <$b1 $b2 $b3>

    Parameters
    ----------
    pg
        Physical-group label whose volume cells are realized as
        :class:`FourNodeTetrahedron` elements at fan-out time.
    material
        The :class:`NDMaterial` that supplies the constitutive law.
    body_force
        Optional ``(b1, b2, b3)`` body-force vector. Defaults to
        ``None`` (no body-force triple is appended to the command).
    """

    pg: str
    material: NDMaterial
    body_force: tuple[float, float, float] | None = None

    def dependencies(self) -> tuple[Primitive, ...]:
        return (self.material,)

    def _emit(self, emitter: "Emitter", tag: int) -> None:
        nodes = current_element_nodes(emitter)
        if len(nodes) != 4:
            raise ValueError(
                f"FourNodeTetrahedron: expected 4 node tags, got {len(nodes)}."
            )
        mat_tag = resolve_tag(emitter, self.material)
        args: list[int | float | str] = [*nodes, mat_tag]
        if self.body_force is not None:
            args += list(self.body_force)
        emitter.element("FourNodeTetrahedron", tag, *args)


@dataclass(frozen=True, kw_only=True, slots=True)
class TenNodeTetrahedron(Element):
    """``element TenNodeTetrahedron`` — 10-node quadratic tetrahedron.

    Tcl signature::

        element TenNodeTetrahedron $tag $n1 $n2 ... $n10 $matTag <$b1 $b2 $b3>

    Parameters
    ----------
    pg
        Physical-group label whose volume cells are realized as
        :class:`TenNodeTetrahedron` elements at fan-out time.
    material
        The :class:`NDMaterial` that supplies the constitutive law.
    body_force
        Optional ``(b1, b2, b3)`` body-force vector. Defaults to
        ``None``.
    """

    pg: str
    material: NDMaterial
    body_force: tuple[float, float, float] | None = None

    def dependencies(self) -> tuple[Primitive, ...]:
        return (self.material,)

    def _emit(self, emitter: "Emitter", tag: int) -> None:
        nodes = current_element_nodes(emitter)
        if len(nodes) != 10:
            raise ValueError(
                f"TenNodeTetrahedron: expected 10 node tags, got {len(nodes)}."
            )
        mat_tag = resolve_tag(emitter, self.material)
        args: list[int | float | str] = [*nodes, mat_tag]
        if self.body_force is not None:
            args += list(self.body_force)
        emitter.element("TenNodeTetrahedron", tag, *args)


@dataclass(frozen=True, kw_only=True, slots=True)
class stdBrick(Element):  # noqa: N801 — class name mirrors the OpenSees Tcl token
    """``element stdBrick`` — 8-node trilinear continuum brick.

    Tcl signature::

        element stdBrick $tag $n1 $n2 $n3 $n4 $n5 $n6 $n7 $n8 $matTag \\
            <$b1 $b2 $b3>

    Parameters
    ----------
    pg
        Physical-group label whose volume cells are realized as
        :class:`stdBrick` elements at fan-out time.
    material
        The :class:`NDMaterial` that supplies the constitutive law.
    body_force
        Optional ``(b1, b2, b3)`` body-force vector. Defaults to
        ``None``.
    """

    pg: str
    material: NDMaterial
    body_force: tuple[float, float, float] | None = None

    def dependencies(self) -> tuple[Primitive, ...]:
        return (self.material,)

    def _emit(self, emitter: "Emitter", tag: int) -> None:
        nodes = current_element_nodes(emitter)
        if len(nodes) != 8:
            raise ValueError(
                f"stdBrick: expected 8 node tags, got {len(nodes)}."
            )
        mat_tag = resolve_tag(emitter, self.material)
        args: list[int | float | str] = [*nodes, mat_tag]
        if self.body_force is not None:
            args += list(self.body_force)
        emitter.element("stdBrick", tag, *args)


# ---------------------------------------------------------------------------
# 2-D plane elements (quad, tri)
# ---------------------------------------------------------------------------
#
# The OpenSees Tcl parser for the 4-node plane quad expects the type
# token ``"quad"`` (not ``"FourNodeQuad"``) and the 3-node plane tri
# expects ``"tri31"`` (not ``"Tri31"``). We name the Python classes
# after the OpenSees C++ class names (``FourNodeQuad``, ``Tri31``) so
# they read naturally in user code, and write the lowercase token
# explicitly inside ``_emit``.
# ---------------------------------------------------------------------------

@dataclass(frozen=True, kw_only=True, slots=True)
class FourNodeQuad(Element):
    """``element quad`` — 4-node plane (2D) quadrilateral.

    Tcl signature::

        element quad $tag $i $j $k $l $thick $type $matTag \\
            <$pressure $rho $b1 $b2>

    Parameters
    ----------
    pg
        Physical-group label whose surface cells are realized as
        :class:`FourNodeQuad` elements at fan-out time.
    thickness
        Out-of-plane thickness. Must be strictly positive.
    material
        The :class:`NDMaterial` that supplies the constitutive law
        (must be a 2D-capable nD material — typically
        ``ElasticIsotropic``).
    plane_type
        ``"PlaneStrain"`` or ``"PlaneStress"``. Defaults to
        ``"PlaneStrain"``.
    pressure
        Optional surface pressure. Defaults to ``None``.
    rho
        Optional mass density (overrides the material's ``rho`` for
        this element). Defaults to ``None``.
    body_force
        Optional ``(b1, b2)`` body-force vector. Defaults to ``None``.

    Notes
    -----
    OpenSees's quad parser groups the optional tail
    ``<$pressure $rho $b1 $b2>`` positionally. Per the parser, when any
    of the tail params is supplied the leading ones must be too: this
    class is **lenient** at construction (each may be ``None``
    independently), and the ``_emit`` method substitutes ``0.0`` for
    any missing leading slot when later slots are populated, matching
    OpenSees's "all-or-none in order" expectation.
    """

    pg: str
    thickness: float
    material: NDMaterial
    plane_type: str = "PlaneStrain"
    pressure: float | None = None
    rho: float | None = None
    body_force: tuple[float, float] | None = None

    def __post_init__(self) -> None:
        if self.thickness <= 0:
            raise ValueError(
                f"FourNodeQuad: thickness must be > 0, got "
                f"{self.thickness!r}."
            )
        if self.plane_type not in _PLANE_TYPES:
            raise ValueError(
                f"FourNodeQuad: plane_type must be one of {_PLANE_TYPES}, "
                f"got {self.plane_type!r}."
            )
        if self.rho is not None and self.rho < 0:
            raise ValueError(
                f"FourNodeQuad: rho must be >= 0 if supplied, got "
                f"{self.rho!r}."
            )

    def dependencies(self) -> tuple[Primitive, ...]:
        return (self.material,)

    def _emit(self, emitter: "Emitter", tag: int) -> None:
        nodes = current_element_nodes(emitter)
        if len(nodes) != 4:
            raise ValueError(
                f"FourNodeQuad: expected 4 node tags, got {len(nodes)}."
            )
        mat_tag = resolve_tag(emitter, self.material)
        args: list[int | float | str] = [
            *nodes, self.thickness, self.plane_type, mat_tag,
        ]
        tail = _quad_tri_optional_tail(
            self.pressure, self.rho, self.body_force, body_force_dim=2
        )
        args.extend(tail)
        emitter.element("quad", tag, *args)


@dataclass(frozen=True, kw_only=True, slots=True)
class Tri31(Element):
    """``element tri31`` — 3-node plane (2D) triangle.

    Tcl signature::

        element tri31 $tag $i $j $k $thick $type $matTag \\
            <$pressure $rho $b1 $b2>

    Parameters
    ----------
    pg
        Physical-group label whose surface cells are realized as
        :class:`Tri31` elements at fan-out time.
    thickness
        Out-of-plane thickness. Must be strictly positive.
    material
        The :class:`NDMaterial` that supplies the constitutive law.
    plane_type
        ``"PlaneStrain"`` or ``"PlaneStress"``. Defaults to
        ``"PlaneStrain"``.
    pressure
        Optional surface pressure. Defaults to ``None``.
    rho
        Optional mass density. Defaults to ``None``.
    body_force
        Optional ``(b1, b2)`` body-force vector. Defaults to ``None``.

    Notes
    -----
    See :class:`FourNodeQuad` for the optional-tail handling — Tri31
    follows the same rule.
    """

    pg: str
    thickness: float
    material: NDMaterial
    plane_type: str = "PlaneStrain"
    pressure: float | None = None
    rho: float | None = None
    body_force: tuple[float, float] | None = None

    def __post_init__(self) -> None:
        if self.thickness <= 0:
            raise ValueError(
                f"Tri31: thickness must be > 0, got {self.thickness!r}."
            )
        if self.plane_type not in _PLANE_TYPES:
            raise ValueError(
                f"Tri31: plane_type must be one of {_PLANE_TYPES}, "
                f"got {self.plane_type!r}."
            )
        if self.rho is not None and self.rho < 0:
            raise ValueError(
                f"Tri31: rho must be >= 0 if supplied, got {self.rho!r}."
            )

    def dependencies(self) -> tuple[Primitive, ...]:
        return (self.material,)

    def _emit(self, emitter: "Emitter", tag: int) -> None:
        nodes = current_element_nodes(emitter)
        if len(nodes) != 3:
            raise ValueError(
                f"Tri31: expected 3 node tags, got {len(nodes)}."
            )
        mat_tag = resolve_tag(emitter, self.material)
        args: list[int | float | str] = [
            *nodes, self.thickness, self.plane_type, mat_tag,
        ]
        tail = _quad_tri_optional_tail(
            self.pressure, self.rho, self.body_force, body_force_dim=2
        )
        args.extend(tail)
        emitter.element("tri31", tag, *args)


@dataclass(frozen=True, kw_only=True, slots=True)
class SixNodeTri(Element):
    """``element tri6n`` — 6-node quadratic plane (2D) triangle.

    Tcl signature::

        element tri6n $tag $i $j $k $l $n5 $n6 $thick $type $matTag \\
            <$pressure $rho $b1 $b2>

    Six nodes ordered as three corners (``i, j, k``) followed by the
    three mid-edge nodes ``l`` (mid-edge ``i-j``), ``n5`` (mid-edge
    ``j-k``), and ``n6`` (mid-edge ``k-i``) — identical to the Gmsh
    ``tri6`` (etype 9) ordering. Three integration points at
    barycentric coordinates ``(2/3, 1/6)``, ``(1/6, 2/3)``,
    ``(1/6, 1/6)`` with weights ``1/6``.

    Parameters
    ----------
    pg
        Physical-group label whose surface cells are realized as
        :class:`SixNodeTri` elements at fan-out time.
    thickness
        Out-of-plane thickness. Must be strictly positive.
    material
        The :class:`NDMaterial` that supplies the constitutive law.
    plane_type
        One of ``"PlaneStrain"``, ``"PlaneStress"``,
        ``"PlaneStrain2D"``, or ``"PlaneStress2D"``. Defaults to
        ``"PlaneStrain"``. (SixNodeTri's upstream parser accepts the
        ``*2D`` variants in addition to the canonical pair — see
        :data:`_PLANE_TYPES_SIXNODETRI`.)
    pressure
        Optional surface pressure. Defaults to ``None``.
    rho
        Optional mass density. Defaults to ``None``.
    body_force
        Optional ``(b1, b2)`` body-force vector. Defaults to ``None``.

    Notes
    -----
    See :class:`FourNodeQuad` for the optional-tail handling —
    SixNodeTri follows the same rule.
    """

    pg: str
    thickness: float
    material: NDMaterial
    plane_type: str = "PlaneStrain"
    pressure: float | None = None
    rho: float | None = None
    body_force: tuple[float, float] | None = None

    def __post_init__(self) -> None:
        if self.thickness <= 0:
            raise ValueError(
                f"SixNodeTri: thickness must be > 0, got {self.thickness!r}."
            )
        if self.plane_type not in _PLANE_TYPES_SIXNODETRI:
            raise ValueError(
                f"SixNodeTri: plane_type must be one of "
                f"{_PLANE_TYPES_SIXNODETRI}, got {self.plane_type!r}."
            )
        if self.rho is not None and self.rho < 0:
            raise ValueError(
                f"SixNodeTri: rho must be >= 0 if supplied, got {self.rho!r}."
            )

    def dependencies(self) -> tuple[Primitive, ...]:
        return (self.material,)

    def _emit(self, emitter: "Emitter", tag: int) -> None:
        nodes = current_element_nodes(emitter)
        if len(nodes) != 6:
            raise ValueError(
                f"SixNodeTri: expected 6 node tags, got {len(nodes)}."
            )
        mat_tag = resolve_tag(emitter, self.material)
        args: list[int | float | str] = [
            *nodes, self.thickness, self.plane_type, mat_tag,
        ]
        tail = _quad_tri_optional_tail(
            self.pressure, self.rho, self.body_force, body_force_dim=2
        )
        args.extend(tail)
        emitter.element("tri6n", tag, *args)


@dataclass(frozen=True, kw_only=True, slots=True)
class BezierTri6(Element):
    """``element BezierTri6`` — 6-node quadratic Bézier (Bernstein) triangle.

    A Ladruno-fork plane (2D) continuum element (Kadapa 2018). **Fork-only:**
    the element exists in the Ladruno OpenSees build; emitting the command
    works on any build, but ``ops.run()`` needs the fork (gated at run, not
    emit). On a straight-sided mesh the control points coincide with the
    Gmsh ``tri6`` nodes, so connectivity is used verbatim.

    Tcl signature::

        element BezierTri6 $tag $i $j $k $l $n5 $n6 $thick $type $matTag \\
            [-bbar] [-cMass] [-pressure $p] [-rho $r] [-bodyForce $b1 $b2]

    Unlike :class:`SixNodeTri` (positional ``<pressure rho b1 b2>`` tail),
    every option here is **flag-prefixed** and independently optional.

    Parameters
    ----------
    pg
        Physical-group label whose surface cells are realized as
        :class:`BezierTri6` elements at fan-out time.
    thickness
        Out-of-plane thickness. Must be strictly positive.
    material
        The :class:`NDMaterial` that supplies the constitutive law.
    plane_type
        ``"PlaneStrain"`` or ``"PlaneStress"`` (the fork factory accepts
        **only** these two — not the ``*2D`` spellings). Defaults to
        ``"PlaneStrain"``.
    bbar
        Enable the B-bar (mean-dilatation) formulation. Valid only for
        ``PlaneStrain``/3D; under ``PlaneStress`` the fork warns and
        disables it, so apeGmsh drops the flag and emits a
        :class:`BezierBBarPlaneStressWarning`. Defaults to ``False``.
    consistent_mass
        Emit ``-cMass`` for a consistent (vs lumped) mass matrix.
        Defaults to ``False``.
    pressure
        Optional surface pressure (``-pressure``). Defaults to ``None``.
    rho
        Optional mass density (``-rho``). Defaults to ``None``.
    body_force
        Optional ``(b1, b2)`` body-force vector (``-bodyForce``).
        Defaults to ``None``.
    """

    pg: str
    thickness: float
    material: NDMaterial
    plane_type: str = "PlaneStrain"
    bbar: bool = False
    consistent_mass: bool = False
    pressure: float | None = None
    rho: float | None = None
    body_force: tuple[float, float] | None = None

    def __post_init__(self) -> None:
        if self.thickness <= 0:
            raise ValueError(
                f"BezierTri6: thickness must be > 0, got {self.thickness!r}."
            )
        if self.plane_type not in _PLANE_TYPES_BEZIER_TRI6:
            raise ValueError(
                f"BezierTri6: plane_type must be one of "
                f"{_PLANE_TYPES_BEZIER_TRI6}, got {self.plane_type!r}."
            )
        if self.rho is not None and self.rho < 0:
            raise ValueError(
                f"BezierTri6: rho must be >= 0 if supplied, got {self.rho!r}."
            )
        if self.bbar and self.plane_type == "PlaneStress":
            warnings.warn(
                "BezierTri6: B-bar is not valid under PlaneStress (the fork "
                "warns and disables it); dropping the -bbar flag. Use "
                "PlaneStrain for a B-bar formulation.",
                BezierBBarPlaneStressWarning,
                stacklevel=2,
            )

    def dependencies(self) -> tuple[Primitive, ...]:
        return (self.material,)

    def _emit(self, emitter: "Emitter", tag: int) -> None:
        nodes = current_element_nodes(emitter)
        if len(nodes) != 6:
            raise ValueError(
                f"BezierTri6: expected 6 node tags, got {len(nodes)}."
            )
        mat_tag = resolve_tag(emitter, self.material)
        args: list[int | float | str] = [
            *nodes, self.thickness, self.plane_type, mat_tag,
        ]
        # Flag-prefixed tail (each independently optional). D5: B-bar is
        # dropped under PlaneStress (warned in __post_init__).
        if self.bbar and self.plane_type != "PlaneStress":
            args.append("-bbar")
        if self.consistent_mass:
            args.append("-cMass")
        if self.pressure is not None:
            args += ["-pressure", self.pressure]
        if self.rho is not None:
            args += ["-rho", self.rho]
        if self.body_force is not None:
            args += ["-bodyForce", *self.body_force]
        emitter.element("BezierTri6", tag, *args)


@dataclass(frozen=True, kw_only=True, slots=True)
class BezierTet10(Element):
    """``element BezierTet10`` — 10-node quadratic Bézier (Bernstein) tet.

    A Ladruno-fork 3D continuum element (Kadapa 2018), the tetrahedral
    sibling of :class:`BezierTri6`. **Fork-only:** emitting the command
    works on any build, but ``ops.run()`` needs the fork. On a
    straight-sided mesh the control points coincide with the Gmsh
    ``tet10`` nodes, so connectivity is used verbatim — the 10 nodes are
    4 corners then 6 mid-edge nodes in :class:`TenNodeTetrahedron` order
    ``(1-2, 2-3, 1-3, 1-4, 3-4, 2-4)``, byte-identical to Gmsh etype 11.

    Tcl signature::

        element BezierTet10 $tag $n1 ... $n10 $matTag \\
            [-bbar] [-cMass] [-rho $r] [-bodyForce $b1 $b2 $b3] [-pressure $p]

    Every option is flag-prefixed and independently optional. Unlike
    :class:`BezierTri6` there is **no plane-stress degeneracy**, so B-bar
    is always valid (no warn-and-drop guard).

    Parameters
    ----------
    pg
        Physical-group label whose volume cells are realized as
        :class:`BezierTet10` elements at fan-out time.
    material
        The 3D :class:`NDMaterial` that supplies the constitutive law.
    bbar
        Enable the B-bar (near-incompressibility) formulation. Defaults
        to ``False``.
    consistent_mass
        Emit ``-cMass`` for a consistent (vs lumped) mass matrix.
        Defaults to ``False``.
    rho
        Optional mass density (``-rho``); else taken from the material.
        Defaults to ``None``.
    body_force
        Optional ``(b1, b2, b3)`` body-force vector (``-bodyForce``).
        Defaults to ``None``.
    pressure
        Optional volume-pressure term (``-pressure``). Defaults to
        ``None``.
    """

    pg: str
    material: NDMaterial
    bbar: bool = False
    consistent_mass: bool = False
    rho: float | None = None
    body_force: tuple[float, float, float] | None = None
    pressure: float | None = None

    def __post_init__(self) -> None:
        if self.rho is not None and self.rho < 0:
            raise ValueError(
                f"BezierTet10: rho must be >= 0 if supplied, got {self.rho!r}."
            )

    def dependencies(self) -> tuple[Primitive, ...]:
        return (self.material,)

    def _emit(self, emitter: "Emitter", tag: int) -> None:
        nodes = current_element_nodes(emitter)
        if len(nodes) != 10:
            raise ValueError(
                f"BezierTet10: expected 10 node tags, got {len(nodes)}."
            )
        mat_tag = resolve_tag(emitter, self.material)
        args: list[int | float | str] = [*nodes, mat_tag]
        # Flag-prefixed tail (each independently optional). No D5 guard —
        # B-bar is always valid in 3D (no plane-stress degeneracy, D5').
        if self.bbar:
            args.append("-bbar")
        if self.consistent_mass:
            args.append("-cMass")
        if self.pressure is not None:
            args += ["-pressure", self.pressure]
        if self.rho is not None:
            args += ["-rho", self.rho]
        if self.body_force is not None:
            args += ["-bodyForce", *self.body_force]
        emitter.element("BezierTet10", tag, *args)


# ---------------------------------------------------------------------------
# Helper for the shared ``<pressure rho b1 b2>`` tail
# ---------------------------------------------------------------------------

def _quad_tri_optional_tail(
    pressure: float | None,
    rho: float | None,
    body_force: tuple[float, float] | None,
    *,
    body_force_dim: int,
) -> list[float]:
    """Build the ``<pressure rho b1 b2>`` tail for plane-quad / tri31.

    The OpenSees parser expects these positionally and in order:
    if a later slot is supplied, the earlier slots must be too.
    Missing leading slots are filled with ``0.0``. If nothing is
    supplied, return an empty list.
    """
    nothing_supplied = (
        pressure is None and rho is None and body_force is None
    )
    if nothing_supplied:
        return []
    p = 0.0 if pressure is None else pressure
    r = 0.0 if rho is None else rho
    if body_force is None:
        # pressure and/or rho present but no body_force: emit just
        # whichever leading params are needed to reach the last
        # populated slot.
        if rho is not None:
            return [p, r]
        return [p]
    if len(body_force) != body_force_dim:
        raise ValueError(
            f"body_force must have {body_force_dim} components, "
            f"got {len(body_force)}."
        )
    return [p, r, *body_force]
