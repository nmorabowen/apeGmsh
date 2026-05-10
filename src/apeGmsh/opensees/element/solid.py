"""
Typed primitives for OpenSees solid (continuum) ``element`` commands.

Phase 2δ ships five classes:

* :class:`FourNodeTetrahedron` — 4-node linear tet
* :class:`TenNodeTetrahedron`  — 10-node quadratic tet
* :class:`stdBrick`            — 8-node linear brick (``Brick`` / ``stdBrick``)
* :class:`FourNodeQuad`        — 4-node 2D plane quad (Tcl token ``quad``)
* :class:`Tri31`               — 3-node 2D plane tri  (Tcl token ``tri31``)

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

Note that for the 2D elements the **Python class name and the OpenSees
type token differ**: ``FourNodeQuad`` emits ``"quad"`` and ``Tri31``
emits ``"tri31"``. The Python class name is the user-facing identity
(matches the OpenSees C++ class name); the lowercase token is what the
OpenSees Tcl parser expects.

Priority-2 classes (``bbarBrick``, ``SSPbrick``, ``SSPquad``) are
deferred — their parameters / penalty parameters need an OpenSees-
expert sign-off before being locked into the typed surface.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from .._internal.tag_resolution import current_element_nodes, resolve_tag
from .._internal.types import Element, NDMaterial, Primitive

if TYPE_CHECKING:
    from ..emitter.base import Emitter


__all__ = [
    "FourNodeQuad",
    "FourNodeTetrahedron",
    "TenNodeTetrahedron",
    "Tri31",
    "stdBrick",
]


_PLANE_TYPES: tuple[str, ...] = ("PlaneStrain", "PlaneStress")


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
