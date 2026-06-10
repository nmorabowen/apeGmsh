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

Body-force semantics (``body_force=``)
--------------------------------------
The optional ``body_force`` triple/​pair becomes the element's constructor
``b1 b2 b3`` (or ``-bodyForce`` for the Bézier/Ladruno forms).  Verified
against the OpenSees source (``Brick.cpp:1268-1274`` /
``FourNodeQuad.cpp:890-906``) and a live single-element probe, its semantics
are:

* **Always-on.**  The continuum elements integrate the constructor ``b``
  into the resisting force **every step**, with **no** ``eleLoad`` and **no**
  load pattern (the ``applyLoad == 0`` branch).  An ``eleLoad -selfWeight``
  only *replaces* it with a pattern-factored copy; it is not required to
  switch it on.  So ``body_force`` is **not** rampable, **not** scalable by a
  time series, and **not** frozen by ``loadConst`` — it is on from the first
  step of the first stage onward.
* **Force density, not acceleration.**  ``b`` has units of force per unit
  volume (2-D: per unit area·thickness).  For gravity supply
  ``b = (0, 0, -ρ g)`` directly — the element does **not** multiply by the
  material density.
* **Consistent load vector.**  The contribution is the full Gauss-quadrature
  ``∫ Nᵀ b dV`` (the variationally consistent body-force vector), identical to
  what ``eleLoad -selfWeight`` would assemble.

For a *staged* gravity solve (gravity ramped with — or held before — an
``s.initial_stress`` K0 install, then frozen by ``loadConst``), prefer the
pattern-controlled route instead: author ``g.loads.gravity(...)`` at the
geometry and import it with ``p.from_model(case)`` inside the stage.  Mixing
the two — a ``body_force`` element whose nodes also receive a ``from_model``
gravity case — double-counts self-weight and raises
:class:`~apeGmsh.opensees._internal.build.WarnBodyForceDoubleCount` at build.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import TYPE_CHECKING

from .._internal.tag_resolution import (
    current_element_nodes,
    damp_args,
    resolve_tag,
)
from .._internal.types import Damping, Element, NDMaterial, Primitive

if TYPE_CHECKING:
    from ..emitter.base import Emitter


__all__ = [
    "BezierBBarPlaneStressWarning",
    "BezierTet10",
    "BezierTri6",
    "FourNodeQuad",
    "FourNodeTetrahedron",
    "LadrunoBrick",
    "LadrunoQuad",
    "SixNodeTri",
    "TenNodeTetrahedron",
    "Tri31",
    "stdBrick",
]


_PLANE_TYPES: tuple[str, ...] = ("PlaneStrain", "PlaneStress")

# Ladruno solid geometry/kinematics methods (the ``-geom`` selector shared by
# BezierTet10 and LadrunoBrick): linear (small strain), corot (large rotation /
# small strain, EICR), finite (large strain, updated-Lagrangian).
_GEOM_METHODS: tuple[str, ...] = ("linear", "corot", "finite")

# LadrunoBrick F-bar variant (the ``-fbar`` selector, only meaningful with
# B-bar + finite). centroid = single centroid J₀; mean_dilatation = volume-avg.
_FBAR_MODES: tuple[str, ...] = ("centroid", "mean_dilatation")

# LadrunoBrick ``-formulation`` selector (strain interpolation / anti-locking):
# std (full integration), bbar (mean-dilatation), uri (1-pt reduced + hourglass),
# ssp (stabilized single-point), eas (true Simo-Rifai enhanced assumed strain).
_BRICK_FORMULATIONS: tuple[str, ...] = ("std", "bbar", "uri", "ssp", "eas")

# LadrunoBrick formulations that support ``-geom corot|finite`` and ``-damp``
# (OPS_LadrunoBrick.cpp parse guards): only std and bbar.
_BRICK_GEOM_FORMULATIONS: frozenset[str] = frozenset({"std", "bbar"})

# LadrunoBrick ``-hourglass`` type (uri only): viscous (explicit-only rate
# damping), stiffness (Flanagan-Belytschko, implicit-safe), physical
# (Belytschko-Bindeman assumed strain).
_BRICK_HOURGLASS_TYPES: tuple[str, ...] = ("viscous", "stiffness", "physical")

# LadrunoQuad ``-formulation`` selector (ADR 25): std (full 2×2 Gauss, the
# honest default), bbar (mean-dilatation B-bar, PlaneStrain only), ssp
# (stabilized single-point — the cheap nonlinear/explicit quad). ``eas``
# (Simo-Rifai Q1/E4) is reserved and parser-refused in the fork (ADR 25
# Phase 3); apeGmsh rejects it at construction rather than emitting a deck the
# fork would reject at parse.
_QUAD_FORMULATIONS: tuple[str, ...] = ("std", "bbar", "ssp")

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
        Applied **every step**, no load pattern needed; ``b`` is a force
        density (gravity: ``-ρ g``).  See *Body-force semantics* in the
        module docstring.
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
        ``None``.  Applied **every step**, no load pattern needed; ``b``
        is a force density (gravity: ``-ρ g``).  See *Body-force
        semantics* in the module docstring.
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
        ``None``.  Applied **every step**, no load pattern needed; ``b``
        is a force density (gravity: ``-ρ g``).  See *Body-force
        semantics* in the module docstring.
    """

    pg: str
    material: NDMaterial
    body_force: tuple[float, float, float] | None = None
    damp: Damping | None = None

    def dependencies(self) -> tuple[Primitive, ...]:
        if self.damp is not None:
            return (self.material, self.damp)
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
        elif self.damp is not None:
            # The brick parser greedily reads up to three doubles (b1 b2 b3)
            # before scanning flags (Brick.cpp:77-87); zero-fill the body
            # force so a trailing -damp is not misread as a double.
            args += [0.0, 0.0, 0.0]
        args.extend(damp_args(emitter, self.damp))
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
        Applied **every step**, no load pattern needed; ``b`` is a force
        density (gravity: ``-ρ g``).  See *Body-force semantics* in the
        module docstring.

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
    damp: Damping | None = None

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
        if self.damp is not None:
            return (self.material, self.damp)
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
        # When -damp follows, the quad parser greedily reads four tail
        # doubles before scanning flags (FourNodeQuad.cpp:99-110), so the
        # full <pressure rho b1 b2> tail must be emitted first.
        tail = _quad_tri_optional_tail(
            self.pressure, self.rho, self.body_force, body_force_dim=2,
            force_full=self.damp is not None,
        )
        args.extend(tail)
        args.extend(damp_args(emitter, self.damp))
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
        Applied **every step**, no load pattern needed; ``b`` is a force
        density (gravity: ``-ρ g``).  See *Body-force semantics* in the
        module docstring.

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
        Applied **every step**, no load pattern needed; ``b`` is a force
        density (gravity: ``-ρ g``).  See *Body-force semantics* in the
        module docstring.

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
        Defaults to ``None``.  Applied **every step**, no load pattern
        needed; ``b`` is a force density (gravity: ``-ρ g``).  See
        *Body-force semantics* in the module docstring.
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
            [-bbar] [-cMass] [-rho $r] [-bodyForce $b1 $b2 $b3] [-pressure $p] \\
            [-geom linear|corot|finite] [-fbar centroid|mean_dilatation]

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
        Under ``geom="finite"`` this must be a finite-strain material
        driven by ``setTrialF(F)`` (e.g. ``nDMaterial LogStrain``); the
        fork rejects a small-strain material there at run time (gated at
        run, not emit — apeGmsh does not model finite-strain-ness).
    bbar
        Enable the B-bar (near-incompressibility) formulation. Defaults
        to ``False``. With ``geom="finite"`` this selects the **F-bar**
        element (the large-strain volumetric-locking cure), whose tangent
        is generally **unsymmetric** — use an unsymmetric solver
        (``FullGeneral``/``UmfPack``/``SparseGEN``).
    consistent_mass
        Emit ``-cMass`` for a consistent (vs lumped) mass matrix.
        Defaults to ``False``.
    rho
        Optional mass density (``-rho``); else taken from the material.
        Defaults to ``None``.
    body_force
        Optional ``(b1, b2, b3)`` body-force vector (``-bodyForce``).
        Defaults to ``None``.  Applied **every step**, no load pattern
        needed; ``b`` is a force density (gravity: ``-ρ g``).  See
        *Body-force semantics* in the module docstring.
    pressure
        Optional volume-pressure term (``-pressure``). Defaults to
        ``None``. **Rejected** under ``geom="corot"`` or ``"finite"``
        (unvalidated in the fork v1; ``OPS_BezierTet10.cpp``).
    geom
        Geometry/kinematics method — ``"linear"`` (default, small strain),
        ``"corot"`` (large rotation / small strain, EICR), or ``"finite"``
        (large strain, updated-Lagrangian). The default elides the flag so
        existing decks stay byte-identical.
    fbar
        F-bar variant, only meaningful with ``bbar=True`` and
        ``geom="finite"``: ``"centroid"`` (default, single centroid J₀) or
        ``"mean_dilatation"`` (volume-averaged J̄). Setting it to a
        non-default value without ``bbar``+``finite`` raises.
    """

    pg: str
    material: NDMaterial
    bbar: bool = False
    consistent_mass: bool = False
    rho: float | None = None
    body_force: tuple[float, float, float] | None = None
    pressure: float | None = None
    geom: str = "linear"
    fbar: str = "centroid"

    def __post_init__(self) -> None:
        if self.rho is not None and self.rho < 0:
            raise ValueError(
                f"BezierTet10: rho must be >= 0 if supplied, got {self.rho!r}."
            )
        if self.geom not in _GEOM_METHODS:
            raise ValueError(
                f"BezierTet10: geom must be one of {_GEOM_METHODS}, "
                f"got {self.geom!r}."
            )
        if self.fbar not in _FBAR_MODES:
            raise ValueError(
                f"BezierTet10: fbar must be one of {_FBAR_MODES}, "
                f"got {self.fbar!r}."
            )
        # The fork rejects the +z pressure hack under corot/finite at parse
        # time (OPS_BezierTet10.cpp:213-228) — unvalidated against the
        # co-rotating / large-strain load contract in v1.
        if self.pressure is not None and self.geom in ("corot", "finite"):
            raise ValueError(
                f"BezierTet10: pressure is not supported under "
                f"geom={self.geom!r} (the fork rejects it in v1); "
                "use geom='linear' or drop pressure."
            )
        # -fbar only does anything with the F-bar element (bbar + finite);
        # the fork warns it is a no-op otherwise. apeGmsh fails loud instead.
        if self.fbar != "centroid" and not (self.bbar and self.geom == "finite"):
            raise ValueError(
                "BezierTet10: fbar='mean_dilatation' requires bbar=True and "
                "geom='finite' (F-bar is off otherwise)."
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
        # Geometry method + F-bar variant. Defaults (linear / centroid) are
        # elided so existing decks stay byte-identical.
        if self.geom != "linear":
            args += ["-geom", self.geom]
        if self.fbar != "centroid":
            args += ["-fbar", self.fbar]
        emitter.element("BezierTet10", tag, *args)


@dataclass(frozen=True, kw_only=True, slots=True)
class LadrunoBrick(Element):
    """``element LadrunoBrick`` — unified 8-node hexahedral solid.

    A Ladruno-fork 3D continuum element (class tag ``33002``) that exposes
    the anti-locking treatment as a single ``-formulation`` selector
    (``std``/``bbar``/``uri``/``ssp``/``eas``) and an orthogonal ``-geom``
    selector for the kinematic regime (``linear``/``corot``/``finite``).
    One class reproduces upstream ``Brick``/``bbarBrick``/``SSPbrick``
    bit-for-bit where they overlap and adds the cheap explicit hex.
    **Fork-only:** emitting the command works on any build, but
    ``ops.run()`` needs the fork (gated at run, not emit).

    Tcl signature::

        element LadrunoBrick $tag $n1 ... $n8 $matTag \\
            [-formulation std|bbar|uri|ssp|eas] [-geom linear|corot|finite] \\
            [-hourglass viscous|stiffness|physical [$coeff]] \\
            [-lumped] [-b $bx $by $bz] [-damp $dampTag]

    The 8 nodes are the standard ``Brick`` order (nodes 0–3 on the
    ``ζ=-1`` face, 4–7 on ``ζ=+1``), byte-identical to Gmsh hex8 (etype 5).

    Parameters
    ----------
    pg
        Physical-group label whose volume cells are realized as
        :class:`LadrunoBrick` elements at fan-out time.
    material
        The 3D :class:`NDMaterial` constitutive law. Under
        ``geom="finite"`` this must be a finite-strain material driven by
        ``setTrialF(F)`` (e.g. ``nDMaterial LogStrain``); the fork rejects a
        small-strain material there at run time.
    formulation
        Strain interpolation / anti-locking treatment — one of
        ``"std"`` (default), ``"bbar"``, ``"uri"``, ``"ssp"``, ``"eas"``.
    geom
        Kinematics method — ``"linear"`` (default), ``"corot"`` (large
        rotation / small strain), ``"finite"`` (large strain). ``corot``
        and ``finite`` support **only** ``std``/``bbar`` (the fork rejects
        ``uri``/``ssp``/``eas`` there); with ``finite`` + ``bbar`` you get
        the F-bar element (unsymmetric tangent — use an unsymmetric solver).
    hourglass
        Hourglass-control flavour for ``formulation="uri"`` only —
        ``"viscous"`` (explicit-only), ``"stiffness"``, or ``"physical"``.
        Must be ``None`` for every other formulation. Defaults to ``None``.
    hourglass_coeff
        Optional numeric coefficient for the hourglass control. Requires
        ``hourglass`` to be set. Defaults to ``None``.
    lumped
        Emit ``-lumped`` for a diagonal (row-sum) mass matrix — required
        for explicit integrators. Defaults to ``False``.
    body_force
        Optional ``(bx, by, bz)`` body force per unit volume (``-b``).
        Defaults to ``None``.  Applied **every step**, no load pattern
        needed (gravity: ``-ρ g``).  See *Body-force semantics* in the
        module docstring.
    damp
        Optional :class:`Damping` object attached via the element's
        ``-damp`` flag (ADR 0053 element-flag attach). The fork honours
        ``-damp`` **only** with ``std``/``bbar`` formulations; apeGmsh
        rejects it for the others rather than letting the fork silently
        drop it. Defaults to ``None``.
    """

    pg: str
    material: NDMaterial
    formulation: str = "std"
    geom: str = "linear"
    hourglass: str | None = None
    hourglass_coeff: float | None = None
    lumped: bool = False
    body_force: tuple[float, float, float] | None = None
    damp: Damping | None = None

    def __post_init__(self) -> None:
        if self.formulation not in _BRICK_FORMULATIONS:
            raise ValueError(
                f"LadrunoBrick: formulation must be one of "
                f"{_BRICK_FORMULATIONS}, got {self.formulation!r}."
            )
        if self.geom not in _GEOM_METHODS:
            raise ValueError(
                f"LadrunoBrick: geom must be one of {_GEOM_METHODS}, "
                f"got {self.geom!r}."
            )
        # Hourglass control is meaningful only for the reduced-integration
        # (uri) kernel (OPS_LadrunoBrick.cpp — the other kernels ignore it).
        if self.hourglass is not None:
            if self.hourglass not in _BRICK_HOURGLASS_TYPES:
                raise ValueError(
                    f"LadrunoBrick: hourglass must be one of "
                    f"{_BRICK_HOURGLASS_TYPES}, got {self.hourglass!r}."
                )
            if self.formulation != "uri":
                raise ValueError(
                    "LadrunoBrick: hourglass is only valid with "
                    f"formulation='uri', got formulation={self.formulation!r}."
                )
        if self.hourglass_coeff is not None and self.hourglass is None:
            raise ValueError(
                "LadrunoBrick: hourglass_coeff requires hourglass to be set."
            )
        # corot/finite ship std|bbar only (the single-point/EAS kernels under
        # large rotation/strain are deferred follow-ups — fork parse reject).
        if (
            self.geom in ("corot", "finite")
            and self.formulation not in _BRICK_GEOM_FORMULATIONS
        ):
            raise ValueError(
                f"LadrunoBrick: geom={self.geom!r} supports only "
                "formulation='std' or 'bbar' "
                f"(got {self.formulation!r}); uri/ssp/eas are reserved there."
            )
        # The fork wires -damp only through the std/bbar kernel and drops it
        # (with a warning) for the others. apeGmsh fails loud instead.
        if (
            self.damp is not None
            and self.formulation not in _BRICK_GEOM_FORMULATIONS
        ):
            raise ValueError(
                "LadrunoBrick: -damp is only supported with formulation='std' "
                f"or 'bbar' (got {self.formulation!r})."
            )
        # A finite-strain material (LogStrain / LadrunoJ2Finite / InitDefGrad)
        # is driven by setTrialF(F) — under geom != "finite" the element never
        # calls the F-interface, so it would integrate zero stress. The fork
        # rejects this at run; apeGmsh fails loud at construction (the forward
        # case — geom="finite" needing a finite material — is left to the fork,
        # since apeGmsh only marks the finite materials it models).
        if self.geom != "finite" and getattr(
            self.material, "is_finite_strain", False
        ):
            raise ValueError(
                f"LadrunoBrick: geom={self.geom!r} cannot use the finite-strain "
                f"material {type(self.material).__name__!r} (a "
                "FiniteStrainNDMaterial is driven by setTrialF and yields zero "
                "stress without the F-interface); use geom='finite'."
            )

    def dependencies(self) -> tuple[Primitive, ...]:
        if self.damp is not None:
            return (self.material, self.damp)
        return (self.material,)

    def _emit(self, emitter: "Emitter", tag: int) -> None:
        nodes = current_element_nodes(emitter)
        if len(nodes) != 8:
            raise ValueError(
                f"LadrunoBrick: expected 8 node tags, got {len(nodes)}."
            )
        mat_tag = resolve_tag(emitter, self.material)
        args: list[int | float | str] = [*nodes, mat_tag]
        # Every option is flag-prefixed and order-independent (the fork
        # factory scans a while-loop). The std/linear defaults are elided.
        if self.formulation != "std":
            args += ["-formulation", self.formulation]
        if self.geom != "linear":
            args += ["-geom", self.geom]
        if self.hourglass is not None:
            args += ["-hourglass", self.hourglass]
            if self.hourglass_coeff is not None:
                args.append(self.hourglass_coeff)
        if self.lumped:
            args.append("-lumped")
        if self.body_force is not None:
            args += ["-b", *self.body_force]
        # All options above are flag-prefixed, so a trailing -damp parses
        # cleanly with no zero-fill (unlike stdBrick's greedy body-force tail).
        args.extend(damp_args(emitter, self.damp))
        emitter.element("LadrunoBrick", tag, *args)


@dataclass(frozen=True, kw_only=True, slots=True)
class LadrunoQuad(Element):
    """``element LadrunoQuad`` — unified 4-node plane (2D) continuum.

    A Ladruno-fork plane-stress / plane-strain element (class tag ``33007``)
    that exposes the anti-locking treatment as a single ``-formulation``
    selector — ``std`` (full 2×2 Gauss), ``bbar`` (mean-dilatation B-bar,
    plane-strain only), ``ssp`` (stabilized single-point) — the 2D sibling of
    :class:`LadrunoBrick`. It reduces to upstream ``FourNodeQuad`` (``std``),
    ``ConstantPressureVolumeQuad`` (``bbar``) and ``SSPquad`` (``ssp``) where
    they overlap, and overrides ``getCharacteristicLength`` with the true
    element area (``√A``) so crack-band materials regularize correctly.
    **Fork-only:** emitting the command works on any build, but ``ops.run()``
    needs the fork (gated at run, not emit).

    Tcl signature::

        element LadrunoQuad $tag $n1 $n2 $n3 $n4 $matTag \\
            [-formulation std|bbar|ssp] [-type PlaneStrain|PlaneStress] \\
            [-thick $t] [-rho $r] [-body $bx $by] [-pressure $p]

    The 4 nodes are the standard CCW quad order, byte-identical to Gmsh quad4
    (etype 3). The model **must** be ``ndm 2, ndf 2`` (the fork parser refuses
    anything else); apeGmsh brackets the element block with ``model -ndf 2`` so
    it survives a mixed-ndf envelope.

    Parameters
    ----------
    pg
        Physical-group label whose surface cells are realized as
        :class:`LadrunoQuad` elements at fan-out time.
    material
        The 2D :class:`NDMaterial` constitutive law. It must support the
        requested plane view (``getCopy("PlaneStrain")`` / ``"PlaneStress"``);
        a material that returns 0 for the view fails at run.
    thickness
        Out-of-plane thickness (``-thick``). Must be strictly positive.
    formulation
        Anti-locking treatment — ``"std"`` (default), ``"bbar"``, ``"ssp"``.
        ``"bbar"`` is **plane-strain only** (volumetric locking is a
        plane-strain / incompressible phenomenon); ``"eas"`` is reserved in
        the fork and rejected here.
    plane_type
        ``"PlaneStrain"`` (default) or ``"PlaneStress"``.
    pressure
        Optional uniform edge pressure (``-pressure``). Defaults to ``None``.
    rho
        Optional mass density override (``-rho``; else the material's
        ``getRho()``). Defaults to ``None``.
    body_force
        Optional ``(bx, by)`` body force per unit volume (``-body``).
        Defaults to ``None``. Applied **every step**, no load pattern needed
        (gravity: ``-ρ g``). See *Body-force semantics* in the module
        docstring.
    """

    pg: str
    material: NDMaterial
    thickness: float
    formulation: str = "std"
    plane_type: str = "PlaneStrain"
    pressure: float | None = None
    rho: float | None = None
    body_force: tuple[float, float] | None = None

    def __post_init__(self) -> None:
        if self.thickness <= 0:
            raise ValueError(
                f"LadrunoQuad: thickness must be > 0, got {self.thickness!r}."
            )
        # ``eas`` is a reserved-but-refused token in the fork (ADR 25 Phase 3):
        # give a targeted message rather than the generic "must be one of".
        if self.formulation == "eas":
            raise ValueError(
                "LadrunoQuad: formulation 'eas' is reserved but not yet "
                "implemented in the fork (ADR 25 Phase 3); use 'std', 'bbar', "
                "or 'ssp'."
            )
        if self.formulation not in _QUAD_FORMULATIONS:
            raise ValueError(
                f"LadrunoQuad: formulation must be one of {_QUAD_FORMULATIONS}, "
                f"got {self.formulation!r}."
            )
        if self.plane_type not in _PLANE_TYPES:
            raise ValueError(
                f"LadrunoQuad: plane_type must be one of {_PLANE_TYPES}, "
                f"got {self.plane_type!r}."
            )
        # The fork refuses bbar + PlaneStress at parse (volumetric locking is a
        # plane-strain issue, so B-bar there is meaningless). Fail loud here.
        if self.formulation == "bbar" and self.plane_type == "PlaneStress":
            raise ValueError(
                "LadrunoQuad: formulation='bbar' is for plane_type='PlaneStrain' "
                "only (volumetric locking is a plane-strain / incompressible "
                "phenomenon); use 'std' or 'ssp' for PlaneStress."
            )
        if self.rho is not None and self.rho < 0:
            raise ValueError(
                f"LadrunoQuad: rho must be >= 0 if supplied, got {self.rho!r}."
            )

    def dependencies(self) -> tuple[Primitive, ...]:
        return (self.material,)

    def _emit(self, emitter: "Emitter", tag: int) -> None:
        nodes = current_element_nodes(emitter)
        if len(nodes) != 4:
            raise ValueError(
                f"LadrunoQuad: expected 4 node tags, got {len(nodes)}."
            )
        mat_tag = resolve_tag(emitter, self.material)
        args: list[int | float | str] = [*nodes, mat_tag]
        # Every option is flag-prefixed and order-independent (the fork factory
        # scans a while-loop). The std / PlaneStrain defaults are elided; the
        # required thickness is always emitted.
        if self.formulation != "std":
            args += ["-formulation", self.formulation]
        if self.plane_type != "PlaneStrain":
            args += ["-type", self.plane_type]
        args += ["-thick", self.thickness]
        if self.rho is not None:
            args += ["-rho", self.rho]
        if self.body_force is not None:
            args += ["-body", *self.body_force]
        if self.pressure is not None:
            args += ["-pressure", self.pressure]
        emitter.element("LadrunoQuad", tag, *args)


# ---------------------------------------------------------------------------
# Helper for the shared ``<pressure rho b1 b2>`` tail
# ---------------------------------------------------------------------------

def _quad_tri_optional_tail(
    pressure: float | None,
    rho: float | None,
    body_force: tuple[float, float] | None,
    *,
    body_force_dim: int,
    force_full: bool = False,
) -> list[float]:
    """Build the ``<pressure rho b1 b2>`` tail for plane-quad / tri31.

    The OpenSees parser expects these positionally and in order:
    if a later slot is supplied, the earlier slots must be too.
    Missing leading slots are filled with ``0.0``. If nothing is
    supplied, return an empty list.

    ``force_full`` zero-fills the **complete** tail (pressure, rho, and
    all body-force components). It is required when a trailing ``-damp``
    flag follows: the OpenSees quad parser greedily reads up to four
    doubles for the tail *before* scanning for flags
    (``FourNodeQuad.cpp:99-110``), so a ``-damp`` token left inside that
    window would be misread as a double. Emitting the full tail consumes
    exactly those slots and leaves ``-damp`` for the flag loop.
    """
    if force_full:
        p = 0.0 if pressure is None else pressure
        r = 0.0 if rho is None else rho
        bf = body_force if body_force is not None else (0.0,) * body_force_dim
        if len(bf) != body_force_dim:
            raise ValueError(
                f"body_force must have {body_force_dim} components, "
                f"got {len(bf)}."
            )
        return [p, r, *bf]
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
