"""
Beam-column elements — typed primitives for OpenSees line-element commands.

This module ships the priority-1 and priority-2 beam-column primitives for
Phase 2α:

* :class:`elasticBeamColumn` — ``element elasticBeamColumn`` (2-D + 3-D)
* :class:`forceBeamColumn`   — ``element forceBeamColumn`` (single-section
  ``-section sec_tag n_ip`` integration form)
* :class:`dispBeamColumn`    — ``element dispBeamColumn`` (single-section)
* :class:`ElasticTimoshenkoBeam` — ``element ElasticTimoshenkoBeam`` (3-D
  is the primary form; the 2-D form drops ``Iy``, ``J``, and ``Avz``)

OpenSees command shapes
=======================

Per the OpenSees Tcl manual + ``SRC/element/elasticBeamColumn/`` and
``SRC/element/forceBeamColumn/`` parsers:

* 2-D ``elasticBeamColumn``::

    element elasticBeamColumn $tag $iNode $jNode $A $E $Iz $transfTag \
        [-mass $m] [-cMass]

* 3-D ``elasticBeamColumn``::

    element elasticBeamColumn $tag $iNode $jNode $A $E $G $J $Iy $Iz \
        $transfTag [-mass $m] [-cMass]

* 2-D / 3-D ``forceBeamColumn`` (single-section integration)::

    element forceBeamColumn $tag $iNode $jNode $transfTag \
        -section $secTag $numIntgrPts \
        [-mass $m] [-iter $maxIter $tol]

* 2-D / 3-D ``dispBeamColumn`` (single-section integration)::

    element dispBeamColumn $tag $iNode $jNode $numIntgrPts \
        $secTag $transfTag [-mass $m] [-cMass] [-integration $intType]

  (Phase 2α emits the simple form and exposes ``mass``; the optional
  ``-integration`` keyword is deferred — see "Open" notes in the docstring.)

* 2-D ``ElasticTimoshenkoBeam``::

    element ElasticTimoshenkoBeam $tag $iNode $jNode $E $G $A $Iz $Avy \
        $transfTag [-mass $m] [-cMass]

* 3-D ``ElasticTimoshenkoBeam``::

    element ElasticTimoshenkoBeam $tag $iNode $jNode $E $G $A $Jx $Iy $Iz \
        $Avy $Avz $transfTag [-mass $m] [-cMass]

Element fan-out contract
========================

Each ``_emit`` reads the per-element node tags from the emitter context
(set by the bridge via
:func:`apeGmsh.opensees._internal.tag_resolution.set_element_nodes`) and
the dependency tags via
:func:`apeGmsh.opensees._internal.tag_resolution.resolve_tag`. Phase 4 wires
the actual fan-out; for Phase 2α tests install a manual context and
resolver before driving ``_emit`` directly.

Open / deferred for Phase 2α
============================

* **Multi-section integration for** ``forceBeamColumn`` /
  ``dispBeamColumn`` (``-sections n s1 s2 ... sN``) — the typed dataclass
  exposes a single ``section=`` field (per-element single section,
  ``n_ip`` integration points). Per-IP heterogeneous sections is a
  follow-up. Same for the ``-integration`` family of keywords.

* **CatenaryCable** — rare; deferred.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from .._internal.tag_resolution import (
    current_element_nodes,
    damp_args,
    resolve_tag,
)
from .._internal.types import BeamIntegration, Damping, Element, Primitive
from ..transform import Corotational, Linear, PDelta


if TYPE_CHECKING:
    from ..emitter.base import Emitter


__all__ = [
    "ElasticTimoshenkoBeam",
    "dispBeamColumn",
    "elasticBeamColumn",
    "forceBeamColumn",
]


# Union of GeomTransf concrete types accepted on the ``transf=`` parameter.
# We intentionally use a concrete union (rather than the abstract
# :class:`GeomTransf`) so mypy can statically verify a CS-aware concrete
# transform is supplied.
_AnyTransf = Linear | PDelta | Corotational


# ---------------------------------------------------------------------------
# Shared validation helpers (pure functions, no state).
# ---------------------------------------------------------------------------

def _check_two_nodes(type_name: str, nodes: tuple[int, ...]) -> None:
    """Raise ``ValueError`` if ``nodes`` does not have exactly two entries.

    Beam-column elements are line elements with exactly two nodes
    (i, j). The bridge fan-out feeds the per-element node tuple via
    :func:`current_element_nodes`; if a non-line PG sneaks through the
    error surfaces here.
    """
    if len(nodes) == 3:
        # Specific guidance for the most common cause: a 2nd-order
        # continuum mesh (quadratic shell, tet10, etc.) propagated
        # its order to every line entity in the model, including the
        # frame PG.  The mid-side node is a real Gmsh node but
        # OpenSees beams are strictly 2-node 1st-order.  The broker
        # has a verb that demotes Line3 -> 2x Line2 in place.
        # Per ADR 0037, the cleanest fix is to re-mesh the frame PG
        # at order 1 (separate Gmsh model) when the calibrated
        # nonlinear column response matters; ``policy='split'`` is the
        # in-place fix that doubles integration points; ``policy='forbid'``
        # is the build-time invariant lock.
        raise ValueError(
            f"{type_name}: expected 2 node tags (line element i, j), "
            f"got 3.  This PG contains a 3-node line (Gmsh Line3, "
            "typical when the continuum part of the mesh is 2nd-order). "
            "Cleanest fix: re-mesh the frame curves at order 1 in a "
            "separate Gmsh model and merge.  Alternative (in-place): "
            "call g.mesh.editing.split_higher_order_lines(\"YourPG\", "
            "policy='split') BEFORE g.mesh.queries.get_fem_data(...); "
            "this doubles integration points per beam (see ADR 0037). "
            "Use policy='forbid' to lock in 1st-order lines as a build "
            "invariant."
        )
        # NOTE: policy='constrain' is reserved but not implemented
        # this round; see ADR 0037 for the deferred linear-interp
        # mid-node constraint path (gated on upstream OpenSees work
        # per ADR 0036).
    if len(nodes) == 4:
        # Cubic edge from an order-3 continuum mesh (Gmsh Line4,
        # type 26).  No split path is implemented this round —
        # split_higher_order_lines explicitly raises NotImplementedError
        # for dim != 1 / cubic lines.  Pointing the user at the same
        # verb is still the right hint: the deferral message will tell
        # them to re-mesh at order <= 2 (or wait for the line4
        # generalisation tracked in ADR 0037 §Future work).
        raise ValueError(
            f"{type_name}: expected 2 node tags (line element i, j), "
            f"got 4.  This PG contains a 4-node line (Gmsh Line4 / "
            "cubic edge, typical of order-3 continuum meshes).  No "
            "in-place demotion is available this round — re-mesh the "
            "frame curves at order 1 in a separate Gmsh model and "
            "merge.  Future work generalising "
            "g.mesh.editing.split_higher_order_lines to Line4 is "
            "tracked in ADR 0037 §Future work."
        )
    if len(nodes) != 2:
        raise ValueError(
            f"{type_name}: expected 2 node tags (line element i, j), "
            f"got {len(nodes)}."
        )


def _check_optional_mass(type_name: str, mass: float | None) -> None:
    """Reject ``mass`` < 0 if supplied. ``None`` and ``0.0`` are both fine."""
    if mass is not None and mass < 0:
        raise ValueError(
            f"{type_name}: mass must be >= 0 if supplied, got {mass}."
        )


# ---------------------------------------------------------------------------
# elasticBeamColumn — section properties as scalars + geomTransf.
# ---------------------------------------------------------------------------

@dataclass(frozen=True, kw_only=True, slots=True)
class elasticBeamColumn(Element):
    """``element elasticBeamColumn`` — linear-elastic beam-column.

    The 3-D variant is selected by supplying any of the 3-D-only
    parameters ``Iy``, ``G``, ``J``; in that case **all** of ``Iy``,
    ``G``, ``J`` are required. Otherwise the 2-D form is emitted.

    Parameters
    ----------
    pg
        Physical-group name the spec applies to. The bridge fans this
        spec out across the PG's line elements at build time.
    transf
        Geometric transform (Linear / PDelta / Corotational).
    A, E
        Section area and Young's modulus (required in both 2-D and 3-D).
    Iz
        Second moment about local z (required in both forms).
    Iy
        Second moment about local y (3-D only).
    G
        Shear modulus (3-D only).
    J
        Torsional moment (3-D only).
    mass
        Optional ``-mass <m>`` per-unit-length mass.
    c_mass
        If ``True``, append the ``-cMass`` flag (consistent-mass form).
    """

    pg: str
    transf: _AnyTransf
    A: float
    E: float
    Iz: float
    Iy: float | None = None
    G: float | None = None
    J: float | None = None
    mass: float | None = None
    c_mass: bool = False
    damp: Damping | None = None

    def __post_init__(self) -> None:
        if not self.pg:
            raise ValueError("elasticBeamColumn: pg must be a non-empty name.")
        if self.A <= 0:
            raise ValueError(
                f"elasticBeamColumn: A must be > 0, got {self.A}."
            )
        if self.E <= 0:
            raise ValueError(
                f"elasticBeamColumn: E must be > 0, got {self.E}."
            )
        if self.Iz <= 0:
            raise ValueError(
                f"elasticBeamColumn: Iz must be > 0, got {self.Iz}."
            )

        # 3-D variant detection.
        is_3d = any(v is not None for v in (self.Iy, self.G, self.J))
        if is_3d:
            missing = [
                name for name, val in (
                    ("Iy", self.Iy), ("G", self.G), ("J", self.J),
                )
                if val is None
            ]
            if missing:
                raise ValueError(
                    "elasticBeamColumn: 3-D variant requires Iy, G, J. "
                    f"Missing: {', '.join(missing)}."
                )

        if self.Iy is not None and self.Iy <= 0:
            raise ValueError(
                f"elasticBeamColumn: Iy must be > 0, got {self.Iy}."
            )
        if self.G is not None and self.G <= 0:
            raise ValueError(
                f"elasticBeamColumn: G must be > 0, got {self.G}."
            )
        if self.J is not None and self.J <= 0:
            raise ValueError(
                f"elasticBeamColumn: J must be > 0, got {self.J}."
            )

        _check_optional_mass("elasticBeamColumn", self.mass)

    def _is_3d(self) -> bool:
        return any(v is not None for v in (self.Iy, self.G, self.J))

    def dependencies(self) -> tuple[Primitive, ...]:
        if self.damp is not None:
            return (self.transf, self.damp)
        return (self.transf,)

    def _emit(self, emitter: "Emitter", tag: int) -> None:
        nodes = current_element_nodes(emitter)
        _check_two_nodes("elasticBeamColumn", nodes)
        i_node, j_node = nodes
        transf_tag = resolve_tag(emitter, self.transf)

        args: list[int | float | str] = [i_node, j_node]
        if self._is_3d():
            # 3-D: A E G J Iy Iz transfTag.
            assert self.G is not None
            assert self.J is not None
            assert self.Iy is not None
            args.extend(
                [self.A, self.E, self.G, self.J, self.Iy, self.Iz, transf_tag]
            )
        else:
            # 2-D: A E Iz transfTag.
            args.extend([self.A, self.E, self.Iz, transf_tag])

        if self.mass is not None:
            args.extend(["-mass", self.mass])
        if self.c_mass:
            args.append("-cMass")
        args.extend(damp_args(emitter, self.damp))

        emitter.element("elasticBeamColumn", tag, *args)


# ---------------------------------------------------------------------------
# forceBeamColumn — force-based; integration via -section sec_tag n_ip.
# ---------------------------------------------------------------------------

@dataclass(frozen=True, kw_only=True, slots=True)
class forceBeamColumn(Element):
    """``element forceBeamColumn`` — force-based distributed-plasticity beam.

    Modern OpenSees command shape (the only form openseespy parses)::

        element forceBeamColumn tag iNode jNode transfTag integrationTag
                                [-mass m] [-iter maxIter tol]

    The integration rule is a separate registered primitive (see
    :mod:`apeGmsh.opensees.integration`). The user constructs it via
    ``ops.beamIntegration.<Type>(...)`` and references it on the
    element via ``integration=``.

    For the common case of "single section, N Lobatto IPs" the user
    can construct the rule inline:

    .. code-block:: python

        sec = ops.section.Fiber(...)
        integ = ops.beamIntegration.Lobatto(section=sec, n_ip=5)
        ops.element.forceBeamColumn(
            pg="Cols", transf=t, integration=integ,
        )

    For concentrated-plasticity:

    .. code-block:: python

        hinge = ops.beamIntegration.HingeRadau(
            section_i=plastic_sec, lp_i=0.1,
            section_j=plastic_sec, lp_j=0.1,
            section_interior=elastic_sec,
        )
        ops.element.forceBeamColumn(pg="Cols", transf=t, integration=hinge)

    Parameters
    ----------
    pg
        Physical-group name the spec applies to.
    transf
        Geometric transform.
    integration
        The :class:`BeamIntegration` rule that places IPs along the
        element and composes sections.
    mass
        Optional ``-mass <m>`` per-unit-length mass.
    max_iter
        Optional ``-iter`` max-iterations (paired with ``tol``).
    tol
        Optional ``-iter`` tolerance (paired with ``max_iter``).

    Notes
    -----
    OpenSees couples the ``-iter`` flag's two scalars: supplying
    one without the other is a parser error. We mirror that: both
    must be set, or both must be ``None``.
    """

    pg: str
    transf: _AnyTransf
    integration: BeamIntegration
    mass: float | None = None
    max_iter: int | None = None
    tol: float | None = None
    damp: Damping | None = None

    def __post_init__(self) -> None:
        if not self.pg:
            raise ValueError("forceBeamColumn: pg must be a non-empty name.")
        _check_optional_mass("forceBeamColumn", self.mass)

        if (self.max_iter is None) != (self.tol is None):
            raise ValueError(
                "forceBeamColumn: max_iter and tol must be supplied "
                "together (or both omitted)."
            )
        if self.max_iter is not None and self.max_iter <= 0:
            raise ValueError(
                f"forceBeamColumn: max_iter must be > 0, got {self.max_iter}."
            )
        if self.tol is not None and self.tol <= 0:
            raise ValueError(
                f"forceBeamColumn: tol must be > 0, got {self.tol}."
            )

    def dependencies(self) -> tuple[Primitive, ...]:
        if self.damp is not None:
            return (self.integration, self.transf, self.damp)
        return (self.integration, self.transf)

    def _emit(self, emitter: "Emitter", tag: int) -> None:
        nodes = current_element_nodes(emitter)
        _check_two_nodes("forceBeamColumn", nodes)
        i_node, j_node = nodes
        transf_tag = resolve_tag(emitter, self.transf)
        integ_tag = resolve_tag(emitter, self.integration)

        args: list[int | float | str] = [
            i_node, j_node, transf_tag, integ_tag,
        ]
        if self.mass is not None:
            args.extend(["-mass", self.mass])
        if self.max_iter is not None and self.tol is not None:
            args.extend(["-iter", self.max_iter, self.tol])
        args.extend(damp_args(emitter, self.damp))

        emitter.element("forceBeamColumn", tag, *args)


# ---------------------------------------------------------------------------
# dispBeamColumn — displacement-based distributed-plasticity beam.
# ---------------------------------------------------------------------------

@dataclass(frozen=True, kw_only=True, slots=True)
class dispBeamColumn(Element):
    """``element dispBeamColumn`` — displacement-based beam-column.

    Modern OpenSees command shape::

        element dispBeamColumn tag iNode jNode transfTag integrationTag
                               [-mass m] [-cMass]

    Like :class:`forceBeamColumn`, the integration rule is a separate
    registered primitive — see :mod:`apeGmsh.opensees.integration`.

    Parameters
    ----------
    pg
        Physical-group name the spec applies to.
    transf
        Geometric transform.
    integration
        The :class:`BeamIntegration` rule that places IPs along the
        element and composes sections.
    mass
        Optional ``-mass <m>`` per-unit-length mass.
    c_mass
        If ``True``, append the ``-cMass`` flag (consistent-mass form).
    """

    pg: str
    transf: _AnyTransf
    integration: BeamIntegration
    mass: float | None = None
    c_mass: bool = False
    damp: Damping | None = None

    def __post_init__(self) -> None:
        if not self.pg:
            raise ValueError("dispBeamColumn: pg must be a non-empty name.")
        _check_optional_mass("dispBeamColumn", self.mass)

    def dependencies(self) -> tuple[Primitive, ...]:
        if self.damp is not None:
            return (self.integration, self.transf, self.damp)
        return (self.integration, self.transf)

    def _emit(self, emitter: "Emitter", tag: int) -> None:
        nodes = current_element_nodes(emitter)
        _check_two_nodes("dispBeamColumn", nodes)
        i_node, j_node = nodes
        transf_tag = resolve_tag(emitter, self.transf)
        integ_tag = resolve_tag(emitter, self.integration)

        args: list[int | float | str] = [
            i_node, j_node, transf_tag, integ_tag,
        ]
        if self.mass is not None:
            args.extend(["-mass", self.mass])
        if self.c_mass:
            args.append("-cMass")
        args.extend(damp_args(emitter, self.damp))

        emitter.element("dispBeamColumn", tag, *args)


# ---------------------------------------------------------------------------
# ElasticTimoshenkoBeam — shear-flexible elastic beam (closed-form stiffness).
# ---------------------------------------------------------------------------

@dataclass(frozen=True, kw_only=True, slots=True)
class ElasticTimoshenkoBeam(Element):
    """``element ElasticTimoshenkoBeam`` — closed-form Timoshenko beam.

    The 3-D variant is the primary form. The 2-D form is selected when
    none of the 3-D-only parameters (``Iy``, ``J``, ``Avz``) are
    supplied; in that case ``Iz`` and ``Avy`` are sufficient.

    2-D Tcl signature::

        element ElasticTimoshenkoBeam $tag $iNode $jNode $E $G $A $Iz \
            $Avy $transfTag [-mass $m] [-cMass]

    3-D Tcl signature::

        element ElasticTimoshenkoBeam $tag $iNode $jNode $E $G $A $Jx \
            $Iy $Iz $Avy $Avz $transfTag [-mass $m] [-cMass]

    Parameters
    ----------
    pg
        Physical-group name the spec applies to.
    transf
        Geometric transform.
    E, G, A
        Young's modulus, shear modulus, cross-section area.
    Iz, Avy
        Second moment about local z and shear-area in local y (both forms).
    Iy
        Second moment about local y (3-D only).
    J
        Torsional moment (3-D only).
    Avz
        Shear-area in local z (3-D only).
    mass
        Optional ``-mass <m>`` per-unit-length mass.
    c_mass
        If ``True``, append the ``-cMass`` flag.
    """

    pg: str
    transf: _AnyTransf
    E: float
    G: float
    A: float
    Iz: float
    Avy: float
    Iy: float | None = None
    J: float | None = None
    Avz: float | None = None
    mass: float | None = None
    c_mass: bool = False

    def __post_init__(self) -> None:
        if not self.pg:
            raise ValueError(
                "ElasticTimoshenkoBeam: pg must be a non-empty name."
            )
        if self.E <= 0:
            raise ValueError(
                f"ElasticTimoshenkoBeam: E must be > 0, got {self.E}."
            )
        if self.G <= 0:
            raise ValueError(
                f"ElasticTimoshenkoBeam: G must be > 0, got {self.G}."
            )
        if self.A <= 0:
            raise ValueError(
                f"ElasticTimoshenkoBeam: A must be > 0, got {self.A}."
            )
        if self.Iz <= 0:
            raise ValueError(
                f"ElasticTimoshenkoBeam: Iz must be > 0, got {self.Iz}."
            )
        if self.Avy <= 0:
            raise ValueError(
                f"ElasticTimoshenkoBeam: Avy must be > 0, got {self.Avy}."
            )

        # 3-D variant detection — supplying any of Iy / J / Avz selects 3-D
        # and requires all three.
        is_3d = any(v is not None for v in (self.Iy, self.J, self.Avz))
        if is_3d:
            missing = [
                name for name, val in (
                    ("Iy", self.Iy), ("J", self.J), ("Avz", self.Avz),
                )
                if val is None
            ]
            if missing:
                raise ValueError(
                    "ElasticTimoshenkoBeam: 3-D variant requires Iy, J, "
                    f"Avz. Missing: {', '.join(missing)}."
                )

        if self.Iy is not None and self.Iy <= 0:
            raise ValueError(
                f"ElasticTimoshenkoBeam: Iy must be > 0, got {self.Iy}."
            )
        if self.J is not None and self.J <= 0:
            raise ValueError(
                f"ElasticTimoshenkoBeam: J must be > 0, got {self.J}."
            )
        if self.Avz is not None and self.Avz <= 0:
            raise ValueError(
                f"ElasticTimoshenkoBeam: Avz must be > 0, got {self.Avz}."
            )

        _check_optional_mass("ElasticTimoshenkoBeam", self.mass)

    def _is_3d(self) -> bool:
        return any(v is not None for v in (self.Iy, self.J, self.Avz))

    def dependencies(self) -> tuple[Primitive, ...]:
        return (self.transf,)

    def _emit(self, emitter: "Emitter", tag: int) -> None:
        nodes = current_element_nodes(emitter)
        _check_two_nodes("ElasticTimoshenkoBeam", nodes)
        i_node, j_node = nodes
        transf_tag = resolve_tag(emitter, self.transf)

        args: list[int | float | str] = [i_node, j_node]
        if self._is_3d():
            # 3-D: E G A J Iy Iz Avy Avz transfTag.
            assert self.J is not None
            assert self.Iy is not None
            assert self.Avz is not None
            args.extend(
                [
                    self.E, self.G, self.A,
                    self.J, self.Iy, self.Iz,
                    self.Avy, self.Avz,
                    transf_tag,
                ]
            )
        else:
            # 2-D: E G A Iz Avy transfTag.
            args.extend(
                [self.E, self.G, self.A, self.Iz, self.Avy, transf_tag]
            )

        if self.mass is not None:
            args.extend(["-mass", self.mass])
        if self.c_mass:
            args.append("-cMass")

        emitter.element("ElasticTimoshenkoBeam", tag, *args)
