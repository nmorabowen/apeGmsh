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
from .._internal.types import (
    BeamIntegration,
    Damping,
    Element,
    NDMaterial,
    Primitive,
    UniaxialMaterial,
)
from ..transform import Corotational, Linear, PDelta
from .frame_modifiers import (
    FrameModifiers,
    ModifiedCopies,
    check_frame_modifiers,
    emitted_integration,
    is_reference_stub,
    scaled,
    scaled_mass,
)


if TYPE_CHECKING:
    from ..emitter.base import Emitter


__all__ = [
    "ElasticTimoshenkoBeam",
    "dispBeamColumn",
    "elasticBeamColumn",
    "forceBeamColumn",
    "LadrunoDispBeamColumn",
    "LadrunoForceBeamColumn",
    "LadrunoIMKBeam",
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


def _check_integration_modifiers(
    type_name: str,
    integration: BeamIntegration,
    mods: FrameModifiers | None,
    mass: float | None,
) -> None:
    """Refuse ``mods`` a force-/disp-based element cannot apply.

    The section-side checks (an ``ElasticSection`` in every slot, the
    property each non-unit modifier needs) run by building the modified
    rule once, in a throwaway memo: the copy the element references is
    made when a bridge registers it (``_internal/frame_modifier_copies``).
    """
    check_frame_modifiers(
        type_name, mods, lacks={}, has_mass=mass is not None,
    )
    if (mods is not None and not mods.is_identity
            and not is_reference_stub(integration)):
        ModifiedCopies().integration(type_name, integration, mods)


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
    modifiers
        Optional :class:`~.frame_modifiers.FrameModifiers`. The emitted
        ``A``, ``Iz``, ``Iy``, ``J`` and ``-mass`` are the declared values
        times ``A``, ``Iz``, ``Iy``, ``J`` and ``mass``; ``E`` and ``G`` are
        never scaled. An Euler-Bernoulli element has no shear area, so a
        non-unit ``Avy`` / ``Avz`` is refused, as are ``Iy`` / ``J`` in 2-D.
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
    modifiers: FrameModifiers | None = None

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
        euler = ("an Euler-Bernoulli element carries no shear deformation; "
                 "use ElasticTimoshenkoBeam")
        lacks = {"Avy": euler, "Avz": euler}
        if not is_3d:
            lacks["Iy"] = "a 2-D element has no second moment about local y"
            lacks["J"] = "a 2-D element has no torsion"
        check_frame_modifiers(
            "elasticBeamColumn", self.modifiers, lacks=lacks,
            has_mass=self.mass is not None,
        )

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

        m = self.modifiers or FrameModifiers()
        A, Iz = scaled(self.A, m.A), scaled(self.Iz, m.Iz)
        args: list[int | float | str] = [i_node, j_node]
        if self._is_3d():
            # 3-D: A E G J Iy Iz transfTag.
            assert self.G is not None
            assert self.J is not None
            assert self.Iy is not None
            args.extend(
                [A, self.E, self.G, scaled(self.J, m.J),
                 scaled(self.Iy, m.Iy), Iz, transf_tag]
            )
        else:
            # 2-D: A E Iz transfTag.
            args.extend([A, self.E, Iz, transf_tag])

        mass = scaled_mass(self.mass, self.modifiers)
        if mass is not None:
            args.extend(["-mass", mass])
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
    modifiers
        Optional :class:`~.frame_modifiers.FrameModifiers`. Every section
        of ``integration`` must then be an ``ElasticSection``: the element
        references a modified copy of the rule and of each section (one
        copy per distinct section and modifier set; the declared ones are
        untouched), and ``-mass`` is scaled by ``mass``. The bridge
        registers the copies when it registers the element.

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
    modifiers: FrameModifiers | None = None

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
        _check_integration_modifiers(
            "forceBeamColumn", self.integration, self.modifiers, self.mass,
        )

    def dependencies(self) -> tuple[Primitive, ...]:
        integ = emitted_integration(self)
        if self.damp is not None:
            return (integ, self.transf, self.damp)
        return (integ, self.transf)

    def _emit(self, emitter: "Emitter", tag: int) -> None:
        nodes = current_element_nodes(emitter)
        _check_two_nodes("forceBeamColumn", nodes)
        i_node, j_node = nodes
        transf_tag = resolve_tag(emitter, self.transf)
        integ_tag = resolve_tag(emitter, emitted_integration(self))

        args: list[int | float | str] = [
            i_node, j_node, transf_tag, integ_tag,
        ]
        mass = scaled_mass(self.mass, self.modifiers)
        if mass is not None:
            args.extend(["-mass", mass])
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
    modifiers
        Optional :class:`~.frame_modifiers.FrameModifiers`, as on
        :class:`forceBeamColumn`.
    """

    pg: str
    transf: _AnyTransf
    integration: BeamIntegration
    mass: float | None = None
    c_mass: bool = False
    damp: Damping | None = None
    modifiers: FrameModifiers | None = None

    def __post_init__(self) -> None:
        if not self.pg:
            raise ValueError("dispBeamColumn: pg must be a non-empty name.")
        _check_optional_mass("dispBeamColumn", self.mass)
        _check_integration_modifiers(
            "dispBeamColumn", self.integration, self.modifiers, self.mass,
        )

    def dependencies(self) -> tuple[Primitive, ...]:
        integ = emitted_integration(self)
        if self.damp is not None:
            return (integ, self.transf, self.damp)
        return (integ, self.transf)

    def _emit(self, emitter: "Emitter", tag: int) -> None:
        nodes = current_element_nodes(emitter)
        _check_two_nodes("dispBeamColumn", nodes)
        i_node, j_node = nodes
        transf_tag = resolve_tag(emitter, self.transf)
        integ_tag = resolve_tag(emitter, emitted_integration(self))

        args: list[int | float | str] = [
            i_node, j_node, transf_tag, integ_tag,
        ]
        mass = scaled_mass(self.mass, self.modifiers)
        if mass is not None:
            args.extend(["-mass", mass])
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
    modifiers
        Optional :class:`~.frame_modifiers.FrameModifiers`: the emitted
        ``A``, ``Iz``, ``Avy`` (and in 3-D ``J``, ``Iy``, ``Avz``) and
        ``-mass`` are the declared values times the matching modifier;
        ``E`` and ``G`` are never scaled. ``Iy`` / ``J`` / ``Avz`` are
        refused in 2-D.
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
    modifiers: FrameModifiers | None = None

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
        lacks: dict[str, str] = {}
        if not is_3d:
            lacks["Iy"] = "a 2-D element has no second moment about local y"
            lacks["J"] = "a 2-D element has no torsion"
            lacks["Avz"] = "a 2-D element has no shear area along local z"
        check_frame_modifiers(
            "ElasticTimoshenkoBeam", self.modifiers, lacks=lacks,
            has_mass=self.mass is not None,
        )

    def _is_3d(self) -> bool:
        return any(v is not None for v in (self.Iy, self.J, self.Avz))

    def dependencies(self) -> tuple[Primitive, ...]:
        return (self.transf,)

    def _emit(self, emitter: "Emitter", tag: int) -> None:
        nodes = current_element_nodes(emitter)
        _check_two_nodes("ElasticTimoshenkoBeam", nodes)
        i_node, j_node = nodes
        transf_tag = resolve_tag(emitter, self.transf)

        m = self.modifiers or FrameModifiers()
        A, Iz = scaled(self.A, m.A), scaled(self.Iz, m.Iz)
        Avy = scaled(self.Avy, m.Avy)
        args: list[int | float | str] = [i_node, j_node]
        if self._is_3d():
            # 3-D: E G A J Iy Iz Avy Avz transfTag.
            assert self.J is not None
            assert self.Iy is not None
            assert self.Avz is not None
            args.extend(
                [
                    self.E, self.G, A,
                    scaled(self.J, m.J), scaled(self.Iy, m.Iy), Iz,
                    Avy, scaled(self.Avz, m.Avz),
                    transf_tag,
                ]
            )
        else:
            # 2-D: E G A Iz Avy transfTag.
            args.extend([self.E, self.G, A, Iz, Avy, transf_tag])

        mass = scaled_mass(self.mass, self.modifiers)
        if mass is not None:
            args.extend(["-mass", mass])
        if self.c_mass:
            args.append("-cMass")

        emitter.element("ElasticTimoshenkoBeam", tag, *args)


def _check_optional_pm(
    type_name: str, pm: float | tuple[float, float] | None,
) -> None:
    if pm is None:
        return
    if isinstance(pm, tuple):
        if len(pm) != 2:
            raise ValueError(
                f"{type_name}: pm tuple must have length 2 (P0, alpha), "
                f"got {pm!r}."
            )
        p0, alpha = pm
        if isinstance(p0, bool) or not (p0 > 0) or p0 == float("inf"):
            raise ValueError(
                f"{type_name}: pm P0 must be finite and > 0, got {p0!r}."
            )
        if (
            isinstance(alpha, bool)
            or not (alpha == alpha)
            or alpha in (float("inf"), float("-inf"))
        ):
            raise ValueError(
                f"{type_name}: pm alpha must be finite, got {alpha!r}."
            )
    else:
        if isinstance(pm, bool) or not (pm > 0) or pm == float("inf"):
            raise ValueError(
                f"{type_name}: pm P0 must be finite and > 0, got {pm!r}."
            )


def _pm_args(pm: float | tuple[float, float] | None) -> list[float | str]:
    if pm is None:
        return []
    if isinstance(pm, tuple):
        return ["-pm", float(pm[0]), float(pm[1])]
    return ["-pm", float(pm)]


def _check_optional_pm_bal(
    type_name: str,
    pm_bal: tuple[float, float, float, float] | None,
    pm: float | tuple[float, float] | None = None,
) -> None:
    if pm_bal is None:
        return
    if pm is not None:
        raise ValueError(
            f"{type_name}: pm and pm_bal are mutually exclusive."
        )
    if not isinstance(pm_bal, tuple) or len(pm_bal) != 4:
        raise ValueError(
            f"{type_name}: pm_bal must be a 4-tuple "
            f"(P0c, P0t, Pbal, Mbal_ratio), got {pm_bal!r}."
        )
    for val in pm_bal:
        if (
            isinstance(val, bool)
            or not isinstance(val, (int, float))
            or not (val == val)
            or val in (float("inf"), float("-inf"))
        ):
            raise ValueError(
                f"{type_name}: pm_bal values must be finite numbers, "
                f"got {pm_bal!r}."
            )
    p0c, p0t, pbal, mbal_ratio = pm_bal
    if not (p0c > 0):
        raise ValueError(
            f"{type_name}: pm_bal P0c must be > 0, got {p0c!r}."
        )
    if not (p0t > 0):
        raise ValueError(
            f"{type_name}: pm_bal P0t must be > 0, got {p0t!r}."
        )
    if not (0.0 < pbal < p0c):
        raise ValueError(
            f"{type_name}: pm_bal Pbal must satisfy 0 < Pbal < P0c, "
            f"got Pbal={pbal!r}, P0c={p0c!r}."
        )
    if not (mbal_ratio >= 1.0):
        raise ValueError(
            f"{type_name}: pm_bal Mbal_ratio must be >= 1.0, "
            f"got {mbal_ratio!r}."
        )


def _pm_bal_args(
    pm_bal: tuple[float, float, float, float] | None,
) -> list[float | str]:
    if pm_bal is None:
        return []
    return [
        "-pmBal",
        float(pm_bal[0]),
        float(pm_bal[1]),
        float(pm_bal[2]),
        float(pm_bal[3]),
    ]


# ---------------------------------------------------------------------------
# LadrunoDispBeamColumn — disp-based beam-column with fork regularization +
# embedded cohesive hinges (Ladruno fork).
# ---------------------------------------------------------------------------

@dataclass(frozen=True, kw_only=True, slots=True)
class LadrunoDispBeamColumn(Element):
    r"""``element LadrunoDispBeamColumn`` — disp-based beam-column (Ladruno fork).

    OpenSees command (Ladruno fork; ``ELE_TAG`` **33013** (2D) / **33014**
    (3D), ndm-dispatched)::

        element LadrunoDispBeamColumn tag iNode jNode transfTag integrationTag \
            [-mass m] [-cMass] [-damp dampTag] [-lch ip|element|<value>] \
            [-nl] [-spanLoad | -noSpanLoad] [-axialRelease] \
            [-hinge matTag] [-hingeI matTagI] [-hingeJ matTagJ] \
            [-hingeIJ matTagI [matTagJ]] \
            [-hingeY matTag] [-hingeYI matTagYI] [-hingeYJ matTagYJ] \
            [-hingeYIJ matTagYI [matTagYJ]] \
            [-hingeBiaxial ndMatTag] [-hingeBiaxialI ndMatTagI] \
            [-hingeBiaxialJ ndMatTagJ] [-hingeBiaxialIJ ndMatTagI [ndMatTagJ]] \
            [-hingeLoc xiI [xiJ]] [-implex] [-pm P0 [alpha]] \
            [-pmBal P0c P0t Pbal Mbal_ratio]

    A displacement-based distributed-plasticity beam-column (the
    :class:`dispBeamColumn` shape: a :class:`BeamIntegration` rule places the
    integration points and composing sections) extended with the fork's
    Tier-1 per-IP crack-band regularization (``-lch``), the optional ``-nl``
    bowing strain, exact span-load curvature recovery + discrete-orthogonalized
    quartic bubble enhancement (``-spanLoad`` default; ``-noSpanLoad`` for
    classic cubic Hermite interpolation), axial-moment coupling release for RC
    floor beams (``-axialRelease``), and the Tier-2 embedded
    strong-discontinuity rotation hinges (single-hinge ``-hinge`` / ``-hingeY`` /
    ``-hingeBiaxial`` or dual-end ``-hingeI`` / ``-hingeJ`` / ``-hingeIJ`` /
    ``-hingeY*`` / ``-hingeBiaxial*`` with ``-hingeLoc``, ``-pm``, ``-pmBal``,
    and ``-implex``).

    .. note::
       Fork-only. Emission produces a deck line on any build; the element is
       unavailable on stock ``openseespy`` and bites only at ``ops.run()``.

    Parameters
    ----------
    pg
        Physical-group name the spec applies to.
    transf
        Geometric transform (Linear / PDelta / Corotational).
    integration
        The :class:`BeamIntegration` rule placing IPs and composing sections.
    mass
        Optional ``-mass <m>`` per-unit-length mass.
    c_mass
        If ``True``, append ``-cMass`` (consistent-mass form).
    lch
        Crack-band characteristic-length mode (``-lch``): ``"ip"`` (default,
        per-IP from the section), ``"element"`` (debug, the element length),
        or a positive numeric band width. Tier-1 regularization.
    nl
        If ``True``, append ``-nl`` (½θ² bowing strain — ½(θy²+θz²) in 3D).
        Mutually exclusive with any hinge.
    span_load
        If ``None`` (default), omit the flag and rely on the C++ default
        (span-load enhancement enabled). If ``True``, explicitly emit
        ``-spanLoad``. If ``False``, emit ``-noSpanLoad`` (classic cubic
        Hermite interpolation).
    hinge
        Single-hinge strong-axis (Mz) embedded cohesive rotation hinge — a
        :class:`UniaxialMaterial` (typically
        :class:`~apeGmsh.opensees.material.uniaxial.LadrunoCohesiveHinge`).
    hinge_i, hinge_j, hinge_ij
        Dual-end strong-axis (Mz) embedded rotation hinges (``-hingeI``,
        ``-hingeJ``, ``-hingeIJ``). ``hinge_ij`` accepts either a single
        :class:`UniaxialMaterial` (symmetric) or a ``(mat_i, mat_j)`` pair.
    hinge_y
        Single-hinge weak-axis (My) cohesive hinge (3-D only, ``-hingeY``);
        requires ``hinge`` to be set.
    hinge_yi, hinge_yj, hinge_yij
        Dual-end weak-axis (My) embedded rotation hinges (3-D only,
        ``-hingeYI``, ``-hingeYJ``, ``-hingeYIJ``); require the corresponding
        strong-axis dual-end hinge(s).
    hinge_biaxial
        Single-hinge coupled Mz–My cohesive interaction surface (3-D only,
        ``-hingeBiaxial``) — an :class:`NDMaterial` (typically
        :class:`~apeGmsh.opensees.material.nd.LadrunoCohesiveHingeBiaxial`).
        Mutually exclusive with uniaxial hinges.
    hinge_biaxial_i, hinge_biaxial_j, hinge_biaxial_ij
        Dual-end coupled Mz–My cohesive interaction hinges (3-D only,
        ``-hingeBiaxialI``, ``-hingeBiaxialJ``, ``-hingeBiaxialIJ``).
        Mutually exclusive with uniaxial hinges.
    hinge_loc
        Normalized hinge location(s) in ``[0, 1]`` (``-hingeLoc xiI
        [xiJ]``). Requires at least one hinge (single or dual-end).
    pm
        Optional symmetric parabolic axial-moment interaction ``P0`` or
        ``(P0, alpha)`` (``-pm P0 [alpha]``, ``P0 > 0``). Mutually exclusive
        with ``pm_bal``.
    pm_bal
        Optional asymmetric RC balanced-point axial-moment interaction
        4-tuple ``(P0c, P0t, Pbal, Mbal_ratio)`` (``-pmBal P0c P0t Pbal
        Mbal_ratio``, with ``P0c > 0``, ``P0t > 0``, ``0 < Pbal < P0c``, and
        ``Mbal_ratio >= 1.0``). Mutually exclusive with ``pm``.
    axial_release
        If ``True``, append ``-axialRelease`` to release basic axial force
        (:math:`N = 0`, solving for the neutral-axis strain :math:`\varepsilon_0(x_i)`
        at each IP and statically condensing the section axial-flexural coupling,
        eliminating spurious frame axial-elongation restraint in RC beams).
    implex
        If ``True``, append ``-implex`` for element-level IMPLEX hinge
        extrapolation.
    """

    pg: str
    transf: _AnyTransf
    integration: BeamIntegration
    mass: float | None = None
    c_mass: bool = False
    lch: str | float = "ip"
    nl: bool = False
    span_load: bool | None = None
    hinge: UniaxialMaterial | None = None
    hinge_i: UniaxialMaterial | None = None
    hinge_j: UniaxialMaterial | None = None
    hinge_ij: UniaxialMaterial | tuple[UniaxialMaterial, UniaxialMaterial] | None = None
    hinge_y: UniaxialMaterial | None = None
    hinge_yi: UniaxialMaterial | None = None
    hinge_yj: UniaxialMaterial | None = None
    hinge_yij: UniaxialMaterial | tuple[UniaxialMaterial, UniaxialMaterial] | None = None
    hinge_biaxial: NDMaterial | None = None
    hinge_biaxial_i: NDMaterial | None = None
    hinge_biaxial_j: NDMaterial | None = None
    hinge_biaxial_ij: NDMaterial | tuple[NDMaterial, NDMaterial] | None = None
    hinge_loc: float | tuple[float, float] | None = None
    pm: float | tuple[float, float] | None = None
    pm_bal: tuple[float, float, float, float] | None = None
    axial_release: bool = False
    implex: bool = False
    damp: Damping | None = None

    def __post_init__(self) -> None:
        if not self.pg:
            raise ValueError("LadrunoDispBeamColumn: pg must be a non-empty name.")
        _check_optional_mass("LadrunoDispBeamColumn", self.mass)

        if self.span_load is not None and not isinstance(self.span_load, bool):
            raise ValueError(
                f"LadrunoDispBeamColumn: span_load must be a bool or None, got {self.span_load!r}."
            )
        if not isinstance(self.axial_release, bool):
            raise ValueError(
                f"LadrunoDispBeamColumn: axial_release must be a bool, got {self.axial_release!r}."
            )

        if isinstance(self.lch, str):
            if self.lch not in ("ip", "element"):
                raise ValueError(
                    "LadrunoDispBeamColumn: lch must be 'ip', 'element', or a "
                    f"positive number, got {self.lch!r}."
                )
        elif isinstance(self.lch, bool) or not (self.lch > 0) or self.lch == float("inf"):
            # ``not (lch > 0)`` rejects NaN and non-positive; the explicit
            # inf check rejects +inf (a non-finite band width is nonsense).
            raise ValueError(
                "LadrunoDispBeamColumn: numeric lch must be finite and "
                f"> 0, got {self.lch!r}."
            )

        for fname, val in (
            ("hinge_ij", self.hinge_ij),
            ("hinge_yij", self.hinge_yij),
            ("hinge_biaxial_ij", self.hinge_biaxial_ij),
        ):
            if isinstance(val, tuple) and len(val) != 2:
                raise ValueError(
                    f"LadrunoDispBeamColumn: {fname} tuple must have length 2, "
                    f"got {len(val)}."
                )

        has_single = any(
            x is not None for x in (self.hinge, self.hinge_y, self.hinge_biaxial)
        )
        has_dual = any(
            x is not None
            for x in (
                self.hinge_i, self.hinge_j, self.hinge_ij,
                self.hinge_yi, self.hinge_yj, self.hinge_yij,
                self.hinge_biaxial_i, self.hinge_biaxial_j, self.hinge_biaxial_ij,
            )
        )
        if has_single and has_dual:
            raise ValueError(
                "LadrunoDispBeamColumn: single-hinge (hinge/hinge_y/hinge_biaxial) "
                "and dual-hinge (hinge_i/j/ij, hinge_yi/yj/yij, "
                "hinge_biaxial_i/j/ij) modes are mutually exclusive."
            )
        if self.hinge_ij is not None and (
            self.hinge_i is not None or self.hinge_j is not None
        ):
            raise ValueError(
                "LadrunoDispBeamColumn: hinge_ij is mutually exclusive with "
                "hinge_i / hinge_j."
            )
        if self.hinge_yij is not None and (
            self.hinge_yi is not None or self.hinge_yj is not None
        ):
            raise ValueError(
                "LadrunoDispBeamColumn: hinge_yij is mutually exclusive with "
                "hinge_yi / hinge_yj."
            )
        if self.hinge_biaxial_ij is not None and (
            self.hinge_biaxial_i is not None or self.hinge_biaxial_j is not None
        ):
            raise ValueError(
                "LadrunoDispBeamColumn: hinge_biaxial_ij is mutually exclusive "
                "with hinge_biaxial_i / hinge_biaxial_j."
            )

        has_uniaxial = any(
            x is not None
            for x in (
                self.hinge, self.hinge_i, self.hinge_j, self.hinge_ij,
                self.hinge_y, self.hinge_yi, self.hinge_yj, self.hinge_yij,
            )
        )
        has_biaxial = any(
            x is not None
            for x in (
                self.hinge_biaxial, self.hinge_biaxial_i,
                self.hinge_biaxial_j, self.hinge_biaxial_ij,
            )
        )
        if self.nl and (has_uniaxial or has_biaxial):
            raise ValueError(
                "LadrunoDispBeamColumn: -nl is mutually exclusive with any "
                "hinge (-hinge/-hingeY/-hingeBiaxial)."
            )
        if has_biaxial and has_uniaxial:
            raise ValueError(
                "LadrunoDispBeamColumn: hinge_biaxial (coupled) is mutually "
                "exclusive with hinge / hinge_y (block-diagonal)."
            )
        if self.hinge_y is not None and self.hinge is None:
            raise ValueError(
                "LadrunoDispBeamColumn: hinge_y requires hinge to be set "
                "(the strong axis must carry a hinge before the weak axis)."
            )
        has_zi = self.hinge_i is not None or self.hinge_ij is not None
        has_zj = self.hinge_j is not None or self.hinge_ij is not None
        if self.hinge_yi is not None and not has_zi:
            raise ValueError(
                "LadrunoDispBeamColumn: hinge_yi requires hinge_i or hinge_ij "
                "to be set."
            )
        if self.hinge_yj is not None and not has_zj:
            raise ValueError(
                "LadrunoDispBeamColumn: hinge_yj requires hinge_j or hinge_ij "
                "to be set."
            )
        if self.hinge_yij is not None and not (has_zi and has_zj):
            raise ValueError(
                "LadrunoDispBeamColumn: hinge_yij requires strong-axis hinges "
                "at both ends (hinge_ij or hinge_i + hinge_j)."
            )

        if self.hinge_loc is not None:
            if not (has_single or has_dual):
                raise ValueError(
                    "LadrunoDispBeamColumn: hinge_loc requires a hinge "
                    "(hinge/hinge_i/hinge_j/hinge_ij or hinge_biaxial*)."
                )
            locs = (
                self.hinge_loc
                if isinstance(self.hinge_loc, tuple)
                else (self.hinge_loc,)
            )
            if len(locs) not in (1, 2):
                raise ValueError(
                    "LadrunoDispBeamColumn: hinge_loc tuple must have length 2, "
                    f"got {self.hinge_loc!r}."
                )
            for xi in locs:
                if (
                    isinstance(xi, bool)
                    or not (0.0 <= xi <= 1.0)
                ):
                    raise ValueError(
                        "LadrunoDispBeamColumn: hinge_loc values must be in "
                        f"[0, 1], got {self.hinge_loc!r}."
                    )

        _check_optional_pm("LadrunoDispBeamColumn", self.pm)
        _check_optional_pm_bal("LadrunoDispBeamColumn", self.pm_bal, self.pm)

    def dependencies(self) -> tuple[Primitive, ...]:
        deps: list[Primitive] = [self.integration, self.transf]
        for entry in (
            self.hinge, self.hinge_i, self.hinge_j, self.hinge_ij,
            self.hinge_y, self.hinge_yi, self.hinge_yj, self.hinge_yij,
            self.hinge_biaxial, self.hinge_biaxial_i,
            self.hinge_biaxial_j, self.hinge_biaxial_ij,
        ):
            if entry is None:
                continue
            if isinstance(entry, tuple):
                for m in entry:
                    if m not in deps:
                        deps.append(m)
            elif entry not in deps:
                deps.append(entry)
        if self.damp is not None:
            deps.append(self.damp)
        return tuple(deps)

    def _emit(self, emitter: "Emitter", tag: int) -> None:
        nodes = current_element_nodes(emitter)
        _check_two_nodes("LadrunoDispBeamColumn", nodes)
        i_node, j_node = nodes
        transf_tag = resolve_tag(emitter, self.transf)
        integ_tag = resolve_tag(emitter, self.integration)

        args: list[int | float | str] = [i_node, j_node, transf_tag, integ_tag]
        if self.mass is not None:
            args.extend(["-mass", self.mass])
        if self.c_mass:
            args.append("-cMass")
        if self.lch != "ip":
            args.extend(["-lch", str(self.lch)])
        if self.nl:
            args.append("-nl")
        if self.span_load is True:
            args.append("-spanLoad")
        elif self.span_load is False:
            args.append("-noSpanLoad")
        if self.hinge is not None:
            args.extend(["-hinge", resolve_tag(emitter, self.hinge)])
        if self.hinge_i is not None:
            args.extend(["-hingeI", resolve_tag(emitter, self.hinge_i)])
        if self.hinge_j is not None:
            args.extend(["-hingeJ", resolve_tag(emitter, self.hinge_j)])
        if self.hinge_ij is not None:
            if isinstance(self.hinge_ij, tuple):
                args.extend([
                    "-hingeIJ",
                    resolve_tag(emitter, self.hinge_ij[0]),
                    resolve_tag(emitter, self.hinge_ij[1]),
                ])
            else:
                args.extend(["-hingeIJ", resolve_tag(emitter, self.hinge_ij)])
        if self.hinge_y is not None:
            args.extend(["-hingeY", resolve_tag(emitter, self.hinge_y)])
        if self.hinge_yi is not None:
            args.extend(["-hingeYI", resolve_tag(emitter, self.hinge_yi)])
        if self.hinge_yj is not None:
            args.extend(["-hingeYJ", resolve_tag(emitter, self.hinge_yj)])
        if self.hinge_yij is not None:
            if isinstance(self.hinge_yij, tuple):
                args.extend([
                    "-hingeYIJ",
                    resolve_tag(emitter, self.hinge_yij[0]),
                    resolve_tag(emitter, self.hinge_yij[1]),
                ])
            else:
                args.extend(["-hingeYIJ", resolve_tag(emitter, self.hinge_yij)])
        if self.hinge_biaxial is not None:
            args.extend(
                ["-hingeBiaxial", resolve_tag(emitter, self.hinge_biaxial)]
            )
        if self.hinge_biaxial_i is not None:
            args.extend(
                ["-hingeBiaxialI", resolve_tag(emitter, self.hinge_biaxial_i)]
            )
        if self.hinge_biaxial_j is not None:
            args.extend(
                ["-hingeBiaxialJ", resolve_tag(emitter, self.hinge_biaxial_j)]
            )
        if self.hinge_biaxial_ij is not None:
            if isinstance(self.hinge_biaxial_ij, tuple):
                args.extend([
                    "-hingeBiaxialIJ",
                    resolve_tag(emitter, self.hinge_biaxial_ij[0]),
                    resolve_tag(emitter, self.hinge_biaxial_ij[1]),
                ])
            else:
                args.extend([
                    "-hingeBiaxialIJ",
                    resolve_tag(emitter, self.hinge_biaxial_ij),
                ])
        if self.hinge_loc is not None:
            if isinstance(self.hinge_loc, tuple):
                args.extend([
                    "-hingeLoc",
                    float(self.hinge_loc[0]),
                    float(self.hinge_loc[1]),
                ])
            else:
                args.extend(["-hingeLoc", float(self.hinge_loc)])
        args.extend(_pm_args(self.pm))
        args.extend(_pm_bal_args(self.pm_bal))
        if self.axial_release:
            args.append("-axialRelease")
        if self.implex:
            args.append("-implex")
        args.extend(damp_args(emitter, self.damp))

        emitter.element("LadrunoDispBeamColumn", tag, *args)


# ---------------------------------------------------------------------------
# LadrunoForceBeamColumn — force-based beam-column with fork regularization +
# embedded cohesive hinges + linear Timoshenko shear (Ladruno fork).
# ---------------------------------------------------------------------------

@dataclass(frozen=True, kw_only=True, slots=True)
class LadrunoForceBeamColumn(Element):
    r"""``element LadrunoForceBeamColumn`` — force-based beam-column (Ladruno fork).

    OpenSees command (Ladruno fork; ``ELE_TAG`` **33024** (2D) / **33025**
    (3D), ndm-dispatched)::

        element LadrunoForceBeamColumn tag iNode jNode transfTag integrationTag \
            [-mass m] [-cMass] [-damp dampTag] [-iter maxIter tol] \
            [-subdivide maxSubLevels [subdivideFactor]] [-lch ip|element|<value>] \
            [-spanLoad | -noSpanLoad] [-axialRelease] \
            [-hinge matTag] [-hingeI matTagI] [-hingeJ matTagJ] \
            [-hingeIJ matTagI [matTagJ]] \
            [-hingeY matTag] [-hingeYI matTagYI] [-hingeYJ matTagYJ] \
            [-hingeYIJ matTagYI [matTagYJ]] \
            [-hingeBiaxial ndMatTag] [-hingeBiaxialI ndMatTagI] \
            [-hingeBiaxialJ ndMatTagJ] [-hingeBiaxialIJ ndMatTagI [ndMatTagJ]] \
            [-hingeLoc xiI [xiJ]] [-pm P0 [alpha]] \
            [-pmBal P0c P0t Pbal Mbal_ratio] [-shear GAvy [GAvz]] \
            [-implex]

    A force-based (flexibility) distributed-plasticity beam-column (exact linear
    moment interpolation ``s(xi) = b(xi) q + s_p(xi)`` with full-span and partial
    trapezoidal ``beamUniform`` + ``beamPoint`` member loads) extended with the
    fork's Tier-1 per-IP crack-band regularization (``-lch``), axial-moment
    coupling release for RC floor beams (``-axialRelease``), Tier-2 embedded
    strong-discontinuity rotation hinges (``-hinge*``, ``-hingeY*``,
    ``-hingeBiaxial*``, ``-hingeLoc``, ``-pm``, ``-pmBal``, ``-implex``), and
    additive element-level linear Timoshenko shear flexibility (``-shear``).

    .. note::
       Fork-only. Emission produces a deck line on any build; the element is
       unavailable on stock ``openseespy`` and bites only at ``ops.run()``.

    Parameters
    ----------
    pg
        Physical-group name the spec applies to.
    transf
        Geometric transform (Linear / PDelta / Corotational).
    integration
        The :class:`BeamIntegration` rule placing IPs and composing sections.
    mass
        Optional ``-mass <m>`` per-unit-length mass.
    c_mass
        If ``True``, append ``-cMass`` (consistent-mass form).
    max_iter, tol
        Optional ``-iter maxIter tol`` element state-determination loop controls
        (must be supplied together or both omitted).
    subdivide
        Optional ``-subdivide maxSubLevels [subdivideFactor]`` step subdivision
        controls: an integer ``max_sub_levels >= 1`` or a 2-tuple
        ``(max_sub_levels, subdivide_factor)`` with ``subdivide_factor > 1.0``.
    lch
        Crack-band characteristic-length mode (``-lch``): ``"ip"`` (default,
        per-IP ``w_i * L``), ``"element"`` (debug, ``L``), or a positive numeric
        band width. Tier-1 regularization.
    span_load
        If ``None`` (default), omit the flag and rely on the C++ default
        (exact internal member-load force interpolation enabled). If ``True``,
        explicitly emit ``-spanLoad``. If ``False``, emit ``-noSpanLoad``.
    hinge, hinge_i, hinge_j, hinge_ij
        Strong-axis (Mz) embedded cohesive rotation hinges.
    hinge_y, hinge_yi, hinge_yj, hinge_yij
        Weak-axis (My) embedded cohesive rotation hinges (3-D only).
    hinge_biaxial, hinge_biaxial_i, hinge_biaxial_j, hinge_biaxial_ij
        Coupled 2-DOF (Mz, My) biaxial cohesive rotation hinges (3-D only).
    hinge_loc
        Normalized hinge location(s) in ``[0, 1]`` (``-hingeLoc xiI [xiJ]``).
        Requires at least one hinge (single or dual-end).
    pm
        Optional symmetric parabolic axial-moment interaction ``P0`` or
        ``(P0, alpha)`` (``-pm P0 [alpha]``, ``P0 > 0``). Mutually exclusive
        with ``pm_bal``.
    pm_bal
        Optional asymmetric RC balanced-point axial-moment interaction
        4-tuple ``(P0c, P0t, Pbal, Mbal_ratio)`` (``-pmBal P0c P0t Pbal
        Mbal_ratio``, with ``P0c > 0``, ``P0t > 0``, ``0 < Pbal < P0c``, and
        ``Mbal_ratio >= 1.0``). Mutually exclusive with ``pm``.
    axial_release
        If ``True``, append ``-axialRelease`` to release basic axial force
        (:math:`N = 0` at every section IP while preserving full fiber flexural
        response, eliminating spurious frame axial-elongation restraint in RC beams).
    shear
        Optional additive linear Timoshenko shear rigidity (``-shear GAvy`` in
        2-D or ``-shear GAvy [GAvz]`` in 3-D; finite and ``> 0``).
    implex
        If ``True``, append ``-implex`` for element-level IMPLEX hinge
        extrapolation.
    damp
        Optional element-level damping object (``-damp``).
    """

    pg: str
    transf: _AnyTransf
    integration: BeamIntegration
    mass: float | None = None
    c_mass: bool = False
    max_iter: int | None = None
    tol: float | None = None
    subdivide: int | tuple[int, float] | None = None
    lch: str | float = "ip"
    span_load: bool | None = None
    hinge: UniaxialMaterial | None = None
    hinge_i: UniaxialMaterial | None = None
    hinge_j: UniaxialMaterial | None = None
    hinge_ij: UniaxialMaterial | tuple[UniaxialMaterial, UniaxialMaterial] | None = None
    hinge_y: UniaxialMaterial | None = None
    hinge_yi: UniaxialMaterial | None = None
    hinge_yj: UniaxialMaterial | None = None
    hinge_yij: UniaxialMaterial | tuple[UniaxialMaterial, UniaxialMaterial] | None = None
    hinge_biaxial: NDMaterial | None = None
    hinge_biaxial_i: NDMaterial | None = None
    hinge_biaxial_j: NDMaterial | None = None
    hinge_biaxial_ij: NDMaterial | tuple[NDMaterial, NDMaterial] | None = None
    hinge_loc: float | tuple[float, float] | None = None
    pm: float | tuple[float, float] | None = None
    pm_bal: tuple[float, float, float, float] | None = None
    axial_release: bool = False
    shear: float | tuple[float, float] | None = None
    implex: bool = False
    damp: Damping | None = None

    def __post_init__(self) -> None:
        if not self.pg:
            raise ValueError("LadrunoForceBeamColumn: pg must be a non-empty name.")
        _check_optional_mass("LadrunoForceBeamColumn", self.mass)

        if self.span_load is not None and not isinstance(self.span_load, bool):
            raise ValueError(
                f"LadrunoForceBeamColumn: span_load must be a bool or None, got {self.span_load!r}."
            )
        if not isinstance(self.axial_release, bool):
            raise ValueError(
                f"LadrunoForceBeamColumn: axial_release must be a bool, got {self.axial_release!r}."
            )

        if (self.max_iter is None) != (self.tol is None):
            raise ValueError(
                "LadrunoForceBeamColumn: max_iter and tol must be supplied "
                "together (or both omitted)."
            )
        if self.max_iter is not None and (
            isinstance(self.max_iter, bool) or self.max_iter <= 0
        ):
            raise ValueError(
                f"LadrunoForceBeamColumn: max_iter must be > 0, got {self.max_iter!r}."
            )
        if self.tol is not None and (
            isinstance(self.tol, bool)
            or not (self.tol > 0)
            or self.tol == float("inf")
        ):
            raise ValueError(
                f"LadrunoForceBeamColumn: tol must be finite and > 0, got {self.tol!r}."
            )
        if self.subdivide is not None:
            if isinstance(self.subdivide, tuple):
                if len(self.subdivide) != 2:
                    raise ValueError(
                        "LadrunoForceBeamColumn: subdivide tuple must have length 2 "
                        f"(max_sub_levels, subdivide_factor), got {self.subdivide!r}."
                    )
                sub_levels, sub_factor = self.subdivide
                if (
                    isinstance(sub_levels, bool)
                    or not isinstance(sub_levels, int)
                    or sub_levels < 1
                ):
                    raise ValueError(
                        "LadrunoForceBeamColumn: subdivide max_sub_levels must be "
                        f"an int >= 1, got {self.subdivide!r}."
                    )
                if (
                    isinstance(sub_factor, bool)
                    or not isinstance(sub_factor, (int, float))
                    or not (sub_factor > 1.0)
                    or sub_factor == float("inf")
                ):
                    raise ValueError(
                        "LadrunoForceBeamColumn: subdivide factor must be finite "
                        f"and > 1.0, got {self.subdivide!r}."
                    )
            elif (
                isinstance(self.subdivide, bool)
                or not isinstance(self.subdivide, int)
                or self.subdivide < 1
            ):
                raise ValueError(
                    "LadrunoForceBeamColumn: subdivide must be an int >= 1 "
                    "or (max_sub_levels, subdivide_factor), "
                    f"got {self.subdivide!r}."
                )

        if isinstance(self.lch, str):
            if self.lch not in ("ip", "element"):
                raise ValueError(
                    "LadrunoForceBeamColumn: lch must be 'ip', 'element', or a "
                    f"positive number, got {self.lch!r}."
                )
        elif isinstance(self.lch, bool) or not (self.lch > 0) or self.lch == float("inf"):
            raise ValueError(
                "LadrunoForceBeamColumn: numeric lch must be finite and "
                f"> 0, got {self.lch!r}."
            )

        for fname, val in (
            ("hinge_ij", self.hinge_ij),
            ("hinge_yij", self.hinge_yij),
            ("hinge_biaxial_ij", self.hinge_biaxial_ij),
        ):
            if isinstance(val, tuple) and len(val) != 2:
                raise ValueError(
                    f"LadrunoForceBeamColumn: {fname} tuple must have length 2, "
                    f"got {len(val)}."
                )

        has_single = any(
            x is not None for x in (self.hinge, self.hinge_y, self.hinge_biaxial)
        )
        has_dual = any(
            x is not None
            for x in (
                self.hinge_i, self.hinge_j, self.hinge_ij,
                self.hinge_yi, self.hinge_yj, self.hinge_yij,
                self.hinge_biaxial_i, self.hinge_biaxial_j, self.hinge_biaxial_ij,
            )
        )
        if has_single and has_dual:
            raise ValueError(
                "LadrunoForceBeamColumn: single-hinge (hinge/hinge_y/hinge_biaxial) "
                "and dual-hinge (hinge_i/j/ij, hinge_yi/yj/yij, "
                "hinge_biaxial_i/j/ij) modes are mutually exclusive."
            )
        if self.hinge_ij is not None and (
            self.hinge_i is not None or self.hinge_j is not None
        ):
            raise ValueError(
                "LadrunoForceBeamColumn: hinge_ij is mutually exclusive with "
                "hinge_i / hinge_j."
            )
        if self.hinge_yij is not None and (
            self.hinge_yi is not None or self.hinge_yj is not None
        ):
            raise ValueError(
                "LadrunoForceBeamColumn: hinge_yij is mutually exclusive with "
                "hinge_yi / hinge_yj."
            )
        if self.hinge_biaxial_ij is not None and (
            self.hinge_biaxial_i is not None or self.hinge_biaxial_j is not None
        ):
            raise ValueError(
                "LadrunoForceBeamColumn: hinge_biaxial_ij is mutually exclusive "
                "with hinge_biaxial_i / hinge_biaxial_j."
            )

        has_uniaxial = any(
            x is not None
            for x in (
                self.hinge, self.hinge_i, self.hinge_j, self.hinge_ij,
                self.hinge_y, self.hinge_yi, self.hinge_yj, self.hinge_yij,
            )
        )
        has_biaxial = any(
            x is not None
            for x in (
                self.hinge_biaxial, self.hinge_biaxial_i,
                self.hinge_biaxial_j, self.hinge_biaxial_ij,
            )
        )
        if has_biaxial and has_uniaxial:
            raise ValueError(
                "LadrunoForceBeamColumn: hinge_biaxial (coupled) is mutually "
                "exclusive with hinge / hinge_y (block-diagonal)."
            )
        if self.hinge_y is not None and self.hinge is None:
            raise ValueError(
                "LadrunoForceBeamColumn: hinge_y requires hinge to be set "
                "(the strong axis must carry a hinge before the weak axis)."
            )
        has_zi = self.hinge_i is not None or self.hinge_ij is not None
        has_zj = self.hinge_j is not None or self.hinge_ij is not None
        if self.hinge_yi is not None and not has_zi:
            raise ValueError(
                "LadrunoForceBeamColumn: hinge_yi requires hinge_i or hinge_ij "
                "to be set."
            )
        if self.hinge_yj is not None and not has_zj:
            raise ValueError(
                "LadrunoForceBeamColumn: hinge_yj requires hinge_j or hinge_ij "
                "to be set."
            )
        if self.hinge_yij is not None and not (has_zi and has_zj):
            raise ValueError(
                "LadrunoForceBeamColumn: hinge_yij requires strong-axis hinges "
                "at both ends (hinge_ij or hinge_i + hinge_j)."
            )

        if self.hinge_loc is not None:
            if not (has_single or has_dual):
                raise ValueError(
                    "LadrunoForceBeamColumn: hinge_loc requires a hinge "
                    "(hinge/hinge_i/hinge_j/hinge_ij or hinge_biaxial*)."
                )
            locs = (
                self.hinge_loc
                if isinstance(self.hinge_loc, tuple)
                else (self.hinge_loc,)
            )
            if len(locs) not in (1, 2):
                raise ValueError(
                    "LadrunoForceBeamColumn: hinge_loc tuple must have length 2, "
                    f"got {self.hinge_loc!r}."
                )
            for xi in locs:
                if isinstance(xi, bool) or not (0.0 <= xi <= 1.0):
                    raise ValueError(
                        "LadrunoForceBeamColumn: hinge_loc values must be in "
                        f"[0, 1], got {self.hinge_loc!r}."
                    )

        _check_optional_pm("LadrunoForceBeamColumn", self.pm)
        _check_optional_pm_bal("LadrunoForceBeamColumn", self.pm_bal, self.pm)

        if self.shear is not None:
            if isinstance(self.shear, tuple):
                if len(self.shear) != 2:
                    raise ValueError(
                        "LadrunoForceBeamColumn: shear tuple must have length 2 "
                        f"(GAvy, GAvz), got {self.shear!r}."
                    )
                svals: tuple[float, ...] = self.shear
            else:
                svals = (self.shear,)
            for sv in svals:
                if isinstance(sv, bool) or not (sv > 0) or sv == float("inf"):
                    raise ValueError(
                        "LadrunoForceBeamColumn: shear values must be finite "
                        f"and > 0, got {self.shear!r}."
                    )

    def dependencies(self) -> tuple[Primitive, ...]:
        deps: list[Primitive] = [self.integration, self.transf]
        for entry in (
            self.hinge, self.hinge_i, self.hinge_j, self.hinge_ij,
            self.hinge_y, self.hinge_yi, self.hinge_yj, self.hinge_yij,
            self.hinge_biaxial, self.hinge_biaxial_i,
            self.hinge_biaxial_j, self.hinge_biaxial_ij,
        ):
            if entry is None:
                continue
            if isinstance(entry, tuple):
                for m in entry:
                    if m not in deps:
                        deps.append(m)
            elif entry not in deps:
                deps.append(entry)
        if self.damp is not None:
            deps.append(self.damp)
        return tuple(deps)

    def _emit(self, emitter: "Emitter", tag: int) -> None:
        nodes = current_element_nodes(emitter)
        _check_two_nodes("LadrunoForceBeamColumn", nodes)
        i_node, j_node = nodes
        transf_tag = resolve_tag(emitter, self.transf)
        integ_tag = resolve_tag(emitter, self.integration)

        args: list[int | float | str] = [i_node, j_node, transf_tag, integ_tag]
        if self.mass is not None:
            args.extend(["-mass", self.mass])
        if self.c_mass:
            args.append("-cMass")
        if self.max_iter is not None and self.tol is not None:
            args.extend(["-iter", int(self.max_iter), float(self.tol)])
        if self.subdivide is not None:
            if isinstance(self.subdivide, tuple):
                args.extend([
                    "-subdivide",
                    int(self.subdivide[0]),
                    float(self.subdivide[1]),
                ])
            else:
                args.extend(["-subdivide", int(self.subdivide)])
        if self.lch != "ip":
            args.extend(["-lch", str(self.lch)])
        if self.span_load is True:
            args.append("-spanLoad")
        elif self.span_load is False:
            args.append("-noSpanLoad")
        if self.hinge is not None:
            args.extend(["-hinge", resolve_tag(emitter, self.hinge)])
        if self.hinge_i is not None:
            args.extend(["-hingeI", resolve_tag(emitter, self.hinge_i)])
        if self.hinge_j is not None:
            args.extend(["-hingeJ", resolve_tag(emitter, self.hinge_j)])
        if self.hinge_ij is not None:
            if isinstance(self.hinge_ij, tuple):
                args.extend([
                    "-hingeIJ",
                    resolve_tag(emitter, self.hinge_ij[0]),
                    resolve_tag(emitter, self.hinge_ij[1]),
                ])
            else:
                args.extend(["-hingeIJ", resolve_tag(emitter, self.hinge_ij)])
        if self.hinge_y is not None:
            args.extend(["-hingeY", resolve_tag(emitter, self.hinge_y)])
        if self.hinge_yi is not None:
            args.extend(["-hingeYI", resolve_tag(emitter, self.hinge_yi)])
        if self.hinge_yj is not None:
            args.extend(["-hingeYJ", resolve_tag(emitter, self.hinge_yj)])
        if self.hinge_yij is not None:
            if isinstance(self.hinge_yij, tuple):
                args.extend([
                    "-hingeYIJ",
                    resolve_tag(emitter, self.hinge_yij[0]),
                    resolve_tag(emitter, self.hinge_yij[1]),
                ])
            else:
                args.extend(["-hingeYIJ", resolve_tag(emitter, self.hinge_yij)])
        if self.hinge_biaxial is not None:
            args.extend(
                ["-hingeBiaxial", resolve_tag(emitter, self.hinge_biaxial)]
            )
        if self.hinge_biaxial_i is not None:
            args.extend(
                ["-hingeBiaxialI", resolve_tag(emitter, self.hinge_biaxial_i)]
            )
        if self.hinge_biaxial_j is not None:
            args.extend(
                ["-hingeBiaxialJ", resolve_tag(emitter, self.hinge_biaxial_j)]
            )
        if self.hinge_biaxial_ij is not None:
            if isinstance(self.hinge_biaxial_ij, tuple):
                args.extend([
                    "-hingeBiaxialIJ",
                    resolve_tag(emitter, self.hinge_biaxial_ij[0]),
                    resolve_tag(emitter, self.hinge_biaxial_ij[1]),
                ])
            else:
                args.extend([
                    "-hingeBiaxialIJ",
                    resolve_tag(emitter, self.hinge_biaxial_ij),
                ])
        if self.hinge_loc is not None:
            if isinstance(self.hinge_loc, tuple):
                args.extend([
                    "-hingeLoc",
                    float(self.hinge_loc[0]),
                    float(self.hinge_loc[1]),
                ])
            else:
                args.extend(["-hingeLoc", float(self.hinge_loc)])
        args.extend(_pm_args(self.pm))
        args.extend(_pm_bal_args(self.pm_bal))
        if self.axial_release:
            args.append("-axialRelease")
        if self.shear is not None:
            if isinstance(self.shear, tuple):
                args.extend(["-shear", float(self.shear[0]), float(self.shear[1])])
            else:
                args.extend(["-shear", float(self.shear)])
        if self.implex:
            args.append("-implex")
        args.extend(damp_args(emitter, self.damp))

        emitter.element("LadrunoForceBeamColumn", tag, *args)


# ---------------------------------------------------------------------------
# LadrunoIMKBeam — concentrated-plasticity beam, moment-rotation
# IMK / cohesive hinges at the ends (Ladruno fork).
# ---------------------------------------------------------------------------

@dataclass(frozen=True, kw_only=True, slots=True)
class LadrunoIMKBeam(Element):
    r"""``element LadrunoIMKBeam`` — concentrated-plasticity IMK beam (Ladruno fork).

    OpenSees command (Ladruno fork; ``ELE_TAG`` **33003** (3D) / **33004**
    (2D), ndm-dispatched)::

        # 2D:
        element LadrunoIMKBeam tag iNode jNode A E Iz transfTag \
            [-hinge both|i|j|none] [-matZ tag] [-matZi tag] [-matZj tag] \
            [-shear GAvy | -shear G Avy] [-shearHinge matTag] \
            [-endZones dI dJ] [-mass rho] [-cMass] [-pm P0 [alpha]] \
            [-pmBal P0c P0t Pbal Mbal_ratio] [-implex]
        # 3D:
        element LadrunoIMKBeam tag iNode jNode A E G Jx Iy Iz transfTag \
            [-hinge both|i|j|none] [-matZ tag] [-matY tag] \
            [-matZi/-matZj/-matYi/-matYj tag] \
            [-hingeBiaxial ndMatTag] [-hingeBiaxialI ndMatTagI] \
            [-hingeBiaxialJ ndMatTagJ] [-hingeBiaxialIJ ndMatTagI [ndMatTagJ]] \
            [-shear GAvy GAvz | -shear G Avy Avz] \
            [-shearHinge matTagVy [matTagVz]] \
            [-shearHingeY matTagVy] [-shearHingeZ matTagVz] \
            [-endZones dI dJ] [-mass rho] [-cMass] [-pm P0 [alpha]] \
            [-pmBal P0c P0t Pbal Mbal_ratio] [-implex]

    A 2-node elastic Euler-Bernoulli / Timoshenko beam carrying concentrated
    moment-rotation hinges at its ends (uniaxial Mz/My or 3-D coupled Mz–My
    biaxial ``NDMaterial``), optional series inelastic shear hinges
    (``-shearHinge`` in 2-D or 3-D; ``-shearHingeY`` / ``-shearHingeZ`` in 3-D),
    optional rigid end zones (``-endZones``), optional symmetric parabolic or
    asymmetric RC balanced-point axial-moment interaction (``-pm`` / ``-pmBal``),
    and element-level IMPLEX extrapolation (``-implex``).

    The 3-D variant is selected by supplying any of the 3-D-only properties
    ``G`` / ``Jx`` / ``Iy`` (all three then required), mirroring
    :class:`elasticBeamColumn`.

    .. note::
        Fork-only. Emission produces a deck line on any build; the element is
        unavailable on stock ``openseespy`` and bites only at ``ops.run()``.

    Parameters
    ----------
    pg, transf
        Physical-group name and geometric transform.
    A, E, Iz
        Cross-section area, Young's modulus, strong-axis inertia (both forms).
    G, Jx, Iy
        Shear modulus, torsion constant, weak-axis inertia (3-D only; supply
        all three together).
    ends
        Which ends receive the symmetric ``mat_z`` / ``mat_y`` laws
        (``-hinge``): ``"both"`` (default), ``"i"``, ``"j"``, or ``"none"``.
    mat_z, mat_y
        Symmetric strong-axis (Mz) and weak-axis (My, 3-D only) hinge laws
        applied to the ``ends``-selected ends.
    mat_zi, mat_zj, mat_yi, mat_yj
        Per-end overrides (``-matZi`` / ``-matZj`` / ``-matYi`` / ``-matYj``)
        — take precedence over the symmetric laws at that end. The ``*y*``
        overrides are 3-D only.
    hinge_biaxial, hinge_biaxial_i, hinge_biaxial_j, hinge_biaxial_ij
        3-D coupled 2-DOF (Mz, My) biaxial end hinges (``-hingeBiaxial``,
        ``-hingeBiaxialI``, ``-hingeBiaxialJ``, ``-hingeBiaxialIJ``); mutually
        exclusive with uniaxial ``mat_z*`` / ``mat_y*`` hinges.
    shear
        Optional Timoshenko shear rigidity (``-shear``): in 2-D, ``GAvy`` or
        ``(G, Avy)``; in 3-D, ``(GAvy, GAvz)`` or ``(G, Avy, Avz)``. All
        values must be finite and ``> 0``.
    shear_hinge
        Optional series inelastic shear hinge (``-shearHinge``): in 2-D, a
        single :class:`UniaxialMaterial` for Vy–δy; in 3-D, a single
        :class:`UniaxialMaterial` (Vy–δy) or a ``(mat_vy, mat_vz)`` pair.
    shear_hinge_y, shear_hinge_z
        Optional 3-D series inelastic Vy–δy / Vz–δz shear hinges
        (``-shearHingeY`` / ``-shearHingeZ``): each a single
        :class:`UniaxialMaterial` (one shear hinge per axis).
    end_zones
        Optional rigid end-zone offsets ``(dI, dJ)`` (``-endZones dI dJ``,
        both finite and ``>= 0``).
    mass
        Optional ``-mass <rho>`` per-unit-length mass.
    c_mass
        If ``True``, append ``-cMass`` (consistent Hermite beam mass matrix).
    pm
        Optional symmetric parabolic axial-moment interaction ``P0`` or
        ``(P0, alpha)`` (``-pm P0 [alpha]``, ``P0 > 0``). Mutually exclusive
        with ``pm_bal``.
    pm_bal
        Optional asymmetric RC balanced-point axial-moment interaction
        4-tuple ``(P0c, P0t, Pbal, Mbal_ratio)`` (``-pmBal P0c P0t Pbal
        Mbal_ratio``, with ``P0c > 0``, ``P0t > 0``, ``0 < Pbal < P0c``, and
        ``Mbal_ratio >= 1.0``). Mutually exclusive with ``pm``.
    implex
        If ``True``, append ``-implex`` for element-level IMPLEX hinge
        extrapolation.
    """

    pg: str
    transf: _AnyTransf
    A: float
    E: float
    Iz: float
    G: float | None = None
    Jx: float | None = None
    Iy: float | None = None
    ends: str = "both"
    mat_z: UniaxialMaterial | None = None
    mat_y: UniaxialMaterial | None = None
    mat_zi: UniaxialMaterial | None = None
    mat_zj: UniaxialMaterial | None = None
    mat_yi: UniaxialMaterial | None = None
    mat_yj: UniaxialMaterial | None = None
    hinge_biaxial: NDMaterial | None = None
    hinge_biaxial_i: NDMaterial | None = None
    hinge_biaxial_j: NDMaterial | None = None
    hinge_biaxial_ij: NDMaterial | tuple[NDMaterial, NDMaterial] | None = None
    shear: float | tuple[float, float] | tuple[float, float, float] | None = None
    shear_hinge: UniaxialMaterial | tuple[UniaxialMaterial, UniaxialMaterial] | None = None
    shear_hinge_y: UniaxialMaterial | None = None
    shear_hinge_z: UniaxialMaterial | None = None
    end_zones: tuple[float, float] | None = None
    mass: float | None = None
    c_mass: bool = False
    pm: float | tuple[float, float] | None = None
    pm_bal: tuple[float, float, float, float] | None = None
    implex: bool = False

    def __post_init__(self) -> None:
        if not self.pg:
            raise ValueError("LadrunoIMKBeam: pg must be a non-empty name.")
        if self.A <= 0:
            raise ValueError(f"LadrunoIMKBeam: A must be > 0, got {self.A}.")
        if self.E <= 0:
            raise ValueError(f"LadrunoIMKBeam: E must be > 0, got {self.E}.")
        if self.Iz <= 0:
            raise ValueError(f"LadrunoIMKBeam: Iz must be > 0, got {self.Iz}.")

        is_3d = any(v is not None for v in (self.G, self.Jx, self.Iy))
        if is_3d:
            missing = [
                name for name, val in (
                    ("G", self.G), ("Jx", self.Jx), ("Iy", self.Iy),
                )
                if val is None
            ]
            if missing:
                raise ValueError(
                    "LadrunoIMKBeam: 3-D variant requires G, Jx, Iy. "
                    f"Missing: {', '.join(missing)}."
                )
            for name, val in (("G", self.G), ("Jx", self.Jx), ("Iy", self.Iy)):
                if val is not None and val <= 0:
                    raise ValueError(
                        f"LadrunoIMKBeam: {name} must be > 0, got {val}."
                    )

        if self.ends not in ("both", "i", "j", "none"):
            raise ValueError(
                f"LadrunoIMKBeam: ends must be 'both', 'i', 'j', or 'none', "
                f"got {self.ends!r}."
            )

        if not is_3d and any(
            m is not None for m in (self.mat_y, self.mat_yi, self.mat_yj)
        ):
            raise ValueError(
                "LadrunoIMKBeam: weak-axis hinges (mat_y / mat_yi / mat_yj) "
                "are 3-D only — supply G, Jx, Iy for a 3-D element."
            )

        if (
            isinstance(self.hinge_biaxial_ij, tuple)
            and len(self.hinge_biaxial_ij) != 2
        ):
            raise ValueError(
                "LadrunoIMKBeam: hinge_biaxial_ij tuple must have length 2, "
                f"got {len(self.hinge_biaxial_ij)}."
            )

        for fname, mval in (
            ("shear_hinge_y", self.shear_hinge_y),
            ("shear_hinge_z", self.shear_hinge_z),
        ):
            if isinstance(mval, tuple):
                raise ValueError(
                    f"LadrunoIMKBeam: {fname} must be a single UniaxialMaterial "
                    "(one shear hinge per axis, not per end)."
                )

        if isinstance(self.shear_hinge, tuple):
            if not is_3d:
                raise ValueError(
                    "LadrunoIMKBeam: 2-D shear_hinge must be a single "
                    "UniaxialMaterial (one Vy shear hinge per element); a "
                    "(mat_vy, mat_vz) tuple is 3-D only."
                )
            if len(self.shear_hinge) != 2:
                raise ValueError(
                    "LadrunoIMKBeam: shear_hinge tuple must have length 2 "
                    f"(mat_vy, mat_vz), got {len(self.shear_hinge)}."
                )

        has_biaxial = any(
            x is not None
            for x in (
                self.hinge_biaxial, self.hinge_biaxial_i,
                self.hinge_biaxial_j, self.hinge_biaxial_ij,
            )
        )
        has_uniaxial_flex = any(
            x is not None
            for x in (
                self.mat_z, self.mat_y,
                self.mat_zi, self.mat_zj, self.mat_yi, self.mat_yj,
            )
        )
        if not is_3d and has_biaxial:
            raise ValueError(
                "LadrunoIMKBeam: biaxial hinges (hinge_biaxial*) are 3-D only "
                "— supply G, Jx, Iy for a 3-D element."
            )
        if has_biaxial and has_uniaxial_flex:
            raise ValueError(
                "LadrunoIMKBeam: hinge_biaxial* (coupled) is mutually exclusive "
                "with uniaxial flexural hinges (mat_z* / mat_y*)."
            )
        if self.hinge_biaxial is not None and any(
            x is not None
            for x in (
                self.hinge_biaxial_i, self.hinge_biaxial_j, self.hinge_biaxial_ij,
            )
        ):
            raise ValueError(
                "LadrunoIMKBeam: hinge_biaxial is mutually exclusive with "
                "hinge_biaxial_i / hinge_biaxial_j / hinge_biaxial_ij."
            )
        if self.hinge_biaxial_ij is not None and (
            self.hinge_biaxial_i is not None or self.hinge_biaxial_j is not None
        ):
            raise ValueError(
                "LadrunoIMKBeam: hinge_biaxial_ij is mutually exclusive with "
                "hinge_biaxial_i / hinge_biaxial_j."
            )

        if not is_3d and (
            self.shear_hinge_y is not None or self.shear_hinge_z is not None
        ):
            raise ValueError(
                "LadrunoIMKBeam: shear_hinge_y / shear_hinge_z are 3-D only — "
                "use shear_hinge in 2-D or supply G, Jx, Iy for a 3-D element."
            )
        if self.shear_hinge is not None and (
            self.shear_hinge_y is not None or self.shear_hinge_z is not None
        ):
            raise ValueError(
                "LadrunoIMKBeam: shear_hinge is mutually exclusive with "
                "shear_hinge_y / shear_hinge_z."
            )

        if self.shear is not None:
            if isinstance(self.shear, tuple):
                if is_3d and len(self.shear) not in (2, 3):
                    raise ValueError(
                        "LadrunoIMKBeam: 3-D shear must be a 2-tuple "
                        "(GAvy, GAvz) or 3-tuple (G, Avy, Avz), got "
                        f"{self.shear!r}."
                    )
                if not is_3d and len(self.shear) != 2:
                    raise ValueError(
                        "LadrunoIMKBeam: 2-D shear must be a scalar GAvy or "
                        f"2-tuple (G, Avy), got {self.shear!r}."
                    )
                vals: tuple[float, ...] = self.shear
            else:
                if is_3d:
                    raise ValueError(
                        "LadrunoIMKBeam: 3-D shear requires (GAvy, GAvz) or "
                        f"(G, Avy, Avz), got scalar {self.shear!r}."
                    )
                vals = (self.shear,)
            for sv in vals:
                if isinstance(sv, bool) or not (sv > 0) or sv == float("inf"):
                    raise ValueError(
                        "LadrunoIMKBeam: shear values must be finite and > 0, "
                        f"got {self.shear!r}."
                    )

        if self.end_zones is not None:
            if not isinstance(self.end_zones, tuple) or len(self.end_zones) != 2:
                raise ValueError(
                    "LadrunoIMKBeam: end_zones must be a 2-tuple (dI, dJ), "
                    f"got {self.end_zones!r}."
                )
            for dz in self.end_zones:
                if isinstance(dz, bool) or not (dz >= 0) or dz == float("inf"):
                    raise ValueError(
                        "LadrunoIMKBeam: end_zones values must be finite and "
                        f">= 0, got {self.end_zones!r}."
                    )

        _check_optional_mass("LadrunoIMKBeam", self.mass)
        _check_optional_pm("LadrunoIMKBeam", self.pm)
        _check_optional_pm_bal("LadrunoIMKBeam", self.pm_bal, self.pm)

    def _is_3d(self) -> bool:
        return any(v is not None for v in (self.G, self.Jx, self.Iy))

    def dependencies(self) -> tuple[Primitive, ...]:
        deps: list[Primitive] = [self.transf]
        for entry in (
            self.mat_z, self.mat_y, self.mat_zi, self.mat_zj,
            self.mat_yi, self.mat_yj,
            self.hinge_biaxial, self.hinge_biaxial_i,
            self.hinge_biaxial_j, self.hinge_biaxial_ij,
            self.shear_hinge, self.shear_hinge_y, self.shear_hinge_z,
        ):
            if entry is None:
                continue
            if isinstance(entry, tuple):
                for m in entry:
                    if m not in deps:
                        deps.append(m)
            elif entry not in deps:
                deps.append(entry)
        return tuple(deps)

    def _emit(self, emitter: "Emitter", tag: int) -> None:
        nodes = current_element_nodes(emitter)
        _check_two_nodes("LadrunoIMKBeam", nodes)
        i_node, j_node = nodes
        transf_tag = resolve_tag(emitter, self.transf)

        args: list[int | float | str] = [i_node, j_node]
        if self._is_3d():
            assert self.G is not None
            assert self.Jx is not None
            assert self.Iy is not None
            args.extend(
                [self.A, self.E, self.G, self.Jx, self.Iy, self.Iz, transf_tag]
            )
        else:
            args.extend([self.A, self.E, self.Iz, transf_tag])

        if self.ends != "both":
            args.extend(["-hinge", self.ends])
        if self.mat_z is not None:
            args.extend(["-matZ", resolve_tag(emitter, self.mat_z)])
        if self.mat_y is not None:
            args.extend(["-matY", resolve_tag(emitter, self.mat_y)])
        if self.mat_zi is not None:
            args.extend(["-matZi", resolve_tag(emitter, self.mat_zi)])
        if self.mat_zj is not None:
            args.extend(["-matZj", resolve_tag(emitter, self.mat_zj)])
        if self.mat_yi is not None:
            args.extend(["-matYi", resolve_tag(emitter, self.mat_yi)])
        if self.mat_yj is not None:
            args.extend(["-matYj", resolve_tag(emitter, self.mat_yj)])
        if self.hinge_biaxial is not None:
            args.extend(["-hingeBiaxial", resolve_tag(emitter, self.hinge_biaxial)])
        if self.hinge_biaxial_i is not None:
            args.extend(["-hingeBiaxialI", resolve_tag(emitter, self.hinge_biaxial_i)])
        if self.hinge_biaxial_j is not None:
            args.extend(["-hingeBiaxialJ", resolve_tag(emitter, self.hinge_biaxial_j)])
        if self.hinge_biaxial_ij is not None:
            if isinstance(self.hinge_biaxial_ij, tuple):
                args.extend([
                    "-hingeBiaxialIJ",
                    resolve_tag(emitter, self.hinge_biaxial_ij[0]),
                    resolve_tag(emitter, self.hinge_biaxial_ij[1]),
                ])
            else:
                args.extend([
                    "-hingeBiaxialIJ",
                    resolve_tag(emitter, self.hinge_biaxial_ij),
                ])
        if self.shear is not None:
            if isinstance(self.shear, tuple):
                args.extend(["-shear", *(float(v) for v in self.shear)])
            else:
                args.extend(["-shear", float(self.shear)])
        if self.shear_hinge is not None:
            if isinstance(self.shear_hinge, tuple):
                args.extend([
                    "-shearHinge",
                    resolve_tag(emitter, self.shear_hinge[0]),
                    resolve_tag(emitter, self.shear_hinge[1]),
                ])
            else:
                args.extend(["-shearHinge", resolve_tag(emitter, self.shear_hinge)])
        if self.shear_hinge_y is not None:
            args.extend(["-shearHingeY", resolve_tag(emitter, self.shear_hinge_y)])
        if self.shear_hinge_z is not None:
            args.extend(["-shearHingeZ", resolve_tag(emitter, self.shear_hinge_z)])
        if self.end_zones is not None:
            args.extend([
                "-endZones",
                float(self.end_zones[0]),
                float(self.end_zones[1]),
            ])
        if self.mass is not None:
            args.extend(["-mass", self.mass])
        if self.c_mass:
            args.append("-cMass")
        args.extend(_pm_args(self.pm))
        args.extend(_pm_bal_args(self.pm_bal))
        if self.implex:
            args.append("-implex")

        emitter.element("LadrunoIMKBeam", tag, *args)

