"""
Shell element primitives — typed wrappers over the OpenSees ``element
Shell*`` family.

Five element types live here:

* :class:`ShellMITC4` — 4-node MITC shell (membrane + bending).
* :class:`ShellDKGQ` — 4-node Discrete Kirchhoff (Quadrilateral) shell.
* :class:`ShellNLDKGQ` — its geometrically nonlinear (updated Lagrangian)
  twin, whose tangent carries the initial-stress stiffness.
* :class:`ASDShellQ4` — 4-node ASD shell (Petracca/ASDEA Software).
* :class:`ASDShellT3` — 3-node ASD shell.

The OpenSees commands per the manual:

* ``element ShellMITC4 tag i j k l secTag``
* ``element ShellDKGQ  tag i j k l secTag``
* ``element ShellNLDKGQ tag i j k l secTag``
* ``element ASDShellQ4 tag i j k l secTag [-corotational]``
  ``[-drillingNT alpha] [-localCS x1 x2 x3 y1 y2 y3]``
* ``element ASDShellT3 tag i j k secTag [-corotational]``
  ``[-drillingDOF dof_id] [-localCS x1 x2 x3 y1 y2 y3]``

``ShellMITC3`` was removed (#2092): no OpenSees build has the element, so
every model that used it failed at emit. ``ASDShellT3`` is the 3-node shell
to use. ``ops.element.ShellMITC3`` and a lookup of the class by name raise
:data:`~apeGmsh.opensees._element_capabilities.REMOVED_ELEMENT_CLASSES`'s
message, which names the replacements.

Element fan-out contract
========================

Each element class composes one :class:`Section` (referenced via its
allocated tag at emit time) and one or more nodes. Per the contract in
:mod:`apeGmsh.opensees._internal.tag_resolution`, the bridge fans out
the element's physical group at build time and sets:

* the section's allocated tag via the resolver attached to the emitter
  (looked up via :func:`resolve_tag`).
* the node tags for **the current element** of the fan-out via
  :func:`set_element_nodes` on the emitter (read by ``_emit`` via
  :func:`current_element_nodes`).

Tests install both contexts manually with :func:`set_tag_resolver` +
:func:`set_element_nodes`.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import ClassVar, Sequence, TYPE_CHECKING

from .._internal.tag_resolution import (
    ATTR_ELEMENT_NODES,
    clear_element_emit_context,
    current_element_nodes,
    current_membrane_context,
    damp_args,
    resolve_tag,
)
from .._internal.types import Damping, Element, NDMaterial, Primitive, Section
from ..node import Node, NodeSet
from ..section.plate import ElasticMembranePlateSection

if TYPE_CHECKING:
    from ..emitter.base import Emitter


__all__ = [
    "ShellMITC4",
    "ShellDKGQ",
    "ShellNLDKGQ",
    "ASDShellQ4",
    "ASDShellT3",
    "LadrunoMembrane",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _check_local_cs(type_name: str, local_cs: tuple[float, ...]) -> None:
    """``-local`` takes the local x axis: exactly three floats, not all zero.

    The ASD shells' parsers (``ASDShellQ4.cpp`` / ``ASDShellT3.cpp``,
    ``strcmp(type, "-local")``) read three components and build the rest of
    the frame themselves (x projected onto the shell plane, z the normal,
    y = z cross x). They have no ``-localCS``: that token, and its six
    values, fell through the option loop unread, so the element silently
    kept its default frame. A 6-tuple is refused rather than truncated: a
    script that passed one was running on the default frame, and quietly
    switching it to the requested one would change its answers unannounced.
    """
    if len(local_cs) == 6:
        raise ValueError(
            f"{type_name}: local_cs= takes the local x axis only, "
            f"(x1, x2, x3), emitted as '-local x1 x2 x3'. The old 6-tuple "
            f"form emitted '-localCS', which the element does not parse, so "
            f"it always ran on the default frame. Pass local_cs="
            f"{tuple(local_cs[:3])!r}; the element derives y from the "
            f"normal."
        )
    if len(local_cs) != 3:
        raise ValueError(
            f"{type_name}: local_cs= must be a 3-tuple (x1, x2, x3), the "
            f"local x axis; got {len(local_cs)} entries."
        )
    if not any(float(c) != 0.0 for c in local_cs):
        raise ValueError(
            f"{type_name}: local_cs= must not be the zero vector (the element "
            f"reads a zero -local vector as 'use the default frame')."
        )


def _check_p_delta(type_name: str, p_delta: bool, corotational: bool) -> None:
    """``-pDelta`` and ``-corotational`` are exclusive (the fork refuses
    the element when both are given)."""
    if p_delta and corotational:
        raise ValueError(
            f"{type_name}: p_delta=True and corotational=True are mutually "
            f"exclusive (the parser refuses -pDelta with -corotational). "
            f"-pDelta is linear kinematics plus the initial-stress "
            f"stiffness; -corotational already carries the geometric "
            f"nonlinearity."
        )


# ---------------------------------------------------------------------------
# ShellMITC4 — 4-node MITC shell
# ---------------------------------------------------------------------------

@dataclass(frozen=True, kw_only=True, slots=True)
class ShellMITC4(Element):
    """``element ShellMITC4`` — 4-node MITC shell.

    Parameters
    ----------
    pg
        Physical group whose surface (quadrilateral) cells receive
        this element. The bridge fans out at build time; per element
        the four node tags are read from the emitter context by
        :func:`current_element_nodes`.
    section
        The plate / shell :class:`Section` (typically
        :class:`~apeGmsh.opensees.section.plate.ElasticMembranePlateSection`,
        :class:`~apeGmsh.opensees.section.plate.LayeredShell`, or
        :class:`~apeGmsh.opensees.section.plate.LayeredShellFiberSection`).
    """

    pg: str
    section: Section
    damp: Damping | None = None

    def _emit(self, emitter: "Emitter", tag: int) -> None:
        nodes = current_element_nodes(emitter)
        if len(nodes) != 4:
            raise ValueError(
                f"ShellMITC4: expected 4 node tags, got {len(nodes)}."
            )
        sec_tag = resolve_tag(emitter, self.section)
        emitter.element(
            "ShellMITC4", tag, *nodes, sec_tag,
            *damp_args(emitter, self.damp),
        )

    def dependencies(self) -> tuple[Primitive, ...]:
        if self.damp is not None:
            return (self.section, self.damp)
        return (self.section,)


# ---------------------------------------------------------------------------
# ShellDKGQ — 4-node Discrete Kirchhoff (Quadrilateral) shell
# ---------------------------------------------------------------------------

@dataclass(frozen=True, kw_only=True, slots=True)
class ShellDKGQ(Element):
    """``element ShellDKGQ`` — 4-node Discrete Kirchhoff Quadrilateral.

    Parameters
    ----------
    pg
        Physical group whose surface (quadrilateral) cells receive
        this element.
    section
        The plate / shell :class:`Section`.
    """

    pg: str
    section: Section
    damp: Damping | None = None

    def _emit(self, emitter: "Emitter", tag: int) -> None:
        nodes = current_element_nodes(emitter)
        if len(nodes) != 4:
            raise ValueError(
                f"ShellDKGQ: expected 4 node tags, got {len(nodes)}."
            )
        sec_tag = resolve_tag(emitter, self.section)
        emitter.element(
            "ShellDKGQ", tag, *nodes, sec_tag,
            *damp_args(emitter, self.damp),
        )

    def dependencies(self) -> tuple[Primitive, ...]:
        if self.damp is not None:
            return (self.section, self.damp)
        return (self.section,)


# ---------------------------------------------------------------------------
# ShellNLDKGQ — geometrically nonlinear Discrete Kirchhoff quadrilateral
# ---------------------------------------------------------------------------

@dataclass(frozen=True, kw_only=True, slots=True)
class ShellNLDKGQ(Element):
    """``element ShellNLDKGQ`` — geometrically nonlinear 4-node DKGQ shell.

    The updated-Lagrangian twin of :class:`ShellDKGQ`: its tangent carries
    the initial-stress (geometric) stiffness of the current membrane
    forces, so an ``eigen`` after a gravity preload sees the P-Delta
    softening of a wall (ADR 0124). Under ``algorithm Linear`` its nodal
    forces read stale (``shell_probe``); solve with ``Newton``.

    Parameters
    ----------
    pg
        Physical group whose surface (quadrilateral) cells receive
        this element.
    section
        The plate / shell :class:`Section`.
    """

    pg: str
    section: Section
    damp: Damping | None = None

    def _emit(self, emitter: "Emitter", tag: int) -> None:
        nodes = current_element_nodes(emitter)
        if len(nodes) != 4:
            raise ValueError(
                f"ShellNLDKGQ: expected 4 node tags, got {len(nodes)}."
            )
        sec_tag = resolve_tag(emitter, self.section)
        emitter.element(
            "ShellNLDKGQ", tag, *nodes, sec_tag,
            *damp_args(emitter, self.damp),
        )

    def dependencies(self) -> tuple[Primitive, ...]:
        if self.damp is not None:
            return (self.section, self.damp)
        return (self.section,)


# ---------------------------------------------------------------------------
# ASDShellQ4 — 4-node ASD shell
# ---------------------------------------------------------------------------

@dataclass(frozen=True, kw_only=True, slots=True)
class ASDShellQ4(Element):
    """``element ASDShellQ4`` — 4-node ASD shell.

    Optional flags, as ``OPS_ASDShellQ4`` parses them: ``-corotational``
    / ``-pDelta`` (mutually exclusive), ``-noeas``, ``-drillingStab $v`` /
    ``-drillingNL`` (mutually exclusive) and ``-local $x1 $x2 $x3``.

    Parameters
    ----------
    pg
        Physical group whose surface (quadrilateral) cells receive
        this element.
    section
        The plate / shell :class:`Section`.
    corotational
        Append the ``-corotational`` flag.
    p_delta
        Append ``-pDelta`` (**Ladruno fork**, at or after
        ``stage_emit.C83_FORK_COMMIT`` c83ee812e): linear kinematics
        plus the initial-stress (geometric) stiffness of the current
        membrane forces, the shell analogue of ``geomTransf PDelta``, for a
        gravity, ``loadConst``, ``eigen`` sequence. Refused with
        ``corotational=True`` (the fork refuses the pair). Stock OpenSees
        and older fork builds accept the token and drop the stiffness, so
        the live route refuses any build not measured to carry c83ee812e
        (``stage_emit.FORK_BUILD_HAS_C83``) and a Tcl / py deck checks
        ``ladrunoBuild`` at its start (``ASDSHELL_PDELTA_GUARD``).
    no_eas
        Append ``-noeas``: turn off the enhanced membrane (the AGQI
        incompatible modes and their EAS parameters), leaving the plain
        bilinear membrane. The default keeps the enhancement.
    drilling_stab
        If supplied, append ``-drillingStab <v>``: the drilling
        stabilization factor, in ``[0, 1]`` (the parser clamps to that
        range; the default without the flag is 0.01).
    drilling_nl
        Append ``-drillingNL`` (nonlinear drilling DOF treatment). The
        parser refuses it together with ``-drillingStab``.
    drilling_nt_alpha
        **Refused.** It emitted ``-drillingNT``, which ASDShellQ4 does not
        parse (the token fell through unread). Use ``drilling_stab`` or
        ``drilling_nl``.
    local_cs
        The local x axis ``(x1, x2, x3)``, emitted as ``-local x1 x2 x3``.
        The element projects it onto the shell plane and takes y = z cross
        x. This is also the section frame the layers of a layered section
        see, so a ``PlateRebar`` angle is measured from it.
    """

    pg: str
    section: Section
    corotational: bool = False
    p_delta: bool = False
    no_eas: bool = False
    drilling_stab: float | None = None
    drilling_nl: bool = False
    drilling_nt_alpha: float | None = None
    local_cs: tuple[float, ...] | None = None
    damp: Damping | None = None

    def __post_init__(self) -> None:
        _check_p_delta("ASDShellQ4", self.p_delta, self.corotational)
        if self.local_cs is not None:
            _check_local_cs("ASDShellQ4", self.local_cs)
        if self.drilling_nt_alpha is not None:
            raise ValueError(
                "ASDShellQ4: drilling_nt_alpha emitted '-drillingNT', which "
                "the element does not parse; it was silently ignored. Use "
                "drilling_stab=<v in [0, 1]> ('-drillingStab') or "
                "drilling_nl=True ('-drillingNL')."
            )
        if self.drilling_stab is not None:
            if not (0.0 <= float(self.drilling_stab) <= 1.0):
                raise ValueError(
                    f"ASDShellQ4: drilling_stab must be in [0, 1], got "
                    f"{self.drilling_stab!r} (the parser would clamp it)."
                )
            if self.drilling_nl:
                raise ValueError(
                    "ASDShellQ4: drilling_stab and drilling_nl are mutually "
                    "exclusive (the parser refuses -drillingStab with "
                    "-drillingNL)."
                )

    def _emit(self, emitter: "Emitter", tag: int) -> None:
        nodes = current_element_nodes(emitter)
        if len(nodes) != 4:
            raise ValueError(
                f"ASDShellQ4: expected 4 node tags, got {len(nodes)}."
            )
        sec_tag = resolve_tag(emitter, self.section)
        args: list[int | float | str] = [*nodes, sec_tag]
        if self.corotational:
            args.append("-corotational")
        if self.p_delta:
            args.append("-pDelta")
        if self.no_eas:
            args.append("-noeas")
        if self.drilling_stab is not None:
            args.append("-drillingStab")
            args.append(float(self.drilling_stab))
        if self.drilling_nl:
            args.append("-drillingNL")
        if self.local_cs is not None:
            args.append("-local")
            args.extend(float(c) for c in self.local_cs)
        args.extend(damp_args(emitter, self.damp))
        emitter.element("ASDShellQ4", tag, *args)

    def dependencies(self) -> tuple[Primitive, ...]:
        if self.damp is not None:
            return (self.section, self.damp)
        return (self.section,)


# ---------------------------------------------------------------------------
# ASDShellT3 — 3-node ASD shell
# ---------------------------------------------------------------------------

@dataclass(frozen=True, kw_only=True, slots=True)
class ASDShellT3(Element):
    """``element ASDShellT3`` — 3-node ASD shell.

    Phase 2γ scope ships the canonical positional arguments plus the
    Optional flags, as ``OPS_ASDShellT3`` parses them: ``-corotational``
    / ``-pDelta`` (mutually exclusive), ``-drillingNL`` and ``-local $x1
    $x2 $x3``. ``-reducedIntegration`` is not exposed.

    Parameters
    ----------
    pg
        Physical group whose surface (triangular) cells receive this
        element.
    section
        The plate / shell :class:`Section`.
    corotational
        Append the ``-corotational`` flag.
    p_delta
        Append ``-pDelta`` (**Ladruno fork**; see ``ASDShellQ4``).
        Refused with ``corotational=True``.
    drilling_nl
        Append ``-drillingNL`` (nonlinear drilling DOF treatment).
    drilling_dof
        **Refused.** It emitted ``-drillingDOF <dof_id>``, which ASDShellT3
        does not parse (the token and its value fell through unread). The
        drilling DOF is always the element's sixth; use ``drilling_nl`` for
        the only drilling option the parser has.
    local_cs
        The local x axis ``(x1, x2, x3)``, emitted as ``-local x1 x2 x3``
        (the parser has no ``-localCS``; see ``ASDShellQ4``).
    """

    pg: str
    section: Section
    corotational: bool = False
    p_delta: bool = False
    drilling_nl: bool = False
    drilling_dof: int | None = None
    local_cs: tuple[float, ...] | None = None
    damp: Damping | None = None

    def __post_init__(self) -> None:
        _check_p_delta("ASDShellT3", self.p_delta, self.corotational)
        if self.local_cs is not None:
            _check_local_cs("ASDShellT3", self.local_cs)
        if self.drilling_dof is not None:
            raise ValueError(
                "ASDShellT3: drilling_dof emitted '-drillingDOF', which the "
                "element does not parse; it was silently ignored. The only "
                "drilling option ASDShellT3 has is drilling_nl=True "
                "('-drillingNL')."
            )

    def _emit(self, emitter: "Emitter", tag: int) -> None:
        nodes = current_element_nodes(emitter)
        if len(nodes) != 3:
            raise ValueError(
                f"ASDShellT3: expected 3 node tags, got {len(nodes)}."
            )
        sec_tag = resolve_tag(emitter, self.section)
        args: list[int | float | str] = [*nodes, sec_tag]
        if self.corotational:
            args.append("-corotational")
        if self.p_delta:
            args.append("-pDelta")
        if self.drilling_nl:
            args.append("-drillingNL")
        if self.local_cs is not None:
            args.append("-local")
            args.extend(float(c) for c in self.local_cs)
        args.extend(damp_args(emitter, self.damp))
        emitter.element("ASDShellT3", tag, *args)

    def dependencies(self) -> tuple[Primitive, ...]:
        return (self.section,)


# ---------------------------------------------------------------------------
# LadrunoMembrane — N-node polygon membrane + load-distribution macro-element
# ---------------------------------------------------------------------------

_MEMBRANE_METHODS: frozenset[str] = frozenset({
    "tributary",
    "oneway",
    "fea_poisson",
    "fea_plate",
})

MembraneNodeRef = int | str | Node


def _normalize_node_ref(cls_name: str, field_name: str, val: object) -> int | str:
    if isinstance(val, bool) or not isinstance(val, (int, str, Node)):
        raise TypeError(
            f"{cls_name}: {field_name} must be an integer node tag, Node, or "
            f"point label/PG name, got {type(val).__name__}."
        )
    if isinstance(val, str):
        if not val:
            raise ValueError(f"{cls_name}: {field_name} name must be non-empty.")
        return val
    tag = int(val.tag) if isinstance(val, Node) else int(val)
    if tag <= 0:
        raise ValueError(
            f"{cls_name}: {field_name} node tag must be > 0, got {tag}."
        )
    return tag


def _normalize_node_list(
    cls_name: str, field_name: str, value: object,
) -> tuple[int | str, ...]:
    """A node ref, ``NodeSet`` or sequence of them, as tags and names.

    ``()`` and an empty ``NodeSet`` (a group that matched nothing) are
    refused alike: either would drop the flag from the deck in silence.
    """
    if isinstance(value, (int, str, Node, NodeSet)) and not isinstance(value, bool):
        raw: tuple[object, ...] = (value,)
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        raw = tuple(value)
    else:
        raise TypeError(
            f"{cls_name}: {field_name}= must be a node tag, Node, NodeSet, "
            f"PG/label name, or sequence thereof."
        )
    if not raw:
        raise ValueError(f"{cls_name}: {field_name}= must not be empty.")
    out: list[int | str] = []
    for idx, item in enumerate(raw):
        where = f"{field_name}[{idx}]"
        if isinstance(item, NodeSet):
            if not item.tags:
                raise ValueError(
                    f"{cls_name}: {field_name}= must not be empty ({where} is "
                    f"a NodeSet with no nodes)."
                )
            out.extend(_normalize_node_ref(cls_name, where, int(t)) for t in item.tags)
        elif isinstance(item, bool) or not isinstance(item, (int, str, Node)):
            raise TypeError(
                f"{cls_name}: {where} must be an int tag, Node, NodeSet, or "
                f"PG/label name, got {type(item).__name__}."
            )
        else:
            out.append(_normalize_node_ref(cls_name, where, item))
    return tuple(out)


def _resolve_node_ref(cls_name: str, field_name: str, val: object) -> int:
    norm = _normalize_node_ref(cls_name, field_name, val)
    if isinstance(norm, str):
        raise RuntimeError(
            f"{cls_name}: {field_name}={norm!r} requires emitting through "
            "ops.build().emit(...)."
        )
    return norm


@dataclass(frozen=True, kw_only=True, slots=True)
class LadrunoMembrane(Element):
    """``element LadrunoMembrane`` — polygon floor/wall diaphragm and load distributor.

    Ladruno fork (``ELE_TAG = 33022``). Emits one macro-element per polygon
    panel rather than fanning out per Gmsh cell:

    ``element LadrunoMembrane $tag -nodes $n1 $n2 $n3 ... [-frames $f1 ...]``
    ``[-supportNodes $sn1 ...] [-hole $h1 $h2 $h3 ...]``
    ``[-diaphragmNodes $d1 ...] [-method $method]``
    ``[-spanAxis $dx $dy [$dz]] [-orthoRatio $eta] [-pressure $q]``
    ``[-thick $t -mat $matTag | -stiff $E $nu $t] [-f11 $v] [-f22 $v]``
    ``[-f12 $v] [-rho $rho] [-lumped] [-meshSize $hMax] [-damp $dampTag]``

    With a stiffness (``stiff=``, ``E=``/``nu=``/``thickness=``, ``mat=``
    with ``thickness=``, or ``section=``), every outer, hole and support
    node gets the panel's condensed in-plane (diaphragm) stiffness and
    mass. ``diaphragm_nodes=`` adds nodes that get that stiffness and mass
    but no gravity share, such as the nodes of an edge beam or of a wall
    along an opening. It takes the forms ``support_nodes=`` takes (tags,
    ``Node``, ``NodeSet``, PG or label names). ``f11=``, ``f22=`` and
    ``f12=`` scale the plane-stress modulus as ``D_ij * sqrt(f_i * f_j)``
    in the element's local axes (e1 = global X and e2 = global Y for a
    horizontal slab), as ``LadrunoShellModifier`` does; the mass is not
    scaled. ``mesh_size=`` also sets the diaphragm mesh, and point
    connections depend on it. ``c83ee812e`` parses these options but
    gives an in-plane stiffness 26 to 28 percent low; the exact one needs
    fork build ``b6d3e3d2`` (fork WP-208) or later. The live route refuses
    them on a build not measured to carry it
    (``stage_emit.FORK_BUILD_HAS_MEMBRANE_WP208``), and a Tcl / py deck
    checks ``ladrunoBuild`` at its start (``MEMBRANE_WP208_GUARD``).

    The element answers ``membraneForces`` (``N11, N22, N12`` per diaphragm
    triangle), ``meanMembraneForce``, ``diaphragmForces`` (the in-plane
    force on each element node), ``diaphragmVertices`` and
    ``diaphragmTriangles``, through ``eleResponse`` and element recorders.

    ``pressure=`` does not load the slab by itself. The fork reads it only
    inside ``LadrunoMembrane::addLoad``, as the reference value that a
    pattern's ``surface_load`` scales. A nonzero ``self_weight`` scales it
    too, but only when the membrane has no ``rho`` x thickness (``rho=``,
    the section's ``rho``, or the ``mat=`` density); otherwise it scales
    ``rho`` x thickness. ``beam_uniform`` never reads it. Activate it from a
    ``Plain`` pattern, top level or a stage's ``s.pattern(...)``, by its
    handle (``ele=slab``) or ``pg=``::

        with ops.pattern.Plain(series=ops.timeSeries.Linear()) as p:
            p.surface_load(ele=slab)

    ``ops.build()`` refuses a nonzero ``pressure=`` that no ``Plain``
    pattern reads on one of those paths: without one, every support
    reaction reads 0.0 (#2151). Drop ``pressure=`` for a modal or
    stiffness-only model. ``label=`` does not reach a membrane, which has
    no mesh cell.

    ``rho=`` with a thickness is both the self-weight density that
    ``self_weight`` scales and the mass of the element's mass matrix, so a
    modal or ``UniformExcitation`` run reads it with no ``Plain`` pattern.
    """

    ELE_TAG: ClassVar[int] = 33022

    pg: str | None = None
    nodes: tuple[MembraneNodeRef, ...] | None = None
    frames: tuple[int | str | Element, ...] | None = None
    support_nodes: tuple[int | str | Node | NodeSet, ...] | None = None
    holes: tuple[tuple[MembraneNodeRef, ...] | str, ...] | None = None
    diaphragm_nodes: tuple[int | str | Node | NodeSet, ...] | None = None
    method: str = "tributary"
    span_axis: tuple[float, ...] | None = None
    ortho_ratio: float | None = None
    pressure: float | None = None
    section: Section | None = None
    mat: NDMaterial | None = None
    E: float | None = None
    nu: float | None = None
    thickness: float | None = None
    thick: float | None = None
    stiff: tuple[float, float, float] | None = None
    rho: float | None = None
    lumped: bool = False
    mesh_size: float | None = None
    f11: float | None = None
    f22: float | None = None
    f12: float | None = None
    damp: Damping | None = None

    def __post_init__(self) -> None:
        has_pg = self.pg is not None
        has_nodes = self.nodes is not None
        if has_pg == has_nodes:
            raise ValueError(
                "LadrunoMembrane: pass exactly one of pg= or nodes=(...); got "
                f"pg={self.pg!r}, nodes={self.nodes!r}."
            )
        if has_pg and (not isinstance(self.pg, str) or not self.pg):
            raise ValueError(
                f"LadrunoMembrane: pg= must be a non-empty string, got {self.pg!r}."
            )
        if self.nodes is not None:
            if not isinstance(self.nodes, Sequence) or isinstance(self.nodes, (str, bytes)):
                raise TypeError(
                    "LadrunoMembrane: nodes= must be a sequence of >= 3 node refs."
                )
            raw_n = tuple(self.nodes)
            if len(raw_n) < 3:
                raise ValueError(
                    f"LadrunoMembrane: nodes= requires at least 3 boundary nodes, "
                    f"got {len(raw_n)}."
                )
            n_tup = tuple(
                _normalize_node_ref("LadrunoMembrane", f"nodes[{idx}]", n)
                for idx, n in enumerate(raw_n)
            )
            if len(set(n_tup)) < 3:
                raise ValueError(
                    f"LadrunoMembrane: nodes= requires at least 3 distinct boundary "
                    f"nodes, got {n_tup!r}."
                )
            object.__setattr__(self, "nodes", n_tup)

        if self.frames is not None:
            if isinstance(self.frames, (int, str, Element)) and not isinstance(self.frames, bool):
                f_tup: tuple[int | str | Element, ...] = (self.frames,)
            elif isinstance(self.frames, Sequence) and not isinstance(self.frames, (str, bytes)):
                f_tup = tuple(self.frames)
            else:
                raise TypeError(
                    "LadrunoMembrane: frames= must be an Element, PG/label name, "
                    "integer tag, or sequence thereof."
                )
            for idx, item in enumerate(f_tup):
                if isinstance(item, bool) or not isinstance(item, (int, str, Element)):
                    raise TypeError(
                        f"LadrunoMembrane: frames[{idx}] must be an int tag, PG/label "
                        f"name, or Element spec, got {type(item).__name__}."
                    )
                if isinstance(item, int) and item <= 0:
                    raise ValueError(
                        f"LadrunoMembrane: frames[{idx}] tag must be positive, got {item}."
                    )
                if isinstance(item, str) and not item:
                    raise ValueError(
                        f"LadrunoMembrane: frames[{idx}] name must be non-empty."
                    )
            object.__setattr__(self, "frames", f_tup)

        if self.support_nodes is not None:
            object.__setattr__(
                self, "support_nodes",
                _normalize_node_list(
                    "LadrunoMembrane", "support_nodes", self.support_nodes,
                ),
            )

        if self.diaphragm_nodes is not None:
            object.__setattr__(
                self, "diaphragm_nodes",
                _normalize_node_list(
                    "LadrunoMembrane", "diaphragm_nodes", self.diaphragm_nodes,
                ),
            )

        if self.holes is not None:
            if isinstance(self.holes, str):
                if not self.holes:
                    raise ValueError("LadrunoMembrane: holes= PG name must be non-empty.")
                h_tup: tuple[tuple[MembraneNodeRef, ...] | str, ...] = (self.holes,)
            elif isinstance(self.holes, Sequence) and not isinstance(self.holes, (str, bytes)):
                raw_holes = tuple(self.holes)
                if raw_holes and all(
                    isinstance(x, (int, Node)) and not isinstance(x, bool)
                    for x in raw_holes
                ):
                    raw_holes = (raw_holes,)  # type: ignore[assignment]
                norm_holes: list[tuple[MembraneNodeRef, ...] | str] = []
                for h_idx, h in enumerate(raw_holes):
                    if isinstance(h, str):
                        if not h:
                            raise ValueError(
                                f"LadrunoMembrane: holes[{h_idx}] name must be non-empty."
                            )
                        norm_holes.append(h)
                    elif isinstance(h, Sequence) and not isinstance(h, (str, bytes)):
                        raw_loop = tuple(h)
                        if len(raw_loop) < 3:
                            raise ValueError(
                                f"LadrunoMembrane: holes[{h_idx}] requires at least 3 "
                                f"nodes, got {len(raw_loop)}."
                            )
                        loop = tuple(
                            _normalize_node_ref(
                                "LadrunoMembrane", f"holes[{h_idx}][{n_idx}]", n
                            )
                            for n_idx, n in enumerate(raw_loop)
                        )
                        if len(set(loop)) < 3:
                            raise ValueError(
                                f"LadrunoMembrane: holes[{h_idx}] requires at least 3 "
                                f"distinct nodes, got {loop!r}."
                            )
                        norm_holes.append(loop)
                    else:
                        raise TypeError(
                            f"LadrunoMembrane: holes[{h_idx}] must be a sequence of "
                            f">= 3 node refs or a PG/label name."
                        )
                h_tup = tuple(norm_holes)
            else:
                raise TypeError(
                    "LadrunoMembrane: holes= must be a sequence of hole loops or PG names."
                )
            object.__setattr__(self, "holes", h_tup)

        if self.method not in _MEMBRANE_METHODS:
            raise ValueError(
                f"LadrunoMembrane: method must be one of {sorted(_MEMBRANE_METHODS)}, "
                f"got {self.method!r}."
            )

        if self.span_axis is not None:
            if not isinstance(self.span_axis, Sequence) or isinstance(
                self.span_axis, (str, bytes)
            ):
                raise TypeError("LadrunoMembrane: span_axis= must be a 2- or 3-tuple of floats.")
            sa = tuple(float(c) for c in self.span_axis)
            if len(sa) not in (2, 3):
                raise ValueError(
                    f"LadrunoMembrane: span_axis= must have 2 or 3 components, got {len(sa)}."
                )
            if not all(math.isfinite(c) for c in sa) or math.hypot(*sa) <= 0.0:
                raise ValueError(
                    "LadrunoMembrane: span_axis= must be a finite non-zero vector."
                )
            object.__setattr__(self, "span_axis", sa)

        if self.ortho_ratio is not None:
            eta = float(self.ortho_ratio)
            if not math.isfinite(eta) or eta <= 0.0 or eta > 1.0:
                raise ValueError(
                    "LadrunoMembrane: ortho_ratio= must be finite and in (0, 1], "
                    f"got {self.ortho_ratio!r}."
                )
            object.__setattr__(self, "ortho_ratio", eta)

        if self.pressure is not None:
            q = float(self.pressure)
            if not math.isfinite(q):
                raise ValueError(
                    f"LadrunoMembrane: pressure= must be finite, got {self.pressure!r}."
                )
            object.__setattr__(self, "pressure", q)

        if self.thickness is not None and self.thick is not None:
            if float(self.thickness) != float(self.thick):
                raise ValueError(
                    "LadrunoMembrane: pass at most one of thickness= or thick=."
                )
        eff_thick = self.thickness if self.thickness is not None else self.thick

        if self.stiff is not None:
            if (
                eff_thick is not None
                or self.E is not None
                or self.nu is not None
                or self.mat is not None
                or self.section is not None
            ):
                raise ValueError(
                    "LadrunoMembrane: stiff= is mutually exclusive with "
                    "E=/nu=/thickness=/thick=/mat= and section=."
                )
            if not isinstance(self.stiff, Sequence) or isinstance(self.stiff, (str, bytes)):
                raise TypeError("LadrunoMembrane: stiff= must be a 3-tuple (E, nu, t).")
            st = tuple(float(c) for c in self.stiff)
            if len(st) != 3:
                raise ValueError(
                    f"LadrunoMembrane: stiff= must be a 3-tuple (E, nu, t), got {len(st)} items."
                )
            E_val, nu_val, t_val = st
            if not (math.isfinite(E_val) and E_val > 0.0):
                raise ValueError(f"LadrunoMembrane: stiff E must be > 0, got {E_val!r}.")
            if not (math.isfinite(nu_val) and -1.0 < nu_val < 0.5):
                raise ValueError(
                    f"LadrunoMembrane: stiff nu must be in (-1, 0.5), got {nu_val!r}."
                )
            if not (math.isfinite(t_val) and t_val > 0.0):
                raise ValueError(f"LadrunoMembrane: stiff t must be > 0, got {t_val!r}.")
            object.__setattr__(self, "stiff", (E_val, nu_val, t_val))
            object.__setattr__(self, "E", E_val)
            object.__setattr__(self, "nu", nu_val)
            object.__setattr__(self, "thickness", t_val)
            object.__setattr__(self, "thick", t_val)
        elif self.E is not None or self.nu is not None:
            if self.mat is not None or self.section is not None:
                raise ValueError(
                    "LadrunoMembrane: E=/nu=/thickness= is mutually exclusive with "
                    "mat= and section=."
                )
            if self.E is None or self.nu is None or eff_thick is None:
                raise ValueError(
                    "LadrunoMembrane: E=, nu=, and thickness= must all be provided "
                    "together for isotropic -stiff."
                )
            E_val = float(self.E)
            nu_val = float(self.nu)
            t_val = float(eff_thick)
            if not (math.isfinite(E_val) and E_val > 0.0):
                raise ValueError(f"LadrunoMembrane: E= must be > 0, got {self.E!r}.")
            if not (math.isfinite(nu_val) and -1.0 < nu_val < 0.5):
                raise ValueError(
                    f"LadrunoMembrane: nu= must be in (-1, 0.5), got {self.nu!r}."
                )
            if not (math.isfinite(t_val) and t_val > 0.0):
                raise ValueError(
                    f"LadrunoMembrane: thickness= must be > 0, got {eff_thick!r}."
                )
            object.__setattr__(self, "E", E_val)
            object.__setattr__(self, "nu", nu_val)
            object.__setattr__(self, "thickness", t_val)
            object.__setattr__(self, "thick", t_val)
            object.__setattr__(self, "stiff", (E_val, nu_val, t_val))
        elif self.mat is not None:
            if self.section is not None:
                raise ValueError(
                    "LadrunoMembrane: section= is mutually exclusive with mat=/thickness=."
                )
            if eff_thick is None:
                raise ValueError(
                    "LadrunoMembrane: thickness= (or thick=) and mat= must be supplied "
                    "together (-thick $t -mat $matTag)."
                )
            t_val = float(eff_thick)
            if not math.isfinite(t_val) or t_val <= 0.0:
                raise ValueError(
                    f"LadrunoMembrane: thickness= must be > 0, got {eff_thick!r}."
                )
            if not isinstance(self.mat, NDMaterial):
                raise TypeError(
                    f"LadrunoMembrane: mat= must be an NDMaterial, got {type(self.mat).__name__}."
                )
            object.__setattr__(self, "thickness", t_val)
            object.__setattr__(self, "thick", t_val)
        elif eff_thick is not None:
            raise ValueError(
                "LadrunoMembrane: thickness= (or thick=) requires either mat= or both E= and nu=."
            )
        elif self.section is not None:
            if not isinstance(self.section, ElasticMembranePlateSection):
                raise TypeError(
                    "LadrunoMembrane: section= must be an ElasticMembranePlateSection "
                    "(or pass mat=/thickness= or E=/nu=/thickness=); got "
                    f"{type(self.section).__name__}."
                )

        if self.rho is not None:
            r_val = float(self.rho)
            if not math.isfinite(r_val) or r_val < 0.0:
                raise ValueError(
                    f"LadrunoMembrane: rho= must be >= 0, got {self.rho!r}."
                )
            object.__setattr__(self, "rho", r_val)

        if self.mesh_size is not None:
            ms_val = float(self.mesh_size)
            if not math.isfinite(ms_val) or ms_val <= 0.0:
                raise ValueError(
                    f"LadrunoMembrane: mesh_size= must be > 0, got {self.mesh_size!r}."
                )
            object.__setattr__(self, "mesh_size", ms_val)

        for f_name in ("f11", "f22", "f12"):
            f_raw = getattr(self, f_name)
            if f_raw is None:
                continue
            f_val = float(f_raw)
            if not (math.isfinite(f_val) and f_val > 0.0):
                raise ValueError(
                    f"LadrunoMembrane: {f_name}= must be finite and > 0, got "
                    f"{f_raw!r}."
                )
            object.__setattr__(self, f_name, f_val)
        if self.stiff is None and self.mat is None and self.section is None:
            given = [
                n for n in ("diaphragm_nodes", "f11", "f22", "f12")
                if getattr(self, n) is not None
            ]
            if given:
                names = ", ".join(f"{n}=" for n in given)
                raise ValueError(
                    f"LadrunoMembrane: {names} act on the in-plane stiffness, "
                    f"and this membrane has none. Give it one (stiff=, "
                    f"E=/nu=/thickness=, mat= with thickness=, or section=), "
                    f"or drop {names}."
                )

    def _resolve_standalone_context(
        self,
    ) -> tuple[tuple[int, ...], tuple[int, ...], tuple[tuple[int, ...], ...]]:
        """Fallback resolution when ``_emit`` is called outside the bridge build pipeline."""
        frames_out: list[int] = []
        if self.frames is not None:
            for f in self.frames:
                if isinstance(f, int) and not isinstance(f, bool):
                    frames_out.append(int(f))
                else:
                    raise RuntimeError(
                        "LadrunoMembrane: frames= containing PG/label names or Element "
                        "specs requires emitting through ops.build().emit(...)."
                    )
        supp_out: list[int] = []
        if self.support_nodes is not None:
            for sn in self.support_nodes:
                if isinstance(sn, int) and not isinstance(sn, bool):
                    supp_out.append(int(sn))
                elif isinstance(sn, Node):
                    supp_out.append(_resolve_node_ref("LadrunoMembrane", "support_nodes", sn))
                elif isinstance(sn, NodeSet):
                    supp_out.extend(int(t) for t in sn.tags)
                else:
                    raise RuntimeError(
                        "LadrunoMembrane: support_nodes= containing PG/label names "
                        "requires emitting through ops.build().emit(...)."
                    )
        holes_out: list[tuple[int, ...]] = []
        if self.holes is not None:
            for h_idx, h in enumerate(self.holes):
                if isinstance(h, str):
                    raise RuntimeError(
                        "LadrunoMembrane: holes= containing PG/label names "
                        "requires emitting through ops.build().emit(...)."
                    )
                holes_out.append(
                    tuple(
                        _resolve_node_ref("LadrunoMembrane", f"holes[{h_idx}][{i}]", n)
                        for i, n in enumerate(h)
                    )
                )
        return tuple(frames_out), tuple(supp_out), tuple(holes_out)

    def _resolve_standalone_diaphragm_nodes(self) -> tuple[int, ...]:
        """``diaphragm_nodes=`` as tags outside the bridge build pipeline."""
        out: list[int] = []
        for dn in self.diaphragm_nodes or ():
            if isinstance(dn, str):
                raise RuntimeError(
                    "LadrunoMembrane: diaphragm_nodes= containing PG/label "
                    "names requires emitting through ops.build().emit(...)."
                )
            out.append(_resolve_node_ref("LadrunoMembrane", "diaphragm_nodes", dn))
        return tuple(out)

    def _emit(self, emitter: "Emitter", tag: int) -> None:
        if getattr(emitter, ATTR_ELEMENT_NODES, None) is not None:
            nodes = current_element_nodes(emitter)
        elif self.nodes is not None:
            nodes = tuple(
                _resolve_node_ref("LadrunoMembrane", f"nodes[{i}]", n)
                for i, n in enumerate(self.nodes)
            )
        else:
            nodes = current_element_nodes(emitter)

        if len(nodes) < 3:
            raise ValueError(
                f"LadrunoMembrane: expected at least 3 node tags, got {len(nodes)}."
            )

        ctx = current_membrane_context(emitter)
        if ctx is not None:
            frames_tags, support_node_tags, hole_loops, diaphragm_tags = ctx
        else:
            frames_tags, support_node_tags, hole_loops = self._resolve_standalone_context()
            diaphragm_tags = self._resolve_standalone_diaphragm_nodes()

        args: list[int | float | str] = ["-nodes", *(int(n) for n in nodes)]
        if frames_tags:
            args.extend(["-frames", *(int(f) for f in frames_tags)])
        if support_node_tags:
            args.extend(["-supportNodes", *(int(sn) for sn in support_node_tags)])
        for hole_loop in hole_loops:
            args.extend(["-hole", *(int(h) for h in hole_loop)])
        if diaphragm_tags:
            args.extend(["-diaphragmNodes", *(int(d) for d in diaphragm_tags)])
        args.extend(["-method", self.method])
        if self.span_axis is not None:
            args.extend(["-spanAxis", *(float(c) for c in self.span_axis)])
        if self.ortho_ratio is not None:
            args.extend(["-orthoRatio", float(self.ortho_ratio)])
        if self.pressure is not None:
            args.extend(["-pressure", float(self.pressure)])
        if self.thickness is not None and self.mat is not None:
            mat_tag = resolve_tag(emitter, self.mat)
            args.extend(["-thick", float(self.thickness), "-mat", mat_tag])
        elif self.stiff is not None:
            args.extend(["-stiff", *(float(c) for c in self.stiff)])
        elif isinstance(self.section, ElasticMembranePlateSection):
            args.extend([
                "-stiff",
                float(self.section.E),
                float(self.section.nu),
                float(self.section.h),
            ])
        for f_name in ("f11", "f22", "f12"):
            f_val = getattr(self, f_name)
            if f_val is not None:
                args.extend([f"-{f_name}", float(f_val)])
        eff_rho = self.rho
        if (
            eff_rho is None
            and isinstance(self.section, ElasticMembranePlateSection)
            and float(self.section.rho) > 0.0
        ):
            eff_rho = float(self.section.rho)
        if eff_rho is not None:
            args.extend(["-rho", float(eff_rho)])
        if self.lumped:
            args.append("-lumped")
        if self.mesh_size is not None:
            args.extend(["-meshSize", float(self.mesh_size)])
        args.extend(damp_args(emitter, self.damp))

        # Clear element-nodes / fem_eid context before emitter.element so
        # H5Emitter archives the entire flag-prefixed command ("-nodes", ...)
        # in args/args_str with connectivity=(), preserving round-trip fidelity.
        clear_element_emit_context(emitter)
        emitter.element("LadrunoMembrane", tag, *args)

    def dependencies(self) -> tuple[Primitive, ...]:
        deps: list[Primitive] = []
        if self.section is not None:
            deps.append(self.section)
        if self.mat is not None:
            deps.append(self.mat)
        if self.damp is not None:
            deps.append(self.damp)
        if self.frames is not None:
            for f in self.frames:
                if isinstance(f, Element):
                    deps.append(f)
        return tuple(deps)
