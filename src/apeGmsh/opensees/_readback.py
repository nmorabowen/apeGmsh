"""Instant read on the live route (ADR 0115 D1-D2).

After a live :meth:`~apeGmsh.opensees.apesees.apeSees.analyze`, the
bridge keeps a :class:`ReadSession`: the live emitter that solved the
model, the model's ``ndm``, the element tag plan the emit read, and the
domain epoch (:func:`~apeGmsh.opensees.emitter.live.domain_epoch`) at the
moment the solve returned. ``Node.read`` / ``NodeSet.read`` and
``ops.elements.get|one(...).read`` answer from it:

- reads go through ``LiveOpsEmitter.ops`` (a ``SIDE_CHANNELS["live"]``
  name), never through a verb or ``command()`` (D1);
- a node component maps to a DOF through the bridge's own DOF layout
  (``_load_dof_layout``), per node, from the node's live ``ndf`` (D2);
- an element read looks the FEM id up in the tag plan the session
  captured, so a read never allocates a tag and never builds
  (Invariant 2), and a default build and an ``element_tags="fem"`` build
  read the same values (Invariant 3);
- a session another rebuild of the domain has overtaken raises, naming
  the cause (Invariant 4); a component the node lacks raises, naming the
  node and its ``ndf`` (Invariant 5), and so does a rotational component
  on a node whose rotational DOF slot no element establishes (a u-p
  node's DOF 3 is pore pressure; see :meth:`ReadSession.rotation_refusal`).

The staleness guard sees bridge-driven rebuilds only (every
``LiveOpsEmitter``): a raw ``get_ops().wipe()`` followed by a raw rebuild
is outside it. A raw wipe alone makes a node read raise (the node is no
longer in the domain) and an element read raise (no response).

Every refusal that does not depend on the live domain (an unknown
component, an empty or unknown selector, a stale or missing session, an
element id no element was emitted at) raises before any call into
openseespy.

All imports from the build module live here (S2-a moves it into a
package; this is the one place to repoint).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Iterable, Iterator, Sequence, Union

import numpy as np

from ._internal.build.errors import BridgeError
from ._internal.build.ndf import _load_dof_layout
from ._internal.tag_resolution import MISSING_FEM_ELEMENT_ID

from .element.absorbing import ASDAbsorbingBoundary2D, ASDAbsorbingBoundary3D
from .element.beam_column import (
    ElasticTimoshenkoBeam,
    LadrunoDispBeamColumn,
    LadrunoIMKBeam,
    dispBeamColumn,
    elasticBeamColumn,
    forceBeamColumn,
)
from .element.shell import (
    ASDShellQ4,
    ASDShellT3,
    ShellDKGQ,
    ShellMITC3,
    ShellMITC4,
)
from .element.solid import (
    BezierTet10,
    LadrunoUP,
    BezierTri6,
    FourNodeQuad,
    FourNodeTetrahedron,
    LadrunoBrick,
    LadrunoBrick20,
    LadrunoCST,
    LadrunoLST,
    LadrunoQuad,
    SixNodeTri,
    TenNodeTetrahedron,
    Tri31,
    stdBrick,
)
from .element.truss import CorotTruss, InertiaTruss, Truss
from .element.two_node_link import TwoNodeLink
from .element.zero_length import (
    CoupledZeroLength,
    ZeroLength,
    ZeroLengthSection,
)

# ---------------------------------------------------------------------------
# Rotation roles: the one table ``.read``, the ``recorder.declare`` emit
# (``_internal/build/recorders.py``) and ``ops.results``
# (``_results_route.py``) consult before calling a DOF slot a rotation.
# ---------------------------------------------------------------------------

#: Element families that establish a rotation in the DOF slots
#: ``_load_dof_layout`` calls rotations: beam-columns, shells, and the
#: zero-length family and ``twoNodeLink`` (a rotational spring, a fiber
#: section at a base, or a link's rotational ``-dir`` 3 in 2-D / 4-6 in 3-D
#: acts on those slots).
_ROTATIONAL_FAMILIES: tuple[type, ...] = (
    elasticBeamColumn, forceBeamColumn, dispBeamColumn,
    ElasticTimoshenkoBeam, LadrunoDispBeamColumn, LadrunoIMKBeam,
    ShellMITC4, ShellMITC3, ShellDKGQ, ASDShellQ4, ASDShellT3,
    ZeroLength, ZeroLengthSection, CoupledZeroLength, TwoNodeLink,
)

#: Families that use the translational DOFs only, so they neither give
#: nor take away a meaning for a rotational slot: trusses (a bracing truss
#: on a frame node; ``InertiaTruss`` is a mass-only truss) and the
#: displacement-only continua (a shell on a solid).
_TRANSLATIONAL_ONLY: tuple[type, ...] = (
    Truss, CorotTruss, InertiaTruss,
    FourNodeQuad, Tri31, SixNodeTri, BezierTri6, LadrunoQuad, LadrunoCST,
    LadrunoLST, FourNodeTetrahedron, TenNodeTetrahedron, BezierTet10,
    stdBrick, LadrunoBrick, LadrunoBrick20,
)

# Every other class BLOCKS a rotational slot on the nodes it touches: a u-p
# element (``LadrunoUP``) carries pore pressure at DOF ``ndm + 1``, and a
# class this table does not name is not guessed.

#: u-p families: their pressure DOF sits in a slot an element-less node's
#: layout calls a rotation, so a constraint could tie the two. A model
#: holding one refuses rotations on element-less nodes.
# Any element class added later that carries a pressure DOF must join this tuple.
_UP_FAMILIES: tuple[type, ...] = (LadrunoUP,)


@dataclass(frozen=True)
class RotationRoles:
    """What the element plan says about the rotational DOF slots of a set
    of nodes (:func:`rotation_roles`)."""

    #: Nodes at least one :data:`_ROTATIONAL_FAMILIES` element touches.
    rotational: frozenset[int]
    #: Node -> the first blocking class that touches it.
    blocked: "dict[int, str]"
    #: Nodes any element touches.
    touched: frozenset[int]
    #: A u-p class anywhere in the model (:data:`_UP_FAMILIES`), or ``None``.
    #: On the token route it may instead name an element token the
    #: table does not know (:attr:`model_blocker_unknown`).
    model_blocker: "str | None"
    #: ``model_blocker`` is an unknown element token, not a u-p class.
    model_blocker_unknown: bool = False

    def refusal(self, tag: int) -> "str | None":
        """Why node ``tag``'s rotational DOF slots carry no rotation, or
        ``None`` when they do. A blocking class wins over a rotational one
        on the same node. An element-less node (a decoupled or diaphragm
        master) carries a rotation only in a model with no u-p class:
        a constraint could tie its slot to a pore-pressure DOF."""
        if tag in self.blocked:
            return (
                f"a {self.blocked[tag]} touches it, whose DOF layout is not "
                "a beam/shell rotation (a u-p element carries pore pressure "
                "there; any other family, or one this table does not know, "
                "is left undefined)"
            )
        if tag in self.rotational:
            return None
        if tag in self.touched:
            return (
                "no beam-column or shell (or zero-length) element touches it, so "
                "no element establishes a rotation in that DOF slot"
            )
        if self.model_blocker is not None and self.model_blocker_unknown:
            return (
                "no element touches it, and the model holds an element "
                f"token this table does not know ({self.model_blocker}), so "
                "what a constraint could tie that DOF slot to is unknown"
            )
        if self.model_blocker is not None:
            return (
                "no element touches it, and the model holds a "
                f"{self.model_blocker}, so a constraint could tie that DOF "
                "slot to a pore pressure"
            )
        return None


def rotation_roles(
    element_plan: "Iterable[tuple[Element, ElementPlanRows]]",
    targets: "Iterable[int] | None" = None,
) -> RotationRoles:
    """The :class:`RotationRoles` of ``targets`` (every node when ``None``)
    under ``element_plan`` (each element spec with its rows)."""
    rows: list[tuple["type | None", str, np.ndarray]] = []
    for spec, sub in element_plan:
        if len(sub) == 0:
            continue
        conn = sub.conn
        if conn.ndim == 2:
            flat = np.unique(np.asarray(conn, dtype=np.int64))
        else:
            flat = np.unique(np.fromiter(
                (int(n) for row in conn for n in row), dtype=np.int64))
        rows.append((type(spec), type(spec).__name__, flat))
    return _roles_from_rows(rows, targets)


def _roles_from_rows(
    rows: "Iterable[tuple[type | None, str, np.ndarray]]",
    targets: "Iterable[int] | None",
) -> RotationRoles:
    """The one classification behind :func:`rotation_roles` and
    :func:`rotation_roles_from_tokens`: each row is an element class (or
    ``None`` for one this table cannot name), the name a refusal shows,
    and the nodes the class's elements touch. ``None`` blocks, and counts
    as a model blocker like a u-p class: what it carries is not known."""
    want = (None if targets is None
            else np.unique(np.asarray(list(targets), dtype=np.int64)))
    rotational: set[int] = set()
    touched: set[int] = set()
    blocked: dict[int, str] = {}
    model_blocker: "str | None" = None
    unknown = False
    for cls, name, flat in rows:
        is_rot = cls is not None and issubclass(cls, _ROTATIONAL_FAMILIES)
        is_trans = cls is not None and issubclass(cls, _TRANSLATIONAL_ONLY)
        if model_blocker is None and (
                cls is None or issubclass(cls, _UP_FAMILIES)):
            model_blocker, unknown = name, cls is None
        hit = flat if want is None else np.intersect1d(flat, want)
        nodes = [int(n) for n in hit]
        touched.update(nodes)
        if is_rot:
            rotational.update(nodes)
        elif not is_trans:
            for n in nodes:
                blocked.setdefault(n, name)
    return RotationRoles(
        rotational=frozenset(rotational), blocked=blocked,
        touched=frozenset(touched), model_blocker=model_blocker,
        model_blocker_unknown=unknown,
    )


#: OpenSees ``element`` token -> the bridge class that emits it, for every
#: ``Element`` class with an ``_emit``, in the family tuples or not (an
#: absorbing boundary blocks its own nodes, as on the declared path, but is
#: no model blocker). Each class's ``_emit`` writes the token;
#: ``tests/test_results_recorder_emit.py`` reads it back from the source
#: and checks no ``Element`` subclass is missing. A rehydrated model keeps
#: only the token (``ElementRecord.type_token``). A token not here is an
#: unknown class: it blocks the rotational slots of the nodes it touches,
#: and of element-less nodes.
_CLASS_FOR_TOKEN: "dict[str, type]" = {
    "elasticBeamColumn": elasticBeamColumn,
    "forceBeamColumn": forceBeamColumn,
    "dispBeamColumn": dispBeamColumn,
    "ElasticTimoshenkoBeam": ElasticTimoshenkoBeam,
    "LadrunoDispBeamColumn": LadrunoDispBeamColumn,
    "LadrunoIMKBeam": LadrunoIMKBeam,
    "ShellMITC4": ShellMITC4,
    "ShellMITC3": ShellMITC3,
    "ShellDKGQ": ShellDKGQ,
    "ASDShellQ4": ASDShellQ4,
    "ASDShellT3": ASDShellT3,
    "zeroLength": ZeroLength,
    "zeroLengthSection": ZeroLengthSection,
    "CoupledZeroLength": CoupledZeroLength,
    "twoNodeLink": TwoNodeLink,
    "Truss": Truss,
    "CorotTruss": CorotTruss,
    "InertiaTruss": InertiaTruss,
    "quad": FourNodeQuad,
    "tri31": Tri31,
    "tri6n": SixNodeTri,
    "BezierTri6": BezierTri6,
    "LadrunoQuad": LadrunoQuad,
    "LadrunoCST": LadrunoCST,
    "LadrunoLST": LadrunoLST,
    "FourNodeTetrahedron": FourNodeTetrahedron,
    "TenNodeTetrahedron": TenNodeTetrahedron,
    "BezierTet10": BezierTet10,
    "stdBrick": stdBrick,
    "LadrunoBrick": LadrunoBrick,
    "LadrunoBrick20": LadrunoBrick20,
    "LadrunoUP": LadrunoUP,
    "ASDAbsorbingBoundary2D": ASDAbsorbingBoundary2D,
    "ASDAbsorbingBoundary3D": ASDAbsorbingBoundary3D,
}


def rotation_roles_from_tokens(
    rows: "Iterable[tuple[str, Iterable[int]]]",
    targets: "Iterable[int] | None" = None,
) -> RotationRoles:
    """:func:`rotation_roles` for a model that keeps only element tokens.

    ``rows`` are ``(type_token, node tags)``, one per element, as a
    rehydrated :class:`~apeGmsh.opensees.OpenSeesModel` holds them. Each
    token maps to its class through :data:`_CLASS_FOR_TOKEN` and the same
    family tables; an unknown token blocks the rotational slots of its
    nodes and, like a u-p class, of element-less nodes.
    """
    by_token: dict[str, list[int]] = {}
    for token, nodes in rows:
        by_token.setdefault(str(token), []).extend(int(n) for n in nodes)
    return _roles_from_rows(
        ((_CLASS_FOR_TOKEN.get(tok), tok,
          np.unique(np.asarray(nodes, dtype=np.int64)))
         for tok, nodes in by_token.items()),
        targets,
    )

if TYPE_CHECKING:
    from ._internal.build.pg_expand import ElementPlanRows
    from ._internal.types import Element
    from .apesees import apeSees
    from .emitter.live import LiveOpsEmitter


__all__ = [
    "DroppedSession",
    "ElementHandle",
    "ElementSet",
    "NODE_COMPONENTS",
    "ELEMENT_COMPONENTS",
    "ReadSession",
    "ReadState",
    "_ElementAccessor",
    "read_nodes",
    "read_elements",
]


# ---------------------------------------------------------------------------
# Component vocabulary (the results package's canonical names)
# ---------------------------------------------------------------------------

#: Canonical prefix -> (openseespy query, spatial offset). The spatial
#: index addresses ``(x, y, z, rx, ry, rz)``, the order
#: ``_load_dof_layout`` places at each DOF.
_NODE_PREFIXES: dict[str, tuple[str, int]] = {
    "displacement":         ("nodeDisp", 0),
    "rotation":             ("nodeDisp", 3),
    "velocity":             ("nodeVel", 0),
    "angular_velocity":     ("nodeVel", 3),
    "acceleration":         ("nodeAccel", 0),
    "angular_acceleration": ("nodeAccel", 3),
    "reaction_force":       ("nodeReaction", 0),
    "reaction_moment":      ("nodeReaction", 3),
}
_AXES: dict[str, int] = {"x": 0, "y": 1, "z": 2}

#: Every node component ``.read`` answers.
NODE_COMPONENTS: tuple[str, ...] = tuple(
    f"{p}_{a}" for p in _NODE_PREFIXES for a in _AXES)

#: Element component -> the ``eleResponse`` token that answers it with one
#: value per element (Q7). R2 serves the truss axial force only.
_ELEMENT_TOKENS: dict[str, str] = {"axial_force": "axialForce"}

#: Every element component ``.read`` answers.
ELEMENT_COMPONENTS: tuple[str, ...] = tuple(_ELEMENT_TOKENS)

#: The DOF counts ``_load_dof_layout`` names exactly; any other ``ndf``
#: falls back to a legacy guess there, so a read refuses it.
_LAID_OUT_NDF: frozenset[int] = frozenset({1, 2, 3, 6})

_RESULTS_HINT = (
    "Record it (ops.recorder.MPCO / ops.recorder.declare) and read it with "
    "apeGmsh's Results (Results.from_mpco / Results.from_recorders)."
)


def _node_component(component: str, where: str) -> tuple[str, int]:
    """``(query, spatial index)`` of a canonical node component."""
    if isinstance(component, str) and "_" in component:
        prefix, axis = component.rsplit("_", 1)
        entry = _NODE_PREFIXES.get(prefix)
        if entry is not None and axis in _AXES:
            query, offset = entry
            return query, offset + _AXES[axis]
    raise BridgeError(
        f"{where}: unknown node component {component!r}. .read answers "
        f"{', '.join(NODE_COMPONENTS)}."
    )


def _element_token(component: str, where: str) -> str:
    token = _ELEMENT_TOKENS.get(component) if isinstance(
        component, str) else None
    if token is None:
        raise BridgeError(
            f"{where}: .read answers only per-element scalars "
            f"({', '.join(ELEMENT_COMPONENTS)}, on Truss and CorotTruss); "
            f"{component!r} is not one. {_RESULTS_HINT}"
        )
    return token


# ---------------------------------------------------------------------------
# The read session
# ---------------------------------------------------------------------------


class ReadSession:
    """What a live ``analyze`` leaves for ``.read`` (ADR 0115 D2).

    ``element_plan`` is the element tag plan the live emit read (each
    element spec with its columnar rows); the FEM-id index over it is
    built on the first element read and cached here.
    """

    __slots__ = ("emitter", "ndm", "epoch", "element_plan", "_index",
                 "_rot_roles")

    def __init__(
        self,
        *,
        emitter: "LiveOpsEmitter",
        ndm: int,
        epoch: int,
        element_plan: "tuple[tuple[Element, ElementPlanRows], ...]",
    ) -> None:
        self.emitter = emitter
        self.ndm = int(ndm)
        self.epoch = int(epoch)
        self.element_plan = element_plan
        self._index: "tuple[np.ndarray, np.ndarray, np.ndarray] | None" = None
        self._rot_roles: "RotationRoles | None" = None

    def rotation_refusal(self, tag: int) -> "str | None":
        """Why node ``tag``'s rotational DOF slots have no established
        meaning, or ``None`` when they do (:meth:`RotationRoles.refusal`,
        the rule the ``recorder.declare`` emit and ``ops.results`` apply
        too)."""
        if self._rot_roles is None:
            self._rot_roles = rotation_roles(self.element_plan)
        return self._rot_roles.refusal(int(tag))

    def element_rows(
        self, ids: Sequence[int], where: str,
    ) -> "list[tuple[Element, int]]":
        """``(spec, OpenSees tag)`` of each FEM element id, from the plan.

        Raises when an id has no planned element (no declaration fans
        out over it) or more than one (two declarations fan out over it,
        so the id names no single element).
        """
        sorted_eids, sorted_tags, sorted_spec = self._element_index()
        want = np.asarray([int(e) for e in ids], dtype=np.int64)
        lo = np.searchsorted(sorted_eids, want, side="left")
        hi = np.searchsorted(sorted_eids, want, side="right")
        missing = [int(e) for e, a, b in zip(want, lo, hi) if a == b]
        if missing:
            raise BridgeError(
                f"{where}: no element was emitted at FEM element id(s) "
                f"{missing[:10]}{' ...' if len(missing) > 10 else ''}; no "
                "ops.element declaration covers them."
            )
        twice = [int(e) for e, a, b in zip(want, lo, hi) if b - a > 1]
        if twice:
            raise BridgeError(
                f"{where}: FEM element id(s) {twice[:10]} are fanned out by "
                "more than one ops.element declaration, so an id names no "
                "single OpenSees element. Read them through Results."
            )
        return [
            (self.element_plan[int(sorted_spec[a])][0], int(sorted_tags[a]))
            for a in lo
        ]

    def _element_index(self) -> "tuple[np.ndarray, np.ndarray, np.ndarray]":
        if self._index is None:
            eids: list[np.ndarray] = []
            tags: list[np.ndarray] = []
            specs: list[np.ndarray] = []
            for k, (_spec, sub) in enumerate(self.element_plan):
                n = len(sub)
                if n == 0:
                    continue
                e = np.asarray(sub.eids, dtype=np.int64)
                t = (sub.tag_start + np.arange(n, dtype=np.int64)
                     if sub.tags is None
                     else np.asarray(sub.tags, dtype=np.int64))
                keep = e != MISSING_FEM_ELEMENT_ID
                eids.append(e[keep])
                tags.append(t[keep])
                specs.append(np.full(int(keep.sum()), k, dtype=np.int64))
            all_e = np.concatenate(eids) if eids else np.empty(0, np.int64)
            all_t = np.concatenate(tags) if tags else np.empty(0, np.int64)
            all_s = np.concatenate(specs) if specs else np.empty(0, np.int64)
            order = np.argsort(all_e, kind="stable")
            self._index = (all_e[order], all_t[order], all_s[order])
        return self._index


@dataclass(frozen=True, slots=True)
class DroppedSession:
    """A read session the bridge closed itself, and why (Invariant 4)."""

    cause: str


#: What ``apeSees._read_state`` holds: ``None`` before any live analyze.
ReadState = Union[ReadSession, DroppedSession, None]

#: The cause a deck run leaves: its results are not in this process.
DECK_RUN_CAUSE = (
    "a deck run ({verb}) solved the model out of process, so "
    "the in-process domain does not hold its results; read them with "
    "ops.results(path) (ADR 0115 D3; it lands with slice R2-c)"
)


def open_session(
    emitter: "LiveOpsEmitter",
    *,
    ndm: int,
    element_plan: "tuple[tuple[Element, ElementPlanRows], ...]",
) -> ReadSession:
    """A session over the domain ``emitter`` just solved, at the current
    epoch."""
    from .emitter.live import domain_epoch

    return ReadSession(
        emitter=emitter, ndm=ndm, epoch=domain_epoch(),
        element_plan=element_plan,
    )


def live_session(bridge: "apeSees", where: str) -> ReadSession:
    """The bridge's open read session; raises, naming why, when there is
    none or it no longer describes the domain."""
    from .emitter.live import domain_epoch

    state = bridge._read_state
    if state is None:
        raise BridgeError(
            f"{where}: no live analyze has run on this bridge. .read answers "
            "on the live route only, after ops.analyze(...). A deck run "
            "(ops.tcl / ops.py with run=True) is read with ops.results(path) "
            "(ADR 0115 D3; it lands with slice R2-c)."
        )
    if isinstance(state, DroppedSession):
        raise BridgeError(
            f"{where}: the read session of this bridge's last analyze is "
            f"closed: {state.cause}."
        )
    now = domain_epoch()
    if state.epoch != now:
        raise BridgeError(
            f"{where}: the in-process OpenSees domain was rebuilt after this "
            "bridge's analyze (by another apeSees bridge, or a live procedure "
            "such as modal_properties, frf, an explicit run or ops.run); it "
            "no longer holds this bridge's solved state. Call "
            "ops.analyze(...) again before reading."
        )
    return state


# ---------------------------------------------------------------------------
# Node reads
# ---------------------------------------------------------------------------


def _node_ndf(ops: object, tag: int, where: str) -> int:
    try:
        raw = ops.getNDF(int(tag))  # type: ignore[attr-defined]
    except Exception as e:  # the node is not in the domain
        raise BridgeError(
            f"{where}: node {tag} is not in the live domain ({e})."
        ) from e
    if isinstance(raw, (list, tuple)):
        return int(raw[0])
    return int(raw)


def read_nodes(
    bridge: "apeSees", tags: Sequence[int], component: str, *, where: str,
) -> np.ndarray:
    """``component`` at each node of ``tags``, in order, in model units at
    the last committed state."""
    query, spatial = _node_component(component, where)
    session = live_session(bridge, where)
    ops = session.emitter.ops
    dofs: list[int] = []
    for tag in tags:
        ndf = _node_ndf(ops, tag, where)
        if ndf not in _LAID_OUT_NDF:
            raise BridgeError(
                f"{where}: node {tag} has ndf={ndf}; the bridge's DOF layout "
                f"names components only for ndf in {sorted(_LAID_OUT_NDF)}. "
                f"{_RESULTS_HINT}"
            )
        layout = _load_dof_layout(ndf, session.ndm)
        if spatial not in layout:
            raise BridgeError(
                f"{where}: node {tag} has no {component!r}: it has ndf={ndf} "
                f"(ndm={session.ndm}), whose DOFs carry "
                f"{_layout_names(layout, query)}."
            )
        if spatial >= 3:
            why = session.rotation_refusal(int(tag))
            if why is not None:
                raise BridgeError(
                    f"{where}: node {tag} (ndf={ndf}, ndm={session.ndm}) has "
                    f"no established {component!r}: DOF "
                    f"{layout.index(spatial) + 1} is a rotation only on a "
                    f"beam-column, shell or zero-length node, and {why}. {_RESULTS_HINT}"
                )
        dofs.append(layout.index(spatial) + 1)
    if query == "nodeReaction":
        ops.reactions()
    fetch = getattr(ops, query)
    return np.asarray(
        [float(fetch(int(t), d)) for t, d in zip(tags, dofs)],
        dtype=np.float64,
    )


def _layout_names(layout: tuple[int, ...], query: str) -> str:
    trans, rot = {
        "nodeDisp": ("displacement", "rotation"),
        "nodeVel": ("velocity", "angular_velocity"),
        "nodeAccel": ("acceleration", "angular_acceleration"),
        "nodeReaction": ("reaction_force", "reaction_moment"),
    }[query]
    axes = "xyz"
    return ", ".join(
        f"{trans}_{axes[i]}" if i < 3 else f"{rot}_{axes[i - 3]}"
        for i in layout
    )


# ---------------------------------------------------------------------------
# Element reads
# ---------------------------------------------------------------------------


def read_elements(
    bridge: "apeSees", ids: Sequence[int], component: str, *, where: str,
) -> np.ndarray:
    """``component`` of each FEM element of ``ids``, in order."""
    token = _element_token(component, where)
    session = live_session(bridge, where)
    rows = session.element_rows(ids, where)
    for eid, (spec, _tag) in zip(ids, rows):
        if not isinstance(spec, (Truss, CorotTruss)):
            raise BridgeError(
                f"{where}: element {int(eid)} is a {type(spec).__name__}; "
                f"{component!r} is read only on Truss and CorotTruss. "
                f"{_RESULTS_HINT}"
            )
    ops = session.emitter.ops
    out: list[float] = []
    for eid, (_spec, tag) in zip(ids, rows):
        got = ops.eleResponse(tag, token)
        if not got:
            raise BridgeError(
                f"{where}: element {int(eid)} (OpenSees tag {tag}) answered no "
                f"{token!r}: the live domain no longer holds it. A raw "
                "get_ops().wipe() (or a rebuild outside the bridge) is not "
                "seen by the read session; call ops.analyze(...) again."
            )
        out.append(float(got[0]))
    return np.asarray(out, dtype=np.float64)


class ElementHandle:
    """One FEM element of a bridge, addressed by its FEM id (ADR 0115 D2).

    Returned by ``ops.elements.one(...)``. ``id`` is the FEM element id,
    whatever OpenSees tag the build gave it (Invariant 3).
    """

    __slots__ = ("id", "_bridge")

    def __init__(self, *, eid: int, bridge: "apeSees") -> None:
        self.id = int(eid)
        self._bridge = bridge

    def read(self, component: str) -> float:
        """``component`` of this element at the last committed state.

        Serves the per-element scalars of :data:`ELEMENT_COMPONENTS`
        (``axial_force`` on Truss and CorotTruss); anything else raises
        and points to ``Results``. Raises :class:`BridgeError` when no
        live analyze has run on the bridge or its session is stale.
        """
        return float(read_elements(
            self._bridge, (self.id,), component,
            where=f"ops.elements element {self.id}.read")[0])

    def __int__(self) -> int:
        return self.id

    def __repr__(self) -> str:
        return f"ElementHandle(id={self.id})"


class ElementSet:
    """FEM elements of a bridge, in selection order (ADR 0115 D2)."""

    __slots__ = ("_ids", "_bridge")

    def __init__(self, *, ids: tuple[int, ...], bridge: "apeSees") -> None:
        self._ids = tuple(int(e) for e in ids)
        self._bridge = bridge

    @property
    def ids(self) -> tuple[int, ...]:
        """FEM element ids, in the set's order."""
        return self._ids

    def read(self, component: str) -> np.ndarray:
        """``component`` of every element, ``ndarray (E,)`` in :attr:`ids`
        order, at the last committed state.

        Serves the per-element scalars of :data:`ELEMENT_COMPONENTS`
        (``axial_force`` on Truss and CorotTruss); anything else raises
        and points to ``Results``.
        """
        return read_elements(
            self._bridge, self._ids, component, where="ElementSet.read")

    def __iter__(self) -> Iterator[ElementHandle]:
        return (ElementHandle(eid=e, bridge=self._bridge) for e in self._ids)

    def __len__(self) -> int:
        return len(self._ids)

    def __bool__(self) -> bool:
        return bool(self._ids)

    def __getitem__(self, idx: int) -> ElementHandle:
        return ElementHandle(eid=self._ids[idx], bridge=self._bridge)

    def __repr__(self) -> str:
        return f"ElementSet(n={len(self)})"


class _ElementAccessor:
    """Backs ``ops.elements``: FEM elements by ``pg=`` / ``label=`` /
    ``ids=`` (ADR 0115 D2, Q2), mirroring ``ops.nodes``.

    ``ops.element`` (singular) holds the element constructors; this
    accessor only selects and reads.
    """

    __slots__ = ("_bridge",)

    def __init__(self, bridge: "apeSees") -> None:
        self._bridge = bridge

    def get(
        self,
        *,
        pg: str | None = None,
        label: str | Sequence[str] | None = None,
        ids: Sequence[int] | None = None,
    ) -> ElementSet:
        """An :class:`ElementSet` of the FEM elements one selector picks
        (every element with none). Always a set, whatever its size (D5);
        :meth:`one` is the single-element form."""
        return ElementSet(
            ids=self._select(pg, label, ids, where="ops.elements.get",
                             allow_none=True),
            bridge=self._bridge,
        )

    def one(
        self,
        *,
        pg: str | None = None,
        label: str | Sequence[str] | None = None,
        ids: Sequence[int] | None = None,
    ) -> ElementHandle:
        """The single element exactly one selector picks; raises
        :class:`BridgeError`, naming the count, otherwise."""
        found = self._select(pg, label, ids, where="ops.elements.one",
                             allow_none=False)
        if len(found) != 1:
            raise BridgeError(
                f"ops.elements.one selects {len(found)} elements, not exactly "
                f"one (ids {list(found)[:10]}"
                f"{' ...' if len(found) > 10 else ''}). Use "
                "ops.elements.get(...) for a set."
            )
        return ElementHandle(eid=found[0], bridge=self._bridge)

    def __len__(self) -> int:
        return len(self._bridge.fem.elements.ids)

    def __repr__(self) -> str:
        return f"_ElementAccessor(n={len(self)})"

    def _select(
        self,
        pg: str | None,
        label: str | Sequence[str] | None,
        ids: Sequence[int] | None,
        *,
        where: str,
        allow_none: bool,
    ) -> tuple[int, ...]:
        given = [k for k, v in (("pg", pg), ("label", label), ("ids", ids))
                 if v is not None]
        if len(given) > 1 or (not given and not allow_none):
            raise ValueError(
                f"{where}: supply {'at most' if allow_none else 'exactly'} "
                f"one of pg=, label= or ids= (got {', '.join(given) or 'none'})."
            )
        elements = self._bridge.fem.elements
        if pg is not None:
            try:
                found = elements.select(pg=pg).ids
            except (KeyError, ValueError) as e:
                raise KeyError(
                    f"{where}(pg={pg!r}): no such element PG in the FEM "
                    "snapshot."
                ) from e
            return self._snapshot_order(found)
        if label is not None:
            return self._snapshot_order(
                resolve_element_label(self._bridge, label, where=where))
        if ids is not None:
            out = tuple(int(e) for e in ids)
            if not out:
                raise ValueError(f"{where}: ids= is empty.")
            for eid in out:
                elements.index(eid)  # KeyError names the valid range
            return out
        return tuple(int(e) for e in elements.ids)

    def _snapshot_order(self, found: Sequence[int]) -> tuple[int, ...]:
        elements = self._bridge.fem.elements
        return tuple(sorted({int(e) for e in found}, key=elements.index))


def resolve_element_label(
    bridge: "apeSees", label: str | Sequence[str], *, where: str,
) -> tuple[int, ...]:
    """FEM element ids under the geometry label(s) ``label`` (ADR 0115 D4).

    One name or a non-empty list; an unknown name raises, listing the
    labels that exist, and never falls back to a physical group.
    """
    if isinstance(label, str):
        names: tuple[str, ...] = (label,)
    elif isinstance(label, (list, tuple)):
        names = tuple(label)
    else:
        raise ValueError(
            f"{where}: label= must be a string or a list of strings, got "
            f"{type(label).__name__} ({label!r})."
        )
    if not names:
        raise ValueError(f"{where}: label= is empty; name at least one label.")
    bad = [n for n in names if not isinstance(n, str) or not n.strip()]
    if bad:
        raise ValueError(
            f"{where}: every label must be a non-blank string, got {bad!r}."
        )
    elements = bridge.fem.elements
    labels = elements.labels
    missing = [n for n in names if n not in labels]
    if missing:
        raise BridgeError(
            f"{where}: no label {', '.join(repr(n) for n in missing)} on the "
            f"FEM snapshot. Labels that exist: {sorted(set(labels.names()))}. "
            "A label never falls back to a physical group of the same name."
        )
    out: list[int] = []
    seen: set[int] = set()
    for n in names:
        try:
            got = [int(e) for e in labels.element_ids(n)]
        except ValueError:  # a point-only (dim-0) label carries no elements
            got = []
        for eid in got:
            if eid not in seen:
                seen.add(eid)
                out.append(eid)
    if not out:
        raise BridgeError(
            f"{where}: label={label!r} selects 0 elements on the FEM snapshot."
        )
    return tuple(out)
