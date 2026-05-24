"""
Typed ``recorder`` primitives.

Phase 3B ships three concrete recorder classes mirroring the OpenSees
``recorder`` command:

  * :class:`Node`    — ``recorder Node ...``
  * :class:`Element` — ``recorder Element ...``
  * :class:`MPCO`    — ``recorder mpco ...`` (HDF5)

Each class is a ``@dataclass(frozen=True, kw_only=True, slots=True)``;
the matching :class:`apeGmsh.opensees._internal.ns.recorder._RecorderNS`
methods take the same kwargs and call ``self._bridge._register(Cls(...))``.

Recorders never compose other primitives (``dependencies()`` returns
``()``). They are leaves in the dependency graph; the build pipeline
emits them after the topology + analysis chain so that each ``recorder``
command sees fully-allocated node and element tags.

The ``pg=`` form (physical-group fan-out into node/element tags) is
materialized at build time by
:func:`apeGmsh.opensees._internal.build.emit_recorder_spec`, which
resolves ``pg`` through the FEM snapshot, rewrites the spec to its
explicit ``nodes=`` / ``elements=`` form via :func:`dataclasses.replace`,
and then delegates to ``_emit``. End users drive this through
``apeSees(fem).tcl(...) / .py(...) / .run()`` — never call ``_emit``
directly with a ``pg`` spec, which raises :class:`NotImplementedError`
as a defense-in-depth guard.

OpenSees command shapes
-----------------------

::

    recorder Node    -file fname [-time] [-dT dT] [-node n...]
                                 -dof d... response
    recorder Element -file fname [-time] [-dT dT] [-ele e...]
                                 response_tokens...
    recorder mpco    fname.mpco  [-N nodal_responses...]
                                 [-E elem_responses...]
                                 [-T dt $dt | -T nsteps $n]

The ``-time`` flag (when ``time_format="dt"``) instructs OpenSees to
include the simulation-time column in the output file. The default
``time_format="step"`` writes only the response columns.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

from .._vocabulary import (
    DERIVED_SCALARS,
    FIBER,
    LINE_DIAGRAMS,
    MATERIAL_STATE,
    NODAL_FORCES,
    NODAL_KINEMATICS,
    PER_ELEMENT_NODAL_FORCES,
    STRAIN,
    STRESS,
    is_canonical,
)
from ._internal.types import Primitive, Recorder

if TYPE_CHECKING:
    from apeGmsh.mesh.FEMData import FEMData

    from ._internal.tag_allocator import TagAllocator
    from .emitter.base import Emitter


__all__ = [
    # Typed primitives (Phase 3B)
    "Node",
    "Element",
    "MPCO",
    # Unified declaration (Phase 9)
    "RecorderRecord",
    "RecorderDeclaration",
    "ALL_RECORDER_CATEGORIES",
]


# ---------------------------------------------------------------------------
# Node — ``recorder Node ...``
# ---------------------------------------------------------------------------

@dataclass(frozen=True, kw_only=True, slots=True)
class Node(Recorder):
    """``recorder Node`` — record nodal response history.

    OpenSees command::

        recorder Node -file fname [-time] [-dT dT]
                      (-node n1 n2 ... | -nodeRange first last)
                      -dof d1 d2 ... response

    Exactly one of ``nodes=`` (explicit list) or ``pg=`` (physical-group
    label) must be supplied. The bridge build pipeline materializes the
    ``pg=`` form against the FEM snapshot before driving ``_emit``;
    direct ``_emit`` calls on a ``pg=`` spec raise
    :class:`NotImplementedError` as a defense-in-depth guard.

    Parameters
    ----------
    file
        Output file path.
    response
        OpenSees response token (``"disp"``, ``"vel"``, ``"accel"``,
        ``"reaction"``, ``"unbalance"``, ...).
    nodes
        Explicit tuple of node tags. Mutually exclusive with ``pg``.
    pg
        Physical-group label whose nodes the recorder targets.
        Mutually exclusive with ``nodes``. Resolved by the bridge
        build pipeline at emit time.
    dofs
        DOF indices (1-based, OpenSees convention). At least one
        required.
    dT
        Optional cadence — record only every ``dT`` simulation
        seconds. ``None`` records every step.
    time_format
        ``"step"`` (default) writes only response columns;
        ``"dt"`` emits the OpenSees ``-time`` flag, prepending the
        simulation-time column.
    """

    file: str
    response: str
    nodes: tuple[int, ...] | None = None
    pg: str | None = None
    dofs: tuple[int, ...]
    dT: float | None = None
    time_format: str = "step"

    def __post_init__(self) -> None:
        if (self.nodes is None) == (self.pg is None):
            raise ValueError(
                "Node recorder: supply exactly one of nodes= or pg= "
                f"(got nodes={self.nodes!r}, pg={self.pg!r})."
            )
        if not self.dofs:
            raise ValueError(
                "Node recorder: at least one dof required."
            )
        if self.time_format not in ("step", "dt"):
            raise ValueError(
                "Node recorder: time_format must be 'step' or 'dt', "
                f"got {self.time_format!r}."
            )

    def dependencies(self) -> tuple[Primitive, ...]:
        return ()

    def materialize(
        self,
        emitter: "Emitter",
        fem: "FEMData",
        tags: "TagAllocator | None",
        fem_eid_to_ops_tag: "dict[int, int] | None" = None,
    ) -> "Node":
        # ``fem_eid_to_ops_tag`` is accepted for signature parity with
        # element-targeting recorders but ignored here: OpenSees node
        # tags equal FEM node ids (nodes are never rebased through an
        # allocator), so no translation is needed.
        del fem_eid_to_ops_tag
        if self.pg is None:
            return self
        from ._internal.build import expand_pg_to_nodes
        node_ids = expand_pg_to_nodes(fem, self.pg)
        return replace(self, pg=None, nodes=node_ids)

    def _emit(self, emitter: "Emitter", tag: int) -> None:
        args: list[int | float | str] = ["-file", self.file]
        if self.dT is not None:
            args += ["-dT", self.dT]
        if self.time_format == "dt":
            args += ["-time"]
        if self.nodes is not None:
            args += ["-node", *self.nodes]
        else:
            # pg → node fan-out is owned by :meth:`Node.materialize`;
            # the bridge calls it before _emit. Reaching this branch
            # means someone bypassed the bridge.
            raise NotImplementedError(
                "Node recorder pg= must be resolved by the bridge "
                "build pipeline. Drive emission through "
                "apeSees(fem).tcl()/py()/run() instead of calling "
                "_emit directly with a pg= spec."
            )
        args += ["-dof", *self.dofs, self.response]
        emitter.recorder("Node", *args)


# ---------------------------------------------------------------------------
# Element — ``recorder Element ...``
# ---------------------------------------------------------------------------

@dataclass(frozen=True, kw_only=True, slots=True)
class Element(Recorder):
    """``recorder Element`` — record element-level response history.

    OpenSees command::

        recorder Element -file fname [-time] [-dT dT]
                         (-ele e1 e2 ... | -eleRange first last)
                         response_tokens...

    ``response`` is a tuple of OpenSees response tokens — the simplest
    case is ``("globalForce",)`` or ``("stresses",)``; element types
    that nest responses (e.g. fiber sections) take multi-token forms
    such as ``("section", "1", "force")``.

    Exactly one of ``elements=`` (explicit list) or ``pg=`` (physical-
    group label) must be supplied. ``pg=`` is resolved against the FEM
    snapshot by the bridge build pipeline before driving ``_emit``;
    direct ``_emit`` calls on a ``pg=`` spec raise
    :class:`NotImplementedError` as a defense-in-depth guard.

    Parameters
    ----------
    file
        Output file path.
    response
        Tuple of OpenSees response tokens (at least one).
    elements
        Explicit tuple of element tags. Mutually exclusive with ``pg``.
    pg
        Physical-group label whose elements the recorder targets.
        Mutually exclusive with ``elements``. Resolved by the bridge
        build pipeline at emit time.
    dT
        Optional cadence — record only every ``dT`` simulation
        seconds. ``None`` records every step.
    time_format
        ``"step"`` (default) writes only response columns;
        ``"dt"`` emits the OpenSees ``-time`` flag.
    """

    file: str
    response: tuple[str, ...]
    elements: tuple[int, ...] | None = None
    pg: str | None = None
    dT: float | None = None
    time_format: str = "step"

    def __post_init__(self) -> None:
        if (self.elements is None) == (self.pg is None):
            raise ValueError(
                "Element recorder: supply exactly one of elements= or "
                f"pg= (got elements={self.elements!r}, pg={self.pg!r})."
            )
        if not self.response:
            raise ValueError(
                "Element recorder: response token required."
            )
        if self.time_format not in ("step", "dt"):
            raise ValueError(
                "Element recorder: time_format must be 'step' or "
                f"'dt', got {self.time_format!r}."
            )

    def dependencies(self) -> tuple[Primitive, ...]:
        return ()

    def materialize(
        self,
        emitter: "Emitter",
        fem: "FEMData",
        tags: "TagAllocator | None",
        fem_eid_to_ops_tag: "dict[int, int] | None" = None,
    ) -> "Element":
        if self.pg is None:
            return self
        from ._internal.build import BridgeError, expand_pg_to_elements
        # ``expand_pg_to_elements`` yields FEM-side element ids; the
        # OpenSees element tags assigned by the bridge's fan-out differ
        # whenever an element primitive consumed an allocator slot in
        # ``_register`` (one ``ops.element.X(pg="Rock")`` declaration
        # → element-kind tag 1 for the spec, fan-out instance → tag 2).
        # Translate through the bridge-built ``{fem_eid: ops_tag}`` map
        # so the recorder's ``-ele ...`` list references the emitted
        # OpenSees tags, not the raw FEM eids.
        fem_eids = tuple(
            eid for eid, _conn in expand_pg_to_elements(fem, self.pg)
        )
        if fem_eid_to_ops_tag is None:
            # Legacy direct callers (unit tests of materialize() with no
            # bridge) get the raw FEM eids; the bridge always supplies
            # the map on the recorder emit pass.
            return replace(self, pg=None, elements=fem_eids)
        ops_tags: list[int] = []
        for eid in fem_eids:
            ops_tag = fem_eid_to_ops_tag.get(int(eid))
            if ops_tag is None:
                raise BridgeError(
                    f"Element recorder pg={self.pg!r} resolves to FEM "
                    f"eid {eid} but no element was emitted at that "
                    "eid — declare an ``ops.element.X(pg=...)`` "
                    f"primitive whose pg includes {self.pg!r}."
                )
            ops_tags.append(int(ops_tag))
        return replace(self, pg=None, elements=tuple(ops_tags))

    def _emit(self, emitter: "Emitter", tag: int) -> None:
        args: list[int | float | str] = ["-file", self.file]
        if self.dT is not None:
            args += ["-dT", self.dT]
        if self.time_format == "dt":
            args += ["-time"]
        if self.elements is not None:
            args += ["-ele", *self.elements]
        else:
            # pg → element fan-out is owned by :meth:`Element.materialize`;
            # the bridge calls it before _emit. Reaching this branch
            # means someone bypassed the bridge.
            raise NotImplementedError(
                "Element recorder pg= must be resolved by the bridge "
                "build pipeline. Drive emission through "
                "apeSees(fem).tcl()/py()/run() instead of calling "
                "_emit directly with a pg= spec."
            )
        args += list(self.response)
        emitter.recorder("Element", *args)


# ---------------------------------------------------------------------------
# MPCO — ``recorder mpco ...`` (HDF5)
# ---------------------------------------------------------------------------

@dataclass(frozen=True, kw_only=True, slots=True)
class MPCO(Recorder):
    """``recorder mpco`` — write a single HDF5 ``.mpco`` file.

    OpenSees command::

        recorder mpco fname.mpco [-N nodal_responses...]
                                 [-E elem_responses...]
                                 [-T dt $dt | -T nsteps $n]
                                 [-R $regTag]

    The MPCO recorder captures the full response tensor for each
    requested token (no per-DOF selection at write time); STKO /
    apeGmsh consumers filter at read time. At least one of
    ``nodal_responses`` or ``elem_responses`` must be non-empty.

    Cadence is selected by exactly one of ``dT`` (seconds) or
    ``nsteps`` (analysis steps). Supplying both raises ``ValueError``;
    supplying neither records every analysis step.

    **Filtering** — MPCO records the whole model by default. To
    restrict output to a subset of nodes/elements, supply any of
    ``nodes=`` / ``nodes_pg=`` / ``elements=`` / ``elements_pg=``:
    the build pipeline auto-emits an OpenSees ``region $tag -node ...
    -ele ...`` command before the recorder and passes ``-R $tag`` to
    MPCO. ``nodes=`` is mutually exclusive with ``nodes_pg=``; the
    same applies to the element pair. When all four are ``None``
    (the default) MPCO records the whole model and no region is
    emitted.

    Parameters
    ----------
    file
        Output ``.mpco`` (HDF5) file path.
    nodal_responses
        Tuple of MPCO ``-N`` tokens (e.g. ``("displacement",
        "reactionForce")``). Empty tuple means no nodal recording.
    elem_responses
        Tuple of MPCO ``-E`` tokens (e.g. ``("stresses",
        "section.fiber.stress")``). Empty tuple means no element
        recording.
    dT
        Optional time-based cadence (seconds). Mutually exclusive
        with ``nsteps``.
    nsteps
        Optional step-based cadence (every N analysis steps).
        Mutually exclusive with ``dT``.
    nodes
        Explicit tuple of node tags to include in the region filter.
        Mutually exclusive with ``nodes_pg``.
    nodes_pg
        Physical-group label whose nodes the region filter targets.
        Mutually exclusive with ``nodes``. Resolved by the bridge
        build pipeline at emit time.
    elements
        Explicit tuple of element tags to include in the region
        filter. Mutually exclusive with ``elements_pg``.
    elements_pg
        Physical-group label whose elements the region filter
        targets. Mutually exclusive with ``elements``. Resolved by
        the bridge build pipeline at emit time.

    Note
    ----
    The bridge does not interpret ``-R``-bearing MPCO arg tails when
    ``_emit`` is called directly (outside the build pipeline); the
    ``pg=`` form is materialised by
    :func:`apeGmsh.opensees._internal.build.emit_recorder_spec`,
    which resolves selectors, allocates a region tag, emits the
    region, and replaces the spec via :func:`dataclasses.replace`
    with explicit ``nodes=``/``elements=`` before driving ``_emit``.
    """

    file: str
    nodal_responses: tuple[str, ...] = ()
    elem_responses: tuple[str, ...] = ()
    dT: float | None = None
    nsteps: int | None = None
    nodes: tuple[int, ...] | None = None
    nodes_pg: str | None = None
    elements: tuple[int, ...] | None = None
    elements_pg: str | None = None
    # When the build pipeline materialises the ``*_pg=`` form into a
    # concrete region, it replays the spec with ``_region_tag=$tag``
    # and ``nodes=``/``elements=`` populated; ``_emit`` reads
    # ``_region_tag`` to append ``-R $tag`` to the MPCO command. End
    # users never set this directly.
    _region_tag: int | None = None

    def __post_init__(self) -> None:
        if not (self.nodal_responses or self.elem_responses):
            raise ValueError(
                "MPCO: at least one of nodal_responses or "
                "elem_responses required."
            )
        if self.dT is not None and self.nsteps is not None:
            raise ValueError(
                "MPCO: supply only one of dT or nsteps "
                f"(got dT={self.dT!r}, nsteps={self.nsteps!r})."
            )
        if self.nodes is not None and self.nodes_pg is not None:
            raise ValueError(
                "MPCO: supply only one of nodes= or nodes_pg= "
                f"(got nodes={self.nodes!r}, nodes_pg={self.nodes_pg!r})."
            )
        if self.elements is not None and self.elements_pg is not None:
            raise ValueError(
                "MPCO: supply only one of elements= or elements_pg= "
                f"(got elements={self.elements!r}, "
                f"elements_pg={self.elements_pg!r})."
            )
        # Reject silent-empty-output asymmetric filter combos.  An
        # OpenSees ``region`` populated with only ``-node ...`` does NOT
        # auto-derive elements (``MeshRegion::setNodes`` is one-way; the
        # reverse ``setElements`` -> nodes IS auto-derived), so MPCO
        # filtered by a node-only region produces an empty element
        # stream — a silent runtime bug.  Refuse the combo at
        # construction time.
        node_filter = self.nodes is not None or self.nodes_pg is not None
        elem_filter = (
            self.elements is not None or self.elements_pg is not None
        )
        if (
            node_filter and not elem_filter and self.elem_responses
        ):
            raise ValueError(
                "MPCO: node-only filter (nodes= or nodes_pg=) cannot be "
                "combined with elem_responses — the auto-emitted region "
                "would carry no -ele entries, and OpenSees MeshRegion "
                "does not auto-derive elements from nodes, so MPCO "
                "would produce an empty element stream.  Supply "
                "elements= / elements_pg= (or drop elem_responses)."
            )
        if (
            elem_filter and not node_filter and self.nodal_responses
        ):
            # The symmetric case is less dangerous (OpenSees does
            # auto-derive nodes from elements via the connectivity), but
            # rejecting it keeps the API contract uniform and forces
            # the user to be explicit about the nodal scope of the
            # filter.
            raise ValueError(
                "MPCO: element-only filter (elements= or elements_pg=) "
                "cannot be combined with nodal_responses without an "
                "explicit nodes= / nodes_pg= — the region's auto-derived "
                "node set is implicit and varies with element type. "
                "Supply nodes= / nodes_pg= explicitly (or drop "
                "nodal_responses)."
            )

    def dependencies(self) -> tuple[Primitive, ...]:
        return ()

    def has_filter(self) -> bool:
        """True iff any node/element selector was supplied.

        Used by the partition-aware build pipeline (ADR 0027 INV-4) to
        decide whether the recorder needs a per-rank region pass — a
        whole-model MPCO (no filter) emits one ``recorder mpco`` line
        and nothing else.
        """
        return (
            self.nodes is not None
            or self.nodes_pg is not None
            or self.elements is not None
            or self.elements_pg is not None
        )

    def resolve_filter_ids(
        self, fem: "FEMData",
    ) -> tuple[tuple[int, ...], tuple[int, ...]]:
        """Resolve ``nodes`` / ``nodes_pg`` / ``elements`` / ``elements_pg``
        to explicit id tuples — no emission, no tag allocation.

        Returns ``(node_ids, elem_ids)``. Either may be empty when its
        side was not requested; an empty *result* on a *requested* side
        (e.g. ``nodes_pg="X"`` resolving to zero nodes) raises
        :class:`BridgeError` to mirror the OpenSees runtime rejection of
        an empty region.

        This is the partition-aware split-point of the legacy single-
        pass :meth:`materialize`: the partition orchestrator calls
        ``resolve_filter_ids`` once globally to determine the full
        filter id set, then intersects per-rank before emitting the
        region.  Whole-model recording (``has_filter() is False``) is
        a no-op pass-through — callers should not invoke this method
        in that case.
        """
        from ._internal.build import (
            BridgeError,
            expand_pg_to_elements,
            expand_pg_to_nodes,
        )

        # Resolve node-side selector.
        node_ids: tuple[int, ...] = ()
        if self.nodes_pg is not None:
            node_ids = expand_pg_to_nodes(fem, self.nodes_pg)
            if not node_ids:
                raise BridgeError(
                    f"MPCO recorder filter: nodes_pg={self.nodes_pg!r} "
                    "resolved to zero nodes against the FEM snapshot. "
                    "An empty region is rejected by OpenSees at runtime; "
                    "check the PG name spelling and that the PG was "
                    "populated before get_fem_data."
                )
        elif self.nodes is not None:
            node_ids = tuple(int(n) for n in self.nodes)
            if not node_ids:
                raise BridgeError(
                    "MPCO recorder filter: nodes=() is empty.  An empty "
                    "region is rejected by OpenSees at runtime; supply a "
                    "non-empty tuple or drop the nodes= kwarg."
                )

        # Resolve element-side selector.
        elem_ids: tuple[int, ...] = ()
        if self.elements_pg is not None:
            elem_ids = tuple(
                eid for eid, _conn in expand_pg_to_elements(fem, self.elements_pg)
            )
            if not elem_ids:
                raise BridgeError(
                    f"MPCO recorder filter: elements_pg={self.elements_pg!r} "
                    "resolved to zero elements against the FEM snapshot. "
                    "An empty region is rejected by OpenSees at runtime; "
                    "check the PG name spelling and that elements were "
                    "registered against it before get_fem_data."
                )
        elif self.elements is not None:
            elem_ids = tuple(int(e) for e in self.elements)
            if not elem_ids:
                raise BridgeError(
                    "MPCO recorder filter: elements=() is empty.  An "
                    "empty region is rejected by OpenSees at runtime; "
                    "supply a non-empty tuple or drop the elements= kwarg."
                )

        return node_ids, elem_ids

    def materialize(
        self,
        emitter: "Emitter",
        fem: "FEMData",
        tags: "TagAllocator | None",
        fem_eid_to_ops_tag: "dict[int, int] | None" = None,
    ) -> "MPCO":
        """Resolve filter selectors against the FEM and emit the region.

        Whole-model recording (no filter selectors) is a no-op pass-
        through.  When any of ``nodes`` / ``nodes_pg`` / ``elements`` /
        ``elements_pg`` is set, this method:

        1. Resolves ``*_pg`` to explicit id tuples via the bridge's
           PG-expansion helpers; refuses empty resolutions with
           :class:`BridgeError` (an empty OpenSees region is rejected
           at runtime).
        2. Allocates one fresh region tag from ``tags`` (must be
           supplied — the bridge build pipeline forwards the
           ``TagAllocator``).
        3. Emits one ``region $tag -node ... -ele ...`` line on
           ``emitter``.
        4. Returns a clone with the filter selectors cleared and
           ``_region_tag`` populated, so the subsequent ``_emit``
           appends ``-R $tag`` to the MPCO command.

        Used by the flat / unpartitioned emit path.  The partitioned
        emit path (ADR 0027 INV-4) invokes :meth:`resolve_filter_ids`
        once and emits the per-rank region line itself; it then
        injects ``_region_tag=`` onto the spec via
        :func:`dataclasses.replace` directly, bypassing this method.
        """
        if not self.has_filter():
            return self

        from ._internal.build import BridgeError

        if tags is None:
            raise BridgeError(
                "MPCO with nodes=/elements=/nodes_pg=/elements_pg= filter "
                "requires a TagAllocator on emit_recorder_spec(..., tags=); "
                "the bridge build pipeline supplies one — tests that "
                "bypass the bridge must pass it explicitly."
            )

        # TODO: when ``elements_pg=`` is set, ``elem_ids`` here are FEM
        # eids, but the region's ``-ele`` arg needs OpenSees element
        # tags.  Same drift as the Element recorder (closed by
        # ``Element.materialize`` above).  Translation requires both
        # the ``fem_eid_to_ops_tag`` map AND a reformulation of the
        # partitioned ``element_owner`` intersection in
        # ``_emit_mpco_filter_regions_for_rank`` (which is keyed by
        # FEM eid today).  Tracked as a follow-up.
        del fem_eid_to_ops_tag
        node_ids, elem_ids = self.resolve_filter_ids(fem)

        # Allocate one region tag for this MPCO recorder and emit it.
        # One ``region`` command can carry both ``-node`` and ``-ele``
        # flags; MPCO's ``-R`` then filters both nodal and element
        # results.  At least one of node_ids / elem_ids is guaranteed
        # non-empty (empty-resolution branches in resolve_filter_ids
        # raise before we get here, and __post_init__ already verified
        # at least one selector was supplied).
        region_tag = tags.allocate("region")
        region_args: list[int | float | str] = []
        if node_ids:
            region_args += ["-node", *node_ids]
        if elem_ids:
            region_args += ["-ele", *elem_ids]
        emitter.region(region_tag, *region_args)

        return replace(
            self,
            nodes_pg=None,
            elements_pg=None,
            nodes=node_ids if node_ids else None,
            elements=elem_ids if elem_ids else None,
            _region_tag=region_tag,
        )

    def _emit(self, emitter: "Emitter", tag: int) -> None:
        # ``pg=`` selectors must be materialised by the build pipeline
        # before _emit runs; reaching here with one set means someone
        # bypassed the bridge.
        if self.nodes_pg is not None or self.elements_pg is not None:
            raise NotImplementedError(
                "MPCO nodes_pg=/elements_pg= must be resolved by the "
                "bridge build pipeline. Drive emission through "
                "apeSees(fem).tcl()/py()/run() instead of calling "
                "_emit directly with a pg= spec."
            )
        args: list[int | float | str] = [self.file]
        if self.nodal_responses:
            args += ["-N", *self.nodal_responses]
        if self.elem_responses:
            args += ["-E", *self.elem_responses]
        if self.dT is not None:
            args += ["-T", "dt", self.dT]
        elif self.nsteps is not None:
            args += ["-T", "nsteps", self.nsteps]
        if self._region_tag is not None:
            args += ["-R", self._region_tag]
        emitter.recorder("mpco", *args)


# ---------------------------------------------------------------------------
# RecorderDeclaration — unified recorder spec (Phase 9)
# ---------------------------------------------------------------------------
#
# Two recorder declaration systems coexist on `main` until Phase 9:
#   - `Node` / `Element` / `MPCO` typed primitives (above)
#   - `apeGmsh.results.spec.Recorders` fluent helper (legacy)
#
# Phase 9 introduces a single bridge-side declaration that subsumes
# both surfaces. The typed primitives stay as the "exact OpenSees
# command shape" path; `RecorderDeclaration` is the "broad canonical
# vocabulary" path that supports nodes / elements / line_stations /
# gauss / fibers / layers / modal categories with shorthand expansion.
#
# This commit (Phase 9 commit 2) defines the types only; emit-time
# fan-out lands in commit 3 (`_internal/build.py::emit_recorder_spec`).

ALL_RECORDER_CATEGORIES: tuple[str, ...] = (
    "nodes",
    "elements",
    "line_stations",
    "gauss",
    "fibers",
    "layers",
    "modal",
)

# Per-category canonical component sets. Components outside the
# category's allowed set fail validation unless supplied via ``raw=``
# (the escape hatch for non-catalogued OpenSees tokens).
_CATEGORY_CANONICALS: dict[str, frozenset[str]] = {
    "nodes":         frozenset(NODAL_KINEMATICS + NODAL_FORCES),
    "elements":      frozenset(PER_ELEMENT_NODAL_FORCES),
    "line_stations": frozenset(LINE_DIAGRAMS),
    "gauss":         frozenset(STRESS + STRAIN + DERIVED_SCALARS + MATERIAL_STATE),
    "fibers":        frozenset(FIBER + MATERIAL_STATE),
    "layers":        frozenset(FIBER + MATERIAL_STATE),
    "modal":         frozenset(),  # no per-component vocabulary; n_modes only
}


@dataclass(frozen=True, kw_only=True, slots=True)
class RecorderRecord:
    """One category-level declaration entry within a RecorderDeclaration.

    Stores already-expanded canonical components (or raw OpenSees
    tokens via the ``raw=`` escape hatch). Shorthand expansion
    (``"displacement"`` → ``displacement_x/y/z``) happens at
    construction in the namespace method (Phase 9 commit 3), not in
    this dataclass — by the time a record is built, components are
    fully expanded.

    Parameters
    ----------
    category
        One of :data:`ALL_RECORDER_CATEGORIES`.
    components
        Tuple of canonical component names. Validated against
        :data:`_CATEGORY_CANONICALS` per category, plus indexed
        canonicals (``state_variable_<n>``, ``fiber_stress_<n>``,
        ``spring_force_<n>``) recognized via :func:`is_canonical`.
    raw
        Escape hatch for non-canonical OpenSees tokens (e.g. a
        custom recorder response). Bypasses canonical validation.
    pg / label / selection / ids
        Target selectors. ``ids=`` is mutually exclusive with the
        named selectors. Resolution against FEMData happens at
        emit time (commit 3).
    dt / n_steps
        Recording cadence. At most one may be set; both ``None``
        records every step.
    name
        Optional user-supplied name for this record; auto-generated
        when ``None``.
    n_modes
        Required for ``category="modal"``; rejected for other
        categories.
    element_class_name
        Optional OpenSees C++ class name override for element-level
        records. Used by the .out transcoder to disambiguate
        elements that share a flat response size (e.g. tri31 vs
        SSPquad). Carried from the legacy ``Recorders.elements``
        contract.
    """

    category: str
    components: tuple[str, ...] = ()
    raw: tuple[str, ...] = ()
    pg: tuple[str, ...] = ()
    label: tuple[str, ...] = ()
    selection: tuple[str, ...] = ()
    ids: tuple[int, ...] | None = None
    dt: float | None = None
    n_steps: int | None = None
    name: str | None = None
    n_modes: int | None = None
    element_class_name: str | None = None

    def __post_init__(self) -> None:
        # Category
        if self.category not in ALL_RECORDER_CATEGORIES:
            raise ValueError(
                f"RecorderRecord: unknown category {self.category!r}. "
                f"Allowed: {ALL_RECORDER_CATEGORIES}."
            )

        # Components — must be canonical and category-allowed (or
        # match an indexed-canonical pattern). Raw tokens bypass
        # validation.
        allowed = _CATEGORY_CANONICALS.get(self.category, frozenset())
        for comp in self.components:
            if comp in allowed:
                continue
            # Allow indexed canonicals (state_variable_N, fiber_stress_N,
            # spring_force_N etc.) in element-level categories.
            if is_canonical(comp) and self.category in (
                "elements", "line_stations", "gauss", "fibers", "layers",
            ):
                continue
            raise ValueError(
                f"RecorderRecord(category={self.category!r}): component "
                f"{comp!r} not in canonical vocabulary or category-allowed "
                f"set. Use raw= for non-canonical OpenSees tokens."
            )

        # Cadence: at most one of dt / n_steps
        if self.dt is not None and self.n_steps is not None:
            raise ValueError(
                "RecorderRecord: supply at most one of dt= or n_steps= "
                f"(got dt={self.dt!r}, n_steps={self.n_steps!r})."
            )

        # Selectors: ids is mutually exclusive with named selectors
        named_used = bool(self.pg or self.label or self.selection)
        if self.ids is not None and named_used:
            raise ValueError(
                "RecorderRecord: ids= is mutually exclusive with "
                f"pg=/label=/selection= (got ids={self.ids!r}, "
                f"pg={self.pg!r}, label={self.label!r}, "
                f"selection={self.selection!r})."
            )

        # Modal: requires n_modes; rejects components / selectors
        if self.category == "modal":
            if self.n_modes is None or self.n_modes < 1:
                raise ValueError(
                    "RecorderRecord(category='modal'): n_modes >= 1 required."
                )
        elif self.n_modes is not None:
            raise ValueError(
                f"RecorderRecord(category={self.category!r}): "
                "n_modes is only valid for category='modal'."
            )


@dataclass(frozen=True, kw_only=True, slots=True)
class RecorderDeclaration(Recorder):
    """A bundle of recorder records, registered as a single Primitive.

    Captures the bridge's ``ndm`` and ``ndf`` at construction time
    (Phase 9 D8 — implicit source-of-truth binding). Drives the
    file-emit path via :func:`emit_recorder_spec` in
    :mod:`apeGmsh.opensees._internal.build`.

    Parameters
    ----------
    records
        Tuple of :class:`RecorderRecord` entries. Each is one
        category-level declaration; emit fans them out into one or
        more concrete OpenSees ``recorder`` commands.
    name
        Identifier for this declaration (defaults to ``"default"``).
        Multiple named declarations can coexist on one bridge.
    ndm, ndf
        Snapshot of the bridge's ``ndm``/``ndf`` at construction time.
        Used downstream for shorthand expansion and validation. The
        bridge passes these in (Phase 9 D8 — user never repeats
        ``ops.model(ndm=, ndf=)`` values).
    file_root
        Directory prefix for emitted ``.out`` files. Each record fans
        out to ``<file_root>/<decl.name>__<record_name>__<token>.out``.
        Defaults to ``"."`` (current working directory).
    """

    records: tuple[RecorderRecord, ...]
    name: str = "default"
    ndm: int = 3
    ndf: int = 6
    file_root: str = "."

    def dependencies(self) -> tuple[Primitive, ...]:
        return ()

    def _emit(self, emitter: "Emitter", tag: int) -> None:
        # Emit fan-out lives in emit_recorder_spec (Phase 9 commit 3).
        # Direct ._emit calls are not supported because per-record
        # translation needs FEM context (for pg/label/selection
        # resolution) that the bridge fan-out supplies.
        raise NotImplementedError(
            "RecorderDeclaration._emit is driven by emit_recorder_spec "
            "in apeGmsh.opensees._internal.build; call it via the "
            "bridge build pipeline rather than directly."
        )
