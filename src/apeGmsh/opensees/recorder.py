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

from typing import Iterable

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
    expand_many,
    is_canonical,
)
from ._internal.types import Primitive, Recorder

if TYPE_CHECKING:
    from apeGmsh.mesh.FEMData import FEMData

    from ._internal.build import FemToOpsTagMap
    from ._internal.tag_allocator import TagAllocator
    from .emitter.base import Emitter


__all__ = [
    # Typed primitives (Phase 3B)
    "Node",
    "Element",
    # Shared region-filter base (MPCO + Ladruno)
    "FilterableRecorder",
    "MPCO",
    # Ladruno fork recorder (recorder-integration L1)
    "Ladruno",
    # Ladruno live-telemetry monitor (fork-only SWMR sink)
    "Monitor",
    # Unified declaration (Phase 9)
    "RecorderRecord",
    "RecorderDeclaration",
    "ALL_RECORDER_CATEGORIES",
    # Shared declaration builder (used by the bridge namespace AND by
    # ModelData's hand-written-deck recorder surface)
    "build_recorder_declaration",
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
        fem_eid_to_ops_tag: "FemToOpsTagMap | dict[int, int] | None" = None,
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
        fem_eid_to_ops_tag: "FemToOpsTagMap | dict[int, int] | None" = None,
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
# FilterableRecorder — shared node/element region-filter machinery
# ---------------------------------------------------------------------------

@dataclass(frozen=True, kw_only=True, slots=True)
class FilterableRecorder(Recorder):
    """Base for HDF5 recorders that share MPCO's region-filter surface.

    Carries the four mutually-paired selectors (``nodes`` / ``nodes_pg``
    / ``elements`` / ``elements_pg``) and the region-emit machinery that
    turns them into an OpenSees ``region $tag -node ... -ele ...`` line
    plus a ``-R $tag`` on the recorder command. :class:`MPCO` and
    :class:`Ladruno` both inherit it; they differ only in the recorder
    *kind* token and (Ladruno) the trailing ``-G energy`` channel.

    Carries the shared value channels too (``file`` + ``nodal_responses``
    / ``elem_responses`` ``-N``/``-E`` + ``dT``/``nsteps`` ``-T`` cadence),
    so :meth:`_value_channel_args` builds the common ``-N ... -E ... -T ...
    -R ...`` tail once for both subclasses — only the recorder *kind*
    token and (Ladruno) the trailing ``-G energy`` are subclass-specific.

    Subclasses own only their required-response rule (MPCO needs nodal or
    element responses; Ladruno also accepts ``energy``); the cadence
    mutex (:meth:`_validate_cadence`) and the four selector guards
    (:meth:`_validate_filter`) are shared. The partition-aware build
    pipeline (ADR 0027 INV-4) keys its per-rank region pass on
    ``isinstance(spec, FilterableRecorder)`` via :meth:`has_filter` /
    :meth:`resolve_filter_ids`.
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
    # ``_region_tag`` to append ``-R $tag`` to the recorder command. End
    # users never set this directly.
    _region_tag: int | None = None

    # NOTE: ``dependencies()`` is intentionally NOT defined here.  The
    # cross-family primitive contract (test_primitive_base) requires every
    # *concrete* primitive to declare ``dependencies`` in its own
    # ``__dict__``; MPCO and Ladruno each define it directly (both return
    # ``()`` — recorders are leaves).

    def _validate_cadence(self) -> None:
        """Reject ``dT`` + ``nsteps`` together (call from ``__post_init__``)."""
        if self.dT is not None and self.nsteps is not None:
            raise ValueError(
                f"{type(self).__name__}: supply only one of dT or nsteps "
                f"(got dT={self.dT!r}, nsteps={self.nsteps!r})."
            )

    def _validate_filter(self) -> None:
        """The four shared selector guards (call from ``__post_init__``).

        The recorder name in each message comes from
        ``type(self).__name__`` so MPCO and Ladruno self-identify; the
        ``has_*`` booleans are derived from the base's own response fields.
        """
        kind = type(self).__name__
        has_nodal = bool(self.nodal_responses)
        has_elem = bool(self.elem_responses)
        if self.nodes is not None and self.nodes_pg is not None:
            raise ValueError(
                f"{kind}: supply only one of nodes= or nodes_pg= "
                f"(got nodes={self.nodes!r}, nodes_pg={self.nodes_pg!r})."
            )
        if self.elements is not None and self.elements_pg is not None:
            raise ValueError(
                f"{kind}: supply only one of elements= or elements_pg= "
                f"(got elements={self.elements!r}, "
                f"elements_pg={self.elements_pg!r})."
            )
        # Reject silent-empty-output asymmetric filter combos.  An
        # OpenSees ``region`` populated with only ``-node ...`` does NOT
        # auto-derive elements (``MeshRegion::setNodes`` is one-way; the
        # reverse ``setElements`` -> nodes IS auto-derived), so a recorder
        # filtered by a node-only region produces an empty element
        # stream — a silent runtime bug.  Refuse the combo at
        # construction time.
        node_filter = self.nodes is not None or self.nodes_pg is not None
        elem_filter = (
            self.elements is not None or self.elements_pg is not None
        )
        if node_filter and not elem_filter and has_elem:
            raise ValueError(
                f"{kind}: node-only filter (nodes= or nodes_pg=) cannot be "
                "combined with elem_responses — the auto-emitted region "
                "would carry no -ele entries, and OpenSees MeshRegion "
                "does not auto-derive elements from nodes, so the recorder "
                "would produce an empty element stream.  Supply "
                "elements= / elements_pg= (or drop elem_responses)."
            )
        if elem_filter and not node_filter and has_nodal:
            # The symmetric case is less dangerous (OpenSees does
            # auto-derive nodes from elements via the connectivity), but
            # rejecting it keeps the API contract uniform and forces
            # the user to be explicit about the nodal scope of the
            # filter.
            raise ValueError(
                f"{kind}: element-only filter (elements= or elements_pg=) "
                "cannot be combined with nodal_responses without an "
                "explicit nodes= / nodes_pg= — the region's auto-derived "
                "node set is implicit and varies with element type. "
                "Supply nodes= / nodes_pg= explicitly (or drop "
                "nodal_responses)."
            )

    def _require_materialized(self) -> None:
        """Guard ``_emit`` against an unmaterialised ``*_pg=`` selector.

        ``pg=`` selectors must be resolved by the build pipeline before
        ``_emit`` runs; reaching ``_emit`` with one set means someone
        bypassed the bridge.
        """
        if self.nodes_pg is not None or self.elements_pg is not None:
            raise NotImplementedError(
                f"{type(self).__name__} nodes_pg=/elements_pg= must be "
                "resolved by the bridge build pipeline. Drive emission "
                "through apeSees(fem).tcl()/py()/run() instead of calling "
                "_emit directly with a pg= spec."
            )

    def _value_channel_args(self) -> "list[int | float | str]":
        """The shared ``-N ... -E ... -T ... -R ...`` arg tail.

        Built once for both MPCO and Ladruno so the value-channel +
        region wire grammar cannot drift between them (the ``.ladruno``
        recorder is contracted to reproduce ``MPCORecorder``'s ``-N`` /
        ``-E`` / ``-T`` grammar bit-for-bit). Excludes the leading
        ``file`` and any subclass-specific trailing flag (Ladruno's
        ``-G energy``). ``_region_tag`` is read with ``is not None`` so a
        legitimate region tag of ``0`` still emits ``-R 0``.
        """
        args: list[int | float | str] = []
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
        return args

    def has_filter(self) -> bool:
        """True iff any node/element selector was supplied.

        Used by the partition-aware build pipeline (ADR 0027 INV-4) to
        decide whether the recorder needs a per-rank region pass — a
        whole-model recorder (no filter) emits one ``recorder`` line
        and nothing else.
        """
        return (
            self.nodes is not None
            or self.nodes_pg is not None
            or self.elements is not None
            or self.elements_pg is not None
        )

    def resolve_filter_ids(
        self,
        fem: "FEMData",
        fem_eid_to_ops_tag: "FemToOpsTagMap | dict[int, int] | None" = None,
    ) -> tuple[tuple[int, ...], tuple[int, ...]]:
        """Resolve ``nodes`` / ``nodes_pg`` / ``elements`` / ``elements_pg``
        to explicit id tuples — no emission, no tag allocation.

        Returns ``(node_ids, elem_ids)``. Either may be empty when its
        side was not requested; an empty *result* on a *requested* side
        (e.g. ``nodes_pg="X"`` resolving to zero nodes) raises
        :class:`BridgeError` to mirror the OpenSees runtime rejection of
        an empty region.

        ``fem_eid_to_ops_tag`` is the bridge-built ``{fem_eid: ops_tag}``
        map for element fan-out.  When supplied AND ``elements_pg`` is
        set, the resolved FEM eids are translated to OpenSees element
        tags before they flow into the region's ``-ele`` arg list
        (same drift as the Element recorder, closed by
        :meth:`Element.materialize`).  Lookup miss → :class:`BridgeError`.
        When ``None`` (legacy direct callers) the FEM eids are returned
        verbatim — the partition orchestrator uses this form so it can
        intersect per-rank in FEM-eid space, then translate at the
        final region-emit step.

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

        kind = type(self).__name__

        # Resolve node-side selector.
        node_ids: tuple[int, ...] = ()
        if self.nodes_pg is not None:
            node_ids = expand_pg_to_nodes(fem, self.nodes_pg)
            if not node_ids:
                raise BridgeError(
                    f"{kind} recorder filter: nodes_pg={self.nodes_pg!r} "
                    "resolved to zero nodes against the FEM snapshot. "
                    "An empty region is rejected by OpenSees at runtime; "
                    "check the PG name spelling and that the PG was "
                    "populated before get_fem_data."
                )
        elif self.nodes is not None:
            node_ids = tuple(int(n) for n in self.nodes)
            if not node_ids:
                raise BridgeError(
                    f"{kind} recorder filter: nodes=() is empty.  An empty "
                    "region is rejected by OpenSees at runtime; supply a "
                    "non-empty tuple or drop the nodes= kwarg."
                )

        # Resolve element-side selector.
        elem_ids: tuple[int, ...] = ()
        if self.elements_pg is not None:
            fem_eids = tuple(
                eid for eid, _conn in expand_pg_to_elements(fem, self.elements_pg)
            )
            if not fem_eids:
                raise BridgeError(
                    f"{kind} recorder filter: elements_pg={self.elements_pg!r} "
                    "resolved to zero elements against the FEM snapshot. "
                    "An empty region is rejected by OpenSees at runtime; "
                    "check the PG name spelling and that elements were "
                    "registered against it before get_fem_data."
                )
            if fem_eid_to_ops_tag is None:
                # Legacy / partition-orchestrator form: return FEM eids
                # verbatim so the caller can intersect per-rank in
                # FEM-eid space.  The bridge's flat path always supplies
                # the map; the partition path translates at the final
                # per-rank region-emit step.
                elem_ids = fem_eids
            else:
                ops_tags: list[int] = []
                for eid in fem_eids:
                    ops_tag = fem_eid_to_ops_tag.get(int(eid))
                    if ops_tag is None:
                        raise BridgeError(
                            f"{kind} recorder filter: elements_pg="
                            f"{self.elements_pg!r} resolves to FEM eid "
                            f"{eid} but no element was emitted at that "
                            "eid — declare an "
                            "``ops.element.X(pg=...)`` primitive whose "
                            f"pg includes {self.elements_pg!r}."
                        )
                    ops_tags.append(int(ops_tag))
                elem_ids = tuple(ops_tags)
        elif self.elements is not None:
            elem_ids = tuple(int(e) for e in self.elements)
            if not elem_ids:
                raise BridgeError(
                    f"{kind} recorder filter: elements=() is empty.  An "
                    "empty region is rejected by OpenSees at runtime; "
                    "supply a non-empty tuple or drop the elements= kwarg."
                )

        return node_ids, elem_ids

    def materialize(
        self,
        emitter: "Emitter",
        fem: "FEMData",
        tags: "TagAllocator | None",
        fem_eid_to_ops_tag: "FemToOpsTagMap | dict[int, int] | None" = None,
    ) -> "FilterableRecorder":
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
           appends ``-R $tag`` to the recorder command.

        Used by the flat / unpartitioned emit path.  The partitioned
        emit path (ADR 0027 INV-4) invokes :meth:`resolve_filter_ids`
        once and emits the per-rank region line itself; it then
        injects ``_region_tag=`` onto the spec via
        :func:`dataclasses.replace` directly, bypassing this method.
        """
        if not self.has_filter():
            return self

        from ._internal.build import BridgeError

        kind = type(self).__name__

        if tags is None:
            raise BridgeError(
                f"{kind} with nodes=/elements=/nodes_pg=/elements_pg= filter "
                "requires a TagAllocator on emit_recorder_spec(..., tags=); "
                "the bridge build pipeline supplies one — tests that "
                "bypass the bridge must pass it explicitly."
            )

        # ``elements_pg=`` resolution translates FEM eids → OpenSees
        # element tags via the bridge-built map (same drift as the
        # Element recorder, closed by ``Element.materialize`` above).
        # The partitioned emit path drives this method-bypass via
        # ``_plan_partitioned_mpco_recorders`` + ``_emit_mpco_filter_
        # regions_for_rank`` (which keeps the resolution in FEM-eid
        # space so the per-rank ``element_owner`` intersection works,
        # then translates at the final region-emit step).
        node_ids, elem_ids = self.resolve_filter_ids(
            fem, fem_eid_to_ops_tag=fem_eid_to_ops_tag,
        )

        # Allocate one region tag for this recorder and emit it.
        # One ``region`` command can carry both ``-node`` and ``-ele``
        # flags; the recorder's ``-R`` then filters both nodal and
        # element results.  At least one of node_ids / elem_ids is
        # guaranteed non-empty (empty-resolution branches in
        # resolve_filter_ids raise before we get here, and
        # __post_init__ already verified at least one selector was
        # supplied).
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


# ---------------------------------------------------------------------------
# MPCO — ``recorder mpco ...`` (HDF5)
# ---------------------------------------------------------------------------

@dataclass(frozen=True, kw_only=True, slots=True)
class MPCO(FilterableRecorder):
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

    # ``file`` + value-channel + filter fields are inherited from
    # FilterableRecorder; MPCO adds no fields of its own.

    def __post_init__(self) -> None:
        if not (self.nodal_responses or self.elem_responses):
            raise ValueError(
                "MPCO: at least one of nodal_responses or "
                "elem_responses required."
            )
        self._validate_cadence()
        self._validate_filter()

    def dependencies(self) -> tuple[Primitive, ...]:
        return ()

    def _emit(self, emitter: "Emitter", tag: int) -> None:
        self._require_materialized()
        emitter.recorder("mpco", self.file, *self._value_channel_args())


# ---------------------------------------------------------------------------
# Ladruno — ``recorder ladruno ...`` (HDF5, fork-only canonical recorder)
# ---------------------------------------------------------------------------

@dataclass(frozen=True, kw_only=True, slots=True)
class Ladruno(FilterableRecorder):
    """``recorder ladruno`` — write a single HDF5 ``.ladruno`` file.

    **Fork-only.** The ``ladruno`` recorder exists only in the Ladruno
    fork build of OpenSees (``nmorabowen/OpenSees@ladruno``); stock
    ``openseespy`` does not have it. Per the opt-in contract, *emission*
    works on **any** build (the deck line is just ``recorder ladruno
    ...``); the fork requirement bites only when the deck actually runs
    (``ops.run()`` / the live emitter). Gate at the point of use, never
    at import.

    OpenSees command (value channels + region filter + energy balance)::

        recorder ladruno fname.ladruno [-N nodal_responses...]
                                       [-E elem_responses...]
                                       [-T dt $dt | -T nsteps $n]
                                       [-R $regTag]
                                       [-G energy]

    The ``.ladruno`` recorder is forked from the frozen ``MPCORecorder``
    and shares its value-channel command grammar (the ``-N``/``-E``/
    ``-T`` channels reproduce the frozen recorder to 1e-12), so this
    dataclass mirrors :class:`MPCO` for those channels and inherits the
    same region-filter machinery from :class:`FilterableRecorder`. It
    diverges only in the recorder *kind* token (``ladruno`` vs ``mpco``),
    the output extension (``.ladruno``), and the trailing ``-G energy``
    channel.

    Cadence is selected by exactly one of ``dT`` (seconds) or ``nsteps``
    (analysis steps). Supplying both raises ``ValueError``; supplying
    neither records every analysis step.

    **Filtering** — like MPCO, Ladruno records the whole model by
    default. Supply any of ``nodes=`` / ``nodes_pg=`` / ``elements=`` /
    ``elements_pg=`` to restrict output: the build pipeline auto-emits an
    OpenSees ``region $tag -node ... -ele ...`` command before the
    recorder and passes ``-R $tag`` to ``ladruno``. ``nodes=`` is
    mutually exclusive with ``nodes_pg=`` (same for the element pair).
    A node-only filter cannot be combined with ``elem_responses`` (and
    vice versa) — the auto-region would carry no entries on the other
    side and produce an empty stream. The ``-R $tag`` is emitted
    **before** ``-G energy`` so the energy flag stays last.

    ``energy=True`` adds the fork's whole-model energy-balance channel
    (``-G energy`` → ``RESULTS/ON_DOMAIN/energyBalance``, components
    ``KE/IE/DW/ULW/RES/ERR``), read back via ``Results.energy()``. The
    flag is emitted **last**: the fork's ``-G`` parser eagerly consumes
    trailing region-tag integers and cannot rewind past a following
    flag, so ``-G energy -T nsteps 10`` is a parse error while
    ``-T nsteps 10 -G energy`` runs (run-verified on the fork build).

    **Energy balance.** Three forms, all run-verified on the fork build
    (which always writes the whole-model balance,
    ``RESULTS/ON_DOMAIN/energyBalance``, and adds a per-region balance,
    ``RESULTS/ON_REGIONS/energyBalance``, whenever a region tag is given):

    * ``energy=True`` alone → whole-model (``-G energy``, no tag).
    * ``energy=True`` + a value filter → energy over the **same** region
      the ``-R`` filter targets, reusing the filter's already-allocated
      tag (``-G energy $filterTag``). The *coupled* form.
    * ``energy_pg="X"`` → energy over an **independent** region X
      (``-G energy $tagX``), decoupled from the ``-R`` value filter. X may
      differ from (or exist without) the value-channel filter; it gets its
      own region tag. The *decoupled* form. ``energy_pg`` takes precedence
      over the coupled form, and implies energy recording (no need to also
      set ``energy=True``).

    All three ride the flat / staged / partitioned region plumbing — the
    decoupled region gets its own per-rank fan-out under partitioning, just
    like the value filter (ADR 0064 §4). Read any region's balance via
    ``Results.energy(region=<tag>)``; whole-model via ``Results.energy()``.

    Parameters
    ----------
    file
        Output ``.ladruno`` (HDF5) file path.
    nodal_responses
        Tuple of ``-N`` tokens (e.g. ``("displacement",
        "reactionForce")``). Empty tuple means no nodal recording.
    elem_responses
        Tuple of ``-E`` tokens (e.g. ``("stresses",
        "section.fiber.stress")``). Empty tuple means no element
        recording.
    dT
        Optional time-based cadence (seconds). Mutually exclusive with
        ``nsteps``.
    nsteps
        Optional step-based cadence (every N analysis steps). Mutually
        exclusive with ``dT``.
    energy
        Record the energy balance (``-G energy``). Whole-model when no
        filter is set; per-region (over the ``-R`` filter region, plus
        whole-model) when combined with a ``nodes=``/``elements=`` filter.
    energy_pg
        Physical-group label for a **decoupled** energy region — records
        energy over that PG (``-G energy $tag``) independent of the value
        filter. Implies energy recording. Resolved by the bridge build
        pipeline; gets its own auto-emitted region (per-rank under
        partitioning).
    nodes
        Explicit tuple of node tags for the region filter. Mutually
        exclusive with ``nodes_pg``.
    nodes_pg
        Physical-group label whose nodes the region filter targets.
        Mutually exclusive with ``nodes``. Resolved by the bridge build
        pipeline at emit time.
    elements
        Explicit tuple of element tags for the region filter. Mutually
        exclusive with ``elements_pg``.
    elements_pg
        Physical-group label whose elements the region filter targets.
        Mutually exclusive with ``elements``. Resolved by the bridge
        build pipeline at emit time.
    """

    # ``file`` + value-channel + filter fields are inherited from
    # FilterableRecorder; Ladruno adds the energy channel + an optional
    # decoupled energy region.
    energy: bool = False
    energy_pg: str | None = None
    # Region tags for ``-G energy $tag...`` (decoupled energy regions),
    # populated by ``materialize`` / the partition planner from
    # ``energy_pg``. End users never set this directly.
    _energy_region_tags: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        if not (
            self.nodal_responses
            or self.elem_responses
            or self.energy
            or self.energy_pg is not None
        ):
            raise ValueError(
                "Ladruno: at least one of nodal_responses, "
                "elem_responses, energy or energy_pg required."
            )
        self._validate_cadence()
        self._validate_filter()

    def dependencies(self) -> tuple[Primitive, ...]:
        return ()

    def resolve_energy_ids(
        self,
        fem: "FEMData",
        fem_eid_to_ops_tag: "FemToOpsTagMap | dict[int, int] | None" = None,
    ) -> tuple[int, ...]:
        """Resolve ``energy_pg`` to element ids for the energy region.

        Mirrors the element side of :meth:`resolve_filter_ids`: returns
        FEM eids verbatim when ``fem_eid_to_ops_tag is None`` (the
        partition orchestrator intersects per-rank in FEM-eid space),
        else translates to OpenSees element tags. Empty resolution →
        :class:`BridgeError`. Energy is an element quantity, so the
        region carries ``-ele`` only (the fork auto-derives the nodes).
        """
        from ._internal.build import BridgeError, expand_pg_to_elements

        assert self.energy_pg is not None  # caller-guarded
        fem_eids = tuple(
            eid for eid, _conn in expand_pg_to_elements(fem, self.energy_pg)
        )
        if not fem_eids:
            raise BridgeError(
                f"Ladruno energy region: energy_pg={self.energy_pg!r} "
                "resolved to zero elements against the FEM snapshot. "
                "An empty region is rejected by OpenSees at runtime; check "
                "the PG name spelling and that elements were registered "
                "against it before get_fem_data."
            )
        if fem_eid_to_ops_tag is None:
            return fem_eids
        ops_tags: list[int] = []
        for eid in fem_eids:
            ops_tag = fem_eid_to_ops_tag.get(int(eid))
            if ops_tag is None:
                raise BridgeError(
                    f"Ladruno energy region: energy_pg={self.energy_pg!r} "
                    f"resolves to FEM eid {eid} but no element was emitted "
                    "at that eid — declare an ``ops.element.X(pg=...)`` "
                    f"primitive whose pg includes {self.energy_pg!r}."
                )
            ops_tags.append(int(ops_tag))
        return tuple(ops_tags)

    def materialize(
        self,
        emitter: "Emitter",
        fem: "FEMData",
        tags: "TagAllocator | None",
        fem_eid_to_ops_tag: "FemToOpsTagMap | dict[int, int] | None" = None,
    ) -> "FilterableRecorder":
        """Emit the value-filter region (base) **and** the decoupled
        energy region (``energy_pg``), each as its own OpenSees ``region``.

        The energy region is independent of the ``-R`` value filter (the
        fork's ``-G energy <tag>`` list is orthogonal to ``-R``); it gets
        its own tag, recorded in ``_energy_region_tags`` and referenced by
        ``_emit`` as ``-G energy $tag``. Used by the flat path; the
        partitioned path builds the equivalent spec in
        :meth:`BuiltModel._plan_partitioned_mpco_recorders`.
        """
        # NOTE: explicit base call, not zero-arg ``super()`` — these are
        # ``@dataclass(slots=True)`` classes, which the decorator rebuilds,
        # leaving the ``super()`` ``__class__`` cell stale (TypeError).
        spec = FilterableRecorder.materialize(
            self, emitter, fem, tags, fem_eid_to_ops_tag,
        )
        if self.energy_pg is None:
            return spec
        from ._internal.build import BridgeError

        if tags is None:
            raise BridgeError(
                "Ladruno energy_pg= requires a TagAllocator on "
                "emit_recorder_spec(..., tags=); the bridge build pipeline "
                "supplies one — tests bypassing the bridge must pass it."
            )
        assert isinstance(spec, Ladruno)
        elem_ids = spec.resolve_energy_ids(fem, fem_eid_to_ops_tag)
        energy_tag = tags.allocate("region")
        emitter.region(energy_tag, "-ele", *elem_ids)
        return replace(spec, energy_pg=None, _energy_region_tags=(energy_tag,))

    def _require_materialized(self) -> None:
        # Explicit base call (slots-dataclass ``super()`` caveat, see
        # ``materialize`` above).
        FilterableRecorder._require_materialized(self)
        if self.energy_pg is not None:
            raise NotImplementedError(
                "Ladruno energy_pg= must be resolved by the bridge build "
                "pipeline. Drive emission through apeSees(fem).tcl()/py()/"
                "run() instead of calling _emit directly with an energy_pg= "
                "spec."
            )

    def _emit(self, emitter: "Emitter", tag: int) -> None:
        self._require_materialized()
        # ``_value_channel_args`` puts ``-R $tag`` last (when filtered);
        # the energy flag must trail it — the fork's ``-G`` parser eagerly
        # eats trailing ints and cannot rewind past a following flag, so
        # ``-G energy [$tag...]`` is always the final option.
        args: list[int | float | str] = [self.file, *self._value_channel_args()]
        if self.energy or self._energy_region_tags:
            args += ["-G", "energy"]
            if self._energy_region_tags:
                # Decoupled energy region(s) from ``energy_pg`` — energy
                # over their own region(s), independent of the ``-R`` value
                # filter (takes precedence over the coupled form below).
                args += list(self._energy_region_tags)
            elif self._region_tag is not None:
                # Coupled: ``energy=True`` + a value filter records energy
                # over the SAME region the ``-R`` filter targets (reuses
                # the tag ``materialize`` already allocated).
                args.append(self._region_tag)
        emitter.recorder("ladruno", *args)


@dataclass(frozen=True, kw_only=True, slots=True)
class Monitor(Recorder):
    """``recorder Monitor`` — live SWMR-HDF5 nodal-telemetry sink (fork-only).

    **Fork-only.** The ``Monitor`` recorder exists only in the Ladruno fork
    build; stock ``openseespy`` does not have it. Emission works on any
    build (the deck line is just ``recorder Monitor ...``); the fork
    requirement bites only when the deck runs. Gate at the point of use.

    Unlike the canonical :class:`Ladruno` recorder, the Monitor is a
    *lightweight live-telemetry sidecar*: it streams a few selected nodal
    scalars to a small SWMR-HDF5 file a viewer process can **tail while the
    analysis is still running**. The same file is a valid at-rest result
    once the run finishes — read both via
    :func:`apeGmsh.results.read_monitor` / :func:`apeGmsh.results.tail_monitor`.

    OpenSees command::

        recorder Monitor (-node n1 n2 ... | -region tag) -dof d1 d2 ...
                         [-resp disp|vel|accel|reaction]
                         -sink fname.h5 [-every K] [-hz H]

    Exactly one of ``nodes=`` (explicit tags) or ``pg=`` (physical-group
    label, resolved against the FEM snapshot by the bridge build pipeline)
    must be supplied. The recorded channels are the cartesian product of
    nodes × dofs, labelled ``node<N>.<resp>.dof<D>`` in node-major order.

    Parameters
    ----------
    sink
        Output ``.h5`` SWMR sink path.
    dofs
        DOF indices (1-based, OpenSees convention). At least one required.
    nodes
        Explicit tuple of node tags. Mutually exclusive with ``pg``.
    pg
        Physical-group label whose nodes are monitored. Mutually exclusive
        with ``nodes``; resolved at emit time.
    resp
        Nodal response — one of ``disp`` / ``vel`` / ``accel`` /
        ``reaction`` (the fork v1 set). Default ``disp``.
    every
        Step decimation — emit a frame every ``K`` analysis steps. ``None``
        records every step.
    hz
        Wall-clock throttle — emit at most ``H`` frames per second of real
        time (the first frame always passes). ``None`` means no throttle.
        Independent of ``every``; both may bound the stream.
    """

    sink: str
    dofs: tuple[int, ...]
    nodes: tuple[int, ...] | None = None
    pg: str | None = None
    resp: str = "disp"
    every: int | None = None
    hz: float | None = None

    def __post_init__(self) -> None:
        if (self.nodes is None) == (self.pg is None):
            raise ValueError(
                "Monitor recorder: supply exactly one of nodes= or pg= "
                f"(got nodes={self.nodes!r}, pg={self.pg!r})."
            )
        if not self.dofs:
            raise ValueError("Monitor recorder: at least one dof required.")
        if self.resp not in ("disp", "vel", "accel", "reaction"):
            raise ValueError(
                "Monitor recorder: resp must be one of disp|vel|accel|"
                f"reaction (got {self.resp!r})."
            )
        if self.every is not None and self.every < 1:
            raise ValueError(
                f"Monitor recorder: every must be >= 1 (got {self.every!r})."
            )
        if self.hz is not None and self.hz <= 0:
            raise ValueError(
                f"Monitor recorder: hz must be > 0 (got {self.hz!r})."
            )

    def dependencies(self) -> tuple[Primitive, ...]:
        return ()

    def materialize(
        self,
        emitter: "Emitter",
        fem: "FEMData",
        tags: "TagAllocator | None",
        fem_eid_to_ops_tag: "FemToOpsTagMap | dict[int, int] | None" = None,
    ) -> "Monitor":
        # Node tags equal FEM node ids (never rebased), so no element-tag
        # translation is needed — mirrors :meth:`Node.materialize`.
        del fem_eid_to_ops_tag
        if self.pg is None:
            return self
        from ._internal.build import expand_pg_to_nodes
        return replace(self, pg=None, nodes=expand_pg_to_nodes(fem, self.pg))

    def _emit(self, emitter: "Emitter", tag: int) -> None:
        if self.nodes is None:
            # pg → node fan-out is owned by :meth:`materialize`; the bridge
            # calls it before _emit. Reaching here means the bridge was
            # bypassed.
            raise NotImplementedError(
                "Monitor recorder pg= must be resolved by the bridge build "
                "pipeline. Drive emission through apeSees(fem).tcl()/py()/"
                "run() instead of calling _emit directly with a pg= spec."
            )
        args: list[int | float | str] = ["-node", *self.nodes, "-dof", *self.dofs]
        args += ["-resp", self.resp, "-sink", self.sink]
        if self.every is not None:
            args += ["-every", self.every]
        if self.hz is not None:
            args += ["-hz", self.hz]
        emitter.recorder("Monitor", *args)


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


# ---------------------------------------------------------------------------
# Shared declaration builder
# ---------------------------------------------------------------------------


def _normalize_str_selector(
    value: "Iterable[str] | str | None",
) -> tuple[str, ...]:
    """Coerce a ``str`` / iterable-of-str / ``None`` into a tuple of strs.

    A single string is wrapped in a one-element tuple (so the caller can
    write ``pg="Top"`` instead of ``pg=("Top",)``). ``None`` becomes the
    empty tuple.
    """
    if value is None:
        return ()
    if isinstance(value, str):
        return (value,)
    return tuple(value)


def build_recorder_declaration(
    *,
    ndm: int,
    ndf: int,
    nodes: "Iterable[str] | str" = (),
    elements: "Iterable[str] | str" = (),
    line_stations: "Iterable[str] | str" = (),
    gauss: "Iterable[str] | str" = (),
    raw_nodes: "Iterable[str] | str | None" = None,
    raw_elements: "Iterable[str] | str | None" = None,
    raw_line_stations: "Iterable[str] | str | None" = None,
    raw_gauss: "Iterable[str] | str | None" = None,
    pg: "str | Iterable[str] | None" = None,
    label: "str | Iterable[str] | None" = None,
    selection: "str | Iterable[str] | None" = None,
    ids: "Iterable[int] | None" = None,
    dt: float | None = None,
    n_steps: int | None = None,
    name: str = "default",
    record_name: str | None = None,
    element_class_name: str | None = None,
    file_root: str = ".",
) -> RecorderDeclaration:
    """Construct a :class:`RecorderDeclaration` from declarative kwargs.

    Single source of truth for shorthand expansion (``"displacement"``
    → ``displacement_x/y/z`` via the bound ``ndm``/``ndf``) and
    per-category record construction.  Shared by
    :meth:`apeGmsh.opensees._internal.ns.recorder._RecorderNS.declare`
    (bridge-owned models) and
    :meth:`apeGmsh.opensees.ModelData.recorders` (hand-written decks).

    ``ndm`` / ``ndf`` are supplied by the caller (the bridge or
    ``ModelData`` binds them at declaration time, Phase 9 D8 — the user
    never repeats ``ndm=``/``ndf=`` here).
    """
    pg_tuple = _normalize_str_selector(pg)
    label_tuple = _normalize_str_selector(label)
    selection_tuple = _normalize_str_selector(selection)
    ids_tuple = tuple(int(i) for i in ids) if ids is not None else None

    records: list[RecorderRecord] = []

    # Per-category record construction. Each category produces at most
    # one record (canonical components + raw tokens combined). A record
    # is skipped only when both are empty for that category.
    category_inputs: tuple[
        tuple[str, "Iterable[str] | str", "Iterable[str] | str | None"], ...
    ] = (
        ("nodes",         nodes,         raw_nodes),
        ("elements",      elements,      raw_elements),
        ("line_stations", line_stations, raw_line_stations),
        ("gauss",         gauss,         raw_gauss),
    )
    for category, canonical_kw, raw_kw in category_inputs:
        canonical_seq = _normalize_str_selector(canonical_kw)
        raw_seq = _normalize_str_selector(raw_kw)
        if not canonical_seq and not raw_seq:
            continue
        components = (
            expand_many(canonical_seq, ndm=ndm, ndf=ndf)
            if canonical_seq else ()
        )
        records.append(
            RecorderRecord(
                category=category,
                components=components,
                raw=raw_seq,
                pg=pg_tuple,
                label=label_tuple,
                selection=selection_tuple,
                ids=ids_tuple,
                dt=dt,
                n_steps=n_steps,
                name=record_name,
                element_class_name=(
                    element_class_name if category != "nodes" else None
                ),
            )
        )

    return RecorderDeclaration(
        records=tuple(records),
        name=name,
        ndm=ndm,
        ndf=ndf,
        file_root=file_root,
    )
