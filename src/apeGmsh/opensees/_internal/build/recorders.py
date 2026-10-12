"""recorders: part of the ``build/`` package (S2-a split of ``_internal/build.py``)."""
from __future__ import annotations
from typing import TYPE_CHECKING

from typing import Iterable

from ..types import Recorder
from ...recorder import RecorderDeclaration, RecorderRecord
from .errors import BridgeError
from .pg_expand import (
    _expand_label_to_nodes,
    expand_pg_to_elements,
    expand_pg_to_nodes,
)
from .ownership import FemToOpsTagMap


if TYPE_CHECKING:
    from ...emitter.base import Emitter
    from apeGmsh.mesh.FEMData import FEMData
    from ..tag_plan import TagPlan
    from ..types import Element
    from typing import Mapping

    from .pg_expand import ElementPlanRows

    #: The element tag plan: each element spec with its columnar rows.
    ElementPlan = tuple[tuple[Element, ElementPlanRows], ...]


# ---------------------------------------------------------------------------
# Recorder fan-out — Node / Element recorders with pg= resolved to ids
# ---------------------------------------------------------------------------

def emit_recorder_spec(
    spec: Recorder,
    emitter: "Emitter",
    tag: int,
    fem: "FEMData",
    *,
    tag_plan: "TagPlan | None" = None,
    fem_eid_to_ops_tag: "FemToOpsTagMap | None" = None,
) -> None:
    """Drive a recorder's emit through its :meth:`Recorder.materialize`.

    :class:`RecorderDeclaration` follows a different shape (one
    declaration fans out to many ``recorder`` lines) and stays on its
    own branch.  Every other recorder routes through
    :meth:`Recorder.materialize` which resolves any ``pg=``-style
    selectors against ``fem``, emits any auxiliary declarations
    (e.g. MPCO's ``region`` line) on ``emitter`` directly, and
    returns a clone of itself with the build-time selectors cleared.
    The dispatcher then invokes ``_emit`` on the materialised spec.

    Recorders that carry no build-time selectors (e.g. a Node recorder
    constructed with explicit ``nodes=``) inherit the default
    no-op :meth:`Recorder.materialize` and pass through unchanged.

    ``fem_eid_to_ops_tag`` is the bridge-built ``{fem_eid: ops_tag}``
    map — element-targeting recorders (``Element``) use it to translate
    FEM eids resolved from ``pg=`` into the actual OpenSees element
    tags emitted by the element fan-out.  Without it (legacy direct
    callers), ``Element`` recorders fall back to FEM eids verbatim,
    which silently writes recorder lines that target the wrong
    elements whenever an element primitive consumed an allocator slot
    in ``_register`` — the bridge always supplies the map on the
    recorder emit pass.
    """
    if isinstance(spec, RecorderDeclaration):
        _emit_recorder_declaration(
            spec, emitter, fem, fem_eid_to_ops_tag=fem_eid_to_ops_tag,
            element_plan=(
                tag_plan.elements.specs if tag_plan is not None else None),
        )
        return
    materialised = spec.materialize(
        emitter, fem, tag_plan, fem_eid_to_ops_tag=fem_eid_to_ops_tag,
    )
    materialised._emit(emitter, tag)


# ---------------------------------------------------------------------------
# RecorderDeclaration emit fan-out (Phase 9 commit 3)
# ---------------------------------------------------------------------------

def _emit_recorder_declaration(
    decl: RecorderDeclaration,
    emitter: "Emitter",
    fem: "FEMData",
    *,
    fem_eid_to_ops_tag: "FemToOpsTagMap | None" = None,
    element_plan: "ElementPlan | None" = None,
) -> None:
    """Walk a :class:`RecorderDeclaration` and emit one
    ``emitter.recorder(...)`` call per (ops_token, target_set) group.

    ``element_plan`` is the bridge's element tag plan (each element spec
    with its rows). Node components map to DOFs through the per-node
    layout (:func:`_node_record_layout`): with the plan, each node's own
    ndf, and a rotational component only where the node's rotation role
    allows it; without it (``ModelData``, a hand-written deck) every node
    carries the declared ``(decl.ndm, decl.ndf)`` layout.

    Per Phase 9 commit 3a, only the ``"nodes"`` category is handled
    end-to-end. Other categories raise :class:`NotImplementedError`
    pointing at the follow-up commits.

    ``fem_eid_to_ops_tag`` is the bridge-built ``{fem_eid: ops_tag}``
    map.  Element-level records (``elements`` / ``line_stations`` /
    ``gauss``) translate FEM eids through this map before writing
    ``-ele`` arg lists — without it the recorder would target the raw
    FEM eids, which silently differ from the emitted OpenSees tags
    whenever an Element primitive consumed an allocator slot in
    ``_register``.  Mirrors the typed-``Element`` recorder
    materialise path (``recorder.py``) so both emit routes resolve to
    the same ops_tag list.  Legacy direct callers (no bridge) pass
    ``None`` and fall back to raw FEM eids.
    """
    from ..._recorder_translate import (
        element_record_response_tokens,
        group_node_components_by_ops_token,
    )

    for record in decl.records:
        if record.category in ("fibers", "layers", "modal"):
            raise NotImplementedError(
                f"RecorderDeclaration record(category={record.category!r}) "
                "is not file-emit-able; use DomainCapture instead (Phase 9 "
                "commit 5 provides the bridge-friendly entry point)."
            )
        if record.category not in (
            "nodes", "elements", "line_stations", "gauss",
        ):  # pragma: no cover  - guarded by RecorderRecord validation
            raise ValueError(
                f"unrecognized category {record.category!r} on "
                f"RecorderDeclaration record"
            )

        # Schema 2.3.0: bracket every fan-out with declaration
        # metadata so the H5 emitter can archive the original intent
        # alongside each emitted ``recorder(...)`` call. Lean emitters
        # (Tcl / Py / Live / Recording) treat these calls as no-ops.
        emitter.recorder_declaration_begin(
            declaration_name=decl.name,
            record_name=record.name,
            category=record.category,
            components=record.components,
            raw=record.raw,
            pg=record.pg,
            label=record.label,
            selection=record.selection,
            ids=record.ids,
            dt=record.dt,
            n_steps=record.n_steps,
            file_root=decl.file_root,
        )
        try:
            if record.category == "nodes":
                _emit_nodes_record(record, decl, emitter, fem,
                                   group_node_components_by_ops_token,
                                   element_plan=element_plan)
            else:
                _emit_element_level_record(
                    record, decl, emitter, fem,
                    element_record_response_tokens,
                    fem_eid_to_ops_tag=fem_eid_to_ops_tag,
                )
        finally:
            emitter.recorder_declaration_end()


def _emit_nodes_record(
    record: RecorderRecord,
    decl: RecorderDeclaration,
    emitter: "Emitter",
    fem: "FEMData",
    group: object,  # callable; passed in to keep imports local to caller
    *,
    element_plan: "ElementPlan | None" = None,
) -> None:
    """Emit one node-level :class:`RecorderRecord`.

    Resolves selectors (``pg``, ``label``, ``selection``, ``ids``) to
    a flat node-tag tuple, groups canonical components by their
    OpenSees recorder token, and emits one ``recorder Node`` call per
    canonical (``ops_token``, ``target_set``) group plus one extra
    ``recorder Node`` per ``raw=`` token (with dofs defaulting to all
    DOFs from ``decl.ndf``).

    File path convention:
      * canonical: ``<file_root>/<decl.name>__<record_name>__<token>.out``
      * raw: ``<file_root>/<decl.name>__<record_name>__raw_<token>.out``
    """
    node_ids = _resolve_node_targets(record, fem)
    if not node_ids:
        return  # nothing to record — silent skip mirrors typed-primitive behavior

    # Group canonical components by ops token (e.g. "disp": (1, 2)),
    # each mapped through the targets' DOF layout. Caller passes the
    # translator to keep its import scoped to the
    # _emit_recorder_declaration function.
    grouped: dict[str, tuple[int, ...]] = {}
    if record.components:
        node_ndf = (
            _inferred_node_ndf(fem, element_plan, int(decl.ndm))
            if element_plan is not None else None)
        layout = _node_record_layout(
            record, decl, node_ids, element_plan, node_ndf=node_ndf)
        grouped = group(record.components, layout)  # type: ignore[operator]
    record_name = record.name or "default"

    for ops_token, dofs in grouped.items():
        file_path = _recorder_file_path(
            decl.file_root, decl.name, record_name, ops_token,
        )
        args: list[int | float | str] = ["-file", file_path]
        if record.dt is not None:
            args += ["-dT", record.dt]
        # Default time_format is "dt" for declared records — they're
        # broader-vocabulary and time-aware consumers (results) expect
        # the leading time column.
        args += ["-time"]
        args += ["-node", *node_ids]
        args += ["-dof", *dofs]
        args.append(ops_token)
        emitter.recorder("Node", *args)

    # Raw escape hatch: one extra recorder Node per raw token. Dofs
    # default to all DOFs (1..ndf) since raw tokens bypass the
    # canonical→dof translation.
    if record.raw:
        all_dofs = tuple(range(1, decl.ndf + 1))
        for raw_token in record.raw:
            file_path = _recorder_file_path(
                decl.file_root, decl.name, record_name,
                f"raw_{_sanitize_raw_token(raw_token)}",
            )
            args = ["-file", file_path]
            if record.dt is not None:
                args += ["-dT", record.dt]
            args += ["-time"]
            args += ["-node", *node_ids]
            args += ["-dof", *all_dofs]
            args.append(raw_token)
            emitter.recorder("Node", *args)


#: The ``ndf`` values ``_load_dof_layout`` lays out exactly; any other
#: (a 3-D u-p node's 4) names only its leading ``ndm`` translations here.
_LAID_OUT_NDF: frozenset[int] = frozenset({1, 2, 3, 6})


def _inferred_node_ndf(
    fem: "FEMData", element_plan: "ElementPlan", ndm: int,
) -> "dict[int, int]":
    """``{node: ndf}`` the bridge infers from the planned element specs
    (ADR 0048). A node absent from it (element-less, or touched only by
    adaptive elements) takes the ``ops.model`` envelope."""
    from .ndf import infer_node_ndf

    return {int(k): int(v) for k, v in infer_node_ndf(
        fem, [spec for spec, _sub in element_plan], ndm).items()}


def _node_layout(ndf: int, ndm: int) -> tuple[int, ...]:
    """The spatial slot each DOF of an ``(ndm, ndf)`` node carries."""
    from .ndf import _load_dof_layout

    if ndf in _LAID_OUT_NDF:
        return _load_dof_layout(ndf, ndm)
    return tuple(range(min(ndf, ndm)))


def _node_record_layout(
    record: RecorderRecord,
    decl: RecorderDeclaration,
    node_ids: tuple[int, ...],
    element_plan: "ElementPlan | None",
    *,
    node_ndf: "Mapping[int, int] | None" = None,
) -> tuple[int, ...]:
    """The DOF layout that maps ``record``'s components to the same DOFs
    on every node of ``node_ids``; raise a :class:`BridgeError` naming a
    node where one has no DOF.

    Shared by the ``recorder.declare`` emit and ``ops.results`` (which
    re-derives the deck's columns), so the two agree.

    - Without an element plan (``ModelData``: a hand-written deck) every
      node carries the deck's declared ``(decl.ndm, decl.ndf)`` layout.
    - With the plan, each node carries the layout of its own ndf
      (``node_ndf``, the envelope ``decl.ndf`` where absent), and a
      rotational component also needs the node's rotation role
      (:meth:`~apeGmsh.opensees._readback.RotationRoles.refusal`, the
      rule ``.read`` applies): a u-p node's DOF ``ndm + 1`` is pore
      pressure, never a rotation.
    - ``pore_pressure`` refuses: OpenSees' ``recorder Node ... pressure``
      response is the PFEM pressure constraint, not a u-p pressure DOF.

    The layouts ``_load_dof_layout`` names place a spatial slot at the
    same DOF wherever they hold it, so one ``-dof`` list serves every
    node and a recorder line never needs splitting; a disagreement raises.
    """
    from ..._readback import rotation_roles
    from ..._recorder_translate import (
        group_node_components_by_ops_token,
        node_component_spatial,
    )
    from .ndf import _load_dof_layout

    where = f"recorder.declare({decl.name!r})"
    ndm = int(decl.ndm)
    spatial = {c: node_component_spatial(c) for c in record.components}
    pressure = [c for c, s in spatial.items() if s is not None and s[1] < 0]
    if pressure:
        raise BridgeError(
            f"{where}: {pressure[0]!r} is not recorded by a declared node "
            "record: the OpenSees 'pressure' node response is the PFEM "
            "pressure constraint, not a u-p pressure DOF. Record the u-p "
            f"pressure DOF ({ndm + 1}) with ops.recorder.Node(response="
            "'disp', dofs=...) on the u-p vertex nodes, or with "
            "ops.recorder.MPCO."
        )

    if element_plan is None:
        layouts = {n: _load_dof_layout(int(decl.ndf), ndm) for n in node_ids}
    else:
        rotational = [c for c, s in spatial.items()
                      if s is not None and s[1] >= 3]
        if rotational:
            roles = rotation_roles(element_plan, node_ids)
            refused = [(n, why) for n in node_ids
                       if (why := roles.refusal(n)) is not None]
            if refused:
                shown = "; ".join(f"node {n} ({why})" for n, why in refused[:3])
                more = len(refused) - min(len(refused), 3)
                raise BridgeError(
                    f"{where}: {rotational[0]!r} has no established DOF at "
                    f"{shown}" + (f"; and {more} more node(s)" if more else "")
                    + ". Narrow the selector to nodes whose elements carry "
                    "a rotation, or drop the rotational component."
                )
        ndf_of = node_ndf or {}
        layouts = {n: _node_layout(int(ndf_of.get(n, decl.ndf)), ndm)
                   for n in node_ids}

    first: "tuple[int, dict[str, tuple[int, ...]]] | None" = None
    for n, layout in layouts.items():
        try:
            grouped = group_node_components_by_ops_token(
                record.components, layout)
        except ValueError as e:
            envelope = (element_plan is not None
                        and n not in (node_ndf or {}))
            raise BridgeError(
                f"{where}: node {n} (ndm={ndm}) has no DOF for a declared "
                f"component: {e}"
                + (f" No element sizes node {n}, so the recorder emit takes "
                   f"the ops.model envelope ndf={decl.ndf} for it. If "
                   f"ops.ndf raised node {n} above that, the emit cannot "
                   "see it yet and refuses rather than guess a DOF: a "
                   "temporary regression, tracked in #1745."
                   if envelope else "")
            ) from e
        if first is None:
            first = (n, grouped)
        elif grouped != first[1]:
            raise BridgeError(  # pragma: no cover - laid-out layouts agree
                f"{where}: nodes {first[0]} and {n} carry the components at "
                f"different DOFs ({first[1]} vs {grouped}); declare them in "
                "separate records."
            )
    return layouts[node_ids[0]]


def _resolve_node_targets(
    record: RecorderRecord, fem: "FEMData",
) -> tuple[int, ...]:
    """Resolve a node-category :class:`RecorderRecord`'s selectors to
    a flat tuple of node tags.

    Supports ``ids=`` (mutex with named selectors) and the named
    selectors ``pg=`` / ``label=`` / ``selection=`` (composable — the
    resulting target sets are unioned and deduplicated, mirroring the
    legacy ``Recorders`` helper semantics).
    """
    if record.ids is not None:
        return tuple(int(i) for i in record.ids)

    chunks: list[Iterable[int]] = []
    for pg_name in record.pg:
        chunks.append(expand_pg_to_nodes(fem, pg_name))
    for label_name in record.label:
        chunks.append(_expand_label_to_nodes(fem, label_name))
    for sel_name in record.selection:
        chunks.append(_expand_selection_to_nodes(fem, sel_name))

    if not chunks:
        return ()
    out: list[int] = []
    seen: set[int] = set()
    for chunk in chunks:
        for tag in chunk:
            t = int(tag)
            if t not in seen:
                seen.add(t)
                out.append(t)
    return tuple(out)


def _expand_selection_to_nodes(fem: "FEMData", sel_name: str) -> tuple[int, ...]:
    """Return node IDs registered under ``sel_name`` on ``fem.mesh_selection``.

    Raises :class:`BridgeError` if the FEM snapshot has no
    ``mesh_selection`` store or the selection name is unknown.
    """
    store = getattr(fem, "mesh_selection", None)
    if store is None:
        raise BridgeError(
            f"selection {sel_name!r} requested but FEM snapshot has no "
            f"mesh_selection store (no post-mesh selections were declared "
            f"on the session)."
        )
    try:
        ids = store.node_ids(sel_name)
    except (KeyError, ValueError) as e:
        raise BridgeError(
            f"node selection {sel_name!r} not found on FEM snapshot."
        ) from e
    return tuple(int(n) for n in ids)


# ---------------------------------------------------------------------------
# Element-level emit (Phase 9 commit 3b)
# ---------------------------------------------------------------------------


def _emit_element_level_record(
    record: RecorderRecord,
    decl: RecorderDeclaration,
    emitter: "Emitter",
    fem: "FEMData",
    response_tokens: object,  # callable; passed in to keep imports local
    *,
    fem_eid_to_ops_tag: "FemToOpsTagMap | None" = None,
) -> None:
    """Emit one element-level :class:`RecorderRecord` (elements / gauss /
    line_stations).

    Resolves selectors to element IDs, picks the OpenSees response
    phrase based on the record's components (via the catalog-driven
    ``element_record_response_tokens`` helper), and issues one
    ``emitter.recorder("Element", ...)`` call for the canonical group
    plus one per ``raw=`` token.

    For ``line_stations`` records, also emits a paired
    ``integrationPoints`` recorder writing to ``<file>_gpx.out`` —
    consumed by the .out transcoder when reading the line-station
    results back into a :class:`LineStationSlab`.

    ``fem_eid_to_ops_tag`` translates resolved FEM eids into emitted
    OpenSees element tags so ``-ele`` arg lists match the OpenSees
    domain.  ``None`` (legacy direct callers / no-bridge tests) falls
    back to raw FEM eids, which is only correct when no Element
    primitive consumed an allocator slot.
    """
    elem_ids = _resolve_element_targets(
        record, fem, fem_eid_to_ops_tag=fem_eid_to_ops_tag,
    )
    if not elem_ids:
        return

    record_name = record.name or "default"
    emitted_canonical = False

    if record.components:
        # ``None`` = "the question does not apply here", which the
        # translator distinguishes from a checked-and-incapable ``False``
        # so it does not warn about something it cannot know or that is
        # already fine.  Two such cases: no element plan to check against
        # (legacy direct callers / ModelData's fem-eid-verbatim recorder
        # rendering), and a 3-D model, whose ``stresses`` response carries
        # σ_zz already — the plane-strain promotion is a 2-D affair.
        sigma_zz_capable = (
            None if (fem_eid_to_ops_tag is None or decl.ndm != 2)
            else fem_eid_to_ops_tag.all_sigma_zz_capable(elem_ids)
        )
        tokens = response_tokens(  # type: ignore[operator]
            record.category, record.components, record_name=record.name,
            sigma_zz_capable=sigma_zz_capable,
        )
        if tokens is not None:
            file_path = _recorder_file_path(
                decl.file_root, decl.name, record_name, record.category,
            )
            args: list[int | float | str] = ["-file", file_path]
            if record.dt is not None:
                args += ["-dT", record.dt]
            args += ["-time"]
            args += ["-ele", *elem_ids]
            args += list(tokens)
            emitter.recorder("Element", *args)
            emitted_canonical = True

    # Raw escape hatch: one extra recorder Element per raw token.
    if record.raw:
        for raw_token in record.raw:
            file_path = _recorder_file_path(
                decl.file_root, decl.name, record_name,
                f"raw_{_sanitize_raw_token(raw_token)}",
            )
            args = ["-file", file_path]
            if record.dt is not None:
                args += ["-dT", record.dt]
            args += ["-time"]
            args += ["-ele", *elem_ids]
            args.append(raw_token)
            emitter.recorder("Element", *args)

    # line_stations IP pairing: the .out transcoder needs per-element
    # integration-point positions to map the section.force samples back
    # to physical xi*L coordinates. Emit one gpx file per record (shared
    # across canonical + raw tokens — the GP geometry is independent of
    # the response token).
    if record.category == "line_stations" and (
        emitted_canonical or record.raw
    ):
        canonical_path = _recorder_file_path(
            decl.file_root, decl.name, record_name, record.category,
        )
        gpx_path = _line_station_gpx_path(canonical_path)
        args = ["-file", gpx_path]
        if record.dt is not None:
            args += ["-dT", record.dt]
        args += ["-time"]
        args += ["-ele", *elem_ids]
        args.append("integrationPoints")
        emitter.recorder("Element", *args)


def _resolve_element_targets(
    record: RecorderRecord, fem: "FEMData",
    *,
    fem_eid_to_ops_tag: "FemToOpsTagMap | None" = None,
) -> tuple[int, ...]:
    """Resolve an element-level :class:`RecorderRecord`'s selectors to
    a flat tuple of element tags.

    Supports ``ids=`` (mutex with named selectors) and the named
    selectors ``pg=`` / ``label=`` / ``selection=`` (composable — same
    union/dedup contract as :func:`_resolve_node_targets`).

    When ``fem_eid_to_ops_tag`` is supplied (the standard bridge
    path), the resolved FEM eids are translated to emitted OpenSees
    element tags before being returned — so the caller can write them
    straight into ``-ele`` arg lists without re-translating.  A FEM
    eid that maps to no ops_tag raises :class:`BridgeError` (the user
    targeted an element id that no ``ops.element.X(pg=...)``
    primitive emitted).  ``None`` falls back to raw FEM eids (legacy
    direct callers / no-bridge tests).
    """
    if record.ids is not None:
        fem_eids = tuple(int(i) for i in record.ids)
        return _translate_to_ops_tags(fem_eids, fem_eid_to_ops_tag, record)

    chunks: list[Iterable[int]] = []
    for pg_name in record.pg:
        chunks.append(eid for eid, _conn in expand_pg_to_elements(fem, pg_name))
    for label_name in record.label:
        chunks.append(_expand_label_to_elements(fem, label_name))
    for sel_name in record.selection:
        chunks.append(_expand_selection_to_elements(fem, sel_name))

    if not chunks:
        return ()
    out: list[int] = []
    seen: set[int] = set()
    for chunk in chunks:
        for tag in chunk:
            t = int(tag)
            if t not in seen:
                seen.add(t)
                out.append(t)
    return _translate_to_ops_tags(tuple(out), fem_eid_to_ops_tag, record)


def _translate_to_ops_tags(
    fem_eids: tuple[int, ...],
    fem_eid_to_ops_tag: "FemToOpsTagMap | None",
    record: "RecorderRecord",
) -> tuple[int, ...]:
    """Translate FEM eids → emitted OpenSees element tags.

    Mirrors the typed-``Element`` recorder ``materialize`` path
    (``recorder.py``).  ``None`` map = legacy direct caller, returns
    fem_eids verbatim.  A FEM eid not present in the map indicates the
    user targeted an element that no Element primitive emitted →
    :class:`BridgeError`.
    """
    if fem_eid_to_ops_tag is None:
        return fem_eids
    out: list[int] = []
    for eid in fem_eids:
        ops_tag = fem_eid_to_ops_tag.get(int(eid))
        if ops_tag is None:
            raise BridgeError(
                f"recorder declaration record (name={record.name!r}, "
                f"category={record.category!r}) resolves to FEM element "
                f"id {int(eid)} but no Element primitive was emitted at "
                f"that id (would silently target a wrong OpenSees tag). "
                "Either drop it from the recorder's selectors, or declare "
                "an ``ops.element.X(pg=...)`` primitive whose pg includes "
                "that element."
            )
        out.append(int(ops_tag))
    return tuple(out)


def _expand_label_to_elements(
    fem: "FEMData", label_name: str,
) -> tuple[int, ...]:
    """Return element IDs registered under ``label_name`` on
    ``fem.elements.labels``."""
    elements_obj = getattr(fem, "elements", None)
    labels = (
        getattr(elements_obj, "labels", None)
        if elements_obj is not None
        else None
    )
    if labels is None:
        raise BridgeError(
            f"label {label_name!r} requested but FEM snapshot has no "
            f"elements.labels accessor."
        )
    try:
        ids = labels.element_ids(label_name)
    except (KeyError, ValueError) as e:
        raise BridgeError(
            f"element label {label_name!r} not found on FEM snapshot."
        ) from e
    return tuple(int(e) for e in ids)


def _expand_selection_to_elements(
    fem: "FEMData", sel_name: str,
) -> tuple[int, ...]:
    """Return element IDs registered under ``sel_name`` on
    ``fem.mesh_selection``."""
    store = getattr(fem, "mesh_selection", None)
    if store is None:
        raise BridgeError(
            f"selection {sel_name!r} requested but FEM snapshot has no "
            f"mesh_selection store (no post-mesh selections were declared "
            f"on the session)."
        )
    try:
        ids = store.element_ids(sel_name)
    except (KeyError, ValueError) as e:
        raise BridgeError(
            f"element selection {sel_name!r} not found on FEM snapshot."
        ) from e
    return tuple(int(e) for e in ids)


# ---------------------------------------------------------------------------
# File-path helpers for RecorderDeclaration emit
# ---------------------------------------------------------------------------

def _recorder_file_path(file_root: str, *parts: str) -> str:
    """Build a recorder ``.out`` path from ``file_root`` and a sequence
    of basename parts joined by ``__``.

    Mirrors the legacy ``_build_file_path`` convention in
    ``apeGmsh.results.spec._emit`` — handles empty ``file_root`` (no
    prefix) and trailing slashes on the directory portion.
    """
    fname = "__".join(parts) + ".out"
    if not file_root:
        return fname
    sep = "" if file_root.endswith(("/", "\\")) else "/"
    return f"{file_root}{sep}{fname}"


def _line_station_gpx_path(line_station_file_path: str) -> str:
    """Return the paired ``integrationPoints`` recorder path for a
    line-stations file.

    Replaces the ``.out`` suffix with ``_gpx.out``. Matches the
    convention defined in ``apeGmsh.results.spec._emit.line_station_gpx_path``
    so the legacy .out transcoder locates the paired file unchanged.
    """
    if line_station_file_path.endswith(".out"):
        return line_station_file_path[:-4] + "_gpx.out"
    return line_station_file_path + "_gpx"


def _sanitize_raw_token(token: str) -> str:
    """Return a filename-safe form of ``token`` for raw= file paths.

    Replaces any character that isn't alphanumeric or underscore with
    ``_``. Raw tokens are user-supplied OpenSees response strings that
    may contain spaces, hyphens, or other shell-sensitive chars; the
    sanitized form is used only in the output ``.out`` filename — the
    raw token reaches OpenSees verbatim as the recorder response.
    """
    return "".join(c if (c.isalnum() or c == "_") else "_" for c in token)
