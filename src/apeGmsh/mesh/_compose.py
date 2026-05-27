"""apeGmsh.mesh._compose — Compose facade scaffold (Phase 3B.1 / ADR 0038).

The :class:`Compose` facade is a session-level entry point that lands the
shell of ADR 0038's ``g.compose(...)`` API: input validation, the
``compose_inspect`` / ``compose_list`` companion helpers, the
:class:`ComposedModule` handle, and the typed exception hierarchy.

The merge engine itself — tag-offset reservation, namespace prefix
sweep, record rewrite + verifier — is intentionally **deferred** to
Phase 3B.2.  Calling :meth:`Compose.compose` here raises
:class:`NotImplementedError` after the input gates pass; ``inspect`` and
``list`` are fully functional because they only read H5 metadata or walk
the current broker's ``fem.composed_from``.

Cross-references
----------------
* ADR 0038 §"g.compose() signature" — the entry-point contract.
* ADR 0038 §"Companion helpers (v1)" — inspect / list shape.
* ADR 0038 §"Tag-collision verifier" — the typed errors defined here.
* Phase 3A.1 substrate (PR #361):
  :class:`apeGmsh._kernel.records._compose.ComposeRecord` +
  :class:`apeGmsh._kernel.record_sets.ComposeSet` carry the provenance
  this facade will produce in Phase 3B.2 and inspects today.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass, field, replace as _dc_replace
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from .._kernel.records._compose import ComposeRecord

if TYPE_CHECKING:
    from .._core import apeGmsh
    from .FEMData import FEMData


# ---------------------------------------------------------------------------
# Exception hierarchy (ADR 0038 §"Tag-collision verifier")
# ---------------------------------------------------------------------------


class ComposeError(Exception):
    """Base for all compose-time errors per ADR 0038."""


class ComposeLabelError(ComposeError, ValueError):
    """Invalid ``label=`` per ADR 0038 §"g.compose() signature" line 94.

    Raised when ``label`` is empty, carries the namespace separator
    ``.``, the depth-boundary separator ``/``, whitespace, or starts /
    ends with ``_`` (the reserved-prefix convention).
    """


class ComposeAnchorError(ComposeError, ValueError):
    """``anchor=`` combined with a non-zero ``translate=`` per ADR 0038
    §"g.compose() signature" line 104.

    ``anchor`` is sugar that resolves to a translate; the two are
    mutually exclusive."""


class ComposeCapacityError(ComposeError, ValueError):
    """Source span exceeds the configured reservation cap per ADR 0038
    §"Tag-collision verifier" check 5.

    Raised by the merge engine in Phase 3B.2 when an explicit
    ``compose_size_per_module=N`` is smaller than the source's actual
    tag span.  The exception type lives in 3B.1 so callers writing
    forward-compatible try/excepts can catch it today.
    """


class ComposeDepthExceededError(ComposeError, ValueError):
    """Nested-compose depth exceeds ``max_compose_depth`` per ADR 0038
    §"Namespace rule".

    Raised by the merge engine in Phase 3B.2; defined here so the
    typed-error surface for compose-time failures is complete.
    """


class ComposeNamespaceCollisionError(ComposeError, ValueError):
    """Post-rewrite PG-name collision per ADR 0038 §"Tag-collision
    verifier" check 4.

    Raised by the verifier in Phase 3B.2; defined here so callers can
    write forward-compatible exception handlers.
    """


class ComposeFilterWarning(UserWarning):
    """One-line warning per filtered-with-warning record kind per
    compose call per ADR 0038 §"Merge semantics".

    Emitted by the merge engine in Phase 3B.2 for the stages /
    time-series / load-patterns kinds.  Defined here so the warning
    class lives with its sibling errors.
    """


# ---------------------------------------------------------------------------
# Rewritten bundle + rewrite engine (Phase 3B.2a / ADR 0038)
# ---------------------------------------------------------------------------
#
# The bundle is the output of the read+rewrite half of compose.  Phase
# 3B.2b will consume it to merge into the host ``FEMData``; this PR
# only ships the producer.  See ADR 0038 §"Tag-offset scheme" and
# §"Tag-reference rewrite checklist".


@dataclass(frozen=True)
class _RewrittenBundle:
    """Internal: rewritten IMPORT-verdict records ready to merge.

    Produced by :func:`_rewrite_source_for_compose`; consumed by Phase
    3B.2b's host-merge step.  Carries the rewritten record lists keyed
    by record kind plus the compose metadata needed to populate the
    host's :class:`~apeGmsh._kernel.records._compose.ComposeRecord`.

    FILTER + DISCARD verdicts are dropped silently in this slice —
    3B.2b emits the user-facing :class:`ComposeFilterWarning`.  The
    bundle is a flat container — no analysis happens during
    construction; rewrites are applied before instantiation.
    """

    # ── Compose metadata (populates ComposeRecord in 3B.2b) ────────
    label: str
    source_path: str
    source_fem_hash: str
    source_neutral_schema_version: str
    translate: tuple[float, float, float]
    rotate: tuple[float, float, float, float] | None
    partition_rank: int | None
    properties: dict
    composed_at: str

    # ── Reservation window ─────────────────────────────────────────
    base: int
    size: int
    source_span: int
    source_min_tag: int

    # ── Rewritten IMPORT records ───────────────────────────────────
    #
    # Container shape matches what 3B.2b will iterate to splice into
    # the host broker.  Nodes / elements are stored as raw numpy
    # arrays (the broker's native shape) — the rewriter rebuilds
    # them from a fresh ``read_fem_h5`` call so the bundle owns its
    # own buffers (no aliasing back to the source file).  PG / label
    # records remain in their broker dict shape because the broker
    # has no dataclass wrapping; constraint / load / mass / SP
    # records carry the dataclass instances with offset-rewritten
    # tag fields and ``{label}.``-prefixed name fields.
    node_ids: np.ndarray
    node_coords: np.ndarray
    node_ndf: np.ndarray | None
    # elements: ``{type_code: ElementGroup}`` mirroring the broker's
    # ``ElementComposite._groups``.  Each group's ``ids`` and
    # ``connectivity`` are offset-rewritten.
    element_groups: dict
    # Per-side PG / label dict — the broker's native
    # ``{(dim, tag): info_dict}`` shape; ``info_dict`` carries the
    # rewritten ``node_ids`` / ``element_ids`` arrays and the
    # ``{label}.``-prefixed ``name`` string.
    node_physical: dict
    elem_physical: dict
    node_labels: dict
    elem_labels: dict
    # Mesh selections: ``MeshSelectionStore`` instance (or ``None``).
    mesh_selection: Any
    # Parts: ``{label_str: set[int]}`` (node-side, element-side).
    part_node_map: dict
    part_elem_map: dict
    # Constraint / load / mass / SP record lists.  Each entry is a
    # dataclass instance with offset-rewritten tag fields.
    node_constraints: tuple
    elem_constraints: tuple
    nodal_loads: tuple
    element_loads: tuple
    sp_records: tuple
    mass_records: tuple


# ── Helper: schema-version + tag-span reader ───────────────────────


def _compute_source_span(
    source_path: "str | Path",
) -> tuple[int, int, int]:
    """Return ``(span, min_tag, max_tag)`` from a 2.8/2.9 source H5.

    ADR 0038 §"Tag-offset scheme — per-module auto-sizing".  For
    2.9.x sources, ``span`` is read from the ``/meta/@tag_span_max``
    attribute (O(1)).  For 2.8.x sources lacking that attribute, falls
    back to a dataset scan over ``/nodes/ids`` + ``/elements/{type}/ids``
    and emits one :class:`UserWarning` per call:

        "source written by pre-2.9.0 apeGmsh; tag span computed by
         dataset scan — re-save under 2.9.0 to skip this on future
         composes"

    The min/max include both nodes and elements (compose reserves one
    window covering both classes uniformly — see ADR 0038 line 294).

    Raises
    ------
    SchemaVersionError
        When the source's neutral schema is outside the reader's two-
        version window (e.g. pre-2.8.x).
    """
    import h5py

    from apeGmsh.opensees.emitter.h5_reader import MalformedH5Error
    from apeGmsh.opensees._internal.schema_version import (
        NEUTRAL,
        read_zone_version,
        reader_version,
        validate_zone_version,
    )

    p = Path(source_path)
    with h5py.File(str(p), "r") as f:
        if "meta" not in f:
            raise MalformedH5Error(
                f"{p}: missing /meta group; not an apeGmsh model.h5"
            )
        meta_attrs = f["meta"].attrs
        # Schema gate — pre-2.8.x is outside the reader window and
        # surfaces a typed SchemaVersionError (the same surface
        # ``read_fem_h5`` raises on stale sources).
        file_version = read_zone_version(meta_attrs, NEUTRAL)
        if file_version is not None:
            validate_zone_version(
                file_version, reader_version(NEUTRAL), zone=NEUTRAL,
            )

        # 2.9.x fast path: /meta/@tag_span_max is present.
        if "tag_span_max" in meta_attrs:
            span = int(meta_attrs["tag_span_max"])
            min_tag, max_tag = _scan_min_max_tags(f)
            # Defensive: if the span attr disagrees with the scan
            # (re-saved file?), trust the scan — span is derived from
            # min/max in the writer.  Tests assert the 2.9 happy path
            # uses the attr value, so prefer it when consistent.
            if max_tag - min_tag + 1 == span:
                return span, min_tag, max_tag
            return max_tag - min_tag + 1, min_tag, max_tag

        # 2.8.x fallback: one-shot warning + dataset scan.
        warnings.warn(
            f"source written by pre-2.9.0 apeGmsh; tag span computed by "
            f"dataset scan — re-save under 2.9.0 to skip this on future "
            f"composes",
            UserWarning,
            stacklevel=2,
        )
        min_tag, max_tag = _scan_min_max_tags(f)
        return max_tag - min_tag + 1, min_tag, max_tag


def _scan_min_max_tags(f: Any) -> tuple[int, int]:
    """Dataset scan over ``/nodes/ids`` + ``/elements/{type}/ids``.

    Returns ``(min_tag, max_tag)`` covering both nodes and elements.
    An empty H5 returns ``(0, 0)`` — the compose engine handles that
    edge case at the reservation step.
    """
    mins: list[int] = []
    maxs: list[int] = []
    if "nodes" in f and "ids" in f["nodes"]:
        nids = f["nodes/ids"][...]
        if nids.size > 0:
            mins.append(int(nids.min()))
            maxs.append(int(nids.max()))
    if "elements" in f:
        for type_name in f["elements"].keys():
            sub = f["elements"][type_name]
            if hasattr(sub, "keys") and "ids" in sub:
                eids = sub["ids"][...]
                if eids.size > 0:
                    mins.append(int(eids.min()))
                    maxs.append(int(eids.max()))
    if not mins:
        return 0, 0
    return min(mins), max(maxs)


# ── Helper: reservation window math ────────────────────────────────


def _compute_reservation(
    source_span: int,
    host_max_tag: int,
    previous_reservations: tuple[tuple[int, int], ...] = (),
    *,
    granularity: int = 1_000_000,
    compose_size_per_module: int | None = None,
) -> tuple[int, int]:
    """Return ``(base, size)`` per ADR 0038 §"Tag-offset scheme".

    Parameters
    ----------
    source_span : int
        Source's ``max_tag - min_tag + 1`` (from
        :func:`_compute_source_span`).
    host_max_tag : int
        The host's current maximum tag — the reservation must sit
        strictly above it.  Pass ``0`` for an empty host.
    previous_reservations : tuple of (base, size)
        Reservations already issued under previous compose calls.
        When non-empty, the new base is ``previous_base + previous_size``
        of the last entry rather than rounded up from ``host_max_tag``.
    granularity : int, default 1_000_000
        Power-of-10 round-up unit (``Compose.RESERVATION_GRANULARITY``).
    compose_size_per_module : int or None
        Explicit reservation-size floor.  When supplied, ``size`` is
        ``max(auto_size, compose_size_per_module)``.

    Returns
    -------
    (int, int)
        ``(base, size)`` of the new reservation window.
    """
    # Auto-size from the source's actual span.
    auto_size = (
        (source_span + granularity - 1) // granularity
    ) * granularity
    # ``compose_size_per_module`` is a FLOOR (advisory headroom for
    # users who expect the source to grow — ADR 0038 line 264-270);
    # the rewriter still computes the natural size and takes the
    # larger of the two.
    if compose_size_per_module is not None:
        size = max(auto_size, compose_size_per_module)
    else:
        size = auto_size
    # Empty source edge case — span 0 round-trips to size 0 which
    # would collapse the reservation.  Snap to one granularity unit so
    # the rest of the pipeline has a non-degenerate window.
    if size == 0:
        size = granularity

    if previous_reservations:
        prev_base, prev_size = previous_reservations[-1]
        base = prev_base + prev_size
    else:
        # First compose: round host_max_tag up to the next granularity
        # boundary so the reservation sits clearly above existing tags.
        base = (
            (host_max_tag + granularity) // granularity
        ) * granularity
    return base, size


# ── Helper: geometric transform ────────────────────────────────────


def _apply_geometric_transform(
    xyz: np.ndarray,
    *,
    translate: tuple[float, float, float],
    rotate: tuple[float, float, float, float] | None,
) -> np.ndarray:
    """Apply rotate-then-translate to an ``(N, 3)`` node-coord array.

    ``rotate`` is axis-angle ``(x, y, z, theta)`` with ``theta`` in
    radians, matching ``gmsh.model.occ.rotate(...)`` (ADR 0038 line 101).
    ``translate`` is a 3-vector applied AFTER rotation.  Returns a NEW
    array; input is not mutated.  When ``rotate is None`` AND
    ``translate == (0, 0, 0)``, returns the input unchanged
    (no-copy fast path).
    """
    arr = np.asarray(xyz, dtype=np.float64)
    is_identity_translate = all(float(t) == 0.0 for t in translate)
    if rotate is None and is_identity_translate:
        return arr  # no-copy fast path

    out = arr
    if rotate is not None:
        ax, ay, az, theta = (
            float(rotate[0]), float(rotate[1]),
            float(rotate[2]), float(rotate[3]),
        )
        # Normalise the axis (Gmsh's rotate is axis-angle; the axis
        # must be a unit vector for the Rodrigues formula).
        axis = np.array([ax, ay, az], dtype=np.float64)
        norm = float(np.linalg.norm(axis))
        if norm == 0.0:
            # Degenerate axis with non-zero theta is ill-defined; fall
            # back to identity rotation (consistent with gmsh's tolerant
            # behaviour).
            out = arr.copy()
        else:
            axis = axis / norm
            c = float(np.cos(theta))
            s = float(np.sin(theta))
            one_minus_c = 1.0 - c
            kx, ky, kz = float(axis[0]), float(axis[1]), float(axis[2])
            # Rodrigues rotation matrix.
            R = np.array([
                [c + kx * kx * one_minus_c,
                 kx * ky * one_minus_c - kz * s,
                 kx * kz * one_minus_c + ky * s],
                [ky * kx * one_minus_c + kz * s,
                 c + ky * ky * one_minus_c,
                 ky * kz * one_minus_c - kx * s],
                [kz * kx * one_minus_c - ky * s,
                 kz * ky * one_minus_c + kx * s,
                 c + kz * kz * one_minus_c],
            ], dtype=np.float64)
            out = arr @ R.T

    if not is_identity_translate:
        out = out + np.array(translate, dtype=np.float64)
    return out


# ── Helper: per-record tag-offset + namespace rewrite ──────────────


def _rewrite_record(
    rec: Any,
    *,
    offset: int,
    label: str,
) -> Any:
    """Return a copy of ``rec`` with tag fields offset + name fields
    namespace-prefixed.

    Iterates the record's ``tag_rewrite_spec`` class attribute (ADR 0038
    §"Tag-reference rewrite checklist").  A ``None`` spec means the
    record is opt-out (DISCARD / DEFER verdicts) and the caller skips
    it without invoking this helper.

    Nested record lists (e.g. ``NodeToSurfaceRecord.rigid_link_records``)
    are walked recursively via the ``nested_records`` key.

    Returns a NEW record instance; ``rec`` is not mutated.
    """
    spec = getattr(rec, "tag_rewrite_spec", None)
    if spec is None:
        raise TypeError(
            f"record kind {type(rec).__name__} has no tag_rewrite_spec; "
            f"cover-set drift — add a ClassVar declaration per "
            f"ADR 0038 §'Tag-reference rewrite checklist'"
        )

    # Build a kwargs dict of the changed fields and pass to dataclasses.replace.
    changes: dict[str, Any] = {}

    for fname in spec.get("tag_fields_scalar", ()):
        current = getattr(rec, fname)
        if current is None:
            continue
        changes[fname] = int(current) + offset

    for fname in spec.get("tag_fields_array", ()):
        current = getattr(rec, fname)
        if current is None:
            continue
        # Preserve container type: list-in -> list-out; ndarray-in ->
        # ndarray-out.  Both shapes occur on the resolved records
        # (NodeGroupRecord.slave_nodes is a list; phantom_nodes is too;
        # NodeToSurfaceRecord.phantom_coords is ndarray — but coords
        # aren't tag-bearing, so they're not in tag_fields_array).
        if isinstance(current, np.ndarray):
            changes[fname] = current.astype(np.int64) + np.int64(offset)
        else:
            changes[fname] = [int(x) + offset for x in current]

    for fname in spec.get("name_fields", ()):
        current = getattr(rec, fname)
        if current is None:
            continue
        # ``pattern`` defaults to ``"default"`` on LoadRecord — still
        # namespaced so loads in different modules carrying the same
        # pattern name don't collide on the host.
        changes[fname] = f"{label}.{current}"

    for fname in spec.get("nested_records", ()):
        current = getattr(rec, fname)
        if current is None:
            continue
        changes[fname] = [
            _rewrite_record(child, offset=offset, label=label)
            for child in current
        ]

    return _dc_replace(rec, **changes)


# ── Helper: rewrite the broker dict-shaped PG / label entries ──────


def _rewrite_named_groups(
    groups: dict,
    *,
    offset: int,
    label: str,
) -> dict:
    """Return a copy of a ``{(dim, tag): info_dict}`` mapping with
    ``node_ids`` / ``element_ids`` / ``connectivity`` offset and the
    ``name`` namespaced.

    The ``info_dict`` carries object-dtype arrays under ``node_ids`` /
    ``element_ids`` (cast by :class:`NamedGroupSet.__init__`); we work
    in int64 space for the math and let the consumer recoerce.
    """
    out: dict = {}
    for key, info in groups.items():
        new_info: dict = {}
        for k, v in info.items():
            if k == "name":
                new_info[k] = f"{label}.{v}"
            elif k in ("node_ids", "element_ids", "connectivity"):
                if v is None:
                    new_info[k] = v
                else:
                    arr = np.asarray(v, dtype=np.int64)
                    if arr.size == 0:
                        new_info[k] = arr
                    else:
                        new_info[k] = arr + np.int64(offset)
            elif k == "node_coords":
                # Coords are not tag-bearing; copy as-is (geometric
                # transform applies to fem.nodes.coords, not the
                # PG-side mirror copies which the rewriter re-derives
                # from the rebuilt node table when 3B.2b merges).  In
                # 3B.2a we preserve the source coords; 3B.2b will
                # decide whether to re-fetch from the rewritten node
                # table or keep these PG-local copies.
                new_info[k] = v
            else:
                # Forward unknown keys (e.g. nested per-type 'groups')
                # untouched — 3B.2b will handle the rewrite if needed.
                new_info[k] = v
        out[key] = new_info
    return out


# ── Helper: rewrite the parts maps ─────────────────────────────────


def _rewrite_part_map(
    part_map: dict,
    *,
    offset: int,
    label: str,
) -> dict:
    """Return a copy of ``{part_label: set[int]}`` with each set's int
    members offset and each part_label namespace-prefixed.
    """
    out: dict = {}
    for plabel, members in part_map.items():
        new_label = f"{label}.{plabel}"
        out[new_label] = {int(x) + offset for x in members}
    return out


# ── Helper: rewrite the mesh-selection store ───────────────────────


def _rewrite_mesh_selection(
    store: Any,
    *,
    offset: int,
    label: str,
) -> Any:
    """Return a fresh ``MeshSelectionStore`` with offset tag arrays and
    namespaced selection names.

    Returns ``None`` when ``store`` is ``None`` (mirrors the input
    nullable signal).
    """
    if store is None:
        return None
    from .MeshSelectionSet import MeshSelectionStore

    sets: dict = {}
    for (dim, tag), info in store._sets.items() if hasattr(store, "_sets") \
            else {}.items():
        new_info: dict = {}
        for k, v in info.items():
            if k == "name":
                new_info[k] = f"{label}.{v}"
            elif k in ("node_ids", "element_ids", "connectivity"):
                if v is None:
                    new_info[k] = v
                else:
                    arr = np.asarray(v, dtype=np.int64)
                    if arr.size == 0:
                        new_info[k] = arr
                    else:
                        new_info[k] = arr + np.int64(offset)
            else:
                new_info[k] = v
        sets[(dim, tag)] = new_info
    if not sets:
        return None
    return MeshSelectionStore(sets)


# ── Top-level rewrite entry point ──────────────────────────────────


def _rewrite_source_for_compose(
    source_path: "str | Path",
    *,
    label: str,
    translate: tuple[float, float, float],
    rotate: tuple[float, float, float, float] | None,
    partition_rank: int | None,
    properties: dict,
    base: int,
    size: int,
    source_span: int,
    source_min_tag: int,
) -> _RewrittenBundle:
    """Read the source H5, rewrite all IMPORT records, return a bundle.

    Pipeline (ADR 0038 §"Tag-offset scheme" + §"Tag-reference rewrite
    checklist" + §"Namespace rule"):

    1. ``read_fem_h5(source_path)`` → source :class:`FEMData`.
    2. Compute offset = ``base - source_min_tag``.
    3. For each registered record kind, apply ``tag_rewrite_spec``:

       * DISCARD verdict (``tag_rewrite_spec = None``):
         drop silently — PartitionRecord, ComposeRecord (deferred to
         3E.1's nested-provenance graft).
       * IMPORT verdict: offset every tag-bearing field; prefix every
         name-bearing field with ``"{label}."``.

    4. Apply geometric transform to node coordinates.
    5. Drop FILTER kinds silently — 3B.2b owns the
       :class:`ComposeFilterWarning` emission.

    The bundle is **read-only output**.  NEVER touches the host
    FEMData; NEVER calls into :class:`Compose` facade state; NEVER
    fires the verifier (3B.2c).
    """
    from ._femdata_h5_io import read_fem_h5
    from ._element_types import ElementGroup
    from .._kernel.records._partitions import PartitionRecord  # noqa: F401

    offset = base - source_min_tag
    source = read_fem_h5(str(source_path))

    # 1. Nodes — offset ids, apply geometric transform to coords.
    src_node_ids = np.asarray(source.nodes.ids, dtype=np.int64)
    src_node_coords = np.asarray(source.nodes.coords, dtype=np.float64)
    new_node_ids = src_node_ids + np.int64(offset)
    new_node_coords = _apply_geometric_transform(
        src_node_coords, translate=translate, rotate=rotate,
    )
    src_node_ndf = getattr(source.nodes, "_ndf", None)
    new_node_ndf = (
        np.asarray(src_node_ndf, dtype=np.int8)
        if src_node_ndf is not None
        else None
    )

    # 2. Elements — offset ids + connectivity per type.
    new_element_groups: dict = {}
    for type_code, group in source.elements._groups.items():
        old_ids = np.asarray(group.ids, dtype=np.int64)
        old_conn = np.asarray(group.connectivity, dtype=np.int64)
        new_ids = old_ids + np.int64(offset)
        new_conn = old_conn + np.int64(offset)
        new_element_groups[type_code] = ElementGroup(
            element_type=group.element_type,
            ids=new_ids,
            connectivity=new_conn,
        )

    # 3. Physical groups + labels — rewrite the dict-shaped name maps
    #    on both node-side and element-side composites.
    new_node_physical = _rewrite_named_groups(
        source.nodes.physical._groups, offset=offset, label=label,
    )
    new_elem_physical = _rewrite_named_groups(
        source.elements.physical._groups, offset=offset, label=label,
    )
    new_node_labels = _rewrite_named_groups(
        source.nodes.labels._groups, offset=offset, label=label,
    )
    new_elem_labels = _rewrite_named_groups(
        source.elements.labels._groups, offset=offset, label=label,
    )

    # 4. Mesh selections (optional).
    new_mesh_selection = _rewrite_mesh_selection(
        source.mesh_selection, offset=offset, label=label,
    )

    # 5. Parts maps — namespace the part_label keys + offset members.
    new_part_node_map = _rewrite_part_map(
        getattr(source.nodes, "_part_node_map", {}) or {},
        offset=offset, label=label,
    )
    new_part_elem_map = _rewrite_part_map(
        getattr(source.elements, "_part_elem_map", {}) or {},
        offset=offset, label=label,
    )

    # 6. Constraint / load / mass / SP records — apply tag_rewrite_spec.
    new_node_constraints = tuple(
        _rewrite_record(rec, offset=offset, label=label)
        for rec in source.nodes.constraints
    )
    new_elem_constraints = tuple(
        _rewrite_record(rec, offset=offset, label=label)
        for rec in source.elements.constraints
    )
    new_nodal_loads = tuple(
        _rewrite_record(rec, offset=offset, label=label)
        for rec in source.nodes.loads
    )
    new_element_loads = tuple(
        _rewrite_record(rec, offset=offset, label=label)
        for rec in source.elements.loads
    )
    new_sp_records = tuple(
        _rewrite_record(rec, offset=offset, label=label)
        for rec in source.nodes.sp
    )
    new_mass_records = tuple(
        _rewrite_record(rec, offset=offset, label=label)
        for rec in source.nodes.masses
    )

    # ── DISCARD / DEFER kinds (silently dropped per ADR 0038 §"Merge
    # semantics"; 3B.2b will emit warnings for FILTER kinds): the
    # module's own PartitionSet (line 168) and nested-compose
    # ComposeRecord (line 211, deferred to 3E.1).  ``read_fem_h5``
    # already populates these on the source FEMData; we deliberately
    # don't carry them through the bundle.  FILTER kinds (stages,
    # time-series, load-patterns, recorders, analysis settings,
    # /results/) are not part of the neutral zone ``read_fem_h5``
    # parses — they live in the OpenSees zone and never enter the
    # bundle in 3B.2a.

    composed_at = datetime.now(tz=timezone.utc).isoformat()

    return _RewrittenBundle(
        label=label,
        source_path=str(source_path),
        source_fem_hash=str(source.snapshot_id),
        source_neutral_schema_version=_read_neutral_schema_str(source_path),
        translate=translate,
        rotate=rotate,
        partition_rank=partition_rank,
        properties=dict(properties or {}),
        composed_at=composed_at,
        base=base,
        size=size,
        source_span=source_span,
        source_min_tag=source_min_tag,
        node_ids=new_node_ids,
        node_coords=new_node_coords,
        node_ndf=new_node_ndf,
        element_groups=new_element_groups,
        node_physical=new_node_physical,
        elem_physical=new_elem_physical,
        node_labels=new_node_labels,
        elem_labels=new_elem_labels,
        mesh_selection=new_mesh_selection,
        part_node_map=new_part_node_map,
        part_elem_map=new_part_elem_map,
        node_constraints=new_node_constraints,
        elem_constraints=new_elem_constraints,
        nodal_loads=new_nodal_loads,
        element_loads=new_element_loads,
        sp_records=new_sp_records,
        mass_records=new_mass_records,
    )


def _read_neutral_schema_str(source_path: "str | Path") -> str:
    """Best-effort: read ``/meta/@neutral_schema_version`` as a string.

    Used by :func:`_rewrite_source_for_compose` to populate the
    bundle's ``source_neutral_schema_version`` for the future
    :class:`ComposeRecord`.  Empty string when absent (very old files).
    """
    import h5py

    with h5py.File(str(source_path), "r") as f:
        if "meta" not in f:
            return ""
        attrs = f["meta"].attrs
        raw = attrs.get("neutral_schema_version", "")
        if isinstance(raw, (bytes, bytearray)):
            return raw.decode("utf-8", errors="replace")
        return str(raw)


# ---------------------------------------------------------------------------
# ComposedModule — live handle to one composed source module
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ComposedModule:
    """Live handle to one composed source module in the host session.

    Wraps a :class:`ComposeRecord` provenance entry plus an optional
    back-reference to the host :class:`FEMData` so the handle can later
    introspect the broker (PG inventory, label inventory, record counts
    contributed by this module).

    Phase 3B.1 ships the handle's identity surface (``label`` /
    ``source_path`` / ``translate`` / ``rotate`` / ``partition_rank``);
    the introspection methods (:meth:`pgs`, :meth:`labels`,
    :meth:`record_counts`) are stubbed pending the Phase 3B.2 merge
    engine that populates the ``module_label`` parallel datasets they
    walk.
    """

    record: "ComposeRecord"
    _fem: "FEMData | None" = field(default=None, repr=False, compare=False)

    # ── Identity passthroughs ───────────────────────────────────

    @property
    def label(self) -> str:
        """Namespace label assigned at compose time."""
        return self.record.label

    @property
    def source_path(self) -> str:
        """Path of the source ``model.h5`` that contributed this module."""
        return self.record.source_path

    @property
    def translate(self) -> tuple[float, float, float]:
        """XYZ translation applied at compose time."""
        return self.record.translate

    @property
    def rotate(self) -> tuple[float, float, float, float] | None:
        """Optional axis-angle rotation applied at compose time."""
        return self.record.rotate

    @property
    def partition_rank(self) -> int | None:
        """Layer-2 partition-rank hint per ADR 0038 §"Rank model"."""
        return self.record.partition_rank

    # ── Introspection (stubbed until Phase 3B.2) ────────────────

    def pgs(self) -> tuple[str, ...]:
        """PG names contributed by this module.

        Lands in Phase 3B.2 — needs the ``module_label`` parallel
        dataset populated by the merge engine.
        """
        raise NotImplementedError(
            "ComposedModule.pgs() needs the module_label parallel "
            "dataset populated by the merge engine; lands in Phase 3B.2."
        )

    def labels(self) -> tuple[str, ...]:
        """Label names contributed by this module.

        Lands in Phase 3B.2 — needs the ``module_label`` parallel
        dataset populated by the merge engine.
        """
        raise NotImplementedError(
            "ComposedModule.labels() needs the module_label parallel "
            "dataset populated by the merge engine; lands in Phase 3B.2."
        )

    def record_counts(self) -> dict[str, int]:
        """Per-record-kind counts contributed by this module.

        Lands in Phase 3B.2 — needs the ``module_label`` parallel
        dataset populated by the merge engine.
        """
        raise NotImplementedError(
            "ComposedModule.record_counts() needs the module_label "
            "parallel dataset populated by the merge engine; lands in "
            "Phase 3B.2."
        )


# ---------------------------------------------------------------------------
# Compose — session-level facade
# ---------------------------------------------------------------------------


class Compose:
    """Facade for compose-time model assembly per ADR 0038.

    Single per-session instance, exposed through the three session-level
    entry points :meth:`apeGmsh.compose`, :meth:`apeGmsh.compose_inspect`,
    and :meth:`apeGmsh.compose_list`.

    Phase 3B.1 (this PR) scaffolds the facade — input validation, the
    list / inspect helpers, exception types, the
    :data:`RESERVATION_GRANULARITY` knob, and the :class:`ComposedModule`
    handle.  The merge engine behind :meth:`compose` raises
    :class:`NotImplementedError` pending Phase 3B.2.
    """

    #: Reservation granularity for per-module tag windows per ADR 0038
    #: §"Tag-offset scheme — per-module auto-sizing".  Each compose call
    #: rounds the source's tag-span up to a multiple of this value when
    #: computing the host-side reservation.  Power-of-10 keeps the log
    #: messages human-readable; Phase 3B.2 uses it inside the merge
    #: engine.
    RESERVATION_GRANULARITY: int = 1_000_000

    def __init__(self, session: "apeGmsh") -> None:
        self._session = session

    # ── Public API ────────────────────────────────────────────────

    def compose(
        self,
        source: "str | Path",
        *,
        label: str,
        translate: tuple[float, float, float] = (0.0, 0.0, 0.0),
        rotate: tuple[float, float, float, float] | None = None,
        anchor: str | None = None,
        partition_rank: int | None = None,
        properties: "dict[str, Any] | None" = None,
        max_compose_depth: int = 3,
        compose_size_per_module: int | None = None,
    ) -> ComposedModule:
        """Merge a previously-saved apeGmsh model into the host session.

        Phase 3B.1 (this PR) validates inputs eagerly and raises
        :class:`NotImplementedError` before any H5 read or broker
        mutation; the merge engine ships in Phase 3B.2.

        Parameters
        ----------
        source : str | Path
            Path to the source ``model.h5`` (H5-only in v1 per ADR 0038
            §"g.compose() signature").
        label : str
            Namespace prefix assigned to every imported string-keyed
            record.  Required.  Must be non-empty, contain no ``.``,
            ``/`` or whitespace, and must not start or end with ``_``.
        translate, rotate : tuple
            Rigid-body placement of the module in the host's coordinate
            system.  ``rotate`` is axis-angle ``(x, y, z, theta)``.
        anchor : str | None
            PG-name sugar over ``translate``.  Mutually exclusive with a
            non-zero ``translate`` — see ADR 0038 §"g.compose() signature"
            line 104.
        partition_rank : int | None
            Layer-2 rank hint per ADR 0038 §"Rank model".  ``K >= 0``.
        properties : dict | None
            Free-form provenance dict round-tripped through
            ``/composed_from/{label}/properties`` on the host's next
            ``g.save()``.
        max_compose_depth : int, default 3
            Hard cap on nested-compose depth — raises
            :class:`ComposeDepthExceededError` from the verifier in
            Phase 3B.2.  No-op in 3B.1 (validation only; the engine
            stub fires first).
        compose_size_per_module : int | None
            Explicit reservation-size floor per ADR 0038 §"Tag-offset
            scheme".  ``None`` means "auto-size from the source's
            actual span".  Phase 3B.2 honours the override; 3B.1
            validates only ``> 0``.

        Returns
        -------
        ComposedModule
            Phase 3B.2 returns the live handle; in 3B.1 this method
            raises :class:`NotImplementedError` after validation.

        Raises
        ------
        ComposeLabelError
            ``label=`` violates the lexical rules.
        ComposeAnchorError
            ``anchor=`` combined with a non-zero ``translate=``.
        ValueError
            ``partition_rank < 0`` or ``compose_size_per_module <= 0``.
        NotImplementedError
            Always, after validation, until Phase 3B.2 wires the merge
            engine.
        """
        # Eager input validation — fail before any H5 read so misuse
        # surfaces at call time instead of half-way through the merge.
        self._validate_label(label)
        self._validate_translate_rotate_anchor(translate, anchor)
        self._validate_partition_rank(partition_rank)
        self._validate_compose_size(compose_size_per_module)
        # ``properties`` is exercised by the merge engine; ``max_compose_depth``
        # is accepted today but enforced by Phase 3E.1's nested-compose
        # verifier (no-op here per ADR 0038 §"Namespace rule").
        _ = max_compose_depth  # acknowledged, not enforced in 3B.2c

        parent = self._session

        # Ensure the session has a current ``_fem`` chain head.  On a
        # newly-begun session this triggers the canonical extraction
        # from gmsh + def lists; on a chain-phase session (built via
        # ``apeGmsh.from_h5``) ``_fem`` is already populated.
        if getattr(parent, "_fem", None) is None:
            parent._fem = parent.mesh.queries.get_fem_data()

        # Run the canonical transform.  ``FEMData.compose`` validates
        # again (cheap; keeps the primitive callable standalone) and
        # returns a new FEMData with the bundle merged in plus a
        # ``_last_compose_bundle`` attribute carrying the bundle for
        # replay.
        new_fem = parent._fem.compose(
            source,
            label=label,
            translate=translate,
            rotate=rotate,
            anchor=anchor,
            partition_rank=partition_rank,
            properties=properties,
            compose_size_per_module=compose_size_per_module,
        )

        # Update session state.
        parent._fem = new_fem
        bundle = getattr(new_fem, "_last_compose_bundle", None)
        if bundle is not None:
            existing = getattr(parent, "_compose_bundles", ())
            parent._compose_bundles = (*existing, bundle)

        # Bump + re-mark fresh so subsequent ``get_fem_data()`` calls
        # see the merged snapshot as the current chain head.
        if hasattr(parent, "_bump_fem_counter"):
            parent._bump_fem_counter()
        if hasattr(parent, "_mark_fem_fresh"):
            parent._mark_fem_fresh()

        # Provenance handle — the new ComposeRecord is the last entry
        # in the result's composed_from chain.  ``ComposeSet`` iterates
        # in ascending-label order, so look up by label rather than
        # index.
        new_record = new_fem.composed_from[label]
        return ComposedModule(record=new_record, _fem=new_fem)

    # ── Internal: rewrite half of the merge engine (Phase 3B.2a) ─

    def _rewrite_for_compose(
        self,
        source: "str | Path",
        *,
        label: str,
        translate: tuple[float, float, float] = (0.0, 0.0, 0.0),
        rotate: tuple[float, float, float, float] | None = None,
        partition_rank: int | None = None,
        properties: "dict | None" = None,
        compose_size_per_module: int | None = None,
    ) -> "_RewrittenBundle":
        """Internal: produce the rewritten bundle. Phase 3B.2b calls
        into this to obtain the offset/namespaced records before
        merging them into the host :class:`FEMData`.

        Validates inputs (delegating to the existing 3B.1 validators),
        reads the source H5 via :func:`_compute_source_span`, computes
        the reservation window against the current host FEMData (if
        any) plus the existing ``fem.composed_from`` reservations, and
        applies the rewrite pipeline.  Returns a :class:`_RewrittenBundle`
        — never mutates the host.
        """
        self._validate_label(label)
        # No anchor here — _rewrite_for_compose is the engine half;
        # ``anchor`` resolution belongs to the public ``compose(...)``
        # method which lands in 3B.2b.
        self._validate_partition_rank(partition_rank)
        self._validate_compose_size(compose_size_per_module)

        # Compute the source span (with the 2.8.x fallback path).
        source_span, source_min_tag, _source_max_tag = (
            _compute_source_span(source)
        )

        # Walk the host's current composed_from entries to seed the
        # previous-reservations list.  The bundle's ``base`` must sit
        # strictly above the host's tag watermark AND above any
        # already-issued module windows.  3B.2a does not write back
        # to fem.composed_from — 3B.2b owns the merge — but we still
        # honour the existing chain when present so the bundle's
        # base/size is self-consistent for 3B.2b to splice in.
        fem = self._current_fem()
        if fem is None:
            host_max_tag = 0
            previous: tuple[tuple[int, int], ...] = ()
        else:
            host_max_tag = _host_max_tag(fem)
            previous = _previous_reservations(fem)

        base, size = _compute_reservation(
            source_span=source_span,
            host_max_tag=host_max_tag,
            previous_reservations=previous,
            granularity=type(self).RESERVATION_GRANULARITY,
            compose_size_per_module=compose_size_per_module,
        )

        # Capacity check (ADR 0038 §"Tag-collision verifier" check 5).
        # Fires only when the caller supplied an explicit
        # ``compose_size_per_module`` smaller than the actual span.
        if compose_size_per_module is not None and source_span > size:
            raise ComposeCapacityError(
                f"compose(compose_size_per_module={compose_size_per_module}) "
                f"is smaller than the source's tag span ({source_span}); "
                f"reservation size {size} would not fit the imported tags."
            )

        return _rewrite_source_for_compose(
            source_path=source,
            label=label,
            translate=translate,
            rotate=rotate,
            partition_rank=partition_rank,
            properties=properties or {},
            base=base,
            size=size,
            source_span=source_span,
            source_min_tag=source_min_tag,
        )

    def compose_inspect(self, path: "str | Path") -> dict:
        """Read a module's H5 header without composing it.

        Returns a metadata-only summary — does NOT parse bulk record
        bodies.  ADR 0038 §"Companion helpers (v1)" line 121.

        Keys
        ----
        ``fem_hash`` : str
            ``snapshot_id`` from ``/meta``; empty string when absent.
        ``neutral_schema_version`` : str
            ``neutral_schema_version`` from ``/meta``; empty string
            when absent (e.g. very old files).
        ``tag_span_max`` : int
            ``tag_span_max`` from ``/meta``; ``0`` for pre-2.9.0 files
            that lacked the attribute.
        ``pg_inventory`` : tuple[str, ...]
            Sorted physical-group names.
        ``label_inventory`` : tuple[str, ...]
            Sorted label names.
        ``record_counts`` : dict[str, int]
            Counts for the major record kinds present on disk.
        ``composed_from`` : tuple[ComposeRecord, ...]
            ``ComposeRecord`` entries from ``/composed_from/`` (empty
            for uncomposed sources).
        ``properties`` : dict
            File-level properties slot.  ADR 0038 stores ``properties``
            per-module under each ``/composed_from/{label}/properties/``
            sub-group; the inspect-level entry is reserved for a future
            file-wide annotation surface and is ``{}`` today.
        """
        import h5py  # local import — keeps mesh package import-time light

        p = Path(path)
        with h5py.File(str(p), "r") as f:
            meta_attrs: dict[str, Any] = {}
            if "meta" in f:
                meta_attrs = dict(f["meta"].attrs)

            fem_hash = str(meta_attrs.get("snapshot_id", ""))
            neutral_schema_version = str(
                meta_attrs.get("neutral_schema_version", "")
            )
            tag_span_max = int(meta_attrs.get("tag_span_max", 0))

            pg_inventory = _read_named_group_inventory(f, "physical_groups")
            label_inventory = _read_named_group_inventory(f, "labels")
            record_counts = _read_record_counts(f)

            # ``_read_composed_from`` is the canonical reader for
            # ``/composed_from/`` and tolerates absence by returning ().
            from ._femdata_h5_io import _read_composed_from
            composed_from = _read_composed_from(
                f["composed_from"] if "composed_from" in f else None
            )

        return {
            "fem_hash": fem_hash,
            "neutral_schema_version": neutral_schema_version,
            "tag_span_max": tag_span_max,
            "pg_inventory": pg_inventory,
            "label_inventory": label_inventory,
            "record_counts": record_counts,
            "composed_from": composed_from,
            "properties": {},
        }

    def compose_list(self) -> tuple[ComposedModule, ...]:
        """Composed modules currently on the host session.

        Returns modules in compose-call order (the
        :class:`ComposeSet`'s ascending-label order, which matches the
        compose-order-independent canonicalisation of ADR 0038
        §"Lineage chain extension").  Empty tuple when no modules are
        composed or no FEM has been extracted yet.
        """
        fem = self._current_fem()
        if fem is None:
            return ()

        composed_from = getattr(fem, "composed_from", None)
        if not composed_from:
            return ()

        return tuple(
            ComposedModule(record=rec, _fem=fem) for rec in composed_from
        )

    # ── Defensive accessors ───────────────────────────────────────

    def _current_fem(self) -> "FEMData | None":
        """Best-effort fetch of the host session's current ``FEMData``.

        Returns ``None`` when no FEM has been extracted yet (e.g. the
        session was constructed but never ``begin()``-ed, or
        ``get_fem_data()`` would raise because gmsh has no mesh).
        Used by :meth:`compose_list` to gracefully degrade.

        Chain-phase sessions (built via :meth:`apeGmsh.from_h5`) carry
        ``_fem`` directly with no gmsh state behind them; fall back to
        that cached snapshot when ``mesh.queries`` is unavailable.
        """
        # Chain-phase short-circuit: prefer the cached chain head.
        cached = getattr(self._session, "_fem", None)
        if cached is not None and getattr(
            self._session, "_fem_from_h5", False,
        ):
            return cached
        try:
            return self._session.mesh.queries.get_fem_data()
        except Exception:
            # Intentionally swallow — compose_list is read-only and
            # must not blow up when called pre-mesh.  Real errors
            # surface elsewhere.
            return cached  # may still be None — caller handles it

    # ── Validators ────────────────────────────────────────────────

    @staticmethod
    def _validate_label(label: str) -> None:
        """Enforce ADR 0038 §"g.compose() signature" line 94 label rules.

        Required: non-empty string, no ``.`` (namespace separator), no
        ``/`` (depth-boundary separator), no whitespace, no leading or
        trailing ``_`` (reserved-prefix convention).
        """
        if not isinstance(label, str):
            raise ComposeLabelError(
                f"compose label must be a string, got {type(label).__name__}"
            )
        if not label:
            raise ComposeLabelError(
                "compose label must be non-empty per ADR 0038 "
                "§'g.compose() signature'."
            )
        if "." in label:
            raise ComposeLabelError(
                f"compose label {label!r} contains '.' (the namespace "
                "separator); ADR 0038 §'Namespace rule' reserves it."
            )
        if "/" in label:
            raise ComposeLabelError(
                f"compose label {label!r} contains '/' (the "
                "depth-boundary separator); ADR 0038 §'Namespace rule' "
                "reserves it."
            )
        if any(ch.isspace() for ch in label):
            raise ComposeLabelError(
                f"compose label {label!r} contains whitespace."
            )
        if label.startswith("_") or label.endswith("_"):
            raise ComposeLabelError(
                f"compose label {label!r} cannot start or end with '_'."
            )

    @staticmethod
    def _validate_translate_rotate_anchor(
        translate: tuple[float, float, float],
        anchor: str | None,
    ) -> None:
        """Enforce ADR 0038 line 104: ``anchor=`` and a non-zero
        ``translate=`` are mutually exclusive.
        """
        if anchor is None:
            return
        # Anchor is set — translate must be the identity.
        if any(float(x) != 0.0 for x in translate):
            raise ComposeAnchorError(
                f"compose() got anchor={anchor!r} together with a "
                f"non-zero translate={translate}; per ADR 0038 "
                "§'g.compose() signature' line 104 the two are "
                "mutually exclusive."
            )

    @staticmethod
    def _validate_partition_rank(partition_rank: int | None) -> None:
        """Enforce ADR 0038 §"Layer 2" line 420: ``K >= 0``.

        Phase 3B.1 surfaces a plain :class:`ValueError` because the
        constraint is a simple integer-range check; no compose-specific
        semantic context applies.  Callers catching :class:`ValueError`
        (which :class:`ComposeError` subclasses share) still see it.
        """
        if partition_rank is None:
            return
        if not isinstance(partition_rank, int) or isinstance(
            partition_rank, bool
        ):
            raise ValueError(
                f"compose(partition_rank=...) must be an int or None, "
                f"got {type(partition_rank).__name__}"
            )
        if partition_rank < 0:
            raise ValueError(
                f"compose(partition_rank={partition_rank}) must be "
                ">= 0 per ADR 0038 §'Rank model — Layer 2'."
            )

    @staticmethod
    def _validate_compose_size(compose_size_per_module: int | None) -> None:
        """Enforce ``compose_size_per_module > 0`` when supplied."""
        if compose_size_per_module is None:
            return
        if (
            not isinstance(compose_size_per_module, int)
            or isinstance(compose_size_per_module, bool)
        ):
            raise ValueError(
                "compose(compose_size_per_module=...) must be an int "
                "or None, got "
                f"{type(compose_size_per_module).__name__}"
            )
        if compose_size_per_module <= 0:
            raise ValueError(
                "compose(compose_size_per_module=...) must be > 0; "
                f"got {compose_size_per_module}."
            )


# ---------------------------------------------------------------------------
# Helpers used by compose_inspect
# ---------------------------------------------------------------------------


def _read_named_group_inventory(f: Any, group_name: str) -> tuple[str, ...]:
    """Return sorted names of a named-index group (PG / labels) or ()."""
    if group_name not in f:
        return ()
    parent = f[group_name]
    names: list[str] = []
    for key in parent.keys():
        sub = parent[key]
        if not hasattr(sub, "attrs"):
            continue
        name_attr = sub.attrs.get("name", key)
        names.append(
            name_attr.decode("utf-8")
            if isinstance(name_attr, (bytes, bytearray))
            else str(name_attr)
        )
    return tuple(sorted(names))


def _read_record_counts(f: Any) -> dict[str, int]:
    """Count rows in the major record kinds present on the file.

    Probes optional groups with ``in`` per the h5py optional-child
    ``.get()`` hazard (project_h5py_optional_child_get_hazard).
    """
    counts: dict[str, int] = {}

    if "nodes" in f and "ids" in f["nodes"]:
        counts["nodes"] = int(f["nodes/ids"].shape[0])
    else:
        counts["nodes"] = 0

    elements_total = 0
    if "elements" in f:
        for type_name in f["elements"].keys():
            sub = f["elements"][type_name]
            if hasattr(sub, "keys") and "ids" in sub:
                elements_total += int(sub["ids"].shape[0])
    counts["elements"] = elements_total

    for kind, dataset in (
        ("constraints", "constraints"),
        ("loads", "loads"),
        ("masses", "masses"),
    ):
        total = 0
        if dataset in f:
            node = f[dataset]
            if hasattr(node, "shape"):
                # Top-level dataset (e.g. /masses, /loads).
                total = int(node.shape[0])
            elif hasattr(node, "keys"):
                # Sub-grouped (e.g. /constraints/{kind} datasets).
                for child in node.keys():
                    sub = node[child]
                    if hasattr(sub, "shape"):
                        total += int(sub.shape[0])
        counts[kind] = total

    return counts


# ---------------------------------------------------------------------------
# Helpers used by Compose._rewrite_for_compose
# ---------------------------------------------------------------------------


def _host_max_tag(fem: "FEMData") -> int:
    """Return the max tag across the host's nodes + elements.

    Mirrors :func:`_compute_tag_span_max` in
    :mod:`apeGmsh.mesh._femdata_h5_io` but returns only the upper
    bound — that's what the reservation formula needs to compute the
    base for the first compose call.  Empty broker → 0.
    """
    maxs: list[int] = []
    node_ids = np.asarray(fem.nodes.ids, dtype=np.int64)
    if node_ids.size > 0:
        maxs.append(int(node_ids.max()))
    for group in fem.elements:
        ids = np.asarray(group.ids, dtype=np.int64)
        if ids.size > 0:
            maxs.append(int(ids.max()))
    return max(maxs) if maxs else 0


def _previous_reservations(
    fem: "FEMData",
) -> tuple[tuple[int, int], ...]:
    """Recover ``(base, size)`` pairs from the host's ``fem.composed_from``.

    The bundle does NOT yet carry the reservation extents on the
    :class:`ComposeRecord` (the schema bumps that lift them onto disk
    are a separate consideration); we approximate by reading the
    rewritten module's tag span out of the host broker.  This is good
    enough for 3B.2a's bundle production — 3B.2b will assemble the
    authoritative reservation chain when it stitches modules together.

    Returns an empty tuple when the host carries no composed modules.
    """
    composed = getattr(fem, "composed_from", None)
    if not composed:
        return ()
    # In 3B.2a there is no on-host ``base`` / ``size`` field on the
    # ComposeRecord.  The merge engine in 3B.2b will own that, so we
    # return an empty tuple here and let the first reservation just
    # round up from ``host_max_tag``.  Callers that need cumulative
    # tracking pass their own tuple via the rewrite engine in 3B.2b.
    return ()


# ---------------------------------------------------------------------------
# Merge engine (Phase 3B.2c / ADR 0038)
# ---------------------------------------------------------------------------


def _merge_bundle_into_fem(
    fem: "FEMData",
    bundle: "_RewrittenBundle",
) -> "FEMData":
    """Return a new :class:`FEMData` extending ``fem`` with ``bundle``.

    Pure transform — ``fem`` is unchanged; the returned :class:`FEMData`
    carries every record in both predecessors merged into a single
    snapshot.  The bundle's records were offset + namespace-rewritten
    by :func:`_rewrite_source_for_compose`, so there are no tag /
    PG-name collisions to resolve here (ADR 0038 §"Tag-offset scheme"
    + §"Namespace rule" guarantees that).

    Merge rules (ADR 0038 §"Merge semantics"):

    * Nodes / elements / PGs / labels / mesh-selections / parts /
      constraints / loads / masses / SP: concatenate (host first,
      bundle second).  PG / label dicts merge by key with no
      duplicates by construction (namespace rule).
    * ``fem.composed_from``: extended with a new :class:`ComposeRecord`
      built from the bundle's metadata.
    * ``module_label`` parallel arrays: host rows keep their existing
      label (empty string for host-owned rows, prior compose label
      for prior-composed rows); bundle's rows are stamped with the
      bundle's label.
    """
    from .FEMData import (
        FEMData, NodeComposite, ElementComposite, MeshInfo,
        _compute_bandwidth,
    )
    from ._element_types import ElementGroup
    from .._kernel.records._compose import ComposeRecord
    from .._kernel.record_sets import ComposeSet

    # ── 1. Concatenate node ids + coords ─────────────────────────
    host_node_ids = np.asarray(fem.nodes.ids, dtype=np.int64)
    host_node_coords = np.asarray(fem.nodes.coords, dtype=np.float64)
    bundle_node_ids = np.asarray(bundle.node_ids, dtype=np.int64)
    bundle_node_coords = np.asarray(bundle.node_coords, dtype=np.float64)
    new_node_ids = np.concatenate([host_node_ids, bundle_node_ids])
    new_node_coords = (
        np.concatenate([host_node_coords, bundle_node_coords], axis=0)
        if host_node_coords.size or bundle_node_coords.size
        else np.zeros((0, 3), dtype=np.float64)
    )

    # ── 2. Node ndf — concatenate when either side declares it ──
    host_ndf = getattr(fem.nodes, "_ndf", None)
    bundle_ndf = bundle.node_ndf
    if host_ndf is None and bundle_ndf is None:
        new_ndf: "np.ndarray | None" = None
    else:
        host_part = (
            np.asarray(host_ndf, dtype=np.int8)
            if host_ndf is not None
            else np.zeros(host_node_ids.shape, dtype=np.int8)
        )
        bundle_part = (
            np.asarray(bundle_ndf, dtype=np.int8)
            if bundle_ndf is not None
            else np.zeros(bundle_node_ids.shape, dtype=np.int8)
        )
        new_ndf = np.concatenate([host_part, bundle_part])

    # ── 3. Node module_label — host rows keep their label; bundle
    #      rows are stamped with the bundle's label.  Always allocate
    #      so the writer + ``ComposedModule.pgs()`` introspection
    #      have a deterministic shape after the first compose.
    host_ml = getattr(fem.nodes, "_module_label", None)
    if host_ml is None:
        host_ml_arr = np.array([""] * host_node_ids.size, dtype=object)
    else:
        host_ml_arr = np.asarray(host_ml, dtype=object)
    bundle_ml_arr = np.array(
        [bundle.label] * bundle_node_ids.size, dtype=object,
    )
    new_node_module_label = np.concatenate([host_ml_arr, bundle_ml_arr])

    # ── 4. Elements — merge per-type groups ─────────────────────
    new_element_groups: dict[int, ElementGroup] = dict(fem.elements._groups)
    # Build a parallel new module_label dict aligned with the result
    # groups.  Seed with what's already on the host (may be None).
    host_elem_ml = getattr(fem.elements, "_module_label", None)
    new_elem_module_label: dict[int, np.ndarray] = {}
    for code, host_group in new_element_groups.items():
        host_size = host_group.ids.size
        if host_elem_ml is not None and code in host_elem_ml:
            new_elem_module_label[code] = np.asarray(
                host_elem_ml[code], dtype=object,
            )
        else:
            new_elem_module_label[code] = np.array(
                [""] * host_size, dtype=object,
            )

    for code, bundle_group in bundle.element_groups.items():
        b_ids = np.asarray(bundle_group.ids, dtype=np.int64)
        b_conn = np.asarray(bundle_group.connectivity, dtype=np.int64)
        b_size = b_ids.size
        b_label_arr = np.array([bundle.label] * b_size, dtype=object)
        if code in new_element_groups:
            host_group = new_element_groups[code]
            merged_ids = np.concatenate([
                np.asarray(host_group.ids, dtype=np.int64), b_ids,
            ])
            merged_conn = np.concatenate([
                np.asarray(host_group.connectivity, dtype=np.int64),
                b_conn,
            ], axis=0)
            new_element_groups[code] = ElementGroup(
                element_type=host_group.element_type,
                ids=merged_ids,
                connectivity=merged_conn,
            )
            new_elem_module_label[code] = np.concatenate([
                new_elem_module_label[code], b_label_arr,
            ])
        else:
            # New type only present in the bundle.
            new_element_groups[code] = ElementGroup(
                element_type=bundle_group.element_type,
                ids=b_ids,
                connectivity=b_conn,
            )
            new_elem_module_label[code] = b_label_arr

    # ── 5. Physical groups + labels — merge dicts (namespace-prefixed
    #      keys / names guarantee no collisions per ADR 0038 §"Namespace
    #      rule")
    new_node_pgs = _merged_named_groups(
        fem.nodes.physical._groups, bundle.node_physical,
    )
    new_elem_pgs = _merged_named_groups(
        fem.elements.physical._groups, bundle.elem_physical,
    )
    new_node_labels = _merged_named_groups(
        fem.nodes.labels._groups, bundle.node_labels,
    )
    new_elem_labels = _merged_named_groups(
        fem.elements.labels._groups, bundle.elem_labels,
    )

    # ── 6. Mesh selection store — combine host's + bundle's ────
    new_mesh_selection = _merged_mesh_selection(
        fem.mesh_selection, bundle.mesh_selection,
    )

    # ── 7. Parts maps — merge dicts (keys namespace-prefixed) ──
    new_part_node_map: dict = dict(
        getattr(fem.nodes, "_part_node_map", {}) or {}
    )
    for k, v in (bundle.part_node_map or {}).items():
        new_part_node_map[k] = set(v)
    new_part_elem_map: dict = dict(
        getattr(fem.elements, "_part_elem_map", {}) or {}
    )
    for k, v in (bundle.part_elem_map or {}).items():
        new_part_elem_map[k] = set(v)

    # ── 8. Build the new FEMData with merged composites ────────
    from ._group_set import PhysicalGroupSet, LabelSet

    new_nodes = NodeComposite(
        node_ids=new_node_ids,
        node_coords=new_node_coords,
        physical=PhysicalGroupSet(new_node_pgs),
        labels=LabelSet(new_node_labels),
        constraints=list(fem.nodes.constraints),
        loads=list(fem.nodes.loads),
        sp=list(fem.nodes.sp),
        masses=list(fem.nodes.masses),
        partitions=getattr(fem.nodes, "_partitions", None) or None,
        part_node_map=new_part_node_map or None,
        ndf=new_ndf,
        module_label=new_node_module_label,
    )
    new_elements = ElementComposite(
        groups=new_element_groups,
        physical=PhysicalGroupSet(new_elem_pgs),
        labels=LabelSet(new_elem_labels),
        constraints=list(fem.elements.constraints),
        loads=list(fem.elements.loads),
        partitions=getattr(fem.elements, "_partitions", None) or None,
        part_elem_map=new_part_elem_map or None,
        module_label=new_elem_module_label,
    )

    # ── 9. Recompute MeshInfo so summary / bandwidth reflect the
    #      merged geometry.  Element-type infos are rebuilt from the
    #      merged groups (counts change).
    new_types = []
    for code, grp in new_element_groups.items():
        et = grp.element_type
        # ``ElementTypeInfo`` is a frozen dataclass; recreate with
        # the updated count.
        from ._element_types import make_type_info
        new_types.append(make_type_info(
            code=et.code, gmsh_name=et.gmsh_name, dim=et.dim,
            order=et.order, npe=et.npe, count=int(grp.ids.size),
        ))
    new_info = MeshInfo(
        n_nodes=int(new_node_ids.size),
        n_elems=int(sum(g.ids.size for g in new_element_groups.values())),
        bandwidth=_compute_bandwidth(new_element_groups),
        types=new_types,
    )

    # ── 10. Extend composed_from with a new ComposeRecord ──────
    new_record = ComposeRecord(
        label=bundle.label,
        source_path=bundle.source_path,
        source_fem_hash=bundle.source_fem_hash,
        source_neutral_schema_version=bundle.source_neutral_schema_version,
        translate=bundle.translate,
        rotate=bundle.rotate,
        partition_rank=bundle.partition_rank,
        composed_at=bundle.composed_at,
        properties=dict(bundle.properties),
    )
    existing_records = tuple(fem.composed_from) if fem.composed_from else ()
    new_composed_from = ComposeSet((*existing_records, new_record))

    new_fem = FEMData(
        nodes=new_nodes,
        elements=new_elements,
        info=new_info,
        mesh_selection=new_mesh_selection,
        composed_from=new_composed_from,
    )

    # ── 11. Now append the bundle's records via the with_*
    #       transforms.  This routes through the existing
    #       constraint / load / mass dispatch and avoids re-implementing
    #       the routing logic.  Cheap per-record clone of the
    #       FEMData (shallow copy + record-set replace) — fine for
    #       compose-time merges.
    for rec in bundle.node_constraints:
        new_fem = new_fem.with_constraint(rec)
    for rec in bundle.elem_constraints:
        new_fem = new_fem.with_constraint(rec)
    for rec in bundle.nodal_loads:
        new_fem = new_fem.with_load(rec)
    for rec in bundle.element_loads:
        new_fem = new_fem.with_load(rec)
    for rec in bundle.sp_records:
        new_fem = new_fem.with_load(rec)
    for rec in bundle.mass_records:
        new_fem = new_fem.with_mass(rec)

    return new_fem


def _merged_named_groups(
    host_groups: dict, bundle_groups: dict,
) -> dict:
    """Merge two ``{(dim, tag): info_dict}`` mappings.

    Bundle keys are guaranteed distinct from host keys by the
    namespace rule (ADR 0038 §"Namespace rule") — bundle ``info["name"]``
    has been prefixed with ``"{label}."`` so PG-name uniqueness is
    structural.  Key collisions on the ``(dim, tag)`` int pair are
    avoided because the bundle's rewriter offset the tags into the
    bundle's reservation window.
    """
    out = dict(host_groups)
    for key, info in bundle_groups.items():
        # If by some accident a (dim, tag) collision occurs (e.g. dim=0
        # tag=0 PG-name-only entry), bump the tag side by a large offset
        # so the host's entry survives.  This is defensive — the
        # namespace rule should make this branch unreachable in
        # practice.
        if key in out:
            d, t = key
            shifted_key = (d, t + 1_000_000_000)
            out[shifted_key] = info
        else:
            out[key] = info
    return out


def _merged_mesh_selection(host_store: Any, bundle_store: Any) -> Any:
    """Merge two :class:`MeshSelectionStore` instances into one.

    ``None`` on either side falls back to the other; both ``None``
    returns ``None``.  Both populated returns a fresh store with the
    union of their entries (namespace rule prevents name collisions
    on the bundle side).
    """
    if host_store is None and bundle_store is None:
        return None
    if host_store is None:
        return bundle_store
    if bundle_store is None:
        return host_store
    from .MeshSelectionSet import MeshSelectionStore

    merged: dict = {}
    if hasattr(host_store, "_sets"):
        merged.update(host_store._sets)
    if hasattr(bundle_store, "_sets"):
        for key, info in bundle_store._sets.items():
            # Defensive collision handling (same as _merged_named_groups).
            if key in merged:
                d, t = key
                merged[(d, t + 1_000_000_000)] = info
            else:
                merged[key] = info
    if not merged:
        return None
    return MeshSelectionStore(merged)


# ---------------------------------------------------------------------------
# FILTER-warning emission (Phase 3B.2c / ADR 0038)
# ---------------------------------------------------------------------------


def _emit_filter_warnings(source_path: "str | Path", label: str) -> None:
    """Emit one :class:`ComposeFilterWarning` per FILTER kind found.

    Reads the source H5's metadata (no bulk record reads) and emits a
    user-visible warning for each FILTER-verdict record kind present
    that is *not* silent.  Per ADR 0038 §"Merge semantics":

    =============== ==================================================
    FILTER kind     Warning?
    =============== ==================================================
    stages          yes — analysis-time, not inherited
    time-series     yes
    load-patterns   yes
    recorders       silent
    analysis        silent
    results         silent
    =============== ==================================================
    """
    import h5py

    p = Path(source_path)
    if not p.exists():
        return  # caller already handled missing-file; nothing to warn about.

    try:
        with h5py.File(str(p), "r") as f:
            # FILTER-verdict groups live in the ``/opensees/`` zone;
            # absent zone means no analysis content to filter and the
            # function silently returns.
            ops_zone = f.get("opensees") if "opensees" in f else None
            if ops_zone is None or not hasattr(ops_zone, "keys"):
                return
            # ``stages`` — STKO-style stage definitions
            if "stages" in ops_zone:
                stages = ops_zone["stages"]
                n = (
                    len(list(stages.keys()))
                    if hasattr(stages, "keys")
                    else 0
                )
                if n > 0:
                    warnings.warn(
                        f"module {label!r} carries {n} stages; "
                        f"stages are analysis-time and not inherited "
                        f"under compose. Re-declare on the host.",
                        ComposeFilterWarning,
                        stacklevel=3,
                    )
            # ``time_series`` — independent time-series defs
            if "time_series" in ops_zone:
                ts = ops_zone["time_series"]
                n = len(list(ts.keys())) if hasattr(ts, "keys") else 0
                if n > 0:
                    warnings.warn(
                        f"module {label!r} carries {n} time-series; "
                        f"time-series are analysis-time and not "
                        f"inherited under compose. Re-declare on the "
                        f"host.",
                        ComposeFilterWarning,
                        stacklevel=3,
                    )
            # ``patterns`` — load patterns
            if "patterns" in ops_zone:
                pats = ops_zone["patterns"]
                n = len(list(pats.keys())) if hasattr(pats, "keys") else 0
                if n > 0:
                    warnings.warn(
                        f"module {label!r} carries {n} load patterns; "
                        f"load patterns are analysis-time and not "
                        f"inherited under compose. Re-declare on the "
                        f"host.",
                        ComposeFilterWarning,
                        stacklevel=3,
                    )
    except (OSError, KeyError):
        # Read errors are non-fatal for the warning probe — the merge
        # engine has already loaded the IMPORT records successfully via
        # the rewrite step.
        return
