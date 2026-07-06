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
from ..core._compose_errors import (
    ComposeDepthExceededError as _CoreComposeDepthExceededError,
)
from ..core._compose_errors import ComposeInterfaceSizeWarning

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


class ComposeDepthExceededError(
    _CoreComposeDepthExceededError, ComposeError,
):
    """Nested-compose depth exceeds ``max_compose_depth`` per ADR 0038
    §"Nested composition".

    Phase 3E.1 wires the check into the merge engine.  The canonical
    error class lives at
    :class:`apeGmsh.core._compose_errors.ComposeDepthExceededError`;
    this facade-side subclass also inherits from :class:`ComposeError`
    so callers using ``except ComposeError`` continue to catch it
    alongside the other facade compose errors
    (:class:`ComposeLabelError`, :class:`ComposeAnchorError`,
    :class:`ComposeCapacityError`,
    :class:`ComposeNamespaceCollisionError`).

    Inheritance order: ``_CoreComposeDepthExceededError`` first so
    ``isinstance(err, apeGmsh.core._compose_errors.ComposeDepthExceededError)``
    holds for code outside the mesh package.
    """


class ComposeNamespaceCollisionError(ComposeError, ValueError):
    """Post-rewrite PG-name collision per ADR 0038 §"Tag-collision
    verifier" check 4.

    Raised by the verifier in Phase 3B.2; defined here so callers can
    write forward-compatible exception handlers.
    """


class ComposeReinforceCrossPartError(ComposeError, ValueError):
    """An embedded-reinforcement tie spans two source Parts (ADR 0067 P5.1).

    A ``LadrunoEmbeddedRebar`` tie inverse-maps a rebar node into the
    shape functions of ONE host element, so its ``rebar_node`` and
    ``host_nodes`` must live in the same Part. When a composed source's
    Part structure puts them in different Parts, the offset rewrite would
    silently produce a tie with broken conformal topology — so compose
    rejects it. (Unlike a tied-contact ``SurfaceCouplingRecord``, which is
    *designed* to bridge surfaces and may legitimately span Parts, an
    embedded tie's bond assumes a co-meshed host.) Recovery: flatten the
    source's Part structure, or keep each rebar within one Part.
    """


class ComposeFilterWarning(UserWarning):
    """One-line warning per filtered-with-warning record kind per
    compose call per ADR 0038 §"Merge semantics".

    Emitted by the merge engine in Phase 3B.2 for the stages /
    time-series / load-patterns kinds.  Defined here so the warning
    class lives with its sibling errors.
    """


# ---------------------------------------------------------------------------
# Nested composition — depth tracking + separator alternation (Phase 3E.1)
# ---------------------------------------------------------------------------
#
# ADR 0038 §"Nested composition" — three coupled rules for composing
# a source that is itself a composed assembly:
#
#   1. Depth limit: ``source_depth >= max_compose_depth`` raises
#      :class:`ComposeDepthExceededError`.  Default ``max_compose_depth = 3``.
#   2. Separator alternation: at depth N the outer namespace separator
#      flips ``.`` ↔ ``/`` so depth boundaries stay parseable.  Convention:
#      depth 1 = ``.``, depth 2 = ``/``, depth 3 = ``.``, ... (odd = ``.``,
#      even = ``/``).
#   3. Provenance graft: source's existing ``composed_from`` records get
#      their labels re-prefixed with the outer label (using the depth-N
#      separator) and surface in the host's flat ``composed_from`` chain.

#: Default cap on nested-compose depth per ADR 0038 §"Nested composition".
#: Class constant on :class:`Compose`; the public ``compose(...)`` method
#: also accepts ``max_compose_depth=N`` as a per-call override.
DEFAULT_MAX_COMPOSE_DEPTH: int = 3

#: Advisory threshold on a compose's interface size per ADR 0038
#: §"v1 scope gate" (Phase 3F.1).  The Phase 1 microbenchmark passed
#: at 10k x 4 ranks but breached emit / parse / RSS thresholds at
#: 100k x 8 ranks; the middle-branch decision was to ship compose at
#: full scope and emit a :class:`ComposeInterfaceSizeWarning` when the
#: caller's compose lands in the regime where downstream cross-rank
#: emit / parse cost may become a problem.
#:
#: "Interface size" here is the count of MP-style constraint records
#: in the rewritten bundle (node-side + element-side) — each replicates
#: across ranks per ADR 0027's cross-partition emission rules, so the
#: replication cost scales with the constraint count regardless of
#: rank count.  Constants in :data:`apeGmsh.mesh._compose` rather than
#: a sub-config because the threshold is a property of the Phase 1
#: gate, not user configuration.
WARN_INTERFACE_SIZE: int = 50_000

#: Separators allowed at compose-namespace boundaries.  The first entry
#: is the canonical "outer" separator at depth 1; the alternation rule
#: walks this tuple modulo ``len(_DEPTH_SEPARATORS)``.
_DEPTH_SEPARATORS: tuple[str, ...] = (".", "/")


def _separator_for_depth(depth: int) -> str:
    """Return the namespace separator for a compose at *result* depth.

    ADR 0038 §"Nested composition" — depth 1 (first compose) uses ``.``;
    depth 2 uses ``/``; depth 3 uses ``.``; ... and so on (odd → ``.``,
    even → ``/``).  The alternation makes nesting depth structurally
    visible to anyone parsing the joined label.

    Parameters
    ----------
    depth : int
        The result depth — i.e. ``source_depth + 1``.  Must be ``>= 1``;
        depth 0 (no compose at all) does not appear in any label and
        callers should not query the separator for it.

    Returns
    -------
    str
        ``"."`` when ``depth`` is odd, ``"/"`` when ``depth`` is even.
    """
    if depth < 1:
        raise ValueError(
            f"depth must be >= 1, got {depth} — depth 0 has no separator"
        )
    # depth 1 → index 0 → "."; depth 2 → index 1 → "/"; depth 3 → 0 → ".".
    return _DEPTH_SEPARATORS[(depth - 1) % len(_DEPTH_SEPARATORS)]


def _label_depth(label: str) -> int:
    """Return the compose depth encoded in a joined module label.

    A label without any separator is depth 1 (the leaf of a single
    compose); each additional separator (either ``.`` or ``/``) marks
    one more level of nesting.

    Examples
    --------
    >>> _label_depth("bolt")
    1
    >>> _label_depth("partA.bolt_head")
    2
    >>> _label_depth("assemblyM/partA.bolt_head")
    3
    >>> _label_depth("bayP.assemblyM/partA.bolt_head")
    4
    """
    if not label:
        return 0
    seps = sum(1 for ch in label if ch in _DEPTH_SEPARATORS)
    return seps + 1


def _compose_depth_of_records(records: "tuple[ComposeRecord, ...]") -> int:
    """Return the max compose depth across a ``composed_from`` tuple.

    Returns 0 when ``records`` is empty (an uncomposed FEMData).
    Otherwise the result is the max :func:`_label_depth` across the
    record labels — a flat-graft tuple's depth equals the highest
    joined-label depth in the chain.
    """
    if not records:
        return 0
    return max(_label_depth(rec.label) for rec in records)


def _join_module_label(outer: str, inner: str, *, result_depth: int) -> str:
    """Prefix ``inner`` with ``outer`` using the depth-N separator.

    Used by the provenance graft when ``outer`` is being composed
    around a source whose ``composed_from`` already carries
    ``inner``-labeled records.  The result's depth is ``result_depth``
    (``= _label_depth(inner) + 1`` for the grafted entry, or
    ``result_depth = 1 + source_depth`` for the new top-level entry
    when ``inner`` is empty / leaf-only).

    Parameters
    ----------
    outer : str
        The new top-level compose label assigned by the caller.
        Validated upstream to contain no separators.
    inner : str
        The pre-existing joined label from the source's
        ``composed_from``.  May contain mixed ``.`` / ``/`` separators
        from earlier composes.
    result_depth : int
        The depth at which the joined label sits in the resulting
        host's ``composed_from`` chain.  Drives the separator choice.

    Returns
    -------
    str
        ``"{outer}{sep}{inner}"`` where ``sep`` = depth-``result_depth``
        separator.  When ``inner`` is empty, returns ``outer``.
    """
    if not inner:
        return outer
    sep = _separator_for_depth(result_depth)
    return f"{outer}{sep}{inner}"


# ── Tree view (compose_tree) — derived from flat composed_from ─────


def _split_joined_label(label: str) -> tuple[str, ...]:
    """Inverse of :func:`_join_module_label`: split a joined label into
    its per-depth component tuple.

    A joined label is built by progressively joining components with
    the depth-N separator picked by :func:`_separator_for_depth`.  For
    a result-depth-N label, the LEFTMOST separator is at depth N (the
    outermost join), the next is at depth N-1, and so on down to
    depth 2 (the join between the last two components).  This helper
    walks the label left-to-right, validates that each separator
    matches the expected depth-N alternation, and returns the
    components as a tuple in outer-to-inner order.

    Examples
    --------
    >>> _split_joined_label("partA")
    ('partA',)
    >>> _split_joined_label("outer/inner")
    ('outer', 'inner')
    >>> _split_joined_label("top.assemblyM/partA")
    ('top', 'assemblyM', 'partA')

    Raises
    ------
    ComposeError
        When a separator at position k from the left does not match
        ``_separator_for_depth(N - k + 1)`` — i.e. the label was not
        produced by :func:`_join_module_label` and the alternation
        rule is violated.  This is the fail-loud signal the spec
        calls out (a label like ``"top/foo/bar"`` — `/` at the
        outermost depth where `.` is expected).
    """
    if not label:
        return ()
    n = _label_depth(label)
    if n == 1:
        return (label,)
    # Collect separator positions in left-to-right order.
    sep_positions = [
        i for i, ch in enumerate(label) if ch in _DEPTH_SEPARATORS
    ]
    # n - 1 == len(sep_positions) by construction of _label_depth.
    # Validate each separator's identity against the alternation rule.
    for k, idx in enumerate(sep_positions, start=1):
        expected_depth = n - k + 1
        expected_sep = _separator_for_depth(expected_depth)
        if label[idx] != expected_sep:
            raise ComposeError(
                f"compose_tree: joined label {label!r} violates the "
                f"separator-alternation rule at position {idx} "
                f"(separator {label[idx]!r} at depth {expected_depth}, "
                f"expected {expected_sep!r} per ADR 0038 §'Nested "
                f"composition'). The label was not produced by "
                f"_join_module_label and cannot be parsed."
            )
    # Split at the validated separator positions.
    components: list[str] = []
    prev = 0
    for idx in sep_positions:
        components.append(label[prev:idx])
        prev = idx + 1
    components.append(label[prev:])
    # Defensive: empty component slots indicate two adjacent separators
    # or a leading/trailing separator — fail-loud (cannot round-trip).
    for c in components:
        if not c:
            raise ComposeError(
                f"compose_tree: joined label {label!r} has an empty "
                f"component (adjacent separators or leading/trailing "
                f"separator); not a valid _join_module_label output."
            )
    return tuple(components)


@dataclass(frozen=True)
class ComposeTreeNode:
    """One node in a derived compose-tree view of ``fem.composed_from``.

    A :class:`ComposeTreeNode` reconstructs the nested-compose
    hierarchy from the flat-graft storage shipped by PR #369: each
    record in :attr:`FEMData.composed_from` carries its full joined
    label (``"outer.middle/inner"`` etc.) and surfaces as a top-level
    entry.  :meth:`FEMData.compose_tree` parses those joined labels
    via :func:`_split_joined_label` and returns a tuple of root
    nodes, each carrying its direct children recursively.

    Parameters
    ----------
    label : str
        The component name at this level — i.e. the original
        user-supplied ``label=`` from the compose call that produced
        this node.  For nested composes this is the leaf component
        (e.g. ``"partA"``), NOT the joined label
        (e.g. ``"outer/partA"``); the joined label lives on
        :attr:`record`.
    record : ComposeRecord
        The flat :class:`ComposeRecord` from ``fem.composed_from``
        corresponding to this node.  Its ``label`` is the joined
        joined label (full path from the root).
    children : tuple[ComposeTreeNode, ...]
        Direct child nodes — composes that were nested INSIDE this
        one at the next depth.  Empty tuple for leaf nodes.

    Notes
    -----
    Frozen and hashable — mirrors the rest of the compose package's
    frozen-dataclass conventions (:class:`ComposeRecord`,
    :class:`ComposedModule`, :class:`_RewrittenBundle`).
    """

    label: str
    record: "ComposeRecord"
    children: "tuple[ComposeTreeNode, ...]" = ()


def _build_compose_tree(
    records: "tuple[ComposeRecord, ...]",
) -> "tuple[ComposeTreeNode, ...]":
    """Build a tuple of root :class:`ComposeTreeNode` from a flat
    ``composed_from`` tuple.

    Each record's joined label is parsed via
    :func:`_split_joined_label` into per-depth components; the tree
    is assembled by inserting each (components, record) pair at the
    path described by the components.

    Per the flat-graft contract (PR #369), every non-leaf path-prefix
    also appears as its own top-level :class:`ComposeRecord` in the
    flat list (e.g. depth-3 host has ``["bayP", "bayP/assemblyM",
    "bayP.assemblyM/partA"]``), so the tree-builder finds a record at
    every node it constructs.

    Parameters
    ----------
    records : tuple[ComposeRecord, ...]
        Flat compose records, typically ``tuple(fem.composed_from)``.

    Returns
    -------
    tuple[ComposeTreeNode, ...]
        Root nodes in sorted (compose-label) order.  Empty tuple
        when ``records`` is empty (an uncomposed FEMData).
    """
    if not records:
        return ()
    # Index records by their joined label for O(1) lookup during the
    # depth-first tree build.  ComposeSet already enforces uniqueness
    # by joined label, but we accept any tuple here.
    by_label: dict[str, "ComposeRecord"] = {rec.label: rec for rec in records}

    # Group records by their depth-1 root component.  Walk every
    # record once, split it, and bucket by ``components[0]``.
    roots_to_descendants: dict[str, list[tuple[tuple[str, ...], "ComposeRecord"]]] = {}
    for rec in records:
        components = _split_joined_label(rec.label)
        # Defensive: _split_joined_label returned () only for the
        # empty-label case; ComposeLabelError forbids empty user labels
        # so this should be unreachable on valid records.
        if not components:
            raise ComposeError(
                f"compose_tree: record {rec!r} has an empty joined label; "
                "this should be unreachable per ComposeLabelError."
            )
        root = components[0]
        roots_to_descendants.setdefault(root, []).append((components, rec))

    def _build_subtree(
        prefix_components: tuple[str, ...],
    ) -> "ComposeTreeNode":
        """Build the subtree rooted at the record whose components
        equal ``prefix_components``."""
        # Reconstruct the joined label inside-out: start from the
        # innermost (leaf) component at depth 1, then wrap each outer
        # component using the depth-N separator from
        # :func:`_join_module_label`.  E.g. for components
        # ``("bayP", "assemblyM", "partA")``: start ``"partA"`` →
        # wrap with ``"assemblyM"`` at depth 2 → ``"assemblyM/partA"``
        # → wrap with ``"bayP"`` at depth 3 →
        # ``"bayP.assemblyM/partA"``.
        join_label = prefix_components[-1]
        for k in range(len(prefix_components) - 1, 0, -1):
            outer = prefix_components[k - 1]
            result_depth = len(prefix_components) - k + 1
            join_label = _join_module_label(
                outer, join_label, result_depth=result_depth,
            )
        record = by_label.get(join_label)
        if record is None:
            # Shouldn't happen given the flat-graft contract, but
            # surface it loudly if a record is missing — caller
            # passed a malformed records tuple.
            raise ComposeError(
                f"compose_tree: missing record for joined label "
                f"{join_label!r}; the flat-graft contract requires "
                f"every ancestor path to appear as a top-level "
                f"ComposeRecord."
            )
        # Find direct children: any record whose components are
        # exactly ``prefix_components + (next_comp,)`` for some next.
        prefix_len = len(prefix_components)
        child_components_seen: set[tuple[str, ...]] = set()
        for components, _rec in roots_to_descendants[prefix_components[0]]:
            if len(components) == prefix_len + 1 and (
                components[:prefix_len] == prefix_components
            ):
                child_components_seen.add(components)
        children = tuple(
            _build_subtree(c)
            for c in sorted(child_components_seen)
        )
        return ComposeTreeNode(
            label=prefix_components[-1],
            record=record,
            children=children,
        )

    roots = tuple(
        _build_subtree((root_name,))
        for root_name in sorted(roots_to_descendants.keys())
    )
    return roots


def _read_source_composed_from(source_path: "str | Path") -> tuple:
    """Best-effort read of a source H5's ``/composed_from/`` group.

    Returns an empty tuple when the source is uncomposed (no
    ``/composed_from/`` group on disk) or pre-2.9.0 (the group can't
    exist on those schemas).  Used by the depth check to validate
    nested-compose limits before any merge work runs.

    Probes optional children with ``in`` per the h5py optional-child
    ``.get()`` hazard (``project_h5py_optional_child_get_hazard``).
    """
    import h5py

    from ._femdata_h5_io import _read_composed_from

    p = Path(source_path)
    with h5py.File(str(p), "r") as f:
        if "composed_from" not in f:
            return ()
        return _read_composed_from(f["composed_from"])


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
    # Constraint / load / SP record lists.  Each entry is a dataclass
    # instance with offset-rewritten tag fields.  Masses are NOT a
    # record tuple here — they live columnar on ``mass_set`` below
    # (ADR 0065 v2 review hardening: the old eager
    # ``mass_records=tuple(mass_set)`` boxed one MassRecord per node on
    # every compose — ~GBs at multi-M nodes — and nothing consumed it).
    node_constraints: tuple
    elem_constraints: tuple
    nodal_loads: tuple
    element_loads: tuple
    sp_records: tuple
    # Nested provenance graft (Phase 3E.1 / ADR 0038 §"Nested
    # composition"): the source's own ``composed_from`` records,
    # each with its ``label`` already re-prefixed with the outer
    # compose label via :func:`_join_module_label` (so the merge
    # engine can splice them directly into the host's flat
    # ``composed_from`` chain).  Empty tuple when the source is
    # uncomposed (the depth-1 case).
    grafted_compose_records: tuple = ()
    # ADR 0065 v2 / plan_emit_memory_columnar.md C1–C3 — the columnar
    # rewritten masses (offset node_ids), the vectorized sibling of
    # the retired ``mass_records`` tuple.  The merge adopts these columns
    # directly instead of the O(N²) per-record ``with_mass`` append loop.
    # ``None`` only for a hand-built bundle with no masses.
    mass_set: "Any | None" = None
    # Pre-joined module_label arrays for nodes / elements (Phase 3E.1).
    # When the source is depth-0 (uncomposed) these are ``None`` and
    # the merge engine falls back to the simple "stamp every row with
    # ``bundle.label``" path.  When the source is nested, each row's
    # joined label is ``label`` for source-host-owned rows or
    # ``{label}{sep}{inner}`` for rows that came from a prior compose.
    node_module_label_joined: "np.ndarray | None" = None
    element_module_label_joined: "dict[int, np.ndarray] | None" = None
    # Per-node provenance (decoupled nodes — ADR 0049), aligned to
    # ``node_ids``.  ``None`` when the source has no decoupled nodes.
    node_provenance: "np.ndarray | None" = None
    # Embedded-reinforcement ties (ADR 0067 P5.1): the source's
    # ``elements.reinforce_ties`` (ReinforceTieRecord), offset-rewritten
    # via tag_rewrite_spec (rebar_node scalar + host_nodes array; name +
    # bond namespace-prefixed).  Empty when the source has no rebar.
    reinforce_ties: tuple = ()


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


def _prefix_namespaced_name(
    outer_label: str, inner_name: str | None,
) -> str | None:
    """Prefix a namespaced *name* (PG name, label, part label, …) with
    ``outer_label`` honouring the depth-N alternation rule (Phase
    3E.1 / ADR 0038 §"Nested composition").

    Convention for names that are *not* compose labels: an
    uncomposed leaf carries 0 separators; each prior compose adds
    one separator at the outer boundary.  The new outer prefix's
    separator is therefore at depth = ``(seps in inner) + 1``:

    * ``inner = "top_flange"`` (0 seps) → depth 1, sep ``.``,
      result ``"outer.top_flange"``.
    * ``inner = "conn_a.top_flange"`` (1 sep) → depth 2, sep ``/``,
      result ``"outer/conn_a.top_flange"``.
    * ``inner = "frame/conn_a.top_flange"`` (2 seps) → depth 3,
      sep ``.``, result ``"outer.frame/conn_a.top_flange"``.

    Note that compose *labels* (entries in ``fem.composed_from``)
    are a different namespace where the leaf is at depth 1 rather
    than depth 0; use :func:`_join_module_label` for those.

    Returns ``None`` unchanged when ``inner_name`` is ``None``; the
    caller decides whether to short-circuit.
    """
    if inner_name is None:
        return None
    inner = str(inner_name)
    seps = sum(1 for ch in inner if ch in _DEPTH_SEPARATORS)
    depth = seps + 1
    sep = _separator_for_depth(depth)
    return f"{outer_label}{sep}{inner}"


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

    Name prefixing honours the depth-N separator alternation rule
    (Phase 3E.1 / ADR 0038 §"Nested composition") via
    :func:`_prefix_namespaced_name` — a leaf name uses ``.``; a name
    that already carries one inner separator uses ``/`` at the outer
    boundary; and so on.

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
        # pattern name don't collide on the host.  Depth-N alternation
        # (Phase 3E.1) preserves the inner separator structure when
        # the source carries nested-compose names.
        changes[fname] = _prefix_namespaced_name(label, current)

    for fname in spec.get("nested_records", ()):
        current = getattr(rec, fname)
        if current is None:
            continue
        changes[fname] = [
            _rewrite_record(child, offset=offset, label=label)
            for child in current
        ]

    return _dc_replace(rec, **changes)


def _rewrite_mass_set(source_masses: Any, *, offset: int, label: str) -> Any:
    """Vectorized offset-rewrite of a columnar :class:`MassSet`.

    ADR 0065 v2 / plan_emit_memory_columnar.md C1–C3 — the columnar fast
    path for the mass leg of ``tag_rewrite_spec``.  ``MassRecord`` only has
    a scalar tag field (``node_id``) and a name field, so the rewrite is a
    single ``node_ids + offset`` array add plus a namespace-prefix on the
    (sparse) named rows — identical results to per-record
    :func:`_rewrite_record`, but without boxing one record per node.

    Falls back to the generic record path for any non-columnar mass set.
    """
    from apeGmsh._kernel.record_sets import MassSet

    node_ids_fn = getattr(source_masses, "node_ids", None)
    mass_arr_fn = getattr(source_masses, "mass_array", None)
    if not (callable(node_ids_fn) and callable(mass_arr_fn)):
        # Generic fallback — record-by-record (small / stub sets).
        return MassSet([
            _rewrite_record(rec, offset=offset, label=label)
            for rec in source_masses
        ])

    old_ids = np.asarray(node_ids_fn(), dtype=np.int64)
    new_ids = old_ids + np.int64(offset)
    # Mass values are copied verbatim (single copy) — untouched by the
    # tag rewrite, so deck float ``repr`` is preserved bit-for-bit.
    new_mass = np.array(mass_arr_fn(), dtype=np.float64, copy=True)
    old_names = getattr(source_masses, "_names", {}) or {}
    new_names: dict[int, str] = {
        int(i): str(_prefix_namespaced_name(label, str(nm)))
        for i, nm in old_names.items()
    }
    return MassSet(node_ids=new_ids, mass=new_mass, names=new_names)


def _concat_mass_sets(host_masses: Any, bundle_mass_set: Any) -> Any:
    """Concatenate host + bundle mass columns into one :class:`MassSet`.

    ADR 0065 v2 / plan_emit_memory_columnar.md C1–C3 — the columnar merge
    that replaces the O(N²) per-record ``with_mass`` append loop.  Host
    rows come first, then bundle rows (byte-identical ordering to the
    previous ``list(host) + [bundle…]`` append order).  Falls back to
    concatenating record lists when either side is non-columnar.
    """
    from apeGmsh._kernel.record_sets import MassSet

    host_ids_fn = getattr(host_masses, "node_ids", None)
    host_mass_fn = getattr(host_masses, "mass_array", None)
    b_ids_fn = getattr(bundle_mass_set, "node_ids", None)
    b_mass_fn = getattr(bundle_mass_set, "mass_array", None)

    if not (callable(host_ids_fn) and callable(host_mass_fn)
            and callable(b_ids_fn) and callable(b_mass_fn)):
        # Record-path fallback (stub sets or a legacy bundle without
        # ``mass_set``): rebuild from records, preserving host-then-bundle
        # order.
        recs = list(host_masses)
        if bundle_mass_set is not None:
            recs.extend(bundle_mass_set)
        return MassSet(recs)

    host_ids = np.asarray(host_ids_fn(), dtype=np.int64)
    host_mass = np.asarray(host_mass_fn(), dtype=np.float64)
    b_ids = np.asarray(b_ids_fn(), dtype=np.int64)
    b_mass = np.asarray(b_mass_fn(), dtype=np.float64)

    node_ids = np.concatenate([host_ids, b_ids])
    mass = np.concatenate([host_mass, b_mass], axis=0)

    # Names are sparse dicts keyed by row index — shift the bundle's keys
    # past the host block.
    host_names = dict(getattr(host_masses, "_names", {}) or {})
    off = len(host_ids)
    for i, nm in (getattr(bundle_mass_set, "_names", {}) or {}).items():
        host_names[off + i] = nm
    return MassSet(node_ids=node_ids, mass=mass, names=host_names)


def _guard_reinforce_cross_part(source: Any, ties: Any, *, label: str) -> None:
    """Raise :class:`ComposeReinforceCrossPartError` if any embedded tie's
    ``rebar_node`` + ``host_nodes`` span two different source Parts (ADR
    0067 P5.1) — the offset rewrite would otherwise produce a tie with
    broken conformal topology. No-op when the source has fewer than two
    Parts (no cross-Part possible) or carries no Part map (legacy source)."""
    if not ties:
        return
    part_map = getattr(source.nodes, "_part_node_map", None) or {}
    if len(part_map) < 2:
        return
    node_part: dict[int, str] = {}
    for pname, nids in part_map.items():
        for nid in nids:
            node_part[int(nid)] = pname
    for tie in ties:
        nodes = [int(tie.rebar_node), *(int(h) for h in tie.host_nodes)]
        # A node not in any named Part (e.g. a bare embedded rebar node that
        # was never assigned to a Part) is treated as unconstrained and does
        # NOT count toward the span — the guard fires only on >= 2 distinct
        # *named* Parts. This is deliberate: raising on no-Part nodes would
        # false-positive on legitimate single-Part-with-partial-map sources;
        # the real corruption case (host nodes split across two named Parts)
        # is still caught (adversarial-review C15).
        parts = {node_part[n] for n in nodes if n in node_part}
        if len(parts) > 1:
            raise ComposeReinforceCrossPartError(
                f"compose: reinforce tie at rebar node {tie.rebar_node} in "
                f"source {label!r} spans Parts {sorted(parts)} — its "
                f"rebar_node + host_nodes {list(tie.host_nodes)} are not in "
                f"one Part. An embedded tie's bond assumes a co-meshed host; "
                f"keep each rebar within one Part, or flatten the source's "
                f"Part structure."
            )


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
                # Phase 3E.1: depth-N alternation when ``v`` carries
                # inner separators from earlier composes; depth-1 ``.``
                # for leaf names (the common case).
                new_info[k] = _prefix_namespaced_name(label, v)
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

    Phase 3E.1: when a part_label already carries inner separators
    from a nested-compose source, the outer prefix uses the next-
    depth separator per :func:`_prefix_namespaced_name`.
    """
    out: dict = {}
    for plabel, members in part_map.items():
        new_label = _prefix_namespaced_name(label, plabel)
        if new_label is None:  # plabel was None (defensive)
            continue
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
                # Phase 3E.1 depth-N alternation.
                new_info[k] = _prefix_namespaced_name(label, v)
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
    max_compose_depth: int = DEFAULT_MAX_COMPOSE_DEPTH,
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

    # ── Nested composition (Phase 3E.1 / ADR 0038 §"Nested composition")
    #
    # 1. Compute source depth from the source's own ``composed_from``
    #    chain.  Fail-loud BEFORE any rewrite work runs if composing
    #    this source would push past ``max_compose_depth``.
    # 2. Compute the result depth + the depth-N separator used to glue
    #    the outer label onto inner (grafted) labels.
    # 3. Build the grafted ``ComposeRecord`` tuple — each source record
    #    gets its ``label`` re-prefixed with ``{outer_label}{sep}``.
    source_composed_from = tuple(source.composed_from) if (
        source.composed_from
    ) else ()
    source_depth = _compose_depth_of_records(source_composed_from)
    if source_depth >= max_compose_depth:
        raise ComposeDepthExceededError(
            f"compose(label={label!r}, source={str(source_path)!r}) would "
            f"exceed max_compose_depth={max_compose_depth}: source's own "
            f"compose depth is {source_depth} (max label depth in "
            f"source.composed_from), and composing it would create a "
            f"depth-{source_depth + 1} entry on the host. Lift the cap "
            f"with max_compose_depth=N or flatten the source via "
            f"re-baking before composing."
        )
    result_depth = source_depth + 1
    depth_sep = _separator_for_depth(result_depth)

    # Re-prefix the source's existing composed_from records.
    from .._kernel.records._compose import ComposeRecord as _ComposeRecord
    grafted_records = tuple(
        _ComposeRecord(
            label=_join_module_label(label, rec.label, result_depth=(
                _label_depth(rec.label) + 1
            )),
            source_path=rec.source_path,
            source_fem_hash=rec.source_fem_hash,
            source_neutral_schema_version=rec.source_neutral_schema_version,
            translate=rec.translate,
            rotate=rec.rotate,
            partition_rank=rec.partition_rank,
            composed_at=rec.composed_at,
            properties=dict(rec.properties) if rec.properties else {},
        )
        for rec in source_composed_from
    )

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
    src_node_prov = getattr(source.nodes, "_provenance", None)
    new_node_provenance = (
        np.asarray(src_node_prov, dtype=np.int8)
        if src_node_prov is not None
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
    # Masses: vectorized columnar offset-rewrite (ADR 0065 v2 /
    # plan_emit_memory_columnar.md C1–C3). ``new_mass_set`` carries the
    # remapped columns and is the ONLY mass channel on the bundle — no
    # record-tuple view is materialised (review hardening: the eager
    # ``tuple(new_mass_set)`` boxed every node's record on the compose
    # hot path and had no consumer).
    new_mass_set = _rewrite_mass_set(
        source.nodes.masses, offset=offset, label=label,
    )
    # Embedded-reinforcement ties (ADR 0067 P5.1): guard cross-Part ties
    # (broken conformal topology) BEFORE rewriting, then offset-rewrite
    # rebar_node + host_nodes and namespace-prefix the name + bond.
    source_ties = getattr(source.elements, "reinforce_ties", None) or ()
    _guard_reinforce_cross_part(source, source_ties, label=label)
    new_reinforce_ties = tuple(
        _rewrite_record(rec, offset=offset, label=label)
        for rec in source_ties
    )

    # 7. Joined module_label arrays (Phase 3E.1).  The source's
    #    per-row ``_module_label`` arrays carry inner labels from
    #    earlier composes; we re-prefix each non-empty inner with
    #    ``{label}{sep_for_inner_depth}{inner}``.  Empty source rows
    #    (source's own host content) become a plain ``{label}``
    #    stamp.  When the source is uncomposed (depth 0), the
    #    bundle leaves these as ``None`` so the merge engine takes
    #    its simple stamp-every-row-with-bundle.label path.
    node_module_label_joined: "np.ndarray | None" = None
    element_module_label_joined: "dict[int, np.ndarray] | None" = None
    if source_depth > 0:
        src_node_ml = getattr(source.nodes, "_module_label", None)
        if src_node_ml is not None:
            node_module_label_joined = _rewrite_module_labels(
                src_node_ml, outer_label=label,
            )
        # else: the source carries no nested module_label dataset on
        # nodes — falls through to the merge engine's default stamp.
        src_elem_ml = getattr(source.elements, "_module_label", None)
        if src_elem_ml:
            element_module_label_joined = {
                code: _rewrite_module_labels(arr, outer_label=label)
                for code, arr in src_elem_ml.items()
            }

    # ── DISCARD / DEFER kinds (silently dropped per ADR 0038 §"Merge
    # semantics"; 3B.2b will emit warnings for FILTER kinds): the
    # module's own PartitionSet (line 168) and nested-compose
    # ComposeRecord (line 211, now handled by 3E.1's provenance graft
    # via ``grafted_records`` above).  ``read_fem_h5`` already
    # populates these on the source FEMData; the PartitionSet is
    # deliberately not carried through the bundle.  FILTER kinds
    # (stages, time-series, load-patterns, recorders, analysis
    # settings, /results/) are not part of the neutral zone
    # ``read_fem_h5`` parses — they live in the OpenSees zone and
    # never enter the bundle in 3B.2a.

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
        node_provenance=new_node_provenance,
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
        mass_set=new_mass_set,
        grafted_compose_records=grafted_records,
        node_module_label_joined=node_module_label_joined,
        element_module_label_joined=element_module_label_joined,
        reinforce_ties=new_reinforce_ties,
    )


def _rewrite_module_labels(
    src_labels: np.ndarray,
    *,
    outer_label: str,
) -> np.ndarray:
    """Re-prefix an array of inner module labels with ``outer_label``.

    Phase 3E.1 / ADR 0038 §"Nested composition".  For each row:

    * Empty inner → ``outer_label`` (the source's host content becomes
      the new module-leaf).
    * Non-empty inner → ``{outer_label}{sep}{inner}`` where ``sep`` is
      the depth-N separator for the joined label's depth (``_label_depth
      (inner) + 1``).

    Returns a fresh ``object``-dtype ndarray; the input is not mutated.
    """
    out: list[str] = []
    for raw in src_labels:
        if isinstance(raw, (bytes, bytearray)):
            inner = raw.decode("utf-8", errors="replace")
        else:
            inner = str(raw) if raw is not None else ""
        if not inner:
            out.append(outer_label)
        else:
            depth = _label_depth(inner) + 1
            sep = _separator_for_depth(depth)
            out.append(f"{outer_label}{sep}{inner}")
    return np.array(out, dtype=object)


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

    #: Default cap on nested-compose depth per ADR 0038
    #: §"Nested composition".  Mirrors
    #: :data:`DEFAULT_MAX_COMPOSE_DEPTH` at module scope so callers
    #: can lift the cap class-wide (e.g. for a deep-hierarchy run)
    #: without passing ``max_compose_depth=`` on every call:
    #:
    #: .. code:: python
    #:
    #:     class MyCompose(Compose):
    #:         MAX_COMPOSE_DEPTH = 5
    #:
    #: Per-call ``max_compose_depth=N`` on :meth:`compose` overrides
    #: the class-level default for that single invocation.
    MAX_COMPOSE_DEPTH: int = DEFAULT_MAX_COMPOSE_DEPTH

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
        max_compose_depth: "int | None" = None,
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
        max_compose_depth : int or None, default None
            Per-call override of the class-level
            :data:`MAX_COMPOSE_DEPTH` (default 3).  Raises
            :class:`ComposeDepthExceededError` when the source's own
            ``composed_from`` depth would push the result past the
            cap.  ``None`` falls back to the class-level default; pass
            an explicit integer to lift the cap for a single compose
            call.  See ADR 0038 §"Nested composition" (Phase 3E.1).
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
        # ``properties`` is exercised by the merge engine;
        # ``max_compose_depth`` is forwarded to the rewriter, which
        # performs the nested-compose depth check before any rewrite
        # work (Phase 3E.1 / ADR 0038 §"Nested composition").  ``None``
        # falls back to the class-level :data:`MAX_COMPOSE_DEPTH` so
        # subclasses can override the default without touching every
        # call site.
        if max_compose_depth is None:
            max_compose_depth = type(self).MAX_COMPOSE_DEPTH
        if not isinstance(max_compose_depth, int) or isinstance(
            max_compose_depth, bool,
        ):
            raise ValueError(
                "compose(max_compose_depth=...) must be an int, got "
                f"{type(max_compose_depth).__name__}"
            )
        if max_compose_depth < 1:
            raise ValueError(
                "compose(max_compose_depth=...) must be >= 1, got "
                f"{max_compose_depth}"
            )

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
            max_compose_depth=max_compose_depth,
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
        max_compose_depth: "int | None" = None,
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

        # Phase 3E.1: resolve nested-compose cap from the kwarg or the
        # class-level :data:`MAX_COMPOSE_DEPTH` default.
        depth_cap = (
            max_compose_depth
            if max_compose_depth is not None
            else type(self).MAX_COMPOSE_DEPTH
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
            max_compose_depth=depth_cap,
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
        ``filtered`` : dict[str, int]
            The filtered-audit (ADR 0055 Phase 3 / ADR 0038 §195-196):
            warn-kind analysis content this source carries that
            ``g.compose`` would DROP with a
            :class:`ComposeFilterWarning` — ``{"stages": n,
            "time_series": n, "patterns": n}``, present-and-non-zero
            kinds only.  Empty dict when the source carries no
            droppable analysis content (or no ``/opensees/`` zone at
            all).  Mirrors the compose-time warning probe exactly, so
            a module can be audited BEFORE composing it.
        ``composed_from`` : tuple[ComposeRecord, ...]
            ``ComposeRecord`` entries from ``/composed_from/`` (empty
            for uncomposed sources).
        ``compose_tree`` : tuple[ComposeTreeNode, ...]
            Derived tree-view of the source's nested compose hierarchy,
            reconstructed from the flat ``composed_from`` list via the
            separator-alternation rule (depth-1 ``.``, depth-2 ``/``,
            ...). Empty tuple for uncomposed sources. Useful for
            previewing what ``g.compose_tree()`` would yield AFTER
            composing this source. Identical to
            :meth:`compose_tree` semantics — same builder, same
            ``ComposeTreeNode`` dataclass.
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

            # ADR 0055 Phase 3 / ADR 0038 §195-196: the filtered-audit —
            # which warn-kind analysis content this source carries that
            # compose would DROP (FILTER verdict).  Same counting the
            # compose-time warning probe uses, surfaced read-only so a
            # user can audit a module BEFORE composing it.
            filtered = _filter_kind_counts(f)

            # ``_read_composed_from`` is the canonical reader for
            # ``/composed_from/`` and tolerates absence by returning ().
            from ._femdata_h5_io import _read_composed_from
            composed_from = _read_composed_from(
                f["composed_from"] if "composed_from" in f else None
            )

        # Derived tree view of the source's compose hierarchy —
        # uses the same _build_compose_tree helper as g.compose_tree()
        # so users can preview the nested shape WITHOUT first composing.
        # Returns () for uncomposed sources (composed_from is empty).
        compose_tree = _build_compose_tree(composed_from)

        return {
            "fem_hash": fem_hash,
            "neutral_schema_version": neutral_schema_version,
            "tag_span_max": tag_span_max,
            "pg_inventory": pg_inventory,
            "label_inventory": label_inventory,
            "record_counts": record_counts,
            "filtered": filtered,
            "composed_from": composed_from,
            "compose_tree": compose_tree,
            "properties": {},
        }

    def compose_tree(self) -> "tuple[ComposeTreeNode, ...]":
        """Derived tree view of the host's nested-compose hierarchy.

        Reconstructs the nested-compose tree from the host's flat
        ``fem.composed_from`` chain.  Returns a tuple of root
        :class:`ComposeTreeNode` instances — each carrying its
        :class:`ComposeRecord` plus any direct children parsed from
        the joined labels via the separator-alternation rule
        (depth-1 ``.``, depth-2 ``/``, depth-3 ``.``, ...).

        Empty tuple for an uncomposed FEMData or a session that has
        not yet extracted a FEMData snapshot.

        See :meth:`FEMData.compose_tree` — the canonical primitive
        this session shim delegates to.  Companion to
        :meth:`compose_list` (flat-view) and :meth:`compose_inspect`
        (single-source-H5 view).

        Notes
        -----
        Storage stays flat per PR #369; the tree is a derived view,
        not a separate on-disk representation.  Joined labels parse
        unambiguously because :class:`ComposeLabelError` forbids
        ``.`` and ``/`` in user-supplied ``label=``, so every
        separator in a joined label sits at a known depth.
        """
        fem = self._current_fem()
        if fem is None:
            return ()
        return fem.compose_tree()

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
    """Return sorted names of a named-index group (PG / labels) or ().

    Neutral schema 2.10.0 (B2): the writer splits node-side and
    element-side entries into ``{group_name}/node_side/`` +
    ``{group_name}/element_side/`` sub-trees.  This helper walks both
    and returns the deduplicated union of names — compose-inspect's
    intent ("what named groups does this archive declare?") is
    side-agnostic.
    """
    if group_name not in f:
        return ()
    parent = f[group_name]
    names: set[str] = set()
    for side in ("node_side", "element_side"):
        if side not in parent:
            continue
        side_grp = parent[side]
        if not hasattr(side_grp, "keys"):
            continue
        for key in side_grp.keys():
            sub = side_grp[key]
            if not hasattr(sub, "attrs"):
                continue
            name_attr = sub.attrs.get("name", key)
            names.add(
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
    *,
    compose_size_per_module: int | None = None,
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

    Phase 3B.2d / ADR 0038 — verifier wiring.  Before any merge work
    runs, the five tag-collision verifier checks fire against the
    bundle's reservation + import data against the host's existing
    reservations + PG inventory.  Failures raise the typed exceptions
    :class:`~apeGmsh.core._compose_errors.PartTagCollisionError` /
    :class:`~apeGmsh.core._compose_errors.ComposeInvariantError` /
    :class:`~apeGmsh.core._compose_errors.ComposeCapacityError`.
    """
    from .FEMData import (
        FEMData, NodeComposite, ElementComposite, MeshInfo,
        _compute_bandwidth,
    )
    from ._element_types import ElementGroup
    from .._kernel.records._compose import ComposeRecord
    from .._kernel.record_sets import ComposeSet

    # ── 0. Verifier (Phase 2.2 wiring — ADR 0038 §"Tag-collision
    #      verifier") ──────────────────────────────────────────────
    _run_compose_verifier(
        fem=fem, bundle=bundle,
        compose_size_per_module=compose_size_per_module,
    )

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

    # ── 2b. Node provenance — decoupled nodes (ADR 0049).  Same
    #      concatenate-when-either-side-declares pattern as ndf.
    host_prov = getattr(fem.nodes, "_provenance", None)
    bundle_prov = bundle.node_provenance
    if host_prov is None and bundle_prov is None:
        new_provenance: "np.ndarray | None" = None
    else:
        host_prov_part = (
            np.asarray(host_prov, dtype=np.int8)
            if host_prov is not None
            else np.zeros(host_node_ids.shape, dtype=np.int8)
        )
        bundle_prov_part = (
            np.asarray(bundle_prov, dtype=np.int8)
            if bundle_prov is not None
            else np.zeros(bundle_node_ids.shape, dtype=np.int8)
        )
        new_provenance = np.concatenate([host_prov_part, bundle_prov_part])

    # ── 3. Node module_label — host rows keep their label; bundle
    #      rows are stamped with the bundle's label.  Always allocate
    #      so the writer + ``ComposedModule.pgs()`` introspection
    #      have a deterministic shape after the first compose.
    #      Phase 3E.1: when the bundle carries a pre-joined module
    #      label array (nested-compose case), use it instead of the
    #      flat stamp so depth-N rows inherit the joined inner label.
    host_ml = getattr(fem.nodes, "_module_label", None)
    if host_ml is None:
        host_ml_arr = np.array([""] * host_node_ids.size, dtype=object)
    else:
        host_ml_arr = np.asarray(host_ml, dtype=object)
    if bundle.node_module_label_joined is not None:
        bundle_ml_arr = np.asarray(
            bundle.node_module_label_joined, dtype=object,
        )
    else:
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
        # Phase 3E.1: use bundle's pre-joined element labels when
        # present (nested-compose case); fall back to flat-stamp.
        if (
            bundle.element_module_label_joined is not None
            and code in bundle.element_module_label_joined
        ):
            b_label_arr = np.asarray(
                bundle.element_module_label_joined[code], dtype=object,
            )
        else:
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

    # Masses — concatenate host + bundle columns in ONE shot (ADR 0065 v2 /
    # plan_emit_memory_columnar.md C1–C3). This replaces the previous
    # ``masses=list(host)`` boxing + O(N²) per-record ``with_mass`` append
    # loop; the constraint / load / sp legs still use the record path below.
    new_mass_set = _concat_mass_sets(fem.nodes.masses, bundle.mass_set)

    new_nodes = NodeComposite(
        node_ids=new_node_ids,
        node_coords=new_node_coords,
        physical=PhysicalGroupSet(new_node_pgs),
        labels=LabelSet(new_node_labels),
        constraints=list(fem.nodes.constraints),
        loads=list(fem.nodes.loads),
        sp=list(fem.nodes.sp),
        masses=new_mass_set,
        partitions=getattr(fem.nodes, "_partitions", None) or None,
        part_node_map=new_part_node_map or None,
        ndf=new_ndf,
        module_label=new_node_module_label,
        provenance=new_provenance,
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
        # ADR 0067 P5.1: carry the host's own ties + append the bundle's
        # rewritten ties (the with_* transforms below shallow-copy the
        # ElementComposite, preserving this list by reference).
        reinforce_ties=(list(getattr(fem.elements, "reinforce_ties", []))
                        + list(bundle.reinforce_ties)),
        # ADR 0067 P5.2 / B1a.2: preserve the HOST's auto-emitted rebar
        # elements across the merge (the rebuilt ElementComposite would
        # otherwise drop them). Carrying the SOURCE module's rebar_elements
        # (PG + material-name prefixing, parallel to the reinforce-tie carry
        # above) is a deferred compose teach-in follow-on.
        rebar_elements=list(getattr(fem.elements, "rebar_elements", [])),
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

    # ── 10. Extend composed_from with a new ComposeRecord +
    #        the grafted nested-provenance records (Phase 3E.1).
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
    grafted = tuple(bundle.grafted_compose_records or ())
    new_composed_from = ComposeSet(
        (*existing_records, new_record, *grafted),
    )

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
    # Masses were merged columnar into ``new_nodes`` above (ADR 0065 v2 /
    # plan_emit_memory_columnar.md C1–C3) — no per-record ``with_mass``
    # append loop here.

    return new_fem


def _rebuild_partitions_from_modules(fem: "FEMData") -> "FEMData":
    """Phase 3B.2d / ADR 0038 §"Rank model" — eager populator.

    Walks the host's :attr:`fem.composed_from` chain and writes a
    ``module_label → partition_rank`` map onto the host's node /
    element ``_partitions`` back-stores so the broker reports the
    composed modules as separate partitions.

    Three layers of rank assignment (ADR 0038 §"Rank model"):

    * **Layer 1 (default)** — auto-assign one rank per module in
      compose order: host nodes/elements land on rank 0, the
      first composed module on rank 1, the second on rank 2, etc.
      Activated when every ``ComposeRecord.partition_rank`` is
      ``None``.
    * **Layer 2 (hint)** — when one or more modules carry an
      explicit ``partition_rank=K`` hint on their ``ComposeRecord``,
      those ranks override the Layer-1 default for that module.
      Other modules get auto-assigned around the hints to avoid
      collisions (sequential lowest-unused integer).
    * **Layer 3 (METIS override)** — when the user later runs
      partition refinement that overrides module ranks, the caller
      is responsible for emitting a :class:`UserWarning` noting the
      override.  This populator emits the warning itself when it
      detects that ``fem`` already carries explicit per-node /
      per-element partition assignments AND a Layer-1/2 module map
      would disagree.

    Pure-ish helper: ``fem`` is not deep-copied — the partition
    back-stores are replaced in place on the existing composite
    instances + a new :class:`PartitionSet` is built on the result.
    Returns a new :class:`FEMData` carrying the rebuilt partitions
    so callers can chain transforms.
    """
    composed = getattr(fem, "composed_from", None)
    if not composed:
        return fem

    # Layer 1 / 2 — assign each module a rank.
    rank_by_label: dict[str, int] = {}
    used: set[int] = {0}  # host always owns rank 0
    # First pass — honour Layer-2 hints.
    for rec in composed:
        if rec.partition_rank is not None:
            r = int(rec.partition_rank)
            if r in used:
                # Collision between two Layer-2 hints OR between a hint
                # and the host rank.  Raise a clear error rather than
                # silently overriding — ADR 0038 §"Rank model".
                raise ValueError(
                    f"rank model: module {rec.label!r} has "
                    f"partition_rank={r} which collides with another "
                    f"module's hint or the host's rank-0 reservation."
                )
            used.add(r)
            rank_by_label[rec.label] = r
    # Second pass — auto-assign the unhinted modules from the lowest
    # unused integer.
    auto_cursor = 1
    for rec in composed:
        if rec.label in rank_by_label:
            continue
        while auto_cursor in used:
            auto_cursor += 1
        rank_by_label[rec.label] = auto_cursor
        used.add(auto_cursor)
        auto_cursor += 1

    # Layer 3 — detect METIS override.  We only warn when the existing
    # partition record on the host carries IDs that this compose's
    # rank assignment does NOT cover (i.e. the user ran METIS at some
    # point AND the partition IDs don't match the rank set the
    # populator would assign).  A prior compose's partition record
    # uses {0, rank_by_label...} which by construction matches; only a
    # user-driven METIS run would produce a different set.
    existing_node_parts = getattr(fem.nodes, "_partitions", {}) or {}
    existing_elem_parts = getattr(fem.elements, "_partitions", {}) or {}
    if existing_node_parts or existing_elem_parts:
        expected = {0, *rank_by_label.values()}
        existing = (
            set(existing_node_parts.keys())
            | set(existing_elem_parts.keys())
        )
        if not existing.issubset(expected):
            import warnings
            warnings.warn(
                "compose rank model: overriding existing partition "
                "assignment with one rank per composed module per "
                "ADR 0038 §'Rank model — Layer 3'. Re-run partition "
                "refinement (g.mesh.partitioning.partition) AFTER "
                "compose to keep a METIS-driven partition.",
                UserWarning,
                stacklevel=2,
            )

    # ── Build the partition dicts ────────────────────────────────
    node_ids = np.asarray(fem.nodes.ids, dtype=np.int64)
    node_ml_arr = (
        np.asarray(fem.nodes._module_label, dtype=object)
        if fem.nodes._module_label is not None
        else np.array([""] * node_ids.size, dtype=object)
    )

    new_node_parts: dict[int, dict] = {0: {"node_ids": [], "element_ids": []}}
    for r in rank_by_label.values():
        new_node_parts[r] = {"node_ids": [], "element_ids": []}

    for i, nid in enumerate(node_ids):
        lbl = str(node_ml_arr[i]) if i < node_ml_arr.size else ""
        r = rank_by_label.get(lbl, 0)
        new_node_parts[r]["node_ids"].append(int(nid))

    new_elem_parts: dict[int, dict] = {0: {"node_ids": [], "element_ids": []}}
    for r in rank_by_label.values():
        new_elem_parts[r] = {"node_ids": [], "element_ids": []}

    elem_ml = getattr(fem.elements, "_module_label", None) or {}
    for code, grp in fem.elements._groups.items():
        eids = np.asarray(grp.ids, dtype=np.int64)
        ml_arr = (
            np.asarray(elem_ml.get(code), dtype=object)
            if elem_ml.get(code) is not None
            else np.array([""] * eids.size, dtype=object)
        )
        for i, eid in enumerate(eids):
            lbl = str(ml_arr[i]) if i < ml_arr.size else ""
            r = rank_by_label.get(lbl, 0)
            new_elem_parts[r]["element_ids"].append(int(eid))

    # Convert lists → int64 arrays for the broker's contract.
    for r, info in new_node_parts.items():
        info["node_ids"] = np.array(info["node_ids"], dtype=np.int64)
        info["element_ids"] = np.array(info["element_ids"], dtype=np.int64)
    for r, info in new_elem_parts.items():
        info["element_ids"] = np.array(info["element_ids"], dtype=np.int64)
        info["node_ids"] = np.array(info["node_ids"], dtype=np.int64)

    # Mutate the existing composites in place (private back-stores
    # are documented as the broker's source of truth for partitions).
    fem.nodes._partitions = new_node_parts
    fem.elements._partitions = new_elem_parts

    # Rebuild the public PartitionSet by constructing a fresh
    # :class:`FEMData` — its ``__init__`` reads the partition dicts
    # straight off the composites' ``_partitions`` back-stores, so
    # the returned snapshot exposes the just-assigned ranks via
    # ``fem.partitions``.  Pass mesh_selection + composed_from + info
    # through unchanged so the chain head is otherwise intact.
    from .FEMData import FEMData as _FEMData
    return _FEMData(
        nodes=fem.nodes,
        elements=fem.elements,
        info=fem.info,
        mesh_selection=fem.mesh_selection,
        composed_from=fem.composed_from,
    )


def _run_compose_verifier(
    *,
    fem: "FEMData",
    bundle: "_RewrittenBundle",
    compose_size_per_module: int | None,
) -> None:
    """Run the 5 ADR 0038 tag-collision checks for one bundle merge.

    Pulls reservations + import data from the host FEMData + the
    rewritten bundle and dispatches to
    :func:`apeGmsh.core._tag_collision_verifier.tag_collision_verify`.

    Reservation reconstruction
    --------------------------
    Each :class:`~apeGmsh._kernel.records._compose.ComposeRecord` does
    not yet carry on-disk ``(base, size)`` extents — the schema can
    grow that in a follow-up — so previously-composed modules'
    reservations are reconstructed as ``[host_max_tag + 1, total]``
    sweeps over the host's ``module_label`` arrays.  This is a
    conservative approximation: the verifier's check 2 (disjoint
    reservations) is the only check that uses the historical
    reservations, and reconstructing from the actual occupied tag
    range gives the same disjointness verdict the writer's offset
    formula did at the time.
    """
    from ..core._tag_collision_verifier import (
        ImportedRecords,
        ReservationRecord,
        tag_collision_verify,
    )

    # New reservation — the bundle's own (base, size) pair.
    new_reservation = ReservationRecord(
        label=bundle.label, base=bundle.base, size=bundle.size,
    )
    # Reconstruct prior reservations from the host's composed_from
    # chain.  When the chain is empty (first compose) this is the
    # only reservation handed to the verifier.
    prior: list[ReservationRecord] = []
    composed = getattr(fem, "composed_from", None)
    if composed:
        for prev_rec in composed:
            base, size = _reconstruct_reservation_for_label(
                fem, prev_rec.label,
            )
            if size > 0:
                prior.append(ReservationRecord(
                    label=prev_rec.label, base=base, size=size,
                ))
    reservations: tuple[ReservationRecord, ...] = (*prior, new_reservation)

    # Host PG name inventory — node-side + element-side.
    host_pg_names = _collect_pg_names(fem)

    # Imports — the bundle's rewritten tag inventory.
    imported_tags: list[int] = []
    imported_tags.extend(int(x) for x in bundle.node_ids)
    for grp in bundle.element_groups.values():
        imported_tags.extend(int(x) for x in grp.ids)
    # Source-side PG name inventory before namespacing (so check 4's
    # ``{label}.``-prefix can run uniformly).
    source_pg_names: list[str] = []
    for entry in bundle.node_physical.values():
        n = entry.get("name", "")
        # Bundle entries already carry namespace-prefixed names
        # (``_rewrite_named_groups`` applied ``{label}.`` to them);
        # strip the prefix for the verifier so check 4 reproduces
        # the prefix itself.
        if isinstance(n, str) and n.startswith(f"{bundle.label}."):
            source_pg_names.append(n[len(bundle.label) + 1:])
        elif isinstance(n, str):
            source_pg_names.append(n)
    for entry in bundle.elem_physical.values():
        n = entry.get("name", "")
        if isinstance(n, str) and n.startswith(f"{bundle.label}."):
            source_pg_names.append(n[len(bundle.label) + 1:])
        elif isinstance(n, str):
            source_pg_names.append(n)

    constraint_refs = tuple(
        _bundle_constraint_refs(bundle)
    )

    module_imports = {
        bundle.label: ImportedRecords(
            tags=imported_tags,
            pg_names=source_pg_names,
            constraint_refs=constraint_refs,
            source_span=bundle.source_span,
        ),
    }

    tag_collision_verify(
        reservations=reservations,
        host_pg_names=host_pg_names,
        module_imports=module_imports,
        compose_size_per_module=compose_size_per_module,
    )


def _reconstruct_reservation_for_label(
    fem: "FEMData", label: str,
) -> tuple[int, int]:
    """Return ``(base, size)`` covering the tag range owned by ``label``.

    Walks the host's ``module_label`` parallel arrays on nodes +
    elements, gathers every tag stamped with ``label``, and computes
    ``(min, max - min + 1)``.  Returns ``(0, 0)`` when ``label`` owns
    no rows (defensive — empty module).
    """
    mins: list[int] = []
    maxs: list[int] = []
    node_ml = getattr(fem.nodes, "_module_label", None)
    if node_ml is not None:
        node_ml_arr = np.asarray(node_ml, dtype=object)
        mask = np.array(
            [str(x) == label for x in node_ml_arr], dtype=bool,
        )
        if mask.any():
            node_ids = np.asarray(fem.nodes.ids, dtype=np.int64)[mask]
            if node_ids.size > 0:
                mins.append(int(node_ids.min()))
                maxs.append(int(node_ids.max()))
    elem_ml = getattr(fem.elements, "_module_label", None)
    if elem_ml is not None:
        for code, grp in fem.elements._groups.items():
            arr = elem_ml.get(code)
            if arr is None:
                continue
            ml_arr = np.asarray(arr, dtype=object)
            mask = np.array(
                [str(x) == label for x in ml_arr], dtype=bool,
            )
            if mask.any():
                eids = np.asarray(grp.ids, dtype=np.int64)[mask]
                if eids.size > 0:
                    mins.append(int(eids.min()))
                    maxs.append(int(eids.max()))
    if not mins:
        return 0, 0
    base = min(mins)
    size = max(maxs) - base + 1
    return base, size


def _collect_pg_names(fem: "FEMData") -> tuple[str, ...]:
    """Gather every PG name from the host (node + element side)."""
    out: list[str] = []
    for entry in fem.nodes.physical._groups.values():
        n = entry.get("name", "")
        if isinstance(n, str) and n:
            out.append(n)
    for entry in fem.elements.physical._groups.values():
        n = entry.get("name", "")
        if isinstance(n, str) and n:
            out.append(n)
    return tuple(out)


def _bundle_constraint_refs(bundle: "_RewrittenBundle"):
    """Yield :class:`ConstraintReference` for every tag-bearing field on
    every constraint record in ``bundle``.

    Iterates ``tag_rewrite_spec`` to find which fields are tags so the
    verifier can confirm each lands inside the bundle's reservation
    window (check 3 — cover-set drift detection).
    """
    from ..core._tag_collision_verifier import ConstraintReference

    record_streams = (
        bundle.node_constraints,
        bundle.elem_constraints,
    )
    for stream in record_streams:
        for rec in stream:
            spec = getattr(rec, "tag_rewrite_spec", None)
            if spec is None:
                continue
            kind = type(rec).__name__
            for fname in spec.get("tag_fields_scalar", ()):
                val = getattr(rec, fname, None)
                if val is None:
                    continue
                yield ConstraintReference(
                    kind=kind, field_name=fname, tag=int(val),
                )
            for fname in spec.get("tag_fields_array", ()):
                vals = getattr(rec, fname, None)
                if vals is None:
                    continue
                for i, v in enumerate(vals):
                    yield ConstraintReference(
                        kind=kind, field_name=f"{fname}[{i}]",
                        tag=int(v),
                    )


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


#: Warn-kind FILTER groups under ``/opensees/``: H5 child name →
#: human noun used in the :class:`ComposeFilterWarning` message.
#: Recorders / analysis / results are FILTER-silent and deliberately
#: absent.  Shared by the compose-time warning probe and the
#: ``compose_inspect`` ``filtered`` audit (ADR 0055 Phase 3).
_FILTER_WARN_KINDS: "tuple[tuple[str, str], ...]" = (
    ("stages", "stages"),
    ("time_series", "time-series"),
    ("patterns", "load patterns"),
)


def _filter_kind_counts(f: "Any") -> "dict[str, int]":
    """Count the warn-kind FILTER groups present in an open source H5.

    Returns ``{h5_child_name: n_children}`` for each
    :data:`_FILTER_WARN_KINDS` group present with at least one child;
    empty dict when the ``/opensees/`` zone is absent or carries no
    droppable analysis content.  Metadata-only (no bulk record reads).
    """
    counts: "dict[str, int]" = {}
    ops_zone = f.get("opensees") if "opensees" in f else None
    if ops_zone is None or not hasattr(ops_zone, "keys"):
        return counts
    for child, _noun in _FILTER_WARN_KINDS:
        if child in ops_zone:
            grp = ops_zone[child]
            n = len(list(grp.keys())) if hasattr(grp, "keys") else 0
            if n > 0:
                counts[child] = n
    return counts


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
            counts = _filter_kind_counts(f)
    except (OSError, KeyError):
        # Read errors are non-fatal for the warning probe — the merge
        # engine has already loaded the IMPORT records successfully via
        # the rewrite step.
        return
    nouns = dict(_FILTER_WARN_KINDS)
    for child, n in counts.items():
        noun = nouns[child]
        warnings.warn(
            f"module {label!r} carries {n} {noun}; "
            f"{noun} are analysis-time and not inherited "
            f"under compose. Re-declare on the host.",
            ComposeFilterWarning,
            stacklevel=3,
        )


# ---------------------------------------------------------------------------
# Interface-size advisory (Phase 3F.1 / ADR 0038 §"v1 scope gate")
# ---------------------------------------------------------------------------


def _count_interface_size(bundle: "_RewrittenBundle") -> int:
    """Return the count of MP-style constraint records in ``bundle``.

    Phase 3F.1 / ADR 0038 §"v1 scope gate".  The gate matrix
    (``tests/benchmarks/test_cross_rank_constraint_cost.py``) measured
    emit / parse cost as a function of an ``interface_size`` parameter
    that counts the embedded interpolation constraints in the
    fixture.  At compose time the same quantity is the count of
    constraint records carried by the rewritten bundle — each of
    those records will, at emit time, fan out across ranks per
    ADR 0027's cross-partition emission rules (``equalDOF``,
    ``rigidLink``, ``rigidDiaphragm``, ``embeddedNode``,
    ``mp_constraint_comment``), so the cost scales with the
    constraint count regardless of rank count.

    Counts the node-side + element-side constraint streams on the
    bundle.  Other interface-class proxies (interface node count, PG
    member counts) are correlated but indirect; the constraint count
    matches the gate fixture's framing exactly.
    """
    return len(bundle.node_constraints) + len(bundle.elem_constraints)


def _warn_interface_size(
    bundle: "_RewrittenBundle",
    *,
    threshold: "int | None" = None,
) -> None:
    """Emit a :class:`ComposeInterfaceSizeWarning` when ``bundle`` sits
    above the advisory threshold.

    Phase 3F.1 / ADR 0038 §"v1 scope gate".  Predicate is
    strictly-greater-than so the threshold value itself does NOT
    trip the warning — only counts that exceed it.  ``threshold=None``
    disables the warning entirely (escape hatch for callers who have
    already accepted the cost).

    Emits at most one warning per call regardless of how many
    constraint records the bundle carries.
    """
    if threshold is None:
        return
    count = _count_interface_size(bundle)
    if count <= threshold:
        return
    warnings.warn(
        f"compose(label={bundle.label!r}): module carries {count} "
        f"interface-class constraints (> {threshold}); downstream "
        f"cross-rank emit / parse cost may dominate per ADR 0038 "
        f"§\"v1 scope gate\". Consider splitting the source into "
        f"smaller modules, or silence with warnings.simplefilter("
        f"\"ignore\", ComposeInterfaceSizeWarning).",
        ComposeInterfaceSizeWarning,
        stacklevel=3,
    )
