"""
Labels — geometry-time entity naming via internal physical groups.
==================================================================

Accessed via ``g.labels``.  Labels are human-readable names for
geometry entities that survive slicing, boolean operations, and
the STEP round-trip.  They are backed by Gmsh physical groups
whose names carry an internal ``_label:`` prefix so they are
invisible to solver-facing code (``g.physical``, ``fem.physical``,
the OpenSees exporter).

Two-tier naming
---------------
* **Labels** (``g.labels``) — geometry-time bookkeeping.
  Created automatically when ``label=`` is passed to any
  ``g.model.geometry.add_*`` method.  Used for addressing
  entities by name during boolean ops, slicing, and Part
  composition.  NOT visible to the solver.

* **Physical groups** (``g.physical``) — solver-time naming.
  Created explicitly by the user for boundary conditions,
  materials, and loads.  Visible to ``fem.physical`` and the
  OpenSees exporter.

The user promotes a label to a physical group when they are
ready to expose it to the solver::

    # Build and slice
    g.model.geometry.add_box(0, 0, 0, 1, 1, 3, label="shaft")
    g.model.geometry.slice("shaft", axis='z', offset=1.5)

    # Promote to a solver-facing PG
    tags = g.labels.entities("shaft")
    g.physical.add_volume(tags, name="column_shaft")

Boolean safety
--------------
Both label PGs and user PGs survive every OCC boolean operation
(``fragment``, ``fuse``, ``cut``, ``intersect``).  Two module-level
functions — :func:`snapshot_physical_groups` and
:func:`remap_physical_groups` — implement a snapshot-then-remap
pattern that every boolean call site wraps around the OCC call.
Users never need to re-resolve labels after a boolean.

Naming convention
-----------------
At the Gmsh level, a label named ``"shaft"`` is stored as a
physical group with name ``"_label:shaft"``.  The prefix is
stripped by every method on this composite so the user never
sees it.  ``g.physical`` filters OUT any PG whose name starts
with ``_label:`` so the two tiers do not interfere.
"""
from __future__ import annotations

import warnings
from contextlib import contextmanager
from typing import TYPE_CHECKING, Iterator

import gmsh
import numpy as np

from apeGmsh._logging import _HasLogging
from apeGmsh._types import DimTag

if TYPE_CHECKING:
    from apeGmsh._types import SessionProtocol as _SessionBase

# The internal ``_label:`` prefix predicates were RELOCATED to
# apeGmsh._kernel._label_prefix (selection-unification-v2 P1-K, HT4):
# core/Labels.py is NOT wholesale-relocatable (it eagerly ``import
# gmsh`` and hosts a gmsh-driven class), but these four helpers are
# pure ``str`` logic.  Re-exported here via a downward ``core`` ->
# ``_kernel`` edge (the intended layering direction) so the in-file
# uses below, ``from apeGmsh.core.Labels import add_prefix`` /
# ``is_label_pg`` / ``strip_prefix`` / ``LABEL_PREFIX``, and the
# byte-unchanged contract/pin tests keep resolving.  Flagged as a
# P3/P4 internal-cleanup candidate.
from apeGmsh._kernel._label_prefix import (  # noqa: F401
    LABEL_PREFIX,
    add_prefix,
    is_label_pg,
    strip_prefix,
)


# =====================================================================
# Boolean-safe PG preservation
# =====================================================================
#
# Gmsh's OCC boolean operations (fragment, fuse, cut, intersect)
# destroy physical group membership after synchronize().  These two
# functions implement a snapshot-then-remap pattern that every boolean
# call site must wrap around the OCC call:
#
#     snap = snapshot_physical_groups()
#     result, result_map = gmsh.model.occ.fragment(obj, tool, ...)
#     gmsh.model.occ.synchronize()
#     remap_physical_groups(snap, obj + tool, result_map)
#
# Or use the context manager for simpler call sites:
#
#     with pg_preserved() as pg:
#         result, result_map = gmsh.model.occ.fragment(obj, tool, ...)
#         pg.set_result(obj + tool, result_map)
# =====================================================================


# ─────────────────────────────────────────────────────────────────────
# Geometric signature matching
# ─────────────────────────────────────────────────────────────────────
#
# OCC's ``outDimTagsMap`` maps inputs to outputs at the input dim
# (volumes -> volumes for a 3D cut).  Sub-entity lineage — which
# pre-op face/edge/vertex became which post-op one — is **not**
# exposed by the gmsh Python API, so the snapshot/remap pattern alone
# cannot track sub-entities through a boolean.  The two failure modes
# from this gap were "Cannot remap (sub-topology renumbering)" and
# "is now empty" warnings for line/surface PGs after a cut.
#
# ─── Escape hatch if this approach hits its limits ──────────────────
# Gmsh's ``OCC_Internals::booleanOperator`` (gmsh source,
# ``src/geo/GModelIO_OCC.cpp``) already calls OCC's
# ``BRepAlgoAPI::Modified()/Generated()/IsDeleted()`` on every input
# shape — it just iterates top-level inputs only (no
# ``TopExp_Explorer`` descent into sub-shapes), and the builder is
# stack-local so we can't query it post-hoc.  Two paths to recover
# the real OCC sub-shape lineage if/when the geometric heuristic
# becomes the bottleneck:
#
# 1. **Upstream patch (preferred).**  ~120 LOC adding a TopExp
#    descent in each boolean ``case`` and a sidecar
#    ``OCC_Internals::_lastBooleanLineage`` exposed via
#    ``gmsh.model.occ.getLastBooleanLineage()``.  The data is
#    already computed; only the binding is missing.  File the issue
#    on gitlab.onelab.info/gmsh/gmsh with the patch sketch.
#
# 2. **pythonocc-core sidecar (fallback if upstream rejects).**
#    Run the boolean through ``pythonocc-core`` in parallel,
#    harvest ``Modified()/Generated()/IsDeleted()`` per sub-shape,
#    map back to gmsh tags via the same geometric signatures.
#    Closes the cut-interface gap (``Generated()`` returns those
#    explicitly).  Caveat: gmsh internally runs ``SimplifyResult()``
#    after every boolean (collinear-edge consolidation, coplanar-
#    face merge) which pythonocc won't see — the two BReps can
#    diverge on cleanup ops.  Mitigation: replace gmsh's boolean
#    call with pythonocc's entirely and re-import the result, OR
#    accept the divergence on the small fraction of ops where
#    simplification fires.
#
# Effort estimates: option 1 is ~8-16h if upstream accepts;
# option 2 is ~2-3 days including the dependency story.  Don't
# implement either pre-emptively — the geometric heuristic below
# handles the common case (splits, merges, simple cuts) cleanly.
#
# These helpers add a geometric fallback: capture a tag-agnostic
# signature (bbox + centroid + dim-specific orientation) for every
# PG entity pre-op, and after the boolean match each pre-op signature
# against current entities at the same dim by bidirectional
# centroid-in-bbox containment plus a direction/normal-parallel
# check.  Handles splits, simple cuts, fragments, and merges
# generically.  See the module docstring for the failure modes that
# remain (new entities introduced by the cut interface, fully-
# consumed entities, coincident-then-ambiguous geometry).

# Angular tolerance for direction/normal parallelism — cos of ~5°.
_PARALLEL_COS_TOL = 0.9962


def _model_tolerance() -> float:
    """Geometric tolerance scaled to the live model's bounding box.

    ``1e-7 * model_diag`` matches OCC's default precision setting
    and is comfortably above the floating-point noise OCC adds when
    rebuilding a BRep across a boolean.  Falls back to ``1e-6`` for
    an empty model or a getBoundingBox failure (rare; some Gmsh
    builds raise on a fresh model before any entity exists).
    """
    try:
        bb = gmsh.model.getBoundingBox(-1, -1)
        diag = max(bb[3] - bb[0], bb[4] - bb[1], bb[5] - bb[2])
        if diag > 0:
            return max(1e-9, diag * 1e-7)
    except Exception:
        pass
    return 1e-6


def _entity_signature(dim: int, tag: int) -> dict | None:
    """Compute a tag-agnostic geometric signature for an entity.

    Returns ``None`` when the entity is not in the model (caller
    must call this on a known-live tag, typically pre-op).

    Signature shape::

        {
            'bbox':      (xmin, ymin, zmin, xmax, ymax, zmax),
            'centroid':  (cx, cy, cz),
            'direction': (dx, dy, dz) | None,   # dim=1 only
            'normal':    (nx, ny, nz) | None,   # dim=2 only
            'kind':      str,                   # gmsh.model.getType
            'mass':      float | None,          # length/area/volume
        }

    ``kind`` is the OCC type name (``'Line'``, ``'Circle'``,
    ``'Plane'``, ``'Cylinder'``, ``'BSpline curve'``, etc.).
    Cheap categorical filter that rejects matches across surface
    types (a planar face cannot match a cylindrical one no matter
    how their bboxes overlap).

    ``mass`` is length for 1D, area for 2D, volume for 3D.  Used
    as a *post-match sanity check* (Σ child masses should be
    close to the parent mass under a clean split or no-op) rather
    than a per-pair rejection — preserves matches when masses
    drift slightly under OCC simplification, and surfaces a
    warning when they drift badly.
    """
    try:
        bb = gmsh.model.getBoundingBox(dim, tag)
    except Exception:
        return None
    bbox = tuple(float(v) for v in bb)

    sig: dict = {
        'bbox': bbox,
        'direction': None,
        'normal': None,
        'kind': '',
        'mass': None,
    }

    try:
        sig['kind'] = str(gmsh.model.getType(dim, tag))
    except Exception:
        sig['kind'] = ''

    if dim >= 1:
        try:
            m = gmsh.model.occ.getMass(dim, tag)
            if m is not None:
                sig['mass'] = float(m)
        except Exception:
            sig['mass'] = None

    if dim == 0:
        try:
            pos = gmsh.model.getValue(0, tag, [])
        except Exception:
            return None
        sig['centroid'] = (float(pos[0]), float(pos[1]), float(pos[2]))
        return sig

    # For dim>=1, use OCC's center of mass; fall back to bbox center
    # if the entity is degenerate enough that getCenterOfMass throws.
    try:
        com = gmsh.model.occ.getCenterOfMass(dim, tag)
        sig['centroid'] = (float(com[0]), float(com[1]), float(com[2]))
    except Exception:
        sig['centroid'] = (
            0.5 * (bbox[0] + bbox[3]),
            0.5 * (bbox[1] + bbox[4]),
            0.5 * (bbox[2] + bbox[5]),
        )

    if dim == 1:
        # Endpoint-to-endpoint direction.  Curved edges may have a
        # direction at their endpoints that doesn't match the local
        # tangent at the midpoint — for the matching rule, what
        # matters is that pre-op and post-op halves of the SAME curve
        # share the same endpoint chord direction, which they do
        # under any well-behaved OCC split.
        try:
            bnd = gmsh.model.getBoundary(
                [(1, tag)], oriented=False, recursive=False,
            )
            pts = [b for b in bnd if int(b[0]) == 0]
            if len(pts) >= 2:
                p0 = gmsh.model.getValue(0, int(pts[0][1]), [])
                p1 = gmsh.model.getValue(0, int(pts[-1][1]), [])
                d = (p1[0] - p0[0], p1[1] - p0[1], p1[2] - p0[2])
                n = (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]) ** 0.5
                if n > 1e-12:
                    sig['direction'] = (d[0] / n, d[1] / n, d[2] / n)
        except Exception:
            pass
    elif dim == 2:
        # Face normal at the parametric midpoint.  ``getNormal`` wants
        # a flat list of ``[u0, v0, u1, v1, ...]`` so we pass the
        # midpoint of the parameter bounds.  Some surface types
        # (trimmed, periodic) may not honour this; on failure the
        # signature leaves normal=None and the matching rule falls
        # back to bbox + centroid alone.
        try:
            pb = gmsh.model.getParametrizationBounds(2, tag)
            u_mid = 0.5 * (pb[0][0] + pb[1][0])
            v_mid = 0.5 * (pb[0][1] + pb[1][1])
            nrm = gmsh.model.getNormal(tag, [u_mid, v_mid])
            n = (
                nrm[0] * nrm[0] + nrm[1] * nrm[1] + nrm[2] * nrm[2]
            ) ** 0.5
            if n > 1e-12:
                sig['normal'] = (nrm[0] / n, nrm[1] / n, nrm[2] / n)
        except Exception:
            pass

    return sig


# The compatibility rule (bidirectional centroid-in-bbox + kind
# equality + direction/normal parallelism within ~5°) is implemented
# vectorized inside ``remap_physical_groups._geometric_match``.
# The matching axes:
#
# * Splits: child centroid lies inside parent bbox (one direction).
# * Merges: parent centroid lies inside child bbox (the other
#   direction).  Captures fuse and fragment merge-products.
# * ``kind`` (gmsh.model.getType): hard categorical filter — a
#   planar face cannot match a cylindrical one regardless of bbox
#   overlap.  Empty ``kind`` on either side skips the check (don't
#   reject when OCC didn't report a type).
# * Direction (1D) or normal (2D): cos(angle) >= ``_PARALLEL_COS_TOL``
#   when both sides have it; skipped when either is missing.


class _PGPreserver:
    """Collects boolean result info and remaps PGs on context exit."""

    __slots__ = (
        '_snap', '_input_dts', '_result_map', '_result', '_absorbed',
        '_skip_names',
    )

    def __init__(self, snap: list[dict]) -> None:
        self._snap = snap
        self._input_dts: list[DimTag] | None = None
        self._result_map: list[list[DimTag]] | None = None
        self._result: list[DimTag] | None = None
        self._absorbed = False
        self._skip_names: set[str] = set()

    def set_result(
        self,
        input_dimtags: list[DimTag],
        result_map: list[list[DimTag]],
        *,
        result: list[DimTag] | None = None,
        absorbed_into_result: bool = False,
    ) -> None:
        self._input_dts = input_dimtags
        self._result_map = result_map
        self._result = result
        self._absorbed = absorbed_into_result

    def skip(self, names: set[str] | list[str]) -> None:
        """Mark PG names to bypass during the post-op remap.

        The caller takes ownership: snapshot entries with these names
        are silently dropped (no warning, no recreate) on context
        exit.  Used by Part-side topology-rebuild hooks (e.g. DRM
        box line PGs) that re-derive these PGs from a stored
        predicate after the boolean has finished — the snapshot data
        would be stale and the standard remap can't track sub-entity
        renumbering for them anyway.
        """
        self._skip_names.update(names)


@contextmanager
def pg_preserved_identity() -> Iterator[None]:
    """Snapshot PGs on entry, recreate them on exit with the same tags.

    For OCC operations that preserve entity tags but still wipe
    physical-group membership on ``synchronize()`` — transforms
    (translate, rotate, scale, mirror), ``copy``, and the sweep ops
    (extrude, revolve) where the input entities survive unchanged.

    Use :func:`pg_preserved` instead for boolean ops, which need
    ``result_map``-driven remapping.
    """
    snap = snapshot_physical_groups()
    yield
    remap_physical_groups(snap, [], [])


@contextmanager
def pg_preserved() -> Iterator[_PGPreserver]:
    """Context manager that snapshots PGs on entry and remaps on exit.

    Usage::

        with pg_preserved() as pg:
            result, result_map = gmsh.model.occ.fragment(obj, tool, ...)
            pg.set_result(obj + tool, result_map, result=result)
        # PGs are automatically remapped here
    """
    snap = snapshot_physical_groups()
    ctx = _PGPreserver(snap)
    yield ctx
    if ctx._input_dts is not None and ctx._result_map is not None:
        remap_physical_groups(
            ctx._snap, ctx._input_dts, ctx._result_map,
            result=ctx._result,
            absorbed_into_result=ctx._absorbed,
            skip_pg_names=ctx._skip_names,
        )


def snapshot_physical_groups() -> list[dict]:
    """Capture every physical group (both ``_label:*`` and user PGs).

    Call this **before** any OCC boolean + synchronize sequence.

    Each entry carries per-entity geometric signatures
    (``entity_signatures``) so the post-op remap can fall back to
    geometric containment matching when ``outDimTagsMap`` doesn't
    cover an entity (i.e. for all sub-entities of a boolean — OCC
    doesn't expose sub-shape lineage to gmsh's Python API).

    Returns
    -------
    list[dict]
        One entry per PG: ``{'dim', 'pg_tag', 'name', 'entity_tags',
        'entity_signatures'}``.  ``entity_signatures`` is a
        ``{tag: signature}`` dict; tags whose signature could not
        be computed (zero-extent, unknown surface type) are omitted
        — the geometric fallback simply skips them.
    """
    snapshot: list[dict] = []
    for dim, pg_tag in gmsh.model.getPhysicalGroups():
        name = gmsh.model.getPhysicalName(dim, pg_tag)
        ent_tags = list(gmsh.model.getEntitiesForPhysicalGroup(dim, pg_tag))
        sigs: dict[int, dict] = {}
        for t in ent_tags:
            t_int = int(t)
            sig = _entity_signature(dim, t_int)
            if sig is not None:
                sigs[t_int] = sig
        snapshot.append({
            'dim':               dim,
            'pg_tag':            pg_tag,
            'name':              name,
            'entity_tags':       [int(t) for t in ent_tags],
            'entity_signatures': sigs,
        })
    return snapshot


def remap_physical_groups(
    snapshot: list[dict],
    input_dimtags: list[DimTag],
    result_map: list[list[DimTag]],
    *,
    result: list[DimTag] | None = None,
    absorbed_into_result: bool = False,
    skip_pg_names: set[str] | None = None,
) -> None:
    """Recreate PGs with remapped entity tags after a boolean operation.

    Must be called **after** ``occ.synchronize()``.

    Parameters
    ----------
    snapshot
        Value returned by :func:`snapshot_physical_groups`.
    input_dimtags
        The ``obj + tool`` dimtags that were passed to the OCC boolean
        call — same ordering as ``result_map``.
    result_map
        The second return value of the OCC boolean call.  ``result_map[i]``
        lists the dimtags that ``input_dimtags[i]`` became.
    result
        The first return value of the OCC boolean call — the list of
        all surviving dimtags.  Required for the absorbed-fallback
        when ``absorbed_into_result=True``: OCC's fuse returns
        ``result_map=[[], []]`` even though the merged volume exists,
        so ``result_map`` alone cannot tell us where to remap to.
    absorbed_into_result : bool, default False
        When True, entities whose ``result_map`` entry is empty are
        remapped to the **result** entities at the same dimension
        (the material merged, so the PG should follow).  Use this
        for ``fuse`` and ``intersect``.  When False (the default,
        appropriate for ``cut``), empty mappings mean the entity was
        consumed and a warning is emitted.
    skip_pg_names : set[str], optional
        PG names whose snapshot entries should be silently dropped
        on this pass — the old PG is removed, no recreate, no
        warnings.  Used by topology-rebuild hooks (e.g. DRM box
        line PGs) that re-derive these PGs from a stored predicate
        after the boolean.  The standard remap can't track
        sub-entity renumbering for them anyway — OCC doesn't expose
        edge-level parent→child lineage.

    Notes
    -----
    * Entities in a PG that were **inputs** to the boolean are remapped
      through ``result_map``.
    * Entities that were **not** inputs are kept if they still exist in
      the model.  If they disappeared (sub-topology casualty of a
      higher-dim boolean), a warning is emitted and they are dropped.
    * If a PG becomes entirely empty, a warning is emitted and the PG
      is not recreated.
    """
    if not snapshot:
        return
    skip_set: set[str] = set(skip_pg_names) if skip_pg_names else set()

    # -- Build old-dimtag → [new-dimtags] mapping ----------------------
    dt_map: dict[DimTag, list[DimTag]] = {}
    for old_dt, new_dts in zip(input_dimtags, result_map):
        key = (int(old_dt[0]), int(old_dt[1]))
        dt_map[key] = [(int(d), int(t)) for d, t in new_dts]

    # -- Collect surviving entities per dimension (for absorbed-entity
    # fallback).  Prefer ``result`` (the first OCC return value) when
    # available because OCC's ``fuse`` leaves ``result_map`` empty
    # even when the merged entity exists in ``result``.  Fall back to
    # what ``result_map`` exposes for callers that did not pass it.
    result_by_dim: dict[int, list[int]] = {}
    source = result if result is not None else [
        dt for new_dts in result_map for dt in new_dts
    ]
    for d, t in source:
        result_by_dim.setdefault(int(d), []).append(int(t))
    for d in result_by_dim:
        result_by_dim[d] = sorted(set(result_by_dim[d]))

    # -- Current model entities (post-synchronize) ---------------------
    current_entities: set[DimTag] = set()
    for d in range(4):
        for _, t in gmsh.model.getEntities(d):
            current_entities.add((d, int(t)))

    # -- Geometric-fallback index: vectorized arrays of current-entity
    # signatures, built lazily per dim.  Two optimisations layered on
    # top of the scalar baseline:
    #
    # 1. **Transform fast path** — when ``input_dimtags`` is empty
    #    (``pg_preserved_identity`` for translate/rotate/copy), the
    #    geometric search is a guaranteed no-op (the pre-op signature
    #    won't match the post-transform geometry, and the keep-as-is
    #    branch handles tag preservation).  Skip signature build
    #    entirely.
    #
    # 2. **bbox pre-filter** — for each current entity, we always need
    #    its bbox to know if it's a candidate for any pre-op PG entity
    #    at the same dim.  Skip the expensive ``getCenterOfMass`` /
    #    ``getMass`` / ``getNormal`` calls for entities whose bbox
    #    doesn't overlap any pre-op bbox.  On a 10k-entity model with
    #    sparse PG coverage this typically skips >95% of the OCC work.
    geom_tol = _model_tolerance()
    is_boolean = bool(input_dimtags)

    # -- Untouched-survivor partition (coincident-entity fix) ----------
    # An entity that was NOT a boolean input and whose tag survived the
    # op with an *unchanged* signature (same kind + bbox + centroid
    # within tolerance) was never operated on by OCC — e.g. a standalone
    # curve that merely coincides geometrically with a genuine
    # sub-entity of an input.  Two consequences, both required to keep
    # coincident-but-distinct curves from collapsing onto each other:
    #
    #   (a) the survivor keeps its own tag verbatim instead of being
    #       re-discovered geometrically (handled in the main loop), and
    #   (b) it is excluded from the geometric-match candidate pool so a
    #       genuine sub-entity of an input can't absorb it (handled in
    #       ``_ensure_dim_indexed``).
    #
    # The signature check is what separates "OCC never touched this"
    # (keep verbatim) from "OCC split this but kept the parent tag on
    # one half" (signature changes -> still needs geometric matching).
    untouched_tags: dict[int, set[int]] = {}
    if is_boolean:
        for entry in snapshot:
            d_e = int(entry['dim'])
            pre_sigs_e = entry.get('entity_signatures', {})
            for et in entry['entity_tags']:
                if (d_e, et) in dt_map:
                    continue
                if (d_e, et) not in current_entities:
                    continue
                pre_sig = pre_sigs_e.get(et)
                if pre_sig is None:
                    continue
                cur_sig = _entity_signature(d_e, et)
                if cur_sig is None:
                    continue
                if (cur_sig.get('kind') or '') != (pre_sig.get('kind') or ''):
                    continue
                if not (
                    np.allclose(
                        cur_sig['bbox'], pre_sig['bbox'],
                        atol=geom_tol, rtol=0.0,
                    )
                    and np.allclose(
                        cur_sig['centroid'], pre_sig['centroid'],
                        atol=geom_tol, rtol=0.0,
                    )
                ):
                    continue
                untouched_tags.setdefault(d_e, set()).add(et)

    pre_bboxes_by_dim: dict[int, np.ndarray] = {}
    if is_boolean:
        for entry in snapshot:
            d_e = int(entry['dim'])
            sigs = entry.get('entity_signatures', {})
            if not sigs:
                continue
            bboxes = np.array(
                [s['bbox'] for s in sigs.values()], dtype=np.float64,
            )
            if d_e in pre_bboxes_by_dim:
                pre_bboxes_by_dim[d_e] = np.vstack(
                    [pre_bboxes_by_dim[d_e], bboxes]
                )
            else:
                pre_bboxes_by_dim[d_e] = bboxes

    current_idx: dict[int, dict | None] = {}

    def _ensure_dim_indexed(d: int) -> None:
        if d in current_idx:
            return
        if not is_boolean or d not in pre_bboxes_by_dim:
            current_idx[d] = None
            return

        pre_bboxes = pre_bboxes_by_dim[d]
        pre_mins = pre_bboxes[:, 0:3]
        pre_maxs = pre_bboxes[:, 3:6]
        # Exclude untouched survivors: an entity OCC never touched is
        # not a remap target for any other PG entity (see the
        # untouched-survivor partition above).
        untouched_for_d = untouched_tags.get(d, frozenset())

        tags_list: list[int] = []
        bboxes_list: list[tuple[float, ...]] = []
        centroids_list: list[tuple[float, float, float]] = []
        directions_list: list[tuple[float, float, float]] = []
        has_direction_list: list[bool] = []
        normals_list: list[tuple[float, float, float]] = []
        has_normal_list: list[bool] = []
        kinds_list: list[str] = []
        masses_list: list[float] = []

        for _, t in gmsh.model.getEntities(d):
            t_int = int(t)
            if t_int in untouched_for_d:
                continue
            try:
                bb = gmsh.model.getBoundingBox(d, t_int)
            except Exception:
                continue
            bbox = (
                float(bb[0]), float(bb[1]), float(bb[2]),
                float(bb[3]), float(bb[4]), float(bb[5]),
            )
            # bbox pre-filter: vectorized AABB-overlap against every
            # pre-op bbox at this dim.  Skip the full signature
            # build for non-candidates.
            cb_min = np.array(bbox[0:3]) - geom_tol
            cb_max = np.array(bbox[3:6]) + geom_tol
            overlap = (
                (cb_min <= pre_maxs) & (pre_mins <= cb_max)
            ).all(axis=1)
            if not overlap.any():
                continue

            sig = _entity_signature(d, t_int)
            if sig is None:
                continue

            tags_list.append(t_int)
            bboxes_list.append(sig['bbox'])
            centroids_list.append(sig['centroid'])
            dir_v = sig.get('direction')
            if dir_v is not None:
                directions_list.append(dir_v)
                has_direction_list.append(True)
            else:
                directions_list.append((0.0, 0.0, 0.0))
                has_direction_list.append(False)
            n_v = sig.get('normal')
            if n_v is not None:
                normals_list.append(n_v)
                has_normal_list.append(True)
            else:
                normals_list.append((0.0, 0.0, 0.0))
                has_normal_list.append(False)
            kinds_list.append(sig.get('kind', '') or '')
            m = sig.get('mass')
            masses_list.append(float(m) if m is not None else float('nan'))

        if not tags_list:
            current_idx[d] = None
            return

        current_idx[d] = {
            'tags':          np.array(tags_list, dtype=np.int64),
            'bboxes':        np.array(bboxes_list, dtype=np.float64),
            'centroids':     np.array(centroids_list, dtype=np.float64),
            'directions':    np.array(directions_list, dtype=np.float64),
            'has_direction': np.array(has_direction_list, dtype=bool),
            'normals':       np.array(normals_list, dtype=np.float64),
            'has_normal':    np.array(has_normal_list, dtype=bool),
            'kinds':         np.array(kinds_list, dtype='<U32'),
            'masses':        np.array(masses_list, dtype=np.float64),
        }

    def _geometric_match(
        dim: int, pre_sig: dict | None,
    ) -> tuple[list[int], float]:
        """Find current entities compatible with ``pre_sig``.

        Returns ``(tags, child_mass_sum)``.  The mass sum is computed
        inline to avoid a second pass through the array for the
        mass-balance check; entries whose mass is ``NaN`` (signature
        computation couldn't reach OCC) are excluded via
        ``np.nansum``.
        """
        if pre_sig is None:
            return [], 0.0
        _ensure_dim_indexed(dim)
        idx = current_idx.get(dim)
        if idx is None:
            return [], 0.0

        # ── dim 0: position-distance only ──────────────────────────
        if dim == 0:
            pre_c = np.array(pre_sig['centroid'], dtype=np.float64)
            deltas = idx['centroids'] - pre_c
            dist_sq = (deltas * deltas).sum(axis=1)
            mask = dist_sq < geom_tol * geom_tol
            if not mask.any():
                return [], 0.0
            return idx['tags'][mask].tolist(), 0.0

        # ── Bidirectional centroid-in-bbox ────────────────────────
        pre_bbox = np.asarray(pre_sig['bbox'], dtype=np.float64)
        pre_min, pre_max = pre_bbox[0:3], pre_bbox[3:6]
        pre_c = np.array(pre_sig['centroid'], dtype=np.float64)

        centroids = idx['centroids']
        bboxes = idx['bboxes']

        child_in_pre = (
            (centroids >= pre_min - geom_tol).all(axis=1)
            & (centroids <= pre_max + geom_tol).all(axis=1)
        )
        pre_in_child = (
            (pre_c >= bboxes[:, 0:3] - geom_tol).all(axis=1)
            & (pre_c <= bboxes[:, 3:6] + geom_tol).all(axis=1)
        )
        mask = child_in_pre | pre_in_child

        # ── Kind filter ───────────────────────────────────────────
        pre_kind = (pre_sig.get('kind') or '')
        if pre_kind:
            kinds = idx['kinds']
            mask &= (kinds == '') | (kinds == pre_kind)

        # ── Direction (1D) or normal (2D) parallelism ─────────────
        if dim == 1:
            pre_dir = pre_sig.get('direction')
            if pre_dir is not None:
                pre_dir_arr = np.asarray(pre_dir, dtype=np.float64)
                dots = np.abs(idx['directions'] @ pre_dir_arr)
                mask &= (
                    ~idx['has_direction'] | (dots >= _PARALLEL_COS_TOL)
                )
        elif dim == 2:
            pre_n = pre_sig.get('normal')
            if pre_n is not None:
                pre_n_arr = np.asarray(pre_n, dtype=np.float64)
                dots = np.abs(idx['normals'] @ pre_n_arr)
                mask &= (
                    ~idx['has_normal'] | (dots >= _PARALLEL_COS_TOL)
                )

        if not mask.any():
            return [], 0.0

        matched_tags = idx['tags'][mask].tolist()
        matched_masses = idx['masses'][mask]
        child_mass_sum = float(np.nansum(matched_masses))
        return matched_tags, child_mass_sum

    # Mass-balance tolerance: 5% slack absorbs OCC's BRep rebuilding
    # noise on every-day cuts while still flagging the egregious
    # "matched twice the area we should have" case that means the
    # match picked up a spurious sibling.
    _MASS_BALANCE_TOL = 0.05

    def _check_mass_balance(
        dim: int, pre_sig: dict | None, matches: list[int],
        child_mass_sum: float, pg_name: str, pre_tag: int,
    ) -> None:
        """Warn when matched-children total mass disagrees with parent.

        Validation only — never rejects a match.  Geometric matching
        can pick up a spurious sibling whose bbox happens to overlap
        the parent; the mass sum then exceeds the parent by a wide
        margin.  Conversely, a hole left by a cut (some children
        consumed) shows as a deficit — informational, since the
        ``was consumed`` warning already covers the wholly-deleted
        case but mass adds a quantitative signal for partial loss.
        """
        if pre_sig is None or not matches:
            return
        parent_mass = pre_sig.get('mass')
        if parent_mass is None or parent_mass <= 0.0:
            return
        if child_mass_sum <= 0.0:
            return
        ratio = child_mass_sum / parent_mass
        # Tolerate splits (ratio < 1 + tol) and merges where the
        # matched entity is the union of several parents (ratio
        # well above 1 is then expected from THIS parent's view).
        # Flag only an over-match by ≥ (1 + 2·tol) on a single
        # parent — that's the "false sibling" signal.
        if ratio > 1.0 + 2.0 * _MASS_BALANCE_TOL:
            warnings.warn(
                f"Physical group '{pg_name}' (dim={dim}): geometric "
                f"match for entity {pre_tag} produced children with "
                f"{ratio:.2f}× the parent mass — possible false "
                f"sibling match.  Inspect the PG before relying on "
                f"it.",
                stacklevel=3,
            )
        elif ratio < 1.0 - _MASS_BALANCE_TOL and len(matches) == 1:
            # Single match with significant mass deficit usually
            # means the entity was partly consumed — informational.
            warnings.warn(
                f"Physical group '{pg_name}' (dim={dim}): geometric "
                f"match for entity {pre_tag} recovered only "
                f"{ratio:.2f}× the parent mass — part of the entity "
                f"was consumed by the boolean.",
                stacklevel=3,
            )

    # -- Remove surviving stale PGs, then recreate ---------------------
    for entry in snapshot:
        dim = entry['dim']
        old_pg_tag = entry['pg_tag']
        name = entry['name']
        old_tags = entry['entity_tags']

        # Caller owns this PG name — they recreated it from a stored
        # predicate before this loop ran (e.g. DRM-box line PGs).
        # Skip BEFORE touching ``removePhysicalGroups``: synchronize()
        # wipes PG membership, and Gmsh may reuse the snapshot's
        # ``old_pg_tag`` for the caller's freshly-created PG —
        # blindly removing ``(dim, old_pg_tag)`` would destroy the
        # caller's work.
        if name in skip_set:
            continue

        # Remove the old PG if it survived synchronize with stale data.
        # Expected to fail when the PG was already destroyed by
        # synchronize() — Gmsh raises bare Exception, so we can't
        # narrow the catch.
        try:
            gmsh.model.removePhysicalGroups([(dim, old_pg_tag)])
        except Exception:
            pass

        # Remap each entity tag.
        #
        # For entities at the input dim (volumes for a 3D cut), the
        # OCC ``result_map`` gives us a precise parent→child list and
        # we use it directly.  For sub-entities (faces / edges /
        # vertices) gmsh has no equivalent map, so we fall back to a
        # geometric search by signature — *always*, even when the
        # old tag happens to still exist in ``current_entities``.
        # The reason for "always": OCC's cut can split a sub-entity
        # while keeping the parent's tag on one of the halves; if we
        # short-circuited on "old tag still alive" we'd capture that
        # half but miss its newly-tagged sibling.  The geometric
        # search picks up both pieces — the centroid of each surviving
        # half lies inside the pre-op bbox, and direction/normal
        # agreement disambiguates against unrelated parallel edges.
        pre_sigs = entry.get('entity_signatures', {})
        new_tags: list[int] = []
        for et in old_tags:
            old_dt = (dim, et)
            if old_dt in dt_map:
                # Entity was a boolean input — remap via result_map.
                mapped = [t for d, t in dt_map[old_dt] if d == dim]
                if mapped:
                    new_tags.extend(mapped)
                    continue
                if absorbed_into_result:
                    fallback = result_by_dim.get(dim, [])
                    if fallback:
                        new_tags.extend(fallback)
                        continue
                # Empty mapping at the input dim: ``outDimTagsMap``
                # is authoritative here (e.g. a cut's tool is
                # consumed, OCC returns ``[]``).  Don't run the
                # geometric fallback — for input-dim entities it
                # tends to find spurious "merge products" (the cut
                # survivor has a centroid inside the tool's bbox),
                # producing false-sibling matches at multiples of
                # the parent mass.  Trust the API; emit the
                # ``was consumed`` warning.
                warnings.warn(
                    f"Physical group '{name}' (dim={dim}): entity "
                    f"{et} was consumed by the boolean operation.",
                    stacklevel=3,
                )
                continue

            # Not a boolean input — sub-entity path.
            #
            # Fast path: when there was no boolean at all
            # (``input_dimtags`` empty -> ``pg_preserved_identity``
            # for translate/rotate/copy), tags preserve through the
            # operation and the geometric search would just waste
            # cycles computing post-transform signatures that can't
            # match the pre-op ones.  Skip straight to keep-as-is.
            if not is_boolean:
                if old_dt in current_entities:
                    new_tags.append(et)
                else:
                    warnings.warn(
                        f"Physical group '{name}' (dim={dim}): entity "
                        f"{et} was lost (entity no longer in the model "
                        f"after a non-boolean operation).",
                        stacklevel=3,
                    )
                continue

            # Untouched survivor: tag is stable and geometrically
            # unchanged — OCC never operated on it (e.g. a standalone
            # curve coincident with a genuine sub-entity of an input).
            # Keep it verbatim; geometric matching here would re-absorb
            # the coincident sibling and collapse the two PGs together.
            if et in untouched_tags.get(dim, frozenset()):
                new_tags.append(et)
                continue

            # Boolean sub-entity path.  Always go through geometric
            # matching even when the old tag is alive: OCC can split
            # a sub-entity while keeping the parent's tag on one of
            # the halves; the keep-as-is shortcut would capture that
            # half but miss the newly-tagged sibling.
            pre_sig = pre_sigs.get(et)
            geo_matches, child_mass_sum = _geometric_match(dim, pre_sig)
            if geo_matches:
                new_tags.extend(geo_matches)
                _check_mass_balance(
                    dim, pre_sig, geo_matches, child_mass_sum, name, et,
                )
            elif old_dt in current_entities:
                # Signature couldn't be computed pre-op but the tag
                # still resolves — keep it.
                new_tags.append(et)
            else:
                warnings.warn(
                    f"Physical group '{name}' (dim={dim}): entity {et} "
                    f"was lost (no geometric descendant in the post-op "
                    f"model).",
                    stacklevel=3,
                )

        new_tags = sorted(set(new_tags))
        if new_tags:
            new_pg = gmsh.model.addPhysicalGroup(dim, new_tags)
            gmsh.model.setPhysicalName(dim, new_pg, name)
        elif old_tags:
            warnings.warn(
                f"Physical group '{name}' (dim={dim}) is now empty — "
                f"no geometric descendants survived the boolean.",
                stacklevel=3,
            )


# =====================================================================
# Label PG cleanup after entity removal
# =====================================================================


def cleanup_label_pgs(removed_dimtags: list[DimTag]) -> None:
    """Remove deleted entity tags from label PGs.

    Call after ``occ.remove()`` + ``synchronize()`` when the caller
    knows exactly which entities were deleted.  Drops any label PG
    that becomes empty after the cleanup.

    Only touches ``_label:*`` PGs — user-facing PGs are left alone
    because the caller may want different semantics there.
    """
    if not removed_dimtags:
        return
    removed_set = {(int(d), int(t)) for d, t in removed_dimtags}
    for dim, pg_tag in list(gmsh.model.getPhysicalGroups()):
        name = gmsh.model.getPhysicalName(dim, pg_tag)
        if not is_label_pg(name):
            continue
        ent_tags = list(gmsh.model.getEntitiesForPhysicalGroup(dim, pg_tag))
        new_tags = [int(t) for t in ent_tags if (dim, int(t)) not in removed_set]
        if len(new_tags) == len(ent_tags):
            continue  # nothing changed
        # Gmsh raises bare Exception — can't narrow the catch.
        try:
            gmsh.model.removePhysicalGroups([(dim, pg_tag)])
        except Exception:
            pass
        if new_tags:
            new_pg = gmsh.model.addPhysicalGroup(dim, new_tags)
            gmsh.model.setPhysicalName(dim, new_pg, name)


def reconcile_label_pgs() -> None:
    """Remove stale entity tags from ALL label PGs.

    Walks every ``_label:*`` PG and drops any tag whose entity no
    longer exists in the Gmsh model.  Use this after operations
    like ``removeAllDuplicates()`` that renumber entities without
    providing a result map.
    """
    current: set[DimTag] = set()
    for d in range(4):
        for _, t in gmsh.model.getEntities(d):
            current.add((d, int(t)))

    for dim, pg_tag in list(gmsh.model.getPhysicalGroups()):
        name = gmsh.model.getPhysicalName(dim, pg_tag)
        if not is_label_pg(name):
            continue
        ent_tags = list(gmsh.model.getEntitiesForPhysicalGroup(dim, pg_tag))
        new_tags = [int(t) for t in ent_tags if (dim, int(t)) in current]
        if len(new_tags) == len(ent_tags):
            continue  # nothing changed
        # Gmsh raises bare Exception — can't narrow the catch.
        try:
            gmsh.model.removePhysicalGroups([(dim, pg_tag)])
        except Exception:
            pass
        if new_tags:
            new_pg = gmsh.model.addPhysicalGroup(dim, new_tags)
            gmsh.model.setPhysicalName(dim, new_pg, name)


class Labels(_HasLogging):
    """Geometry-time entity naming composite (``g.labels``).

    Backed by Gmsh physical groups with an internal ``_label:``
    prefix.  See the module docstring for the two-tier naming
    architecture.
    """

    _log_prefix = "Labels"

    def __init__(self, parent: "_SessionBase") -> None:
        self._parent = parent

    # ------------------------------------------------------------------
    # Create / update
    # ------------------------------------------------------------------

    def add(self, dim: int, tags: list[int], name: str) -> int:
        """Create a label for the given entities.

        If a label with the same name and dimension already exists,
        the tags are **merged** into the existing PG rather than
        creating a duplicate.

        Parameters
        ----------
        dim : int
            Entity dimension (0–3).
        tags : list[int]
            Entity tags to label.
        name : str
            Human-readable label name (without prefix).

        Returns
        -------
        int
            The Gmsh physical-group tag backing this label.
        """
        # Phase 3B.2d / ADR 0038 — labels round-trip via the
        # FEMData broker; mutating them post-extraction would diverge
        # the broker from gmsh.
        from ._compose_errors import chain_phase_guard
        chain_phase_guard(self._parent, f"g.labels.add({name!r})")
        prefixed = add_prefix(name)

        # Build a name→(dim, pg_tag) index in one pass over all label
        # PGs.  This replaces up to 4 separate _find_pg_tag scans with
        # a single O(n) scan + O(1) dict lookups.
        label_index = self._label_index()

        # Check if this label already exists at this dim — merge
        # rather than duplicate.
        existing_tag = label_index.get((dim, prefixed))
        if existing_tag is not None:
            existing_ents = list(
                gmsh.model.getEntitiesForPhysicalGroup(dim, existing_tag)
            )
            new_tags = set(int(t) for t in tags)
            truly_new = new_tags - set(existing_ents)
            if truly_new:
                warnings.warn(
                    f"Label {name!r} (dim={dim}) already exists with "
                    f"{len(existing_ents)} entity(ies). Merging "
                    f"{len(truly_new)} new tag(s) into it. If this is "
                    f"unintentional, use a different label name.",
                    stacklevel=3,
                )
            merged = sorted(set(existing_ents) | new_tags)
            gmsh.model.removePhysicalGroups([(dim, existing_tag)])
            pg_tag = gmsh.model.addPhysicalGroup(dim, merged)
            gmsh.model.setPhysicalName(dim, pg_tag, prefixed)
            self._log(f"add({name!r}, dim={dim}) merged into pg_tag={pg_tag}")
            return pg_tag

        # Check if the same label name exists at a DIFFERENT dim —
        # warn about cross-dim shadowing.
        for other_dim in range(4):
            if other_dim == dim:
                continue
            if (other_dim, prefixed) in label_index:
                warnings.warn(
                    f"Label {name!r} already exists at dim={other_dim}, "
                    f"now also being created at dim={dim}. This may "
                    f"cause ambiguous lookups when dim= is not specified.",
                    stacklevel=3,
                )
                break

        pg_tag = gmsh.model.addPhysicalGroup(dim, [int(t) for t in tags])
        gmsh.model.setPhysicalName(dim, pg_tag, prefixed)
        self._log(f"add({name!r}, dim={dim}, tags={tags}) -> pg_tag={pg_tag}")
        return pg_tag

    @staticmethod
    def _label_index() -> dict[tuple[int, str], int]:
        """Build ``(dim, prefixed_name) -> pg_tag`` for all label PGs.

        One pass over ``getPhysicalGroups(-1)`` replaces repeated
        per-dimension scans.
        """
        index: dict[tuple[int, str], int] = {}
        for d, t in gmsh.model.getPhysicalGroups(-1):
            pg_name = gmsh.model.getPhysicalName(d, t)
            if is_label_pg(pg_name):
                index[(d, pg_name)] = t
        return index

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    def entities(self, name: str, *, dim: int | None = None) -> list[int]:
        """Return entity tags for a label.

        Parameters
        ----------
        name : str
            Label name (without prefix).
        dim : int, optional
            Restrict to a single dimension.  When None, searches
            all dimensions.  If the label exists at exactly one
            dimension, returns those entities.  If it exists at
            multiple dimensions, raises ``ValueError`` asking the
            caller to specify ``dim=``.

        Returns
        -------
        list[int]
            Entity tags.

        Raises
        ------
        KeyError
            When no label with this name exists.
        ValueError
            When ``dim=None`` and the label exists at multiple
            dimensions.
        """
        prefixed = add_prefix(name)

        if dim is not None:
            # Direct lookup at a specific dimension
            for pg_dim, pg_tag in gmsh.model.getPhysicalGroups(dim):
                pg_name = gmsh.model.getPhysicalName(pg_dim, pg_tag)
                if pg_name == prefixed:
                    return [
                        int(t)
                        for t in gmsh.model.getEntitiesForPhysicalGroup(
                            pg_dim, pg_tag,
                        )
                    ]
            available = self.get_all()
            raise KeyError(
                f"no label {name!r} found at dim={dim}. "
                f"Available labels: {available}"
            )

        # dim=None — search all dimensions, require unambiguous match
        matches: list[tuple[int, int]] = []  # (pg_dim, pg_tag)
        for d in range(4):
            for pg_dim, pg_tag in gmsh.model.getPhysicalGroups(d):
                if gmsh.model.getPhysicalName(pg_dim, pg_tag) == prefixed:
                    matches.append((pg_dim, pg_tag))

        if not matches:
            available = self.get_all()
            raise KeyError(
                f"no label {name!r} found. Available labels: {available}"
            )

        if len(matches) == 1:
            pg_dim, pg_tag = matches[0]
            return [
                int(t)
                for t in gmsh.model.getEntitiesForPhysicalGroup(
                    pg_dim, pg_tag,
                )
            ]

        dims_found = sorted(set(d for d, _ in matches))
        raise ValueError(
            f"Label {name!r} exists at multiple dimensions "
            f"{dims_found}. Specify dim= to disambiguate, e.g. "
            f"g.labels.entities({name!r}, dim={dims_found[-1]})"
        )

    def get_all(self, *, dim: int = -1) -> list[str]:
        """Return all label names (without prefix).

        Parameters
        ----------
        dim : int, default -1
            Filter by dimension.  ``-1`` returns all dimensions.
        """
        names: list[str] = []
        for d, t in gmsh.model.getPhysicalGroups(dim):
            pg_name = gmsh.model.getPhysicalName(d, t)
            if is_label_pg(pg_name):
                names.append(strip_prefix(pg_name))
        return sorted(set(names))

    def summary(self):
        """DataFrame describing every label in the model.

        Mirrors :meth:`PhysicalGroups.summary` but returns only the
        internal label PGs (with the ``_label:`` prefix stripped).

        Returns
        -------
        pd.DataFrame  indexed by ``(dim, pg_tag)`` with columns
        ``name``, ``n_entities``, ``entity_tags``.
        """
        import pandas as pd
        rows: list[dict] = []
        for d, t in gmsh.model.getPhysicalGroups():
            pg_name = gmsh.model.getPhysicalName(d, t)
            if not is_label_pg(pg_name):
                continue
            entities = gmsh.model.getEntitiesForPhysicalGroup(d, t)
            rows.append({
                'dim'        : d,
                'pg_tag'     : t,
                'name'       : strip_prefix(pg_name),
                'n_entities' : len(entities),
                'entity_tags': ", ".join(str(x) for x in entities),
            })
        if not rows:
            return pd.DataFrame(
                columns=['dim', 'pg_tag', 'name', 'n_entities', 'entity_tags']
            )
        return (
            pd.DataFrame(rows)
            .set_index(['dim', 'pg_tag'])
            .sort_index()
        )

    def has(self, name: str, *, dim: int | None = None) -> bool:
        """Return True if a label with this name exists."""
        try:
            self.entities(name, dim=dim)
            return True
        except KeyError:
            return False

    # ------------------------------------------------------------------
    # Remove / rename
    # ------------------------------------------------------------------

    def remove(self, name: str, *, dim: int | None = None) -> None:
        """Delete a label (and its backing physical group).

        Parameters
        ----------
        name : str
            Label name (without prefix).
        dim : int, optional
            Restrict to a single dimension.  When None, removes the
            label at **all** dimensions where it exists.

        Raises
        ------
        KeyError
            When no label with this name exists.
        """
        prefixed = add_prefix(name)
        dims = [dim] if dim is not None else [0, 1, 2, 3]
        removed = False
        for d in dims:
            for pg_dim, pg_tag in list(gmsh.model.getPhysicalGroups(d)):
                if gmsh.model.getPhysicalName(pg_dim, pg_tag) == prefixed:
                    gmsh.model.removePhysicalGroups([(pg_dim, pg_tag)])
                    removed = True
        if not removed:
            raise KeyError(
                f"no label {name!r} found"
                + (f" at dim={dim}" if dim is not None else "")
                + f". Available labels: {self.get_all()}"
            )
        self._log(f"remove({name!r}, dim={dim})")

    def rename(self, old_name: str, new_name: str, *, dim: int | None = None) -> None:
        """Rename a label in place, preserving its entity membership.

        Parameters
        ----------
        old_name : str
            Current label name (without prefix).
        new_name : str
            New label name (without prefix).
        dim : int, optional
            Restrict to a single dimension.  When None, renames the
            label at **all** dimensions where it exists.

        Raises
        ------
        KeyError
            When no label with *old_name* exists.
        """
        old_prefixed = add_prefix(old_name)
        new_prefixed = add_prefix(new_name)
        dims = [dim] if dim is not None else [0, 1, 2, 3]
        renamed = False
        for d in dims:
            for pg_dim, pg_tag in list(gmsh.model.getPhysicalGroups(d)):
                if gmsh.model.getPhysicalName(pg_dim, pg_tag) == old_prefixed:
                    # Read entities, remove old PG, create new one
                    ent_tags = list(
                        gmsh.model.getEntitiesForPhysicalGroup(pg_dim, pg_tag)
                    )
                    gmsh.model.removePhysicalGroups([(pg_dim, pg_tag)])
                    new_pg = gmsh.model.addPhysicalGroup(pg_dim, [int(t) for t in ent_tags])
                    gmsh.model.setPhysicalName(pg_dim, new_pg, new_prefixed)
                    renamed = True
        if not renamed:
            raise KeyError(
                f"no label {old_name!r} found"
                + (f" at dim={dim}" if dim is not None else "")
                + f". Available labels: {self.get_all()}"
            )
        self._log(f"rename({old_name!r} -> {new_name!r}, dim={dim})")

    # ------------------------------------------------------------------
    # Promote to physical group
    # ------------------------------------------------------------------

    def promote_to_physical(
        self,
        label_name: str,
        *,
        pg_name: str | None = None,
        dim: int | None = None,
    ) -> int:
        """Copy a label's entities into a solver-facing physical group.

        The label remains intact — this is a **copy**, not a move.
        The new PG is visible to ``g.physical``, ``fem.physical``,
        and the OpenSees exporter.

        Parameters
        ----------
        label_name : str
            Label to promote.
        pg_name : str, optional
            Name for the new physical group.  Defaults to the
            label name (without prefix).
        dim : int, optional
            Dimension to promote.  Required when the label exists
            at multiple dimensions.

        Returns
        -------
        int
            Physical-group tag of the new PG.
        """
        tags = self.entities(label_name, dim=dim)
        out_name = pg_name or label_name

        # Resolve the dim from the label's PG
        prefixed = add_prefix(label_name)
        resolved_dim = dim
        if resolved_dim is None:
            for d in [3, 2, 1, 0]:
                for pd, pt in gmsh.model.getPhysicalGroups(d):
                    if gmsh.model.getPhysicalName(pd, pt) == prefixed:
                        resolved_dim = d
                        break
                if resolved_dim is not None:
                    break
        if resolved_dim is None:
            raise KeyError(f"label {label_name!r} not found")

        pg_tag = gmsh.model.addPhysicalGroup(resolved_dim, tags)
        gmsh.model.setPhysicalName(resolved_dim, pg_tag, out_name)
        self._log(
            f"promote_to_physical({label_name!r}) -> "
            f"PG {out_name!r} (dim={resolved_dim}, {len(tags)} entities)"
        )
        return pg_tag

    def reverse_map(self, *, dim: int = -1) -> dict[DimTag, str]:
        """Build a ``(dim, tag) -> label_name`` reverse lookup.

        Useful when callers need to find labels for many entities at
        once without repeated ``entities()`` calls.

        Parameters
        ----------
        dim : int, default -1
            Filter by dimension.  ``-1`` returns all dimensions.
        """
        result: dict[DimTag, str] = {}
        for d, pg_tag in gmsh.model.getPhysicalGroups(dim):
            pg_name = gmsh.model.getPhysicalName(d, pg_tag)
            if not is_label_pg(pg_name):
                continue
            name = strip_prefix(pg_name)
            for t in gmsh.model.getEntitiesForPhysicalGroup(d, pg_tag):
                result[(int(d), int(t))] = name
        return result

    def labels_for_entity(self, dim: int, tag: int) -> list[str]:
        """Return all label names that contain the given entity."""
        names: list[str] = []
        for d, pg_tag in gmsh.model.getPhysicalGroups(dim):
            pg_name = gmsh.model.getPhysicalName(d, pg_tag)
            if not is_label_pg(pg_name):
                continue
            ent_tags = gmsh.model.getEntitiesForPhysicalGroup(d, pg_tag)
            if tag in ent_tags:
                names.append(strip_prefix(pg_name))
        return names

    def __repr__(self) -> str:
        try:
            labels = self.get_all()
            return f"Labels({len(labels)} labels: {labels[:5]}{'...' if len(labels) > 5 else ''})"
        except Exception:
            return "Labels(session closed)"
