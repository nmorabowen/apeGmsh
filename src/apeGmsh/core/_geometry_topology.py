"""Topology-driven sweep that removes orphan dim<=2 geometry left
behind by OCC boolean / fragment / cut operations.

Shared by :meth:`_Geometry.slice`, :meth:`_Geometry.cut_by_surface`,
and :meth:`_Boolean.fragment` so all three cleanup paths agree on
what "orphan" means.  Before this module, each call site carried
its own definition (slice used a per-call snapshot, fragment used a
centroid-in-bbox heuristic, the other bool ops did nothing) — the
audit found that each definition missed at least one failure mode.

The single rule encoded here: a dim<=2 entity stays IFF it bounds a
registered dim=3 volume at some depth OR is user-intentional (a
metadata-registered standalone like ``add_rectangle`` / ``add_point``
/ ``add_cutting_plane``, or carries a user label).  Everything else
is reaped, including stale ``_metadata`` keys whose tags no longer
exist in OCC.

This module owns no public API on its own — :meth:`_Geometry.find_orphans`
/ :meth:`_Geometry.remove_orphans` / :meth:`_Geometry.validate_pre_mesh`
are the user-facing surface; the bool / cut ops call this internally.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import gmsh

from .Labels import cleanup_label_pgs

if TYPE_CHECKING:
    from .Model import Model


DimTag = tuple[int, int]


def _gather_keep_set(model: "Model") -> set[DimTag]:
    """Return every dimtag that the sweep must preserve.

    Two protection seeds:

    1. **Every registered volume** plus its full boundary closure
       (surfaces → curves → points).  Volumes are always kept; the
       sweep is dim<=2 only.
    2. **Every user-intentional non-volume entity** (in
       ``model._metadata`` or carrying a label) plus its own boundary
       closure.  This is what protects standalone embedded shells
       and pure 2D meshable surfaces — their boundary curves and
       points bound no volume, but they bound a user-intentional
       surface, which is the same load-bearing semantics.

    ``gmsh.model.getBoundary(..., recursive=True)`` returns ONLY the
    leaf points, not the intermediate surfaces/curves — so the walk
    is done explicitly dim-by-dim with ``recursive=False``.
    """
    keep: set[DimTag] = set()

    # Seed: every volume + every user-intentional non-volume.
    seed_by_dim: dict[int, list[int]] = {0: [], 1: [], 2: [], 3: []}
    for _, v in gmsh.model.getEntities(3):
        seed_by_dim[3].append(int(v))
        keep.add((3, int(v)))
    for d in (2, 1, 0):
        for _, t in gmsh.model.getEntities(d):
            if _user_intentional(model, d, int(t)):
                seed_by_dim[d].append(int(t))
                keep.add((d, int(t)))

    # Walk boundary closure of every seeded dim — dim-by-dim, top down.
    # ``getBoundary`` requires inputs at the same dim, so feed one dim
    # at a time.  Each iteration accumulates its children into the
    # next-lower dim's seed list, so a dim=3 seed propagates all the
    # way down to dim=0 in three passes.
    for parent_dim in (3, 2, 1):
        parents = seed_by_dim[parent_dim]
        if not parents:
            continue
        children = gmsh.model.getBoundary(
            [(parent_dim, t) for t in parents],
            oriented=False, recursive=False, combined=False,
        )
        for d_c, t_c in children:
            d_c_abs, t_c_abs = abs(int(d_c)), abs(int(t_c))
            if (d_c_abs, t_c_abs) not in keep:
                keep.add((d_c_abs, t_c_abs))
                seed_by_dim[d_c_abs].append(t_c_abs)

    return keep


def _user_intentional(
    model: "Model", d: int, t: int,
) -> bool:
    """True iff (d, t) is something the user explicitly created and
    wants preserved even though it does not bound a volume.

    Two channels for "user-intentional":

    * ``model._metadata`` — every ``add_*`` primitive registers here,
      so a standalone rectangle or point or cutting plane survives.
    * Label PG membership — if the user (or a subsystem like
      :meth:`Labels.add`) attached a label to ``(d, t)``, that is an
      explicit declaration of intent.

    Callers that want to override the metadata channel for an
    operation-specific tool (e.g. a cutting plane being consumed)
    should pass it via ``also_remove`` on :func:`sweep_dangling`.
    """
    if (d, t) in model._metadata:
        return True
    labels_comp = getattr(model._parent, "labels", None)
    if labels_comp is None:
        return False
    try:
        names = labels_comp.labels_for_entity(d, t)
    except Exception:
        return False
    return bool(names)


def sweep_dangling(
    model: "Model",
    *,
    max_dim: int = 2,
    also_remove: set[DimTag] | None = None,
    dry_run: bool = False,
) -> dict[int, list[int]]:
    """Remove dim<=``max_dim`` entities that bound no registered
    volume and are not user-intentional.  Also reaps stale
    ``model._metadata`` entries whose tag is no longer in OCC.

    Parameters
    ----------
    model
        The :class:`Model` instance whose ``_metadata`` and labels
        composite drive the "user-intentional" check.
    max_dim
        Highest dimension to sweep (default 2 — surfaces and below).
        Volumes are never swept by this helper.
    also_remove
        Dimtags that must be removed even if they would otherwise be
        protected by the metadata / labels checks.  Used by the cut
        operations to drop the cutting plane after the fragment
        consumes it.
    dry_run
        When True, return the dimtags that *would* be removed without
        touching OCC or ``model._metadata``.

    Returns
    -------
    dict[int, list[int]]
        ``{dim: [tags]}`` for every dim in ``range(max_dim + 1)``,
        listing the tags that were (or would be) removed.
    """
    gmsh.model.occ.synchronize()

    also = set(also_remove or ())
    keep_dimtags = _gather_keep_set(model)

    removed: dict[int, list[int]] = {d: [] for d in range(max_dim + 1)}
    removed_dts: list[DimTag] = []

    # Sweep top-down (surfaces -> curves -> points) so that a curve
    # left dangling once its parent surface is removed gets picked
    # up on this same pass.  Sync between dims so that ``getEntities``
    # at each level reflects what ``occ.remove(..., recursive=True)``
    # already cascaded away — otherwise the sweep tries to drop
    # already-dead tags and OCC emits "Unknown entity" warnings.
    for d in range(max_dim, -1, -1):
        if not dry_run:
            gmsh.model.occ.synchronize()
        for _, t in list(gmsh.model.getEntities(d)):
            t = int(t)
            dt = (d, t)
            forced = dt in also
            if not forced:
                if dt in keep_dimtags:
                    continue
                if _user_intentional(model, d, t):
                    continue
            if dry_run:
                removed[d].append(t)
                continue
            try:
                gmsh.model.occ.remove([dt], recursive=True)
            except Exception:
                # OCC refuses when the entity is still bound by
                # something outside the sweep envelope.  That is a
                # legitimate "leave it alone" signal; record nothing
                # and move on.
                continue
            model._metadata.pop(dt, None)
            removed[d].append(t)
            removed_dts.append(dt)

    # Reap stale metadata: a key that names a (d, t) no longer in
    # OCC is by definition pointing at consumed geometry.  Done after
    # the sweep so we don't reap the same keys twice.
    if not dry_run:
        live: set[DimTag] = set()
        for d in range(4):
            for _, t in gmsh.model.getEntities(d):
                live.add((d, int(t)))
        for dt in list(model._metadata):
            if dt not in live:
                model._metadata.pop(dt, None)

        gmsh.model.occ.synchronize()
        if removed_dts:
            cleanup_label_pgs(removed_dts)

    return removed
