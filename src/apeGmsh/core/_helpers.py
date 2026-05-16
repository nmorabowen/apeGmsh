"""
Shared helper functions used by multiple composites.

Avoids duplicating logic across Model, Mesh, and other modules.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import gmsh

if TYPE_CHECKING:
    from apeGmsh._session import _SessionBase

from apeGmsh._types import DimTag, Tag, TagsLike, EntityRefs  # noqa: F401  — ``Tag``, ``DimTag``, ``TagsLike`` are re-exported from this module by every ``_model_*`` sibling.


def resolve_dim(tag: int, default_dim: int) -> int:
    """Look up *tag*'s dimension by querying the live Gmsh model.

    Searches dimensions 3 → 0 and returns the first match.
    If the tag is not found at any dimension, returns *default_dim*.
    """
    for d in (3, 2, 1, 0):
        for _, t in gmsh.model.getEntities(d):
            if t == tag:
                return d
    return default_dim


def as_dimtags(
    tags: TagsLike,
    default_dim: int = 3,
) -> list[DimTag]:
    """Normalize flexible tag input to ``[(dim, tag), ...]``.

    Accepted forms:
    - ``5``                -> ``[(dim, 5)]``
    - ``[1, 2, 3]``        -> ``[(dim, 1), (dim, 2), (dim, 3)]``
    - ``(2, 5)``           -> ``[(2, 5)]``
    - ``[(2, 5), (2, 6)]`` -> ``[(2, 5), (2, 6)]``

    Bare int tags are resolved to their dimension by querying the
    live Gmsh model.  If not found, *default_dim* is used.
    """
    def _dim(t: int) -> int:
        return resolve_dim(t, default_dim)

    if isinstance(tags, int):
        return [(_dim(tags), tags)]

    # Single (dim, tag) tuple
    if (
        isinstance(tags, tuple)
        and len(tags) == 2
        and all(isinstance(x, int) for x in tags)
    ):
        return [tags]

    out: list[DimTag] = []
    for item in tags:
        if isinstance(item, int):
            out.append((_dim(item), item))
        elif isinstance(item, (tuple, list)) and len(item) == 2:
            out.append((int(item[0]), int(item[1])))
        else:
            raise TypeError(f"Cannot convert {item!r} to a (dim, tag) pair.")
    return out


# =====================================================================
# Shared entity resolver (tag / label / PG flexibility)
# =====================================================================

def resolve_to_tags(
    ref: EntityRefs,
    *,
    dim: int | None = None,
    session: "_SessionBase",
) -> list[int]:
    """Resolve a flexible entity reference to a flat list of tags.

    Accepted inputs
    ---------------
    * ``int`` — raw Gmsh entity tag, passed through **verbatim**
      (explicit user intent; not existence- or dim-validated).
    * ``str`` — resolved in order: label name (Tier 1, ``g.labels``),
      user physical-group name (Tier 2, ``g.physical``), then part
      label (Tier 3, ``g.parts``).  The first match wins.
    * ``(dim, tag)`` — dimtag tuple, tag extracted.
    * ``list`` — each element resolved independently, results
      concatenated.
    * ``None`` — all entities at *dim* (requires ``dim`` to be set).

    Parameters
    ----------
    ref : EntityRef, list[EntityRef], or None
        The reference(s) to resolve.
    dim : int, optional
        Expected entity dimension.  Used to scope the search for
        string references (label scoped or unioned, PG single-dim).
        Raw int tags are passed through unscoped.  When ``None``,
        all dimensions are searched.
    session : _SessionBase
        The owning session — used to access ``session.labels`` and
        ``session.physical``.

    Returns
    -------
    list[int]
        Flat list of resolved entity tags.

    Raises
    ------
    KeyError
        When a string reference is not found as a label, PG, or part.
    ValueError
        When ``ref`` is None but ``dim`` is not set.

    Example
    -------
    ::

        tags = resolve_to_tags("col.start_face", dim=2, session=g)
        tags = resolve_to_tags(42, dim=3, session=g)
        tags = resolve_to_tags(["col.top_flange", 99], dim=3, session=g)
        tags = resolve_to_tags(None, dim=3, session=g)  # all dim=3 entities
    """
    if ref is None:
        if dim is None:
            raise ValueError(
                "resolve_to_tags: ref=None requires dim= to be set "
                "so we know which entities to return."
            )
        return [t for _, t in gmsh.model.getEntities(dim)]

    if isinstance(ref, (list, tuple)) and not _is_dimtag_tuple(ref):
        # It's a list of mixed refs — resolve each.
        tags: list[int] = []
        for item in ref:
            tags.extend(resolve_to_tags(item, dim=dim, session=session))
        return tags

    if isinstance(ref, tuple) and _is_dimtag_tuple(ref):
        # (dim, tag) passthrough
        return [int(ref[1])]

    if isinstance(ref, bool):
        raise TypeError(
            "resolve_to_tags: expected int, str, or (dim, tag); got bool."
        )

    if isinstance(ref, int):
        return [int(ref)]

    if isinstance(ref, str):
        return _resolve_string(ref, dim=dim, session=session)

    raise TypeError(
        f"resolve_to_tags: unsupported ref type {type(ref).__name__!r}. "
        f"Expected int, str, (dim, tag), or list thereof."
    )


def resolve_to_dimtags(
    ref: EntityRefs,
    *,
    default_dim: int,
    session: "_SessionBase",
) -> list[DimTag]:
    """Resolve a flexible entity reference to a flat list of ``(dim, tag)``.

    Companion to :func:`resolve_to_tags` — returns dimtags instead of
    flat tags, which matches the shape every OCC boolean call expects.
    A *label* reference that spans multiple dimensions is enumerated
    into one dimtag per (dim, tag) hit; physical groups map to a
    single dimension.

    Accepted inputs (same shape as :func:`resolve_to_tags`)
    -------------------------------------------------------
    * ``int`` — raw tag; dim resolved from the live model, falling back
      to *default_dim* if not found.
    * ``str`` — label name first, then user physical-group name (same
      order as :func:`resolve_to_tags`).
    * ``(dim, tag)`` — passthrough.
    * ``list`` — mixed refs resolved independently and concatenated.
    * ``None`` — all entities at *default_dim*.

    Raises
    ------
    KeyError
        When a string reference is not found as a label or PG.
    TypeError
        When an element is not one of the supported shapes.
    """
    if ref is None:
        return [(default_dim, int(t))
                for _, t in gmsh.model.getEntities(default_dim)]

    if isinstance(ref, (list, tuple)) and not _is_dimtag_tuple(ref):
        out: list[DimTag] = []
        for item in ref:
            out.extend(
                resolve_to_dimtags(
                    item, default_dim=default_dim, session=session,
                )
            )
        return out

    if isinstance(ref, tuple) and _is_dimtag_tuple(ref):
        return [(int(ref[0]), int(ref[1]))]

    if isinstance(ref, bool):
        raise TypeError(
            "resolve_to_dimtags: expected int, str, or (dim, tag); got bool."
        )

    if isinstance(ref, int):
        return [(resolve_dim(int(ref), default_dim), int(ref))]

    if isinstance(ref, str):
        return _resolve_string_to_dimtags(
            ref, default_dim=default_dim, session=session,
        )

    raise TypeError(
        f"resolve_to_dimtags: unsupported ref type {type(ref).__name__!r}. "
        f"Expected int, str, (dim, tag), or list thereof."
    )


def _resolve_string_to_dimtags(
    name: str,
    *,
    default_dim: int,
    session: "_SessionBase",
) -> list[DimTag]:
    """Resolve a string ref to dimtags — label, then PG, then part.

    Precedence ``label (Tier 1) → physical group (Tier 2) → part
    label``, emitting ``(dim, tag)`` — the same geometry-entity
    precedence loads/masses/FEMData use.  Multi-dim *labels* and
    *parts* are enumerated across every dim they occupy; a physical
    group maps to a single dimension and raises ``ValueError`` if a
    legacy model carries one PG name at several dims.  Mesh
    selections are intentionally **not** a tier here: they are a
    post-mesh node concept, not a geometry entity, so a mesh-
    selection name correctly raises ``KeyError`` (loads/masses
    handle it via their own post-mesh path).
    """
    labels_comp = getattr(session, 'labels', None)
    if labels_comp is not None:
        hits: list[DimTag] = []
        for d in range(4):
            try:
                tags = labels_comp.entities(name, dim=d)
            except KeyError:
                continue
            hits.extend((d, int(t)) for t in tags)
        if hits:
            return hits

    physical_comp = getattr(session, 'physical', None)
    if physical_comp is not None:
        pg_hits: list[DimTag] = []
        for d in range(4):
            pg_tag = physical_comp.get_tag(d, name)
            if pg_tag is None:
                continue
            pg_hits.extend(
                (d, int(t))
                for t in physical_comp.get_entities(d, pg_tag)
            )
        if pg_hits:
            pg_dims = {d for d, _ in pg_hits}
            if len(pg_dims) > 1:
                raise ValueError(
                    f"Physical group {name!r} exists at multiple "
                    f"dimensions {sorted(pg_dims)}. Multi-dimensional "
                    f"physical groups are not supported."
                )
            return pg_hits

    parts_comp = getattr(session, 'parts', None)
    if parts_comp is not None:
        inst = getattr(parts_comp, '_instances', {}).get(name)
        if inst is not None:
            part_hits = [
                (int(d), int(t))
                for d, tags in inst.entities.items()
                for t in tags
            ]
            if part_hits:
                return part_hits

    available_labels: list[str] = []
    available_pgs: list[str] = []
    if labels_comp is not None:
        try:
            available_labels = labels_comp.get_all()
        except Exception:
            pass
    if physical_comp is not None:
        try:
            available_pgs = [
                gmsh.model.getPhysicalName(d, t)
                for d, t in physical_comp.get_all()
            ]
        except Exception:
            pass

    raise KeyError(
        f"Entity reference {name!r} not found as a label, physical "
        f"group, or part.  Pass a label name (Tier 1, g.labels), a "
        f"physical group name (Tier 2, g.physical), or a part label "
        f"(Tier 3, g.parts)."
        f"\n  Available labels: {available_labels}"
        f"\n  Available PGs: {available_pgs}"
    )


def resolve_to_single_dimtag(
    ref,
    *,
    default_dim: int,
    session: "_SessionBase",
    what: str = "entity",
) -> DimTag:
    """Resolve a flexible ref expected to identify a single entity.

    Wraps :func:`resolve_to_dimtags` and enforces a single-hit
    resolution.  Raises ``ValueError`` with a clear, actionable
    message when the ref resolves to zero or multiple dimtags.

    Parameters
    ----------
    ref : int, str, (dim, tag), or list — same shapes as
        :func:`resolve_to_dimtags`.
    default_dim : fallback dim for bare-int refs.
    session : the active session (for label/PG lookup).
    what : human-readable noun for error messages, e.g. ``"surface"``,
        ``"cutting plane"``, ``"volume"``.

    Returns
    -------
    (dim, tag) — exactly one dimtag.
    """
    dimtags = resolve_to_dimtags(ref, default_dim=default_dim, session=session)
    if not dimtags:
        raise ValueError(
            f"Could not resolve {what} from {ref!r} — no entities found."
        )
    if len(dimtags) > 1:
        raise ValueError(
            f"Ambiguous {what} reference {ref!r} — resolved to "
            f"{len(dimtags)} entities {dimtags}. Pass an explicit "
            f"(dim, tag) tuple or a label that identifies a single entity."
        )
    return dimtags[0]


def _is_dimtag_tuple(val) -> bool:
    """True if *val* looks like a ``(dim, tag)`` pair."""
    return (
        isinstance(val, tuple)
        and len(val) == 2
        and isinstance(val[0], int)
        and isinstance(val[1], int)
    )


def _resolve_string(
    name: str,
    *,
    dim: int | None,
    session: "_SessionBase",
) -> list[int]:
    """Resolve a string reference — label, then PG, then part.

    Resolution order (the geometry-entity precedence shared with
    loads/masses/FEMData):
    1. ``session.labels`` — Tier 1 label
    2. ``session.physical.entities(name, dim=dim)`` — Tier 2 PG
    3. ``session.parts`` instance — Tier 3 part label
    4. Raise ``KeyError`` with available names

    Mesh selections are intentionally not a tier: they are a
    post-mesh node concept, not a geometry entity.

    A label may legitimately span dimensions.  With an explicit
    ``dim`` the label is scoped to it; with ``dim=None`` the entities
    are **unioned across every dim the label occupies** (mirroring
    :func:`_resolve_string_to_dimtags`, ``_group_set`` and FEMData) —
    rather than raising the way :meth:`Labels.entities` does for a
    bare multi-dim lookup.  A physical group stays single-dim: with
    ``dim=None`` a multi-dim PG still raises (rule enforced by
    :meth:`PhysicalGroups.entities`).
    """
    # Try label (union across dims when no explicit dim is given)
    labels_comp = getattr(session, 'labels', None)
    if labels_comp is not None:
        if dim is not None:
            try:
                return labels_comp.entities(name, dim=dim)
            except KeyError:
                pass
        else:
            hits: list[int] = []
            for d in range(4):
                try:
                    hits.extend(labels_comp.entities(name, dim=d))
                except KeyError:
                    continue
            if hits:
                return hits

    # Try user PG
    physical_comp = getattr(session, 'physical', None)
    if physical_comp is not None:
        try:
            result = physical_comp.entities(name, dim=dim)
            return result
        except (KeyError, TypeError):
            pass

    # Try part label (Tier 3) — same precedence loads/FEMData use.
    parts_comp = getattr(session, 'parts', None)
    if parts_comp is not None:
        inst = getattr(parts_comp, '_instances', {}).get(name)
        if inst is not None:
            if dim is not None:
                part_tags = [int(t) for t in inst.entities.get(dim, [])]
            else:
                part_tags = [
                    int(t)
                    for tags in inst.entities.values()
                    for t in tags
                ]
            if part_tags:
                return part_tags

    # Build a helpful error message
    available_labels = []
    available_pgs = []
    if labels_comp is not None:
        try:
            available_labels = labels_comp.get_all()
        except Exception:
            pass
    if physical_comp is not None:
        try:
            available_pgs = [
                gmsh.model.getPhysicalName(d, t)
                for d, t in physical_comp.get_all()
            ]
        except Exception:
            pass

    raise KeyError(
        f"Entity reference {name!r} not found as a label, "
        f"physical group, or part"
        + (f" at dim={dim}" if dim is not None else "")
        + f".\n  Available labels: {available_labels}"
        + f"\n  Available PGs: {available_pgs}"
    )
