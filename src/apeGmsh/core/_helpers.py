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
    * ``int`` — raw Gmsh entity tag, passed through after existence
      check.
    * ``str`` — resolved in order: label name (Tier 1, ``g.labels``),
      then user physical-group name (Tier 2, ``g.physical``).  The
      first match wins.
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
        string references and to validate int tags.  When ``None``,
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
        When a string reference is not found as a label or PG.
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
    Handles label / PG references that span multiple dimensions by
    enumerating each dim and emitting one dimtag per (dim, tag) hit;
    callers never have to coerce a single dim when the reference
    naturally lives at several.

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
    """Resolve a string ref to dimtags — label first, then user PG.

    Mirrors :func:`_resolve_string`'s order (label Tier 1, then PG
    Tier 2) but emits ``(dim, tag)`` and handles multi-dim names by
    enumerating all dims the name exists at.
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
        try:
            return [(int(d), int(t))
                    for d, t in physical_comp.dim_tags(name)]
        except (KeyError, TypeError):
            pass

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
        f"Entity reference {name!r} not found as a label or physical "
        f"group.  Pass a label name (Tier 1, g.labels) or a physical "
        f"group name (Tier 2, g.physical)."
        f"\n  Available labels: {available_labels}"
        f"\n  Available PGs: {available_pgs}"
    )


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
    """Resolve a string reference — label first, then user PG.

    Resolution order:
    1. ``session.labels.entities(name, dim=dim)`` — Tier 1 label
    2. ``session.physical.entities(name, dim=dim)`` — Tier 2 PG
    3. Raise ``KeyError`` with available names
    """
    # Try label
    labels_comp = getattr(session, 'labels', None)
    if labels_comp is not None:
        try:
            return labels_comp.entities(name, dim=dim)
        except KeyError:
            pass

    # Try user PG
    physical_comp = getattr(session, 'physical', None)
    if physical_comp is not None:
        try:
            result = physical_comp.entities(name, dim=dim)
            return result
        except (KeyError, TypeError):
            pass

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
        f"Entity reference {name!r} not found as a label or "
        f"physical group"
        + (f" at dim={dim}" if dim is not None else "")
        + f".\n  Available labels: {available_labels}"
        + f"\n  Available PGs: {available_pgs}"
    )
