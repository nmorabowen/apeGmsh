"""
Shared helper functions used by multiple composites.

Avoids duplicating logic across Model, Mesh, and other modules.
"""
from __future__ import annotations

# Type aliases used across the library
Tag = int
DimTag = tuple[int, int]
TagsLike = int | Tag | DimTag | list


def resolve_dim(tag: int, default_dim: int, registry: dict) -> int:
    """Look up *tag*'s dimension from *registry*, fallback to *default_dim*.

    If the tag appears at exactly one dimension, return that dimension.
    If ambiguous (same tag at multiple dims) or missing, return *default_dim*.
    """
    found = [d for (d, t) in registry if t == tag]
    if len(found) == 1:
        return found[0]
    return default_dim


def as_dimtags(
    tags: TagsLike,
    default_dim: int = 3,
    registry: dict | None = None,
) -> list[DimTag]:
    """Normalize flexible tag input to ``[(dim, tag), ...]``.

    Accepted forms:
    - ``5``                → ``[(dim, 5)]``
    - ``[1, 2, 3]``        → ``[(dim, 1), (dim, 2), (dim, 3)]``
    - ``(2, 5)``           → ``[(2, 5)]``
    - ``[(2, 5), (2, 6)]`` → ``[(2, 5), (2, 6)]``

    When *registry* is provided and a bare int tag appears at exactly
    one dimension in the registry, that dimension is used.  Otherwise
    *default_dim* is used.
    """
    def _dim(t: int) -> int:
        if registry is not None:
            return resolve_dim(t, default_dim, registry)
        return default_dim

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
