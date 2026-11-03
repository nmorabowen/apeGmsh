"""Topological dim of one results-file element group, shared by the MPCO
and ``.ladruno`` FEMData-from-file readers (ADR 0131 D10, stage 2).

The code was moved out of ``_femdata_mpco_io`` unchanged, so the
``.ladruno`` reader imports nothing from an ``_mpco*`` module.
"""
from __future__ import annotations

import functools
import warnings
from typing import Iterable


# Substrings on OpenSees class names that map deterministically to a
# topological dim. Names are matched case-insensitively. Searched in
# the order shown — first hit wins, so more specific tokens (e.g.
# "Tri" → triangle) come before more generic ones.
_DIM_BY_CLASS_TOKEN: tuple[tuple[str, int], ...] = (
    # Solids
    ("brick",       3),
    ("hexahedron",  3),
    ("hex",         3),
    ("tet",         3),
    ("wedge",       3),
    ("prism",       3),
    ("pyramid",     3),
    # Surfaces
    ("tri",         2),
    ("quad",        2),
    ("shell",       2),
    ("plate",       2),
    ("membrane",    2),
    # Lines / point elements
    ("beam",        1),
    ("truss",       1),
    ("zerolength",  1),
    ("spring",      1),
    ("dashpot",     1),
    ("frame",       1),
    ("link",        1),
    ("twonodelink", 1),
)


class ElementDimUnknownWarning(UserWarning):
    """A results file names an element class whose topological dim is
    neither in the response catalog nor declared by the file, so the
    FEMData-from-file ``dim`` of that group is a class-name guess.

    Fired once per ``(class, guessed dim)`` per Python session: another
    read of the same file, or of a file holding the same class, adds no
    new fact (#2312)."""


class ElementDimContradictionWarning(UserWarning):
    """A results file declares a topological dim (the ``.ladruno`` BASIS
    ``TOPOLOGY``) that contradicts the response catalog's parent dim for
    that class. The dim is known, so this is not an
    :class:`ElementDimUnknownWarning`: the catalog wins and the file's
    declaration is named as wrong. Fired once per ``(class, declared,
    catalogued)`` per Python session (#2312)."""


# Dim facts already warned this session, so repeated reads warn each once.
# A key enters only after ``warnings.warn`` returns: under an ``error``
# filter the raise leaves it unregistered and the next read raises again.
_DIM_WARNED: "set[tuple]" = set()


@functools.lru_cache(maxsize=1)
def _catalog_parent_dims() -> "dict[str, int]":
    """Element class -> parent (topological) dim, from the response catalog.

    A gauss-level class's dim is the width of its catalogued
    ``natural_coords`` (the parent domain the element integrates over);
    the line-station, closed-form-force and fiber catalogs hold line
    elements (1), the layered-shell catalog surfaces (2), and the
    zero-length catalog classes MPCO files under ``Line_GL_1`` (1). A
    class the catalogs give two dims raises: that is a catalog bug,
    and guessing between them would hide it.
    """
    from ..opensees import _response_catalog as rc

    dims: dict[str, set[int]] = {}
    for (cls, _rule, _token), layout in rc.RESPONSE_CATALOG.items():
        dims.setdefault(cls, set()).add(int(layout.natural_coords.shape[1]))
    for catalog, dim in (
        (rc.CUSTOM_RULE_CATALOG, 1),
        (rc.NODAL_FORCE_CATALOG, 1),
        (rc.FIBER_CATALOG, 1),
        (rc.ZEROLENGTH_CATALOG, 1),
        (rc.LAYER_CATALOG, 2),
    ):
        for cls, _token in catalog:
            dims.setdefault(cls, set()).add(dim)
    conflicting = {c: sorted(d) for c, d in dims.items() if len(d) != 1}
    if conflicting:
        raise RuntimeError(
            f"response catalog gives these element classes more than one "
            f"parent dimension: {conflicting}"
        )
    return {cls: next(iter(d)) for cls, d in dims.items()}


def element_dim(
    class_name: str, npe: int, ndm: int, *, declared: "int | None" = None,
) -> "tuple[int, bool]":
    """Topological dim of one results-file element group.

    Returns ``(dim, known)``. A catalogued class takes its catalog
    parent dimension; otherwise the file's own declaration
    (``declared``, the ``.ladruno`` BASIS ``TOPOLOGY``); otherwise the
    class-name guess, with ``known=False`` so the caller names the
    class through :func:`warn_unknown_dims`. A declaration that
    contradicts the catalog loses to it (the result differs from
    ``declared``); the caller names it through
    :func:`warn_dim_contradictions`.
    """
    catalogued = _catalog_parent_dims().get(class_name)
    if catalogued is not None:
        return catalogued, True
    if declared is not None:
        return declared, True
    return _guess_dim_from_class(class_name, npe, ndm), False


def _warn_new(
    facts: "dict[tuple, str]", head: str, tail: str,
    category: "type[Warning]",
) -> None:
    """One warning listing the facts (key -> text) not yet warned this
    session; the keys are registered only once the warning returns."""
    fresh = {k: t for k, t in facts.items() if k not in _DIM_WARNED}
    if not fresh:
        return
    listing = ", ".join(t for _k, t in sorted(fresh.items()))
    warnings.warn(f"{head}{listing}{tail}", category, stacklevel=4)
    _DIM_WARNED.update(fresh)


def warn_unknown_dims(
    guessed: "Iterable[tuple[str, int]]", *, source: str,
) -> None:
    """One :class:`ElementDimUnknownWarning` naming every guessed
    ``(class, dim)`` pair not yet warned this session. Pairs, not a
    class -> dim map: two groups of one class can guess two dims, and
    each is named."""
    _warn_new(
        {("unknown", c, d): f"{c} (dim={d})" for c, d in guessed},
        f"{source}: element class(es) not in the response catalog and "
        f"with no declared topology, so their dim is a class-name guess: ",
        ". Selection by dim and drawing follow that guess.",
        ElementDimUnknownWarning,
    )


def warn_dim_contradictions(
    contradicted: "Iterable[tuple[str, int, int]]", *, source: str,
) -> None:
    """One :class:`ElementDimContradictionWarning` naming every
    ``(class, declared, catalogued)`` triple not yet warned this session
    (two groups of one class can declare two different dims)."""
    _warn_new(
        {("contradiction", c, d, k): f"{c} (declared {d}, catalog {k})"
         for c, d, k in contradicted},
        f"{source}: the file's declared topological dim contradicts the "
        f"response catalog's parent dim for element class(es) ",
        "; using the catalog's.",
        ElementDimContradictionWarning,
    )


def _guess_dim_from_class(
    class_name: str, npe: int, ndm: int,
) -> int:
    """Best-effort element topological dim.

    Priority:

    1. OpenSees class name substring (most reliable — distinguishes
       a 3-node ``Tri31`` triangle from a 3-node quadratic line).
    2. ``npe`` + ``ndm`` heuristic (the prior behaviour).

    ``ndm`` is the model's spatial dim, captured from the coords
    array shape *before* the 2D-to-3D padding step.
    """
    name = (class_name or "").lower()
    for token, dim in _DIM_BY_CLASS_TOKEN:
        if token in name:
            return dim
    # Fallback: npe + ndm heuristic.
    if npe == 2:
        return 1
    if npe == 3:
        # 3 nodes in a 2-D model is almost certainly a triangle; in a
        # 3-D model it might be a quadratic beam, so honour the model.
        return 2 if ndm == 2 else 1
    if npe in (4, 8) and ndm == 2:
        return 2
    if npe == 4:
        return 3 if ndm == 3 else 2
    if npe in (6, 8, 10, 20, 27):
        return 3
    return ndm
