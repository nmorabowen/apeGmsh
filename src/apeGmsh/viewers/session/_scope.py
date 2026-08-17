"""Scope resolution — one composition axis into ONE cell set.

ADR 0098 §3 INV-MESH-1 / §9: a mesh view's scope is one axis
(physical groups | materials | element types) plus one or more checked
names (or all). Scope chooses **elements**; every layer of the view —
fill, interior grid, outlines, node glyphs, Gauss glyphs, and every
slot — is a function of that one set.

S0 stored the axis + names as opaque tokens; this module resolves them
against ``ViewerData``. Both halves of the cell set are produced,
because the per-kind emit code consumes whichever matches its
topology: element-topology slots (gauss values, line forces) select by
element id, node-topology slots (nodal contour) by node id, and a
selector carrying the wrong one silently paints the wrong thing.

Axis availability (verified against live data, S1-B decision 1):

* ``physical_groups`` — ``ViewerData.elements.physical`` (named index).
* ``element_types`` — ``ViewerData.elements`` groups carry
  ``element_type`` + ``ids``; no selector axis exists, so the names are
  resolved to ids here.
* ``materials`` — **no queryable index exists**: the h5 bridge stores
  material *definitions* under ``/opensees/materials/{family}/{name}``
  and each element's material tag is positional inside
  ``ElementRecord.args``, per element type. Publishing an
  element->material index is a ``data/`` change; until it lands this
  axis refuses loudly rather than drawing an unscoped picture.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import numpy as np

if TYPE_CHECKING:
    from apeGmsh.results.session import Scope
    from ..data import ViewerData


class ScopedSet:
    """The one cell set of a view: element ids + the nodes they touch.

    ``None`` on both halves means "unscoped" — the whole analysis mesh
    (the boot picture), which is NOT the same as an empty set.
    """

    __slots__ = ("element_ids", "node_ids")

    def __init__(
        self,
        element_ids: Optional[np.ndarray] = None,
        node_ids: Optional[np.ndarray] = None,
    ) -> None:
        self.element_ids = element_ids
        self.node_ids = node_ids

    @property
    def is_unscoped(self) -> bool:
        return self.element_ids is None

    def __repr__(self) -> str:  # pragma: no cover - debug aid
        if self.is_unscoped:
            return "<ScopedSet unscoped>"
        return (
            f"<ScopedSet {self.element_ids.size} elements, "
            f"{self.node_ids.size} nodes>"
        )


UNSCOPED = ScopedSet()


def resolve_scope(scope: "Optional[Scope]", view: "ViewerData") -> ScopedSet:
    """The ONE cell set ``scope`` selects (INV-MESH-1)."""
    if scope is None:
        return UNSCOPED
    if scope.axis == "physical_groups":
        element_ids = _physical_group_ids(scope.names, view)
    elif scope.axis == "element_types":
        element_ids = _element_type_ids(scope.names, view)
    elif scope.axis == "materials":
        raise NotImplementedError(
            "The 'materials' scope axis is not realizable yet: no "
            "element->material index is published (the h5 bridge stores "
            "material definitions, and each element's material tag is "
            "positional inside its element args). It needs a data-layer "
            "change to expose the index; use the 'physical_groups' or "
            "'element_types' axis meanwhile."
        )
    else:  # pragma: no cover - Scope.__post_init__ closes the set
        raise ValueError(f"Unknown scope axis {scope.axis!r}.")

    if element_ids.size == 0:
        raise ValueError(
            f"Scope(axis={scope.axis!r}, names={scope.names!r}) selects "
            f"no elements — an empty view is a wrong picture, not an "
            f"empty one. Check the names against the model."
        )
    return ScopedSet(element_ids, _nodes_of_elements(element_ids, view))


def _physical_group_ids(
    names: "Optional[tuple[str, ...]]", view: "ViewerData",
) -> np.ndarray:
    index = view.elements.physical
    wanted = tuple(index.names()) if names is None else names
    _require_known(wanted, index.names(), "physical group")
    chunks = [
        np.asarray(index.element_ids(name), dtype=np.int64)
        for name in wanted
    ]
    return _union(chunks)


def _element_type_ids(
    names: "Optional[tuple[str, ...]]", view: "ViewerData",
) -> np.ndarray:
    available = [t.name for t in view.elements.types]
    wanted = tuple(available) if names is None else names
    _require_known(wanted, available, "element type")
    wanted_set = set(wanted)
    chunks = [
        np.asarray(group.ids, dtype=np.int64)
        for group in view.elements
        if group.element_type.name in wanted_set
    ]
    return _union(chunks)


def _nodes_of_elements(
    element_ids: np.ndarray, view: "ViewerData",
) -> np.ndarray:
    """Every node touched by ``element_ids`` (vectorized per group)."""
    chunks: list[np.ndarray] = []
    for group in view.elements:
        ids = np.asarray(group.ids, dtype=np.int64)
        if ids.size == 0:
            continue
        mask = np.isin(ids, element_ids)
        if not bool(mask.any()):
            continue
        conn = np.asarray(group.connectivity, dtype=np.int64)
        chunks.append(np.unique(conn[mask]))
    return _union(chunks)


def _union(chunks: "list[np.ndarray]") -> np.ndarray:
    if not chunks:
        return np.zeros(0, dtype=np.int64)
    return np.unique(np.concatenate(chunks))


def _require_known(
    wanted: "tuple[str, ...]", available: "list[str] | tuple[str, ...]",
    what: str,
) -> None:
    missing = [name for name in wanted if name not in set(available)]
    if missing:
        raise ValueError(
            f"Unknown {what} name(s) {missing} in this model "
            f"(have: {sorted(available)})."
        )


__all__ = ["ScopedSet", "UNSCOPED", "resolve_scope"]
