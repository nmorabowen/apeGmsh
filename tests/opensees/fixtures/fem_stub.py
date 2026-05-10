"""
Hand-rolled FEMData-shaped stub for Phase 4 integration / parity tests.

Mirrors the surface the build pipeline uses:

  * ``fem.nodes.index(node_id)`` -> int (array index)
  * ``fem.nodes.coords`` -> ndarray (N, 3)
  * ``fem.nodes.get(pg=name).ids`` -> ndarray
  * ``fem.elements.get(pg=name)`` -> iterable yielding objects that
    iterate as ``(eid, conn_tuple)`` pairs

Building a real :class:`apeGmsh.mesh.FEMData` requires a live Gmsh
session; the bridge talks to FEMData through a narrow surface, so a
stub keeps tests fast and pure-Python.

The stub is intentionally read-only and mutation-free; tests build a
fresh instance per test or reuse a session-scoped one.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy import ndarray


@dataclass(frozen=True)
class _NodeResult:
    """Stand-in for :class:`apeGmsh.mesh.FEMData.NodeResult`.

    Carries the ``ids`` and ``coords`` arrays; nothing else is read by
    the build pipeline.
    """

    ids: ndarray
    coords: ndarray


@dataclass(frozen=True)
class _ElementGroupView:
    """Stand-in for :class:`apeGmsh.mesh._element_types.ElementGroup`.

    Iterates as ``(eid, conn_tuple)`` pairs, matching ElementGroup's
    iter contract used by :func:`expand_pg_to_elements`.
    """

    ids: tuple[int, ...]
    connectivity: tuple[tuple[int, ...], ...]

    def __iter__(self):
        for eid, conn in zip(self.ids, self.connectivity):
            yield eid, conn

    def __len__(self) -> int:
        return len(self.ids)


@dataclass(frozen=True)
class _GroupResultView:
    """Iterable of :class:`_ElementGroupView` — one per element type.

    The build pipeline iterates this directly; nothing else is read.
    """

    _groups: tuple[_ElementGroupView, ...]

    def __iter__(self):
        return iter(self._groups)

    def __bool__(self) -> bool:
        return len(self._groups) > 0


class _NodesStub:
    """Stand-in for :class:`apeGmsh.mesh.FEMData.NodeComposite`."""

    def __init__(
        self,
        ids: list[int],
        coords: list[tuple[float, float, float]],
        node_pgs: dict[str, list[int]],
    ) -> None:
        self._ids = np.asarray(ids, dtype=np.int64)
        self._coords = np.asarray(coords, dtype=np.float64)
        self._id_to_idx = {int(n): i for i, n in enumerate(self._ids)}
        self._pgs = {k: list(v) for k, v in node_pgs.items()}

    @property
    def ids(self) -> ndarray:
        return self._ids

    @property
    def coords(self) -> ndarray:
        return self._coords

    def index(self, node_id: int) -> int:
        try:
            return self._id_to_idx[int(node_id)]
        except KeyError as e:
            raise KeyError(
                f"node id {node_id} not in fixture; have "
                f"{sorted(self._id_to_idx.keys())}"
            ) from e

    def get(
        self,
        target: object | None = None,
        *,
        pg: str | None = None,
        label: str | None = None,
        tag: int | None = None,
        partition: int | None = None,
        dim: int | None = None,
    ) -> _NodeResult:
        if pg is None:
            raise ValueError("fem-stub.nodes.get: only pg= is supported")
        if pg not in self._pgs:
            raise KeyError(
                f"node pg {pg!r} not in fixture; have {sorted(self._pgs.keys())}"
            )
        ids = np.asarray(self._pgs[pg], dtype=object)
        idxs = [self._id_to_idx[int(n)] for n in ids]
        coords = self._coords[idxs] if idxs else np.empty((0, 3))
        return _NodeResult(ids=ids, coords=np.asarray(coords))


class _ElementsStub:
    """Stand-in for :class:`apeGmsh.mesh.FEMData.ElementComposite`."""

    def __init__(
        self,
        elem_pgs: dict[str, _ElementGroupView],
    ) -> None:
        self._pgs = dict(elem_pgs)

    def get(
        self,
        target: object | None = None,
        *,
        pg: str | None = None,
        label: str | None = None,
        tag: int | None = None,
        dim: int | None = None,
        element_type: str | int | None = None,
        partition: int | None = None,
    ) -> _GroupResultView:
        if pg is None:
            raise ValueError("fem-stub.elements.get: only pg= is supported")
        if pg not in self._pgs:
            raise KeyError(
                f"element pg {pg!r} not in fixture; have "
                f"{sorted(self._pgs.keys())}"
            )
        return _GroupResultView(_groups=(self._pgs[pg],))


class FEMStub:
    """Hand-rolled FEMData-shaped fixture for Phase-4 tests."""

    def __init__(
        self,
        nodes: _NodesStub,
        elements: _ElementsStub,
    ) -> None:
        self.nodes = nodes
        self.elements = elements


def make_two_node_beam() -> FEMStub:
    """Two nodes + one line element, both in PGs ``"Cols"`` and base.

    Geometry:
      * node 1 at origin
      * node 2 at (0, 0, 1) — vertical column

    PGs:
      * ``"Cols"``: element 1 (the vertical line)
      * ``"Base"``: node 1
      * ``"Top"``:  node 2
    """
    nodes = _NodesStub(
        ids=[1, 2],
        coords=[(0.0, 0.0, 0.0), (0.0, 0.0, 1.0)],
        node_pgs={"Base": [1], "Top": [2]},
    )
    elements = _ElementsStub(
        elem_pgs={
            "Cols": _ElementGroupView(
                ids=(1,), connectivity=((1, 2),),
            ),
        },
    )
    return FEMStub(nodes=nodes, elements=elements)


def make_two_column_frame() -> FEMStub:
    """Two parallel columns sharing a common base PG.

    Geometry:
      * node 1 at (0, 0, 0)   - base of column A
      * node 2 at (0, 0, 1)   - top of column A
      * node 3 at (1, 0, 0)   - base of column B
      * node 4 at (1, 0, 1)   - top of column B

    PGs:
      * ``"Cols"``: elements 1 and 2 (both vertical columns)
      * ``"Base"``: nodes 1 and 3
      * ``"Top"``:  nodes 2 and 4
    """
    nodes = _NodesStub(
        ids=[1, 2, 3, 4],
        coords=[
            (0.0, 0.0, 0.0),
            (0.0, 0.0, 1.0),
            (1.0, 0.0, 0.0),
            (1.0, 0.0, 1.0),
        ],
        node_pgs={"Base": [1, 3], "Top": [2, 4]},
    )
    elements = _ElementsStub(
        elem_pgs={
            "Cols": _ElementGroupView(
                ids=(1, 2),
                connectivity=((1, 2), (3, 4)),
            ),
        },
    )
    return FEMStub(nodes=nodes, elements=elements)


def make_arch_with_csys_fan_out() -> FEMStub:
    """Three-segment arch — non-collinear elements drive distinct vecxz under
    cylindrical CS.

    Geometry: three line elements arranged so the unit tangent rotates
    between elements — a coarse approximation of an arch in the X-Z
    plane.

      * node 1 at (1, 0, 0)
      * node 2 at (cos 30°, 0, sin 30°)
      * node 3 at (cos 60°, 0, sin 60°)
      * node 4 at (0, 0, 1)

    Each consecutive pair forms one element.

    PGs:
      * ``"Arch"``: elements 1, 2, 3 (the three arch segments)
    """
    import math
    nodes = _NodesStub(
        ids=[1, 2, 3, 4],
        coords=[
            (1.0, 0.0, 0.0),
            (math.cos(math.pi / 6), 0.0, math.sin(math.pi / 6)),
            (math.cos(math.pi / 3), 0.0, math.sin(math.pi / 3)),
            (0.0, 0.0, 1.0),
        ],
        node_pgs={"Crown": [4], "Springing": [1]},
    )
    elements = _ElementsStub(
        elem_pgs={
            "Arch": _ElementGroupView(
                ids=(1, 2, 3),
                connectivity=((1, 2), (2, 3), (3, 4)),
            ),
        },
    )
    return FEMStub(nodes=nodes, elements=elements)
