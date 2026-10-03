"""An element selection's accessors agree on order (``MeshSelection``).

``.ids``, ``.coords``, ``.connectivity`` and the ``(eid, conn)`` pair view
must describe the same elements in the same order, so ``zip(sel.ids,
sel.connectivity)`` pairs each element with its own nodes. Before
2026-10 ``.ids`` followed the selection's id order (a set for a ``pg=``
seed) while ``.connectivity`` and the pair view followed the mesh's
storage order, so on a mesh stored unsorted (an STKO import keeps its
own ids) the pairs were silently wrong (ADR 0111 review, MAJOR-2).
"""
from __future__ import annotations

from typing import Any

import gmsh
import numpy as np
import pytest

from apeGmsh import apeGmsh

#: Storage order of the line elements. A ``pg=`` seed resolves them
#: through a set, which iterates 16, 9, 1, so the two orders differ.
_STORED = [9, 1, 16, 4]
_CONN = {9: (1, 2), 1: (2, 3), 16: (3, 4), 4: (4, 5)}


@pytest.fixture
def session() -> Any:
    with apeGmsh(model_name="sel_order", verbose=False) as g:
        ent = gmsh.model.addDiscreteEntity(1)
        gmsh.model.mesh.addNodes(
            1, ent, [1, 2, 3, 4, 5],
            [0, 0, 0, 1, 0, 0, 2, 0, 0, 3, 0, 0, 4, 0, 0])
        flat = [n for e in _STORED for n in _CONN[e]]
        gmsh.model.mesh.addElementsByType(ent, 1, _STORED, flat)
        gmsh.model.addPhysicalGroup(1, [ent], name="L")
        yield g


@pytest.fixture
def fem(session: Any) -> Any:
    return session.mesh.queries.get_fem_data(dim=1)


def _assert_aligned(sel: Any) -> None:
    ids = sel.ids
    conn = np.asarray(sel.connectivity)
    assert conn.shape[0] == len(ids)
    for eid, row in zip(ids, conn):
        assert tuple(int(n) for n in row) == _CONN[eid], eid
    pairs = list(sel)
    assert [eid for eid, _ in pairs] == ids
    for eid, row in pairs:
        assert tuple(int(n) for n in row) == _CONN[eid]
    centroid_x = [(_CONN[e][0] + _CONN[e][1]) / 2.0 - 1.0 for e in ids]
    np.testing.assert_allclose(sel.coords[:, 0], centroid_x)


def test_fixture_seed_order_differs_from_storage(fem: Any) -> None:
    # The premise of the lock: without it the test would pass vacuously.
    assert list(fem.elements.ids) == _STORED
    assert fem.elements.select(pg="L").ids != _STORED


@pytest.mark.parametrize("kw", [
    {"pg": "L"},
    {"pg": "L", "dim": 1},
    {"ids": [16, 9, 4]},
    {},
], ids=["pg", "pg+dim", "ids", "all"])
def test_broker_accessors_share_the_id_order(fem: Any, kw: dict) -> None:
    _assert_aligned(fem.elements.select(**kw))


def test_order_survives_chaining(fem: Any) -> None:
    sel = fem.elements.select(pg="L")
    _assert_aligned(sel - fem.elements.select(ids=[1]))
    _assert_aligned(sel.nearest_to((3.5, 0.0, 0.0), count=3))


def test_unknown_id_fails_loud(fem: Any) -> None:
    sel = fem.elements.select(ids=[9, 999])
    with pytest.raises(KeyError, match="999"):
        sel.connectivity
    with pytest.raises(KeyError, match="999"):
        list(sel)


def test_live_mesh_accessors_share_the_id_order(session: Any) -> None:
    sel = session.mesh_selection.select(
        level="element", dim=1, ids=[16, 9, 4, 1])
    _assert_aligned(sel)
