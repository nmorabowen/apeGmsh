"""
Topology-diagnostic safeguards: the find_coincident_node_pairs diagnostic
on InspectComposite (P3) and the unconditional print on
mesh.remove_duplicate_nodes.

Motivated by the disjoint-wire-at-OCC-arc-endpoint bug pattern: an
add_ellipse(angle1, angle2) + lines wire that fails to weld at the
junctions, producing two distinct nodes at the same XYZ with no element
or constraint bridging them.
"""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest


# =====================================================================
# P3 — find_coincident_node_pairs diagnostic
# =====================================================================

def _make_inspect(
    *,
    node_ids: list[int],
    node_coords: list[tuple[float, float, float]],
    elements: list[tuple[str, list[tuple[int, tuple[int, ...]]]]] | None = None,
    constraint_pairs: list[tuple[int, int, str]] | None = None,
    pg_to_ids: dict[str, list[int]] | None = None,
):
    """Build an InspectComposite over a minimal hand-rolled fake fem.

    Only the fields the diagnostic reads are populated:

    * ``fem.nodes.ids`` — ndarray of node tags
    * ``fem.nodes.coords`` — ndarray of XYZ
    * ``fem.nodes.select(pg=)`` — returns object with ``.ids`` (optional)
    * ``fem.elements`` — iterable of groups (type_name + iter of (eid, conn))
    * ``fem.nodes.constraints.pairs()`` — iterator of NodePair-shaped objs
    """
    from apeGmsh.mesh.FEMData import InspectComposite

    ids_arr = np.asarray(node_ids, dtype=np.int64)
    coords_arr = np.asarray(node_coords, dtype=np.float64)

    pg_map = pg_to_ids or {}

    def _select(pg=None, **_kw):
        if pg is None or pg not in pg_map:
            return SimpleNamespace(ids=ids_arr)
        return SimpleNamespace(ids=np.asarray(pg_map[pg], dtype=np.int64))

    # Element groups
    class _Grp:
        def __init__(self, type_name, rows):
            self.type_name = type_name
            self._rows = rows

        def __iter__(self):
            for eid, conn in self._rows:
                yield int(eid), tuple(int(n) for n in conn)

    elem_groups = [_Grp(name, rows) for name, rows in (elements or [])]

    # Constraint pair shims
    pair_objs = [
        SimpleNamespace(master_node=int(a), slave_node=int(b), kind=k)
        for a, b, k in (constraint_pairs or [])
    ]
    constraints_stub = SimpleNamespace(pairs=lambda: iter(pair_objs))

    nodes_stub = SimpleNamespace(
        ids=ids_arr,
        coords=coords_arr,
        select=_select,
        constraints=constraints_stub,
    )
    fem_stub = SimpleNamespace(nodes=nodes_stub, elements=elem_groups)
    return InspectComposite(fem_stub)


def test_find_coincident_node_pairs_returns_empty_when_no_duplicates():
    inspect = _make_inspect(
        node_ids=[1, 2, 3],
        node_coords=[(0, 0, 0), (1, 0, 0), (0, 1, 0)],
    )
    assert inspect.find_coincident_node_pairs(tol=1e-6) == {}


def test_find_coincident_node_pairs_unbridged_pair_has_empty_refs():
    """The smoking-gun case — two nodes at the same XYZ, no element or
    constraint touching them together. This is the cimbra arc-endpoint
    signature."""
    inspect = _make_inspect(
        node_ids=[1, 2, 3],
        node_coords=[(0, 0, 0), (1, 1, 0), (1, 1, 0)],  # 2 & 3 coincident
    )
    pairs = inspect.find_coincident_node_pairs(tol=1e-6)
    assert pairs == {(2, 3): []}


def test_find_coincident_node_pairs_records_element_reference():
    """A zeroLength-like element straddling the coincident pair must
    appear in the refs list — that's the legitimate-coincidence
    signature."""
    inspect = _make_inspect(
        node_ids=[1, 2, 3, 4],
        node_coords=[(0, 0, 0), (1, 1, 0), (1, 1, 0), (2, 0, 0)],
        elements=[
            ("zeroLength", [(100, (2, 3))]),
            ("elasticBeamColumn", [(200, (1, 4))]),  # unrelated
        ],
    )
    pairs = inspect.find_coincident_node_pairs(tol=1e-6)
    assert (2, 3) in pairs
    refs = pairs[(2, 3)]
    assert any("zeroLength" in r and "#100" in r for r in refs)


def test_find_coincident_node_pairs_records_constraint_reference():
    inspect = _make_inspect(
        node_ids=[1, 2, 3],
        node_coords=[(0, 0, 0), (5, 5, 5), (5, 5, 5)],
        constraint_pairs=[(2, 3, "equal_dof")],
    )
    pairs = inspect.find_coincident_node_pairs(tol=1e-6)
    assert pairs == {(2, 3): ["constraint equal_dof"]}


def test_find_coincident_node_pairs_combined_refs():
    """A pair tied by BOTH an element and a constraint surfaces both
    references (so the user can confirm the connection is intentional)."""
    inspect = _make_inspect(
        node_ids=[1, 2],
        node_coords=[(0, 0, 0), (0, 0, 0)],
        elements=[("zeroLength", [(7, (1, 2))])],
        constraint_pairs=[(1, 2, "equal_dof")],
    )
    pairs = inspect.find_coincident_node_pairs(tol=1e-6)
    assert (1, 2) in pairs
    refs = pairs[(1, 2)]
    assert any("zeroLength" in r for r in refs)
    assert any("equal_dof" in r for r in refs)


def test_find_coincident_node_pairs_pg_filter_restricts_scan():
    """If pg= is given, nodes outside the PG must not appear in
    coincident-pair tuples even when they happen to share coordinates
    with PG members."""
    inspect = _make_inspect(
        node_ids=[1, 2, 3, 4],
        # 1 & 2 coincident; 3 & 4 also coincident (different XYZ)
        node_coords=[(0, 0, 0), (0, 0, 0), (1, 1, 1), (1, 1, 1)],
        pg_to_ids={"only_first_pair": [1, 2]},
    )
    pairs = inspect.find_coincident_node_pairs(
        tol=1e-6, pg="only_first_pair")
    assert pairs == {(1, 2): []}


def test_find_coincident_node_pairs_element_partial_overlap_not_credited():
    """An element that references only ONE of the two coincident nodes
    is NOT a bridge — the pair has no shared element, so the refs list
    should remain empty for that element. (Catches a regression where
    the inverted-index scan over-credits.)"""
    inspect = _make_inspect(
        node_ids=[1, 2, 3, 4],
        node_coords=[(0, 0, 0), (1, 1, 0), (1, 1, 0), (2, 0, 0)],
        elements=[("elasticBeamColumn", [(50, (1, 2))])],  # touches 2 not 3
    )
    pairs = inspect.find_coincident_node_pairs(tol=1e-6)
    assert pairs[(2, 3)] == []  # NOT bridged


# =====================================================================
# Node-removal: unconditional print
# =====================================================================

def test_remove_duplicate_nodes_always_prints(g, capsys):
    """``mesh.editing.remove_duplicate_nodes()`` must always announce
    itself on stdout — the silent ``verbose=False`` mode was removed
    because deleting nodes from a meshed model is destructive and
    should never hide in a long pipeline log."""
    # Minimal model: one volume box, meshed.
    g.model.geometry.add_box(0, 0, 0, 1, 1, 1, label="box")
    g.model.sync()
    g.mesh.generation.generate(3)
    capsys.readouterr()  # drop meshing chatter

    g.mesh.editing.remove_duplicate_nodes()
    out = capsys.readouterr().out
    assert "remove_duplicate_nodes" in out
    # Either branch (merged > 0 OR no duplicates) is acceptable —
    # both go through the unconditional print path.
    assert ("merged" in out) or ("no duplicates" in out)


def test_remove_duplicate_nodes_no_silent_kwarg():
    """The pre-change ``verbose=`` parameter was dropped. Passing it
    must now raise TypeError so callers can't accidentally re-silence
    the print path."""
    from apeGmsh.mesh._mesh_editing import _Editing
    import inspect
    sig = inspect.signature(_Editing.remove_duplicate_nodes)
    # self is the only formal parameter — no verbose / no silent.
    params = list(sig.parameters.keys())
    assert params == ["self"], (
        f"remove_duplicate_nodes should expose no caller-tunable "
        f"verbosity flag; found {params}"
    )
