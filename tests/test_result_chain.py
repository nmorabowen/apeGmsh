"""S3c — ResultChain + ``results.nodes.select()`` / ``results.elements.select()``.

Phase S3c of the selection-unification work
(``docs/plans/selection-unification.md`` §5/§6).  S3a/S3b landed
``GeometryChain`` (entity family) and ``NodeChain``/``ElementChain``
(point family) over the broker.  S3c adds the **results** point-family
chainable: a single ``ResultChain`` whose atoms are node ids *or*
element ids (a tiny ``_ResultChainEngine`` level discriminator carries
which), and wires ``results.nodes.select(...)`` /
``results.elements.select(...)`` as **additive** host hooks that
delegate name resolution verbatim to the *existing* results resolvers
(``_SelectionMixin._resolve_node_ids`` /
``_ElementGeometryMixin._combine_candidates`` → ``_resolve_element_ids``
— the exact methods ``.get()`` / ``.in_box()`` / ``.nearest_to()``
already use).

What this locks:

* ``results.nodes.select(...)`` / ``results.elements.select(...)``
  return a ``ResultChain`` (point family) seeded by delegating to the
  existing ``_resolve_*`` / ``_combine_candidates`` (a spy proves the
  call — delegation, not a re-implementation; the resolvers are
  untouched).
* Node & element daisy-chaining
  ``.select(pg=...).in_box(...).on_plane(...)`` composes, every verb
  returning the same concrete ``ResultChain`` type.
* Point-family ``in_box`` is half-open ``[lo, hi)`` by default and
  closed ``[lo, hi]`` with ``inclusive=True`` (R4).
* Set algebra ``| & - ^`` with insertion-order dedup; cross-engine /
  cross-level combination is loud.
* Terminal ``.get(component=...)`` returns the **existing** slab type
  (``NodeSlab`` / ``ElementSlab``) with id/value parity to
  ``results.<level>.get(ids=<equivalent>, component=...)``.
* ``.result()`` (no component) fails loud with a directive message.
* Element centroids fail loud on an unknown node id (never the
  ``np.clip`` silent row-substitution ``_element_centroids`` does).
* The existing ``results.nodes.get`` / ``.in_box`` / ``.nearest_to``
  byte-behaviour is unchanged (the additive ``.select()`` does not
  perturb it).

No ``openseespy`` dependency: a synthetic native HDF5 results file +
``SimpleNamespace`` mock FEM (the exact pattern
``test_results_spatial_filters.py`` uses), extended with element-level
per-element-node force data so the element terminal reads a real
``ElementSlab``.
"""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from apeGmsh._kernel.chain import SelectionChain
from apeGmsh.results import Results
from apeGmsh.results._composites import (
    ElementResultsComposite,
    NodeResultsComposite,
)
# selection-unification-v2 P3-R (§6.2 / §6.3): the legacy
# ``ResultChain`` class is **deleted**; ``_ResultChainEngine`` +
# ``VALID_LEVELS`` are **relocated** verbatim to
# ``apeGmsh.results._result_engine`` (plan §3 — pure typing-only
# leaf, no behaviour change).  The results ``.select()`` hooks return
# the v2 terminal ``MeshSelection``; the chain results read is the
# verbatim rename ``ResultChain.get`` → ``MeshSelection.values``
# (R5).  The element-centroid path now iterates
# ``fem.elements._groups.values()`` directly (M-STOP-3) so the mock
# FEM exposes ``_groups`` (disposition 4).  The legacy
# ``ResultChain`` structural-shape / __init_subclass__ tests are
# deleted (now covered by ``tests/test_selection_idiom.py``); the
# slab-read parity stays (the typed ``results.<lvl>.get(component=)``
# reader is RETAINED — category E, no rewrite).
from apeGmsh.results._result_engine import VALID_LEVELS, _ResultChainEngine
from apeGmsh.mesh._mesh_selection import MeshSelection
from apeGmsh.results._slabs import ElementSlab, NodeSlab
from apeGmsh.results.writers import NativeWriter

from tests.conftest import _open_model_from_h5


# =====================================================================
# Synthetic results file + mock FEM (no openseespy)
# =====================================================================

def _make_results_with_fem(tmp_path: Path, *, with_selection: bool = False):
    """4 unit-square corner nodes (z=0) + a far node 5; one quad elem 10.

    Nodes:  1→(0,0,0) 2→(1,0,0) 3→(1,1,0) 4→(0,1,0) 5→(5,5,5)
    Element 10 = quad4 over nodes 1-2-3-4 (centroid (0.5,0.5,0.0)).

    Node-level component ``displacement_x`` (T=2) and an element-level
    per-element-node force component ``globalForce`` (T=2, E=1, npe=4)
    so the element terminal ``.get`` reads a real ``ElementSlab``.

    ``with_selection=True`` adds a ``mesh_selection`` mock so the
    ``selection=`` delegation path can be exercised.
    """
    path = tmp_path / "synthetic.h5"
    time = np.array([0.0, 1.0])
    node_ids = np.array([1, 2, 3, 4, 5], dtype=np.int64)
    ux = np.array([[0.1, 0.2, 0.3, 0.4, 0.5],
                   [1.1, 1.2, 1.3, 1.4, 1.5]])

    elem_idx = np.array([10], dtype=np.int64)
    # globalForce: (T=2, E=1, npe=4)
    gforce = np.array(
        [[[10.0, 11.0, 12.0, 13.0]],
         [[20.0, 21.0, 22.0, 23.0]]],
        dtype=np.float64,
    )

    with NativeWriter(path) as w:
        w.open(source_type="domain_capture")
        sid = w.begin_stage(name="static", kind="static", time=time)
        w.write_nodes(sid, "partition_0", node_ids=node_ids,
                      components={"displacement_x": ux})
        w.write_nodal_forces_group(
            sid, "partition_0", "group_0",
            class_tag=1, frame="global",
            element_index=elem_idx,
            components={"globalForce": gforce},
        )
        w.end_stage()

    coords = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 1.0, 0.0],
        [0.0, 1.0, 0.0],
        [5.0, 5.0, 5.0],
    ], dtype=np.float64)

    type_info = SimpleNamespace(name="quad4")

    def _resolve(*, element_type=None):
        return (
            np.array([10], dtype=np.int64),
            np.array([[1, 2, 3, 4]], dtype=np.int64),
        )

    # selection-unification-v2 P3-R / §6.3 M-STOP-3 + disposition 4:
    # the results element-centroid path (``_centroid_map_result``) now
    # iterates ``fem.elements._groups.values()`` directly (byte-
    # equivalent to the removed per-type ``fem.elements.resolve(
    # element_type=)`` loop).  The mock mirrors that: one
    # ``ElementGroup``-shaped group (``.ids`` / ``.connectivity`` /
    # ``.type_name``) carrying the SAME (ids, conn) the legacy
    # ``_resolve`` returned.  ``_egroup`` is the single mutable group so
    # the fail-loud test can corrupt its connectivity (the M-STOP-3
    # path reads ``_groups``, not ``resolve``).
    _egroup = SimpleNamespace(
        ids=np.array([10], dtype=np.int64),
        connectivity=np.array([[1, 2, 3, 4]], dtype=np.int64),
        type_name="quad4",
    )

    nodes_ns = SimpleNamespace(
        ids=node_ids,
        coords=coords,
        physical=SimpleNamespace(node_ids=lambda n: {
            "TopRow": np.array([3, 4], dtype=np.int64),
            "Single": np.array([5], dtype=np.int64),
        }[n]),
        labels=SimpleNamespace(
            node_ids=lambda n: np.array([], dtype=np.int64),
        ),
    )
    elements_ns = SimpleNamespace(
        ids=np.array([10], dtype=np.int64),
        types=[type_info],
        resolve=_resolve,
        _groups={0: _egroup},          # P3-R M-STOP-3 (disposition 4)
        physical=SimpleNamespace(
            element_ids=lambda n: np.array([10], dtype=np.int64),
        ),
        labels=SimpleNamespace(
            element_ids=lambda n: np.array([], dtype=np.int64),
        ),
    )

    fem = SimpleNamespace(
        snapshot_id="testhash",
        nodes=nodes_ns,
        elements=elements_ns,
    )
    if with_selection:
        fem.mesh_selection = SimpleNamespace(
            node_ids=lambda n: np.array([1, 2], dtype=np.int64),
            element_ids=lambda n: np.array([10], dtype=np.int64),
        )

    return Results.from_native(path, fem=fem, model=_open_model_from_h5(path)), fem


def _sids(seq) -> list[int]:
    """Sorted ids from a chain or a plain id sequence.

    selection-unification-v2 P2-I (§6.1 STOP-2(b)): the results host
    now returns ``MeshSelection`` whose ``__iter__`` yields
    ``(id, payload)`` pairs (the ratified HT8 design).  Set-algebra /
    identity is defined on the ``_items`` atoms, which are *unchanged*
    by the pair-view — so for a chain read ``_items``; a plain id
    sequence (e.g. a slab's ``.ids``) iterates as bare ids."""
    if isinstance(seq, SelectionChain):
        return sorted(int(a) for a in seq._items)
    return sorted(int(x) for x in seq)


# =====================================================================
# Class-shape invariant — the host hooks return the v2 terminal; the
# relocated _ResultChainEngine still validates its level.
# (the legacy ResultChain structural-shape + __init_subclass__ gate
# tests are deleted; covered by tests/test_selection_idiom.py)
# =====================================================================

def test_select_returns_point_family_meshselection(tmp_path):
    r, _fem = _make_results_with_fem(tmp_path)
    n = r.nodes.select(ids=[1])
    e = r.elements.select(ids=[10])
    assert isinstance(n, MeshSelection) and isinstance(e, MeshSelection)
    assert issubclass(MeshSelection, SelectionChain)
    assert n.FAMILY == "point" and e.FAMILY == "point"
    assert VALID_LEVELS == ("node", "element")


def test_relocated_engine_rejects_invalid_level(tmp_path):
    # _ResultChainEngine relocated to apeGmsh.results._result_engine
    # (P3-R §3, behaviour-verbatim); its level guard is unchanged.
    r, _fem = _make_results_with_fem(tmp_path)
    with pytest.raises(ValueError, match="level.*invalid"):
        _ResultChainEngine(r, r.nodes, "bogus")


# =====================================================================
# .select() host hook — additive, delegates to the existing resolver
# =====================================================================

def test_node_select_returns_resultchain_node_level(tmp_path):
    r, _fem = _make_results_with_fem(tmp_path)
    sel = r.nodes.select(pg="TopRow")
    assert isinstance(sel, MeshSelection)      # P2-I: was ResultChain
    assert sel.FAMILY == "point"
    assert sel._level == "node"
    assert _sids(sel) == [3, 4]
    # no selector → every domain node
    assert _sids(r.nodes.select()) == [1, 2, 3, 4, 5]
    # explicit ids=
    assert _sids(r.nodes.select(ids=[2, 5])) == [2, 5]


def test_element_select_returns_resultchain_element_level(tmp_path):
    r, _fem = _make_results_with_fem(tmp_path)
    sel = r.elements.select(pg="Beams")
    assert isinstance(sel, MeshSelection)      # P2-I: was ResultChain
    assert sel._level == "element"
    assert _sids(sel) == [10]
    assert _sids(r.elements.select()) == [10]
    assert _sids(r.elements.select(ids=[10])) == [10]


def test_node_select_delegates_to_resolve_node_ids(tmp_path, monkeypatch):
    # Prove the host hook calls the EXISTING results node resolver
    # (_SelectionMixin._resolve_node_ids — the one .get()/.in_box() use)
    # rather than re-implementing the pg/label/selection/ids tiers.
    r, _fem = _make_results_with_fem(tmp_path)
    seen = {}
    real = NodeResultsComposite._resolve_node_ids

    def _spy(self, *, pg, label, selection, ids):
        seen["called"] = True
        seen["pg"] = pg
        return real(self, pg=pg, label=label, selection=selection, ids=ids)

    monkeypatch.setattr(NodeResultsComposite, "_resolve_node_ids", _spy)
    sel = r.nodes.select(pg="TopRow")
    assert seen.get("called") is True
    assert seen.get("pg") == "TopRow"
    assert _sids(sel) == [3, 4]


def test_element_select_delegates_to_combine_candidates(tmp_path, monkeypatch):
    # Prove the element host hook calls the EXISTING
    # _ElementGeometryMixin._combine_candidates (which itself calls the
    # existing _resolve_element_ids) — the exact path the element
    # geometry helpers use. Delegation, not re-implementation.
    r, _fem = _make_results_with_fem(tmp_path)
    seen = {}
    real_combine = ElementResultsComposite._combine_candidates
    real_resolve = ElementResultsComposite._resolve_element_ids

    def _spy_combine(self, *, pg, label, selection, ids, element_type):
        seen["combine"] = True
        return real_combine(
            self, pg=pg, label=label, selection=selection, ids=ids,
            element_type=element_type,
        )

    def _spy_resolve(self, *, pg, label, selection, ids):
        seen["resolve"] = True
        return real_resolve(
            self, pg=pg, label=label, selection=selection, ids=ids,
        )

    monkeypatch.setattr(
        ElementResultsComposite, "_combine_candidates", _spy_combine,
    )
    monkeypatch.setattr(
        ElementResultsComposite, "_resolve_element_ids", _spy_resolve,
    )
    sel = r.elements.select(pg="Beams")
    assert seen.get("combine") is True
    assert seen.get("resolve") is True
    assert _sids(sel) == [10]


def test_select_delegates_selection_tier(tmp_path):
    # selection= must flow through the existing _resolve_* mesh_selection
    # branch (delegation), not be re-implemented in the chain.
    r, _fem = _make_results_with_fem(tmp_path, with_selection=True)
    assert _sids(r.nodes.select(selection="MySet")) == [1, 2]
    assert _sids(r.elements.select(selection="MySet")) == [10]


def test_select_rejects_multiple_selectors(tmp_path):
    # The existing _resolve_* contract (ids= XOR named) is preserved by
    # reuse — select() inherits its loudness.
    r, _fem = _make_results_with_fem(tmp_path)
    with pytest.raises(ValueError, match="not multiple"):
        r.nodes.select(pg="TopRow", ids=[1])
    with pytest.raises(ValueError, match="not multiple"):
        r.elements.select(pg="Beams", ids=[10])


# =====================================================================
# Daisy-chaining + point-family spatial semantics
# =====================================================================

def test_node_chain_daisychains_each_verb_returns_resultchain(tmp_path):
    r, _fem = _make_results_with_fem(tmp_path)
    step1 = r.nodes.select()
    step2 = step1.in_box((-1, -1, -1), (2, 2, 2))
    step3 = step2.on_plane((0, 0, 0), (0, 0, 1), tol=1e-9)
    for s in (step1, step2, step3):
        assert isinstance(s, MeshSelection)    # P2-I: was ResultChain
        assert s._level == "node"
    # full one-liner: z=0 plane keeps the 4 corner nodes (node 5 at z=5)
    chained = (r.nodes.select()
                .in_box((-1, -1, -1), (2, 2, 2))
                .on_plane((0, 0, 0), (0, 0, 1), tol=1e-9))
    assert _sids(chained) == [1, 2, 3, 4]


def test_element_chain_daisychains_on_centroid(tmp_path):
    r, _fem = _make_results_with_fem(tmp_path)
    # The single quad centroid is (0.5, 0.5, 0.0).
    chained = (r.elements.select()
                .in_box((-1, -1, -1), (2, 2, 2))
                .on_plane((0, 0, 0), (0, 0, 1), tol=1e-9))
    assert isinstance(chained, MeshSelection)  # P2-I: was ResultChain
    assert chained._level == "element"
    assert _sids(chained) == [10]
    # A plane that misses the centroid drops it.
    assert _sids(
        r.elements.select().on_plane((0, 0, 5), (0, 0, 1), tol=0.1)
    ) == []


def test_point_family_in_box_half_open_default_and_inclusive(tmp_path):
    r, _fem = _make_results_with_fem(tmp_path)
    alln = r.nodes.select()
    # Tight box on (0,0,0)-(1,1,1): half-open [a,b) excludes x=1 / y=1
    # → only node 1 at the origin survives. inclusive=True restores the
    # 4 corner nodes (node 5 at (5,5,5) still out).
    half = alln.in_box((0.0, 0.0, 0.0), (1.0, 1.0, 1.0))
    closed = alln.in_box((0.0, 0.0, 0.0), (1.0, 1.0, 1.0), inclusive=True)
    assert _sids(half) == [1]
    assert _sids(closed) == [1, 2, 3, 4]


def test_point_family_sphere_plane_nearest_where(tmp_path):
    r, _fem = _make_results_with_fem(tmp_path)
    alln = r.nodes.select()

    # in_sphere: closed ball radius 1.5 at origin → nodes 1,2,3,4
    # (node 3 dist sqrt(2)≈1.414 in; node 5 far out).
    assert _sids(alln.in_sphere((0, 0, 0), 1.5)) == [1, 2, 3, 4]
    assert _sids(alln.in_sphere((5, 5, 5), 0.0)) == [5]

    # nearest_to: order by distance, count caps.
    near1 = alln.nearest_to((0.1, 0.1, 0.0), count=1)
    assert _sids(near1) == [1]
    assert len(alln.nearest_to((0, 0, 0), count=3)) == 3

    # on_plane z=0 → 4 corner nodes; non-unit normal must be normalised.
    assert _sids(
        alln.on_plane((0, 0, 0), (0, 0, 100.0), tol=1e-6)
    ) == [1, 2, 3, 4]

    # where: predicate on the coord row → nodes with x < 0.5 are 1 & 4.
    assert _sids(alln.where(lambda xyz: xyz[0] < 0.5)) == [1, 4]

    # point-family input validation is loud.
    with pytest.raises(ValueError, match="radius must be non-negative"):
        alln.in_sphere((0, 0, 0), -1.0)
    with pytest.raises(ValueError, match="tolerance must be non-negative"):
        alln.on_plane((0, 0, 0), (0, 0, 1), tol=-1.0)
    with pytest.raises(ValueError, match="normal vector has zero length"):
        alln.on_plane((0, 0, 0), (0, 0, 0), tol=1e-6)


# =====================================================================
# Set algebra — insertion-order dedup; cross-* is loud
# =====================================================================

def test_set_algebra_union_intersect_difference_symmetric(tmp_path):
    r, _fem = _make_results_with_fem(tmp_path)
    a = r.nodes.select(ids=[1, 2, 3])
    b = r.nodes.select(ids=[2, 3, 4, 5])

    assert len(a | b) == 5                     # union (3 + 4 - 2 dup)
    assert _sids(a & b) == [2, 3]              # intersection
    assert _sids(a - b) == [1]                 # difference
    assert _sids(a ^ b) == [1, 4, 5]           # symmetric difference
    assert len(a | a) == 3                     # idempotent (one law)
    # compare on the ATOMS (``_items``); P2-I
    # ``MeshSelection.__iter__`` yields ``(id, payload)`` pairs
    # (ndarray payload → ambiguous tuple ``==``), but set-algebra is
    # defined on ``_items`` and is unaffected by the pair-view.
    assert (a.union(b))._items == (a | b)._items
    assert (a.difference(b))._items == (a - b)._items
    for s in (a | b, a & b, a - b, a ^ b):
        assert isinstance(s, MeshSelection)    # P2-I: was ResultChain


def test_cross_level_and_cross_engine_set_algebra_is_loud(tmp_path):
    r, _fem = _make_results_with_fem(tmp_path)
    nc = r.nodes.select(ids=[1])
    ec = r.elements.select(ids=[10])
    # cross-level: both ResultChain but bound to different engine
    # adapters (different identity) → loud (different engines).
    with pytest.raises(TypeError, match="different engines"):
        nc | ec

    # cross-engine: two different Results → loud.
    second = tmp_path / "second"
    second.mkdir()
    r2, _fem2 = _make_results_with_fem(second)
    other = r2.nodes.select(ids=[1])
    with pytest.raises(TypeError, match="different engines"):
        nc | other


# =====================================================================
# Terminal .get(component=) — EXISTING slab type + PARITY with get()
# =====================================================================

def test_node_terminal_get_is_nodeslab_and_parity(tmp_path):
    r, _fem = _make_results_with_fem(tmp_path)
    # select(pg=).get(component=) ≡ get(ids=<equiv>, component=)
    chained = r.nodes.select(pg="TopRow")
    slab = chained.values(component="displacement_x")  # P2-I: .get→.values
    assert isinstance(slab, NodeSlab)
    assert type(slab).__name__ == "NodeSlab"

    equiv = r.nodes.get(ids=[3, 4], component="displacement_x")
    assert _sids(slab.node_ids) == _sids(equiv.node_ids)
    np.testing.assert_array_equal(
        slab.values[:, np.argsort(slab.node_ids)],
        equiv.values[:, np.argsort(equiv.node_ids)],
    )

    # spatial daisy-chain BEFORE the terminal read
    spatial = (r.nodes.select()
                .on_plane((0, 0, 0), (0, 0, 1), tol=1e-9)
                .values(component="displacement_x"))  # P2-I: .get→.values
    eq2 = r.nodes.get(ids=[1, 2, 3, 4], component="displacement_x")
    assert _sids(spatial.node_ids) == _sids(eq2.node_ids) == [1, 2, 3, 4]


def test_element_terminal_get_is_elementslab_and_parity(tmp_path):
    r, _fem = _make_results_with_fem(tmp_path)
    chained = r.elements.select(pg="Beams")
    slab = chained.values(component="globalForce")  # P2-I: .get→.values
    assert isinstance(slab, ElementSlab)
    assert type(slab).__name__ == "ElementSlab"

    equiv = r.elements.get(ids=[10], component="globalForce")
    assert _sids(slab.element_ids) == _sids(equiv.element_ids) == [10]
    np.testing.assert_array_equal(slab.values, equiv.values)

    # centroid spatial filter then terminal read
    spatial = (r.elements.select()
                .in_box((-1, -1, -1), (2, 2, 2))
                .values(component="globalForce"))  # P2-I: .get→.values
    np.testing.assert_array_equal(spatial.values, equiv.values)


def test_terminal_get_passes_time_and_stage(tmp_path):
    r, _fem = _make_results_with_fem(tmp_path)
    # time= forwarded to the existing reader path (single step → T=1).
    slab = r.nodes.select(ids=[1]).values(  # P2-I: .get→.values
        component="displacement_x", time=0,
    )
    equiv = r.nodes.get(ids=[1], component="displacement_x", time=0)
    assert slab.values.shape == equiv.values.shape == (1, 1)
    np.testing.assert_array_equal(slab.values, equiv.values)
    # explicit stage= also forwarded (the only stage here is "static").
    slab2 = r.nodes.select(ids=[1]).values(  # P2-I: .get→.values
        component="displacement_x", stage="static",
    )
    np.testing.assert_array_equal(
        slab2.values,
        r.nodes.get(ids=[1], component="displacement_x",
                    stage="static").values,
    )


def test_bare_result_without_component_fails_loud(tmp_path):
    r, _fem = _make_results_with_fem(tmp_path)
    with pytest.raises(RuntimeError, match="needs .get.component"):
        r.nodes.select(ids=[1]).result()
    with pytest.raises(RuntimeError, match="needs .get.component"):
        r.elements.select(ids=[10]).result()


# =====================================================================
# Fail-loud: element centroid with an unknown node id
# =====================================================================

def test_element_centroid_fails_loud_on_unknown_node(tmp_path):
    """A connectivity entry referencing an absent node must raise — never
    the np.clip silent row-0 substitution _element_centroids does.
    """
    r, fem = _make_results_with_fem(tmp_path)

    # Corrupt the mock fem so element 10's connectivity references a
    # node id that is not in the FEM node set.  P3-R / M-STOP-3: the
    # centroid path now iterates ``fem.elements._groups.values()``
    # (not ``resolve``), so corrupt the group's connectivity.
    fem.elements._groups[0].connectivity = np.array(
        [[1, 2, 3, 10 ** 9]], dtype=np.int64
    )
    with pytest.raises(KeyError, match="not in the FEM node set"):
        r.elements.select().in_box((-9, -9, -9), (9, 9, 9))


def test_chain_without_bound_fem_fails_loud(tmp_path):
    """Spatial / coordinate access needs a bound FEMData (duck-typed,
    same as the existing results spatial helpers)."""
    path = tmp_path / "nofem.h5"
    time = np.array([0.0, 1.0])
    with NativeWriter(path) as w:
        w.open(source_type="domain_capture")
        sid = w.begin_stage(name="s", kind="static", time=time)
        w.write_nodes(
            sid, "partition_0",
            node_ids=np.array([1], dtype=np.int64),
            components={"displacement_x": np.array([[0.0], [1.0]])},
        )
        w.end_stage()

    with Results.from_native(path, model=_open_model_from_h5(path)) as r:
        if r._fem is None:
            # select() with a named selector needs the fem to resolve;
            # explicit ids= seeds atoms but the spatial verb still needs
            # coordinates → fail loud there.
            with pytest.raises(RuntimeError, match="bound FEMData"):
                r.nodes.select(ids=[1]).in_box((-1, -1, -1), (1, 1, 1))


# =====================================================================
# Existing results behaviour is unchanged (additive only)
# =====================================================================

def test_existing_get_inbox_nearest_byte_behaviour_unchanged(tmp_path):
    r, _fem = _make_results_with_fem(tmp_path)

    # .get() unchanged
    slab = r.nodes.get(pg="TopRow", component="displacement_x")
    assert isinstance(slab, NodeSlab)
    assert _sids(slab.node_ids) == [3, 4]

    # .in_box() still half-open, additive with pg=, unchanged
    box = r.nodes.in_box(
        box_min=(-0.5, -0.5, -0.5), box_max=(1.5, 1.5, 1.5),
        component="displacement_x",
    )
    assert _sids(box.node_ids) == [1, 2, 3, 4]
    box_pg = r.nodes.in_box(
        box_min=(-0.5, -0.5, -0.5), box_max=(1.5, 1.5, 1.5),
        component="displacement_x", pg="TopRow",
    )
    assert _sids(box_pg.node_ids) == [3, 4]

    # .nearest_to() unchanged
    near = r.nodes.nearest_to((0.1, 0.1, 0.0), component="displacement_x")
    assert _sids(near.node_ids) == [1]

    # element .get() unchanged
    egf = r.elements.get(ids=[10], component="globalForce")
    assert isinstance(egf, ElementSlab)
    assert _sids(egf.element_ids) == [10]
