"""S3e — the cross-chain *idiom* lock (selection-unification).

Sub-phase S3e of the selection-unification work
(``docs/plans/selection-unification.md`` §5/§6).  S3a–S3d landed the
``SelectionChain`` mixin plus the five concrete chains
(``GeometryChain`` entity-family; ``NodeChain`` / ``ElementChain`` /
``ResultChain`` / ``MeshSelectionChain`` point-family) each with its own
focused smoke test.  This file is the *only* file that looks at all
five chains **together** and locks the cross-cutting idiom:

* exactly those five concrete chains exist (a future 6th that skips the
  ``__init_subclass__`` contract is caught here);
* the public verb *names* and their ``inspect.signature`` are identical
  across all five (arities/keyword-only-ness cannot silently drift);
* ``__init_subclass__`` still rejects a bad FAMILY / a dropped verb / a
  missing hook at *class-definition* time;
* per-family behavioural laws hold (point family: half-open vs
  ``inclusive=`` closed, insertion-order set-algebra, deterministic
  ``nearest_to`` tie-break, loud cross-type/cross-engine; entity family:
  ``inclusive=``→``TypeError``, gmsh-BRep set-algebra laws).

CRITICAL SCOPE NOTE (FP-2 / T15, ratified §3, §5):
This file asserts a shared verb-NAME / signature surface **and**
per-family laws.  It NEVER asserts cross-family *behavioural* equality.
``in_box`` is honestly three irreconcilable semantics — entity-family
``gmsh.getEntitiesInBoundingBox`` (BRep CONTAINMENT, closed,
``Geometry.Tolerance``≈1e-8 expanded, *no* half-open knob) vs
point-family node-coordinate / element-centroid half-open ``[lo, hi)``.
There is deliberately no ``GeometryChain`` vs ``NodeChain`` result
comparison anywhere below; the two families are checked in separate,
family-scoped test bodies.

No ``openseespy`` dependency (curated no-openseespy CI gate): pure
apeGmsh + gmsh + numpy + a tiny synthetic native HDF5 results file.
Fixture *patterns* are mirrored (not imported) from
``tests/test_geometry_chain.py``, ``tests/test_fem_chain.py``,
``tests/test_result_chain.py`` and ``tests/test_mesh_selection_chain.py``.
"""
from __future__ import annotations

import inspect
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from apeGmsh import apeGmsh
from apeGmsh._chain import REQUIRED_VERBS, SelectionChain

# Importing the five host modules registers every concrete subclass.
from apeGmsh.core._selection import GeometryChain
from apeGmsh.mesh._node_chain import NodeChain
from apeGmsh.mesh._elem_chain import ElementChain
from apeGmsh.results._result_chain import ResultChain
from apeGmsh.mesh._mesh_selection_chain import MeshSelectionChain
from apeGmsh.results import Results
from apeGmsh.results.writers import NativeWriter


# The set-algebra surface that rides alongside REQUIRED_VERBS.
_SET_ALGEBRA = (
    "__or__", "__and__", "__sub__", "__xor__",
    "union", "intersect", "difference",
)
# in_box is the one verb GeometryChain overrides with a *different*
# signature (entity family has no `inclusive=` knob — R3).  Its name is
# still required everywhere; only its signature is family-specific, so
# the "identical signature" assertion excludes it and the four
# point-family chains are checked for in_box separately.
_EXPECTED_CHAINS = {
    GeometryChain, NodeChain, ElementChain, ResultChain, MeshSelectionChain,
}
_POINT_CHAINS = (NodeChain, ElementChain, ResultChain, MeshSelectionChain)


def _all_concrete_subclasses(cls) -> set:
    """Every concrete (FAMILY-set) leaf reachable via __subclasses__."""
    out: set = set()
    for sub in cls.__subclasses__():
        if getattr(sub, "FAMILY", ""):
            out.add(sub)
        out |= _all_concrete_subclasses(sub)
    return out


# =====================================================================
# 1. Exactly these five concrete chains
# =====================================================================

def test_exactly_these_concrete_chains():
    """The family is closed at five.

    A future 6th ``XChain(SelectionChain)`` would surface here as an
    extra member of ``__subclasses__`` and break the ``==`` — forcing
    whoever adds it to also wire it into this cross-chain contract
    (and, by ``__init_subclass__``, give it a valid FAMILY + the full
    verb surface + every hook).  Equality (not ``>=``) is deliberate:
    the assertion is "these and *only* these".
    """
    found = _all_concrete_subclasses(SelectionChain)
    assert found == _EXPECTED_CHAINS, (
        "concrete SelectionChain subclasses drifted from the ratified "
        f"five. extra={sorted(c.__name__ for c in found - _EXPECTED_CHAINS)} "
        f"missing={sorted(c.__name__ for c in _EXPECTED_CHAINS - found)}"
    )


# =====================================================================
# 2. Identical public verb surface across all five
# =====================================================================

def test_identical_public_verb_surface():
    """Every required/set-algebra name is callable on all five, and its
    signature is byte-identical across all five — names *and* arities
    cannot drift.  ``in_box`` is the lone family-specific signature
    (R3): asserted callable everywhere, signature-identical across the
    four point chains, and (entity family) separately covered by
    ``test_entity_family_laws``.
    """
    chains = sorted(_EXPECTED_CHAINS, key=lambda c: c.__name__)

    for name in REQUIRED_VERBS + _SET_ALGEBRA:
        for cls in chains:
            assert callable(getattr(cls, name, None)), (
                f"{cls.__name__} is missing callable {name!r}"
            )
        if name == "in_box":
            # Family-specific signature by ratified design (R3).
            sigs = {inspect.signature(getattr(c, "in_box"))
                    for c in _POINT_CHAINS}
            assert len(sigs) == 1, (
                f"in_box signature drifted across point chains: {sigs}"
            )
            continue
        sigs = {cls: inspect.signature(getattr(cls, name))
                for cls in chains}
        distinct = set(sigs.values())
        assert len(distinct) == 1, (
            f"signature of {name!r} is not identical across the five "
            f"chains: "
            + ", ".join(f"{c.__name__}{s}" for c, s in sigs.items())
        )


# =====================================================================
# 3. __init_subclass__ still rejects bad shapes (definition-time)
# =====================================================================

def test_init_subclass_rejects_bad_family():
    """A concrete subclass with an unknown FAMILY fails at the `class`
    statement (stronger than a CI test — it is an ImportError-class
    failure)."""
    with pytest.raises(TypeError, match="FAMILY.*invalid"):
        class _BadFamily(SelectionChain):
            FAMILY = "bogus"


def test_init_subclass_rejects_dropped_verb():
    """FAMILY is valid but a required verb is shadowed to a non-callable
    — the gate refuses the class."""
    with pytest.raises(TypeError, match="missing required selection verb"):
        class _DroppedVerb(SelectionChain):
            FAMILY = "point"
            nearest_to = None  # required verb shadowed to non-callable

            # full hook set so we fail on the verb, not a hook
            def _coords_of(self, atoms):
                ...

            def _spatial_box(self, atoms, lo, hi, *, inclusive):
                ...

            def _spatial_sphere(self, atoms, center, radius):
                ...

            def _spatial_plane(self, atoms, point, normal, tol):
                ...

            def _materialize(self):
                ...


def test_init_subclass_rejects_missing_hook():
    """FAMILY valid + every verb present, but a required hook is left as
    the base ``NotImplementedError`` stub — refused at definition."""
    with pytest.raises(TypeError, match="must implement.*hook"):
        class _MissingHook(SelectionChain):
            FAMILY = "point"
            # every required verb is inherited & callable; _materialize
            # (and the others) are NOT overridden -> still the base stub.


# =====================================================================
# Fixtures — smallest deterministic seed per domain (patterns mirrored
# from each domain's existing focused test, NOT imported)
# =====================================================================

@pytest.fixture(scope="module")
def cube_fem():
    """3x3x3 lattice -> 27 nodes at {0,0.5,1}^3, 8 hex8 cells.

    Pattern from ``tests/test_fem_chain.py::cube_fem``: a structured
    transfinite box so every boundary count is an exact integer.  Used
    by the NodeChain / ElementChain point-family rows.
    """
    g = apeGmsh(model_name="s3e_cube", verbose=False)
    g.begin()
    try:
        g.model.geometry.add_box(0.0, 0.0, 0.0, 1.0, 1.0, 1.0,
                                 label="box")
        g.physical.add_volume("box", name="Body")
        g.mesh.structured.set_transfinite_box("box", n=3)
        g.mesh.generation.generate(dim=3)
        fem = g.mesh.queries.get_fem_data(dim=3)
    finally:
        g.end()
    return fem


@pytest.fixture
def live():
    """Live session, 3x3x3 lattice — pattern from
    ``tests/test_mesh_selection_chain.py::live``.  Stays open so
    ``g.mesh_selection`` reads the live ``gmsh.model.mesh``."""
    g = apeGmsh(model_name="s3e_live", verbose=False)
    g.begin()
    try:
        g.model.geometry.add_box(0.0, 0.0, 0.0, 1.0, 1.0, 1.0,
                                 label="box")
        g.physical.add_volume("box", name="Body")
        g.mesh.structured.set_transfinite_box("box", n=3)
        g.mesh.generation.generate(dim=3)
        yield g
    finally:
        g.end()


def _make_results_with_fem(tmp_path: Path):
    """4 unit-square corner nodes (z=0) + far node 5; one quad elem 10.

    Pattern from ``tests/test_result_chain.py::_make_results_with_fem``
    (synthetic native HDF5 + SimpleNamespace mock FEM, no openseespy).
    """
    path = tmp_path / "synthetic.h5"
    time = np.array([0.0, 1.0])
    node_ids = np.array([1, 2, 3, 4, 5], dtype=np.int64)
    ux = np.array([[0.1, 0.2, 0.3, 0.4, 0.5],
                   [1.1, 1.2, 1.3, 1.4, 1.5]])
    elem_idx = np.array([10], dtype=np.int64)
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

    nodes_ns = SimpleNamespace(
        ids=node_ids,
        coords=coords,
        physical=SimpleNamespace(node_ids=lambda n: {
            "TopRow": np.array([3, 4], dtype=np.int64),
        }[n]),
        labels=SimpleNamespace(
            node_ids=lambda n: np.array([], dtype=np.int64),
        ),
    )
    elements_ns = SimpleNamespace(
        ids=np.array([10], dtype=np.int64),
        types=[type_info],
        resolve=_resolve,
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
    return Results.from_native(path, fem=fem)


def _ids(seq) -> list[int]:
    return sorted(int(x) for x in seq)


def _seed_point_chain(kind, *, cube_fem=None, live=None, tmp_path=None):
    """Build the smallest representative chain for one point domain.

    Returns ``(chain_over_all, chain_a, chain_b, on_hi_box, on_hi_pt)``
    where ``on_hi_box``/``on_hi_pt`` exercise the half-open vs
    ``inclusive=`` boundary.
    """
    if kind == "node":
        allc = cube_fem.nodes.select(pg="Body")          # 27 nodes
        a = cube_fem.nodes.select(ids=[1, 2, 3])
        b = cube_fem.nodes.select(ids=[2, 3, 4])
        # box [0,0,0]-[1,1,1]: half-open drops the x|y|z==1 face nodes,
        # inclusive=True keeps all 27.
        return allc, a, b, ((0, 0, 0), (1, 1, 1)), None
    if kind == "element":
        allc = cube_fem.elements.select(pg="Body")        # 8 hex
        eids = sorted(int(x) for x in allc)
        a = cube_fem.elements.select(ids=eids[:3])
        b = cube_fem.elements.select(ids=eids[1:4])
        # centroids at {0.25,0.75}^3; box ..-(0.75,0.75,0.75):
        # half-open keeps the single (0.25,0.25,0.25) cell, inclusive
        # keeps all 8 (0.75 on the closed upper face).
        return allc, a, b, ((0.0, 0.0, 0.0), (0.75, 0.75, 0.75)), None
    if kind == "result":
        r = _make_results_with_fem(tmp_path)
        allc = r.nodes.select()                            # ids 1..5
        a = r.nodes.select(ids=[1, 2, 3])
        b = r.nodes.select(ids=[2, 3, 4, 5])
        # 4 corner nodes in z=0 unit square + far node 5.  box
        # [0,0,0]-[1,1,0]: half-open drops nodes whose x|y == 1
        # (2,3,4); inclusive=True keeps all four corners (1,2,3,4).
        return allc, a, b, ((0.0, 0.0, 0.0), (1.0, 1.0, 0.0)), None
    if kind == "mesh_selection":
        ms = live.mesh_selection
        allc = ms.select()                                 # 27 nodes
        a = ms.select(ids=[1, 2, 3])
        b = ms.select(ids=[2, 3, 4])
        return allc, a, b, ((0, 0, 0), (1, 1, 1)), None
    raise AssertionError(kind)


# =====================================================================
# 4. Point-family behavioural laws (parametrised over the 4 point chains)
# =====================================================================

@pytest.mark.parametrize(
    "kind", ["node", "element", "result", "mesh_selection"]
)
def test_point_family_laws(kind, cube_fem, live, tmp_path):
    allc, a, b, box, _ = _seed_point_chain(
        kind, cube_fem=cube_fem, live=live, tmp_path=tmp_path
    )

    # ── concrete-class shape ────────────────────────────────
    assert allc.FAMILY == "point"
    assert isinstance(allc, SelectionChain)
    cls = type(allc)

    # ── in_box: half-open default vs inclusive=True closed ──
    lo, hi = box
    half = allc.in_box(lo, hi)
    closed = allc.in_box(lo, hi, inclusive=True)
    h_ids, c_ids = _ids(half), _ids(closed)
    # the on-`hi` boundary atoms are excluded half-open, included closed
    assert set(h_ids) < set(c_ids), (
        f"{kind}: inclusive=True must be a strict superset of half-open "
        f"(some atom sits on the upper face). half={h_ids} closed={c_ids}"
    )

    # ── set-algebra laws on a non-trivial pair ──────────────
    assert _ids(a | a) == _ids(a)                       # idempotent ∪
    assert _ids(a & a) == _ids(a)                       # idempotent ∩
    assert list(a - a) == []                            # self-difference
    assert list(a ^ a) == []                            # self-sym-diff
    # commutative as id-sets (insertion order may differ, sets equal)
    assert _ids(a | b) == _ids(b | a)
    assert set(a & b) == set(b & a)
    # named aliases == operators
    assert _ids(a.union(b)) == _ids(a | b)
    assert _ids(a.intersect(b)) == _ids(a & b)
    assert _ids(a.difference(b)) == _ids(a - b)

    # ── insertion-order preservation (the one dedup law) ────
    a_first = list(a)
    # a|b keeps a's order first, then b's new atoms, deduped
    union_seq = list(a | b)
    assert union_seq[:len(a_first)] == a_first, (
        f"{kind}: union must preserve self's insertion order first"
    )
    assert len(union_seq) == len(set(union_seq))         # deduped
    # difference preserves self order
    assert list(a - b) == [x for x in a_first if x not in set(b)]

    # ── nearest_to deterministic + stable lowest-index tie ──
    # The contract (`_chain._nearest`): sort key is
    # ``(squared_distance, i)`` where ``i`` is the atom's *position in
    # the current ordered atom tuple*.  So nearest_to is fully
    # deterministic for a given ordering, and ties break by *lowest
    # position* (a stable sort).  It is NOT invariant under input
    # reordering when exact distance ties exist — that is inherent to a
    # positional tie-break and itself deterministic; this file asserts
    # the guarantees the code actually makes, not a stronger one.
    near_point = (0.0, 0.0, 0.0)
    r1 = list(allc.nearest_to(near_point, count=2))
    r2 = list(allc.nearest_to(near_point, count=2))
    r3 = list(allc.nearest_to(near_point, count=2))
    assert r1 == r2 == r3, (
        f"{kind}: nearest_to not deterministic across repeated calls "
        f"on the same ordering ({r1} vs {r2} vs {r3})"
    )
    assert len(r1) == 2
    # stable lowest-position tie-break: full ordering by the same key
    # the verb uses is identical when recomputed, and equal-distance
    # atoms keep their input positional order (sorted() is stable, ties
    # only broken by the explicit `i`).
    import math as _math
    atoms_seq = tuple(allc)
    coords = allc._coords_of(atoms_seq)
    by_key = sorted(
        range(len(atoms_seq)),
        key=lambda i: (
            _math.fsum((coords[i][k] - near_point[k]) ** 2
                       for k in range(3)),
            i,
        ),
    )
    full = list(allc.nearest_to(near_point, count=len(atoms_seq)))
    assert full == [atoms_seq[i] for i in by_key], (
        f"{kind}: nearest_to order is not the stable "
        f"(distance, lowest-position) sort the contract specifies"
    )
    assert r1 == full[:2]
    # positions strictly increase among any equal-distance run
    dists = [_math.fsum((coords[i][k] - near_point[k]) ** 2
                        for k in range(3)) for i in by_key]
    for j in range(1, len(by_key)):
        if dists[j] == dists[j - 1]:
            assert by_key[j] > by_key[j - 1], (
                f"{kind}: equal-distance tie not broken by lowest "
                f"position (unstable sort)"
            )

    # ── chaining returns the SAME concrete class (covariant) ─
    assert type(allc.in_box(lo, hi)) is cls
    assert type(allc.in_box(lo, hi).where(lambda xyz: True)) is cls
    assert type(a | b) is cls
    assert type(a - b) is cls
    assert type(a ^ b) is cls

    # ── cross-type set-algebra is loud (point vs entity) ────
    geom_other = GeometryChain((), _engine=None)
    with pytest.raises(TypeError):
        a | geom_other
    with pytest.raises(TypeError):
        a & geom_other
    with pytest.raises(TypeError):
        a - geom_other

    # ── cross-engine set-algebra is loud (same class, other engine) ─
    foreign = type(a)(tuple(a), _engine=object())
    with pytest.raises(TypeError):
        a | foreign


# =====================================================================
# 5. Entity-family behavioural laws (GeometryChain only)
#
# SCOPE: this body asserts entity-family laws in isolation.  There is
# deliberately NO comparison against any point chain — cross-family
# behavioural equality is never asserted in this file (FP-2 / T15).
# =====================================================================

@pytest.fixture
def cube_geo():
    """Unit box ``box`` (dim-3) + 6-face PG ``Faces``.

    Pattern from ``tests/test_geometry_chain.py::cube``: faces resolved
    by PG name through the chain (never hard-coded raw tags).
    """
    g = apeGmsh(model_name="s3e_geo", verbose=False)
    g.begin()
    try:
        g.model.geometry.add_box(0.0, 0.0, 0.0, 1.0, 1.0, 1.0,
                                 label="box")
        g.model.sync()
        faces = g.model.queries.boundary("box", dim=3, oriented=False)
        g.physical.add_surface([int(t) for _d, t in faces], name="Faces")
        yield g
    finally:
        g.end()


def test_entity_family_laws(cube_geo):
    g = cube_geo
    faces = g.model.select("Faces")           # 6 dim-2 entities
    assert faces.FAMILY == "entity"
    assert isinstance(faces, SelectionChain)
    cls = type(faces)

    # ── in_box(inclusive=) — any keyword raises TypeError (R3) ──
    big = ((-1.0, -1.0, -1.0), (2.0, 2.0, 2.0))
    with pytest.raises(TypeError, match="inclusive"):
        faces.in_box(*big, inclusive=True)
    with pytest.raises(TypeError, match="inclusive"):
        faces.in_box(*big, inclusive=False)
    # any *other* keyword is equally rejected (entity family has no
    # half-open knob at all — not silently ignored)
    with pytest.raises(TypeError):
        faces.in_box(*big, bogus=1)

    # an enclosing box (BRep CONTAINMENT, ~1e-8 expanded) keeps all 6;
    # a half-space box keeps a proper subset.
    allf = faces.in_box((-0.1, -0.1, -0.1), (1.1, 1.1, 1.1))
    sub = faces.in_box((-1.0, -1.0, -1.0), (0.5, 2.0, 2.0))
    assert len(allf) == 6
    assert 0 < len(sub) < 6

    # ── set-algebra laws on (dim,tag) atoms ─────────────────
    a = allf                                  # all 6 faces
    b = sub                                   # a proper subset
    assert sorted(a | a) == sorted(a)         # idempotent ∪
    assert sorted(a & a) == sorted(a)         # idempotent ∩
    assert list(a - a) == []                  # self-difference empty
    assert sorted(a & b) == sorted(b)         # b ⊂ a  ->  a∩b == b
    assert sorted(a - b) == sorted(
        x for x in list(a) if x not in set(b)
    )
    # insertion-order preserved through union (a's order, then b-new)
    a_seq = list(a)
    u = list(a | b)
    assert u[:len(a_seq)] == a_seq
    assert len(u) == len(set(u))              # deduped

    # ── chaining is covariant (same concrete class) ─────────
    assert type(faces.in_box((-1, -1, -1), (2, 2, 2))) is cls
    assert type(a | b) is cls
    assert type(a - b) is cls
    assert type(faces.where(lambda xyz: True)) is cls

    # ── cross-family combination is loud (entity vs point) ──
    node_other = NodeChain((), _engine=None)
    with pytest.raises(TypeError):
        faces | node_other
    with pytest.raises(TypeError):
        faces - node_other

    # ── terminal stays the legacy Selection (untouched) ─────
    from apeGmsh.core._selection import Selection
    assert isinstance(faces.result(), Selection)
