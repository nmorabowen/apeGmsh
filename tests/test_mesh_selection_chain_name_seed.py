"""S3d follow-on — ``g.mesh_selection.select(name=)`` name-seeding.

Closes ``docs/plans/selection-unification.md`` §9 item 3 for the one
name kind that has a clean, non-registering, non-reimplementing
resolution surface **on** ``MeshSelectionSet``: an **existing**
``g.mesh_selection`` set name.

The headline invariant this file locks:

* ``select(name=N)`` is *id-for-id* the existing
  ``g.mesh_selection`` set ``N`` — because it **delegates verbatim**
  to the surfaces already on the class (``get_tag`` over ``_sets`` +
  ``get_nodes`` / ``get_elements``); it writes **no** new resolver.
* ``select(name=N).<spatial>`` equals the eager ``filter_set`` over
  that same set — so a name-seeded chain narrowed by ``in_box`` /
  ``on_plane`` is the same node/element set the eager path produces,
  for a set built by ``add_nodes`` **and** for one built by
  ``from_physical`` (the equivalence the task names explicitly).
* ``name=`` only **reads** ``_sets`` — no registration, no tag
  allocation (the locked
  ``test_mesh_selection_chain.test_select_does_not_register_a_set``
  invariant, extended to the name path).
* Fail-loud: an unknown name, or a node-set name asked at the element
  level, raises ``KeyError`` (never a silent empty / full-universe
  seed — resolution-contract Rule 6); ``ids=`` + ``name=`` together
  raises ``ValueError``.

Scope note (the consciously-reported boundary): seeding *directly*
from a raw gmsh physical-group name or an apeGmsh label is **not** a
``select()`` parameter — ``MeshSelectionSet`` has no non-registering,
non-reimplementing resolver for those (``from_physical`` *registers* a
set + allocates a tag and is node-only; label/geometry resolution
lives off the class, and a mesh-selection name is deliberately not a
geometry-resolver tier).  The supported existing-surface route is the
two-step ``from_physical(...)`` / ``from_geometric(...)`` **then**
``select(name=...)``, exercised below via the ``from_physical`` set.

No ``openseespy`` dependency (curated no-openseespy CI gate): pure
apeGmsh + gmsh + numpy.  Same deterministic 3x3x3 structured cube the
S3d focused test uses (27 nodes at {0,0.5,1}^3, 8 hex8 cells) plus a
6-face ``Shell`` surface PG so ``from_physical`` yields a non-trivial
26-node subset (all but the single interior node).
"""
from __future__ import annotations

import pytest

from apeGmsh import apeGmsh
from apeGmsh.mesh._mesh_selection_chain import MeshSelectionChain


@pytest.fixture
def live():
    """Live session: 3x3x3 lattice + volume PG ``Body`` + 6-face
    surface PG ``Shell`` (pattern mirrored from
    ``tests/test_mesh_selection_chain.py::live``)."""
    g = apeGmsh(model_name="s3d_nameseed_cube", verbose=False)
    g.begin()
    try:
        g.model.geometry.add_box(0.0, 0.0, 0.0, 1.0, 1.0, 1.0,
                                 label="box")
        g.physical.add_volume("box", name="Body")
        g.model.sync()
        faces = g.model.queries.boundary("box", dim=3, oriented=False)
        g.physical.add_surface([int(t) for _d, t in faces], name="Shell")
        g.mesh.structured.set_transfinite_box("box", n=3)
        g.mesh.generation.generate(dim=3)
        yield g
    finally:
        g.end()


def _sorted_ids(seq) -> list[int]:
    return sorted(int(x) for x in seq)


def _set_node_ids(ms, tag) -> list[int]:
    return _sorted_ids(ms.get_nodes(0, tag)["tags"])


def _set_elem_ids(ms, dim, tag) -> list[int]:
    return _sorted_ids(ms.get_elements(dim, tag)["element_ids"])


# =====================================================================
# Name-seed == the existing set (pure parity, both seed kinds)
# =====================================================================

def test_name_seed_node_equals_eager_add_nodes_set(live):
    """``select(name=N)`` is id-for-id the ``add_nodes(name=N)`` set,
    and the terminal is the same ``get_nodes`` shape."""
    ms = live.mesh_selection
    tag = ms.add_nodes(in_box=(0, 0, 0, 1, 1, 1), name="boxn")
    eager = _set_node_ids(ms, tag)
    assert len(eager) == 8                       # half-open default shell

    seeded = ms.select(name="boxn")
    assert isinstance(seeded, MeshSelectionChain)
    assert seeded.FAMILY == "point"
    assert _sorted_ids(seeded.ids) == eager
    res = seeded.result()
    assert _sorted_ids(res["tags"]) == eager
    assert res["tags"].dtype == object
    assert res["coords"].shape[1] == 3


def test_name_seed_node_equals_from_physical_set(live):
    """The equivalence the task names explicitly: a set built by
    ``from_physical`` re-seeded by ``select(name=)`` is id-for-id that
    ``from_physical`` set (the existing-surface PG route)."""
    ms = live.mesh_selection
    pg_set = ms.from_physical(2, "Shell", ms_name="shell_nodes")
    eager = _set_node_ids(ms, pg_set)
    assert len(eager) == 26                      # all 27 but interior node

    seeded = ms.select(name="shell_nodes")
    assert _sorted_ids(seeded.ids) == eager
    assert _sorted_ids(seeded.result()["tags"]) == eager


def test_name_seed_element_equals_eager_add_elements_set(live):
    """Element level: ``select(level='element', dim=3, name=N)`` is
    id-for-id ``add_elements(dim=3, name=N)``; terminal is the
    ``get_elements`` shape."""
    ms = live.mesh_selection
    # known S3d fixture fact: half-open box upper-z 0.75 keeps the 4
    # cells whose centroid z == 0.25.
    tag = ms.add_elements(dim=3, in_box=(-1.0, -1.0, -1.0, 2.0, 2.0, 0.75),
                          name="boxe")
    eager = _set_elem_ids(ms, 3, tag)
    assert len(eager) == 4

    seeded = ms.select(level="element", dim=3, name="boxe")
    assert _sorted_ids(seeded.ids) == eager
    res = seeded.result()
    assert _sorted_ids(res["element_ids"]) == eager
    assert res["element_ids"].dtype == object
    assert res["connectivity"].dtype == object


# =====================================================================
# Name-seed + spatial == eager filter_set over the same set
# =====================================================================

def test_name_seed_then_spatial_equals_filter_set_add_nodes(live):
    """``select(name=N).in_box(b)`` == ``filter_set(0, tag_N, in_box=b)``
    (half-open both sides) — name-seed + fluent spatial is the eager
    seed + spatial, id-for-id."""
    ms = live.mesh_selection
    tag = ms.add_nodes(in_box=(0, 0, 0, 1, 1, 1), name="boxn")  # 8 nodes

    sub = (-1.0, -1.0, -1.0, 0.5, 0.5, 0.5)
    ft = ms.filter_set(0, tag, in_box=sub, name="boxn_sub")
    chained = (ms.select(name="boxn")
                 .in_box((sub[0], sub[1], sub[2]), (sub[3], sub[4], sub[5])))
    assert _sorted_ids(chained.ids) == _set_node_ids(ms, ft)
    assert len(chained) == 1                      # only the (0,0,0) corner


def test_name_seed_then_spatial_equals_filter_set_from_physical(live):
    """Same equivalence with the ``from_physical`` set:
    ``select(name=shell).on_plane(z=0)`` ==
    ``filter_set(0, shell_tag, on_plane=("z",0))``."""
    ms = live.mesh_selection
    pg_set = ms.from_physical(2, "Shell", ms_name="shell_nodes")

    ft = ms.filter_set(0, pg_set, on_plane=("z", 0.0, 1e-9),
                       name="shell_z0")
    chained = (ms.select(name="shell_nodes")
                 .on_plane((0, 0, 0), (0, 0, 1), tol=1e-9))
    assert _sorted_ids(chained.ids) == _set_node_ids(ms, ft)
    assert len(chained) == 9                       # z=0 lattice face


# =====================================================================
# Daisy-chain + set algebra from name seeds (same engine adapter)
# =====================================================================

def test_name_seed_daisychains_and_set_algebra(live):
    ms = live.mesh_selection
    ta = ms.add(0, [1, 2, 3, 4, 5], name="A")
    ms.add(0, [4, 5, 6, 7], name="B")

    a = ms.select(name="A")
    b = ms.select(name="B")
    # node level on the same MeshSelectionSet -> same memoised engine
    # adapter -> set algebra composes (insertion-order dedup law).
    assert _sorted_ids((a | b).ids) == [1, 2, 3, 4, 5, 6, 7]
    assert _sorted_ids((a & b).ids) == [4, 5]
    assert _sorted_ids((a - b).ids) == [1, 2, 3]
    for s in (a | b, a & b, a - b):
        assert isinstance(s, MeshSelectionChain)

    chained = ms.select(name="A").in_box((-9, -9, -9), (9, 9, 9),
                                          inclusive=True)
    assert _sorted_ids(chained.ids) == _set_node_ids(ms, ta)


# =====================================================================
# Fail-loud — never a silent empty / full-universe seed
# =====================================================================

def test_unknown_name_fails_loud_with_route_hint(live):
    ms = live.mesh_selection
    ms.add_nodes(in_box=(0, 0, 0, 1, 1, 1), name="exists")
    with pytest.raises(KeyError) as ei:
        ms.select(name="nope")
    msg = ei.value.args[0]
    assert "No mesh-selection set named" in msg
    assert "'nope'" in msg
    # the consciously-reported register-then-select route is surfaced
    assert "from_physical" in msg and "from_geometric" in msg
    assert "exists" in msg                          # available list


def test_ids_and_name_are_mutually_exclusive(live):
    ms = live.mesh_selection
    ms.add_nodes(in_box=(0, 0, 0, 1, 1, 1), name="exists")
    with pytest.raises(ValueError, match="mutually exclusive"):
        ms.select(ids=[1], name="exists")


def test_node_set_name_not_found_at_element_level(live):
    """A node set name asked at the element level is a loud miss (the
    (dim, tag)+name identity contract): no cross-dim silent match."""
    ms = live.mesh_selection
    ms.add_nodes(in_box=(0, 0, 0, 1, 1, 1), name="nset")   # dim=0 set
    with pytest.raises(KeyError, match="No mesh-selection set named"):
        ms.select(level="element", dim=3, name="nset")     # dim=3 lookup


# =====================================================================
# Additive — name-seed never registers / allocates (locked invariant)
# =====================================================================

def test_name_seed_does_not_register_or_allocate(live):
    """Persistence stays out of scope: a full
    ``select(name=...).<chain>.result()`` must not mutate ``_sets`` or
    ``_next_tag`` (the locked no-registration invariant, extended to
    the name path)."""
    ms = live.mesh_selection
    ms.add_nodes(in_box=(0, 0, 0, 1, 1, 1), name="boxn")
    ms.from_physical(2, "Shell", ms_name="shell_nodes")

    sets_before = dict(ms._sets)
    next_tag_before = dict(ms._next_tag)

    (ms.select(name="boxn")
       .in_box((0, 0, 0), (1, 1, 1))
       .on_plane((0, 0, 0), (0, 0, 1), tol=1e-9)
       .result())
    (ms.select(name="shell_nodes")
       .in_sphere((0, 0, 0), 5.0)
       .result())

    assert ms._sets == sets_before                  # nothing registered
    assert ms._next_tag == next_tag_before          # no tag allocated
