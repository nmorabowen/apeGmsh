"""Tests for the explicit-only per-node ``ndf`` channel (S1b).

Covers:

- Fail-loud: a node with no declaration raises ``LookupError`` from
  :meth:`NodeComposite.ndf_for` with a message that names both fixes.
- Single targeted ``set`` + ``set_default`` interaction.
- Targeted-only (no default): nodes outside the target raise.
- H5 round-trip (2.7.0) preserves the per-node ``ndf``.
- H5 forward-compat (2.6.0 → 2.7.0): a synthetic 2.6.0 file loads
  with ``_ndf is None`` and every ``ndf_for`` call raises.
- H5 length-validation: a writer that drops the last ndf entry
  raises ``MalformedH5Error`` on read.
- Hash regression: identical geometry + different declarations
  produces different ``snapshot_id``.
- ``set`` after ``get_fem_data()`` emits ``UserWarning``.
- ``from_msh`` yields a broker with no declared ndf (sentinel) —
  every ``ndf_for`` call raises with the helpful message.
- Resolver miss propagates ``KeyError`` out of ``get_fem_data()``.
"""
from __future__ import annotations

import warnings
from pathlib import Path

import h5py
import numpy as np
import pytest

from apeGmsh.mesh.FEMData import FEMData
from apeGmsh.mesh._femdata_h5_io import NEUTRAL_SCHEMA_VERSION


# =====================================================================
# Helpers
# =====================================================================

def _build_two_box_model(g):
    """Build a 1-box-with-two-PGs model used by several tests."""
    g.model.geometry.add_box(0.0, 0.0, 0.0, 10.0, 10.0, 10.0, label='Body')
    g.model.sync()
    # Tag the top face and the bottom face as named PGs.
    top_tag = None
    bot_tag = None
    for d, t in g.model.queries.boundary('Body', dim=2):
        com = g.model.queries.center_of_mass(int(t), dim=int(d))
        if abs(com[2] - 10.0) < 1e-6:
            top_tag = int(t)
        elif abs(com[2] - 0.0) < 1e-6:
            bot_tag = int(t)
    assert top_tag is not None and bot_tag is not None
    g.physical.add_surface([top_tag], name='Top')
    g.physical.add_surface([bot_tag], name='Bottom')
    g.mesh.sizing.set_global_size(5.0)
    g.mesh.generation.generate(dim=3)
    return top_tag, bot_tag


def _top_node_ids(top_tag: int) -> set[int]:
    """Mesh node IDs touching the top face."""
    import gmsh
    nt, _, _ = gmsh.model.mesh.getNodes(
        dim=2, tag=top_tag, includeBoundary=True,
        returnParametricCoord=False,
    )
    return {int(n) for n in nt}


# =====================================================================
# Schema bump pin
# =====================================================================

def test_schema_version_bumped_to_2_7_0():
    """NEUTRAL_SCHEMA_VERSION advanced from 2.6.0 to 2.7.0."""
    assert NEUTRAL_SCHEMA_VERSION == "2.7.0"


# =====================================================================
# 1. Fail-loud: no declaration -> ndf_for raises with the help text
# =====================================================================

def test_ndf_for_undeclared_raises_helpful_lookuperror(g):
    """A FEM built without any g.node_ndf.set(...) call leaves every
    node at the sentinel; ndf_for raises LookupError naming both fixes.
    """
    _build_two_box_model(g)
    fem = g.mesh.queries.get_fem_data(dim=3)

    assert len(fem.nodes) > 0
    nid = int(next(iter(fem.nodes.ids)))

    with pytest.raises(LookupError) as exc_info:
        fem.nodes.ndf_for(nid)
    msg = str(exc_info.value)
    assert "ndf not declared" in msg
    assert "g.node_ndf.set" in msg
    assert "g.node_ndf.set_default" in msg


# =====================================================================
# 2. Single region override + set_default fallback
# =====================================================================

def test_single_region_override_plus_default(g):
    """g.node_ndf.set('Top', ndf=6) + set_default(ndf=3) yields:

    - top-face nodes -> ndf=6
    - every other node -> ndf=3
    """
    top_tag, _ = _build_two_box_model(g)
    g.node_ndf.set('Top', ndf=6)
    g.node_ndf.set_default(ndf=3)

    fem = g.mesh.queries.get_fem_data(dim=3)
    top_nodes = _top_node_ids(top_tag)
    assert top_nodes, "top face must carry at least one mesh node"

    for nid in fem.nodes.ids:
        nid_i = int(nid)
        expected = 6 if nid_i in top_nodes else 3
        assert fem.nodes.ndf_for(nid_i) == expected, (
            f"node {nid_i} expected ndf={expected}, "
            f"got {fem.nodes.ndf_for(nid_i)}"
        )


# =====================================================================
# 3. Targeted-only (no default) -> uncovered nodes raise
# =====================================================================

def test_targeted_only_no_default_uncovered_nodes_raise(g):
    """Without set_default, nodes outside the targeted region remain
    at the sentinel and ndf_for raises for them."""
    top_tag, _ = _build_two_box_model(g)
    g.node_ndf.set('Top', ndf=6)

    fem = g.mesh.queries.get_fem_data(dim=3)
    top_nodes = _top_node_ids(top_tag)
    interior = {int(t) for t in fem.nodes.ids} - top_nodes
    assert interior, "model must have at least one interior node"

    # Top nodes resolved cleanly.
    for nid in top_nodes:
        assert fem.nodes.ndf_for(nid) == 6

    # Interior nodes raise the helpful LookupError.
    sample = next(iter(interior))
    with pytest.raises(LookupError):
        fem.nodes.ndf_for(sample)


# =====================================================================
# 4. H5 round-trip preserves ndf + hash
# =====================================================================

def test_ndf_round_trip_through_h5(g, tmp_path: Path):
    """The per-node ndf vector and snapshot_id survive to_h5/from_h5."""
    top_tag, _ = _build_two_box_model(g)
    g.node_ndf.set('Top', ndf=6)
    g.node_ndf.set_default(ndf=3)

    fem = g.mesh.queries.get_fem_data(dim=3)
    original = {
        int(t): fem.nodes.ndf_for(int(t)) for t in fem.nodes.ids
    }
    original_snap = fem.snapshot_id

    out = tmp_path / "ndf_round_trip.h5"
    fem.to_h5(str(out))

    with h5py.File(out, "r") as f:
        assert "nodes/ndf" in f
        assert f["nodes/ndf"].dtype == np.int8
        assert f["nodes/ndf"].shape == (len(fem.nodes),)

    rebuilt = FEMData.from_h5(str(out))
    rebuilt_map = {
        int(t): rebuilt.nodes.ndf_for(int(t)) for t in rebuilt.nodes.ids
    }
    assert rebuilt_map == original
    assert rebuilt.snapshot_id == original_snap


# =====================================================================
# 5. Forward compat 2.6.0 -> 2.7.0
# =====================================================================

def test_legacy_2_6_0_file_loads_without_ndf(g, tmp_path: Path):
    """A 2.6.0 file (no /nodes/ndf dataset) loads with _ndf is None
    and every ndf_for call raises."""
    g.model.geometry.add_box(0.0, 0.0, 0.0, 10.0, 10.0, 10.0, label='Body')
    g.model.sync()
    g.mesh.sizing.set_global_size(5.0)
    g.mesh.generation.generate(dim=3)
    fem = g.mesh.queries.get_fem_data(dim=3)

    out = tmp_path / "legacy_2_6_0.h5"
    fem.to_h5(str(out))

    # Reshape the on-disk file to look like a 2.6.0 emitter wrote it:
    # strip the ndf dataset, stamp schema attrs back, and clear the
    # snapshot_id (the loader skips the integrity check when the
    # attr is absent — see _femdata_h5_io.py:~1272).
    with h5py.File(out, "r+") as f:
        if "ndf" in f["nodes"]:
            del f["nodes"]["ndf"]
        f["meta"].attrs["schema_version"] = "2.6.0"
        f["meta"].attrs["neutral_schema_version"] = "2.6.0"
        if "snapshot_id" in f["meta"].attrs:
            del f["meta"].attrs["snapshot_id"]

    rebuilt = FEMData.from_h5(str(out))
    assert rebuilt.nodes._ndf is None

    nid = int(next(iter(rebuilt.nodes.ids)))
    with pytest.raises(LookupError):
        rebuilt.nodes.ndf_for(nid)


# =====================================================================
# 6. H5 length validation (malformed file)
# =====================================================================

def test_malformed_ndf_length_raises(g, tmp_path: Path):
    """Truncating /nodes/ndf to a wrong length triggers MalformedH5Error."""
    from apeGmsh.opensees.emitter.h5_reader import MalformedH5Error

    g.model.geometry.add_box(0.0, 0.0, 0.0, 10.0, 10.0, 10.0, label='Body')
    g.model.sync()
    g.mesh.sizing.set_global_size(5.0)
    g.mesh.generation.generate(dim=3)
    g.node_ndf.set_default(ndf=3)
    fem = g.mesh.queries.get_fem_data(dim=3)

    out = tmp_path / "malformed_ndf.h5"
    fem.to_h5(str(out))

    # Truncate the ndf dataset by one element.
    with h5py.File(out, "r+") as f:
        truncated = f["nodes/ndf"][:-1]
        del f["nodes/ndf"]
        f["nodes"].create_dataset("ndf", data=truncated)

    with pytest.raises(MalformedH5Error) as exc_info:
        FEMData.from_h5(str(out))
    assert "/nodes/ndf shape" in str(exc_info.value)


# =====================================================================
# 7. Hash regression — different declarations -> different snapshot_id
# =====================================================================

def test_hash_changes_when_ndf_changes(g, tmp_path: Path):
    """Two FEMs with identical geometry but different ndf declarations
    must hash to different snapshot_ids."""
    # First build — uniform ndf=3.
    g.model.geometry.add_box(0.0, 0.0, 0.0, 10.0, 10.0, 10.0, label='Body')
    g.model.sync()
    g.mesh.sizing.set_global_size(5.0)
    g.mesh.generation.generate(dim=3)
    g.node_ndf.set_default(ndf=3)
    fem_a = g.mesh.queries.get_fem_data(dim=3)
    snap_a = fem_a.snapshot_id

    # Second build — uniform ndf=6.  Drop and re-declare.
    # ``_fem_built`` is reset at the top of every ``_from_gmsh`` call,
    # so the second ``get_fem_data()`` correctly picks up the new
    # declaration without a workaround.
    g.node_ndf.clear()
    g.node_ndf.set_default(ndf=6)
    fem_b = g.mesh.queries.get_fem_data(dim=3)
    snap_b = fem_b.snapshot_id

    assert snap_a != snap_b, (
        "Identical geometry with different ndf declarations must "
        "produce different snapshot_ids."
    )


def test_hash_stable_for_same_declarations(g):
    """Same geometry, same declarations -> same snapshot_id across
    two extractions."""
    g.model.geometry.add_box(0.0, 0.0, 0.0, 10.0, 10.0, 10.0, label='Body')
    g.model.sync()
    g.mesh.sizing.set_global_size(5.0)
    g.mesh.generation.generate(dim=3)
    g.node_ndf.set_default(ndf=3)

    fem_a = g.mesh.queries.get_fem_data(dim=3)
    fem_b = g.mesh.queries.get_fem_data(dim=3)
    assert fem_a.snapshot_id == fem_b.snapshot_id


def test_hash_diverges_when_coverage_differs(g):
    """Two FEMs with the same geometry + same default but different
    *targeted* ndf coverage produce different snapshot_ids.

    FEM A: set('Top', ndf=6) + set_default(ndf=3) -> Top=6, rest=3.
    FEM B: set('Top', ndf=3) + set_default(ndf=3) -> all=3.

    The resolved arrays differ on the top-face nodes, so the
    snapshot_ids must differ — proving the hash actually folds in
    the *per-node* ndf vector rather than a coarse digest.
    """
    # First build — top face overridden to 6.
    _build_two_box_model(g)
    g.node_ndf.set("Top", ndf=6)
    g.node_ndf.set_default(ndf=3)
    fem_a = g.mesh.queries.get_fem_data(dim=3)
    snap_a = fem_a.snapshot_id

    # Second build — top face matches the default (effectively uniform).
    g.node_ndf.clear()
    g.node_ndf.set("Top", ndf=3)
    g.node_ndf.set_default(ndf=3)
    fem_b = g.mesh.queries.get_fem_data(dim=3)
    snap_b = fem_b.snapshot_id

    assert snap_a != snap_b, (
        "Different per-node ndf coverage (Top=6 vs Top=3) must "
        "produce different snapshot_ids."
    )


def test_hash_insensitive_to_declaration_order(g):
    """Two FEMs whose declarations differ only in call order — but
    produce the same final resolved ``_ndf`` array — must hash to the
    same snapshot_id.

    Order A: ``set('Top', ndf=6)`` then ``set_default(ndf=3)``.
    Order B: ``set_default(ndf=3)`` then ``set('Top', ndf=6)``.

    Per the resolver semantics (targeted defs apply first; default
    fills sentinels), both orders yield the same final array
    (Top nodes → 6, rest → 3).  The snapshot_id is over the *resolved
    state* of ``_ndf``, not the declaration list — so the hashes
    must agree.
    """
    # First build — order A.
    _build_two_box_model(g)
    g.node_ndf.set("Top", ndf=6)
    g.node_ndf.set_default(ndf=3)
    fem_a = g.mesh.queries.get_fem_data(dim=3)
    snap_a = fem_a.snapshot_id

    # Second build — order B, same geometry.
    g.node_ndf.clear()
    g.node_ndf.set_default(ndf=3)
    g.node_ndf.set("Top", ndf=6)
    fem_b = g.mesh.queries.get_fem_data(dim=3)
    snap_b = fem_b.snapshot_id

    assert snap_a == snap_b, (
        "Declaration order should not change snapshot_id when the "
        "resolved ndf array is identical."
    )


# =====================================================================
# 8. set after extraction -> UserWarning
# =====================================================================

def test_set_after_extraction_warns(g):
    """Mutating g.node_ndf after get_fem_data() emits UserWarning."""
    g.model.geometry.add_box(0.0, 0.0, 0.0, 10.0, 10.0, 10.0, label='Body')
    g.model.sync()
    g.mesh.sizing.set_global_size(5.0)
    g.mesh.generation.generate(dim=3)
    g.node_ndf.set_default(ndf=3)

    # Build the broker — flips session._fem_built to True.
    _ = g.mesh.queries.get_fem_data(dim=3)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        g.node_ndf.set_default(ndf=6)
    user_warnings = [w for w in caught if issubclass(w.category, UserWarning)]
    assert user_warnings, (
        "expected UserWarning when g.node_ndf is mutated after "
        "get_fem_data() — none was emitted."
    )
    assert "get_fem_data" in str(user_warnings[0].message)


# =====================================================================
# 9. Resolver KeyError propagates
# =====================================================================

def test_set_unknown_target_raises_keyerror_at_extraction(g):
    """A g.node_ndf.set with a missing target propagates KeyError from
    get_fem_data() — per the dimensional resolution contract, the
    factory must not silently swallow it."""
    g.model.geometry.add_box(0.0, 0.0, 0.0, 10.0, 10.0, 10.0, label='Body')
    g.model.sync()
    g.mesh.sizing.set_global_size(5.0)
    g.mesh.generation.generate(dim=3)
    g.node_ndf.set("DoesNotExist", ndf=6)

    with pytest.raises(KeyError):
        g.mesh.queries.get_fem_data(dim=3)


# =====================================================================
# 10. from_msh path leaves ndf undeclared (sentinel)
# =====================================================================

def test_from_msh_leaves_ndf_undeclared(tmp_path: Path):
    """from_msh has no session and no NodeNDFComposite, so the broker
    is built with ndf=None and ndf_for raises with the help text.

    Does NOT take the ``g`` fixture: ``FEMData.from_msh`` opens its
    own gmsh session, and overlapping that with the fixture session
    leaves gmsh in an inconsistent state at teardown.
    """
    from apeGmsh import apeGmsh

    # Phase 1: build the .msh file in its own short-lived session.
    msh_path = tmp_path / "box.msh"
    with apeGmsh(model_name="from_msh_src", verbose=False) as g:
        g.model.geometry.add_box(0.0, 0.0, 0.0, 10.0, 10.0, 10.0, label='Body')
        g.model.sync()
        g.mesh.sizing.set_global_size(5.0)
        g.mesh.generation.generate(dim=3)

        import gmsh
        gmsh.write(str(msh_path))

    # Phase 2: from_msh has no session — broker is built with ndf=None.
    fem = FEMData.from_msh(str(msh_path), dim=3)
    assert fem.nodes._ndf is None

    nid = int(next(iter(fem.nodes.ids)))
    with pytest.raises(LookupError) as exc_info:
        fem.nodes.ndf_for(nid)
    assert "ndf not declared" in str(exc_info.value)


# =====================================================================
# 11. Composite API surface guards
# =====================================================================

def test_set_rejects_out_of_range_ndf(g):
    """set/set_default refuse ndf outside [1, 6]."""
    with pytest.raises(ValueError):
        g.node_ndf.set("Body", ndf=0)
    with pytest.raises(ValueError):
        g.node_ndf.set("Body", ndf=7)
    with pytest.raises(ValueError):
        g.node_ndf.set_default(ndf=0)


def test_set_rejects_invalid_ndf_types(g):
    """set/set_default refuse non-int ndf values (None, bool, float).

    ``bool`` is a subclass of ``int`` in Python so a naked
    ``isinstance(ndf, int)`` accepts it; the validator pre-empts
    that with an explicit ``isinstance(ndf, bool)`` guard.
    """
    # set(...) rejects.
    with pytest.raises(TypeError):
        g.node_ndf.set("Body", ndf=None)
    with pytest.raises(TypeError):
        g.node_ndf.set("Body", ndf=True)
    with pytest.raises(TypeError):
        g.node_ndf.set("Body", ndf=2.0)

    # set_default(...) rejects (same validator).
    with pytest.raises(TypeError):
        g.node_ndf.set_default(ndf=None)
    with pytest.raises(TypeError):
        g.node_ndf.set_default(ndf=True)


def test_set_default_replaces_not_appends(g):
    """Re-calling set_default replaces the existing default — the
    composite never carries two defaults."""
    g.node_ndf.set_default(ndf=3)
    g.node_ndf.set_default(ndf=6)
    defs = g.node_ndf.list()
    defaults = [d for d in defs if d.target is None]
    assert len(defaults) == 1
    assert defaults[0].ndf == 6


def test_composite_list_and_dunders(g):
    """list / __len__ / __iter__ / __repr__ all reflect declared defs."""
    assert len(g.node_ndf) == 0
    g.node_ndf.set("Body", ndf=6)
    g.node_ndf.set_default(ndf=3)
    assert len(g.node_ndf) == 2
    items = g.node_ndf.list()
    assert {d.ndf for d in items} == {3, 6}
    rep = repr(g.node_ndf)
    assert "NodeNDFComposite" in rep
    assert "default" in rep
