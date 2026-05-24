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

def test_legacy_2_6_0_file_loads_with_synthesized_sentinel(g, tmp_path: Path):
    """A 2.6.0-shaped file (no /nodes/ndf dataset) loads cleanly under
    the 2.7.0 reader.

    Bug 3 + 4 fix from the post-#317 audit: the reader synthesises an
    all-zero sentinel array when ``/nodes/ndf`` is absent, so the
    recomputed ``snapshot_id`` equals what was written.  This means
    the stored ``snapshot_id`` survives the strip and the
    integrity-check at ``_femdata_h5_io.py:~1272`` *actually fires*
    — proving the loader handles 2.6.x backcompat for real, not by
    short-circuiting the integrity check the way the original test
    did (Bug 4)."""
    g.model.geometry.add_box(0.0, 0.0, 0.0, 10.0, 10.0, 10.0, label='Body')
    g.model.sync()
    g.mesh.sizing.set_global_size(5.0)
    g.mesh.generation.generate(dim=3)
    fem = g.mesh.queries.get_fem_data(dim=3)
    original_snap = fem.snapshot_id

    out = tmp_path / "legacy_2_6_0.h5"
    fem.to_h5(str(out))

    # Reshape the on-disk file to look like a 2.6.0 emitter wrote it:
    # strip the ndf dataset and stamp schema attrs back.  Crucially,
    # we do NOT delete /meta/snapshot_id — under Bug 3's symmetric
    # initialisation the stored hash must still validate against the
    # rebuilt FEM (otherwise the loader's integrity guard raises).
    with h5py.File(out, "r+") as f:
        assert "ndf" in f["nodes"], (
            "writer regressed: /nodes/ndf was not stored for a "
            "no-declarations FEM."
        )
        del f["nodes"]["ndf"]
        f["meta"].attrs["schema_version"] = "2.6.0"
        f["meta"].attrs["neutral_schema_version"] = "2.6.0"
        assert "snapshot_id" in f["meta"].attrs, (
            "writer regressed: /meta/snapshot_id was not stored."
        )

    # Load — the integrity check at _femdata_h5_io.py:~1272 must
    # pass (recomputed hash == stored hash).  This proves the
    # backcompat path is exercised, not bypassed (Bug 4 fix).
    rebuilt = FEMData.from_h5(str(out))

    # Reader synthesised the all-zero sentinel array.
    assert rebuilt.nodes._ndf is not None
    assert rebuilt.nodes._ndf.dtype == np.int8
    assert int(rebuilt.nodes._ndf.sum()) == 0
    assert rebuilt.snapshot_id == original_snap, (
        "Rebuilt FEM's snapshot_id must equal the originally-stored "
        "value — proving Bug 3's symmetric hash initialisation."
    )

    # Every ndf_for still raises (sentinel-0 means undeclared).
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
    # The first post-extract mutation (``clear()``) correctly warns
    # that the cached broker won't see the change; it also clears
    # ``_fem_built`` so the rest of the batch is silent.  The next
    # ``get_fem_data()`` re-stamps the flag for the next round.
    with pytest.warns(UserWarning, match="get_fem_data"):
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
    # First post-extract mutation warns; subsequent in the batch are
    # silent (Bug 1 fix).
    with pytest.warns(UserWarning, match="get_fem_data"):
        g.node_ndf.clear()
    g.node_ndf.set("Top", ndf=3)
    g.node_ndf.set_default(ndf=3)
    fem_b = g.mesh.queries.get_fem_data(dim=3)
    snap_b = fem_b.snapshot_id

    assert snap_a != snap_b, (
        "Different per-node ndf coverage (Top=6 vs Top=3) must "
        "produce different snapshot_ids."
    )


def test_from_msh_folds_ndf_into_hash_like_from_gmsh(tmp_path: Path):
    """Bug 3 — the ``from_msh`` construction path must fold ``_ndf``
    into the snapshot_id digest the same way ``from_gmsh`` does.

    Pre-fix, ``from_msh`` left ``_ndf=None`` so the hash gate in
    ``_femdata_hash`` skipped the ndf section entirely, while
    ``from_gmsh`` always folded the (zero) array in — same geometry,
    different hash depending on load path.

    Note: we don't simply compare snapshot_ids across the two paths
    end-to-end because the .msh round-trip loses ~1e-16 of
    coordinate precision (verified empirically), so byte-equal
    coords are impossible regardless of Bug 3.  Instead this test
    isolates the *ndf channel fold*: hash the same FEMData with and
    without ``_ndf`` and assert the two values differ, proving the
    ndf array participates in ``from_msh``'s digest.  Combined with
    ``test_hash_changes_when_ndf_changes`` (which proves the same
    for ``from_gmsh``), the channel is symmetric.

    Does NOT take the ``g`` fixture; ``from_msh`` opens its own
    gmsh session.
    """
    from apeGmsh import apeGmsh
    from apeGmsh.mesh._femdata_hash import compute_snapshot_id

    msh_path = tmp_path / "box.msh"
    with apeGmsh(model_name="hash_sym_src", verbose=False) as g:
        g.model.geometry.add_box(0.0, 0.0, 0.0, 10.0, 10.0, 10.0, label='Body')
        g.model.sync()
        g.mesh.sizing.set_global_size(5.0)
        g.mesh.generation.generate(dim=3)
        import gmsh
        gmsh.write(str(msh_path))

    fem = FEMData.from_msh(str(msh_path), dim=3)

    # Sanity: Bug 3 populated ``_ndf`` to the all-zero sentinel.
    assert fem.nodes._ndf is not None, (
        "Bug 3 regressed: from_msh left _ndf=None instead of "
        "synthesising the zero-sentinel array."
    )

    # Snapshot with ``_ndf`` populated (the actual behaviour).
    h_with_ndf = compute_snapshot_id(fem)

    # Toggle ``_ndf`` to None and re-hash — this is what the hash
    # would have been pre-Bug-3.  If the two values agree, ``_ndf``
    # was NOT folded into the digest on the from_msh path.
    saved = fem.nodes._ndf
    try:
        fem.nodes._ndf = None
        h_without_ndf = compute_snapshot_id(fem)
    finally:
        fem.nodes._ndf = saved

    assert h_with_ndf != h_without_ndf, (
        "from_msh's snapshot_id ignored _ndf — Bug 3 fix regressed."
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
    # First post-extract mutation warns; subsequent in the batch are
    # silent (Bug 1 fix).
    with pytest.warns(UserWarning, match="get_fem_data"):
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


def test_clear_after_extraction_warns(g):
    """Calling g.node_ndf.clear() after get_fem_data() emits the same
    UserWarning as set/set_default — the cached broker still holds
    the pre-clear ndf array."""
    g.model.geometry.add_box(0.0, 0.0, 0.0, 10.0, 10.0, 10.0, label='Body')
    g.model.sync()
    g.mesh.sizing.set_global_size(5.0)
    g.mesh.generation.generate(dim=3)
    g.node_ndf.set_default(ndf=3)
    _ = g.mesh.queries.get_fem_data(dim=3)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        g.node_ndf.clear()
    user_warnings = [w for w in caught if issubclass(w.category, UserWarning)]
    assert user_warnings, (
        "expected UserWarning when g.node_ndf.clear() is called after "
        "get_fem_data() — none was emitted."
    )
    assert "clear" in str(user_warnings[0].message)
    assert "get_fem_data" in str(user_warnings[0].message)


def test_post_extraction_warning_batches_within_re_declaration_run(g):
    """A batch of post-extract mutations (clear → set_default → set)
    only warns ONCE — on the first call — because that call also
    clears ``_fem_built``.  Subsequent calls in the same batch are
    silent; the next ``get_fem_data()`` re-stamps the flag so the
    next round of mutations warns again.

    This locks down the Bug-1 fix: without the flag clear, every
    ``clear`` / ``set_default`` / ``set`` after the first build
    would warn even though the user is doing the right thing
    (re-declare then re-extract)."""
    g.model.geometry.add_box(0.0, 0.0, 0.0, 10.0, 10.0, 10.0, label='Body')
    g.model.sync()
    g.mesh.sizing.set_global_size(5.0)
    g.mesh.generation.generate(dim=3)
    g.node_ndf.set_default(ndf=3)
    _ = g.mesh.queries.get_fem_data(dim=3)

    # First batch: first mutation warns, subsequent are silent.
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        g.node_ndf.clear()
        g.node_ndf.set_default(ndf=6)
        g.node_ndf.set_default(ndf=4)
    user_warnings = [w for w in caught if issubclass(w.category, UserWarning)]
    assert len(user_warnings) == 1, (
        f"expected exactly one warning for the first post-extract "
        f"mutation in a batch, got {len(user_warnings)}"
    )

    # Re-extract restores the guard.
    _ = g.mesh.queries.get_fem_data(dim=3)

    # Second batch warns again on its first mutation.
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        g.node_ndf.set_default(ndf=5)
        g.node_ndf.set_default(ndf=3)
    user_warnings = [w for w in caught if issubclass(w.category, UserWarning)]
    assert len(user_warnings) == 1, (
        f"second batch must also warn on its first mutation; got "
        f"{len(user_warnings)} warning(s)"
    )


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
# 10. from_msh path leaves ndf at the all-sentinel array
# =====================================================================

def test_from_msh_leaves_ndf_at_sentinel(tmp_path: Path):
    """from_msh has no session and no NodeNDFComposite, so the broker
    is built with the all-zero sentinel array (Bug 3 — symmetric hash
    initialisation across construction paths) and ndf_for raises with
    the help text for every node.

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

    # Phase 2: from_msh has no session — broker carries the
    # all-sentinel array (zeros), not ``None``.
    fem = FEMData.from_msh(str(msh_path), dim=3)
    assert fem.nodes._ndf is not None
    assert fem.nodes._ndf.dtype == np.int8
    assert fem.nodes._ndf.shape == (len(fem.nodes.ids),)
    assert int(fem.nodes._ndf.sum()) == 0, (
        "from_msh must leave ndf at the all-zero sentinel — no "
        "declarations were made."
    )

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
