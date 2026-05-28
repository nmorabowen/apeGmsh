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

def test_schema_version_pin():
    """NEUTRAL_SCHEMA_VERSION pin — bump in lockstep with intentional
    additive changes to the neutral zone (S1b added /nodes/ndf at
    2.7.0; the ADR-0035 follow-up extended interpolation_payload_dtype
    + surface_coupling sr_* lane to round-trip embedded -K/-KP/-rot/-p
    at 2.8.0; ADR 0038 Phase 3A.1 added /composed_from/, tag_span_max,
    and module_label parallel datasets at 2.9.0; B2 split
    /physical_groups/ + /labels/ into node_side/element_side sub-trees
    and widened snapshot_id to fold element-side PGs + labels at
    2.10.0)."""
    assert NEUTRAL_SCHEMA_VERSION == "2.10.0"


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
# 5. Forward compat 2.7.0 -> 2.8.0
# =====================================================================

def test_legacy_2_7_0_file_without_ndf_loads_with_none(g, tmp_path: Path):
    """A prior-minor-shaped file with no /nodes/ndf dataset loads
    cleanly under the current reader, leaving ``_ndf=None``.

    Originally written for the 2.6.0 → 2.7.0 bump and asserted the
    reader *synthesised* an all-zero sentinel.  After successive
    minor bumps (now at 2.9.0 with the 2.8.x window), 2.7.0 is
    outside the two-version window; the synthesis branch is
    therefore only reachable for pre-2.7.0 files, which the schema-
    validation gate refuses upstream (dead path).  What the reader
    is actually expected to do for 2.7.0+ files with no
    ``/nodes/ndf`` is the second branch: leave ``_ndf=None``.  Both
    None and all-zero hash identically under the S2 fold gate, so
    the snapshot_id integrity check still fires.

    The downgrade stamp here is whatever the current prior-minor
    accepts (2.8.0 at time of 2.9.0 reader).
    """
    g.model.geometry.add_box(0.0, 0.0, 0.0, 10.0, 10.0, 10.0, label='Body')
    g.model.sync()
    g.mesh.sizing.set_global_size(5.0)
    g.mesh.generation.generate(dim=3)
    fem = g.mesh.queries.get_fem_data(dim=3)
    original_snap = fem.snapshot_id

    out = tmp_path / "legacy_2_7_0.h5"
    fem.to_h5(str(out))

    with h5py.File(out, "r+") as f:
        assert "ndf" in f["nodes"], (
            "writer regressed: /nodes/ndf was not stored for a "
            "no-declarations FEM."
        )
        del f["nodes"]["ndf"]
        # Down-stamp into the current prior-minor (within the two-
        # version window).  Tracks NEUTRAL_PRIOR_MINOR — auto-floats
        # with the schema bump.
        from tests.fixtures.schema import NEUTRAL_PRIOR_MINOR
        f["meta"].attrs["schema_version"] = NEUTRAL_PRIOR_MINOR
        f["meta"].attrs["neutral_schema_version"] = NEUTRAL_PRIOR_MINOR
        assert "snapshot_id" in f["meta"].attrs, (
            "writer regressed: /meta/snapshot_id was not stored."
        )

    rebuilt = FEMData.from_h5(str(out))

    # 2.7.0+ with no /nodes/ndf → _ndf stays None (not synthesised).
    assert rebuilt.nodes._ndf is None
    assert rebuilt.snapshot_id == original_snap, (
        "Rebuilt FEM's snapshot_id must equal the originally-stored "
        "value — proving the no-ndf path still hashes symmetrically."
    )

    # _ndf=None means ndf_for still raises the helpful LookupError.
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


def test_hash_symmetric_across_empty_channel_shapes(tmp_path: Path):
    """S2 locked design — the ``_ndf`` hash fold must be skipped for
    every "empty channel" shape, and all such shapes must hash
    identically.

    Empty-channel cases the hash gate handles:
      * ``_ndf is None``         — from_msh (no NodeNDFComposite),
                                   direct test fixtures
      * ``_ndf = np.zeros(N)``   — from_gmsh with no ``g.node_ndf``
                                   declarations
      * ``_ndf = np.zeros(N)``   — from_h5 of a 2.6.x file (reader
                                   synthesises the all-sentinel array)

    All three shapes denote "the user declared no per-node ndf".
    The hash must be identical across them — otherwise from_msh and
    from_gmsh of the same uniform-ndf geometry would hash
    differently, breaking the lineage chain.

    Does NOT take the ``g`` fixture; ``from_msh`` opens its own
    gmsh session.
    """
    from apeGmsh.mesh._femdata_hash import compute_snapshot_id

    # Build a single from_msh FEM and use it as the test vehicle.
    msh_path = tmp_path / "box.msh"
    from apeGmsh import apeGmsh
    with apeGmsh(model_name="hash_sym_src", verbose=False) as g:
        g.model.geometry.add_box(0.0, 0.0, 0.0, 10.0, 10.0, 10.0, label='Body')
        g.model.sync()
        g.mesh.sizing.set_global_size(5.0)
        g.mesh.generation.generate(dim=3)
        import gmsh
        gmsh.write(str(msh_path))

    fem = FEMData.from_msh(str(msh_path), dim=3)

    # Sanity: S2 design — from_msh leaves _ndf=None.
    assert fem.nodes._ndf is None, (
        "S2 regressed: from_msh stamped _ndf array instead of leaving "
        "the channel empty."
    )

    # Hash with _ndf=None (the actual from_msh state).
    h_none = compute_snapshot_id(fem)

    # Hash with _ndf=zeros (mimics from_gmsh-no-declarations state).
    saved = fem.nodes._ndf
    try:
        fem.nodes._ndf = np.zeros(len(fem.nodes.ids), dtype=np.int8)
        h_zeros = compute_snapshot_id(fem)
    finally:
        fem.nodes._ndf = saved

    assert h_none == h_zeros, (
        "S2 hash symmetry regressed: empty channel shapes "
        "(_ndf=None vs _ndf=zeros) hashed differently. The fold gate "
        "in _femdata_hash._hash_nodes must skip both cases."
    )

    # And a declared array (positive values) must hash differently.
    saved = fem.nodes._ndf
    try:
        arr = np.zeros(len(fem.nodes.ids), dtype=np.int8)
        arr[0] = 6
        fem.nodes._ndf = arr
        h_declared = compute_snapshot_id(fem)
    finally:
        fem.nodes._ndf = saved

    assert h_none != h_declared, (
        "S2 hash channel regressed: a declared ndf produced the same "
        "hash as the empty channel. The fold gate must fire when "
        "any element is non-zero."
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
# 10. from_msh path leaves ndf undeclared (None)
# =====================================================================

def test_from_msh_leaves_ndf_undeclared(tmp_path: Path):
    """from_msh has no session and no NodeNDFComposite, so the broker
    is built with ``_ndf=None`` (S2 locked design: leave the channel
    empty rather than synthesise a zero array).  ``ndf_for`` raises
    with the help text for every node.  The S2 emit-side
    ``try/except LookupError`` pattern absorbs this and falls back
    to the apeSees envelope ``ndf=K``, so models loaded via
    ``from_msh`` emit correctly under any uniform-ndf bridge.

    Hash symmetry between ``from_msh`` (``_ndf=None``) and
    ``from_gmsh`` with no declarations (``_ndf=zeros``) is preserved
    by the updated hash fold gate in
    ``_femdata_hash._hash_nodes``: both empty-channel cases skip the
    fold and produce identical digests.

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

    # Phase 2: from_msh has no session — broker carries ``_ndf=None``.
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


# =====================================================================
# 12. S2 emit-side wiring (ADR 0033)
# =====================================================================
# Each test exercises one of the documented S2 contract points:
#   12.1 mixed-ndf shell-on-solid emit (headline use case)
#   12.2 no-overrides byte-identical emit
#   12.3 apeSees.model() validator raises BridgeError
#   12.4 OpenSeesModel.from_h5() validator raises BridgeError
#   12.5 OpenSeesModel.from_compose_buffers() validator raises BridgeError
#   12.6 H5 round-trip preserves per-node ndf
#   12.7 from_msh emits with envelope (no -ndf tokens)
#   12.8 phantom nodes still emit ndf=6 under the new predicate

def _capture_owned_node_calls(fem):
    """Drive ``apeSees(fem).build().emit(RecordingEmitter())`` and
    return only the owned-node calls keyed by tag.

    Returns ``{tag: kwargs}`` for every ``node()`` call captured —
    excludes phantom-emit calls (they sit in the constraints pass,
    which fires AFTER element/node emission for the FEM's own nodes).
    The headline S2 contract is "broker per-node ndf travels to the
    emitted node call"; this helper isolates that surface.
    """
    from apeGmsh.opensees import apeSees
    from apeGmsh.opensees.emitter.recording import RecordingEmitter

    ops = apeSees(fem)
    # Use ndf=6 so the envelope covers any per-node 6 the user declared.
    ops.model(ndm=3, ndf=6)
    bm = ops.build()
    rec = RecordingEmitter()
    bm.emit(rec)
    # The bridge emits FEM-owned nodes first (no MP-constraint pass on
    # this minimal fixture); collect the ``node()`` calls keyed by tag.
    owned: dict[int, dict] = {}
    for name, args, kwargs in rec.calls:
        if name != "node":
            continue
        owned[int(args[0])] = kwargs
    return owned


def test_emit_mixed_ndf_shell_on_solid_flat(g):
    """Headline S2 use case — mixed-ndf model emits per-node ``-ndf K``.

    Build a solid box with a top-face PG, declare the top face as
    ``ndf=6`` (shell-like) and the rest as ``ndf=3`` (solid-like).
    The owned-node emit pass must source the ndf from the broker:
    top-face nodes get ``ndf=6``, every other node gets ``ndf=3``.
    """
    top_tag, _ = _build_two_box_model(g)
    g.node_ndf.set("Top", ndf=6)
    g.node_ndf.set_default(ndf=3)

    fem = g.mesh.queries.get_fem_data(dim=3)
    top_nodes = _top_node_ids(top_tag)
    assert top_nodes, "top face must carry at least one mesh node"

    owned = _capture_owned_node_calls(fem)

    # Every FEM node must have been emitted exactly once.
    expected_tags = {int(t) for t in fem.nodes.ids}
    assert set(owned.keys()) == expected_tags, (
        f"emit missed nodes: expected {expected_tags}, got {set(owned.keys())}"
    )

    for nid, kwargs in owned.items():
        expected = 6 if nid in top_nodes else 3
        assert kwargs.get("ndf") == expected, (
            f"node {nid}: expected emitted ndf={expected}, got {kwargs!r}"
        )


def test_emit_no_overrides_byte_identical(g):
    """A uniform-ndf model with no ``g.node_ndf`` calls emits NO per-node
    ``-ndf`` overrides — the envelope from ``apeSees.model(ndf=K)``
    supplies every node's DOF count.

    Locks down the S2 backcompat promise: existing scripts that never
    touched ``g.node_ndf`` produce byte-identical output (no spurious
    ``-ndf`` tokens added to owned-node emit).  String-search on the
    captured emitter output for the absence of ``-ndf``.
    """
    # Build a uniform-ndf model.  Crucially: NO g.node_ndf calls.
    g.model.geometry.add_box(0.0, 0.0, 0.0, 10.0, 10.0, 10.0, label='Body')
    g.model.sync()
    g.mesh.sizing.set_global_size(5.0)
    g.mesh.generation.generate(dim=3)
    fem = g.mesh.queries.get_fem_data(dim=3)

    owned = _capture_owned_node_calls(fem)
    assert owned, "fixture must emit at least one owned node"

    # No per-node -ndf overrides — every node call has empty kwargs.
    for nid, kwargs in owned.items():
        assert "ndf" not in kwargs, (
            f"S2 backcompat regressed: node {nid} emitted with ndf={kwargs['ndf']} "
            f"despite no g.node_ndf declarations.  Envelope must win for "
            f"undeclared nodes."
        )

    # String-search on a real py-deck emission for the same fixture.
    # The model directive itself carries ``-ndf`` (``ops.model('basic',
    # '-ndm', 3, '-ndf', 3)``) which is correct — we only assert that
    # no PER-NODE override is emitted (``ops.node(..., '-ndf', K)`` or
    # ``-ndf K`` tail on a node line).
    from apeGmsh.opensees import apeSees
    from apeGmsh.opensees.emitter.py import PyEmitter
    ops = apeSees(fem)
    ops.model(ndm=3, ndf=3)
    bm = ops.build()
    emitter = PyEmitter()
    bm.emit(emitter)
    node_lines = [ln for ln in emitter.lines() if "ops.node(" in ln]
    assert node_lines, "fixture must produce at least one ops.node line"
    for line in node_lines:
        assert "-ndf" not in line, (
            f"S2 backcompat regressed: node line contains -ndf override "
            f"despite no g.node_ndf declarations.  Line: {line!r}"
        )


def test_validator_at_apesees_model_raises_bridgeerror(g):
    """``apeSees(fem).model(ndm=3, ndf=3)`` after declaring
    ``g.node_ndf.set('Top', ndf=6)`` raises ``BridgeError`` naming
    the offending node and the fix.  Validator site 1 of 3.
    """
    from apeGmsh.opensees import apeSees
    from apeGmsh.opensees._internal.build import BridgeError

    top_tag, _ = _build_two_box_model(g)
    g.node_ndf.set("Top", ndf=6)
    g.node_ndf.set_default(ndf=3)

    fem = g.mesh.queries.get_fem_data(dim=3)
    ops = apeSees(fem)
    # ndf=3 envelope cannot host a node declared with ndf=6.
    with pytest.raises(BridgeError) as exc_info:
        ops.model(ndm=3, ndf=3)
    msg = str(exc_info.value)
    assert "ndf=6" in msg, f"BridgeError must name the declared ndf: {msg!r}"
    assert "ndf=" in msg and ("raise" in msg or "model" in msg), (
        f"BridgeError must name the fix (raise envelope ndf): {msg!r}"
    )


def test_validator_at_from_h5_raises_bridgeerror(g, tmp_path: Path):
    """``OpenSeesModel.from_h5(path)`` raises ``BridgeError`` when the
    stored envelope is smaller than the largest per-node declaration.
    Validator site 2 of 3 — catches hand-mutated or hand-built H5
    files that bypass ``apeSees.model``.
    """
    from apeGmsh.opensees import apeSees, OpenSeesModel
    from apeGmsh.opensees._internal.build import BridgeError

    # Build + write a valid mixed-ndf model (envelope=6 covers per-node 6).
    top_tag, _ = _build_two_box_model(g)
    g.node_ndf.set("Top", ndf=6)
    g.node_ndf.set_default(ndf=3)
    fem = g.mesh.queries.get_fem_data(dim=3)
    ops = apeSees(fem)
    ops.model(ndm=3, ndf=6)
    out = tmp_path / "mixed_ndf_model.h5"
    ops.h5(str(out))

    # Hand-mutate /meta.ndf down to 3 — the stored per-node ndf=6
    # entries now exceed the envelope.  The bridge stamps the envelope
    # at /meta.ndf (h5_reader.open reads it from there).
    with h5py.File(out, "r+") as f:
        assert "meta" in f, "writer regressed: /meta missing"
        f["meta"].attrs["ndf"] = 3

    with pytest.raises(BridgeError) as exc_info:
        OpenSeesModel.from_h5(str(out))
    assert "ndf=" in str(exc_info.value)


def test_validator_at_from_compose_buffers_raises_bridgeerror(g):
    """``OpenSeesModel.from_compose_buffers`` raises ``BridgeError``
    when the H5Emitter's envelope is below the broker's per-node max.
    Validator site 3 of 3 — programmatic compose flows that bypass
    ``apeSees.model``.
    """
    from apeGmsh.opensees import OpenSeesModel
    from apeGmsh.opensees._internal.build import BridgeError
    from apeGmsh.opensees.emitter.h5 import H5Emitter

    top_tag, _ = _build_two_box_model(g)
    g.node_ndf.set("Top", ndf=6)
    g.node_ndf.set_default(ndf=3)
    fem = g.mesh.queries.get_fem_data(dim=3)

    # Hand-build an emitter with the *wrong* envelope (3 < the broker's 6).
    emitter = H5Emitter()
    emitter.model(ndm=3, ndf=3)  # < max(broker._ndf) = 6

    with pytest.raises(BridgeError) as exc_info:
        OpenSeesModel.from_compose_buffers(
            fem, emitter, snapshot_id="test",
        )
    assert "ndf=" in str(exc_info.value)


def test_h5_round_trip_preserves_per_node_ndf(g, tmp_path: Path):
    """Build a mixed-ndf FEM, write H5, read back via ``OpenSeesModel``,
    re-emit via ``_replay_into``.  The per-node ndf must survive the
    round-trip — query ``fem.nodes.ndf_for(tag)`` on the rehydrated
    broker and confirm shell-node tags still resolve to 6.
    """
    from apeGmsh.opensees import apeSees, OpenSeesModel
    from apeGmsh.opensees.emitter.recording import RecordingEmitter

    top_tag, _ = _build_two_box_model(g)
    g.node_ndf.set("Top", ndf=6)
    g.node_ndf.set_default(ndf=3)
    fem = g.mesh.queries.get_fem_data(dim=3)
    top_nodes = _top_node_ids(top_tag)
    assert top_nodes

    ops = apeSees(fem)
    ops.model(ndm=3, ndf=6)
    out = tmp_path / "mixed_ndf_round_trip.h5"
    ops.h5(str(out))

    # Read back as OpenSeesModel.
    om = OpenSeesModel.from_h5(str(out))

    # 1. Broker survives the round-trip with per-node ndf intact.
    for nid in top_nodes:
        assert om._fem.nodes.ndf_for(int(nid)) == 6, (
            f"shell node {nid} lost its ndf=6 across H5 round-trip"
        )

    # 2. Re-emit via _populate_emitter → _replay_into and confirm
    #    per-node ``-ndf K`` overrides survive.
    rec = RecordingEmitter()
    om._populate_emitter(rec)
    emitted_ndf: dict[int, int | None] = {}
    for name, args, kwargs in rec.calls:
        if name == "node":
            emitted_ndf[int(args[0])] = kwargs.get("ndf")
    for nid in top_nodes:
        assert emitted_ndf.get(int(nid)) == 6, (
            f"shell node {nid} re-emitted with ndf={emitted_ndf.get(int(nid))}, "
            f"expected 6.  Tuple widening in _populate_emitter regressed."
        )


def test_from_msh_emits_with_envelope(tmp_path: Path):
    """Save a model to ``.msh``, load via ``from_msh``,
    ``apeSees(fem).model(ndm=3, ndf=6)``.  ``from_msh`` leaves
    ``_ndf=None`` (S2 design) so every ``ndf_for`` call raises
    ``LookupError`` and the emit-side ``try/except`` absorbs it,
    falling back to the envelope.  The emitted deck must carry NO
    per-node ``-ndf`` tokens — every node inherits the envelope.

    Does NOT take the ``g`` fixture: ``FEMData.from_msh`` opens its
    own gmsh session.
    """
    from apeGmsh import apeGmsh
    from apeGmsh.mesh.FEMData import FEMData
    from apeGmsh.opensees import apeSees
    from apeGmsh.opensees.emitter.py import PyEmitter

    msh_path = tmp_path / "box.msh"
    with apeGmsh(model_name="from_msh_envelope_src", verbose=False) as g:
        g.model.geometry.add_box(0.0, 0.0, 0.0, 10.0, 10.0, 10.0, label='Body')
        g.model.sync()
        g.mesh.sizing.set_global_size(5.0)
        g.mesh.generation.generate(dim=3)
        import gmsh
        gmsh.write(str(msh_path))

    fem = FEMData.from_msh(str(msh_path), dim=3)
    assert fem.nodes._ndf is None  # S2 design — empty channel.

    ops = apeSees(fem)
    ops.model(ndm=3, ndf=6)
    bm = ops.build()
    emitter = PyEmitter()
    bm.emit(emitter)
    # Per-node ``-ndf`` overrides must be absent (model directive's
    # ``-ndf 6`` is the envelope and is correct).  Check only node
    # lines: ``ops.node(...)``.
    node_lines = [ln for ln in emitter.lines() if "ops.node(" in ln]
    assert node_lines, "from_msh fixture must emit owned nodes"
    for line in node_lines:
        assert "-ndf" not in line, (
            f"from_msh path regressed: node line contains -ndf override "
            f"despite _ndf=None broker (envelope must win).  Line: "
            f"{line!r}"
        )


def test_phantom_nodes_unchanged_after_s2(g):
    """The phantom-node emit pass still produces ``ops.node(tag, ...,
    ndf=6)`` for every synthesized phantom, regardless of the FEM's
    per-node ndf channel.  This is the consumer-facing regression
    test for the side-channel → predicate-set refactor (ADR 0033):
    the new ``is_phantom_node(emitter, tag)`` predicate must
    correctly identify phantoms, AND the H5 emitter must still record
    them in ``/opensees/constraints/phantom_node_tags``.

    Uses the existing two-column fixture from
    ``test_constraint_emission_phase7b.py`` (a hand-built
    NodeToSurfaceRecord with deterministic phantom tags 200, 201).
    """
    from typing import cast, Any

    import numpy as np

    from apeGmsh._kernel.records._constraints import (
        NodePairRecord, NodeToSurfaceRecord,
    )
    from apeGmsh._kernel.records._kinds import ConstraintKind
    from apeGmsh.opensees._internal.build import emit_mp_constraints
    from apeGmsh.opensees._internal.tag_allocator import TagAllocator
    from apeGmsh.opensees._internal.tag_resolution import is_phantom_node
    from apeGmsh.opensees.emitter.h5 import H5Emitter
    from apeGmsh.opensees.emitter.recording import RecordingEmitter
    from tests.opensees.fixtures.fem_stub import make_two_column_frame

    fem = make_two_column_frame()
    n2s = NodeToSurfaceRecord(
        kind=ConstraintKind.NODE_TO_SURFACE,
        master_node=100,
        slave_nodes=[2, 4],
        phantom_nodes=[200, 201],
        phantom_coords=np.array([
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 1.0],
        ]),
        rigid_link_records=[
            NodePairRecord(
                kind=ConstraintKind.RIGID_BEAM,
                master_node=100, slave_node=200,
            ),
            NodePairRecord(
                kind=ConstraintKind.RIGID_BEAM,
                master_node=100, slave_node=201,
            ),
        ],
        equal_dof_records=[
            NodePairRecord(
                kind=ConstraintKind.EQUAL_DOF,
                master_node=200, slave_node=2, dofs=[1, 2, 3],
            ),
            NodePairRecord(
                kind=ConstraintKind.EQUAL_DOF,
                master_node=201, slave_node=4, dofs=[1, 2, 3],
            ),
        ],
        dofs=[1, 2, 3],
    )
    fem.add_node_constraints([n2s])

    # 1. RecordingEmitter — phantom nodes emit with ndf=6.
    rec = RecordingEmitter()
    emit_mp_constraints(rec, cast(Any, fem), TagAllocator())
    phantom_emits = [
        (args, kwargs)
        for name, args, kwargs in rec.calls
        if name == "node"
    ]
    assert len(phantom_emits) == 2
    assert phantom_emits[0] == ((200, 0.0, 0.0, 1.0), {"ndf": 6})
    assert phantom_emits[1] == ((201, 1.0, 0.0, 1.0), {"ndf": 6})

    # 2. is_phantom_node predicate — must classify phantoms correctly
    #    after emit_mp_constraints has installed the set.
    assert is_phantom_node(rec, 200) is True
    assert is_phantom_node(rec, 201) is True
    assert is_phantom_node(rec, 100) is False  # master, not phantom
    assert is_phantom_node(rec, 999) is False  # unknown tag

    # 3. H5 emitter — phantoms land in /opensees/constraints/phantom_node_tags.
    h5e = H5Emitter()
    h5e.model(ndm=3, ndf=3)
    # Pre-emit FEM-owned nodes (regulars carry no ndf=6 here).
    for nid, xyz in zip(fem.nodes.ids, fem.nodes.coords):
        h5e.node(int(nid), float(xyz[0]), float(xyz[1]), float(xyz[2]))
    # Then drive MP constraints — populates the phantom predicate set
    # AND emits the phantom nodes.
    emit_mp_constraints(h5e, cast(Any, fem), TagAllocator())
    # The H5 emitter accumulates phantom_node_tags via the predicate.
    assert sorted(h5e._phantom_node_tags) == [200, 201], (
        f"phantom_node_tags must contain only the phantom tags 200, 201; "
        f"got {h5e._phantom_node_tags}"
    )
