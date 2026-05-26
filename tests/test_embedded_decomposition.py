"""Property tests for the embedded-host decomposition contract.

Guards the linear-coupling-over-corners promise that
``ConstraintsComposite._collect_host_subelements`` makes when it
virtualises hex8 / quad4 / higher-order hosts into sub-tris and
sub-tets that ``ASDEmbeddedNodeElement`` can accept.

What is covered
---------------
1. The Kuhn 6-tet table ``HEX8_TO_6_TETS`` uses Gmsh's hex8 vertex
   ordering correctly — each sub-tet has positive volume in the
   canonical reference coordinates fetched live from
   ``gmsh.model.mesh.getElementProperties(5)``.  If Gmsh ever
   renumbers hex8 nodes, this test fails immediately rather than
   letting wrong-signed shape functions silently land in
   ``ASDEmbeddedNodeElement``.
2. The Kuhn 6 tets partition the unit cube — random interior points
   land in exactly one sub-tet with non-negative barycentric weights
   summing to one.
3. The quad4 → 2-triangle split covers the unit square — every
   random interior point lies in at least one sub-triangle.
4. End-to-end: an embedded node at the centroid of a single hex8
   produces an ``InterpolationRecord`` with 4 master nodes drawn
   from the hex's 8 corners, weights summing to 1, all
   non-negative.
"""
from __future__ import annotations

import numpy as np
import pytest

from apeGmsh import apeGmsh
from apeGmsh.core.ConstraintsComposite import (
    ConstraintsComposite,
    HEX8_TO_6_TETS,
    PRISM6_TO_3_TETS,
    PYRAMID5_TO_2_TETS,
)
from apeGmsh._kernel.resolvers._constraint_resolver._resolver import (
    _barycentric_tet4,
    _barycentric_tri3,
)


# ---------------------------------------------------------------------
# 1. Kuhn table uses Gmsh hex8 vertex ordering correctly
# ---------------------------------------------------------------------

def _hex8_reference_coords():
    """Fetch Gmsh's canonical hex8 (etype 5) reference vertex coords
    via the API — never hard-code, so a Gmsh ordering change is
    detected immediately."""
    import gmsh

    gmsh.initialize()
    try:
        _, _, _, num_nodes, local_coord, _ = (
            gmsh.model.mesh.getElementProperties(5))
    finally:
        gmsh.finalize()
    assert num_nodes == 8, f"hex8 should have 8 nodes, got {num_nodes}"
    coords = np.asarray(local_coord, dtype=float).reshape(8, 3)
    return coords


def _signed_tet_volume(v0, v1, v2, v3):
    return float(np.linalg.det(np.column_stack([
        v1 - v0, v2 - v0, v3 - v0]))) / 6.0


def test_kuhn_table_matches_gmsh_hex8_ordering():
    """Every row of ``HEX8_TO_6_TETS``, evaluated against Gmsh's
    canonical hex8 reference coordinates, must yield a right-handed
    (positive-volume) tet.  If Gmsh renumbers hex8 in a future
    release, this test fails — the table is no longer valid.
    """
    ref = _hex8_reference_coords()
    for row_idx, tet_idx in enumerate(HEX8_TO_6_TETS):
        v = ref[tet_idx]
        vol = _signed_tet_volume(v[0], v[1], v[2], v[3])
        assert vol > 0, (
            f"Kuhn row {row_idx} = {list(tet_idx)} gives signed "
            f"volume {vol:.6f} ≤ 0 against Gmsh hex8 ref coords "
            f"{ref.tolist()}.  Either Gmsh changed its hex8 vertex "
            f"ordering or the table is wrong; swap two indices in "
            f"that row to flip orientation."
        )


# ---------------------------------------------------------------------
# 2. Kuhn 6-tet decomposition partitions the unit cube
# ---------------------------------------------------------------------

def test_kuhn_tets_partition_unit_cube():
    """Every random interior point of the canonical hex8 cube falls
    into exactly one of the 6 Kuhn sub-tets (within float tolerance).
    Confirms (a) no gaps — coupling never silently fails for an
    interior embed; (b) no overlap large enough to give two different
    "best" sub-tets, which would make the per-embed master set
    non-deterministic.
    """
    ref = _hex8_reference_coords()
    rng = np.random.default_rng(20260525)
    n_points = 500
    # Sample strictly inside the unit cube by shrinking the range a
    # touch so points don't ride on faces / edges.
    pts = rng.uniform(-0.95, 0.95, size=(n_points, 3))

    bary_tol = 1e-9
    for p in pts:
        n_inside = 0
        for tet_idx in HEX8_TO_6_TETS:
            corners = ref[tet_idx]
            weights, excess, _ = _barycentric_tet4(p, corners)
            if excess is not None and excess <= bary_tol:
                n_inside += 1
                # Within-tet weights should sum to 1 and all be
                # non-negative.
                assert abs(weights.sum() - 1.0) < 1e-9
                assert (weights >= -bary_tol).all()
        assert n_inside == 1, (
            f"point {p} fell into {n_inside} Kuhn sub-tets "
            f"(expected exactly 1).  0 means decomposition has a "
            f"gap; >1 means decomposition has an overlap that "
            f"would make master-node selection non-deterministic."
        )


# ---------------------------------------------------------------------
# 3. Quad4 (0,2)-diagonal split covers a convex quad
# ---------------------------------------------------------------------

def test_quad4_split_covers_unit_square():
    """The (0,2)-diagonal split of a quad4 into 2 triangles must
    cover the quad — every interior point lies in at least one
    sub-triangle.  Failures here mean the resolver would
    fail-loud on a point that's geometrically inside the host quad
    but outside both diagonal-split sub-tris.
    """
    # Canonical quad4 in the (x, y) plane at z=0.
    corners = np.array([
        [-1.0, -1.0, 0.0],
        [+1.0, -1.0, 0.0],
        [+1.0, +1.0, 0.0],
        [-1.0, +1.0, 0.0],
    ])
    rng = np.random.default_rng(20260525)
    n_points = 500
    pts = np.column_stack([
        rng.uniform(-0.95, 0.95, size=n_points),
        rng.uniform(-0.95, 0.95, size=n_points),
        np.zeros(n_points),
    ])

    bary_tol = 1e-9
    for p in pts:
        in_tri_a, _, _ = _barycentric_tri3(p, corners[[0, 1, 2]])
        in_tri_b, _, _ = _barycentric_tri3(p, corners[[0, 2, 3]])
        excess_a = -float(in_tri_a.min())
        excess_b = -float(in_tri_b.min())
        # At least one excess <= bary_tol means "inside that tri".
        assert min(excess_a, excess_b) <= bary_tol, (
            f"interior point {p[:2]} of unit quad landed outside "
            f"both (0,1,2) and (0,2,3) sub-triangles "
            f"(excesses {excess_a:.3e}, {excess_b:.3e}).  The "
            f"(0,2) diagonal split is leaking coverage."
        )


# ---------------------------------------------------------------------
# 4. End-to-end — single hex8 + interior embed
# ---------------------------------------------------------------------

def test_embed_into_hex8_volume_resolves_to_4_corner_subset():
    """Mesh a hex8 volume, embed a node at its geometric centroid,
    and confirm the resolver produces an ``InterpolationRecord``
    whose 4 master nodes are a subset of one host hex's 8 corners,
    weights are non-negative and sum to 1.

    The test does not constrain how many hexes the mesher emits —
    only that the embed lands inside the volume and gets coupled
    linearly to one Kuhn sub-tet's 4 corners.
    """
    import gmsh

    from apeGmsh.core.constraints.defs import EmbeddedDef
    from apeGmsh._kernel.resolvers._constraint_resolver._resolver import (
        ConstraintResolver,
    )

    with apeGmsh(model_name="embed_hex8_volume", verbose=False) as g:
        surf = g.model.geometry.add_rectangle(0.0, 0.0, 0.0, 1.0, 1.0)
        g.model.sync()
        out = gmsh.model.occ.extrude(
            [(2, surf)], 0.0, 0.0, 1.0,
            numElements=[1], recombine=True)
        g.model.sync()
        vol_tag = next(t for d, t in out if d == 3)
        gmsh.option.setNumber("Mesh.RecombineAll", 1)
        gmsh.option.setNumber("Mesh.Recombine3DAll", 1)
        gmsh.option.setNumber("Mesh.SubdivisionAlgorithm", 2)
        g.mesh.sizing.set_global_size(0.5)
        g.mesh.generation.generate(3)

        rows = ConstraintsComposite._collect_host_subelements(
            [(3, vol_tag)])
        # Must be a multiple of 6 (Kuhn sub-tets per hex).
        assert rows.ndim == 2 and rows.shape[1] == 4, (
            f"hex8 host should decompose to length-4 tet rows, got "
            f"shape={rows.shape}")
        assert rows.shape[0] % 6 == 0, (
            f"expected 6 sub-tets per hex, got {rows.shape[0]} "
            f"rows (not a multiple of 6)")

        # Pull coords of every node referenced by the sub-tet rows.
        node_tags = np.unique(rows)
        node_coords = np.array([
            gmsh.model.mesh.getNode(int(t))[0] for t in node_tags])

        # Embed at the volume's geometric centroid (the mean of all
        # corner coords) — guaranteed inside for a convex hex mesh.
        centroid = node_coords.mean(axis=0)
        synth_tag = int(node_tags.max()) + 1
        all_tags = np.concatenate([node_tags, [synth_tag]])
        all_coords = np.vstack([node_coords, centroid[None, :]])

        resolver = ConstraintResolver(all_tags, all_coords)
        defn = EmbeddedDef(
            master_label="host", slave_label="embed",
            tolerance=1e-6, stiffness=1.0e18)
        records = resolver.resolve_embedded(
            defn, rows, {synth_tag})

        assert len(records) == 1, (
            f"one embedded point should produce one record, got "
            f"{len(records)}")
        rec = records[0]
        assert rec.slave_node == synth_tag
        assert len(rec.master_nodes) == 4, (
            f"hex8 host coupling is over 4 tet corners, got "
            f"{len(rec.master_nodes)} masters")
        assert set(rec.master_nodes).issubset(set(node_tags.tolist()))
        w = np.asarray(rec.weights, dtype=float)
        assert abs(w.sum() - 1.0) < 1e-9, (
            f"weights should sum to 1, got {w.sum()}")
        assert (w >= -1e-9).all(), (
            f"weights should all be non-negative for an interior "
            f"embed, got {w.tolist()}")


# ---------------------------------------------------------------------
# 5. host_coupling keyword is reserved
# ---------------------------------------------------------------------

# ---------------------------------------------------------------------
# 6. Prism / pyramid decomposition tables match Gmsh ordering
# ---------------------------------------------------------------------

def _ref_coords(etype: int, expected_n: int) -> np.ndarray:
    """Fetch canonical reference coords for *etype* from Gmsh."""
    import gmsh

    gmsh.initialize()
    try:
        _, _, _, n, local_coord, _ = (
            gmsh.model.mesh.getElementProperties(etype))
    finally:
        gmsh.finalize()
    assert n == expected_n, (
        f"etype {etype} should have {expected_n} nodes, got {n}")
    return np.asarray(local_coord, dtype=float).reshape(n, 3)


def test_prism6_table_matches_gmsh_ordering():
    """Each row of ``PRISM6_TO_3_TETS`` is right-handed in Gmsh's
    canonical prism6 reference coordinates."""
    ref = _ref_coords(6, 6)
    for row_idx, tet_idx in enumerate(PRISM6_TO_3_TETS):
        v = ref[tet_idx]
        vol = _signed_tet_volume(v[0], v[1], v[2], v[3])
        assert vol > 0, (
            f"prism6 sub-tet row {row_idx} = {list(tet_idx)} gives "
            f"signed volume {vol:.6f} ≤ 0 against Gmsh prism6 ref "
            f"coords {ref.tolist()}.  Swap two indices in the row "
            f"to flip orientation.")


def test_pyramid5_table_matches_gmsh_ordering():
    """Each row of ``PYRAMID5_TO_2_TETS`` is right-handed in Gmsh's
    canonical pyramid5 reference coordinates."""
    ref = _ref_coords(7, 5)
    for row_idx, tet_idx in enumerate(PYRAMID5_TO_2_TETS):
        v = ref[tet_idx]
        vol = _signed_tet_volume(v[0], v[1], v[2], v[3])
        assert vol > 0, (
            f"pyramid5 sub-tet row {row_idx} = {list(tet_idx)} gives "
            f"signed volume {vol:.6f} ≤ 0 against Gmsh pyramid5 ref "
            f"coords {ref.tolist()}.")


def test_prism6_tets_partition_reference_prism():
    """Every random interior point of the canonical prism6 falls into
    exactly one of the 3 sub-tets — no gaps, no overlaps."""
    ref = _ref_coords(6, 6)
    rng = np.random.default_rng(20260525)
    n_points = 300
    pts = []
    # Uniform sample inside the standard prism (triangle in (x, y)
    # with x+y <= 1, x >= 0, y >= 0; z in [-1, +1]).  Use rejection
    # against the triangle constraint to keep the sampler simple
    # and uniform.
    while len(pts) < n_points:
        x = rng.uniform(0.02, 0.96)
        y = rng.uniform(0.02, 0.96)
        if x + y >= 0.97:
            continue
        z = rng.uniform(-0.95, 0.95)
        pts.append([x, y, z])
    pts = np.array(pts)

    bary_tol = 1e-9
    for p in pts:
        n_inside = 0
        for tet_idx in PRISM6_TO_3_TETS:
            corners = ref[tet_idx]
            _, excess, _ = _barycentric_tet4(p, corners)
            if excess is not None and excess <= bary_tol:
                n_inside += 1
        assert n_inside == 1, (
            f"interior point {p} of unit prism fell into {n_inside} "
            f"sub-tets (expected exactly 1).")


def test_pyramid5_tets_partition_reference_pyramid():
    """Every random interior point of the canonical pyramid5 falls
    into exactly one of the 2 sub-tets."""
    ref = _ref_coords(7, 5)
    rng = np.random.default_rng(20260525)
    n_points = 300
    pts = []
    # Uniform sample inside the standard pyramid: square base
    # [-1,+1]^2 at z=0, apex at (0,0,1).  Cross-section at height
    # z is a square [-1+z, 1-z]^2.
    while len(pts) < n_points:
        z = rng.uniform(0.05, 0.95)
        half = 1.0 - z
        x = rng.uniform(-0.95 * half, 0.95 * half)
        y = rng.uniform(-0.95 * half, 0.95 * half)
        pts.append([x, y, z])
    pts = np.array(pts)

    bary_tol = 1e-9
    for p in pts:
        n_inside = 0
        for tet_idx in PYRAMID5_TO_2_TETS:
            corners = ref[tet_idx]
            _, excess, _ = _barycentric_tet4(p, corners)
            if excess is not None and excess <= bary_tol:
                n_inside += 1
        assert n_inside == 1, (
            f"interior point {p} of unit pyramid fell into "
            f"{n_inside} sub-tets (expected exactly 1).")


# ---------------------------------------------------------------------
# 7. Mixed-dim host fail-loud (B4)
# ---------------------------------------------------------------------

def test_mixed_dim_host_fails_loud():
    """A host PG that combines 2D and 3D entities must fail-loud at
    the collector — the linear coupling cannot pick between them
    deterministically.
    """
    import gmsh

    with apeGmsh(model_name="mixed_dim_host", verbose=False) as g:
        surf = g.model.geometry.add_rectangle(0.0, 0.0, 0.0, 1.0, 1.0)
        g.model.sync()
        out = gmsh.model.occ.extrude(
            [(2, surf)], 0.0, 0.0, 1.0,
            numElements=[1], recombine=True)
        g.model.sync()
        vol_tag = next(t for d, t in out if d == 3)
        # A separate 2D surface NOT on the brick (so the brick's own
        # surfaces don't double up).
        far_surf = g.model.geometry.add_rectangle(
            5.0, 5.0, 0.0, 1.0, 1.0)
        g.model.sync()
        gmsh.option.setNumber("Mesh.RecombineAll", 1)
        gmsh.option.setNumber("Mesh.Recombine3DAll", 1)
        g.mesh.sizing.set_global_size(2.0)
        g.mesh.generation.generate(3)

        with pytest.raises(ValueError, match="BOTH 2D sub-tris and 3D sub-tets"):
            ConstraintsComposite._collect_host_subelements(
                [(3, vol_tag), (2, far_surf)])


# ---------------------------------------------------------------------
# 8. Higher-order host emits one warning per (etype, entity) (B7)
# ---------------------------------------------------------------------

def test_higher_order_host_warns_once_per_entity():
    """Decomposing a midside-bearing host (tri6 / tet10 / quad8 /
    quad9 / hex20 / prism15 / pyramid13) must emit exactly one
    ``UserWarning`` per (etype, entity) pointing at the
    linear-coupling consequence — not one per element, not silent.
    """
    import gmsh
    import warnings as warnings_module

    with apeGmsh(model_name="tri6_host_warn", verbose=False) as g:
        surf = g.model.geometry.add_rectangle(0.0, 0.0, 0.0, 1.0, 1.0)
        g.model.sync()
        gmsh.option.setNumber("Mesh.RecombineAll", 0)
        gmsh.option.setNumber("Mesh.ElementOrder", 2)
        g.mesh.sizing.set_global_size(0.5)
        g.mesh.generation.generate(2)
        gmsh.model.mesh.setOrder(2)

        etypes, _, _ = gmsh.model.mesh.getElements(dim=2, tag=surf)
        if 9 not in [int(e) for e in etypes]:
            pytest.skip(
                f"expected tri6 (etype 9) after order=2 mesh; "
                f"got {[int(e) for e in etypes]}")

        with warnings_module.catch_warnings(record=True) as caught:
            warnings_module.simplefilter("always")
            ConstraintsComposite._collect_host_subelements([(2, surf)])

        embed_warnings = [
            w for w in caught
            if "embedded" in str(w.message)
            and "linear regardless" in str(w.message)
        ]
        assert len(embed_warnings) == 1, (
            f"expected exactly 1 warning for the tri6 host entity, "
            f"got {len(embed_warnings)}: "
            f"{[str(w.message) for w in embed_warnings]}")


# ---------------------------------------------------------------------
# 9. Sliver-tet behaviour (B6)
# ---------------------------------------------------------------------

def test_sliver_hex_still_resolves_or_fails_loud():
    """A high-aspect-ratio hex (100:1:1) decomposes into 6 slim
    sub-tets.  An embedded point at the centroid must EITHER produce
    a record with weights summing to 1 in [0, 1] (resolver handled
    the sliver) OR raise the fail-loud "outside every host element"
    error (resolver's degeneracy guard fired) — never silently
    produce nonsense weights.

    Documents the resolver's behaviour for badly-conditioned hosts so
    a future regression that silently drops the guard is caught.
    """
    import gmsh

    from apeGmsh.core.constraints.defs import EmbeddedDef
    from apeGmsh._kernel.resolvers._constraint_resolver._resolver import (
        ConstraintResolver,
    )

    with apeGmsh(model_name="sliver_hex_embed", verbose=False) as g:
        # 100 x 1 x 1 box meshed as 1 hex element by transfinite +
        # recombine.  Aspect ratio 100:1:1 produces slivers when
        # Kuhn-decomposed into tets.
        surf = g.model.geometry.add_rectangle(
            0.0, 0.0, 0.0, 100.0, 1.0)
        g.model.sync()
        out = gmsh.model.occ.extrude(
            [(2, surf)], 0.0, 0.0, 1.0,
            numElements=[1], recombine=True)
        g.model.sync()
        vol_tag = next(t for d, t in out if d == 3)
        gmsh.option.setNumber("Mesh.RecombineAll", 1)
        gmsh.option.setNumber("Mesh.Recombine3DAll", 1)
        # Force one element via large size.
        g.mesh.sizing.set_global_size(200.0)
        g.mesh.generation.generate(3)

        rows = ConstraintsComposite._collect_host_subelements(
            [(3, vol_tag)])
        if rows.shape[0] == 0:
            pytest.skip("no hex8 produced — mesher chose tet path")
        assert rows.shape[1] == 4

        node_tags = np.unique(rows)
        node_coords = np.array([
            gmsh.model.mesh.getNode(int(t))[0] for t in node_tags])
        centroid = node_coords.mean(axis=0)
        synth_tag = int(node_tags.max()) + 1
        all_tags = np.concatenate([node_tags, [synth_tag]])
        all_coords = np.vstack([node_coords, centroid[None, :]])

        resolver = ConstraintResolver(all_tags, all_coords)
        defn = EmbeddedDef(
            master_label="host", slave_label="embed",
            tolerance=1e-3, stiffness=1.0e18)

        try:
            records = resolver.resolve_embedded(
                defn, rows, {synth_tag})
        except ValueError as exc:
            # Fail-loud path — acceptable: the resolver caught the
            # degeneracy and raised rather than silently producing
            # nonsense.
            assert "outside every host element" in str(exc) or \
                   "tolerance" in str(exc)
            return

        # Resolved-path: weights must be finite, sum to 1, bounded.
        assert len(records) == 1
        w = np.asarray(records[0].weights, dtype=float)
        assert np.all(np.isfinite(w)), (
            f"sliver-tet resolver produced non-finite weights: "
            f"{w.tolist()}")
        assert abs(w.sum() - 1.0) < 1e-6, (
            f"sliver-tet weights should still sum to 1, got "
            f"{w.sum()}")
        assert (w >= -1e-6).all() and (w <= 1.0 + 1e-6).all(), (
            f"sliver-tet weights should remain bounded in [0, 1]; "
            f"got {w.tolist()}")


# ---------------------------------------------------------------------
# 10. host_coupling keyword reservation (re-asserted)
# ---------------------------------------------------------------------

def test_host_coupling_keyword_rejects_unimplemented_values():
    """Only ``host_coupling="linear"`` is currently supported.
    Reserving the keyword now (rather than after the fact) means a
    future ``"trilinear"`` / ``"biquadratic"`` option can be added
    without breaking existing models.
    """
    from apeGmsh.core.constraints.defs import EmbeddedDef

    EmbeddedDef(  # default value works
        master_label="h", slave_label="e")
    EmbeddedDef(  # explicit "linear" works
        master_label="h", slave_label="e", host_coupling="linear")

    with pytest.raises(ValueError, match="host_coupling.*reserved"):
        EmbeddedDef(
            master_label="h", slave_label="e",
            host_coupling="trilinear")
