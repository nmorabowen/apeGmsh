"""Tests for the ``sweep_dangling`` topology helper and the public
``g.model.geometry.find_orphans`` / ``.remove_orphans`` /
``.validate_pre_mesh`` surface.

The sweep is the single source of truth that ``slice``,
``cut_by_surface``, ``cut_by_plane``, and ``boolean.fragment``
all use to reap orphan dim<=2 geometry — these tests pin its
contract independently of the call sites.
"""
from __future__ import annotations

import gmsh
import pytest


# =====================================================================
# Dry-run / inspection
# =====================================================================

class TestFindOrphansDryRun:
    """``find_orphans()`` must inspect without modifying."""

    def test_sweep_dangling_dry_run_does_not_modify(self, g):
        """A dry-run must leave ``getEntities`` and ``_metadata``
        unchanged even when orphans exist.
        """
        # Build a model with a known orphan: add a standalone rectangle
        # then forcibly drop it from metadata so the sweep would
        # classify it as orphan.
        rect = g.model.geometry.add_rectangle(0, 0, 0, 1, 1)
        g.model._metadata.pop((2, rect), None)
        # Remove any auto-created label PG so labels.labels_for_entity
        # also returns nothing.
        for name in list(g.labels.get_all(dim=2)):
            try:
                if rect in g.labels.entities(name, dim=2):
                    g.labels.remove(name, dim=2)
            except KeyError:
                pass

        before_ents = {
            d: sorted(t for _, t in gmsh.model.getEntities(d))
            for d in range(4)
        }
        before_meta = dict(g.model._metadata)

        result = g.model.geometry.find_orphans()

        after_ents = {
            d: sorted(t for _, t in gmsh.model.getEntities(d))
            for d in range(4)
        }
        after_meta = dict(g.model._metadata)

        assert before_ents == after_ents, (
            f"find_orphans() modified entities: {before_ents} -> {after_ents}"
        )
        assert before_meta == after_meta, "find_orphans() touched _metadata"
        assert rect in result.get(2, []), (
            f"expected rect {rect} in dry-run report, got {result}"
        )


# =====================================================================
# User-intentional preservation
# =====================================================================

class TestSweepProtectsUserGeometry:
    """The sweep must not delete entities the user explicitly created."""

    def test_sweep_protects_2d_only_model_boundary(self, g):
        """In a 2D-only model the user's surface is the highest dim.
        Its bounding curves and points are NOT in volume-boundary
        (there ARE no volumes), but they ARE in the surface's own
        boundary closure — they must survive the sweep.

        Regression: an earlier version of the sweep protected only
        volume-bounding entities, so every 2D mesh setup tripped
        ``validate_pre_mesh`` because the surface's corner points and
        edges showed up as "orphans".
        """
        surf = g.model.geometry.add_rectangle(0, 0, 0, 1, 1, label='quad')
        # No volumes; the surface, its 4 edges, its 4 points are all
        # legitimate model state.
        assert g.model.geometry.find_orphans() == {0: [], 1: [], 2: []}, (
            "2D-only model misreports surface's boundary as orphan"
        )
        # validate_pre_mesh must also accept this clean 2D model.
        g.model.geometry.validate_pre_mesh()

    def test_sweep_protects_embedded_shell_boundary(self, g):
        """3D model with a standalone shell (embedded surface, not
        bounding any volume) — the shell's own boundary curves and
        points must survive even though they bound no volume.

        Regression: the earlier sweep used a "bounds a volume" filter
        only.  Embedded-shell workflows (cohesive crack planes,
        diaphragm shells inside soil) would lose the shell's
        boundary curves and points.
        """
        g.model.geometry.add_box(0, 0, 0, 10, 10, 10, label='soil')
        # An embedded planar shell inside the box.  add_rectangle
        # registers it in metadata so the shell itself is protected;
        # its bounding curves and points must follow.
        g.model.geometry.add_rectangle(2, 2, 5, 6, 6, label='diaphragm')
        assert g.model.geometry.find_orphans() == {0: [], 1: [], 2: []}, (
            "embedded shell's boundary curves/points misreported as orphans"
        )

    def test_sweep_dangling_protects_user_labeled_surface(self, g):
        """A surface that is in ``_metadata`` (created via
        ``add_rectangle``) survives a manual sweep even when it bounds
        no volume.  Its label PG must also survive — sweeping the
        geometry but dropping the label would still corrupt label-
        based resolution downstream.
        """
        g.model.geometry.add_box(0, 0, 0, 1, 1, 1, label='vol')
        rect = g.model.geometry.add_rectangle(
            5, 5, 5, 1, 1, label='floater',
        )
        # The rectangle is far away from the box, no volume bounds it.
        result = g.model.geometry.remove_orphans()
        existing = {t for _, t in gmsh.model.getEntities(2)}
        assert rect in existing, (
            f"sweep deleted user-labeled standalone rectangle {rect}; "
            f"removed dict was {result}"
        )
        assert 'floater' in g.labels.labels_for_entity(2, rect), (
            f"label 'floater' was dropped from surface {rect} during "
            f"the sweep — geometry survived but label-binding did not"
        )

    def test_sweep_dangling_protects_metadata_only_entity(self, g):
        """An entity in ``_metadata`` with no label still survives —
        metadata membership alone marks "user-intentional".
        """
        g.model.geometry.add_box(0, 0, 0, 1, 1, 1)
        # add_point with no label: metadata gets {(0, tag): {'kind': 'point'}}
        pt = g.model.geometry.add_point(10, 10, 10)
        g.model.geometry.remove_orphans()
        existing_pts = {t for _, t in gmsh.model.getEntities(0)}
        assert pt in existing_pts, (
            f"sweep deleted metadata-registered point {pt} (no label)"
        )

    def test_sweep_protects_raw_gmsh_entity_in_user_pg(self, g):
        """A line built via raw ``gmsh.model.occ.addLine`` and attached
        to a raw user physical group (no apeGmsh ``_metadata``, no
        ``g.labels`` entry) must still be classified as user-
        intentional and survive the sweep.

        Regression: PR #378 + #379 lacked this third channel; raw-
        gmsh frame workflows (the canonical FEM-Python pattern) had
        every line classified as orphan because they lived in
        neither ``_metadata`` nor ``g.labels``.  See
        :func:`_user_intentional` docstring channel #3.

        Uses raw OCC primitives instead of GEO to avoid the
        documented GEO/OCC kernel tag-collision issue (different
        kernels can mint the same tag for unrelated entities,
        confusing the sweep).
        """
        p_a = gmsh.model.occ.addPoint(5.0, 5.0, 5.0, 0.5)
        p_b = gmsh.model.occ.addPoint(6.0, 5.0, 5.0, 0.5)
        raw_line = gmsh.model.occ.addLine(p_a, p_b)
        gmsh.model.occ.synchronize()
        # Raw user PG — not an apeGmsh label, just a tagged group.
        gmsh.model.addPhysicalGroup(1, [raw_line], name="StructuralLine")

        # Precondition: the raw line has no _metadata entry.
        assert (1, raw_line) not in g.model._metadata

        # The open-world check must NOT flag the line — channel #3
        # (raw PG membership) protects it.  Endpoints inherit
        # protection via the boundary closure walk.
        orphans = g.model.geometry.find_orphans()
        assert raw_line not in orphans[1], (
            f"raw-gmsh line {raw_line} in user PG 'StructuralLine' "
            f"was misclassified as orphan; got dim=1 orphans {orphans[1]}"
        )
        assert p_a not in orphans[0] and p_b not in orphans[0], (
            f"endpoints of PG-protected line {raw_line} were flagged "
            f"as orphan; got dim=0 orphans {orphans[0]}"
        )

    def test_sweep_flags_raw_gmsh_entity_without_pg(self, g):
        """Companion to the test above: a raw line with NEITHER an
        apeGmsh metadata entry NOR any physical group IS flagged.
        Pins the negative case so the third channel is provably
        opt-in (you have to tag the entity for it to survive).
        """
        p_a = gmsh.model.occ.addPoint(7.0, 7.0, 7.0, 0.5)
        p_b = gmsh.model.occ.addPoint(8.0, 7.0, 7.0, 0.5)
        loose_line = gmsh.model.occ.addLine(p_a, p_b)
        gmsh.model.occ.synchronize()
        # NO addPhysicalGroup — the line is truly anonymous.

        orphans = g.model.geometry.find_orphans()
        assert loose_line in orphans[1], (
            f"raw-gmsh line {loose_line} with no metadata / label / "
            f"PG was NOT flagged; channel #3 must be opt-in. Got "
            f"dim=1 orphans {orphans[1]}"
        )


# =====================================================================
# Stale-metadata reaping
# =====================================================================

class TestSweepReapsStaleMetadata:
    """Stale ``_metadata`` keys (tags no longer in OCC) must be reaped."""

    def test_sweep_dangling_reaps_stale_metadata(self, g):
        """Manually pollute ``_metadata`` with a dead dimtag; the
        sweep must remove it.
        """
        g.model.geometry.add_box(0, 0, 0, 1, 1, 1)
        # Inject a fake metadata entry pointing at a non-existent tag.
        ghost = (2, 9999)
        g.model._metadata[ghost] = {'kind': 'ghost'}
        assert ghost in g.model._metadata

        g.model.geometry.remove_orphans()
        assert ghost not in g.model._metadata, (
            "stale _metadata entry was not reaped by the sweep"
        )


# =====================================================================
# Cross-op invariant
# =====================================================================

class TestNoOrphansAcrossOps:
    """No combination of slice / cut / fuse / fragment / intersect
    should leave orphans behind."""

    def test_no_orphans_after_slice_cut_fuse_chain(self, g):
        """A canonical multi-op chain — fragment + slice + fuse — must
        leave the model clean.
        """
        import warnings
        from apeGmsh.core._geometry_errors import WarnGeomCoincidentFace

        g.model.geometry.add_box(0, 0, 0, 2, 1, 1, label='a')
        g.model.geometry.add_box(1, 0, 0, 2, 1, 1, label='b')
        g.model.boolean.fragment(objects='a', tools='b')
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', WarnGeomCoincidentFace)
            g.model.geometry.slice(axis='z', offset=0.5)
        # Fuse everything back together — the surviving labels still
        # resolve since fragment/slice propagate them.
        g.model.boolean.fuse(
            objects=g.labels.entities('a', dim=3),
            tools=g.labels.entities('b', dim=3),
            label='ab',
        )
        assert g.model.geometry.find_orphans() == {0: [], 1: [], 2: []}

    def test_drm_box_10_consecutive_slices_no_orphan_accumulation(self, g):
        """The DRM workflow slices a single box at 10 evenly-spaced
        z-coordinates — the audit's most stress-case scenario for
        compounding orphan leakage.  Each slice's cutting plane is
        coincident with the previous slice's interior face, so a
        per-slice leak would compound into N=10 stranded surfaces by
        the end.

        The fix must hold: zero orphans at every dim and exactly 10
        volumes in the final model.
        """
        import warnings
        from apeGmsh.core._geometry_errors import WarnGeomCoincidentFace

        g.model.geometry.add_box(0, 0, 0, 1, 1, 1, label='drm')

        with warnings.catch_warnings():
            warnings.simplefilter('ignore', WarnGeomCoincidentFace)
            for i in range(1, 10):
                g.model.geometry.slice(
                    target='drm', axis='z', offset=i / 10.0, label='drm',
                )

        vols = [t for _, t in gmsh.model.getEntities(3)]
        assert len(vols) == 10, (
            f"expected 10 stacked sub-boxes from 9 slices, got {len(vols)}"
        )
        assert g.model.geometry.find_orphans() == {0: [], 1: [], 2: []}, (
            "10-slice DRM box accumulated orphan geometry — the fix "
            "regressed under compound coincident-face slicing"
        )

    def test_metadata_purged_for_every_consumed_entity(self, g):
        """After a chain of cut + fragment + slice ops, every key in
        ``model._metadata`` must point at a tag that currently exists
        in OCC — no stale keys.
        """
        import warnings
        from apeGmsh.core._geometry_errors import WarnGeomCoincidentFace

        g.model.geometry.add_box(-3.3, -0.8, -0.9, 6.6, 1.6, 0.9, label="outer")
        g.model.geometry.add_box(-3.025, -0.675, -0.6, 6.05, 1.35, 0.6, label="inner")
        g.model.boolean.cut(objects=["outer"], tools=["inner"], label="shell")
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', WarnGeomCoincidentFace)
            g.model.geometry.slice(target="shell", axis="z", offset=-0.6)

        gmsh.model.occ.synchronize()
        live = {(d, int(t)) for d in range(4)
                for _, t in gmsh.model.getEntities(d)}
        stale = [dt for dt in g.model._metadata if dt not in live]
        assert stale == [], (
            f"stale _metadata keys after multi-op chain: {stale}"
        )


# =====================================================================
# find_stale_metadata — the closed-world half of the split sweep
# =====================================================================

class TestFindStaleMetadata:
    """``find_stale_metadata`` only inspects ``_metadata`` keys that
    apeGmsh primitives recorded — so it cannot false-positive on raw
    ``gmsh.model.geo.*`` workflows by construction.
    """

    def test_clean_model_has_no_stale_metadata(self, g):
        """Every key the ``add_*`` primitives recorded points at a
        live OCC entity in a freshly built clean model.
        """
        g.model.geometry.add_box(0, 0, 0, 1, 1, 1, label='b')
        g.model.geometry.add_rectangle(5, 5, 5, 1, 1, label='r')
        assert g.model.geometry.find_stale_metadata() == []

    def test_polluted_metadata_is_reported(self, g):
        """Manually injecting a dead dimtag into ``_metadata`` must
        show up as stale.  Mirrors the post-fragment leak shape that
        the audit was chasing.
        """
        g.model.geometry.add_box(0, 0, 0, 1, 1, 1)
        ghost = (2, 9999)
        g.model._metadata[ghost] = {'kind': 'ghost'}
        stale = g.model.geometry.find_stale_metadata()
        assert ghost in stale, (
            f"injected ghost {ghost} not reported by "
            f"find_stale_metadata; got {stale}"
        )

    def test_raw_gmsh_geometry_has_no_stale_metadata(self, g):
        """A model built entirely via raw ``gmsh.model.geo.*`` (no
        apeGmsh primitives) populates no ``_metadata`` entries — so
        ``find_stale_metadata`` returns an empty list regardless of
        how many entities exist.  This is the key invariant: the
        closed-world check cannot punish raw-gmsh workflows because
        it has nothing to inspect for them.
        """
        gmsh.model.geo.addPoint(0, 0, 0, 0.5)
        gmsh.model.geo.addPoint(1, 0, 0, 0.5)
        gmsh.model.geo.addPoint(1, 1, 0, 0.5)
        gmsh.model.geo.addPoint(0, 1, 0, 0.5)
        gmsh.model.geo.addLine(1, 2)
        gmsh.model.geo.addLine(2, 3)
        gmsh.model.geo.addLine(3, 4)
        gmsh.model.geo.addLine(4, 1)
        gmsh.model.geo.synchronize()
        assert g.model.geometry.find_stale_metadata() == []


# =====================================================================
# validate_pre_mesh — split contract (strict default False)
# =====================================================================

class TestValidatePreMesh:
    """``validate_pre_mesh`` has two modes: the default ``strict=False``
    runs the closed-world metadata-stale check (auto-fired by
    ``Mesh.generate``); ``strict=True`` runs the open-world
    ``find_orphans`` check (opt-in).
    """

    def test_validate_pre_mesh_default_passes_on_clean_model(self, g):
        g.model.geometry.add_box(0, 0, 0, 1, 1, 1)
        # Must not raise — clean model, no stale metadata.
        g.model.geometry.validate_pre_mesh()

    def test_validate_pre_mesh_default_raises_on_stale_metadata(self, g):
        """``strict=False`` (default) fires on stale metadata."""
        from apeGmsh.core._geometry_errors import GeometryValidationError

        g.model.geometry.add_box(0, 0, 0, 1, 1, 1)
        # Inject a stale metadata key (apeGmsh primitives don't
        # normally produce these, but post-leak the sweep may).
        g.model._metadata[(2, 9999)] = {'kind': 'leaked'}
        with pytest.raises(GeometryValidationError):
            g.model.geometry.validate_pre_mesh()

    def test_validate_pre_mesh_default_ignores_open_world_orphans(self, g):
        """``strict=False`` does NOT fire on orphan geometry that
        isn't stale metadata.  Raw-gmsh-only models, dim-2 entities
        the user dropped from metadata, etc. — none of these trip
        the default check.  Open-world is opt-in.
        """
        g.model.geometry.add_box(0, 0, 0, 1, 1, 1, label='box')
        # Build an orphan in the open-world sense: a rect whose
        # metadata + labels have been stripped.  ``find_orphans``
        # would flag it; ``validate_pre_mesh()`` must not.
        rect = g.model.geometry.add_rectangle(5, 5, 5, 1, 1)
        g.model._metadata.pop((2, rect), None)
        for name in list(g.labels.get_all(dim=2)):
            try:
                if rect in g.labels.entities(name, dim=2):
                    g.labels.remove(name, dim=2)
            except KeyError:
                pass
        # Precondition: the open-world check WOULD flag the rect.
        assert g.model.geometry.find_orphans()[2], (
            "test precondition: open-world check should flag rect"
        )
        # Default-strict (False) must not raise.
        g.model.geometry.validate_pre_mesh()

    def test_validate_pre_mesh_strict_raises_on_open_world_orphan(self, g):
        """``strict=True`` runs ``find_orphans`` — opt-in open-world.
        """
        from apeGmsh.core._geometry_errors import GeometryValidationError

        g.model.geometry.add_box(0, 0, 0, 1, 1, 1, label='box')
        rect = g.model.geometry.add_rectangle(5, 5, 5, 1, 1)
        g.model._metadata.pop((2, rect), None)
        for name in list(g.labels.get_all(dim=2)):
            try:
                if rect in g.labels.entities(name, dim=2):
                    g.labels.remove(name, dim=2)
            except KeyError:
                pass
        with pytest.raises(GeometryValidationError):
            g.model.geometry.validate_pre_mesh(strict=True)


# =====================================================================
# Mesh.generate auto-wiring (closed-world only)
# =====================================================================

class TestMeshGenerateAutoValidate:
    """``Mesh.generate`` auto-invokes ``validate_pre_mesh()`` with the
    default ``strict=False`` — catches stale metadata without
    false-positiving on raw-gmsh workflows.
    """

    def test_mesh_generate_passes_with_raw_gmsh_geometry(self, g):
        """A model built entirely via raw ``gmsh.model.geo.*`` (no
        apeGmsh primitives, no _metadata population) must mesh
        cleanly.  Regression: the original PR #378 auto-wiring used
        open-world ``find_orphans`` and broke 63 tests of this shape.
        """
        plan, h = 5.0, 3.0
        lc = 1.5
        pts = []
        for x, y, z in [
            (0, 0, 0), (plan, 0, 0), (plan, plan, 0), (0, plan, 0),
            (0, 0, h), (plan, 0, h), (plan, plan, h), (0, plan, h),
        ]:
            pts.append(gmsh.model.geo.addPoint(x, y, z, lc))
        cols = [gmsh.model.geo.addLine(pts[i], pts[i + 4]) for i in range(4)]
        gmsh.model.geo.synchronize()
        gmsh.model.addPhysicalGroup(1, cols, name="Columns")
        g.mesh.sizing.set_global_size(lc)
        # Must not raise.
        g.mesh.generation.generate(1)

    def test_mesh_generate_raises_on_stale_metadata(self, g):
        """The auto-wire's whole point: a stale ``_metadata`` key
        (which means an apeGmsh op left an orphan) raises before the
        slow mesher runs.
        """
        from apeGmsh.core._geometry_errors import GeometryValidationError

        g.model.geometry.add_box(0, 0, 0, 1, 1, 1, label='box')
        g.model._metadata[(2, 9999)] = {'kind': 'leaked'}
        with pytest.raises(GeometryValidationError):
            g.mesh.generation.generate(3)
