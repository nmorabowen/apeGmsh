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
                    solid='drm', axis='z', offset=i / 10.0, label='drm',
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
            g.model.geometry.slice(solid="shell", axis="z", offset=-0.6)

        gmsh.model.occ.synchronize()
        live = {(d, int(t)) for d in range(4)
                for _, t in gmsh.model.getEntities(d)}
        stale = [dt for dt in g.model._metadata if dt not in live]
        assert stale == [], (
            f"stale _metadata keys after multi-op chain: {stale}"
        )


# =====================================================================
# validate_pre_mesh
# =====================================================================

class TestValidatePreMesh:
    """``validate_pre_mesh`` mirrors the other composite's contract:
    raise loudly when the model is unsafe to mesh."""

    def test_validate_pre_mesh_passes_on_clean_model(self, g):
        g.model.geometry.add_box(0, 0, 0, 1, 1, 1)
        # Must not raise.
        g.model.geometry.validate_pre_mesh()

    def test_validate_pre_mesh_raises_on_orphan_present(self, g):
        from apeGmsh.core._geometry_errors import GeometryValidationError

        rect = g.model.geometry.add_rectangle(0, 0, 0, 1, 1)
        # Strip metadata + labels to make it look orphan.
        g.model._metadata.pop((2, rect), None)
        for name in list(g.labels.get_all(dim=2)):
            try:
                if rect in g.labels.entities(name, dim=2):
                    g.labels.remove(name, dim=2)
            except KeyError:
                pass
        # Need a registered volume — without one, the rectangle would
        # be the only surface and the sweep would still flag it; but
        # being explicit avoids accidentally testing the 2D-only edge.
        g.model.geometry.add_box(5, 5, 5, 1, 1, 1)
        with pytest.raises(GeometryValidationError):
            g.model.geometry.validate_pre_mesh()

    def test_mesh_generate_does_not_auto_invoke_geometry_validator(self, g):
        """``Mesh.generate`` does NOT auto-invoke
        ``g.model.geometry.validate_pre_mesh()``.  The geometry
        validator is opt-in because raw ``gmsh.model.geo.*`` /
        ``gmsh.model.occ.*`` workflows and raw user PGs bypass the
        ``_metadata`` and label channels the validator uses to
        decide "user-intentional".  Auto-wiring would false-positive
        on every such workflow.

        Pin the contract: build a model whose orphans would trigger
        ``validate_pre_mesh`` if called, and confirm
        ``Mesh.generate`` succeeds without raising.
        """
        g.model.geometry.add_box(0, 0, 0, 1, 1, 1, label='box')
        rect = g.model.geometry.add_rectangle(5, 5, 5, 1, 1)
        g.model._metadata.pop((2, rect), None)
        for name in list(g.labels.get_all(dim=2)):
            try:
                if rect in g.labels.entities(name, dim=2):
                    g.labels.remove(name, dim=2)
            except KeyError:
                pass

        # find_orphans would flag the rect; mesh.generate must NOT
        # raise GeometryValidationError on its own.
        orphans = g.model.geometry.find_orphans()
        assert orphans[2], "test precondition: orphan rect should be flagged"
        # Should not raise.
        g.mesh.generation.generate(3)
