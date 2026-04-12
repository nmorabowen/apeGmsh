"""
Regression tests — Part physical-group anchors.

These tests exercise the full PG anchor round-trip against a live
Gmsh kernel:

1. Part with ``label=`` on geometry → auto-creates physical groups.
2. Sidecar JSON is written next to the auto-persisted STEP.
3. Sidecar only captures user-named PGs (unlabeled entities are
   skipped).
4. ``parts.add(part, label="col_A")`` re-creates PGs in the
   Assembly with instance-prefixed names (``col_A.shaft``).
5. PGs are accessible via ``g.physical.entities("col_A.shaft")``.
6. Transform cases: no transform, translate, rotate,
   translate + rotate.
7. Multiple instances of the same Part each get their own
   prefixed PGs.
8. ``save(..., write_anchors=False)`` suppresses the sidecar.
9. Missing sidecar falls back cleanly (no PGs, no crash).
"""
from __future__ import annotations

import math

import pytest

from apeGmsh import Part, apeGmsh
from apeGmsh.core._part_anchors import read_sidecar, sidecar_path


# =====================================================================
# Helpers
# =====================================================================

def _make_labeled_column() -> Part:
    """Build a Part with two user-named boxes and one unnamed point."""
    col = Part("column")
    with col:
        col.model.geometry.add_box(0, 0, 0, 1, 1, 3, label="shaft")
        col.model.geometry.add_box(
            0, 0, 2.5, 0.3, 0.3, 0.5, label="top_region",
        )
        col.model.geometry.add_point(5, 5, 5)   # unlabeled
    return col


# =====================================================================
# Sidecar file tests
# =====================================================================

class TestSidecar:

    def test_sidecar_is_written_next_to_cad(self):
        col = _make_labeled_column()
        try:
            side = sidecar_path(col.file_path)
            assert side.exists()
            payload = read_sidecar(col.file_path)
            assert payload is not None
            assert "anchors" in payload
        finally:
            col.cleanup()

    def test_sidecar_captures_only_named_pgs(self):
        col = _make_labeled_column()
        try:
            payload = read_sidecar(col.file_path)
            names = {a["pg_name"] for a in payload["anchors"]}
            # User-named: "shaft" and "top_region".
            # The unlabeled point should NOT appear.
            assert "shaft" in names
            assert "top_region" in names
            assert len(names) == 2
        finally:
            col.cleanup()

    def test_sidecar_suppressed_by_write_anchors_false(self):
        import tempfile
        from pathlib import Path

        col = Part("no_sidecar")
        with col:
            col.model.geometry.add_box(0, 0, 0, 1, 1, 1, label="cube")
            col.save(
                Path(tempfile.mkdtemp()) / "cube.step",
                write_anchors=False,
            )
        try:
            side = sidecar_path(col.file_path)
            assert not side.exists()
        finally:
            col.cleanup()

    def test_missing_sidecar_is_graceful(self):
        """A Part saved without anchors (or with the sidecar deleted)
        still imports fine — the Instance just gets no pg_names."""
        import tempfile
        from pathlib import Path

        col = Part("bare")
        with col:
            col.model.geometry.add_box(0, 0, 0, 1, 1, 1, label="cube")
            col.save(
                Path(tempfile.mkdtemp()) / "bare.step",
                write_anchors=False,
            )
        try:
            with apeGmsh(model_name="asm") as g:
                inst = g.parts.add(col)
                assert inst.pg_names == []
        finally:
            col.cleanup()


# =====================================================================
# Physical group rebinding tests
# =====================================================================

class TestPGRebinding:

    def test_no_transform(self):
        col = _make_labeled_column()
        try:
            with apeGmsh(model_name="asm") as g:
                inst = g.parts.add(col, label="col_A")
                assert "col_A.shaft" in inst.pg_names
                assert "col_A.top_region" in inst.pg_names
                tags = g.physical.entities("col_A.shaft")
                assert len(tags) >= 1
        finally:
            col.cleanup()

    def test_with_translate(self):
        col = _make_labeled_column()
        try:
            with apeGmsh(model_name="asm") as g:
                inst = g.parts.add(col, translate=(100, 0, 0), label="col_A")
                assert "col_A.shaft" in inst.pg_names
                tags = g.physical.entities("col_A.shaft")
                assert len(tags) >= 1
        finally:
            col.cleanup()

    def test_with_rotate(self):
        col = _make_labeled_column()
        try:
            with apeGmsh(model_name="asm") as g:
                inst = g.parts.add(
                    col,
                    rotate=(math.pi / 2, 0, 0, 1),
                    label="col_A",
                )
                assert "col_A.shaft" in inst.pg_names
        finally:
            col.cleanup()

    def test_with_translate_and_rotate(self):
        col = _make_labeled_column()
        try:
            with apeGmsh(model_name="asm") as g:
                inst = g.parts.add(
                    col,
                    translate=(50, 50, 0),
                    rotate=(math.pi / 4, 0, 0, 1),
                    label="col_A",
                )
                assert "col_A.shaft" in inst.pg_names
                assert "col_A.top_region" in inst.pg_names
        finally:
            col.cleanup()


# =====================================================================
# Multiple instances
# =====================================================================

class TestMultipleInstances:

    def test_two_instances_get_independent_pg_names(self):
        col = _make_labeled_column()
        try:
            with apeGmsh(model_name="asm") as g:
                a = g.parts.add(col, translate=(0, 0, 0), label="col_A")
                b = g.parts.add(col, translate=(6, 0, 0), label="col_B")

                assert "col_A.shaft" in a.pg_names
                assert "col_B.shaft" in b.pg_names

                # PGs are distinct in the assembly
                tags_a = g.physical.entities("col_A.shaft")
                tags_b = g.physical.entities("col_B.shaft")
                assert set(tags_a) != set(tags_b)
        finally:
            col.cleanup()


# =====================================================================
# Auto-PG creation from label= on geometry methods
# =====================================================================

class TestAutoPGFromLabel:

    def test_label_creates_pg_in_part_session(self):
        """When label= is passed inside a Part session, a physical
        group is created automatically."""
        import gmsh

        col = Part("pg_test")
        with col:
            col.model.geometry.add_box(0, 0, 0, 1, 1, 1, label="cube")
            # PG should exist in the Part's live gmsh session
            pgs = gmsh.model.getPhysicalGroups()
            names = [
                gmsh.model.getPhysicalName(d, t) for d, t in pgs
            ]
            assert "cube" in names, f"expected 'cube' in PG names, got {names}"
        col.cleanup()

    def test_no_label_no_pg_in_part_session(self):
        """When label= is not passed, no physical group is created."""
        import gmsh

        col = Part("no_pg")
        with col:
            col.model.geometry.add_box(0, 0, 0, 1, 1, 1)
            pgs = gmsh.model.getPhysicalGroups()
            assert len(pgs) == 0
        col.cleanup()

    def test_label_does_not_create_pg_in_assembly_session(self):
        """In the main apeGmsh session, label= only sets the
        registry label — it does NOT auto-create a physical group."""
        with apeGmsh(model_name="asm") as g:
            g.model.geometry.add_box(0, 0, 0, 1, 1, 1, label="cube")
            import gmsh
            pgs = gmsh.model.getPhysicalGroups()
            # No PGs should have been created
            assert len(pgs) == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
