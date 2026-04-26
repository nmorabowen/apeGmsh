"""
Regression tests — Part two-tier naming: labels + physical groups.

Tests the full pipeline:

1. ``label=`` on geometry methods auto-creates a label PG (Tier 1,
   prefixed with ``_label:``) — NOT a user-facing PG.
2. Labels travel through STEP round-trip via sidecar + COM matching.
3. ``g.labels.entities("col_A.shaft")`` resolves to entity tags.
4. ``g.physical.get_all()`` does NOT include labels.
5. ``g.labels.promote_to_physical("name")`` creates a real PG.
6. Transform cases: no transform, translate, rotate, both.
7. Multiple instances get independent label names.
8. Sidecar suppression and missing-sidecar fallback.
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

    def test_sidecar_captures_only_labeled_entities(self):
        col = _make_labeled_column()
        try:
            payload = read_sidecar(col.file_path)
            names = {a["pg_name"] for a in payload["anchors"]}
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
                # Umbrella label is always created; no sidecar means
                # no dotted sub-labels were rebound.
                assert inst.label_names == [inst.label]
                assert not any('.' in n for n in inst.label_names)
        finally:
            col.cleanup()


# =====================================================================
# Label rebinding via sidecar
# =====================================================================

class TestLabelRebinding:

    def test_no_transform(self):
        col = _make_labeled_column()
        try:
            with apeGmsh(model_name="asm") as g:
                inst = g.parts.add(col, label="col_A")
                assert "col_A.shaft" in inst.label_names
                assert "col_A.top_region" in inst.label_names
                tags = g.labels.entities("col_A.shaft")
                assert len(tags) >= 1
        finally:
            col.cleanup()

    def test_with_translate(self):
        col = _make_labeled_column()
        try:
            with apeGmsh(model_name="asm") as g:
                inst = g.parts.add(col, translate=(100, 0, 0), label="col_A")
                assert "col_A.shaft" in inst.label_names
                tags = g.labels.entities("col_A.shaft")
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
                assert "col_A.shaft" in inst.label_names
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
                assert "col_A.shaft" in inst.label_names
                assert "col_A.top_region" in inst.label_names
        finally:
            col.cleanup()


# =====================================================================
# Multiple instances
# =====================================================================

class TestMultipleInstances:

    def test_two_instances_get_independent_label_names(self):
        col = _make_labeled_column()
        try:
            with apeGmsh(model_name="asm") as g:
                a = g.parts.add(col, translate=(0, 0, 0), label="col_A")
                b = g.parts.add(col, translate=(6, 0, 0), label="col_B")

                assert "col_A.shaft" in a.label_names
                assert "col_B.shaft" in b.label_names

                tags_a = g.labels.entities("col_A.shaft")
                tags_b = g.labels.entities("col_B.shaft")
                assert set(tags_a) != set(tags_b)
        finally:
            col.cleanup()


# =====================================================================
# Two-tier separation
# =====================================================================

class TestTwoTierSeparation:

    def test_labels_invisible_to_physical(self):
        """Labels (Tier 1) must not appear in g.physical.get_all()."""
        with apeGmsh(model_name="asm") as g:
            g.model.geometry.add_box(0, 0, 0, 1, 1, 1, label="cube")
            user_pgs = g.physical.get_all()
            assert len(user_pgs) == 0

    def test_promote_creates_user_pg(self):
        """promote_to_physical copies label entities to a real PG."""
        with apeGmsh(model_name="asm") as g:
            g.model.geometry.add_box(0, 0, 0, 1, 1, 1, label="cube")
            assert len(g.physical.get_all()) == 0

            g.labels.promote_to_physical("cube", pg_name="my_cube")
            user_pgs = g.physical.get_all()
            assert len(user_pgs) == 1

    def test_label_created_in_part_session(self):
        """label= inside a Part creates a label PG visible via
        g.labels but NOT via g.physical."""
        import gmsh

        col = Part("pg_test")
        with col:
            col.model.geometry.add_box(0, 0, 0, 1, 1, 1, label="cube")
            # Label PG should exist
            assert col.labels.has("cube")
            # But user PGs should be empty
            user_pgs = col.physical.get_all()
            assert len(user_pgs) == 0
        col.cleanup()

    def test_no_label_no_pg(self):
        """Unlabeled geometry does not create any PGs at all."""
        import gmsh

        col = Part("no_label")
        with col:
            col.model.geometry.add_box(0, 0, 0, 1, 1, 1)
            pgs = gmsh.model.getPhysicalGroups()
            assert len(pgs) == 0
        col.cleanup()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
