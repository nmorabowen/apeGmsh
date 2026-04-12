"""
Tests for ``g.model.queries.*`` — remove, topology queries, registry,
and the ``_temporary_tolerance`` context manager.
"""
from __future__ import annotations

import gmsh
import numpy as np
import pandas as pd
import pytest

from apeGmsh.core._model_queries import _temporary_tolerance
from apeGmsh.core.Labels import LABEL_PREFIX


# =====================================================================
# Helpers
# =====================================================================

def _entity_tags(dim: int) -> list[int]:
    """Return all entity tags at *dim* from the live Gmsh model."""
    return [t for _, t in gmsh.model.getEntities(dim)]


def _label_pg_names() -> list[str]:
    """Return bare label names (prefix stripped) from current PGs."""
    names: list[str] = []
    for d, pg_tag in gmsh.model.getPhysicalGroups():
        name = gmsh.model.getPhysicalName(d, pg_tag)
        if name.startswith(LABEL_PREFIX):
            names.append(name[len(LABEL_PREFIX):])
    return names


# =====================================================================
# Remove
# =====================================================================

class TestRemove:
    """Tests for g.model.queries.remove()."""

    def test_remove_entity(self, g):
        """Adding a box then removing it leaves no dim=3 entities."""
        box = g.model.geometry.add_box(0, 0, 0, 1, 1, 1)
        assert len(_entity_tags(3)) == 1

        g.model.queries.remove(box, dim=3)
        assert len(_entity_tags(3)) == 0

    def test_remove_cleans_label_pgs(self, g):
        """Removing a labeled box cleans up the corresponding label PG."""
        box = g.model.geometry.add_box(0, 0, 0, 1, 1, 1, label="my_box")
        assert "my_box" in _label_pg_names()

        g.model.queries.remove(box, dim=3)
        assert "my_box" not in _label_pg_names()


# =====================================================================
# Remove duplicates
# =====================================================================

class TestRemoveDuplicates:
    """Tests for g.model.queries.remove_duplicates()."""

    def test_remove_duplicates_merges(self, g):
        """Two boxes at the same location should merge into one."""
        g.model.geometry.add_box(0, 0, 0, 1, 1, 1)
        g.model.geometry.add_box(0, 0, 0, 1, 1, 1)
        assert len(_entity_tags(3)) == 2

        g.model.queries.remove_duplicates()
        assert len(_entity_tags(3)) == 1

    def test_remove_duplicates_with_tolerance(self, g):
        """Tolerance parameter temporarily overrides Gmsh options."""
        # Store original tolerance
        orig_tol = gmsh.option.getNumber("Geometry.Tolerance")

        # Two boxes that are very slightly offset — default tolerance
        # might not merge them, but a generous tolerance should.
        g.model.geometry.add_box(0, 0, 0, 1, 1, 1)
        g.model.geometry.add_box(0, 0, 0, 1, 1, 1)

        g.model.queries.remove_duplicates(tolerance=1e-2)

        # After the call, original tolerance must be restored.
        restored_tol = gmsh.option.getNumber("Geometry.Tolerance")
        assert restored_tol == pytest.approx(orig_tol)
        assert len(_entity_tags(3)) == 1


# =====================================================================
# Make conformal
# =====================================================================

class TestMakeConformal:
    """Tests for g.model.queries.make_conformal()."""

    def test_make_conformal_fragments(self, g):
        """Two overlapping curves should be fragmented into more pieces."""
        # Create two overlapping line segments that share a middle portion.
        # Line 1: (0,0,0) -> (2,0,0)
        # Line 2: (1,0,0) -> (3,0,0)
        p1 = g.model.geometry.add_point(0, 0, 0)
        p2 = g.model.geometry.add_point(2, 0, 0)
        p3 = g.model.geometry.add_point(1, 0, 0)
        p4 = g.model.geometry.add_point(3, 0, 0)

        g.model.geometry.add_line(p1, p2)
        g.model.geometry.add_line(p3, p4)

        curves_before = len(_entity_tags(1))
        assert curves_before == 2

        g.model.queries.make_conformal(dims=[1])

        # After fragmentation there should be more curve segments
        # because the overlap at (1,0,0)-(2,0,0) gets split.
        curves_after = len(_entity_tags(1))
        assert curves_after > curves_before


# =====================================================================
# Geometry queries
# =====================================================================

class TestBoundingBox:

    def test_bounding_box(self, g):
        """Unit box at origin has bounding box (0,0,0,1,1,1)."""
        box = g.model.geometry.add_box(0, 0, 0, 1, 1, 1)
        bb = g.model.queries.bounding_box(box, dim=3)
        xmin, ymin, zmin, xmax, ymax, zmax = bb

        assert xmin == pytest.approx(0.0, abs=1e-5)
        assert ymin == pytest.approx(0.0, abs=1e-5)
        assert zmin == pytest.approx(0.0, abs=1e-5)
        assert xmax == pytest.approx(1.0, abs=1e-5)
        assert ymax == pytest.approx(1.0, abs=1e-5)
        assert zmax == pytest.approx(1.0, abs=1e-5)


class TestCenterOfMass:

    def test_center_of_mass(self, g):
        """Unit box at origin has centroid at (0.5, 0.5, 0.5)."""
        box = g.model.geometry.add_box(0, 0, 0, 1, 1, 1)
        cx, cy, cz = g.model.queries.center_of_mass(box, dim=3)

        assert cx == pytest.approx(0.5, abs=1e-10)
        assert cy == pytest.approx(0.5, abs=1e-10)
        assert cz == pytest.approx(0.5, abs=1e-10)


class TestMass:

    def test_mass_volume(self, g):
        """1x1x1 box has volume (mass at dim=3) of 1.0."""
        box = g.model.geometry.add_box(0, 0, 0, 1, 1, 1)
        vol = g.model.queries.mass(box, dim=3)
        assert vol == pytest.approx(1.0, abs=1e-10)

    def test_mass_area(self, g):
        """1x1 rectangle has area (mass at dim=2) of 1.0."""
        rect = g.model.geometry.add_rectangle(0, 0, 0, 1, 1)
        area = g.model.queries.mass(rect, dim=2)
        assert area == pytest.approx(1.0, abs=1e-10)


# =====================================================================
# Boundary & adjacencies
# =====================================================================

class TestBoundary:

    def test_boundary_of_volume(self, g):
        """A box has exactly 6 bounding surfaces."""
        box = g.model.geometry.add_box(0, 0, 0, 1, 1, 1)
        faces = g.model.queries.boundary(box, dim=3, combined=False, oriented=False)

        # Each face is a (2, tag) pair
        face_dims = {d for d, _ in faces}
        assert face_dims == {2}
        assert len(faces) == 6


class TestAdjacencies:

    def test_adjacencies(self, g):
        """A surface of a box has downward curve adjacencies."""
        box = g.model.geometry.add_box(0, 0, 0, 1, 1, 1)

        # Get bounding surfaces via raw Gmsh to control the dim
        surfs = [t for _, t in gmsh.model.getEntities(2)]
        assert len(surfs) == 6
        face_tag = surfs[0]

        up, down = gmsh.model.getAdjacencies(2, face_tag)

        # Each face of a box is bounded by 4 curves
        assert len(down) == 4


# =====================================================================
# Entities in bounding box
# =====================================================================

class TestEntitiesInBoundingBox:

    def test_entities_in_bounding_box(self, g):
        """A box at origin is found inside a slightly padded query region."""
        box = g.model.geometry.add_box(0, 0, 0, 1, 1, 1)

        found = g.model.queries.entities_in_bounding_box(
            -0.1, -0.1, -0.1, 1.1, 1.1, 1.1, dim=3,
        )
        found_tags = [t for _, t in found]
        assert box in found_tags


# =====================================================================
# Registry
# =====================================================================

class TestRegistry:

    def test_registry_dataframe(self, g):
        """Registry returns a DataFrame with the correct columns and rows."""
        g.model.geometry.add_box(0, 0, 0, 1, 1, 1, label="alpha")
        g.model.geometry.add_box(5, 0, 0, 1, 1, 1, label="beta")

        df = g.model.queries.registry()

        assert isinstance(df, pd.DataFrame)
        # The index is (dim, tag); columns include 'kind' and 'label'
        assert 'kind' in df.columns
        assert 'label' in df.columns

        # We created two boxes (dim=3), so there should be at least
        # two rows at dim=3.  Other dimensions (surfaces, curves,
        # points) are also registered.
        dim3 = df.xs(3, level='dim')
        assert len(dim3) >= 2

        # Labels should appear in the DataFrame
        labels = set(df['label'].values)
        assert 'alpha' in labels
        assert 'beta' in labels


# =====================================================================
# _temporary_tolerance context manager
# =====================================================================

class TestTemporaryTolerance:

    def test_temporary_tolerance(self, g):
        """Options are overridden inside the context and restored after."""
        key = "Geometry.Tolerance"
        original = gmsh.option.getNumber(key)

        new_val = 0.12345
        with _temporary_tolerance(new_val, keys=(key,)):
            inside = gmsh.option.getNumber(key)
            assert inside == pytest.approx(new_val)

        restored = gmsh.option.getNumber(key)
        assert restored == pytest.approx(original)

    def test_temporary_tolerance_none_is_noop(self, g):
        """Passing tolerance=None should not change any option."""
        key = "Geometry.Tolerance"
        original = gmsh.option.getNumber(key)

        with _temporary_tolerance(None, keys=(key,)):
            inside = gmsh.option.getNumber(key)
            assert inside == pytest.approx(original)

        restored = gmsh.option.getNumber(key)
        assert restored == pytest.approx(original)
