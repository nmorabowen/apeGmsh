"""
Unit tests for translational DOF masking (``dofs=``) and fixed rotational
inertia (``rotational=``) on all four MassDef kinds.

Tests at the resolver level (synthetic single-element meshes) plus
validator tests at the composite level.
"""
from __future__ import annotations

import unittest

import numpy as np

from apeGmsh.core.MassesComposite import (
    _validate_rotational,
    _validate_translational_dofs,
)
from apeGmsh.core.masses.defs import (
    LineMassDef,
    PointMassDef,
    SurfaceMassDef,
    VolumeMassDef,
)
from apeGmsh._kernel.resolvers._mass_resolver import MassResolver


# ---------------------------------------------------------------------------
# Resolver fixtures — minimum mesh per def kind
# ---------------------------------------------------------------------------

def _point_resolver():
    """Two nodes at distinct coords; node_set targets both."""
    tags = np.array([1, 2], dtype=np.int64)
    coords = np.array([[0, 0, 0], [1, 0, 0]], dtype=float)
    return MassResolver(tags, coords), {1, 2}


def _line_resolver():
    """Single 2-node line, length 4."""
    tags = np.array([1, 2], dtype=np.int64)
    coords = np.array([[0, 0, 0], [4, 0, 0]], dtype=float)
    return MassResolver(tags, coords), [(1, 2)]


def _surface_resolver():
    """Single unit quad (area = 1)."""
    tags = np.array([1, 2, 3, 4], dtype=np.int64)
    coords = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]], dtype=float)
    return MassResolver(tags, coords), [[1, 2, 3, 4]]


def _volume_resolver():
    """Single unit cube (volume = 1)."""
    tags = np.arange(1, 9, dtype=np.int64)
    coords = np.array([
        [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
        [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1],
    ], dtype=float)
    return MassResolver(tags, coords), [np.arange(1, 9, dtype=np.int64)]


def _first_vec(records):
    """Return the length-6 mass vector of the first record."""
    return np.array(records[0].mass)


# =====================================================================
# Defaults — regression: existing callers must see no change
# =====================================================================

class TestDefaultsPreserveCurrentBehavior(unittest.TestCase):

    def test_point_default(self):
        r, ns = _point_resolver()
        defn = PointMassDef(target="pt", mass=3.0)
        vec = _first_vec(r.resolve_point_lumped(defn, ns))
        np.testing.assert_allclose(vec, [3.0, 3.0, 3.0, 0, 0, 0])

    def test_line_default(self):
        r, edges = _line_resolver()
        defn = LineMassDef(target="edge", linear_density=10.0)
        vec = _first_vec(r.resolve_line_lumped(defn, edges))
        # half-share = 0.5 * 10 * 4 = 20
        np.testing.assert_allclose(vec, [20.0, 20.0, 20.0, 0, 0, 0])

    def test_surface_default(self):
        r, faces = _surface_resolver()
        defn = SurfaceMassDef(target="face", areal_density=8.0)
        vec = _first_vec(r.resolve_surface_lumped(defn, faces))
        # share = 8 * 1 / 4 = 2
        np.testing.assert_allclose(vec, [2.0, 2.0, 2.0, 0, 0, 0])

    def test_volume_default(self):
        r, elems = _volume_resolver()
        defn = VolumeMassDef(target="body", density=16.0)
        vec = _first_vec(r.resolve_volume_lumped(defn, elems))
        # share = 16 * 1 / 8 = 2
        np.testing.assert_allclose(vec, [2.0, 2.0, 2.0, 0, 0, 0])


# =====================================================================
# dofs= masks translational positions
# =====================================================================

class TestDofsMask(unittest.TestCase):

    def test_volume_dofs_xy_only(self):
        r, elems = _volume_resolver()
        defn = VolumeMassDef(target="body", density=16.0, dofs=[1, 2])
        vec = _first_vec(r.resolve_volume_lumped(defn, elems))
        np.testing.assert_allclose(vec, [2.0, 2.0, 0.0, 0, 0, 0])

    def test_line_dofs_z_only(self):
        r, edges = _line_resolver()
        defn = LineMassDef(target="edge", linear_density=10.0, dofs=[3])
        vec = _first_vec(r.resolve_line_lumped(defn, edges))
        np.testing.assert_allclose(vec, [0, 0, 20.0, 0, 0, 0])

    def test_surface_dofs_xy_only(self):
        r, faces = _surface_resolver()
        defn = SurfaceMassDef(target="face", areal_density=8.0, dofs=[1, 2])
        vec = _first_vec(r.resolve_surface_lumped(defn, faces))
        np.testing.assert_allclose(vec, [2.0, 2.0, 0.0, 0, 0, 0])

    def test_point_dofs_x_only(self):
        r, ns = _point_resolver()
        defn = PointMassDef(target="pt", mass=3.0, dofs=[1])
        vec = _first_vec(r.resolve_point_lumped(defn, ns))
        np.testing.assert_allclose(vec, [3.0, 0, 0, 0, 0, 0])


# =====================================================================
# rotational= populates positions 4, 5, 6
# =====================================================================

class TestRotational(unittest.TestCase):

    def test_volume_rotational_only(self):
        r, elems = _volume_resolver()
        defn = VolumeMassDef(
            target="body", density=16.0, rotational=(0.1, 0.2, 0.3),
        )
        vec = _first_vec(r.resolve_volume_lumped(defn, elems))
        np.testing.assert_allclose(vec, [2.0, 2.0, 2.0, 0.1, 0.2, 0.3])

    def test_line_rotational(self):
        r, edges = _line_resolver()
        defn = LineMassDef(
            target="edge", linear_density=10.0, rotational=(1.0, 2.0, 3.0),
        )
        vec = _first_vec(r.resolve_line_lumped(defn, edges))
        np.testing.assert_allclose(vec, [20.0, 20.0, 20.0, 1.0, 2.0, 3.0])

    def test_surface_rotational(self):
        r, faces = _surface_resolver()
        defn = SurfaceMassDef(
            target="face", areal_density=8.0, rotational=(0.5, 0.5, 0.5),
        )
        vec = _first_vec(r.resolve_surface_lumped(defn, faces))
        np.testing.assert_allclose(vec, [2.0, 2.0, 2.0, 0.5, 0.5, 0.5])


# =====================================================================
# Composition: dofs= + rotational=
# =====================================================================

class TestDofsAndRotationalCompose(unittest.TestCase):

    def test_volume_dofs_xy_with_rotational(self):
        r, elems = _volume_resolver()
        defn = VolumeMassDef(
            target="body", density=16.0,
            dofs=[1, 2], rotational=(0.1, 0.2, 0.3),
        )
        vec = _first_vec(r.resolve_volume_lumped(defn, elems))
        np.testing.assert_allclose(vec, [2.0, 2.0, 0.0, 0.1, 0.2, 0.3])


# =====================================================================
# Validators — reject bad input at declaration time
# =====================================================================

class TestValidators(unittest.TestCase):

    def test_dofs_none_is_passthrough(self):
        self.assertIsNone(_validate_translational_dofs(None))

    def test_dofs_subset_of_123_ok(self):
        self.assertEqual(_validate_translational_dofs([1, 2]), [1, 2])
        self.assertEqual(_validate_translational_dofs([3]), [3])

    def test_dofs_rejects_rotational_indices(self):
        with self.assertRaises(ValueError) as ctx:
            _validate_translational_dofs([4])
        self.assertIn("rotational=", str(ctx.exception))

    def test_dofs_rejects_zero_and_negative(self):
        with self.assertRaises(ValueError):
            _validate_translational_dofs([0])
        with self.assertRaises(ValueError):
            _validate_translational_dofs([-1])

    def test_dofs_rejects_empty(self):
        with self.assertRaises(ValueError) as ctx:
            _validate_translational_dofs([])
        self.assertIn("non-empty", str(ctx.exception))

    def test_rotational_none_is_passthrough(self):
        self.assertIsNone(_validate_rotational(None))

    def test_rotational_length_3_ok(self):
        self.assertEqual(
            _validate_rotational((1, 2, 3)), (1.0, 2.0, 3.0),
        )

    def test_rotational_wrong_length_rejected(self):
        with self.assertRaises(ValueError):
            _validate_rotational((1, 2))
        with self.assertRaises(ValueError):
            _validate_rotational((1, 2, 3, 4))


if __name__ == "__main__":
    unittest.main()
