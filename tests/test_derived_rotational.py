"""
Derived rotational inertia — about-node parallel-axis form.

`derive_rotational=True` on a Volume/Surface mass def computes per-node
rotational inertia from the element shape functions:

    I_xx^(I) = ρ ∫ N_I (y²+z²) dΩ − M_I (y_I²+z_I²)        (cyclic)

The governing physical invariant is **kinetic-energy consistency**:
the *assembled* rotational KE under a rigid rotation about any global
axis equals the continuum value,

    Σ_I [ M_I r⊥,I² + I_⊥⊥^(I) ]  ==  ρ ∫_Ω r⊥² dΩ

Individual per-node I values may be negative — they are parallel-axis
corrections, not standalone inertias. Tests assert the assembled
identity, never per-node positivity.
"""
from __future__ import annotations

import unittest

import numpy as np

from apeGmsh.core.MassesComposite import _validate_derive_rotational
from apeGmsh.core.masses.defs import SurfaceMassDef, VolumeMassDef
from apeGmsh._kernel.resolvers._mass_resolver import MassResolver


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_CUBE = np.array([
    [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
    [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1],
], dtype=float)

_TET = np.array(
    [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=float,
)

_QUAD = np.array(
    [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]], dtype=float,
)


def _assembled_rot_ke(recs, coords_by_id):
    """(Tx, Ty, Tz) assembled rotational-KE coefficients about the
    three global axes through the origin.
    """
    Tx = Ty = Tz = 0.0
    for r in recs:
        x, y, z = coords_by_id[r.node_id]
        mx, my, mz, ixx, iyy, izz = r.mass
        Tx += mx * (y * y + z * z) + ixx
        Ty += my * (x * x + z * z) + iyy
        Tz += mz * (x * x + y * y) + izz
    return Tx, Ty, Tz


# =====================================================================
# Kinetic-energy consistency — the core invariant
# =====================================================================

class TestKineticEnergyConsistency(unittest.TestCase):

    def test_hex8_unit_cube_all_axes(self):
        tags = np.arange(1, 9, dtype=np.int64)
        r = MassResolver(tags, _CUBE)
        recs = r.resolve_volume_consistent(
            VolumeMassDef(target="c", density=1.0, derive_rotational=True),
            [tags],
        )
        cmap = {i + 1: _CUBE[i] for i in range(8)}
        Tx, Ty, Tz = _assembled_rot_ke(recs, cmap)
        # ∫(y²+z²) over [0,1]³ = 2/3, same for all axes by symmetry
        for T in (Tx, Ty, Tz):
            self.assertAlmostEqual(T, 2.0 / 3.0, places=9)

    def test_offset_cube_parallel_axis_correct(self):
        """Element far from the origin: the parallel-axis subtraction
        must keep the assembled KE equal to the continuum value.
        """
        shift = np.array([10.0, 10.0, 10.0])
        coords = _CUBE + shift
        tags = np.arange(1, 9, dtype=np.int64)
        r = MassResolver(tags, coords)
        recs = r.resolve_volume_consistent(
            VolumeMassDef(target="c", density=1.0, derive_rotational=True),
            [tags],
        )
        cmap = {i + 1: coords[i] for i in range(8)}
        Tx, _, _ = _assembled_rot_ke(recs, cmap)
        # ∫_{10}^{11}(y²+z²) over the unit cube = 2·(11³−10³)/3
        continuum = 2.0 * (11.0**3 - 10.0**3) / 3.0
        self.assertAlmostEqual(Tx, continuum, places=6)

    def test_tet4(self):
        tags = np.array([1, 2, 3, 4], dtype=np.int64)
        r = MassResolver(tags, _TET)
        recs = r.resolve_volume_consistent(
            VolumeMassDef(target="t", density=1.0, derive_rotational=True),
            [tags],
        )
        cmap = {i + 1: _TET[i] for i in range(4)}
        Tx, _, _ = _assembled_rot_ke(recs, cmap)
        # ∫(y²+z²) over the unit ref tet = 2·(1/60) = 1/30
        self.assertAlmostEqual(Tx, 1.0 / 30.0, places=9)

    def test_total_translational_mass_conserved(self):
        tags = np.arange(1, 9, dtype=np.int64)
        r = MassResolver(tags, _CUBE)
        recs = r.resolve_volume_consistent(
            VolumeMassDef(target="c", density=7.0, derive_rotational=True),
            [tags],
        )
        self.assertAlmostEqual(
            sum(x.mass[0] for x in recs), 7.0 * 1.0, places=9,
        )


# =====================================================================
# Per-node values are signed parallel-axis corrections
# =====================================================================

class TestPerNodeValuesAreCorrections(unittest.TestCase):

    def test_some_nodes_negative(self):
        tags = np.arange(1, 9, dtype=np.int64)
        r = MassResolver(tags, _CUBE)
        recs = r.resolve_volume_consistent(
            VolumeMassDef(target="c", density=1.0, derive_rotational=True),
            [tags],
        )
        ixx = [x.mass[3] for x in recs]
        self.assertTrue(min(ixx) < 0.0)   # far corners negative
        self.assertTrue(max(ixx) > 0.0)   # near-origin positive


# =====================================================================
# Surface variant
# =====================================================================

class TestSurfaceDerivedRotational(unittest.TestCase):

    def test_unit_plate_izz(self):
        tags = np.array([1, 2, 3, 4], dtype=np.int64)
        r = MassResolver(tags, _QUAD)
        recs = r.resolve_surface_consistent(
            SurfaceMassDef(
                target="p", areal_density=1.0, derive_rotational=True,
            ),
            [[1, 2, 3, 4]],
        )
        cmap = {i + 1: _QUAD[i] for i in range(4)}
        _, _, Tz = _assembled_rot_ke(recs, cmap)
        # ∫(x²+y²) over the unit square = 2/3
        self.assertAlmostEqual(Tz, 2.0 / 3.0, places=9)


# =====================================================================
# Composition with dofs=
# =====================================================================

class TestComposesWithDofs(unittest.TestCase):

    def test_dofs_masks_translational_not_rotational(self):
        tags = np.arange(1, 9, dtype=np.int64)
        r = MassResolver(tags, _CUBE)
        recs = r.resolve_volume_consistent(
            VolumeMassDef(
                target="c", density=1.0,
                derive_rotational=True, dofs=[1, 2],
            ),
            [tags],
        )
        for x in recs:
            self.assertAlmostEqual(x.mass[2], 0.0)   # z translational masked
        # rotational still present (non-trivial)
        self.assertTrue(any(abs(x.mass[3]) > 1e-9 for x in recs))


# =====================================================================
# Validator
# =====================================================================

class TestValidator(unittest.TestCase):

    def test_default_false(self):
        self.assertFalse(_validate_derive_rotational(
            False, rotational=None, reduction="lumped",
        ))

    def test_ok_on_consistent(self):
        self.assertTrue(_validate_derive_rotational(
            True, rotational=None, reduction="consistent",
        ))

    def test_raises_with_explicit_rotational(self):
        with self.assertRaises(ValueError) as ctx:
            _validate_derive_rotational(
                True, rotational=(1, 2, 3), reduction="consistent",
            )
        self.assertIn("mutually exclusive", str(ctx.exception))

    def test_raises_on_lumped(self):
        with self.assertRaises(ValueError) as ctx:
            _validate_derive_rotational(
                True, rotational=None, reduction="lumped",
            )
        self.assertIn("consistent", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
