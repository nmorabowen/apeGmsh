"""
Physical-correctness tests for the mass-distribution resolver.

These tests verify the *mathematical* invariants that any mass-lumping
scheme must satisfy.  They sit on top of the API-behavior tests in
``test_mass_dofs_rotational.py`` — those check that bytes flow through
the kwargs correctly; these check that the resulting mass distribution
matches the continuum it discretises.

Mass moments hierarchy
----------------------
For a continuous mass distribution ρ(x) over region Ω, define

    M_0 = ∫_Ω ρ dΩ                            (total mass)
    M_1 = ∫_Ω ρ x dΩ                          (first moment → CoM × M_0)
    I   = ∫_Ω ρ (|r|² 𝟙 - r⊗r) dΩ            (inertia tensor)

For a discrete set {m_I, x_I} the lumped moments are

    M_0^L = Σ_I m_I,   M_1^L = Σ_I m_I x_I,   I_ij^L = Σ_I m_I (r_k r_k δ_ij - r_i r_j)

Equal-share lumping (m_I = ρV_e/n per node) preserves:
  * M_0 exactly, always.
  * M_1 exactly for simplices (tet4, tri3, line2) and parallelepipeds
    (axis-aligned cuboid hex8, axis-aligned rectangular quad4) — these
    are the fixtures we use.
  * I — NOT preserved in general; that's an inherent approximation of
    lumping, not a bug.  We do not test I against the continuum value.
"""
from __future__ import annotations

import unittest

import numpy as np

from apeGmsh.core.masses.defs import (
    LineMassDef,
    PointMassDef,
    SurfaceMassDef,
    VolumeMassDef,
)
from apeGmsh._kernel.resolvers._mass_resolver import MassResolver


# ---------------------------------------------------------------------------
# Mesh fixtures — known-volume, known-centroid single elements
# ---------------------------------------------------------------------------

def _resolver(coords_by_tag):
    tags = np.array(sorted(coords_by_tag), dtype=np.int64)
    coords = np.array([coords_by_tag[int(t)] for t in tags], dtype=float)
    return MassResolver(tags, coords)


def _unit_cube_hex8():
    """Unit cube [0,1]³ as a single hex8.  V = 1, centroid = (½, ½, ½)."""
    coords = {
        1: (0, 0, 0), 2: (1, 0, 0), 3: (1, 1, 0), 4: (0, 1, 0),
        5: (0, 0, 1), 6: (1, 0, 1), 7: (1, 1, 1), 8: (0, 1, 1),
    }
    return _resolver(coords), [np.arange(1, 9, dtype=np.int64)], coords


def _unit_tet4():
    """Tet4 with vertices at origin + 3 unit basis vectors.

    V = 1/6, centroid = (¼, ¼, ¼).
    """
    coords = {1: (0, 0, 0), 2: (1, 0, 0), 3: (0, 1, 0), 4: (0, 0, 1)}
    return _resolver(coords), [np.array([1, 2, 3, 4], dtype=np.int64)], coords


def _unit_quad4():
    """Unit square as a single quad4.  A = 1, centroid = (½, ½, 0)."""
    coords = {1: (0, 0, 0), 2: (1, 0, 0), 3: (1, 1, 0), 4: (0, 1, 0)}
    return _resolver(coords), [[1, 2, 3, 4]], coords


def _unit_tri3():
    """Tri3 (0,0)-(1,0)-(0,1).  A = ½, centroid = (⅓, ⅓, 0)."""
    coords = {1: (0, 0, 0), 2: (1, 0, 0), 3: (0, 1, 0)}
    return _resolver(coords), [[1, 2, 3]], coords


def _two_node_line(L=4.0):
    """Line from origin to (L, 0, 0).  Length L, midpoint (L/2, 0, 0)."""
    coords = {1: (0, 0, 0), 2: (L, 0, 0)}
    return _resolver(coords), [(1, 2)], coords


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _by_node(records):
    return {rec.node_id: np.array(rec.mass) for rec in records}


def _total(records, component):
    """Σ_I m_I[component] — the zeroth moment on one channel."""
    return sum(rec.mass[component] for rec in records)


def _com(records, coords_by_tag, component=0):
    """Σ_I m_I x_I / Σ_I m_I — center of mass on one translational channel."""
    total = _total(records, component)
    if total == 0:
        return np.zeros(3)
    weighted = np.zeros(3)
    for rec in records:
        weighted += rec.mass[component] * np.array(coords_by_tag[rec.node_id])
    return weighted / total


# =====================================================================
# Group A — total mass conservation (zeroth moment)
# =====================================================================

class TestTotalMassConservation(unittest.TestCase):
    """M_0^L = ∫_Ω ρ dΩ — must hold exactly for every element type."""

    def test_volume_hex8_unit_cube(self):
        r, conn, _ = _unit_cube_hex8()
        rho = 7850.0
        recs = r.resolve_volume_lumped(
            VolumeMassDef(target="cube", density=rho), conn,
        )
        # V = 1 → total mass = ρV = ρ on each translational DOF
        for c in (0, 1, 2):
            self.assertAlmostEqual(_total(recs, c), rho)

    def test_volume_tet4(self):
        r, conn, _ = _unit_tet4()
        rho = 2400.0
        recs = r.resolve_volume_lumped(
            VolumeMassDef(target="tet", density=rho), conn,
        )
        # V = 1/6
        self.assertAlmostEqual(_total(recs, 0), rho / 6.0)

    def test_surface_quad4(self):
        r, face, _ = _unit_quad4()
        rho_a = 100.0
        recs = r.resolve_surface_lumped(
            SurfaceMassDef(target="face", areal_density=rho_a), face,
        )
        self.assertAlmostEqual(_total(recs, 0), rho_a)  # A = 1

    def test_surface_tri3(self):
        r, face, _ = _unit_tri3()
        rho_a = 50.0
        recs = r.resolve_surface_lumped(
            SurfaceMassDef(target="tri", areal_density=rho_a), face,
        )
        self.assertAlmostEqual(_total(recs, 0), rho_a * 0.5)  # A = 1/2

    def test_line_2node(self):
        r, edges, _ = _two_node_line(L=4.0)
        rho_l = 10.0
        recs = r.resolve_line_lumped(
            LineMassDef(target="edge", linear_density=rho_l), edges,
        )
        self.assertAlmostEqual(_total(recs, 0), rho_l * 4.0)


# =====================================================================
# Group B — first-moment conservation (center of mass)
# =====================================================================

class TestCenterOfMassPreservation(unittest.TestCase):
    """M_1^L / M_0^L = geometric centroid for simplices and parallelepipeds."""

    def test_hex8_unit_cube(self):
        r, conn, coords = _unit_cube_hex8()
        recs = r.resolve_volume_lumped(
            VolumeMassDef(target="cube", density=1.0), conn,
        )
        np.testing.assert_allclose(_com(recs, coords), [0.5, 0.5, 0.5])

    def test_tet4(self):
        r, conn, coords = _unit_tet4()
        recs = r.resolve_volume_lumped(
            VolumeMassDef(target="tet", density=1.0), conn,
        )
        np.testing.assert_allclose(_com(recs, coords), [0.25, 0.25, 0.25])

    def test_quad4_unit(self):
        r, face, coords = _unit_quad4()
        recs = r.resolve_surface_lumped(
            SurfaceMassDef(target="face", areal_density=1.0), face,
        )
        np.testing.assert_allclose(_com(recs, coords), [0.5, 0.5, 0.0])

    def test_tri3(self):
        r, face, coords = _unit_tri3()
        recs = r.resolve_surface_lumped(
            SurfaceMassDef(target="tri", areal_density=1.0), face,
        )
        np.testing.assert_allclose(_com(recs, coords), [1/3, 1/3, 0.0])

    def test_line_midpoint(self):
        r, edges, coords = _two_node_line(L=6.0)
        recs = r.resolve_line_lumped(
            LineMassDef(target="edge", linear_density=2.0), edges,
        )
        np.testing.assert_allclose(_com(recs, coords), [3.0, 0.0, 0.0])


# =====================================================================
# Group C — multi-element accumulation across shared nodes
# =====================================================================

class TestMultiElementAccumulation(unittest.TestCase):
    """Two elements sharing nodes accumulate per-node; total mass adds."""

    def test_two_tets_sharing_face(self):
        # Tet 1: vertices (0,0,0), (1,0,0), (0,1,0), (0,0,1) → V = 1/6
        # Tet 2: vertices (1,1,1), (1,0,0), (0,1,0), (0,0,1) → V = 1/3
        # Shared face: nodes (2, 3, 4)
        coords = {
            1: (0, 0, 0),
            2: (1, 0, 0),
            3: (0, 1, 0),
            4: (0, 0, 1),
            5: (1, 1, 1),
        }
        r = _resolver(coords)
        conn = [
            np.array([1, 2, 3, 4], dtype=np.int64),
            np.array([5, 2, 3, 4], dtype=np.int64),
        ]
        rho = 100.0
        recs = r.resolve_volume_lumped(
            VolumeMassDef(target="bod", density=rho), conn,
        )
        # Total mass = ρ × (V1 + V2) = 100 × (1/6 + 1/3) = 100/2 = 50
        self.assertAlmostEqual(_total(recs, 0), rho * 0.5)
        # Node 1 sees only tet1: ρV1/4 = 100/24
        # Node 5 sees only tet2: ρV2/4 = 100/12 = 200/24
        # Shared nodes 2,3,4 see both: 100/24 + 100/12 = 300/24 = 12.5
        m = _by_node(recs)
        self.assertAlmostEqual(m[1][0], 100/24)
        self.assertAlmostEqual(m[5][0], 100/12)
        for nid in (2, 3, 4):
            self.assertAlmostEqual(m[nid][0], 100/24 + 100/12)


# =====================================================================
# Group D — linearity and superposition
# =====================================================================

class TestLinearityAndSuperposition(unittest.TestCase):

    def test_linear_in_density(self):
        r, conn, _ = _unit_cube_hex8()
        recs1 = _by_node(r.resolve_volume_lumped(
            VolumeMassDef(target="cube", density=10.0), conn))
        recs2 = _by_node(r.resolve_volume_lumped(
            VolumeMassDef(target="cube", density=20.0), conn))
        for nid in recs1:
            np.testing.assert_allclose(recs2[nid][:3], 2.0 * recs1[nid][:3])

    def test_two_defs_same_region_accumulate(self):
        """m_I(ρ_1) + m_I(ρ_2) = m_I(ρ_1 + ρ_2)."""
        r, conn, _ = _unit_cube_hex8()
        r1 = _by_node(r.resolve_volume_lumped(
            VolumeMassDef(target="cube", density=3.0), conn))
        r2 = _by_node(r.resolve_volume_lumped(
            VolumeMassDef(target="cube", density=5.0), conn))
        rs = _by_node(r.resolve_volume_lumped(
            VolumeMassDef(target="cube", density=8.0), conn))
        for nid in r1:
            np.testing.assert_allclose(rs[nid][:3], r1[nid][:3] + r2[nid][:3])


# =====================================================================
# Group E — translation invariance
# =====================================================================

class TestTranslationInvariance(unittest.TestCase):

    def test_mass_values_unchanged(self):
        coords_a = {1: (0, 0, 0), 2: (1, 0, 0), 3: (1, 1, 0), 4: (0, 1, 0),
                    5: (0, 0, 1), 6: (1, 0, 1), 7: (1, 1, 1), 8: (0, 1, 1)}
        shift = np.array([10.0, 20.0, 30.0])
        coords_b = {nid: tuple(np.array(xyz) + shift)
                    for nid, xyz in coords_a.items()}
        conn = [np.arange(1, 9, dtype=np.int64)]
        defn = VolumeMassDef(target="cube", density=1000.0)
        ra = _by_node(_resolver(coords_a).resolve_volume_lumped(defn, conn))
        rb = _by_node(_resolver(coords_b).resolve_volume_lumped(defn, conn))
        for nid in ra:
            np.testing.assert_allclose(ra[nid], rb[nid])

    def test_com_translates_by_shift(self):
        coords_a = {1: (0, 0, 0), 2: (1, 0, 0), 3: (1, 1, 0), 4: (0, 1, 0),
                    5: (0, 0, 1), 6: (1, 0, 1), 7: (1, 1, 1), 8: (0, 1, 1)}
        shift = np.array([7.0, -3.0, 11.0])
        coords_b = {nid: tuple(np.array(xyz) + shift)
                    for nid, xyz in coords_a.items()}
        conn = [np.arange(1, 9, dtype=np.int64)]
        defn = VolumeMassDef(target="cube", density=1.0)
        ra = _resolver(coords_a).resolve_volume_lumped(defn, conn)
        rb = _resolver(coords_b).resolve_volume_lumped(defn, conn)
        np.testing.assert_allclose(_com(rb, coords_b) - _com(ra, coords_a), shift)


# =====================================================================
# Group F — affine scaling
# =====================================================================

class TestAffineScaling(unittest.TestCase):
    """Lumped mass scales with the geometric measure of the element."""

    def test_uniform_scale_volume_cubes(self):
        lam = 3.0
        coords_a = {1: (0, 0, 0), 2: (1, 0, 0), 3: (1, 1, 0), 4: (0, 1, 0),
                    5: (0, 0, 1), 6: (1, 0, 1), 7: (1, 1, 1), 8: (0, 1, 1)}
        coords_b = {nid: tuple(lam * c for c in xyz)
                    for nid, xyz in coords_a.items()}
        conn = [np.arange(1, 9, dtype=np.int64)]
        defn = VolumeMassDef(target="cube", density=1.0)
        ra = _by_node(_resolver(coords_a).resolve_volume_lumped(defn, conn))
        rb = _by_node(_resolver(coords_b).resolve_volume_lumped(defn, conn))
        for nid in ra:
            np.testing.assert_allclose(rb[nid][:3], lam**3 * ra[nid][:3])

    def test_uniform_scale_surface(self):
        lam = 2.5
        coords_a = {1: (0, 0, 0), 2: (1, 0, 0), 3: (1, 1, 0), 4: (0, 1, 0)}
        coords_b = {nid: tuple(lam * c for c in xyz)
                    for nid, xyz in coords_a.items()}
        face = [[1, 2, 3, 4]]
        defn = SurfaceMassDef(target="face", areal_density=1.0)
        ra = _by_node(_resolver(coords_a).resolve_surface_lumped(defn, face))
        rb = _by_node(_resolver(coords_b).resolve_surface_lumped(defn, face))
        for nid in ra:
            np.testing.assert_allclose(rb[nid][:3], lam**2 * ra[nid][:3])

    def test_uniform_scale_line(self):
        lam = 4.0
        coords_a = {1: (0, 0, 0), 2: (1, 0, 0)}
        coords_b = {nid: tuple(lam * c for c in xyz)
                    for nid, xyz in coords_a.items()}
        edges = [(1, 2)]
        defn = LineMassDef(target="edge", linear_density=1.0)
        ra = _by_node(_resolver(coords_a).resolve_line_lumped(defn, edges))
        rb = _by_node(_resolver(coords_b).resolve_line_lumped(defn, edges))
        for nid in ra:
            np.testing.assert_allclose(rb[nid][:3], lam * ra[nid][:3])


# =====================================================================
# Group G — dofs= physical invariants
# =====================================================================

class TestDofsPhysicalInvariants(unittest.TestCase):
    """dofs= projects the same lumping onto a subset of translational DOFs."""

    def test_total_mass_on_enabled_dofs_only(self):
        r, conn, _ = _unit_cube_hex8()
        rho = 1000.0
        recs = r.resolve_volume_lumped(
            VolumeMassDef(target="cube", density=rho, dofs=[1, 2]), conn,
        )
        # Total on enabled channels = ρV; on z = 0
        self.assertAlmostEqual(_total(recs, 0), rho)
        self.assertAlmostEqual(_total(recs, 1), rho)
        self.assertAlmostEqual(_total(recs, 2), 0.0)

    def test_com_preserved_per_enabled_channel(self):
        """Masking zeroes some channels; the spatial distribution of
        each *enabled* channel is unchanged, so CoM-per-channel still
        equals the geometric centroid.
        """
        r, conn, coords = _unit_cube_hex8()
        recs = r.resolve_volume_lumped(
            VolumeMassDef(target="cube", density=1.0, dofs=[1, 2]), conn,
        )
        np.testing.assert_allclose(_com(recs, coords, 0), [0.5, 0.5, 0.5])
        np.testing.assert_allclose(_com(recs, coords, 1), [0.5, 0.5, 0.5])


# =====================================================================
# Group H — rotational= invariants
# =====================================================================

class TestRotationalInvariants(unittest.TestCase):
    """rotational= is a fixed per-node broadcast — geometry-independent."""

    def test_sum_equals_n_nodes_times_value(self):
        """Σ_I I_ii = n_nodes × I_ii_provided."""
        r, conn, _ = _unit_cube_hex8()
        Ix, Iy, Iz = 0.5, 0.8, 1.1
        recs = r.resolve_volume_lumped(
            VolumeMassDef(
                target="cube", density=1.0, rotational=(Ix, Iy, Iz),
            ),
            conn,
        )
        # hex8 has 8 nodes
        self.assertAlmostEqual(_total(recs, 3), 8 * Ix)
        self.assertAlmostEqual(_total(recs, 4), 8 * Iy)
        self.assertAlmostEqual(_total(recs, 5), 8 * Iz)

    def test_independent_of_element_size(self):
        """Rotational inertia per node does NOT scale with volume.

        Confirms apeGmsh does not (and does not pretend to) derive
        rotational inertia from density × volume.
        """
        coords_s = {1: (0, 0, 0), 2: (1, 0, 0), 3: (1, 1, 0), 4: (0, 1, 0),
                    5: (0, 0, 1), 6: (1, 0, 1), 7: (1, 1, 1), 8: (0, 1, 1)}
        coords_b = {nid: tuple(10 * c for c in xyz)
                    for nid, xyz in coords_s.items()}
        conn = [np.arange(1, 9, dtype=np.int64)]
        defn = VolumeMassDef(
            target="cube", density=1.0, rotational=(2.0, 3.0, 4.0),
        )
        ms = _by_node(_resolver(coords_s).resolve_volume_lumped(defn, conn))
        mb = _by_node(_resolver(coords_b).resolve_volume_lumped(defn, conn))
        for nid in ms:
            # Translational scales by 10³; rotational stays put
            np.testing.assert_allclose(mb[nid][:3], 1000.0 * ms[nid][:3])
            np.testing.assert_allclose(mb[nid][3:], ms[nid][3:])

    def test_independent_of_dofs_mask(self):
        """Rotational tuple unaffected by translational dofs= mask."""
        r, conn, _ = _unit_cube_hex8()
        common = dict(target="cube", density=1.0, rotational=(1.0, 2.0, 3.0))
        rf = _by_node(r.resolve_volume_lumped(VolumeMassDef(**common), conn))
        rm = _by_node(r.resolve_volume_lumped(
            VolumeMassDef(**common, dofs=[1]), conn,
        ))
        for nid in rf:
            np.testing.assert_allclose(rf[nid][3:], rm[nid][3:])

    def test_point_mass_rotational_sum(self):
        """Same per-node-broadcast invariant on PointMassDef."""
        r, _, _ = _unit_cube_hex8()
        node_set = {1, 2, 3}
        Ix, Iy, Iz = 0.4, 0.7, 1.3
        recs = r.resolve_point_lumped(
            PointMassDef(target="pt", mass=2.0, rotational=(Ix, Iy, Iz)),
            node_set,
        )
        self.assertAlmostEqual(_total(recs, 3), 3 * Ix)
        self.assertAlmostEqual(_total(recs, 4), 3 * Iy)
        self.assertAlmostEqual(_total(recs, 5), 3 * Iz)


if __name__ == "__main__":
    unittest.main()
