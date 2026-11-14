"""The fork's Contract-1 fixture, re-seeded from a gmsh-free FEMData (ADR 0134 D2).

``particle_set_v1.h5`` (fork commit 55f1831f, vendored by M2-S0) is the model of 232a
section 4: plane strain, t = 1 m, kN_m_s; a 4 x 4 grid of 0.5 m cells with GIMP; body
``soil`` (x 0.5..1.5, y 0.5..1.0; 8 particles, tag 2, rho 2.0) and body ``block``
(x 1.0..1.5, y 1.0..1.5; 4 particles, tag 1, rho 7.85), 2 x 2 particles per cell.

Here the soil is two quad4 and the block two tri3 whose shared diagonal runs through two
of its four lattice points, so the shared-edge rule is on the reproduction's path too.
The fixture's K0 ``STRESS``/``STRESS33`` are initial state, which M2 does not seed.
"""
from __future__ import annotations

import h5py
import numpy as np

from apeGmsh.opensees._mpm import (
    MPMBody,
    MPMGrid,
    Provenance,
    Units,
    preflight_particle_set,
    seed_lattice,
    write_particle_set,
)

from . import _meshes as M
from .conftest import MPM_FIXTURES

FIXTURE = MPM_FIXTURES / "particle_set_v1.h5"


def fixture_fem():
    nodes = {
        1: (0.5, 0.5), 2: (1.0, 0.5), 3: (1.5, 0.5),
        4: (0.5, 1.0), 5: (1.0, 1.0), 6: (1.5, 1.0),
        7: (1.0, 1.5), 8: (1.5, 1.5),
    }
    return M.make_fem(nodes, {
        "soil": [("quad4", [1, 2, 5, 4]), ("quad4", [2, 3, 6, 5])],
        "block": [("tri3", [5, 6, 8]), ("tri3", [5, 8, 7])],
    })


def reseed(tmp_path):
    seeded = seed_lattice(
        fixture_fem(),
        MPMGrid(origin=(0.0, 0.0), cell_size=(0.5, 0.5), num_cells=(4, 4), basis="gimp"),
        [MPMBody("soil", "soil", 2.0), MPMBody("block", "block", 7.85)],
        n=2, thickness=1.0,
    )
    out = tmp_path / "fixture.particles.h5"
    write_particle_set(preflight_particle_set(
        out, seeded, material_tags={"soil": 2, "block": 1},
        units=Units("m", "kN", "s", "kN_m_s"),
        provenance=Provenance("9b2e7c41-5d3a-4f6e-8c1b-2a3d4e5f6a7b", "fixture-wp232a",
                              "unknown"),
    ))
    return seeded, out


def test_reseeding_reproduces_the_fork_fixture(tmp_path, contracts):
    seeded, out = reseed(tmp_path)
    assert seeded.num_particles == 12
    # the block's diagonal (y = x) carries two of its lattice points, each kept once
    block = seeded.position[seeded.body_id == 1]
    assert np.count_nonzero(block[:, 0] == block[:, 1]) == 2

    with h5py.File(FIXTURE, "r") as want, h5py.File(out, "r") as got:
        for name in ("POSITION", "VOLUME", "GIMP_HALF_LENGTH", "BODY_ID",
                     "DENSITY", "MATERIAL_TAG", "ID"):
            w, g = want["PARTICLES"][name], got["PARTICLES"][name]
            assert g.dtype == w.dtype, name
            np.testing.assert_array_equal(g[...], w[...], err_msg=name)
        assert [s.decode() if isinstance(s, bytes) else s for s in want["BODIES/NAME"][...]] \
            == [s.decode() if isinstance(s, bytes) else s for s in got["BODIES/NAME"][...]]
        np.testing.assert_array_equal(got["BODIES/ID"][...], want["BODIES/ID"][...])
        for k in ("ORIGIN", "CELL_SIZE", "NUM_CELLS", "BASIS"):
            np.testing.assert_array_equal(got["GRID"].attrs[k], want["GRID"].attrs[k],
                                          err_msg=k)
        for k in ("FORMAT", "FORMAT_VERSION", "GENERATOR", "NDM", "NUM_PARTICLES",
                  "PLANE_STRAIN_THICKNESS", "UNIT_LENGTH", "UNIT_FORCE", "UNIT_TIME",
                  "UNIT_SYSTEM", "MASS_SOURCE", "STRESS_SIGN", "VOIGT_ORDER"):
            assert got.attrs[k] == want.attrs[k], k
            assert np.asarray(got.attrs[k]).dtype == np.asarray(want.attrs[k]).dtype, k
        for k in ("METHOD", "PARTICLES_PER_CELL"):
            np.testing.assert_array_equal(got["SEEDING"].attrs[k], want["SEEDING"].attrs[k])
        assert got.attrs["SET_ID"] != want.attrs["SET_ID"]  # a new set, a new SET_ID (D1)

    warnings: list[str] = []
    assert contracts.validate_particle_set(str(out), warnings) == []
    assert warnings == []


def test_the_fixture_itself_passes_the_vendored_validator(contracts):
    assert contracts.validate_particle_set(str(FIXTURE)) == []
