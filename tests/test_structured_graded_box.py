"""Tests for g.mesh.structured.build_graded_box() — PM-01 D14's combinator.

The verb replaces a hand-assembly of ``set_transfinite_box`` +
``set_transfinite_curve(mesh_type="Progression")`` +
``set_transfinite_volume`` with one call that produces the footing-on-soil
discretisation PM-01 §10 declares: a uniform **mechanism block** at cell
``h`` around the footprint, a **geometric growth ratio** ``r`` per cell
outside it, and the footprint edges landing exactly on mesh lines.

Two count oracles are used:

* an **arithmetic** one, :func:`_predict`, restating the layout law
  independently of the implementation;
* the **reference-case** figures of PM-01 ``calcs/pm01_domain_cost.py``.
  That script's far-field cell counts come from ``n_geom(span, seed,
  ratio)`` — cells growing by ``ratio`` from ``seed`` until they cross
  ``span``.  ``build_graded_box`` uses the same law with ``seed = h``,
  the declared §10 near-field cell; the script instead seeds at an
  undeclared ``0.275``.  ``test_reference_case_matches_pm01_cost_law``
  pins the *law* against the script's own published rows by feeding the
  seed the script used, and
  ``test_reference_case_counts_at_the_declared_seed`` pins what the verb
  actually builds.  Both are cheap arithmetic; only the small cases mesh.
"""
from __future__ import annotations

import math

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# A coarse case that meshes in well under a second: 9 x 9 x 6 m domain,
# 1.5 m square footprint, h = 0.75 (B = 2 cells), block one cell out and
# one cell deep, growth 1.5.
COARSE = dict(
    extent=(9.0, 9.0, 6.0), footprint=(1.5, 1.5),
    h=0.75, l_mech=0.75, d_mech=0.75, r=1.5,
)


def _n_geom(span: float, seed: float, ratio: float) -> int:
    """``pm01_domain_cost.py``'s far-field cell count, verbatim."""
    n, tot, h = 0, 0.0, seed
    while tot < span - 1e-9:
        tot += h
        n += 1
        h *= ratio
    return n


def _predict(*, extent, footprint, h, l_mech, d_mech, r, seed=None):
    """Grid lines / nodes / elements the layout law implies."""
    bx, ly, hz = extent
    bf, lf = footprint
    seed = h if seed is None else seed
    n_l = round(l_mech / h)
    n_d = max(1, round(d_mech / h))
    a_x, a_y, d_z = bf / 2 + n_l * h, lf / 2 + n_l * h, n_d * h
    nx = round(bf / h) + 2 * n_l + 2 * _n_geom(bx / 2 - a_x, seed, r) + 1
    ny = round(lf / h) + 2 * n_l + 2 * _n_geom(ly / 2 - a_y, seed, r) + 1
    nz = n_d + _n_geom(hz - d_z, seed, r) + 1
    return dict(lines=(nx, ny, nz), nodes=nx * ny * nz,
                elements=(nx - 1) * (ny - 1) * (nz - 1))


def _build(g, **over):
    """Build + mesh a case; return (volume tags, node coords, element ids)."""
    kw = dict(COARSE)
    kw.update(over)
    vols = g.mesh.structured.build_graded_box(**kw)
    g.physical.add_volume(vols, name="soil")
    g.mesh.generation.generate(dim=3)
    fem = g.mesh.queries.get_fem_data(dim=3)
    return vols, np.asarray(fem.nodes.coords), np.asarray(fem.elements.ids)


def _lines(xyz):
    return tuple(len(np.unique(np.round(xyz[:, a], 7))) for a in range(3))


# ---------------------------------------------------------------------------
# Counts — against the arithmetic oracle and against PM-01's cost script
# ---------------------------------------------------------------------------

def test_coarse_case_counts_match_the_layout_law(g):
    want = _predict(**COARSE)
    vols, xyz, eids = _build(g)
    assert len(vols) == 18                      # 3 x 3 x 2 sub-volumes
    assert _lines(xyz) == want["lines"]
    assert len(xyz) == want["nodes"]
    assert len(eids) == want["elements"]


def test_reference_case_matches_pm01_cost_law():
    """The far-field law reproduces ``pm01_domain_cost.py``'s own rows.

    Fed the seed that script uses (0.275), ``_predict`` must return the
    grid lines / nodes / elements published in PM-01 §6 for the three
    geometric-grading rows of the settled 15B x 15B x 12B domain.
    """
    ref = dict(extent=(22.5, 22.5, 18.0), footprint=(1.5, 1.5),
               h=1.5 / 8, l_mech=1.5, d_mech=1.5)
    published = {                    # PM-01 §6, calcs/pm01_domain_cost.py
        1.2: ((49, 49, 24), 57_624, 52_992),
        1.3: ((45, 45, 21), 42_525, 38_720),
        1.4: ((41, 41, 19), 31_939, 28_800),
    }
    for r, (lines, nodes, elements) in published.items():
        got = _predict(**ref, r=r, seed=0.275)
        assert got["lines"] == lines, r
        assert got["nodes"] == nodes, r
        assert got["elements"] == elements, r


def test_reference_case_counts_at_the_declared_seed():
    """What the verb actually builds: the far field seeded at ``h``.

    Seeding at the declared near-field cell instead of the script's
    undeclared 0.275 buys one extra far-field cell per side laterally
    and two in depth — a denser, not a coarser, far field.
    """
    ref = dict(extent=(22.5, 22.5, 18.0), footprint=(1.5, 1.5),
               h=1.5 / 8, l_mech=1.5, d_mech=1.5)
    assert _predict(**ref, r=1.2)["lines"] == (51, 51, 26)
    assert _predict(**ref, r=1.3)["lines"] == (47, 47, 22)
    assert _predict(**ref, r=1.4)["lines"] == (43, 43, 20)


# ---------------------------------------------------------------------------
# The footprint edge is a mesh line
# ---------------------------------------------------------------------------

def test_footprint_edges_are_mesh_lines(g):
    """A surface node sits exactly on each footprint corner (PM-01 D14)."""
    _, xyz, _ = _build(g, footprint=(1.5, 3.0))
    top = xyz[np.abs(xyz[:, 2]) < 1e-9]
    for sx in (-1, 1):
        for sy in (-1, 1):
            hit = (np.abs(top[:, 0] - sx * 0.75) < 1e-9) & \
                  (np.abs(top[:, 1] - sy * 1.5) < 1e-9)
            assert hit.any(), f"no surface node at ({sx * 0.75}, {sy * 1.5}, 0)"


def test_h_must_divide_the_footprint_exactly(g):
    with pytest.raises(ValueError, match="does not divide"):
        g.mesh.structured.build_graded_box(**{**COARSE, "h": 0.7})


# ---------------------------------------------------------------------------
# The growth ratio outside the block
# ---------------------------------------------------------------------------

def test_far_field_layers_grow_by_r(g):
    """Consecutive far-field layer thicknesses are in the ratio ``r``.

    Measured in depth, where the block is one cell thick and everything
    below ``z = -d_mech`` is the graded slab.
    """
    r = COARSE["r"]
    _, xyz, _ = _build(g)
    z = np.unique(np.round(xyz[:, 2], 9))
    far = np.sort(z[z <= -COARSE["d_mech"] + 1e-9])       # -hz .. -d_mech
    t = np.diff(far)                                      # bottom -> up
    assert len(t) >= 3
    # thickness grows downward, i.e. shrinks by 1/r walking up
    assert np.allclose(t[1:] / t[:-1], 1.0 / r, rtol=1e-6)
    # the cell touching the block is never coarser than h
    assert t[-1] <= COARSE["h"] * (1 + 1e-9)


def test_r_equal_one_grades_nothing(g):
    _, xyz, _ = _build(g, r=1.0)
    z = np.unique(np.round(xyz[:, 2], 9))
    far = np.sort(z[z <= -COARSE["d_mech"] + 1e-9])
    t = np.diff(far)
    assert np.allclose(t, COARSE["h"], rtol=1e-9)


def test_r_below_one_is_refused(g):
    with pytest.raises(ValueError, match="r must be >= 1"):
        g.mesh.structured.build_graded_box(**{**COARSE, "r": 0.8})


# ---------------------------------------------------------------------------
# Orientation — the two-orientation refinement study
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("angle", [45.0, 90.0])
def test_orientation_rotates_the_lines_and_keeps_the_counts(g, angle):
    """Same element/node counts, block lines rotated, footprint still exact.

    ``orientation`` lays the grid out axis-aligned and rotates it
    rigidly about z, so the counts cannot change and the footprint —
    which rotates with it — stays on mesh lines in its own frame.
    """
    want = _predict(**COARSE)
    _, xyz, eids = _build(g, orientation=angle)
    assert len(xyz) == want["nodes"]
    assert len(eids) == want["elements"]

    a = math.radians(-angle)                       # back to the grid frame
    ca, sa = math.cos(a), math.sin(a)
    local = np.column_stack([
        xyz[:, 0] * ca - xyz[:, 1] * sa,
        xyz[:, 0] * sa + xyz[:, 1] * ca,
        xyz[:, 2],
    ])
    assert _lines(local) == want["lines"]

    top = local[np.abs(local[:, 2]) < 1e-9]
    hit = (np.abs(top[:, 0] - 0.75) < 1e-7) & (np.abs(top[:, 1] - 0.75) < 1e-7)
    assert hit.any(), "footprint corner is not a mesh node in the grid frame"


def test_orientation_45_is_not_axis_aligned(g):
    """Guard the positive control: at 45 degrees the lines really moved."""
    _, xyz, _ = _build(g, orientation=45.0)
    # an axis-aligned 11 x 11 grid would have 11 distinct x values
    assert _lines(xyz)[0] > 11


# ---------------------------------------------------------------------------
# The block has to fit
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("over", [
    {"l_mech": 6.0},        # block wider than the domain
    {"l_mech": 3.75},       # block exactly the domain half-width
    {"d_mech": 6.0},        # block depth == domain depth
])
def test_block_must_fit_strictly_inside_the_domain(g, over):
    with pytest.raises(ValueError, match="fit strictly inside"):
        g.mesh.structured.build_graded_box(**{**COARSE, **over})
