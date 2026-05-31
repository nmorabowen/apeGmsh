"""Tests — ``loads.line(normal=True)`` is plane-general.

Historically the per-edge in-plane normal was hardcoded to the XY
plane (``(Ty, -Tx, 0)``), so a model built in the XZ plane (the
natural orientation for a tunnel cross-section: x horizontal, z
vertical) got forces pointing purely out-of-plane in ±Y — silently
wrong.  The resolver now fits the plane of the loaded curves and
computes ``T x P`` in that plane.

Covers:
* XZ-plane arch + columns: convergence (inward) vs internal (outward)
  by ``magnitude`` sign, with no out-of-plane leakage.
* XY-plane parity (backward compatibility).
* Surface path (no ``away_from``) generalised to a non-XY plane.
* Fail-loud on a non-planar 3-D curve.
* Collinear single segment recovered via ``away_from``.
"""
from __future__ import annotations

import numpy as np
import pytest


def _tunnel(g, *, plane: str, mag: float, q=5.0,
            L=10.0, H1=4.0, H2=2.0, lc=1.0):
    """Build the arch+columns frame in *plane* and resolve the load.

    Returns ``(inward, outward, max_out_of_plane_fraction)``.
    """
    geo = g.model.geometry
    if plane == "XZ":           # x horizontal, z vertical (y = 0)
        P = lambda x, v: (x, 0.0, v)            # noqa: E731
        ctr = (L / 2, 0.0, H1)
        oop = 1                                  # Y is out of plane
    else:                       # XY: x horizontal, y vertical (z = 0)
        P = lambda x, v: (x, v, 0.0)            # noqa: E731
        ctr = (L / 2, H1, 0.0)
        oop = 2                                  # Z is out of plane
    geo.add_point(*P(0, 0),       mesh_size=lc, label="bl")
    geo.add_point(*P(0, H1),      mesh_size=lc, label="tl")
    geo.add_point(*P(L, 0),       mesh_size=lc, label="br")
    geo.add_point(*P(L, H1),      mesh_size=lc, label="tr")
    geo.add_point(*P(L / 2, H1 + H2), mesh_size=lc, label="tc")
    geo.add_line("bl", "tl", label="cl")
    geo.add_line("br", "tr", label="cr")
    geo.add_arc("tl", "tc", "tr", label="ar", through_point=True)
    g.model.select(None, dim=1).to_physical(name="frames")
    with g.loads.case("p"):
        g.loads.line(target="frames", magnitude=mag,
                      normal=True, away_from=ctr)
    g.mesh.generation.generate(dim=1)
    fem = g.mesh.queries.get_fem_data(dim=1)

    cc = np.asarray(ctr, dtype=float)
    inward = outward = 0
    max_oop = 0.0
    for ld in fem.nodes.loads:
        idx = fem.nodes.index(ld.node_id)
        xyz = np.asarray(fem.nodes.coords[idx], dtype=float)
        f = np.asarray(ld.force_xyz or (0.0, 0.0, 0.0), dtype=float)
        nf = float(np.linalg.norm(f))
        if nf < 1e-9:
            continue
        max_oop = max(max_oop, abs(f[oop]) / nf)
        if float(np.dot(cc - xyz, f)) > 0.0:
            inward += 1
        else:
            outward += 1
    return inward, outward, max_oop


@pytest.mark.parametrize("plane", ["XZ", "XY"])
def test_negative_magnitude_is_convergence(g, plane):
    """``magnitude < 0`` with ``away_from`` = cavity centre pulls every
    node inward (tunnel convergence), in-plane, on any plane."""
    inward, outward, oop = _tunnel(g, plane=plane, mag=-5.0)
    assert inward > 0 and outward == 0
    assert oop < 1e-9          # forces lie in the structure's plane


@pytest.mark.parametrize("plane", ["XZ", "XY"])
def test_positive_magnitude_is_internal_pressure(g, plane):
    """``magnitude > 0`` pushes every node outward (internal
    pressure) — opposite sign, still in-plane."""
    inward, outward, oop = _tunnel(g, plane=plane, mag=+5.0)
    assert outward > 0 and inward == 0
    assert oop < 1e-9


def test_surface_path_generalised_to_non_xy_plane(g):
    """Without ``away_from`` the surface path uses the adjacent face's
    Gmsh-oriented normal — must work for a surface in the XZ plane,
    not only XY."""
    geo = g.model.geometry
    geo.add_point(0, 0, 0, label="p0")
    geo.add_point(4, 0, 0, label="p1")
    geo.add_point(4, 0, 3, label="p2")
    geo.add_point(0, 0, 3, label="p3")
    geo.add_line("p0", "p1", label="e0")
    geo.add_line("p1", "p2", label="e1")
    geo.add_line("p2", "p3", label="e2")
    geo.add_line("p3", "p0", label="e3")
    lp = geo.add_curve_loop(["e0", "e1", "e2", "e3"])
    geo.add_plane_surface(lp, label="face")
    g.model.select("e0").to_physical(name="edge0")
    with g.loads.case("p"):
        g.loads.line(target="edge0", magnitude=10.0, normal=True)
    g.mesh.generation.generate(dim=2)
    fem = g.mesh.queries.get_fem_data(dim=2)

    com = np.array([2.0, 0.0, 1.5])     # surface centroid (XZ plane)
    n_into = n_tot = 0
    for ld in fem.nodes.loads:
        idx = fem.nodes.index(ld.node_id)
        xyz = np.asarray(fem.nodes.coords[idx], dtype=float)
        f = np.asarray(ld.force_xyz or (0.0, 0.0, 0.0), dtype=float)
        if float(np.linalg.norm(f)) < 1e-9:
            continue
        n_tot += 1
        assert abs(f[1]) < 1e-9         # in-plane (no out-of-plane Y)
        if float(np.dot(com - xyz, f)) > 0.0:
            n_into += 1
    assert n_tot > 0 and n_into == n_tot


def test_non_planar_curve_fails_loud(g):
    """A genuinely 3-D space curve has no single in-plane normal —
    the resolver must raise, not emit garbage."""
    geo = g.model.geometry
    geo.add_point(0, 0, 0, label="a")
    geo.add_point(1, 0, 1, label="b")
    geo.add_point(2, 1, 0, label="c")
    geo.add_point(3, 0, 2, label="d")
    geo.add_spline(["a", "b", "c", "d"], label="helix")
    g.model.select(None, dim=1).to_physical(name="sp")
    with g.loads.case("p"):
        g.loads.line(target="sp", magnitude=1.0,
                      normal=True, away_from=(1, 1, 1))
    g.mesh.generation.generate(dim=1)
    with pytest.raises(ValueError, match="not planar"):
        g.mesh.queries.get_fem_data(dim=1)


def test_collinear_segment_recovered_via_away_from(g):
    """A single straight segment is collinear (no unique plane); an
    off-line ``away_from`` defines the plane and the force points away
    from it, in that plane."""
    geo = g.model.geometry
    geo.add_point(0, 0, 0, label="a")
    geo.add_point(0, 0, 5, label="b")          # vertical line along Z
    geo.add_line("a", "b", label="col")
    g.model.select(None, dim=1).to_physical(name="c1")
    with g.loads.case("p"):
        g.loads.line(target="c1", magnitude=3.0, normal=True,
                      away_from=(2, 0, 2.5))     # +x, off the line
    g.mesh.generation.generate(dim=1)
    fem = g.mesh.queries.get_fem_data(dim=1)

    loaded = [np.asarray(ld.force_xyz or (0.0, 0.0, 0.0), dtype=float)
              for ld in fem.nodes.loads]
    loaded = [f for f in loaded if float(np.linalg.norm(f)) > 1e-9]
    assert loaded
    for f in loaded:
        assert f[0] < 0.0                       # away from +x reference
        assert abs(f[1]) < 1e-9 and abs(f[2]) < 1e-9   # in-plane
