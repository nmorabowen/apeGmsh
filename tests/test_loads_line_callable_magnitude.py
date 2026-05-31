"""Tests — ``loads.line(magnitude=callable)`` (spatially varying).

``magnitude`` may be a callable ``q(xyz) -> float`` sampled at each
edge midpoint, so a line load can vary with position — e.g. a
depth-dependent ground / convergence pressure on a tunnel lining.

Covers:
* depth-varying convergence (``normal=True`` + ``away_from``): the
  resolved force grows with depth, points inward, stays in-plane;
* a constant callable reproduces the uniform-float path exactly
  (backward compatibility / no behaviour change);
* the ``direction=`` path, ``reduction="consistent"`` and
  ``target_form="element"`` all honour the callable;
* fail-loud guards: callable ``magnitude`` + ``q_xyz`` is rejected,
  and a callable returning a non-finite value raises.
"""
from __future__ import annotations

import numpy as np
import pytest


def _build_tunnel(g, *, L=10.0, H1=4.0, H2=2.0, lc=1.0):
    """Arch + columns in the XZ plane; PG 'frames' over all curves."""
    geo = g.model.geometry
    geo.add_point(0,   0, 0,      mesh_size=lc, label="bl")
    geo.add_point(0,   0, H1,     mesh_size=lc, label="tl")
    geo.add_point(L,   0, 0,      mesh_size=lc, label="br")
    geo.add_point(L,   0, H1,     mesh_size=lc, label="tr")
    geo.add_point(L / 2, 0, H1 + H2, mesh_size=lc, label="tc")
    geo.add_line("bl", "tl", label="cl")
    geo.add_line("br", "tr", label="cr")
    geo.add_arc("tl", "tc", "tr", label="ar", through_point=True)
    g.model.select(None, dim=1).to_physical(name="frames")
    return L, H1, H2


def _resolved(g, dim=1):
    fem = g.mesh.queries.get_fem_data(dim=dim)
    out = []
    for ld in fem.nodes.loads:
        i = fem.nodes.index(ld.node_id)
        xyz = np.asarray(fem.nodes.coords[i], dtype=float)
        f = np.asarray(ld.force_xyz or (0.0, 0.0, 0.0), dtype=float)
        out.append((xyz, f))
    return out


def test_depth_varying_convergence(g):
    """q grows linearly with depth; every node pulled inward,
    in-plane; deeper nodes get a larger force than the crown."""
    L, H1, H2 = _build_tunnel(g)
    z_top, gamma = H1 + H2, 10.0
    ctr = (L / 2, 0.0, H1)
    with g.loads.case("convergence"):
        g.loads.line(target="frames",
                      magnitude=lambda p: -gamma * (z_top - p[2]),
                      normal=True, away_from=ctr)
    g.mesh.generation.generate(dim=1)
    rows = _resolved(g)

    cc = np.asarray(ctr, dtype=float)
    nz = [(x, f) for x, f in rows if np.linalg.norm(f) > 1e-9]
    assert nz
    for x, f in nz:
        assert float(np.dot(cc - x, f)) > 0.0       # inward
        assert abs(f[1]) < 1e-9                      # in-plane (XZ)
    deep = np.mean([np.linalg.norm(f) for x, f in nz if x[2] < 1.0])
    crown = np.mean([np.linalg.norm(f) for x, f in nz
                     if x[2] > z_top - 0.5])
    assert deep > 5.0 * crown                        # grows with depth


def test_constant_callable_matches_uniform_float(g):
    """A constant callable must resolve identically to the plain
    float — the uniform path is unchanged (backward compatibility)."""
    L, H1, H2 = _build_tunnel(g)
    ctr = (L / 2, 0.0, H1)
    with g.loads.case("a"):
        g.loads.line(target="frames", magnitude=5.0,
                      normal=True, away_from=ctr)
    with g.loads.case("b"):
        g.loads.line(target="frames", magnitude=lambda p: 5.0,
                      normal=True, away_from=ctr)
    g.mesh.generation.generate(dim=1)
    fem = g.mesh.queries.get_fem_data(dim=1)

    by_pat: dict[str, dict] = {"a": {}, "b": {}}
    for ld in fem.nodes.loads:
        i = fem.nodes.index(ld.node_id)
        key = tuple(np.round(fem.nodes.coords[i], 6))
        by_pat[ld.pattern][key] = np.round(
            np.asarray(ld.force_xyz or (0, 0, 0), float), 6)
    assert by_pat["a"].keys() == by_pat["b"].keys()
    for k in by_pat["a"]:
        assert np.allclose(by_pat["a"][k], by_pat["b"][k])


def test_direction_path_callable(g):
    """Non-normal path: q = magnitude(midpoint) * direction."""
    _build_tunnel(g)
    with g.loads.case("p"):
        g.loads.line(target="frames",
                      magnitude=lambda p: -(2.0 + p[2]),
                      direction=(0, 0, 1))
    g.mesh.generation.generate(dim=1)
    nz = [(x, f) for x, f in _resolved(g) if np.linalg.norm(f) > 1e-9]
    assert nz
    for _x, f in nz:
        assert abs(f[0]) < 1e-9 and abs(f[1]) < 1e-9   # pure z
        assert f[2] < 0.0


def test_consistent_reduction_callable(g):
    """``reduction='consistent'`` honours the callable too."""
    L, H1, H2 = _build_tunnel(g)
    z_top = H1 + H2
    ctr = (L / 2, 0.0, H1)
    with g.loads.case("p"):
        g.loads.line(target="frames",
                      magnitude=lambda p: -10.0 * (z_top - p[2]),
                      normal=True, away_from=ctr,
                      reduction="consistent")
    g.mesh.generation.generate(dim=1)
    cc = np.asarray(ctr, dtype=float)
    nz = [(x, f) for x, f in _resolved(g) if np.linalg.norm(f) > 1e-9]
    assert nz
    assert all(float(np.dot(cc - x, f)) > 0.0 for x, f in nz)


def test_element_form_callable_is_per_element(g):
    """``target_form='element'`` emits a per-element ``beamUniform``
    sampled at each element midpoint — distinct values, not one."""
    _build_tunnel(g)
    with g.loads.case("p"):
        g.loads.line(target="frames",
                      magnitude=lambda p: -(1.0 + p[2]),
                      direction=(0, 0, 1), target_form="element")
    g.mesh.generation.generate(dim=1)
    fem = g.mesh.queries.get_fem_data(dim=1)
    recs = list(fem.elements.loads)
    assert recs
    wz = {round(r.params["wz"], 3) for r in recs}
    assert len(wz) > 1                       # genuinely varying
    assert all(r.load_type == "beamUniform" for r in recs)


def test_element_form_uniform_float_unchanged(g):
    """The plain-float element path still emits one constant value."""
    _build_tunnel(g)
    with g.loads.case("p"):
        g.loads.line(target="frames", magnitude=-7.0,
                      direction=(0, 0, 1), target_form="element")
    g.mesh.generation.generate(dim=1)
    fem = g.mesh.queries.get_fem_data(dim=1)
    wz = {round(r.params["wz"], 4) for r in fem.elements.loads}
    assert wz == {-7.0}


def test_callable_magnitude_with_q_xyz_is_rejected(g):
    _build_tunnel(g)
    with pytest.raises(ValueError, match="mutually exclusive"):
        with g.loads.case("p"):
            g.loads.line(target="frames",
                          magnitude=lambda p: 1.0, q_xyz=(1, 0, 0))


def test_non_finite_callable_fails_loud(g):
    L, H1, H2 = _build_tunnel(g)
    with g.loads.case("p"):
        g.loads.line(target="frames",
                      magnitude=lambda p: float("inf"),
                      normal=True, away_from=(L / 2, 0, H1))
    g.mesh.generation.generate(dim=1)
    with pytest.raises(ValueError, match="non-finite"):
        g.mesh.queries.get_fem_data(dim=1)


def _column_total_fz(*, n_elems, reduction, qfun, H=6.0):
    """Own-session straight vertical column of *n_elems* line2
    elements; apply the callable along +z; return summed nodal Fz.

    Spins its own apeGmsh session so it can be called repeatedly
    (mesh-convergence sweep) without stacking geometry.
    """
    from apeGmsh import apeGmsh
    lc = H / n_elems
    with apeGmsh(model_name="col", verbose=False) as m:
        geo = m.model.geometry
        geo.add_point(0, 0, 0, mesh_size=lc, label="a")
        geo.add_point(0, 0, H, mesh_size=lc, label="b")
        geo.add_line("a", "b", label="col")
        m.model.select(None, dim=1).to_physical(name="c1")
        with m.loads.case("p"):
            m.loads.line(target="c1", magnitude=qfun,
                          direction=(0, 0, 1), reduction=reduction)
        m.mesh.generation.generate(dim=1)
        fem = m.mesh.queries.get_fem_data(dim=1)
        return sum(float((ld.force_xyz or (0, 0, 0))[2])
                   for ld in fem.nodes.loads)


def test_consistent_is_gauss_exact_for_nonlinear_field():
    """``reduction='consistent'`` integrates the callable at the
    element Gauss points, so a quadratic field is **exact** even on a
    2-element mesh — it does not over/undershoot ∫q."""
    H = 6.0
    qfun = lambda p: p[2] ** 2          # noqa: E731
    ref = H ** 3 / 3.0                  # ∫₀^H z² dz = 72
    total = _column_total_fz(n_elems=2, reduction="consistent",
                             qfun=qfun, H=H)
    assert total == pytest.approx(ref, rel=1e-9)


def test_tributary_midpoint_undershoots_then_converges():
    """The (documented) flip side: ``reduction='tributary'`` is the
    midpoint rule — it under/overshoots ∫q on a coarse mesh and
    converges O(h²).  Locks the contrast with the consistent path."""
    H = 6.0
    qfun = lambda p: p[2] ** 2          # noqa: E731
    ref = H ** 3 / 3.0
    e2 = abs(_column_total_fz(n_elems=2, reduction="tributary",
                              qfun=qfun, H=H) - ref)
    e8 = abs(_column_total_fz(n_elems=8, reduction="tributary",
                              qfun=qfun, H=H) - ref)
    assert e2 / ref > 0.02              # ~6% off at 2 elements
    assert e8 < 0.3 * e2                # converges as the mesh refines


def test_mesh_viewer_loads_panel_formats_callable_magnitude():
    """The mesh-viewer loads panel must not crash on a callable
    magnitude — ``f"{m:.4g}"`` raises ``TypeError`` on a function, so
    the detail column routes through ``_fmt_magnitude``.

    Pure-function test (no Qt / GPU): exercises the exact formatter
    the panel uses.
    """
    from apeGmsh.viewers.ui.loads_tab import _fmt_magnitude

    def ground_pressure(p):
        return 1.0

    assert _fmt_magnitude(-15000.0) == "-1.5e+04"      # float unchanged
    assert _fmt_magnitude(0.0) == "0"
    assert _fmt_magnitude(lambda p: -18e3 * p[2]) == "q(xyz)"
    assert _fmt_magnitude(ground_pressure) == "ground_pressure(xyz)"
    # never raises, even on something exotic
    assert isinstance(_fmt_magnitude(object()), str)
