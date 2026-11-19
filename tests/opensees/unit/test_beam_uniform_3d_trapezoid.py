"""The fixed-end-force oracle behind the 3-D trapezoid live test (#2550).

``tests/opensees/live/test_beam_uniform_3d_trapezoid_live.py`` compares the
per-node reactions of a fixed-fixed 3-D beam under a trapezoidal
``-beamUniform`` with :func:`fixed_end_reactions_global`. This file holds that
oracle and pins it, without OpenSees, against textbook fixed-end forces
(uniform: ``wL/2``, ``wL^2/12``; triangle 0 to w: ``3wL/20``, ``7wL/20``,
``wL^2/30``, ``wL^2/20``), against the two cases in the #2561 review, and
against its own statics.

Conventions. A transverse load ``q`` acts along local ``+v`` (``v`` = local
y or z). ``F`` is a support force along ``+v``; ``R`` is a support moment in
the sense of ``theta = dv/dx``. That is ``Mz`` for ``v = y`` and ``-My`` for
``v = z``, since a positive rotation about local y turns z toward x.

* Exact: quadrature of the clamped-clamped Euler-Bernoulli influence
  functions. Gauss-Legendre (8 points) on the loaded segment is exact, the
  integrands being polynomials of degree 4 or less.
* Force-based (``quadrature=``): the moment field
  ``M = M0 + m_i (1 - x/L) + m_j x/L`` (``M0`` simply supported, ``M'' = q``)
  with the two compatibility conditions ``int M (1 - x/L) = int M x/L = 0``
  taken at the given points, which is what a force-based element with an
  elastic section solves. The axial force does the same with
  ``int N = 0``. Support forces then follow from statics.
"""
from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence

import numpy as np
import pytest

_GL8 = np.polynomial.legendre.leggauss(8)

# Gauss-Lobatto, 5 points on [-1, 1] (OpenSees ``beamIntegration Lobatto 5``).
LOBATTO_5: tuple[np.ndarray, np.ndarray] = (
    np.array([-1.0, -np.sqrt(3.0 / 7.0), 0.0, np.sqrt(3.0 / 7.0), 1.0]),
    np.array([0.1, 49.0 / 90.0, 32.0 / 45.0, 49.0 / 90.0, 0.1]),
)

Pair = tuple[float, float]


def _integrate(f: Callable[[float], float], lo: float, hi: float) -> float:
    if hi <= lo:
        return 0.0
    x, w = _GL8
    s = 0.5 * (hi - lo) * x + 0.5 * (hi + lo)
    return 0.5 * (hi - lo) * float(sum(wk * f(float(sk)) for sk, wk in zip(s, w)))


def _load(pair: Pair, a: float, b: float) -> Callable[[float], float]:
    wa, wb = pair
    return lambda x: wa + (wb - wa) * (x - a) / (b - a)


def transverse_exact(length: float, pair: Pair, a_l: float, b_l: float
                     ) -> tuple[float, float, float, float]:
    """``(F_i, F_j, R_i, R_j)`` of the clamped-clamped beam, exact."""
    L, a, b = length, a_l * length, b_l * length
    q = _load(pair, a, b)
    f_i = -_integrate(lambda x: q(x) * (L - x) ** 2 * (L + 2 * x) / L**3, a, b)
    f_j = -_integrate(lambda x: q(x) * x**2 * (3 * L - 2 * x) / L**3, a, b)
    r_i = -_integrate(lambda x: q(x) * x * (L - x) ** 2 / L**2, a, b)
    r_j = _integrate(lambda x: q(x) * x**2 * (L - x) / L**2, a, b)
    return f_i, f_j, r_i, r_j


def _points(length: float, quadrature: tuple[np.ndarray, np.ndarray]
            ) -> tuple[np.ndarray, np.ndarray]:
    xi, w = quadrature
    return 0.5 * length * (xi + 1.0), 0.5 * length * w


def transverse_force_based(length: float, pair: Pair, a_l: float, b_l: float,
                           quadrature: tuple[np.ndarray, np.ndarray],
                           ) -> tuple[float, float, float, float]:
    """``(F_i, F_j, R_i, R_j)`` with compatibility taken at the points."""
    L, a, b = length, a_l * length, b_l * length
    q = _load(pair, a, b)

    def m0(x: float) -> float:   # simply supported, M'' = q, M0(0) = M0(L) = 0
        def g(s: float) -> float:
            return -(x * (L - s) if x <= s else s * (L - x)) / L
        mid = min(max(x, a), b)
        return (_integrate(lambda s: g(s) * q(s), a, mid)
                + _integrate(lambda s: g(s) * q(s), mid, b))

    xs, ws = _points(L, quadrature)
    phi = np.array([[1.0 - x / L, x / L] for x in xs])        # (n, 2)
    flex = phi.T @ (ws[:, None] * phi)
    rhs = -phi.T @ (ws * np.array([m0(float(x)) for x in xs]))
    m_i, m_j = np.linalg.solve(flex, rhs)
    r_i, r_j = -float(m_i), float(m_j)
    total = _integrate(q, a, b)
    first = _integrate(lambda x: x * q(x), a, b)
    f_j = -(first + r_i + r_j) / L
    return -total - f_j, f_j, r_i, r_j


def axial(length: float, pair: Pair, a_l: float, b_l: float,
          quadrature: tuple[np.ndarray, np.ndarray] | None = None,
          ) -> tuple[float, float]:
    """``(F_i, F_j)`` of the bar fixed at both ends (EA constant)."""
    L, a, b = length, a_l * length, b_l * length
    q = _load(pair, a, b)
    if quadrature is None:
        return (-_integrate(lambda x: q(x) * (L - x) / L, a, b),
                -_integrate(lambda x: q(x) * x / L, a, b))
    # N = c - Q(x), Q(x) = int_0^x q; compatibility sum w N = 0.
    xs, ws = _points(L, quadrature)
    big_q = [_integrate(q, a, min(max(float(x), a), b)) for x in xs]
    c = float(np.dot(ws, big_q) / ws.sum())
    return -c, c - _integrate(q, a, b)


def local_triad(xj: Sequence[float], vecxz: Sequence[float]) -> np.ndarray:
    """Rows: local x, y, z in global axes, by the ``geomTransf`` rule
    (``x`` from node i to j, ``y = vecxz x x``, ``z = x x y``)."""
    ex = np.array(xj, dtype=float)          # a copy: never normalise the caller's
    ex /= np.linalg.norm(ex)
    ey = np.cross(np.asarray(vecxz, dtype=float), ex)
    ey /= np.linalg.norm(ey)
    return np.array([ex, ey, np.cross(ex, ey)])


def fixed_end_reactions_global(
    length: float, triad: np.ndarray, w: Mapping[str, Pair],
    a_l: float, b_l: float,
    quadrature: tuple[np.ndarray, np.ndarray] | None = None,
) -> np.ndarray:
    """Support reactions in global axes, rows ``(i, j)``, columns
    ``Fx Fy Fz Mx My Mz``; ``w`` maps ``wy``/``wz``/``wx`` to (start, end)."""
    def trans(pair: Pair) -> tuple[float, float, float, float]:
        if quadrature is None:
            return transverse_exact(length, pair, a_l, b_l)
        return transverse_force_based(length, pair, a_l, b_l, quadrature)

    fy_i, fy_j, ry_i, ry_j = trans(w["wy"])     # v = y: Mz = R
    fz_i, fz_j, rz_i, rz_j = trans(w["wz"])     # v = z: My = -R
    fx_i, fx_j = axial(length, w["wx"], a_l, b_l, quadrature)
    local = np.array([
        [fx_i, fy_i, fz_i, 0.0, -rz_i, ry_i],
        [fx_j, fy_j, fz_j, 0.0, -rz_j, ry_j],
    ])
    out = np.empty_like(local)
    out[:, :3] = local[:, :3] @ triad
    out[:, 3:] = local[:, 3:] @ triad
    return out


# ---------------------------------------------------------------------------
# The oracle against textbook fixed-end forces
# ---------------------------------------------------------------------------

_L = 4.0


def test_uniform_load_textbook() -> None:
    w = 3.0
    f_i, f_j, r_i, r_j = transverse_exact(_L, (w, w), 0.0, 1.0)
    assert (f_i, f_j) == pytest.approx((-w * _L / 2, -w * _L / 2), rel=1e-13)
    assert (r_i, r_j) == pytest.approx(
        (-w * _L**2 / 12, w * _L**2 / 12), rel=1e-13)


def test_triangular_load_textbook() -> None:
    w = 5.0
    f_i, f_j, r_i, r_j = transverse_exact(_L, (0.0, w), 0.0, 1.0)
    assert (f_i, f_j) == pytest.approx(
        (-3 * w * _L / 20, -7 * w * _L / 20), rel=1e-13)
    assert (r_i, r_j) == pytest.approx(
        (-w * _L**2 / 30, w * _L**2 / 20), rel=1e-13)


def test_review_case_wy_2_to_10() -> None:
    """#2561 review: |M_i| = 6.933, |M_j| = 9.067; dropping the tail gives
    2.667 at both ends with the same sums."""
    _, _, r_i, r_j = transverse_exact(_L, (2.0, 10.0), 0.0, 1.0)
    assert (-r_i, r_j) == pytest.approx((104 / 15, 136 / 15), rel=1e-13)
    _, _, d_i, d_j = transverse_exact(_L, (2.0, 2.0), 0.0, 1.0)
    assert (-d_i, d_j) == pytest.approx((8 / 3, 8 / 3), rel=1e-13)


def test_partial_uniform_textbook() -> None:
    """Uniform w over [a, b]: compare with a point-load superposition."""
    a_l, b_l, w = 0.25, 0.75, 2.0
    _, _, r_i, r_j = transverse_exact(_L, (w, w), a_l, b_l)
    xs = np.linspace(a_l * _L, b_l * _L, 20001)
    dx = xs[1] - xs[0]
    trap = np.full_like(xs, dx)
    trap[[0, -1]] *= 0.5
    m_i = float(np.sum(w * trap * xs * (_L - xs) ** 2) / _L**2)
    m_j = float(np.sum(w * trap * xs**2 * (_L - xs)) / _L**2)
    assert (-r_i, r_j) == pytest.approx((m_i, m_j), rel=1e-7)


def test_axial_textbook() -> None:
    assert axial(_L, (2.0, 2.0), 0.0, 1.0) == pytest.approx((-4.0, -4.0))
    # Triangle 0 to w: wL/6 at i, wL/3 at j.
    assert axial(_L, (0.0, 3.0), 0.0, 1.0) == pytest.approx((-2.0, -4.0))


@pytest.mark.parametrize("pair", [(2.0, 10.0), (-4.0, -8.0), (1.0, 3.0)])
def test_lobatto_full_span_equals_exact(pair: Pair) -> None:
    """Over the full span the compatibility integrands are polynomials of
    degree 4, which Lobatto-5 integrates exactly."""
    exact = transverse_exact(_L, pair, 0.0, 1.0)
    fb = transverse_force_based(_L, pair, 0.0, 1.0, LOBATTO_5)
    assert fb == pytest.approx(exact, rel=1e-12)
    assert axial(_L, pair, 0.0, 1.0, LOBATTO_5) == pytest.approx(
        axial(_L, pair, 0.0, 1.0), rel=1e-12)


def test_force_based_with_fine_quadrature_converges_to_exact() -> None:
    """The force-based oracle is the exact one with the integrals sampled:
    a fine rule recovers the exact partial-span answer."""
    xi, w = np.polynomial.legendre.leggauss(200)
    exact = transverse_exact(_L, (2.0, 10.0), 0.25, 0.75)
    fine = transverse_force_based(_L, (2.0, 10.0), 0.25, 0.75, (xi, w))
    assert fine == pytest.approx(exact, rel=1e-5)
    coarse = transverse_force_based(_L, (2.0, 10.0), 0.25, 0.75, LOBATTO_5)
    assert coarse != pytest.approx(exact, rel=1e-5)   # the rule matters here


def test_local_triad_follows_vecxz() -> None:
    t = local_triad((4.0, 0.0, 0.0), (0.0, -1.0, 0.0))
    np.testing.assert_allclose(t, [[1, 0, 0], [0, 0, 1], [0, -1, 0]], atol=1e-15)


@pytest.mark.parametrize("quadrature", [None, LOBATTO_5])
def test_global_reactions_balance_the_load(quadrature: object) -> None:
    """Statics in global axes on a skew beam: sum F and sum M about node i."""
    xj = np.array([2.0, 1.0, 2.0])
    length = float(np.linalg.norm(xj))
    tri = local_triad(xj, (0.0, -1.0, 0.0))
    w = {"wy": (2.0, 10.0), "wz": (-4.0, -8.0), "wx": (1.0, 3.0)}
    a_l, b_l = 0.25, 0.75
    r = fixed_end_reactions_global(
        length, tri, w, a_l, b_l, quadrature=quadrature)  # type: ignore[arg-type]
    a, b = a_l * length, b_l * length
    f_local = np.array([
        _integrate(_load(w[k], a, b), a, b) for k in ("wx", "wy", "wz")])
    s_local = np.array([
        _integrate(lambda x, k=k: x * _load(w[k], a, b)(x), a, b)
        for k in ("wx", "wy", "wz")])
    load_f = f_local @ tri
    # The load on the axis at x = s gives r x F = s e_x x (Fy e_y + Fz e_z).
    load_m = np.cross(tri[0], s_local[1] * tri[1] + s_local[2] * tri[2])
    np.testing.assert_allclose(r[0, :3] + r[1, :3], -load_f, atol=1e-12)
    np.testing.assert_allclose(
        r[0, 3:] + r[1, 3:] + np.cross(xj, r[1, :3]), -load_m, atol=1e-12)
