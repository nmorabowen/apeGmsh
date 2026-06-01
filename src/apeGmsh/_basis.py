"""Self-describing finite-element basis front-door ``B(ξ; FAMILY, ORDER)``.

A ``.ladruno`` file describes each element group's geometry with a
``BASIS`` descriptor (``TOPOLOGY``/``FAMILY``/``ORDER``/``NUM_CTRL`` +
``QUADRATURE/GP_PARAM``). The reader reconstructs Gauss-point world
coordinates self-describingly — ``x(ξ) = Σ Rᵢ(ξ)·Xᵢ`` — *from the file*,
with **no per-element-class shape-function table** (recorder plan,
"What's different from MPCO"). This module is that single neutral
evaluator, keyed by the descriptor vocabulary the file speaks.

It is deliberately **non-duplicative**: the *lagrange* families already
live in :mod:`apeGmsh.fem._shape_functions` (Gmsh-keyed, validated), so
this front-door **delegates** lagrange to that library. What it *adds*
is the **bernstein** (Bézier) basis the fork's BezierTri6 / BezierTet10
elements (#3/#4) write — which has no representation elsewhere yet. The
bezier read path (``plan_bezier_elements_integration.md`` B4) imports the
same :func:`basis_values` for ``B(ξ; bernstein, 2)``; settling it here,
once, is the plan's up-front decision (one copy, shared by both).

Node ordering is normative — matched **against the reference BezierTri6 /
BezierTet10 elements** (``C:\\Users\\nmora\\Github\\bezierFEM``, Kadapa
2018 IJNME 117:543 §5), which is what the fork writes into the
``.ladruno`` ``CONNECTIVITY``. Vertices first, then mid-edges. BezierTri6::

    N1=ξ₃²  N2=ξ₁²  N3=ξ₂²            (vertices)
    N4=2ξ₁ξ₃  N5=2ξ₁ξ₂  N6=2ξ₂ξ₃     (mid-edges 1-2, 2-3, 3-1)

BezierTet10 mid-edge order carries the reference's **Larenas N9↔N10 swap
to match Gmsh** — edges ``(1-2, 2-3, 1-3, 1-4, 3-4, 2-4)``, *not* the
naive ``…, 1-4, 2-4, 3-4``. Getting this wrong silently corrupts
``x_global`` (no exception), which is the whole point of pinning it here.

Simplex bernstein ``xi`` is the file's ``GP_PARAM`` form — the **free**
area/volume coords (2 for tri, 3 for tet); the last is derived
(``ξ₃=1−ξ₁−ξ₂`` / ``L4=1−L1−L2−L3``). Full barycentric (3 / 4 cols) is
also accepted.

``basis_values`` returns the basis matrix ``R`` of shape
``(n_points, n_ctrl)`` so a caller forms ``x = R @ X`` with ``X`` the
control-point coordinates in connectivity order.
"""
from __future__ import annotations

import numpy as np
from numpy import ndarray

__all__ = ["basis_values", "BasisError"]


class BasisError(ValueError):
    """Raised when a ``(topology, family, order)`` triple is unsupported."""


# ---------------------------------------------------------------------------
# Lagrange delegation — (topology, total order) → Gmsh element-type code
# ---------------------------------------------------------------------------
# The fork writes ORDER as a scalar (simplex/line) or a per-direction list
# (tensor product). We collapse to the max direction order; the families
# below are isotropic in the orders apeGmsh meshes produce.
_LAGRANGE_GMSH_CODE: dict[tuple[str, int], int] = {
    ("line", 1): 1,   # Line2
    ("tri", 1): 2,    # Tri3
    ("quad", 1): 3,   # Quad4
    ("tet", 1): 4,    # Tet4
    ("hex", 1): 5,    # Hex8
    ("tri", 2): 9,    # Tri6
    ("quad", 2): 10,  # Quad9
    ("tet", 2): 11,   # Tet10
    ("hex", 2): 17,   # Hex20 (serendipity — the order the fork emits)
}


def _max_order(order) -> int:
    if isinstance(order, (list, tuple, np.ndarray)):
        return int(max(int(o) for o in np.asarray(order).ravel()))
    return int(order)


def _as_points(xi, *, dim: int) -> ndarray:
    """Coerce ``xi`` to ``(n_points, dim)`` float64."""
    arr = np.asarray(xi, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1) if dim > 1 else arr.reshape(-1, 1)
    if arr.shape[1] != dim:
        raise BasisError(
            f"expected parametric coords of width {dim}, got shape {arr.shape}."
        )
    return arr


def _as_simplex(xi, *, n_free: int) -> ndarray:
    """Coerce ``xi`` to full barycentric ``(n_points, n_free + 1)``.

    Accepts the file's ``GP_PARAM`` form (``n_free`` columns — the last
    coord is derived as ``1 − Σ``) or already-complete barycentric
    (``n_free + 1`` columns).
    """
    arr = np.asarray(xi, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    w = arr.shape[1]
    if w == n_free:
        last = 1.0 - arr.sum(axis=1, keepdims=True)
        return np.hstack([arr, last])
    if w == n_free + 1:
        return arr
    raise BasisError(
        f"simplex bernstein expects {n_free} free or {n_free + 1} full "
        f"barycentric coords, got width {w}."
    )


# ---------------------------------------------------------------------------
# Bernstein simplex evaluators (the genuinely-new, bezier-shared piece)
# ---------------------------------------------------------------------------

def _bernstein_tri(order: int, bary: ndarray) -> ndarray:
    """Degree-``order`` Bernstein triangle, ``bary = (ξ₁, ξ₂, ξ₃)``.

    Returns ``(n_points, n_ctrl)`` in the contract's vertices-then-edges
    order. Only ``order == 2`` (BezierTri6) is pinned; other orders raise.
    """
    x1, x2, x3 = bary[:, 0], bary[:, 1], bary[:, 2]
    if order == 1:
        return np.stack([x1, x2, x3], axis=1)
    if order == 2:
        # Pinned order: P1=ξ₃², P2=ξ₁², P3=ξ₂², P4=2ξ₁ξ₃, P5=2ξ₁ξ₂, P6=2ξ₂ξ₃.
        return np.stack(
            [x3 * x3, x1 * x1, x2 * x2,
             2 * x1 * x3, 2 * x1 * x2, 2 * x2 * x3],
            axis=1,
        )
    raise BasisError(f"bernstein tri order {order} is not pinned (v1: order 1, 2).")


def _bernstein_tet(order: int, bary: ndarray) -> ndarray:
    """Degree-``order`` Bernstein tetrahedron, ``bary = (L1, L2, L3, L4)``.

    Vertices first, then 6 mid-edges in the reference BezierTet10 order
    (the Larenas N9↔N10 Gmsh swap): edges ``(1-2, 2-3, 1-3, 1-4, 3-4,
    2-4)``. Matches ``bernstein_shape_functions`` in the reference
    ``bezier_tet10_element.py``.
    """
    l1, l2, l3, l4 = bary[:, 0], bary[:, 1], bary[:, 2], bary[:, 3]
    if order == 1:
        return np.stack([l1, l2, l3, l4], axis=1)
    if order == 2:
        return np.stack(
            [l1 * l1, l2 * l2, l3 * l3, l4 * l4,    # N1-N4 vertices
             2 * l1 * l2,                            # N5  edge (1-2)
             2 * l2 * l3,                            # N6  edge (2-3)
             2 * l1 * l3,                            # N7  edge (1-3)
             2 * l1 * l4,                            # N8  edge (1-4)
             2 * l3 * l4,                            # N9  edge (3-4)
             2 * l2 * l4],                           # N10 edge (2-4)
            axis=1,
        )
    raise BasisError(f"bernstein tet order {order} is not pinned (v1: order 1, 2).")


def _bernstein_1d(order: int, t: ndarray) -> ndarray:
    """Univariate Bernstein basis on ``t ∈ [0, 1]`` → ``(n_points, order+1)``."""
    from math import comb
    return np.stack(
        [comb(order, i) * (t ** i) * ((1.0 - t) ** (order - i))
         for i in range(order + 1)],
        axis=1,
    )


def _bernstein_tensor(orders: list[int], xi01: ndarray) -> ndarray:
    """Tensor-product Bernstein on ``[0,1]^d``, lexicographic (fastest in U).

    ``orders`` is one degree per parametric direction. Column index runs
    ``i + (orderU+1)·j + …`` per the contract's tensor ordering.
    """
    dim = len(orders)
    per_dir = [_bernstein_1d(orders[d], xi01[:, d]) for d in range(dim)]
    out = per_dir[0]
    for d in range(1, dim):
        # Kronecker per row with the next direction, fastest index first.
        n_pts = out.shape[0]
        out = (out[:, :, None] * per_dir[d][:, None, :]).reshape(n_pts, -1)
    return out


# ---------------------------------------------------------------------------
# Public front-door
# ---------------------------------------------------------------------------

def basis_values(
    *, topology: str, family: str, order, xi: ndarray,
) -> ndarray:
    """Evaluate the basis ``R(ξ)`` for one element family.

    Parameters
    ----------
    topology
        ``"line"`` / ``"tri"`` / ``"quad"`` / ``"tet"`` / ``"hex"`` — the
        file's ``BASIS/TOPOLOGY``.
    family
        ``"lagrange"`` (delegated to :mod:`apeGmsh.fem._shape_functions`)
        or ``"bernstein"`` (evaluated here).
    order
        Scalar or per-direction list — the file's ``BASIS/ORDER``.
    xi
        Parametric coordinates, ``(n_points, dim)`` (or ``(dim,)`` for a
        single point). Expected in the family's natural domain: ``[-1,1]``
        for lagrange/bernstein tensor topologies, barycentric for simplex
        bernstein (``ξ`` summing to 1).

    Returns
    -------
    ndarray
        ``(n_points, n_ctrl)`` basis matrix; ``x = R @ X``.
    """
    topo = topology.lower()
    fam = family.lower()
    o = _max_order(order)

    if fam == "lagrange":
        code = _LAGRANGE_GMSH_CODE.get((topo, o))
        if code is None:
            raise BasisError(
                f"no lagrange basis for topology {topo!r} order {o} "
                f"(known: {sorted(_LAGRANGE_GMSH_CODE)})."
            )
        from .fem._shape_functions import get_shape_functions
        entry = get_shape_functions(code)
        if entry is None:
            raise BasisError(
                f"shape-function catalog has no entry for Gmsh code {code} "
                f"(topology {topo!r} order {o})."
            )
        n_fn = entry[0]
        dim = 1 if topo == "line" else (3 if topo in ("tet", "hex") else 2)
        return np.asarray(n_fn(_as_points(xi, dim=dim)), dtype=np.float64)

    if fam == "bernstein":
        if topo == "tri":
            return _bernstein_tri(o, _as_simplex(xi, n_free=2))
        if topo == "tet":
            return _bernstein_tet(o, _as_simplex(xi, n_free=3))
        if topo in ("line", "quad", "hex"):
            dim = {"line": 1, "quad": 2, "hex": 3}[topo]
            pts = _as_points(xi, dim=dim)
            orders = (
                [int(x) for x in np.asarray(order).ravel()]
                if isinstance(order, (list, tuple, np.ndarray))
                else [o] * dim
            )
            if len(orders) != dim:
                raise BasisError(
                    f"bernstein {topo} expects {dim} per-direction orders, "
                    f"got {orders}."
                )
            return _bernstein_tensor(orders, pts)
        raise BasisError(f"bernstein topology {topo!r} is not supported.")

    raise BasisError(
        f"family {family!r} is not supported (expected 'lagrange' or "
        f"'bernstein')."
    )
