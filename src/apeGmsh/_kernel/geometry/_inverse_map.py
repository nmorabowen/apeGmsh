"""Guarded isoparametric inverse map — global point → host natural coord ξ.

The geometry core of the ``g.reinforce`` embedded-reinforcement generator
(ADR 20 D3): given a rebar point and a non-matching solid host mesh, find
which host element contains the point and the host's **natural coordinate**
ξ there, plus the shape-function ``weights`` that couple the rebar node to
the host nodes. The ``LadrunoEmbeddedRebar`` element then evaluates
``u_rebar = Σ Nᵢ(ξ)·u_host_i`` each step.

Why a dedicated, *guarded* map (ADR 20 D3). The legacy
``Tcl_generateInterfacePoints.cpp::invIsoMapping`` has confirmed blockers —
on non-convergence it returns −1 but leaves ``inBounds=true``, its solve has
no singularity guard (``det(R)→0`` ⇒ silent inf/nan), and the Newton
tolerance is a hardcoded **absolute** 1e-10. This map instead: uses a
**relative** residual tolerance (scaled by the element size), guards
``det(J)`` before every solve, **checks convergence** and sets
``in_bounds=False`` + warns on failure, and applies an explicit
**out-of-bounds policy** (reject-with-error by default; opt-in snap).

Scope (v1 = straight-sided hosts). Four **linear** host kinds —
``tri3`` / ``quad4`` (2-D) and ``tet4`` / ``hex8`` (3-D). A straight-sided
*higher-order* host (e.g. ``BezierTet10`` / ``stdBrick`` 20-node) has its
geometry and ``-xi`` domain defined by its **corner** sub-element, so the
caller maps with the corner kind (``tet4`` / ``hex8``); the fork host then
computes the full higher-order weights itself on the ``-xi`` path. Curved
high-order inverse maps are deferred (their own convergence study, D3).

``xi`` conventions (matching the OpenSees host's own natural domain):

* ``tri3`` — area (barycentric) free coords ``(L2, L3)``; ``L1 = 1−L2−L3``.
* ``tet4`` — volume (barycentric) free coords ``(L2, L3, L4)``.
* ``quad4`` — ``(ξ, η) ∈ [−1, 1]²``.
* ``hex8`` — ``(ξ, η, ζ) ∈ [−1, 1]³``.

Pure NumPy, solver-agnostic — no Gmsh, no OpenSees imports.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
from numpy import ndarray


__all__ = [
    "InverseMapResult",
    "InverseMapWarning",
    "inverse_map_single",
    "locate_point",
    "HOST_KINDS",
]


class InverseMapWarning(UserWarning):
    """A rebar point did not cleanly invert into a host (non-convergence
    or accepted only via the snap out-of-bounds policy)."""


@dataclass(frozen=True, slots=True)
class InverseMapResult:
    """Outcome of locating one point in a host mesh.

    Attributes
    ----------
    host_index
        Index of the containing host element in the caller's host list
        (``-1`` only on a degenerate all-host failure with ``snap`` off,
        which raises before returning).
    xi
        Host natural coordinates (see the module docstring for the
        per-kind convention). Length = ``ndm``.
    weights
        Shape-function weights ``Nᵢ(ξ)`` for the host element's nodes
        (sum to 1). Length = number of host nodes.
    excess
        Barycentric/parametric overshoot — ``0`` when the point is inside,
        positive when outside (the amount by which ξ violates the
        reference domain).
    in_bounds
        ``True`` iff the map converged **and** ``excess <= tol``.
    converged
        ``True`` iff the Newton solve converged (always ``True`` for the
        closed-form simplex kinds).
    """

    host_index: int
    xi: ndarray
    weights: ndarray
    excess: float
    in_bounds: bool
    converged: bool


# ---------------------------------------------------------------------------
# Shape functions + analytic gradients (gmsh / OpenSees node ordering)
# ---------------------------------------------------------------------------

def _quad4_N(xi: ndarray) -> ndarray:
    x, e = xi
    return 0.25 * np.array([
        (1 - x) * (1 - e),
        (1 + x) * (1 - e),
        (1 + x) * (1 + e),
        (1 - x) * (1 + e),
    ])


def _quad4_dN(xi: ndarray) -> ndarray:
    x, e = xi
    # columns: dN/dxi, dN/deta  -> shape (4, 2)
    return 0.25 * np.array([
        [-(1 - e), -(1 - x)],
        [(1 - e), -(1 + x)],
        [(1 + e), (1 + x)],
        [-(1 + e), (1 - x)],
    ])


# hex8 gmsh/VTK corner order
_HEX8_SIGNS = np.array([
    [-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1],
    [-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1],
], dtype=float)


def _hex8_N(xi: ndarray) -> ndarray:
    s = _HEX8_SIGNS
    return 0.125 * (1 + s[:, 0] * xi[0]) * (1 + s[:, 1] * xi[1]) * (1 + s[:, 2] * xi[2])


def _hex8_dN(xi: ndarray) -> ndarray:
    s = _HEX8_SIGNS
    a = 1 + s[:, 0] * xi[0]
    b = 1 + s[:, 1] * xi[1]
    c = 1 + s[:, 2] * xi[2]
    return 0.125 * np.column_stack([
        s[:, 0] * b * c,
        s[:, 1] * a * c,
        s[:, 2] * a * b,
    ])


# ---------------------------------------------------------------------------
# Closed-form simplex maps (exact, no Newton)
# ---------------------------------------------------------------------------

def _map_tri3(p: ndarray, X: ndarray) -> tuple[ndarray, ndarray, float, bool]:
    """Barycentric of *p* in a tri3 (corners ``X``, may be 3-D coplanar)."""
    A, B, C = X[0], X[1], X[2]
    v0, v1, v2 = B - A, C - A, p - A
    d00 = float(v0 @ v0)
    d01 = float(v0 @ v1)
    d11 = float(v1 @ v1)
    d20 = float(v2 @ v0)
    d21 = float(v2 @ v1)
    denom = d00 * d11 - d01 * d01
    if abs(denom) < 1e-30:
        return np.zeros(2), np.zeros(3), float("inf"), False
    v = (d11 * d20 - d01 * d21) / denom
    w = (d00 * d21 - d01 * d20) / denom
    u = 1.0 - v - w
    weights = np.array([u, v, w])
    return np.array([v, w]), weights, max(-float(weights.min()), 0.0), True


def _map_tet4(p: ndarray, X: ndarray) -> tuple[ndarray, ndarray, float, bool]:
    """Barycentric of *p* in a tet4 (corners ``X``)."""
    A = X[0]
    M = np.column_stack([X[1] - A, X[2] - A, X[3] - A])
    det = float(np.linalg.det(M))
    if abs(det) < 1e-30:
        return np.zeros(3), np.zeros(4), float("inf"), False
    coeffs = np.linalg.solve(M, p - A)
    v, w, x = float(coeffs[0]), float(coeffs[1]), float(coeffs[2])
    u = 1.0 - v - w - x
    weights = np.array([u, v, w, x])
    return np.array([v, w, x]), weights, max(-float(weights.min()), 0.0), True


# ---------------------------------------------------------------------------
# Guarded Newton for the tensor-product kinds (quad4, hex8)
# ---------------------------------------------------------------------------

def _newton_inverse(
    p: ndarray,
    X: ndarray,
    N_fn,
    dN_fn,
    *,
    ndm: int,
    max_iter: int = 30,
    rtol: float = 1e-10,
) -> tuple[ndarray, ndarray, bool]:
    """Solve ``Σ Nᵢ(ξ)·Xᵢ = p`` for ξ by a guarded Newton.

    Returns ``(xi, weights, converged)``. ``converged`` is ``False`` when
    the Jacobian goes singular or the residual fails to fall below a
    **relative** tolerance (scaled by the element size).
    """
    # element size for the relative residual tolerance + det scale
    span = X.max(axis=0) - X.min(axis=0)
    char_len = float(np.linalg.norm(span)) or 1.0
    atol = rtol * char_len
    det_floor = 1e-12 * char_len ** ndm

    xi = np.zeros(ndm)  # reference-domain centre for tensor kinds
    converged = False
    for _ in range(max_iter):
        N = N_fn(xi)
        residual = N @ X - p                 # (3,) or (2,) in physical space
        J = (dN_fn(xi).T @ X).T              # (phys_dim, ndm)
        # square the system onto the parametric dims (J may be (3, ndm))
        Jsq = J[:ndm, :ndm] if J.shape[0] >= ndm else J
        r = residual[:ndm] if residual.shape[0] >= ndm else residual
        if abs(float(np.linalg.det(Jsq))) < det_floor:
            return xi, N_fn(xi), False
        delta = np.linalg.solve(Jsq, -r)
        xi = xi + delta
        if float(np.linalg.norm(residual)) < atol:
            converged = True
            break
    else:
        # exhausted iterations — accept only if the final residual is small
        converged = float(np.linalg.norm(N_fn(xi) @ X - p)) < atol

    return xi, N_fn(xi), converged


def _excess_tensor(xi: ndarray) -> float:
    """Overshoot of ξ outside the ``[-1, 1]^ndm`` reference box."""
    return float(np.maximum(np.abs(xi) - 1.0, 0.0).max())


def _map_quad4(p: ndarray, X: ndarray) -> tuple[ndarray, ndarray, float, bool]:
    # quad host lives in a plane; project the residual within the plane by
    # solving the 2x2 in-plane system (Newton handles the (3,2) J via the
    # ndm-truncation in _newton_inverse when coords are coplanar in xy).
    xi, weights, conv = _newton_inverse(p, X, _quad4_N, _quad4_dN, ndm=2)
    return xi, weights, _excess_tensor(xi), conv


def _map_hex8(p: ndarray, X: ndarray) -> tuple[ndarray, ndarray, float, bool]:
    xi, weights, conv = _newton_inverse(p, X, _hex8_N, _hex8_dN, ndm=3)
    return xi, weights, _excess_tensor(xi), conv


# host kind -> (n_nodes, ndm, mapper)
HOST_KINDS: dict[str, tuple[int, int]] = {
    "tri3": (3, 2),
    "quad4": (4, 2),
    "tet4": (4, 3),
    "hex8": (8, 3),
}

_MAPPERS = {
    "tri3": _map_tri3,
    "quad4": _map_quad4,
    "tet4": _map_tet4,
    "hex8": _map_hex8,
}


def inverse_map_single(
    p: ndarray,
    X: ndarray,
    kind: str,
) -> tuple[ndarray, ndarray, float, bool]:
    """Invert one point into one host element.

    Returns ``(xi, weights, excess, converged)``. Does **not** apply any
    out-of-bounds policy — use :func:`locate_point` for the
    locate-among-many + reject/snap behaviour.
    """
    if kind not in _MAPPERS:
        raise ValueError(
            f"inverse_map_single: unsupported host kind {kind!r} "
            f"(v1 supports {sorted(HOST_KINDS)}; a straight-sided "
            f"higher-order host maps with its corner kind)"
        )
    X = np.asarray(X, dtype=float)
    expected = HOST_KINDS[kind][0]
    if X.shape[0] != expected:
        raise ValueError(
            f"inverse_map_single: {kind} host needs {expected} node "
            f"coords, got {X.shape[0]}"
        )
    return _MAPPERS[kind](np.asarray(p, dtype=float), X)


def locate_point(
    p: ndarray,
    host_coords: list[ndarray],
    host_kinds: list[str],
    *,
    tol: float = 1e-6,
    snap: bool = False,
    k_candidates: int = 16,
    label: str = "",
) -> InverseMapResult:
    """Locate *p* among many host elements and return the best inverse map.

    Searches the ``k_candidates`` nearest hosts (by centroid), inverts into
    each, and keeps the smallest-excess hit. Out-of-bounds policy (ADR 20
    D3): if the best ``excess > tol`` then **reject with an error** by
    default, or — when ``snap=True`` — accept the nearest host and warn
    (the caller may then clamp ξ to the domain).

    Parameters
    ----------
    p
        The rebar point, shape ``(2,)`` or ``(3,)``.
    host_coords
        Per-host node-coordinate arrays, each ``(n_nodes_i, dim)``.
    host_kinds
        Per-host element kind (``"tet4"`` / ``"hex8"`` / ``"quad4"`` /
        ``"tri3"``), parallel to ``host_coords``.
    tol
        Acceptance threshold on the barycentric/parametric ``excess``.
    snap
        ``False`` (default) -> raise when no host contains the point;
        ``True`` -> accept the nearest host, ``in_bounds=False``, + warn.
    label
        Optional generator label, threaded into the error/warning text.
    """
    if len(host_coords) != len(host_kinds):
        raise ValueError(
            "locate_point: host_coords and host_kinds length mismatch "
            f"({len(host_coords)} vs {len(host_kinds)})"
        )
    if not host_coords:
        raise ValueError("locate_point: no host elements supplied")

    p = np.asarray(p, dtype=float)
    centroids = np.array([np.asarray(X, dtype=float).mean(axis=0) for X in host_coords])
    order = np.argsort(np.linalg.norm(centroids - p, axis=1))
    cand = order[: min(k_candidates, len(order))]

    best: InverseMapResult | None = None
    for ei in cand:
        ei = int(ei)
        xi, weights, excess, conv = inverse_map_single(
            p, host_coords[ei], host_kinds[ei]
        )
        if not conv:
            continue
        if best is None or excess < best.excess:
            best = InverseMapResult(
                host_index=ei, xi=xi, weights=weights,
                excess=excess, in_bounds=excess <= tol, converged=True,
            )
        if excess <= tol:
            break

    where = f" (reinforcement {label!r})" if label else ""

    if best is None or not best.in_bounds:
        if best is not None and snap:
            warnings.warn(
                f"locate_point: rebar point {p.tolist()} lies outside every "
                f"host element (best barycentric/parametric excess "
                f"{best.excess:.3e} > tol {tol:.3e}); SNAPPED to the nearest "
                f"host (index {best.host_index}){where}. The tie is "
                f"extrapolated — verify the rebar/host geometry.",
                InverseMapWarning,
                stacklevel=2,
            )
            return best
        best_excess = best.excess if best is not None else float("inf")
        raise ValueError(
            f"locate_point: rebar point {p.tolist()} lies outside every host "
            f"element (best barycentric/parametric excess {best_excess:.3e}, "
            f"tolerance {tol:.3e}){where}. Fix the geometry/mesh so the rebar "
            f"node falls inside a host, widen `tol`, or pass `snap=True` to "
            f"project it onto the nearest host."
        )

    return best
