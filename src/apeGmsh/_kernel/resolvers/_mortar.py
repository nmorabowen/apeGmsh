"""Dual-basis integral-mortar tie kernel (ADR 0086).

Pure-numpy port of the algorithm in the Ladruno fork's
``LadrunoMortarKernel.h`` / ``LadrunoTie::generateMortar`` — extended to
quadratic facets (quad8 slave/master, tri6 master), which is the reason
the port exists: the fork kernel is hard-limited to tri3/quad4 and the
driving use case (hex20 rib faces tied onto hex8 plates) presents quad8.

Scope (ADR 0086 D3 — v1): **flat, coincident, convex** facets. The
interface must lie in one plane to within ``gap_tol``; every facet's
corner polygon must be convex; anything else is a hard
:class:`MortarTieError` — there is no silent zero-force path.

Algorithm
---------
1.  A common interface plane is derived from the master facets' corner
    winding (summed unit normals, fork ``LadrunoTie.cpp:644-649``);
    ``outward=`` overrides; a cancelling sum is a named refusal.
2.  Every facet is projected to 2-D plane coordinates.  Per slave facet,
    the overlap with each master facet is clipped (Sutherland–Hodgman,
    convex ∩ convex), fan-triangulated, and Gauss-integrated (Duffy
    5×5 per triangle — exact past the quad8×quad8 product degree on
    affine cells):

        D_IJ += w·N_I^s·N_J^s       M_IK += w·N_I^s·φ_K^m

    Geometry uses the **corner** (bilinear/affine) map — exact for flat
    straight-edged facets with mid-side nodes at edge midpoints — while
    the full quadratic bases are evaluated at the mapped parametric
    point.
3.  Per-facet dual condensation: ``P^e = (D^e)⁻¹ M^e`` scaled by the
    facet row-sums and accumulated per slave node (the ``-dual``
    biorthogonal route; the fork's P2.1 oracle shows the lumped
    alternative errs 0.28 on the linear patch vs 2e-15 for dual, so no
    toggle is exposed).
4.  Guards (fork parity, sign-safe for quadratic bases): straight
    edges — every midside at its edge midpoint; no master facet
    overlapping another; area coverage per slave facet ≥ 1−1e-3;
    near-zero accumulated row refusal (sign-aware and *relative*, since
    quad8 corner dual masses are legitimately negative); final
    partition of unity |Σ_k P_Ik − 1| ≤ 1e-6 per slave node.

tri6 **slave** facets are refused: the tri6 corner shape functions
integrate to exactly zero over the parent triangle, so the dual row-sum
scaling divides by zero (quad8 corners integrate to −A/12 ≠ 0 and are
fine).  tri6 is accepted on the master side.

Divergences from the fork contract
----------------------------------
The fork kernel is the reference implementation and ADR-78's companion
``78_apegmsh_mortar_crosscheck_requirements.md`` (R1–R8) is the contract
this port satisfies.  Three requirements are met differently, on
purpose:

* **R3 quadrature.** The fork uses a 12-point degree-6 Dunavant rule on
  each sub-triangle; this kernel uses a Duffy-collapsed 5×5 tensor rule
  (25 points, higher order).  Both are exact for the N·φ products on
  *affine* cells, which is why the R7 cross-check agrees to 5e-13.  On a
  non-parallelogram quad the bilinear inverse is rational, neither rule
  is exact, and the two kernels will differ at quadrature level — so a
  future cross-check case with skewed facets must match the fork's rule
  before it can demand 1e-12.
* **R5.2 conforming-gap L1.** Not implemented as a separate measure: the
  stricter v1 scope subsumes it.  Every interface node must already lie
  within ``gap_tol`` of the *common plane* and every master normal must
  be coherent to 0.99, so the accordion surface R5.2 exists to catch
  cannot reach the integration loop.  Lift this scope and the L1 gap
  guard has to come with it.
* **R6 export side.** Emitting hex20 elements whose faces are
  mortar-tied under the explicit ``LadrunoProjection`` handler requires
  ``LadrunoBrick20(lumped=True)`` (HRZ) — row-sum lumping gives negative
  corner masses, which the fork refuses by name.  That knob exists and
  is documented on the element; nothing here verifies the combination.
"""
from __future__ import annotations

from typing import Callable

import numpy as np
from numpy import ndarray

from ._constraint_resolver._geom import SHAPE_FUNCTIONS

__all__ = ["MortarTieError", "compute_dual_mortar_rows"]


class MortarTieError(ValueError):
    """Hard failure of the mortar tie kernel (ADR 0086 D3 fail-loud).

    Subclasses ``ValueError`` so chain-phase routing propagates it —
    :func:`try_chain_phase_route` swallows only KeyError/TypeError.
    """


#: Facet-size-relative area below which a clipped overlap is ignored.
_EPS_AREA_REL = 1.0e-10
#: Fork parity: per-slave-facet covered-area fraction required.
_COVERAGE_TOL = 1.0e-3
#: Fork parity: partition-of-unity tolerance per slave node.
_POU_TOL = 1.0e-6
#: Fork parity: near-zero coefficient filter at row build.
_W_FILTER = 1.0e-12
#: Normal-coherence floor: every facet normal vs the plane normal.
_NORMAL_COHERENCE = 0.99
#: Fork ADR-78 R2: midside straightness, relative to the longest
#: corner edge of the facet.
_MIDSIDE_TOL = 1.0e-6
#: Fork ADR-78 R5.3: master-pair mutual clip area, relative to the
#: smaller facet, above which the two masters count as overlapping.
_SELF_OVERLAP_TOL = 1.0e-6


def _corner_count(nps: int) -> int:
    if nps in (3, 6):
        return 3
    if nps in (4, 8):
        return 4
    raise MortarTieError(
        f"mortar tie: unsupported facet arity {nps} — supported facets "
        f"are tri3/quad4/quad8 (+ tri6 master-side)."
    )


# ── 1-D Gauss (n=5) + Duffy triangle rule ───────────────────────────

_G5_X = np.array([
    0.5 - 0.9061798459386640 / 2, 0.5 - 0.5384693101056831 / 2, 0.5,
    0.5 + 0.5384693101056831 / 2, 0.5 + 0.9061798459386640 / 2,
])
_G5_W = np.array([
    0.2369268850561891, 0.4786286704993665, 0.5688888888888889,
    0.4786286704993665, 0.2369268850561891,
]) / 2.0


def _duffy_points() -> tuple[ndarray, ndarray]:
    """Gauss points/weights on the reference triangle (0,0)-(1,0)-(0,1).

    Duffy collapse of a 5×5 tensor rule: x=u, y=v(1-u), Jacobian (1-u).
    Weights sum to the reference-triangle area 1/2.
    """
    pts, wts = [], []
    for u, wu in zip(_G5_X, _G5_W):
        for v, wv in zip(_G5_X, _G5_W):
            pts.append((u, v * (1.0 - u)))
            wts.append(wu * wv * (1.0 - u))
    return np.asarray(pts), np.asarray(wts)


_TRI_PTS, _TRI_WTS = _duffy_points()


# ── 2-D polygon utilities (convex) ──────────────────────────────────

def _poly_area(poly: ndarray) -> float:
    x, y = poly[:, 0], poly[:, 1]
    return 0.5 * float(
        np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))
    )


def _ensure_ccw_convex(poly: ndarray, what: str) -> ndarray:
    """Return the polygon CCW-oriented; refuse non-convex/degenerate."""
    area = _poly_area(poly)
    scale = float(np.max(np.ptp(poly, axis=0)))
    if abs(area) < (1e-12 * scale * scale + 1e-300):
        raise MortarTieError(f"mortar tie: degenerate {what} facet.")
    if area < 0.0:
        poly = poly[::-1].copy()
    n = len(poly)
    for i in range(n):
        a, b, c = poly[i], poly[(i + 1) % n], poly[(i + 2) % n]
        cross = (b[0] - a[0]) * (c[1] - b[1]) - (b[1] - a[1]) * (c[0] - b[0])
        if cross < -1e-9 * abs(area):
            raise MortarTieError(
                f"mortar tie: non-convex {what} facet — v1 scope is "
                f"flat, coincident, CONVEX facets (ADR 0086 D3)."
            )
    return poly


def _clip_convex(subject: ndarray, clip: ndarray) -> ndarray:
    """Sutherland–Hodgman: subject ∩ clip, both convex CCW."""
    out = list(subject)
    n = len(clip)
    for i in range(n):
        if not out:
            break
        a, b = clip[i], clip[(i + 1) % n]
        ex, ey = b[0] - a[0], b[1] - a[1]

        def _inside(p) -> bool:
            return ex * (p[1] - a[1]) - ey * (p[0] - a[0]) >= -1e-14

        def _intersect(p, q):
            dx, dy = q[0] - p[0], q[1] - p[1]
            denom = ex * dy - ey * dx
            if abs(denom) < 1e-300:
                return q
            t = (ex * (a[1] - p[1]) - ey * (a[0] - p[0])) / denom
            return (p[0] + t * dx, p[1] + t * dy)

        prev, res = out[-1], []
        for cur in out:
            if _inside(cur):
                if not _inside(prev):
                    res.append(_intersect(prev, cur))
                res.append(tuple(cur))
            elif _inside(prev):
                res.append(_intersect(prev, cur))
            prev = cur
        out = res
    return np.asarray(out, dtype=float) if out else np.empty((0, 2))


# ── Corner-map parametric inversion ─────────────────────────────────

def _invert_corner_map(corners2d: ndarray, p: ndarray) -> tuple[float, float]:
    """Parametric (ξ, η) of 2-D point ``p`` under the corner map.

    Triangles: affine (area coordinates, matching ``_shape_tri3``).
    Quads: bilinear Newton inverse (ξ, η ∈ [-1, 1]²).
    """
    if len(corners2d) == 3:
        a, b, c = corners2d
        mat = np.array([[b[0] - a[0], c[0] - a[0]],
                        [b[1] - a[1], c[1] - a[1]]])
        xi, eta = np.linalg.solve(mat, p - a)
        return float(xi), float(eta)

    xi = eta = 0.0
    for _ in range(30):
        n4 = 0.25 * np.array([
            (1 - xi) * (1 - eta), (1 + xi) * (1 - eta),
            (1 + xi) * (1 + eta), (1 - xi) * (1 + eta),
        ])
        r = n4 @ corners2d - p
        if float(r @ r) < 1e-26:
            break
        dxi = 0.25 * np.array([-(1 - eta), (1 - eta), (1 + eta), -(1 + eta)])
        deta = 0.25 * np.array([-(1 - xi), -(1 + xi), (1 + xi), (1 - xi)])
        jac = np.column_stack([dxi @ corners2d, deta @ corners2d])
        step = np.linalg.solve(jac, r)
        xi, eta = xi - float(step[0]), eta - float(step[1])
    return float(xi), float(eta)


# ── Facet preprocessing ─────────────────────────────────────────────

class _Facet:
    __slots__ = ("tags", "nps", "ncor", "poly", "corners2d", "coords2d",
                 "bb_lo", "bb_hi", "area")

    def __init__(self, tags: ndarray, coords2d: ndarray, what: str) -> None:
        self.tags = [int(t) for t in tags]
        self.nps = len(self.tags)
        self.ncor = _corner_count(self.nps)
        self.coords2d = coords2d
        self.corners2d = coords2d[: self.ncor]
        self.poly = _ensure_ccw_convex(self.corners2d.copy(), what)
        self.area = abs(_poly_area(self.poly))
        self.bb_lo = self.poly.min(axis=0)
        self.bb_hi = self.poly.max(axis=0)


def _check_midside_nodes(tags: list[int], xyz: dict, what: str) -> None:
    """Fork ADR-78 R2 (MUST): midside nodes sit at their edge midpoints.

    ALL geometry here runs on the corner sub-facet, which is exact only
    because the serendipity map collapses identically to the corner map
    when every midside is at its edge midpoint.  A curved edge silently
    breaks that identity and nothing else catches it — a midside slid
    10 % along its own edge leaves a 0.30 linear-patch error while
    coverage, partition of unity and every other guard stay clean.  So
    it is a hard error naming the facet, never a silent skip.
    """
    n = len(tags)
    if n not in (6, 8):
        return
    ncor = n // 2
    corners = np.array([xyz[t] for t in tags[:ncor]])
    edges = [(k, (k + 1) % ncor) for k in range(ncor)]
    hmax = max(float(np.linalg.norm(corners[b] - corners[a]))
               for a, b in edges)
    for k, (a, b) in enumerate(edges):
        tag_mid = tags[ncor + k]
        off = float(np.linalg.norm(
            xyz[tag_mid] - 0.5 * (corners[a] + corners[b])))
        if off > _MIDSIDE_TOL * hmax:
            raise MortarTieError(
                f"mortar tie: {what} facet {tags} has a curved edge — "
                f"midside node {tag_mid} lies {off:.6g} off the midpoint "
                f"of edge ({tags[a]}, {tags[b]}), tolerance "
                f"{_MIDSIDE_TOL * hmax:.6g} (1e-6 x longest corner edge). "
                f"The kernel integrates on the corner polygon, which is "
                f"exact only for straight edges; a curved facet would be "
                f"silently mis-tied. Straighten the interface edges."
            )


def _refuse_master_self_overlap(m_facets: "list[_Facet]") -> None:
    """Fork ADR-78 R5.3 (MAJOR): masters may not overlap each other.

    The coverage guard sums pair-clip areas, so it counts
    **multiplicity**: a doubly-listed or mutually overlapping master
    facet can exactly mask an uncovered slave strip.  The masked strip
    is then silently *extrapolated* rather than left free, and neither
    coverage, nor the partition of unity (dual rowsums are identically
    1), nor even a linear-patch test can see it — a linear field
    extrapolates exactly.  Hence a pre-assembly refusal.

    The fork additionally gates on the pair's mean gap so that curved
    master surfaces, whose aux-plane shadows overlap harmlessly, stay
    legal.  Here the v1 flat + coincident scope has already forced every
    master into one plane, so a finite mutual clip *is* a real overlap
    and that sub-gate is vacuous.  Edge-adjacent facets clip to slivers
    and fall under the tolerance naturally.
    """
    for i, mi in enumerate(m_facets):
        for mj in m_facets[i + 1:]:
            if (mj.bb_lo > mi.bb_hi).any() or (mj.bb_hi < mi.bb_lo).any():
                continue
            ov = _clip_convex(mi.poly, mj.poly)
            if len(ov) < 3:
                continue
            a_ov = abs(_poly_area(ov))
            a_ref = min(mi.area, mj.area)
            if a_ov > _SELF_OVERLAP_TOL * a_ref:
                raise MortarTieError(
                    f"mortar tie: master facets {mi.tags} and {mj.tags} "
                    f"overlap each other over {a_ov / a_ref:.4%} of the "
                    f"smaller facet. Overlapping masters inflate the "
                    f"coverage check by multiplicity, which can hide an "
                    f"uncovered slave strip that then gets extrapolated "
                    f"instead of tied — no downstream guard can detect "
                    f"that. Check for duplicated facets or a master "
                    f"selection that picks up two layers of surface."
                )


def _facet_normal(corners: ndarray) -> ndarray:
    """Unit normal from corner winding (fork ``LadrunoTie.cpp:644-649``)."""
    a1 = corners[1] - corners[0]
    a2 = corners[-1] - corners[0]
    n = np.cross(a1, a2)
    norm = float(np.linalg.norm(n))
    if norm < 1e-300:
        raise MortarTieError("mortar tie: degenerate facet (zero normal).")
    return n / norm


# ── Public entry point ──────────────────────────────────────────────

def compute_dual_mortar_rows(
    slave_faces: ndarray,
    master_faces: ndarray,
    coords_of: Callable[[int], ndarray],
    *,
    gap_tol: float,
    outward: "tuple[float, float, float] | None" = None,
) -> list[tuple[int, list[int], ndarray]]:
    """Compute dual-mortar coupling rows ``u_s = Σ_k P_k · u_m_k``.

    Parameters
    ----------
    slave_faces, master_faces : ndarray (n_facets, nodes_per_facet)
        Facet connectivity as node tags. Mixed arities across the two
        arrays are fine; within one array the arity is the column count.
    coords_of : callable ``tag -> (3,) ndarray``
    gap_tol : float
        Absolute out-of-plane tolerance — every interface node must lie
        within this distance of the common plane (coincidence + flatness
        in one check). Unit-sensitive, like ``TieDef.tolerance``.
    outward : optional direction override for the plane normal, mirrors
        the fork's ``-outward`` (only needed when facet winding cancels).

    Returns
    -------
    list of ``(slave_node_tag, master_node_tags, weights)`` rows, one
    per unique slave node, weights partition-of-unity checked.

    Raises
    ------
    MortarTieError
        Non-planar/non-coincident interface, ambiguous normal,
        non-convex or degenerate facets, curved edges (a midside off
        its edge midpoint), tri6 slave facets, master facets
        overlapping one another, coverage gaps, or a failed
        partition-of-unity check.
    """
    slave_faces = np.asarray(slave_faces, dtype=np.int64)
    master_faces = np.asarray(master_faces, dtype=np.int64)
    if slave_faces.ndim != 2 or master_faces.ndim != 2:
        raise MortarTieError("mortar tie: facet arrays must be 2-D.")
    if slave_faces.shape[1] == 6:
        raise MortarTieError(
            "mortar tie: tri6 SLAVE facets are unsupported — the tri6 "
            "corner functions integrate to zero, so the dual row-sum "
            "scaling is singular (ADR 0086). Swap the sides (tri6 is "
            "fine as master) or use method='collocation'."
        )
    _corner_count(slave_faces.shape[1])
    _corner_count(master_faces.shape[1])

    # ── plane from master facet winding (or outward override) ──────
    m_corners3d = []
    for row in master_faces:
        ncor = _corner_count(len(row))
        m_corners3d.append(
            np.array([coords_of(int(t)) for t in row[:ncor]])
        )
    normals = np.array([_facet_normal(c) for c in m_corners3d])
    if outward is not None:
        n0 = np.asarray(outward, dtype=float)
        norm = float(np.linalg.norm(n0))
        if norm < 1e-300:
            raise MortarTieError("mortar tie: outward= is a zero vector.")
        n0 = n0 / norm
    else:
        acc = normals.sum(axis=0)
        if float(np.linalg.norm(acc)) < 1e-6 * len(normals):
            raise MortarTieError(
                "mortar tie: master facet normals cancel (closed or "
                "folded surface) — the interface plane is ambiguous. "
                "Pass outward=(ox, oy, oz) to disambiguate."
            )
        n0 = acc / float(np.linalg.norm(acc))
    bad = np.abs(normals @ n0) < _NORMAL_COHERENCE
    if bad.any():
        raise MortarTieError(
            f"mortar tie: {int(bad.sum())} master facet(s) deviate from "
            f"the common plane normal (min |cos| = "
            f"{float(np.min(np.abs(normals @ n0))):.4f} < "
            f"{_NORMAL_COHERENCE}) — v1 scope is a FLAT interface "
            f"(ADR 0086 D3)."
        )

    # ── coincidence: every interface node within gap_tol of the plane ─
    all_tags = sorted(
        {int(t) for t in slave_faces.ravel()}
        | {int(t) for t in master_faces.ravel()}
    )
    xyz = {t: np.asarray(coords_of(t), dtype=float) for t in all_tags}
    p0 = np.mean([xyz[t] for t in all_tags], axis=0)
    offsets = {t: float((xyz[t] - p0) @ n0) for t in all_tags}
    worst = max(offsets.items(), key=lambda kv: abs(kv[1]))
    if abs(worst[1]) > gap_tol:
        raise MortarTieError(
            f"mortar tie: interface is not coincident-flat — node "
            f"{worst[0]} lies {abs(worst[1]):.6g} off the common plane "
            f"(tolerance {gap_tol:.6g}). v1 scope is a coincident flat "
            f"interface (ADR 0086 D3); increase tolerance= only if the "
            f"gap is a genuine mesh artefact."
        )

    # ── straight edges: the corner-polygon geometry depends on it ───
    for row in slave_faces:
        _check_midside_nodes([int(t) for t in row], xyz, "slave")
    for row in master_faces:
        _check_midside_nodes([int(t) for t in row], xyz, "master")

    # ── 2-D plane basis ─────────────────────────────────────────────
    seed = np.array([1.0, 0.0, 0.0])
    if abs(float(seed @ n0)) > 0.9:
        seed = np.array([0.0, 1.0, 0.0])
    e1 = seed - (seed @ n0) * n0
    e1 /= float(np.linalg.norm(e1))
    e2 = np.cross(n0, e1)

    def _to2d(tags) -> ndarray:
        return np.array(
            [[(xyz[int(t)] - p0) @ e1, (xyz[int(t)] - p0) @ e2]
             for t in tags]
        )

    s_facets = [_Facet(row, _to2d(row), "slave") for row in slave_faces]
    m_facets = [_Facet(row, _to2d(row), "master") for row in master_faces]
    _refuse_master_self_overlap(m_facets)

    # ── integrate per slave facet, accumulate dual rows ─────────────
    numer: dict[int, dict[int, float]] = {}
    denom: dict[int, float] = {}

    for sf in s_facets:
        shape_s = SHAPE_FUNCTIONS[sf.nps]
        d_mat = np.zeros((sf.nps, sf.nps))
        m_cols: dict[int, ndarray] = {}       # master tag -> ∫ N_s · φ_m
        covered = 0.0

        for mf in m_facets:
            if (mf.bb_lo > sf.bb_hi).any() or (mf.bb_hi < sf.bb_lo).any():
                continue
            overlap = _clip_convex(sf.poly, mf.poly)
            if len(overlap) < 3:
                continue
            o_area = abs(_poly_area(overlap))
            if o_area < _EPS_AREA_REL * sf.area:
                continue
            shape_m = SHAPE_FUNCTIONS[mf.nps]
            centroid = overlap.mean(axis=0)
            n_v = len(overlap)
            for i in range(n_v):
                tri = np.array([centroid, overlap[i], overlap[(i + 1) % n_v]])
                a2 = _poly_area(tri) * 2.0     # signed, CCW ⇒ positive
                if abs(a2) < 1e-300:
                    continue
                for (ru, rv), w in zip(_TRI_PTS, _TRI_WTS):
                    p = (tri[0] + ru * (tri[1] - tri[0])
                         + rv * (tri[2] - tri[0]))
                    wj = w * a2
                    xi_s = _invert_corner_map(sf.corners2d, p)
                    n_s = shape_s(*xi_s)
                    xi_m = _invert_corner_map(mf.corners2d, p)
                    n_m = shape_m(*xi_m)
                    d_mat += wj * np.outer(n_s, n_s)
                    covered += wj
                    for k, tag_m in enumerate(mf.tags):
                        col = m_cols.get(tag_m)
                        if col is None:
                            col = np.zeros(sf.nps)
                            m_cols[tag_m] = col
                        col += wj * n_s * float(n_m[k])

        # coverage guard — area-based (sign-safe for quadratic bases,
        # unlike the fork's rowsum ratio which assumes positive N).
        if covered < (1.0 - _COVERAGE_TOL) * sf.area:
            raise MortarTieError(
                f"mortar tie: slave facet {sf.tags} is only "
                f"{covered / sf.area:.4%} covered by the master surface "
                f"— the tie would leave part of the slave surface "
                f"unbonded. Check the surface selections overlap fully."
            )

        # per-facet dual condensation: P^e = (D^e)⁻¹ M^e, accumulated
        # with the facet row-sums as weights (fork -dual route).
        rowsum = d_mat.sum(axis=1)
        try:
            p_e = np.linalg.solve(d_mat, np.column_stack(
                [m_cols[t] for t in sorted(m_cols)]))
        except np.linalg.LinAlgError as exc:
            raise MortarTieError(
                f"mortar tie: singular slave mass matrix on facet "
                f"{sf.tags} ({exc}) — degenerate overlap geometry."
            ) from exc
        m_tags_sorted = sorted(m_cols)
        for i, tag_s in enumerate(sf.tags):
            drow = denom.get(tag_s, 0.0)
            denom[tag_s] = drow + float(rowsum[i])
            nrow = numer.setdefault(tag_s, {})
            for j, tag_m in enumerate(m_tags_sorted):
                nrow[tag_m] = nrow.get(tag_m, 0.0) + \
                    float(rowsum[i]) * float(p_e[i, j])

    if not numer:
        raise MortarTieError(
            "mortar tie: no slave/master facet overlaps found — the two "
            "surfaces do not intersect in the interface plane."
        )

    # ── build rows: near-zero-row guard + PoU + weight filter ───────
    max_denom = max(abs(v) for v in denom.values())
    rows: list[tuple[int, list[int], ndarray]] = []
    for tag_s in sorted(numer):
        d = denom[tag_s]
        if abs(d) < 1e-6 * max_denom:
            raise MortarTieError(
                f"mortar tie: slave node {tag_s} accumulated a near-zero "
                f"dual row (|{d:.3e}| < 1e-6·{max_denom:.3e}) — cannot "
                f"form a well-conditioned coupling."
            )
        items = [(t, w / d) for t, w in numer[tag_s].items()]
        pou = sum(w for _, w in items)
        if abs(pou - 1.0) > _POU_TOL:
            raise MortarTieError(
                f"mortar tie: partition-of-unity failure on slave node "
                f"{tag_s} (Σw = {pou:.8f}) — near-singular dual system "
                f"(fork guard parity, |Σw−1| ≤ {_POU_TOL})."
            )
        items = [(t, w) for t, w in items if abs(w) > _W_FILTER]
        items.sort()
        rows.append((
            int(tag_s),
            [int(t) for t, _ in items],
            np.array([w for _, w in items], dtype=float),
        ))
    return rows
