"""Geometric analysis for the section analyzer (ADR 0078).

One Gauss loop over the snapshot accumulating the modulus-weighted
integrals; everything else (centroid, principal axes, section moduli,
perimeter, mass) derives from them.  No solve — pure quadrature.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, fields, replace

import numpy as np
from numpy import ndarray

from apeGmsh.fem._quadrature import gauss_quad_2d, gauss_tri
from apeGmsh.fem._shape_functions import (
    compute_jacobian_dets,
    compute_physical_coords,
    get_shape_functions,
)

from ._errors import CompositeSectionError, SectionMeshError
from ._snapshot import SectionSnapshot

_TRI_CODES = frozenset({2, 9})
_QUAD_GAUSS_N = 3  # 3×3 tensor rule — degree-5 exact per axis


def _rule_for(code: int) -> tuple[ndarray, ndarray]:
    if code in _TRI_CODES:
        return gauss_tri()
    return gauss_quad_2d(_QUAD_GAUSS_N)


# --------------------------------------------------------------------- #
# Result object                                                          #
# --------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class GeometricProperties:
    """Geometric (modulus-weighted) properties in authoring axes.

    Rigidity-form fields (``EA``, ``EIxx_c``, …) are always valid.  The
    unprefixed accessors (``Ixx_c``, ``Zxx_plus``, …) divide by the
    section's single modulus and raise :class:`CompositeSectionError`
    on a composite — pick a reference with :meth:`transformed`.
    """

    # ── pure geometry — valid in every mode ──────────────────────────
    area: float
    perimeter: float
    mass: float | None
    cx: float
    cy: float
    phi: float                      # principal-axis angle [deg], 11 = major

    # ── rigidity form — always valid ─────────────────────────────────
    EA: float
    EQx: float                      # ∫E·y dA (about global x)
    EQy: float                      # ∫E·x dA (about global y)
    EIxx_g: float
    EIyy_g: float
    EIxy_g: float
    EIxx_c: float
    EIyy_c: float
    EIxy_c: float
    EI11_c: float
    EI22_c: float
    EZxx_plus: float
    EZxx_minus: float
    EZyy_plus: float
    EZyy_minus: float
    EZ11_plus: float
    EZ11_minus: float
    EZ22_plus: float
    EZ22_minus: float

    # ── bookkeeping ──────────────────────────────────────────────────
    e_ref: float | None             # single modulus; None = composite
    material_areas: tuple[float, ...]

    # rigidity-field names the unprefixed law divides by ``e_ref``
    _E_FIELDS = (
        "EA", "EQx", "EQy",
        "EIxx_g", "EIyy_g", "EIxy_g",
        "EIxx_c", "EIyy_c", "EIxy_c",
        "EI11_c", "EI22_c",
        "EZxx_plus", "EZxx_minus", "EZyy_plus", "EZyy_minus",
        "EZ11_plus", "EZ11_minus", "EZ22_plus", "EZ22_minus",
    )

    # ── naming law ───────────────────────────────────────────────────

    def _hom(self, rigidity_field: str) -> float:
        if self.e_ref is None:
            raise CompositeSectionError(
                f"{rigidity_field[1:]} is undefined on a composite section — "
                f"read the rigidity form ({rigidity_field}) or pick an "
                f"explicit reference modulus via transformed(e_ref=...)."
            )
        return getattr(self, rigidity_field) / self.e_ref

    @property
    def Qx(self) -> float: return self._hom("EQx")
    @property
    def Qy(self) -> float: return self._hom("EQy")
    @property
    def Ixx_g(self) -> float: return self._hom("EIxx_g")
    @property
    def Iyy_g(self) -> float: return self._hom("EIyy_g")
    @property
    def Ixy_g(self) -> float: return self._hom("EIxy_g")
    @property
    def Ixx_c(self) -> float: return self._hom("EIxx_c")
    @property
    def Iyy_c(self) -> float: return self._hom("EIyy_c")
    @property
    def Ixy_c(self) -> float: return self._hom("EIxy_c")
    @property
    def I11_c(self) -> float: return self._hom("EI11_c")
    @property
    def I22_c(self) -> float: return self._hom("EI22_c")
    @property
    def Zxx_plus(self) -> float: return self._hom("EZxx_plus")
    @property
    def Zxx_minus(self) -> float: return self._hom("EZxx_minus")
    @property
    def Zyy_plus(self) -> float: return self._hom("EZyy_plus")
    @property
    def Zyy_minus(self) -> float: return self._hom("EZyy_minus")
    @property
    def Z11_plus(self) -> float: return self._hom("EZ11_plus")
    @property
    def Z11_minus(self) -> float: return self._hom("EZ11_minus")
    @property
    def Z22_plus(self) -> float: return self._hom("EZ22_plus")
    @property
    def Z22_minus(self) -> float: return self._hom("EZ22_minus")

    # Radii of gyration are E-ratio quantities — the reference modulus
    # cancels, so they are valid in every mode (transformed-section radii
    # for composites).
    @property
    def rx(self) -> float: return math.sqrt(self.EIxx_c / self.EA)
    @property
    def ry(self) -> float: return math.sqrt(self.EIyy_c / self.EA)
    @property
    def r11(self) -> float: return math.sqrt(self.EI11_c / self.EA)
    @property
    def r22(self) -> float: return math.sqrt(self.EI22_c / self.EA)

    def transformed(self, e_ref: float) -> "GeometricProperties":
        """Same shape with every rigidity divided by ``e_ref`` — the
        classic transformed-section view.  Unprefixed accessors are
        valid on the result."""
        if not e_ref > 0.0:
            raise ValueError(f"transformed: e_ref must be > 0, got {e_ref}.")
        updates = {f: getattr(self, f) / e_ref for f in self._E_FIELDS}
        return replace(self, e_ref=1.0, **updates)

    # ── display ──────────────────────────────────────────────────────

    def _repr_html_(self) -> str:  # pragma: no cover - inspected visually
        rows = []
        for f in fields(self):
            if f.name in ("e_ref", "material_areas"):
                continue
            v = getattr(self, f.name)
            if v is None:
                continue
            rows.append(f"<tr><td><code>{f.name}</code></td>"
                        f"<td style='text-align:right'>{v:.6g}</td></tr>")
        mode = ("composite — rigidity form only; use transformed(e_ref=…)"
                if self.e_ref is None else f"single modulus e_ref={self.e_ref:g}")
        return (
            "<b>GeometricProperties</b> "
            f"<i>({mode})</i>"
            "<table><tr><th>field</th><th>value</th></tr>"
            + "".join(rows) + "</table>"
        )


# --------------------------------------------------------------------- #
# Computation                                                            #
# --------------------------------------------------------------------- #


def compute_geometric(snap: SectionSnapshot) -> GeometricProperties:
    E_by_mat = np.array([m.E for m in snap.materials], dtype=np.float64)

    area = 0.0
    EA = EQx = EQy = 0.0
    EIxx = EIyy = EIxy = 0.0
    mat_areas = np.zeros(len(snap.materials), dtype=np.float64)

    coords3d_all = np.concatenate(
        [snap.coords, np.zeros((len(snap.coords), 1))], axis=1
    )

    for b in snap.blocks:
        pts, wts = _rule_for(b.code)
        shape = get_shape_functions(b.code)
        assert shape is not None  # gated in the snapshot
        N_fn, dN_fn, geom_kind, _ = shape

        elem_xyz = coords3d_all[b.conn]                        # (E, npe, 3)
        detj = compute_jacobian_dets(pts, elem_xyz, dN_fn, geom_kind)
        w = detj * wts[None, :]                                # (E, n_ip)
        ip = compute_physical_coords(pts, elem_xyz, N_fn)      # (E, n_ip, 3)
        x = ip[..., 0]
        y = ip[..., 1]
        Ee = E_by_mat[b.mat_idx][:, None]                      # (E, 1)

        elem_area = w.sum(axis=1)                              # (E,)
        area += float(elem_area.sum())
        np.add.at(mat_areas, b.mat_idx, elem_area)
        EA += float((Ee * w).sum())
        EQx += float((Ee * w * y).sum())
        EQy += float((Ee * w * x).sum())
        EIxx += float((Ee * w * y * y).sum())
        EIyy += float((Ee * w * x * x).sum())
        EIxy += float((Ee * w * x * y).sum())

    if not area > 0.0:
        raise SectionMeshError("section has zero area — degenerate mesh.")

    cx = EQy / EA
    cy = EQx / EA
    EIxx_c = EIxx - cy * cy * EA
    EIyy_c = EIyy - cx * cx * EA
    EIxy_c = EIxy - cx * cy * EA

    delta = EIxx_c - EIyy_c
    h = math.hypot(delta, 2.0 * EIxy_c)
    s = EIxx_c + EIyy_c
    EI11_c = 0.5 * (s + h)
    EI22_c = 0.5 * (s - h)
    theta = 0.5 * math.atan2(-2.0 * EIxy_c, delta)   # major-axis angle
    phi = math.degrees(theta)

    # extreme-fibre distances (straight sides → node coords suffice)
    dx = snap.coords[:, 0] - cx
    dy = snap.coords[:, 1] - cy
    ct, st = math.cos(theta), math.sin(theta)
    d_along_11 = dx * ct + dy * st          # coordinate along the 11 axis
    d_perp_11 = -dx * st + dy * ct          # distance driving M11 bending

    def _z(EI: float, plus: float, minus: float) -> tuple[float, float]:
        return EI / plus, EI / minus

    EZxx_plus, EZxx_minus = _z(EIxx_c, float(dy.max()), float(-dy.min()))
    EZyy_plus, EZyy_minus = _z(EIyy_c, float(dx.max()), float(-dx.min()))
    EZ11_plus, EZ11_minus = _z(
        EI11_c, float(d_perp_11.max()), float(-d_perp_11.min())
    )
    EZ22_plus, EZ22_minus = _z(
        EI22_c, float(d_along_11.max()), float(-d_along_11.min())
    )

    densities = [m.density for m in snap.materials]
    mass = (
        float(sum(d * a for d, a in zip(densities, mat_areas)))
        if all(d is not None for d in densities)
        else None
    )

    single = snap.single_moduli
    e_ref = single[0] if single is not None else None

    return GeometricProperties(
        area=area,
        perimeter=_exterior_perimeter(snap),
        mass=mass,
        cx=cx,
        cy=cy,
        phi=phi,
        EA=EA,
        EQx=EQx,
        EQy=EQy,
        EIxx_g=EIxx,
        EIyy_g=EIyy,
        EIxy_g=EIxy,
        EIxx_c=EIxx_c,
        EIyy_c=EIyy_c,
        EIxy_c=EIxy_c,
        EI11_c=EI11_c,
        EI22_c=EI22_c,
        EZxx_plus=EZxx_plus,
        EZxx_minus=EZxx_minus,
        EZyy_plus=EZyy_plus,
        EZyy_minus=EZyy_minus,
        EZ11_plus=EZ11_plus,
        EZ11_minus=EZ11_minus,
        EZ22_plus=EZ22_plus,
        EZ22_minus=EZ22_minus,
        e_ref=e_ref,
        material_areas=tuple(float(a) for a in mat_areas),
    )


# --------------------------------------------------------------------- #
# Perimeter — exterior boundary loops only (holes excluded)              #
# --------------------------------------------------------------------- #


def _exterior_perimeter(snap: SectionSnapshot) -> float:
    """Sum of exterior-loop lengths, one per connected component.

    Boundary edges are corner edges appearing in exactly one element.
    Loops are chained; per component, the loop with the largest
    enclosed |shoelace area| is the exterior (winding-agnostic) —
    every other loop is a hole and is excluded.
    """
    edge_count: dict[tuple[int, int], int] = {}
    edge_dir: dict[tuple[int, int], tuple[int, int]] = {}
    for b in snap.blocks:
        nc = b.n_corners
        corners = b.conn[:, :nc]
        for row in corners:
            for k in range(nc):
                a, c = int(row[k]), int(row[(k + 1) % nc])
                key = (a, c) if a < c else (c, a)
                edge_count[key] = edge_count.get(key, 0) + 1
                edge_dir[key] = (a, c)

    succ: dict[int, int] = {}
    for key, n in edge_count.items():
        if n != 1:
            continue
        a, c = edge_dir[key]
        if a in succ:
            raise SectionMeshError(
                "non-manifold section boundary (pinched vertex) — cannot "
                "walk boundary loops for the perimeter."
            )
        succ[a] = c

    coords = snap.coords
    comp = snap.node_component
    # component -> (best |area|, best length)
    best: dict[int, tuple[float, float]] = {}
    visited: set[int] = set()
    for start in list(succ):
        if start in visited:
            continue
        loop = [start]
        visited.add(start)
        nxt = succ[start]
        while nxt != start:
            loop.append(nxt)
            visited.add(nxt)
            nxt = succ[nxt]
        pts = coords[loop]
        rolled = np.roll(pts, -1, axis=0)
        area2 = float(np.abs(
            np.sum(pts[:, 0] * rolled[:, 1] - rolled[:, 0] * pts[:, 1])
        ))
        length = float(np.sum(np.linalg.norm(rolled - pts, axis=1)))
        c = int(comp[start])
        if c not in best or area2 > best[c][0]:
            best[c] = (area2, length)
    return float(sum(length for _, length in best.values()))
