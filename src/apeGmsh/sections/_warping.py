"""Warping (Saint-Venant) analysis for the section analyzer (ADR 0078 S2).

Formulation: Pilkey (2002) as implemented by the reference
``sectionproperties`` package (theory docs transcription, 2026-07-16),
in centroidal coordinates:

- **Torsion** ω: ``K ω = f`` with the **per-material G-weighted**
  Laplacian — exact for heterogeneous shear modulus, which is what makes
  the ``SectionMaterial(G=)`` override physically meaningful.
  ``GJ = ∫G(x²+y²)dA − ωᵀKω``.
- **Shear** Ψ/Φ: the package's composite convention — **E-weighted**
  integrals with a single effective ``ν_eff = EA/(2·GA) − 1``.
  Identical to the exact treatment when ν is uniform; documented
  approximation otherwise.
- Pure-Neumann singularity regularized by a **Lagrange row**
  ``∫N dA`` (enforces ∫ω dA = 0) — never node pinning.

Disconnected policy (ADR): ``"raise"`` fails loud with the component
count; ``"sum"`` solves per connected component and combines
(``GJ = ΣGJᵢ``, ``GAs = ΣGAsᵢ``, GJ-weighted shear centre; the
classical equal-twist-rate / no-inter-part-shear-transfer lower bound).
"""
from __future__ import annotations

import math
import warnings
from dataclasses import dataclass, fields, replace

import numpy as np
from numpy import ndarray

from ._errors import (
    CompositeSectionError,
    SectionAccuracyWarning,
    SectionAnalysisError,
)
from ._fe import block_quadrature
from ._geometric import GeometricProperties
from ._snapshot import LINEAR_2D_CODES, SectionSnapshot, _Block


# --------------------------------------------------------------------- #
# Result object                                                          #
# --------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class WarpingProperties:
    """Warping / shear results in authoring axes.

    Rigidity-form fields (``GJ``, ``EGamma``, ``GAs_*``) are always
    valid.  ``J`` / ``Gamma`` / ``As_*`` divide by the single modulus
    (``CompositeSectionError`` on composites; use :meth:`transformed`).
    ``alpha_x`` / ``alpha_y`` are rigidity *ratios* — reference-free.

    ``GAs_xy`` is the shear-coupling term ``Δ²/κ_xy``: for (near-)
    symmetric sections ``κ_xy → 0`` and the value diverges — that IS
    the answer ("no coupling"), same convention as the reference
    package; compare couplings via ``1/GAs_xy``, never the raw value.
    """

    # shear centre, authoring axes (length units).  Modulus-weighted
    # for composites — it moves toward the stiffer material, as it
    # should — but never *divided* by a modulus, so always valid.
    x_sc: float
    y_sc: float
    x_sc_t: float                   # Trefftz
    y_sc_t: float

    # rigidity form — always valid
    GJ: float
    EGamma: float                   # warping rigidity
    GA: float                       # ∫G dA (denominator of the alphas)
    GAs_x: float
    GAs_y: float
    GAs_xy: float

    # monosymmetry constants (lengths) — always valid
    beta_x_plus: float
    beta_x_minus: float
    beta_y_plus: float
    beta_y_minus: float
    beta_11_plus: float
    beta_11_minus: float
    beta_22_plus: float
    beta_22_minus: float

    nu_eff: float                   # EA/(2·GA) − 1

    # bookkeeping
    e_ref: float | None             # single E; None = composite
    g_ref: float | None             # single G; None = composite
    parts: tuple["WarpingProperties", ...] = ()

    _G_FIELDS = ("GJ", "GA", "GAs_x", "GAs_y", "GAs_xy")
    _E_FIELDS = ("EGamma",)

    # ── naming law ───────────────────────────────────────────────────

    def _by_g(self, field: str) -> float:
        if self.g_ref is None:
            raise CompositeSectionError(
                f"{field[1:]} is undefined on a composite section — read "
                f"the rigidity form ({field}) or pick reference moduli via "
                f"transformed(e_ref=..., g_ref=...)."
            )
        return getattr(self, field) / self.g_ref

    @property
    def J(self) -> float: return self._by_g("GJ")
    @property
    def As_x(self) -> float: return self._by_g("GAs_x")
    @property
    def As_y(self) -> float: return self._by_g("GAs_y")
    @property
    def As_xy(self) -> float: return self._by_g("GAs_xy")

    @property
    def Gamma(self) -> float:
        if self.e_ref is None:
            raise CompositeSectionError(
                "Gamma is undefined on a composite section — read EGamma "
                "or pick reference moduli via transformed(e_ref=..., "
                "g_ref=...)."
            )
        return self.EGamma / self.e_ref

    # shear-area factors are rigidity ratios — valid in every mode
    @property
    def alpha_x(self) -> float: return self.GAs_x / self.GA
    @property
    def alpha_y(self) -> float: return self.GAs_y / self.GA

    def transformed(
        self, *, e_ref: float, g_ref: float
    ) -> "WarpingProperties":
        """Rigidities divided by explicit reference moduli; unprefixed
        accessors are valid on the result."""
        if not (e_ref > 0.0 and g_ref > 0.0):
            raise ValueError(
                f"transformed: e_ref and g_ref must be > 0, got "
                f"({e_ref}, {g_ref})."
            )
        updates: dict[str, object] = {
            f: getattr(self, f) / g_ref for f in self._G_FIELDS
        }
        updates |= {f: getattr(self, f) / e_ref for f in self._E_FIELDS}
        # propagate into per-part results so GJ = Σ parts[i].GJ keeps
        # holding on the transformed view
        if self.parts:
            updates["parts"] = tuple(
                p.transformed(e_ref=e_ref, g_ref=g_ref) for p in self.parts
            )
        return replace(self, e_ref=1.0, g_ref=1.0, **updates)

    def _repr_html_(self) -> str:  # pragma: no cover - inspected visually
        rows = []
        for f in fields(self):
            if f.name in ("e_ref", "g_ref", "parts"):
                continue
            v = getattr(self, f.name)
            rows.append(f"<tr><td><code>{f.name}</code></td>"
                        f"<td style='text-align:right'>{v:.6g}</td></tr>")
        mode = ("composite — rigidity form only"
                if self.g_ref is None else
                f"single moduli E={self.e_ref:g}, G={self.g_ref:g}")
        extra = f", {len(self.parts)} parts" if self.parts else ""
        return (
            f"<b>WarpingProperties</b> <i>({mode}{extra})</i>"
            "<table><tr><th>field</th><th>value</th></tr>"
            + "".join(rows) + "</table>"
        )


@dataclass(frozen=True, slots=True)
class _PartSolution:
    """Nodal solution of one connected part — retained for S4 stress
    recovery.  Local node indexing (rows into ``coords``); ``node_rows``
    maps back into the snapshot's global rows."""

    node_rows: ndarray               # (n_local,) global row indices
    coords: ndarray                  # (n_local, 2) authoring axes
    blocks: tuple[_Block, ...]       # local-indexed connectivity
    omega: ndarray                   # (n_local,)
    psi: ndarray
    phi: ndarray
    cx: float                        # part elastic centroid, authoring axes
    cy: float
    EIxx: float                      # part-centroidal E-weighted moments
    EIyy: float
    EIxy: float
    EA: float
    GA: float
    nu_eff: float
    delta_s: float
    GJ: float


# --------------------------------------------------------------------- #
# Entry point                                                            #
# --------------------------------------------------------------------- #


def compute_warping(
    snap: SectionSnapshot,
    geo: GeometricProperties,
    *,
    policy: str,
    handle: str,
) -> tuple[WarpingProperties, tuple[_PartSolution, ...]]:
    if any(b.code in LINEAR_2D_CODES for b in snap.blocks):
        warnings.warn(
            f"{handle}: warping analysis on linear elements (tri3/quad4) "
            f"converges poorly on J and the shear areas — raise the mesh "
            f"to second order with g.mesh.generation.set_order(2).",
            SectionAccuracyWarning,
            stacklevel=3,
        )

    if snap.n_components == 1:
        props, sol = _solve_part(snap, np.arange(len(snap.coords)), parts=())
        return props, (sol,)

    if policy == "raise":
        raise SectionAnalysisError(
            f"{handle}: the section mesh has {snap.n_components} "
            f"disconnected parts — warping needs a connected domain. If "
            f"the parts should touch, fragment the faces so the mesh is "
            f"conformal; if the disconnection is intentional (twin "
            f"girders, spaced boxes), construct SectionProperties with "
            f"disconnected='sum' for the per-part lower bound."
        )

    # ── disconnected="sum": per-component solves, ADR combination ─────
    part_results: list[WarpingProperties] = []
    part_solutions: list[_PartSolution] = []
    for c in range(snap.n_components):
        node_rows = np.flatnonzero(snap.node_component == c)
        p, s = _solve_part(snap, node_rows, parts=())
        part_results.append(p)
        part_solutions.append(s)

    GJ = sum(p.GJ for p in part_results)
    GA = sum(p.GA for p in part_results)
    EA = geo.EA
    nu_eff = EA / (2.0 * GA) - 1.0
    x_sc = sum(p.GJ * p.x_sc for p in part_results) / GJ
    y_sc = sum(p.GJ * p.y_sc for p in part_results) / GJ
    x_sc_t = sum(p.GJ * p.x_sc_t for p in part_results) / GJ
    y_sc_t = sum(p.GJ * p.y_sc_t for p in part_results) / GJ

    # monosymmetry: whole-section integrals about the global centroid
    # with the combined shear centre (connectivity-blind — no solve).
    betas = _monosymmetry(snap, geo, x_sc=x_sc, y_sc=y_sc)

    single = snap.single_moduli
    combined = WarpingProperties(
        x_sc=x_sc,
        y_sc=y_sc,
        x_sc_t=x_sc_t,
        y_sc_t=y_sc_t,
        GJ=GJ,
        EGamma=sum(p.EGamma for p in part_results),
        GA=GA,
        GAs_x=sum(p.GAs_x for p in part_results),
        GAs_y=sum(p.GAs_y for p in part_results),
        GAs_xy=sum(p.GAs_xy for p in part_results),
        **betas,
        nu_eff=nu_eff,
        e_ref=single[0] if single else None,
        g_ref=single[1] if single else None,
        parts=tuple(part_results),
    )
    return combined, tuple(part_solutions)


# --------------------------------------------------------------------- #
# Single-part solve                                                      #
# --------------------------------------------------------------------- #


def _restrict_blocks(
    snap: SectionSnapshot, node_rows: ndarray
) -> tuple[list[_Block], ndarray]:
    """Blocks restricted to the given nodes, re-indexed 0..n_local-1."""
    local = -np.ones(len(snap.coords), dtype=np.int64)
    local[node_rows] = np.arange(len(node_rows))
    out: list[_Block] = []
    for b in snap.blocks:
        keep = np.all(local[b.conn] >= 0, axis=1)
        if not keep.any():
            continue
        out.append(
            _Block(
                code=b.code,
                type_name=b.type_name,
                n_corners=b.n_corners,
                eids=b.eids[keep],
                conn=local[b.conn[keep]],
                mat_idx=b.mat_idx[keep],
            )
        )
    return out, snap.coords[node_rows]


def _solve_part(
    snap: SectionSnapshot,
    node_rows: ndarray,
    *,
    parts: tuple[WarpingProperties, ...],
) -> tuple[WarpingProperties, _PartSolution]:
    from scipy.sparse import bmat, coo_matrix, csc_matrix
    from scipy.sparse.linalg import splu

    blocks, coords = _restrict_blocks(snap, node_rows)
    n = len(coords)
    E_by_mat = np.array([m.E for m in snap.materials])
    G_by_mat = np.array([m.shear_modulus for m in snap.materials])

    # ── pass 1: part centroid (E-weighted) ────────────────────────────
    quads = [
        block_quadrature(b, coords, centroid=(0.0, 0.0)) for b in blocks
    ]
    EA = EQx = EQy = GA = 0.0
    for q in quads:
        Ee = E_by_mat[q.block.mat_idx][:, None]
        Ge = G_by_mat[q.block.mat_idx][:, None]
        EA += float((Ee * q.wdetj).sum())
        GA += float((Ge * q.wdetj).sum())
        EQx += float((Ee * q.wdetj * q.y).sum())
        EQy += float((Ee * q.wdetj * q.x).sum())
    cx, cy = EQy / EA, EQx / EA
    # translation only moves the IP coordinates
    quads = [replace(q, x=q.x - cx, y=q.y - cy) for q in quads]

    nu_eff = EA / (2.0 * GA) - 1.0

    # ── E-weighted centroidal second moments (part) ───────────────────
    EIxx = EIyy = EIxy = 0.0
    for q in quads:
        wE = E_by_mat[q.block.mat_idx][:, None] * q.wdetj
        EIxx += float((wE * q.y * q.y).sum())
        EIyy += float((wE * q.x * q.x).sum())
        EIxy += float((wE * q.x * q.y).sum())

    # ── assemble stiffness (G- and E-weighted) + load vectors ─────────
    rows_l, cols_l, kG_l, kE_l = [], [], [], []
    f_wG = np.zeros(n)      # torsion RHS, G-weighted (the ω solve)
    f_wE = np.zeros(n)      # torsion RHS, E-weighted (shear-centre formula)
    f_psi = np.zeros(n)
    f_phi = np.zeros(n)
    c_vec = np.zeros(n)     # Lagrange row: ∫N dA
    polar_G = 0.0

    for q in quads:
        b = q.block
        Ee = E_by_mat[b.mat_idx]
        Ge = G_by_mat[b.mat_idx]
        wE = Ee[:, None] * q.wdetj                          # (E, n_ip)
        wG = Ge[:, None] * q.wdetj
        polar_G += float((wG * (q.x**2 + q.y**2)).sum())

        # stiffness: k_ab = Σ_ip w · B_ja B_jb
        BtB = np.einsum("eija,eijb->eab", q.B * wG[:, :, None, None], q.B)
        BtB_E = np.einsum("eija,eijb->eab", q.B * wE[:, :, None, None], q.B)
        npe = b.conn.shape[1]
        r_idx = np.repeat(b.conn, npe, axis=1).ravel()
        c_idx = np.tile(b.conn, (1, npe)).ravel()
        rows_l.append(r_idx)
        cols_l.append(c_idx)
        kG_l.append(BtB.ravel())
        kE_l.append(BtB_E.ravel())

        # torsion RHS: f_a = Σ w (B_xa·y − B_ya·x)
        tor = q.B[:, :, 0, :] * q.y[:, :, None] - q.B[:, :, 1, :] * q.x[:, :, None]
        np.add.at(f_wG, b.conn.ravel(),
                  (tor * wG[:, :, None]).sum(axis=1).ravel())
        np.add.at(f_wE, b.conn.ravel(),
                  (tor * wE[:, :, None]).sum(axis=1).ravel())

        # shear RHS (E-weighted, ν_eff):
        r = q.x**2 - q.y**2
        s2 = 2.0 * q.x * q.y
        d1 = EIxx * r - EIxy * s2
        d2 = EIxy * r + EIxx * s2
        h1 = -EIxy * r + EIyy * s2
        h2 = -EIyy * r - EIxy * s2
        lin_psi = EIxx * q.x - EIxy * q.y
        lin_phi = EIyy * q.y - EIxy * q.x
        term_psi = (
            0.5 * nu_eff * (q.B[:, :, 0, :] * d1[:, :, None]
                            + q.B[:, :, 1, :] * d2[:, :, None])
            + 2.0 * (1.0 + nu_eff) * q.N[None, :, :] * lin_psi[:, :, None]
        )
        term_phi = (
            0.5 * nu_eff * (q.B[:, :, 0, :] * h1[:, :, None]
                            + q.B[:, :, 1, :] * h2[:, :, None])
            + 2.0 * (1.0 + nu_eff) * q.N[None, :, :] * lin_phi[:, :, None]
        )
        np.add.at(f_psi, b.conn.ravel(),
                  (term_psi * wE[:, :, None]).sum(axis=1).ravel())
        np.add.at(f_phi, b.conn.ravel(),
                  (term_phi * wE[:, :, None]).sum(axis=1).ravel())

        np.add.at(c_vec, b.conn.ravel(),
                  (q.N[None, :, :] * q.wdetj[:, :, None]).sum(axis=1).ravel())

    rows = np.concatenate(rows_l)
    cols = np.concatenate(cols_l)
    KG = coo_matrix((np.concatenate(kG_l), (rows, cols)), shape=(n, n))
    KE = coo_matrix((np.concatenate(kE_l), (rows, cols)), shape=(n, n))
    c_col = csc_matrix(c_vec[:, None])

    # ω from the G-weighted (exact) torsion system; Ψ/Φ from the
    # E-weighted / ν_eff shear system (package convention).
    A_G = bmat([[KG.tocsc(), c_col], [c_col.T, None]], format="csc")
    lu_G = splu(A_G)
    omega = lu_G.solve(np.concatenate([f_wG, [0.0]]))[:n]

    A_E = bmat([[KE.tocsc(), c_col], [c_col.T, None]], format="csc")
    lu_E = splu(A_E)
    psi = lu_E.solve(np.concatenate([f_psi, [0.0]]))[:n]
    phi = lu_E.solve(np.concatenate([f_phi, [0.0]]))[:n]

    GJ = polar_G - float(omega @ (KG @ omega))

    # ── post-integrals ────────────────────────────────────────────────
    delta_s = 2.0 * (1.0 + nu_eff) * (EIxx * EIyy - EIxy**2)
    sc_x_int = sc_y_int = 0.0
    kap_x = kap_y = kap_xy = 0.0
    Q_om = I_om = I_xom = I_yom = 0.0
    int_x2y_y3 = int_xy2_x3 = 0.0
    int_11 = int_22 = 0.0

    # principal rotation for the 11/22 monosymmetry constants
    delta_I = EIxx - EIyy
    theta = 0.5 * math.atan2(-2.0 * EIxy, delta_I)
    ct, st = math.cos(theta), math.sin(theta)

    for q in quads:
        b = q.block
        Ee = E_by_mat[b.mat_idx]
        wE = Ee[:, None] * q.wdetj

        om_e = omega[b.conn]                                # (E, npe)
        psi_e = psi[b.conn]
        phi_e = phi[b.conn]
        om_ip = np.einsum("ia,ea->ei", q.N, om_e)
        r = q.x**2 - q.y**2
        s2 = 2.0 * q.x * q.y
        d1 = EIxx * r - EIxy * s2
        d2 = EIxy * r + EIxx * s2
        h1 = -EIxy * r + EIyy * s2
        h2 = -EIyy * r - EIxy * s2

        # shear-centre volume integrals
        rho2 = q.x**2 + q.y**2
        sc_x_int += float((wE * (EIyy * q.x + EIxy * q.y) * rho2).sum())
        sc_y_int += float((wE * (EIxx * q.y + EIxy * q.x) * rho2).sum())

        # κ integrands: (BΨ − ν/2 d)·(BΨ − ν/2 d) etc.
        Bpsi = np.einsum("eija,ea->eij", q.B, psi_e)        # (E, n_ip, 2)
        Bphi = np.einsum("eija,ea->eij", q.B, phi_e)
        dv = np.stack([d1, d2], axis=-1)
        hv = np.stack([h1, h2], axis=-1)
        ax = Bpsi - 0.5 * nu_eff * dv
        ay = Bphi - 0.5 * nu_eff * hv
        kap_x += float((wE * np.einsum("eij,eij->ei", ax, ax)).sum())
        kap_y += float((wE * np.einsum("eij,eij->ei", ay, ay)).sum())
        kap_xy += float((wE * np.einsum("eij,eij->ei", ax, ay)).sum())

        # warping-constant integrals
        Q_om += float((wE * om_ip).sum())
        I_om += float((wE * om_ip**2).sum())
        I_xom += float((wE * q.x * om_ip).sum())
        I_yom += float((wE * q.y * om_ip).sum())

        # monosymmetry integrals (part axes + principal axes)
        int_x2y_y3 += float((wE * (q.x**2 * q.y + q.y**3)).sum())
        int_xy2_x3 += float((wE * (q.x * q.y**2 + q.x**3)).sum())
        x1 = q.x * ct + q.y * st
        y1 = -q.x * st + q.y * ct
        int_11 += float((wE * (x1**2 * y1 + y1**3)).sum())
        int_22 += float((wE * (x1 * y1**2 + x1**3)).sum())

    x_s = (0.5 * nu_eff * sc_x_int - float(f_wE @ phi)) / delta_s
    y_s = (0.5 * nu_eff * sc_y_int + float(f_wE @ psi)) / delta_s

    D = EIxx * EIyy - EIxy**2
    x_s_t = (EIxy * I_xom - EIyy * I_yom) / D
    y_s_t = (EIxx * I_xom - EIxy * I_yom) / D

    EGamma = I_om - Q_om**2 / EA - y_s * I_xom + x_s * I_yom

    EAs_x = delta_s**2 / kap_x
    EAs_y = delta_s**2 / kap_y
    EAs_xy = delta_s**2 / kap_xy
    scale = 1.0 / (2.0 * (1.0 + nu_eff))    # E-weighted → G-weighted
    GAs_x = EAs_x * scale
    GAs_y = EAs_y * scale
    GAs_xy = EAs_xy * scale

    beta_x = int_x2y_y3 / EIxx - 2.0 * y_s
    beta_y = int_xy2_x3 / EIyy - 2.0 * x_s
    # principal I and rotated shear centre
    h = math.hypot(delta_I, 2.0 * EIxy)
    EI11 = 0.5 * (EIxx + EIyy + h)
    EI22 = 0.5 * (EIxx + EIyy - h)
    y_s1 = -x_s * st + y_s * ct
    x_s1 = x_s * ct + y_s * st
    beta_11 = int_11 / EI11 - 2.0 * y_s1
    beta_22 = int_22 / EI22 - 2.0 * x_s1

    single = snap.single_moduli
    props = WarpingProperties(
        x_sc=cx + x_s,
        y_sc=cy + y_s,
        x_sc_t=cx + x_s_t,
        y_sc_t=cy + y_s_t,
        GJ=GJ,
        EGamma=EGamma,
        GA=GA,
        GAs_x=GAs_x,
        GAs_y=GAs_y,
        GAs_xy=GAs_xy,
        beta_x_plus=beta_x,
        beta_x_minus=-beta_x,
        beta_y_plus=beta_y,
        beta_y_minus=-beta_y,
        beta_11_plus=beta_11,
        beta_11_minus=-beta_11,
        beta_22_plus=beta_22,
        beta_22_minus=-beta_22,
        nu_eff=nu_eff,
        e_ref=single[0] if single else None,
        g_ref=single[1] if single else None,
        parts=parts,
    )
    solution = _PartSolution(
        node_rows=node_rows,
        coords=coords,
        blocks=tuple(blocks),
        omega=omega,
        psi=psi,
        phi=phi,
        cx=cx,
        cy=cy,
        EIxx=EIxx,
        EIyy=EIyy,
        EIxy=EIxy,
        EA=EA,
        GA=GA,
        nu_eff=nu_eff,
        delta_s=delta_s,
        GJ=GJ,
    )
    return props, solution


# --------------------------------------------------------------------- #
# Whole-section monosymmetry (used by the "sum" combination)             #
# --------------------------------------------------------------------- #


def _monosymmetry(
    snap: SectionSnapshot,
    geo: GeometricProperties,
    *,
    x_sc: float,
    y_sc: float,
) -> dict[str, float]:
    """β from whole-section integrals about the global elastic centroid
    with the given (combined) shear centre — connectivity-blind."""
    E_by_mat = np.array([m.E for m in snap.materials])
    x_s = x_sc - geo.cx
    y_s = y_sc - geo.cy
    theta = math.radians(geo.phi)
    ct, st = math.cos(theta), math.sin(theta)

    int_x = int_y = int_11 = int_22 = 0.0
    for b in snap.blocks:
        q = block_quadrature(b, snap.coords, centroid=(geo.cx, geo.cy))
        wE = E_by_mat[b.mat_idx][:, None] * q.wdetj
        int_x += float((wE * (q.x**2 * q.y + q.y**3)).sum())
        int_y += float((wE * (q.x * q.y**2 + q.x**3)).sum())
        x1 = q.x * ct + q.y * st
        y1 = -q.x * st + q.y * ct
        int_11 += float((wE * (x1**2 * y1 + y1**3)).sum())
        int_22 += float((wE * (x1 * y1**2 + x1**3)).sum())

    beta_x = int_x / geo.EIxx_c - 2.0 * y_s
    beta_y = int_y / geo.EIyy_c - 2.0 * x_s
    y_s1 = -x_s * st + y_s * ct
    x_s1 = x_s * ct + y_s * st
    beta_11 = int_11 / geo.EI11_c - 2.0 * y_s1
    beta_22 = int_22 / geo.EI22_c - 2.0 * x_s1
    return {
        "beta_x_plus": beta_x, "beta_x_minus": -beta_x,
        "beta_y_plus": beta_y, "beta_y_minus": -beta_y,
        "beta_11_plus": beta_11, "beta_11_minus": -beta_11,
        "beta_22_plus": beta_22, "beta_22_minus": -beta_22,
    }
