"""Plastic analysis for the section analyzer (ADR 0078 S3).

Rigid-plastic assumption: every point carries ``±fy`` depending on its
side of the plastic neutral axis.  For each bending axis (centroidal
x / y and principal 11 / 22) the NA position is the **exact zero of the
discretized force balance** ``F(d) = ∫ fy·sign(s − d) dA`` — computed
as the fy-weighted median of the Gauss-point projections (the limit the
ADR's bisection converges to, without a bracketing failure mode).  The
plastic moment is then ``Mp = ∫ fy·|s − d*| dA``.

Shape factors are plastic-over-first-yield: ``sf = Mp / My`` with
``My = min over materials of fy_m · EI_c / (E_m · c_m)`` (the moment at
which the first fibre of any material yields).  For a homogeneous
section this reduces to the classic ``S / Z``.

Documented invalid for strain-softening / nonlinear materials (plain
concrete) — same caveat as the reference package.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, fields

import numpy as np
from numpy import ndarray

from ._errors import CompositeSectionError, SectionAnalysisError
from ._fe import block_quadrature
from ._geometric import GeometricProperties
from ._snapshot import SectionSnapshot


# --------------------------------------------------------------------- #
# Result object                                                          #
# --------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class PlasticProperties:
    """Plastic results in authoring axes.

    ``Mp_*`` (fy-weighted plastic moments) and the shape factors are
    always valid — for a composite the ``Mp_*`` ARE the plastic moment
    capacities.  The unprefixed plastic moduli ``Sxx``/``Syy``/``S11``/
    ``S22`` divide by the section's single ``fy`` and raise
    :class:`CompositeSectionError` when materials carry different yield
    stresses — use the ``Mp_*`` fields directly in that case.
    """

    # plastic neutral-axis positions
    x_pc: float                     # NA for bending about y (authoring x)
    y_pc: float                     # NA for bending about x (authoring y)
    x11_pc: float                   # NA position along 11 (bending about 22)
    y22_pc: float                   # NA position along 22 (bending about 11)

    # fy-weighted form — always valid (composite: these ARE Mp)
    Mp_xx: float
    Mp_yy: float
    Mp_11: float
    Mp_22: float
    sf_xx_plus: float
    sf_xx_minus: float
    sf_yy_plus: float
    sf_yy_minus: float
    sf_11_plus: float
    sf_11_minus: float
    sf_22_plus: float
    sf_22_minus: float

    # bookkeeping
    fy_ref: float | None            # single yield stress; None = mixed fy

    _FY_FIELDS = {
        "Sxx": "Mp_xx", "Syy": "Mp_yy", "S11": "Mp_11", "S22": "Mp_22",
    }

    def _by_fy(self, name: str) -> float:
        if self.fy_ref is None:
            raise CompositeSectionError(
                f"{name} is undefined when materials carry different yield "
                f"stresses — use the fy-weighted plastic moment "
                f"({self._FY_FIELDS[name]}) directly."
            )
        return getattr(self, self._FY_FIELDS[name]) / self.fy_ref

    @property
    def Sxx(self) -> float: return self._by_fy("Sxx")
    @property
    def Syy(self) -> float: return self._by_fy("Syy")
    @property
    def S11(self) -> float: return self._by_fy("S11")
    @property
    def S22(self) -> float: return self._by_fy("S22")

    def _repr_html_(self) -> str:  # pragma: no cover - inspected visually
        rows = []
        for f in fields(self):
            if f.name == "fy_ref":
                continue
            v = getattr(self, f.name)
            rows.append(f"<tr><td><code>{f.name}</code></td>"
                        f"<td style='text-align:right'>{v:.6g}</td></tr>")
        mode = ("mixed fy — use Mp_* directly"
                if self.fy_ref is None else f"single fy = {self.fy_ref:g}")
        return (
            f"<b>PlasticProperties</b> <i>({mode})</i>"
            "<table><tr><th>field</th><th>value</th></tr>"
            + "".join(rows) + "</table>"
        )


# --------------------------------------------------------------------- #
# Computation                                                            #
# --------------------------------------------------------------------- #


def compute_plastic(
    snap: SectionSnapshot,
    geo: GeometricProperties,
    *,
    handle: str,
) -> PlasticProperties:
    # ── fy gate ───────────────────────────────────────────────────────
    if snap.geometric_only:
        raise SectionAnalysisError(
            f"{handle}: plastic analysis needs yield stresses — construct "
            f"SectionProperties with materials= and give every "
            f"SectionMaterial an fy."
        )
    missing = [
        pg for pg, m in zip(snap.material_names, snap.materials)
        if m.fy is None
    ]
    if missing:
        raise SectionAnalysisError(
            f"{handle}: plastic analysis requires fy on every material — "
            f"missing on {missing}. (If a region is not meant to "
            f"contribute plastically, model that choice explicitly; note "
            f"the rigid-plastic assumption is invalid for strain-softening "
            f"materials such as plain concrete.)"
        )

    fy_by_mat = np.array([m.fy for m in snap.materials], dtype=np.float64)
    E_by_mat = np.array([m.E for m in snap.materials], dtype=np.float64)

    # ── gather Gauss-point projections (centroidal axes) ──────────────
    xs, ys, wfy = [], [], []
    for b in snap.blocks:
        q = block_quadrature(b, snap.coords, centroid=(geo.cx, geo.cy))
        fy_ip = fy_by_mat[b.mat_idx][:, None]
        xs.append(q.x.ravel())
        ys.append(q.y.ravel())
        wfy.append((fy_ip * q.wdetj).ravel())
    x = np.concatenate(xs)
    y = np.concatenate(ys)
    w = np.concatenate(wfy)
    if not w.sum() > 0.0:
        raise SectionAnalysisError(f"{handle}: zero plastic capacity.")

    theta = math.radians(geo.phi)
    ct, st = math.cos(theta), math.sin(theta)
    x1 = x * ct + y * st        # coordinate along the 11 axis
    y1 = -x * st + y * ct       # coordinate along the 22 direction

    def _na_and_mp(proj: ndarray) -> tuple[float, float]:
        """fy-weighted median of *proj* and the plastic moment about it."""
        order = np.argsort(proj, kind="stable")
        p = proj[order]
        cw = np.cumsum(w[order])
        half = 0.5 * cw[-1]
        k = int(np.searchsorted(cw, half))
        d = float(p[min(k, len(p) - 1)])
        mp = float(np.sum(w * np.abs(proj - d)))
        return d, mp

    y_na, Mp_xx = _na_and_mp(y)      # bending about x: NA at y = y_na
    x_na, Mp_yy = _na_and_mp(x)      # bending about y
    y22_na, Mp_11 = _na_and_mp(y1)   # bending about 11
    x11_na, Mp_22 = _na_and_mp(x1)   # bending about 22

    # ── first-yield moments per axis/side → shape factors ─────────────
    # per-material extreme fibre distances from node coords
    n_mats = len(snap.materials)
    node_x = snap.coords[:, 0] - geo.cx
    node_y = snap.coords[:, 1] - geo.cy
    node_x1 = node_x * ct + node_y * st
    node_y1 = -node_x * st + node_y * ct
    ext: dict[str, list[tuple[float, float]]] = {
        "y": [], "x": [], "y1": [], "x1": [],
    }
    for m in range(n_mats):
        node_mask = np.zeros(len(snap.coords), dtype=bool)
        for b in snap.blocks:
            sel = b.conn[b.mat_idx == m]
            if len(sel):
                node_mask[sel.ravel()] = True
        for key, arr in (("y", node_y), ("x", node_x),
                         ("y1", node_y1), ("x1", node_x1)):
            vals = arr[node_mask]
            ext[key].append((float(vals.max()), float(-vals.min())))

    def _sf(mp: float, EI: float, key: str) -> tuple[float, float]:
        my_plus = math.inf
        my_minus = math.inf
        for m in range(n_mats):
            c_plus, c_minus = ext[key][m]
            if c_plus > 0.0:
                my_plus = min(my_plus, fy_by_mat[m] * EI / (E_by_mat[m] * c_plus))
            if c_minus > 0.0:
                my_minus = min(
                    my_minus, fy_by_mat[m] * EI / (E_by_mat[m] * c_minus)
                )
        return mp / my_plus, mp / my_minus

    sf_xx_plus, sf_xx_minus = _sf(Mp_xx, geo.EIxx_c, "y")
    sf_yy_plus, sf_yy_minus = _sf(Mp_yy, geo.EIyy_c, "x")
    sf_11_plus, sf_11_minus = _sf(Mp_11, geo.EI11_c, "y1")
    sf_22_plus, sf_22_minus = _sf(Mp_22, geo.EI22_c, "x1")

    fy_set = {m.fy for m in snap.materials}
    fy_ref = next(iter(fy_set)) if len(fy_set) == 1 else None

    return PlasticProperties(
        x_pc=geo.cx + x_na,
        y_pc=geo.cy + y_na,
        x11_pc=x11_na,
        y22_pc=y22_na,
        Mp_xx=Mp_xx,
        Mp_yy=Mp_yy,
        Mp_11=Mp_11,
        Mp_22=Mp_22,
        sf_xx_plus=sf_xx_plus,
        sf_xx_minus=sf_xx_minus,
        sf_yy_plus=sf_yy_plus,
        sf_yy_minus=sf_yy_minus,
        sf_11_plus=sf_11_plus,
        sf_11_minus=sf_11_minus,
        sf_22_plus=sf_22_plus,
        sf_22_minus=sf_22_minus,
        fy_ref=fy_ref,
    )
