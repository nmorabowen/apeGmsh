"""Stress recovery for the section analyzer (ADR 0078 S4).

Everything is a **linear blend of unit-load fields** computed once from
the cached geometric + warping solutions — evaluating a new load vector
is a weighted sum, never a re-solve (this is what makes the S6
inspector's live load inputs cheap).

Sign conventions (documented, equilibrium-tested):

- ``N`` positive in tension; ``σ = E·N/EA``.
- ``Mxx`` produces tension at ``+y``  (``Mxx = ∫σ·ȳ dA``).
- ``Myy`` produces tension at ``+x``  (``Myy = ∫σ·x̄ dA``).
- ``M11``/``M22`` are the same statements in the principal frame.
- ``Mzz`` positive counter-clockwise; ``τ = G·(Mzz/GJ)·(∇ω + (−ȳ, x̄))``.
- ``Vx``/``Vy`` via the Ψ/Φ shear functions (E-weighted, ``ν_eff``).

Recovery is **exact nodal evaluation** (shape-function gradients at the
element nodes — no Gauss→node extrapolation), averaged across elements
*within each material region only*; the flat convenience view takes the
max-|value| across regions at interface nodes, and
``get(component, pg=...)`` returns the exact per-region field.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Mapping

import numpy as np
from numpy import ndarray

from ._fe import block_nodal
from ._geometric import GeometricProperties
from ._snapshot import SectionSnapshot
from ._warping import _PartSolution

if TYPE_CHECKING:  # pragma: no cover
    from matplotlib.axes import Axes

# scalar (σ_zz) unit-load keys and vector (τ) unit-load keys
_SIGMA_ACTIONS = ("n", "mxx", "myy", "m11", "m22")
_TAU_ACTIONS = ("mzz", "vx", "vy")


@dataclass(frozen=True, slots=True)
class _UnitFields:
    """Per-region nodal unit-load fields.

    ``sigma[m][k]`` — (n_nodes,) σ_zz per unit action ``k``; NaN outside
    region ``m``.  ``tau[m][k]`` — (n_nodes, 2) τ per unit action.
    ``region_mask[m]`` — nodes belonging to region ``m``.
    ``triangles`` — corner triangulation for plotting.
    """

    sigma: tuple[dict[str, ndarray], ...]
    tau: tuple[dict[str, ndarray], ...]
    region_mask: tuple[ndarray, ...]
    triangles: ndarray


def compute_unit_fields(
    snap: SectionSnapshot,
    geo: GeometricProperties,
    sol: _PartSolution,
) -> _UnitFields:
    """Build the eight unit-load nodal fields (connected sections)."""
    import math

    n_nodes = len(snap.coords)
    n_mats = len(snap.materials)
    E_by_mat = np.array([m.E for m in snap.materials])
    G_by_mat = np.array([m.shear_modulus for m in snap.materials])

    D = sol.EIxx * sol.EIyy - sol.EIxy**2
    theta = math.radians(geo.phi)
    ct, st = math.cos(theta), math.sin(theta)
    # principal EI from the part solve (equals geo's for connected)
    h = math.hypot(sol.EIxx - sol.EIyy, 2.0 * sol.EIxy)
    EI11 = 0.5 * (sol.EIxx + sol.EIyy + h)
    EI22 = 0.5 * (sol.EIxx + sol.EIyy - h)

    # accumulators: per material, sums + counts for nodal averaging
    sig_sum = [
        {k: np.zeros(n_nodes) for k in _SIGMA_ACTIONS} for _ in range(n_mats)
    ]
    tau_sum = [
        {k: np.zeros((n_nodes, 2)) for k in _TAU_ACTIONS}
        for _ in range(n_mats)
    ]
    count = [np.zeros(n_nodes) for _ in range(n_mats)]
    tris: list[ndarray] = []

    for b in sol.blocks:
        q = block_nodal(b, sol.coords, centroid=(sol.cx, sol.cy))
        conn_g = sol.node_rows[b.conn]                   # global rows
        Ee = E_by_mat[b.mat_idx][:, None]
        Ge = G_by_mat[b.mat_idx][:, None]

        # σ unit fields at the element nodes
        x, y = q.x, q.y                                   # (E, npe)
        x1 = x * ct + y * st
        y1 = -x * st + y * ct
        sig = {
            "n": Ee * np.ones_like(x) / sol.EA,
            "mxx": Ee * (sol.EIyy * y - sol.EIxy * x) / D,
            "myy": Ee * (sol.EIxx * x - sol.EIxy * y) / D,
            "m11": Ee * y1 / EI11,
            "m22": Ee * x1 / EI22,
        }

        # τ unit fields
        om_e = sol.omega[b.conn]
        psi_e = sol.psi[b.conn]
        phi_e = sol.phi[b.conn]
        Bom = np.einsum("eija,ea->eij", q.B, om_e)        # (E, npe, 2)
        Bpsi = np.einsum("eija,ea->eij", q.B, psi_e)
        Bphi = np.einsum("eija,ea->eij", q.B, phi_e)
        r = x**2 - y**2
        s2 = 2.0 * x * y
        d1 = sol.EIxx * r - sol.EIxy * s2
        d2 = sol.EIxy * r + sol.EIxx * s2
        h1 = -sol.EIxy * r + sol.EIyy * s2
        h2 = -sol.EIyy * r - sol.EIxy * s2
        dv = np.stack([d1, d2], axis=-1)
        hv = np.stack([h1, h2], axis=-1)
        tor = Bom + np.stack([-y, x], axis=-1)
        tau = {
            "mzz": Ge[:, :, None] * tor / sol.GJ,
            "vx": Ee[:, :, None] * (Bpsi - 0.5 * sol.nu_eff * dv)
            / sol.delta_s,
            "vy": Ee[:, :, None] * (Bphi - 0.5 * sol.nu_eff * hv)
            / sol.delta_s,
        }

        # scatter-average per material region
        flat = conn_g.ravel()
        for e_mask, m in _per_material(b.mat_idx):
            rows = conn_g[e_mask].ravel()
            np.add.at(count[m], rows, 1.0)
            for k in _SIGMA_ACTIONS:
                np.add.at(sig_sum[m][k], rows, sig[k][e_mask].ravel())
            for k in _TAU_ACTIONS:
                np.add.at(
                    tau_sum[m][k], rows,
                    tau[k][e_mask].reshape(-1, 2),
                )
        del flat

        # corner triangulation for plotting
        nc = b.n_corners
        corners = conn_g[:, :nc]
        if nc == 3:
            tris.append(corners)
        else:
            tris.append(corners[:, [0, 1, 2]])
            tris.append(corners[:, [0, 2, 3]])

    sigma_out, tau_out, masks = [], [], []
    for m in range(n_mats):
        cnt = count[m]
        mask = cnt > 0
        inv = np.where(mask, 1.0 / np.maximum(cnt, 1.0), np.nan)
        sigma_out.append(
            {k: sig_sum[m][k] * inv for k in _SIGMA_ACTIONS}
        )
        tau_out.append(
            {k: tau_sum[m][k] * inv[:, None] for k in _TAU_ACTIONS}
        )
        masks.append(mask)

    return _UnitFields(
        sigma=tuple(sigma_out),
        tau=tuple(tau_out),
        region_mask=tuple(masks),
        triangles=np.concatenate(tris),
    )


def _per_material(mat_idx: ndarray):
    for m in np.unique(mat_idx):
        yield mat_idx == m, int(m)


# --------------------------------------------------------------------- #
# SectionStress                                                          #
# --------------------------------------------------------------------- #


class SectionStress:
    """Linear-elastic stress state for one load vector.

    Per-node arrays over the section mesh (flat views take max-|value|
    across regions at material-interface nodes; ``get(pg=...)`` is the
    exact per-region field, NaN outside the region).
    """

    def __init__(
        self,
        snap: SectionSnapshot,
        fields: _UnitFields,
        loads: Mapping[str, float],
    ) -> None:
        self._snap = snap
        self._fields = fields
        self.loads: dict[str, float] = dict(loads)

        n_nodes = len(snap.coords)
        n_mats = len(fields.sigma)
        sig_w = {
            "n": loads["N"], "mxx": loads["Mxx"], "myy": loads["Myy"],
            "m11": loads["M11"], "m22": loads["M22"],
        }
        tau_w = {"mzz": loads["Mzz"], "vx": loads["Vx"], "vy": loads["Vy"]}

        # per-region combined + per-action fields
        self._sigma_by_region: list[dict[str, ndarray]] = []
        self._tau_by_region: list[dict[str, ndarray]] = []
        for m in range(n_mats):
            s = {
                f"sigma_zz_{k}": fields.sigma[m][k] * sig_w[k]
                for k in _SIGMA_ACTIONS
            }
            s["sigma_zz"] = np.sum([s[f"sigma_zz_{k}"]
                                    for k in _SIGMA_ACTIONS], axis=0)
            t = {}
            for k in _TAU_ACTIONS:
                v = fields.tau[m][k] * tau_w[k]
                t[f"tau_zx_{k}"] = v[:, 0]
                t[f"tau_zy_{k}"] = v[:, 1]
            t["tau_zx"] = np.sum([t[f"tau_zx_{k}"] for k in _TAU_ACTIONS],
                                 axis=0)
            t["tau_zy"] = np.sum([t[f"tau_zy_{k}"] for k in _TAU_ACTIONS],
                                 axis=0)
            self._sigma_by_region.append(s)
            self._tau_by_region.append(t)
        self._n_nodes = n_nodes

    # ── access ───────────────────────────────────────────────────────

    def _flat(self, name: str) -> ndarray:
        """Max-|value| across regions (interface nodes only overlap)."""
        out = np.full(self._n_nodes, np.nan)
        best = np.full(self._n_nodes, -1.0)
        for m in range(len(self._sigma_by_region)):
            src = {**self._sigma_by_region[m], **self._tau_by_region[m]}
            if name not in src:
                raise KeyError(
                    f"unknown stress component {name!r}; available: "
                    f"{sorted(src)} + tau, von_mises"
                )
            v = src[name]
            mask = self._fields.region_mask[m]
            take = mask & (np.abs(np.nan_to_num(v)) > best)
            out[take] = v[take]
            best[take] = np.abs(np.nan_to_num(v))[take]
        return out

    def get(self, component: str, *, pg: str | None = None) -> ndarray:
        """One component; ``pg=`` restricts to a material region (exact
        values, NaN outside)."""
        if component == "tau":
            zx = self.get("tau_zx", pg=pg)
            zy = self.get("tau_zy", pg=pg)
            return np.hypot(zx, zy)
        if component == "von_mises":
            s = self.get("sigma_zz", pg=pg)
            t = self.get("tau", pg=pg)
            return np.sqrt(s**2 + 3.0 * t**2)
        if pg is None:
            return self._flat(component)
        try:
            m = self._snap.material_names.index(pg)
        except ValueError:
            raise KeyError(
                f"unknown material region {pg!r}; available: "
                f"{list(self._snap.material_names)}"
            ) from None
        src = {**self._sigma_by_region[m], **self._tau_by_region[m]}
        if component not in src:
            raise KeyError(
                f"unknown stress component {component!r}; available: "
                f"{sorted(src)} + tau, von_mises"
            )
        v = src[component].copy()
        v[~self._fields.region_mask[m]] = np.nan
        return v

    @property
    def sigma_zz(self) -> ndarray:
        return self._flat("sigma_zz")

    @property
    def tau_zx(self) -> ndarray:
        return self._flat("tau_zx")

    @property
    def tau_zy(self) -> ndarray:
        return self._flat("tau_zy")

    @property
    def tau(self) -> ndarray:
        return self.get("tau")

    @property
    def von_mises(self) -> ndarray:
        return self.get("von_mises")

    # ── plotting ─────────────────────────────────────────────────────

    def plot(
        self,
        component: str = "von_mises",
        *,
        ax: "Axes | None" = None,
        cmap: str = "coolwarm",
        levels: int = 15,
    ) -> "Axes":
        """Filled tricontour of one component over the section mesh."""
        import matplotlib.pyplot as plt
        import matplotlib.tri as mtri

        values = self.get(component)
        if ax is None:
            _, ax = plt.subplots()
        tri = mtri.Triangulation(
            self._snap.coords[:, 0], self._snap.coords[:, 1],
            triangles=self._fields.triangles,
        )
        good = np.isfinite(values)
        vals = np.where(good, values, 0.0)
        tcs = ax.tricontourf(tri, vals, levels=levels, cmap=cmap)
        ax.figure.colorbar(tcs, ax=ax, label=component)
        ax.set_aspect("equal")
        ax.set_title(component)
        return ax

    def __repr__(self) -> str:
        active = {k: v for k, v in self.loads.items() if v}
        return f"<SectionStress loads={active or '{}'}>"
