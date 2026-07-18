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
    sols: tuple[_PartSolution, ...],
) -> _UnitFields:
    """Build the eight unit-load nodal fields.

    Connected sections (one solution) recover from the part solve
    directly.  Under ``disconnected="sum"`` (several solutions) the
    applied actions distribute per the ADR 0078 policy:

    - ``N``/``Mxx``/``Myy``/``M11``/``M22`` use the **global**
      plane-sections composite state (``geo`` — common centroid,
      Steiner terms included), evaluated over every part;
    - ``Mzz`` distributes to parts ∝ ``GJᵢ/ΣGJ`` (equal twist rate),
      each part recovering torsional shear from its own ω solve;
    - ``Vx``/``Vy`` distribute ∝ the part flexural-rigidity shares
      ``EIyyᵢ/ΣEIyy`` / ``EIxxᵢ/ΣEIxx`` (consistent with equal
      curvature), each part recovering shear from its own Ψ/Φ solves.
      The shares are **scalar per axis** — exact when each part's own
      principal axes align with x/y; a part rotated in-plane (part
      ``EIxy ≠ 0``) gets an approximate split, though its recovered
      field itself remains fully coupled through its own Ψ/Φ solves.
    """
    import math

    n_nodes = len(snap.coords)
    n_mats = len(snap.materials)
    E_by_mat = np.array([m.E for m in snap.materials])
    G_by_mat = np.array([m.shear_modulus for m in snap.materials])

    multi = len(sols) > 1
    GJ_tot = sum(s.GJ for s in sols)
    EIxx_tot = sum(s.EIxx for s in sols)
    EIyy_tot = sum(s.EIyy for s in sols)

    theta = math.radians(geo.phi)
    ct, st = math.cos(theta), math.sin(theta)

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

    for sol in sols:
        if multi:
            # global plane-sections state for the σ recovery
            sEA = geo.EA
            sEIxx, sEIyy, sEIxy = geo.EIxx_c, geo.EIyy_c, geo.EIxy_c
            EI11, EI22 = geo.EI11_c, geo.EI22_c
            dx_c, dy_c = sol.cx - geo.cx, sol.cy - geo.cy
            share_vx = sol.EIyy / EIyy_tot
            share_vy = sol.EIxx / EIxx_tot
        else:
            sEA = sol.EA
            sEIxx, sEIyy, sEIxy = sol.EIxx, sol.EIyy, sol.EIxy
            # principal EI from the part solve (equals geo's for connected)
            h = math.hypot(sol.EIxx - sol.EIyy, 2.0 * sol.EIxy)
            EI11 = 0.5 * (sol.EIxx + sol.EIyy + h)
            EI22 = 0.5 * (sol.EIxx + sol.EIyy - h)
            dx_c = dy_c = 0.0
            share_vx = share_vy = 1.0
        D = sEIxx * sEIyy - sEIxy**2

        for b in sol.blocks:
            q = block_nodal(b, sol.coords, centroid=(sol.cx, sol.cy))
            conn_g = sol.node_rows[b.conn]                   # global rows
            Ee = E_by_mat[b.mat_idx][:, None]
            Ge = G_by_mat[b.mat_idx][:, None]

            # σ unit fields at the element nodes — coordinates about the
            # σ-state centroid (part centroid when connected; global
            # centroid under "sum")
            x, y = q.x, q.y                                   # (E, npe)
            xs, ys = x + dx_c, y + dy_c
            x1 = xs * ct + ys * st
            y1 = -xs * st + ys * ct
            sig = {
                "n": Ee * np.ones_like(x) / sEA,
                "mxx": Ee * (sEIyy * ys - sEIxy * xs) / D,
                "myy": Ee * (sEIxx * xs - sEIxy * ys) / D,
                "m11": Ee * y1 / EI11,
                "m22": Ee * x1 / EI22,
            }

            # τ unit fields — always the part's own solves and axes
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
                # part receives Mzzᵢ = Mzz·GJᵢ/ΣGJ and divides by its
                # own GJᵢ → the ΣGJ denominator (== sol.GJ when connected)
                "mzz": Ge[:, :, None] * tor / GJ_tot,
                "vx": share_vx * Ee[:, :, None]
                * (Bpsi - 0.5 * sol.nu_eff * dv) / sol.delta_s,
                "vy": share_vy * Ee[:, :, None]
                * (Bphi - 0.5 * sol.nu_eff * hv) / sol.delta_s,
            }

            # scatter-average per material region
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

    def plot_vector(
        self,
        action: str | None = None,
        *,
        ax: "Axes | None" = None,
        color: str = "k",
        max_arrows: int = 800,
    ) -> "Axes":
        """Quiver of the shear-stress vector ``(τ_zx, τ_zy)`` over the
        section mesh.

        ``action`` restricts to one per-action term — ``"mzz"``,
        ``"vx"``, or ``"vy"`` — instead of the combined field.  A light
        mesh wireframe is drawn underneath; arrow count is thinned to
        ``max_arrows`` on fine meshes.
        """
        import matplotlib.pyplot as plt
        import matplotlib.tri as mtri

        if action is not None and action not in _TAU_ACTIONS:
            raise KeyError(
                f"unknown shear action {action!r}; expected one of "
                f"{sorted(_TAU_ACTIONS)} (or None for the combined field)."
            )
        suffix = f"_{action}" if action is not None else ""
        tzx = np.nan_to_num(self.get(f"tau_zx{suffix}"))
        tzy = np.nan_to_num(self.get(f"tau_zy{suffix}"))

        if ax is None:
            _, ax = plt.subplots()
        coords = self._snap.coords
        tri = mtri.Triangulation(
            coords[:, 0], coords[:, 1],
            triangles=self._fields.triangles,
        )
        ax.triplot(tri, color="0.85", linewidth=0.3)
        idx = np.unique(np.linspace(
            0, len(coords) - 1, min(max_arrows, len(coords)),
        ).astype(int))
        ax.quiver(
            coords[idx, 0], coords[idx, 1], tzx[idx], tzy[idx],
            color=color, width=0.002,
        )
        ax.set_aspect("equal")
        ax.set_title(f"tau{suffix}" if suffix else "tau (combined)")
        return ax

    def _mohr_state(
        self, *, at: tuple[float, float], pg: str | None = None
    ) -> tuple[float, float, int]:
        """``(σ_zz, |τ|, node_row)`` at the mesh node nearest ``at``
        (restricted to region ``pg`` when given)."""
        sig = self.get("sigma_zz", pg=pg)
        tau = self.get("tau", pg=pg)
        valid = np.isfinite(sig)
        if not valid.any():
            raise KeyError(
                f"no nodes carry values for pg={pg!r} — unknown or "
                f"empty material region."
            )
        coords = self._snap.coords
        d2 = np.where(
            valid,
            (coords[:, 0] - at[0]) ** 2 + (coords[:, 1] - at[1]) ** 2,
            np.inf,
        )
        node = int(np.argmin(d2))
        return float(sig[node]), float(tau[node]), node

    def plot_mohrs_circle(
        self,
        *,
        at: tuple[float, float],
        pg: str | None = None,
        ax: "Axes | None" = None,
    ) -> "Axes":
        """Mohr's circle of the beam stress state ``(σ_zz, τ)`` at the
        mesh node nearest ``at`` (authoring coordinates).

        The beam state has one normal stress and one resultant shear,
        so the circle is centred at ``σ/2`` with radius
        ``√((σ/2)² + τ²)``; principal stresses ``σ₁/σ₂ = σ/2 ± R`` are
        annotated.  ``pg=`` picks the exact per-region value at a
        material interface.
        """
        import math

        import matplotlib.pyplot as plt

        sigma, tau, node = self._mohr_state(at=at, pg=pg)
        centre = 0.5 * sigma
        radius = math.hypot(centre, tau)

        if ax is None:
            _, ax = plt.subplots()
        theta = np.linspace(0.0, 2.0 * np.pi, 181)
        ax.plot(centre + radius * np.cos(theta), radius * np.sin(theta),
                "b-", linewidth=1.2)
        ax.plot([sigma, 0.0], [tau, -tau], "ko--", markersize=5,
                linewidth=0.8)
        ax.axhline(0.0, color="0.6", linewidth=0.8)
        ax.axvline(0.0, color="0.6", linewidth=0.8)
        s1, s2 = centre + radius, centre - radius
        ax.plot([s1, s2], [0.0, 0.0], "r.", markersize=8)
        ax.annotate(f"σ₁={s1:.4g}", (s1, 0.0), textcoords="offset points",
                    xytext=(4, 6), fontsize=8)
        ax.annotate(f"σ₂={s2:.4g}", (s2, 0.0), textcoords="offset points",
                    xytext=(4, 6), fontsize=8)
        xy = self._snap.coords[node]
        ax.set_title(
            f"Mohr's circle at ({xy[0]:.4g}, {xy[1]:.4g})"
            + (f" [{pg}]" if pg else "")
        )
        ax.set_xlabel("σ")
        ax.set_ylabel("τ")
        ax.set_aspect("equal")
        return ax

    def __repr__(self) -> str:
        active = {k: v for k, v in self.loads.items() if v}
        return f"<SectionStress loads={active or '{}'}>"
