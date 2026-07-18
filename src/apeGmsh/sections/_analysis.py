"""``SectionProperties`` — the cross-section analyzer broker (ADR 0078).

Consumes a meshed 2-D face (``FEMData``), snapshots it at construction
(session-independent thereafter, ADR 0001 doctrine), and serves memoized
frozen analysis results.  S1 ships geometric analysis; warping / plastic
/ stress land in later slices (ADR 0078 S2–S4).
"""
from __future__ import annotations

from types import MappingProxyType
from typing import TYPE_CHECKING, Literal, Mapping

import numpy as np

from ._errors import SectionAnalysisError, SectionMeshError
from ._geometric import GeometricProperties, compute_geometric
from ._materials import SectionMaterial
from ._plastic import PlasticProperties, compute_plastic
from ._snapshot import SectionSnapshot, build_snapshot
from ._stress import SectionStress, _UnitFields, compute_unit_fields
from ._warping import WarpingProperties, _PartSolution, compute_warping

if TYPE_CHECKING:  # pragma: no cover
    from apeGmsh.mesh.FEMData import FEMData


class SectionProperties:
    """Analyzer + declaration for one meshed cross-section.

    Parameters
    ----------
    fem
        A ``FEMData`` whose 2-D elements mesh the section face in the
        global XY plane (``g.mesh.queries.get_fem_data(dim=2)``).
    materials
        Physical-group name → :class:`SectionMaterial`.  Every 2-D
        element must belong to exactly one named PG.  Omit entirely for
        geometric-only mode (unit moduli — classic geometric numbers).
    name
        Handle used in fail-loud messages and displays.
    disconnected
        Multi-part policy (ADR 0078): ``"raise"`` (default) makes the
        S2 warping solve fail loud on a disconnected mesh — usually the
        forgot-to-fragment authoring bug; ``"sum"`` opts into per-part
        Saint-Venant solves.  Geometric and plastic analyses are
        connectivity-blind in either mode.

    Notes
    -----
    The analyzer is a *declaration*: frozen inputs, memoized frozen
    results.  ``ops.section.ComputedSection(analysis=sec)`` (S5) binds
    it to the OpenSees bridge and resolves lazily at emit.
    """

    def __init__(
        self,
        fem: "FEMData",
        *,
        materials: Mapping[str, SectionMaterial] | None = None,
        name: str | None = None,
        disconnected: Literal["raise", "sum"] = "raise",
    ) -> None:
        if disconnected not in ("raise", "sum"):
            raise ValueError(
                f"SectionProperties: disconnected must be 'raise' or 'sum', "
                f"got {disconnected!r}."
            )
        self._name = name
        self._disconnected: Literal["raise", "sum"] = disconnected
        self._snapshot: SectionSnapshot = build_snapshot(
            fem, materials, name=name
        )
        self._materials: Mapping[str, SectionMaterial] = MappingProxyType(
            dict(zip(self._snapshot.material_names, self._snapshot.materials))
        )
        self._geometric: GeometricProperties | None = None
        self._warping: WarpingProperties | None = None
        self._warp_solutions: tuple[_PartSolution, ...] = ()
        self._plastic: PlasticProperties | None = None
        self._unit_fields: _UnitFields | None = None

    # ── identity ─────────────────────────────────────────────────────

    @property
    def name(self) -> str | None:
        return self._name

    @property
    def materials(self) -> Mapping[str, SectionMaterial]:
        """Read-only PG-name → material view (empty-ish placeholder map
        in geometric-only mode)."""
        return self._materials

    @property
    def disconnected(self) -> str:
        return self._disconnected

    @property
    def geometric_only(self) -> bool:
        return self._snapshot.geometric_only

    @property
    def n_parts(self) -> int:
        """Connected-component count of the section mesh."""
        return self._snapshot.n_components

    # ── analyses (memoized, frozen returns) ──────────────────────────

    def geometric(self) -> GeometricProperties:
        """Area-based (modulus-weighted) properties.  Pure quadrature —
        connectivity-blind, valid for disconnected sections."""
        if self._geometric is None:
            self._geometric = compute_geometric(self._snapshot)
        return self._geometric

    def warping(self) -> WarpingProperties:
        """Saint-Venant warping / shear analysis: ``GJ``, shear centre
        (elasticity + Trefftz), warping rigidity ``EGamma``, shear
        rigidities ``GAs_*``, monosymmetry constants.

        Requires a connected mesh under the default
        ``disconnected="raise"``; ``"sum"`` solves per part (ADR 0078).
        Warns :class:`SectionAccuracyWarning` on linear elements.
        """
        if self._warping is None:
            self._warping, self._warp_solutions = compute_warping(
                self._snapshot,
                self.geometric(),
                policy=self._disconnected,
                handle=self._name or "section",
            )
        return self._warping

    def plastic(self) -> PlasticProperties:
        """Rigid-plastic analysis: plastic centroids, fy-weighted plastic
        moments ``Mp_*``, first-yield shape factors.  Requires ``fy`` on
        every material; connectivity-blind (valid for disconnected
        sections).  Invalid for strain-softening materials."""
        if self._plastic is None:
            self._plastic = compute_plastic(
                self._snapshot,
                self.geometric(),
                handle=self._name or "section",
            )
        return self._plastic

    def stress(
        self,
        *,
        N: float = 0.0,
        Vx: float = 0.0,
        Vy: float = 0.0,
        Mxx: float = 0.0,
        Myy: float = 0.0,
        M11: float = 0.0,
        M22: float = 0.0,
        Mzz: float = 0.0,
    ) -> SectionStress:
        """Linear-elastic stress recovery for one load vector.

        A weighted blend of unit-load fields computed once from the
        cached geometric + warping solutions — calling with a new load
        vector never re-solves anything.  Sign conventions: ``N``
        tension-positive; ``Mxx`` tension at ``+y``; ``Myy`` tension at
        ``+x``; ``M11``/``M22`` likewise in the principal frame;
        ``Mzz`` counter-clockwise.  See :class:`SectionStress` for the
        component list and the per-region access contract.
        """
        self.warping()   # ensures solutions (fail-loud on disconnected)
        if len(self._warp_solutions) != 1:
            raise SectionAnalysisError(
                f"{self._name or 'section'}: stress recovery on a "
                f"disconnected section (disconnected='sum') is not yet "
                f"implemented — analyze the parts as separate sections "
                f"to recover their stress fields."
            )
        if self._unit_fields is None:
            self._unit_fields = compute_unit_fields(
                self._snapshot, self.geometric(), self._warp_solutions[0]
            )
        return SectionStress(
            self._snapshot,
            self._unit_fields,
            {"N": N, "Vx": Vx, "Vy": Vy, "Mxx": Mxx, "Myy": Myy,
             "M11": M11, "M22": M22, "Mzz": Mzz},
        )

    def analyze(self) -> "SectionProperties":
        """Run every available analysis (S1–S3: geometric + warping +
        plastic when fy is available).  Returns self."""
        self.geometric()
        self.warping()
        if not self._snapshot.geometric_only and all(
            m.fy is not None for m in self._snapshot.materials
        ):
            self.plastic()
        return self

    # ── bridge handoff (ADR 0078 S5) ─────────────────────────────────

    def to_elastic_section(
        self,
        *,
        E: float | None = None,
        G: float | None = None,
        ndm: int = 3,
    ):
        """Eagerly lower this analyzer into a plain populated
        :class:`~apeGmsh.opensees.section.ElasticSection`.

        Runs the same shared lowering as
        ``ops.section.ComputedSection(analysis=...)`` — authoring
        ``Ixx_c → Iz``, ``Iyy_c → Iy``, ``J → J``, ``As_y/A → alphaY``,
        ``As_x/A → alphaZ`` — but resolves **now** and returns an
        inspectable, analyzer-decoupled primitive.

        ``E`` / ``G`` default from the single material on a homogeneous
        analyzer; for a **composite** they are required reference
        moduli (transformed-section ``EA/E``, ``EI/E``, ``GJ/G``) and
        for a **geometric-only** analyzer they are required deck
        moduli.  ``ndm=3`` (default) emits the 3-D form; ``ndm=2`` the
        2-D shear-flexible form.
        """
        from apeGmsh.opensees.section.beam import ElasticSection

        from ._lowering import lower_to_elastic

        params = lower_to_elastic(self, E=E, G=G)
        return ElasticSection(**params.section_kwargs(ndm))

    # ── display ──────────────────────────────────────────────────────

    def summary(self) -> str:
        """Plain-text properties report."""
        snap = self._snapshot
        handle = self._name or "section"
        lines = [
            f"SectionProperties '{handle}'",
            f"  elements : {snap.n_elements} "
            f"({', '.join(sorted({b.type_name for b in snap.blocks}))})",
            f"  nodes    : {len(snap.coords)}",
            f"  parts    : {snap.n_components} "
            f"(disconnected policy: {self._disconnected})",
        ]
        if snap.geometric_only:
            lines.append("  materials: geometric-only mode (unit moduli)")
        else:
            for pg, mat, a in zip(
                snap.material_names, snap.materials,
                self.geometric().material_areas,
            ):
                fy = f", fy={mat.fy:g}" if mat.fy is not None else ""
                lines.append(
                    f"  materials: '{pg}' E={mat.E:g} nu={mat.nu:g}"
                    f"{fy}  (A={a:.6g})"
                )
        g = self.geometric()
        lines += [
            f"  area={g.area:.6g}  perimeter={g.perimeter:.6g}"
            + (f"  mass={g.mass:.6g}" if g.mass is not None else ""),
            f"  centroid=({g.cx:.6g}, {g.cy:.6g})  phi={g.phi:.4g} deg",
            f"  EA={g.EA:.6g}",
            f"  EIxx_c={g.EIxx_c:.6g}  EIyy_c={g.EIyy_c:.6g}  "
            f"EIxy_c={g.EIxy_c:.6g}",
            f"  EI11_c={g.EI11_c:.6g}  EI22_c={g.EI22_c:.6g}",
        ]
        if g.e_ref is not None:
            lines.append(
                f"  (single modulus E={g.e_ref:g}: "
                f"A_eff={g.EA / g.e_ref:.6g}, Ixx_c={g.Ixx_c:.6g}, "
                f"Iyy_c={g.Iyy_c:.6g})"
            )
        else:
            lines.append(
                "  (composite: unprefixed accessors raise — use "
                "transformed(e_ref=...))"
            )
        return "\n".join(lines)

    def plot_mesh(self, *, ax=None):
        """Matplotlib wireframe of the section mesh, colored by
        material region."""
        import matplotlib.pyplot as plt
        import matplotlib.tri as mtri

        if ax is None:
            _, ax = plt.subplots()
        snap = self._snapshot
        cmap = plt.get_cmap("tab10")
        for m, pg in enumerate(snap.material_names):
            tris = []
            for b in snap.blocks:
                corners = b.conn[b.mat_idx == m][:, : b.n_corners]
                if not len(corners):
                    continue
                if b.n_corners == 3:
                    tris.append(corners)
                else:
                    tris.append(corners[:, [0, 1, 2]])
                    tris.append(corners[:, [0, 2, 3]])
            if not tris:
                continue
            tri = mtri.Triangulation(
                snap.coords[:, 0], snap.coords[:, 1],
                triangles=np.concatenate(tris),
            )
            ax.triplot(tri, color=cmap(m % 10), linewidth=0.3, label=pg)
        ax.set_aspect("equal")
        if not snap.geometric_only:
            ax.legend(loc="best", fontsize="small")
        return ax

    def plot_section(
        self,
        *,
        centroid: bool = True,
        shear_centre: bool = True,
        principal_axes: bool = True,
        ax=None,
    ):
        """Section outline + glyph overlay: elastic centroid, shear
        centre (triggers :meth:`warping` — pass ``shear_centre=False``
        for disconnected sections under the default policy), principal
        axes at ``phi``."""
        import math


        ax = self.plot_mesh(ax=ax)
        geo = self.geometric()
        if centroid:
            ax.plot(geo.cx, geo.cy, "k+", markersize=12, label="centroid")
        if shear_centre:
            warp = self.warping()
            ax.plot(warp.x_sc, warp.y_sc, "rx", markersize=10,
                    label="shear centre")
        if principal_axes:
            theta = math.radians(geo.phi)
            span = 0.35 * max(
                float(np.ptp(self._snapshot.coords[:, 0])),
                float(np.ptp(self._snapshot.coords[:, 1])),
            )
            for ang, style, lbl in (
                (theta, "-", "11"), (theta + math.pi / 2, "--", "22"),
            ):
                dx, dy = span * math.cos(ang), span * math.sin(ang)
                ax.plot([geo.cx - dx, geo.cx + dx],
                        [geo.cy - dy, geo.cy + dy],
                        linestyle=style, color="0.4", linewidth=0.8)
                ax.annotate(lbl, (geo.cx + dx, geo.cy + dy), color="0.4")
        ax.legend(loc="best", fontsize="small")
        return ax

    def _repr_html_(self) -> str:  # pragma: no cover - inspected visually
        body = self.summary().replace("\n", "<br>")
        return f"<pre style='line-height:1.3'>{body}</pre>"

    def __repr__(self) -> str:
        handle = f" '{self._name}'" if self._name else ""
        snap = self._snapshot
        return (
            f"<SectionProperties{handle}: {snap.n_elements} elements, "
            f"{snap.n_components} part(s), "
            f"{'geometric-only' if snap.geometric_only else 'materials'}>"
        )


__all__ = ["SectionProperties", "SectionMaterial", "SectionMeshError"]
