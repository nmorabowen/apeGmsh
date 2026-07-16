"""``SectionProperties`` — the cross-section analyzer broker (ADR 0078).

Consumes a meshed 2-D face (``FEMData``), snapshots it at construction
(session-independent thereafter, ADR 0001 doctrine), and serves memoized
frozen analysis results.  S1 ships geometric analysis; warping / plastic
/ stress land in later slices (ADR 0078 S2–S4).
"""
from __future__ import annotations

from types import MappingProxyType
from typing import TYPE_CHECKING, Literal, Mapping

from ._errors import SectionMeshError
from ._geometric import GeometricProperties, compute_geometric
from ._materials import SectionMaterial
from ._snapshot import SectionSnapshot, build_snapshot
from ._warping import WarpingProperties, compute_warping

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
            self._warping = compute_warping(
                self._snapshot,
                self.geometric(),
                policy=self._disconnected,
                handle=self._name or "section",
            )
        return self._warping

    def analyze(self) -> "SectionProperties":
        """Run every available analysis (S1–S2: geometric + warping).
        Returns self."""
        self.geometric()
        self.warping()
        return self

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
