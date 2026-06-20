"""
``g.rebar`` — the L2 reinforcement-cage authoring composite (ADR 0066).

Sits **above** the shipped ``g.reinforce`` binding composite: it owns the
L1 spec objects (:mod:`apeGmsh._kernel.defs.rebar`) + geometry generation
+ standardized-member generators, and **delegates** coupling —
*conformal* via ``g.mesh.editing.embed`` (this module, P1) and *embedded*
via ``g.reinforce`` (P2). It never emits an OpenSees element itself.

P1 scope: ``bar`` / ``stirrup`` / ``stirrup_rect`` spec emitters, eager
**polyline** geometry emission (``true_arc`` is deferred to P3), and
``place(cage, into, coupling="conformal")`` which embeds the bar curves
into the host solid before meshing so the host mesh conforms and the bars
share its nodes (perfect bond — the ``ladruno_rc.py`` behaviour
generalised off the grid).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Iterable

from .._kernel.defs.rebar import METADATA, Bar, Cage, Hook, Path, Stirrup, Vec3
from ._compose_errors import chain_phase_guard
from ._helpers import resolve_to_tags

if TYPE_CHECKING:
    from .._core import _ApeGmshSession


# ── resolution-side records (not L1 specs) ───────────────────────────

@dataclass(frozen=True)
class RebarMember:
    """A placed bar/stirrup: the curve physical group + the intent the
    bridge needs to realise a Truss/CorotTruss/DispBeamColumn on it."""
    pg: str
    role: str
    db: float | str
    material: str
    element: str
    line_tags: tuple[int, ...]


@dataclass(frozen=True)
class RebarPlacement:
    """The record of one ``place()`` call."""
    name: str
    host: str
    coupling: str
    members: tuple[RebarMember, ...]


# ── the composite ────────────────────────────────────────────────────

class RebarComposite:
    """``g.rebar`` — reinforcement-cage authoring (ADR 0066)."""

    def __init__(self, parent: "_ApeGmshSession") -> None:
        self._parent = parent
        self._standard: Any = None
        self.placements: list[RebarPlacement] = []

    # ---- detailing standard (used at resolve time, P3) --------------
    def use_standard(self, standard: Any) -> None:
        """Set the default :class:`DetailingStandard` for this session's
        cages (resolves ``"<k>db"`` tokens + hook factories at bind)."""
        self._standard = standard

    # ---- L1 spec emitters (thin) ------------------------------------
    def bar(self, points: Iterable[Vec3], *, db, material,
            role: str = "longitudinal", element: str = "truss",
            start_hook: Hook | None = None, end_hook: Hook | None = None,
            corner_radius=METADATA, name: str | None = None) -> Bar:
        return Bar(path=Path(tuple(points), corner_radius=corner_radius),
                   db=db, material=material, role=role, element=element,
                   start_hook=start_hook, end_hook=end_hook, name=name)

    def stirrup(self, points: Iterable[Vec3], *, db, material,
                closure_hook: Hook | None = None, role: str = "tie",
                corner_radius=METADATA, name: str | None = None) -> Stirrup:
        return Stirrup(path=Path(tuple(points), corner_radius=corner_radius),
                       db=db, material=material, role=role,
                       closure_hook=closure_hook or Hook.seismic_135(),
                       name=name)

    def stirrup_rect(self, bx: float, by: float, cover: float, *,
                     db, material, **kw) -> Stirrup:
        return Stirrup.rect(bx, by, cover, db=db, material=material, **kw)

    # ---- placement / coupling router --------------------------------
    def place(self, cage: Cage, into: str, *, coupling: str = "conformal",
              host_dim: int | None = None, true_arc: bool = False,
              name: str | None = None) -> RebarPlacement:
        """Emit the cage geometry and couple it to host solid ``into``.

        ``coupling="conformal"`` (P1) embeds the bar curves into the host
        so the mesh conforms (shared nodes, perfect bond).
        ``coupling="embedded"`` (P2) forwards to ``g.reinforce``.
        """
        chain_phase_guard(self._parent, "g.rebar.place")
        if not isinstance(cage, Cage):
            raise TypeError(
                f"g.rebar.place: cage must be a Cage, got {type(cage).__name__}."
            )
        if true_arc:
            raise NotImplementedError(
                "g.rebar.place: true_arc fillet geometry is deferred to P3; "
                "use the polyline default (true_arc=False) for now."
            )
        if coupling == "conformal":
            return self._place_conformal(cage, into, host_dim=host_dim,
                                         name=name)
        if coupling == "embedded":
            raise NotImplementedError(
                "g.rebar.place: coupling='embedded' is P2 (forwards to "
                "g.reinforce → LadrunoEmbeddedRebar)."
            )
        raise ValueError(
            f"g.rebar.place: coupling must be 'conformal' or 'embedded', "
            f"got {coupling!r}."
        )

    # ---- conformal (P1) ---------------------------------------------
    def _place_conformal(self, cage: Cage, into: str, *,
                         host_dim: int | None, name: str | None) -> RebarPlacement:
        g = self._parent
        geom = g.model.geometry
        in_dim = host_dim if host_dim is not None else self._detect_host_dim(into)
        base = name or "rebar"

        members: list[RebarMember] = []
        all_line_tags: list[int] = []
        idx = 0
        for default_role, items in (("longitudinal", cage.bars),
                                    ("tie", cage.stirrups)):
            for m in items:
                lts = self._emit_polyline(geom, m.path.points)
                role = getattr(m, "role", default_role)
                pg = f"{base}.{m.name or f'{role}_{idx}'}"
                g.physical.add_curve(lts, name=pg)
                members.append(RebarMember(
                    pg=pg, role=role, db=m.db, material=m.material,
                    element=getattr(m, "element", "truss"),
                    line_tags=tuple(lts),
                ))
                all_line_tags.extend(lts)
                idx += 1

        g.model.sync()
        # Conformal coupling: force the host mesh to conform to the bar
        # curves (gmsh embed) so generate() puts shared nodes on the bars.
        g.mesh.editing.embed(all_line_tags, into, dim=1, in_dim=in_dim)

        placement = RebarPlacement(name=base, host=into, coupling="conformal",
                                   members=tuple(members))
        self.placements.append(placement)
        return placement

    # ---- geometry helpers -------------------------------------------
    def _emit_polyline(self, geom, points: tuple[Vec3, ...]) -> list[int]:
        """Emit a polyline as gmsh points + line segments, returning the
        line tags. A closed loop (first == last) reuses the first point so
        the loop welds into one node ring."""
        closed = len(points) >= 2 and points[0] == points[-1]
        pt_tags: list[int] = []
        first_tag: int | None = None
        n = len(points)
        for i, p in enumerate(points):
            if closed and i == n - 1 and first_tag is not None:
                pt_tags.append(first_tag)
            else:
                t = geom.add_point(p[0], p[1], p[2], sync=False)
                if i == 0:
                    first_tag = t
                pt_tags.append(t)
        return [geom.add_line(pt_tags[i], pt_tags[i + 1], sync=False)
                for i in range(len(pt_tags) - 1)]

    def _detect_host_dim(self, into: str) -> int:
        """Resolve the host's dimension (3D solid preferred, then 2D)."""
        for d in (3, 2):
            try:
                if resolve_to_tags(into, dim=d, session=self._parent):
                    return d
            except Exception:
                continue
        raise ValueError(
            f"g.rebar.place: cannot resolve host {into!r} as a 3-D or 2-D "
            f"entity. Pass host_dim= explicitly or check the label."
        )

    # validate hook — resolution at get_fem_data (P3); nothing pre-mesh yet
    def validate_pre_mesh(self) -> None:
        return None
