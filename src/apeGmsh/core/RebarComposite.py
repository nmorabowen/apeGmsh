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

import math
import warnings
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any, Iterable

import gmsh
import numpy as np

from .._kernel.defs.rebar import (
    METADATA, Bar, BarBuilder, BarLayout, Cage, Hook, Path, Stirrup,
    TieLayout, Vec3,
)
from ..rebar._geometry import hook_primitives, outward_tangent
from ._compose_errors import chain_phase_guard
from ._helpers import resolve_to_tags

_AXIS_TOKENS = {
    "up": (0.0, 0.0, 1.0), "down": (0.0, 0.0, -1.0),
    "+x": (1.0, 0.0, 0.0), "-x": (-1.0, 0.0, 0.0),
    "+y": (0.0, 1.0, 0.0), "-y": (0.0, -1.0, 0.0),
    "+z": (0.0, 0.0, 1.0), "-z": (0.0, 0.0, -1.0),
}

# Transverse-reinforcement roles. A member with one of these roles is a
# tie / hoop / cross-tie, so its end hooks resolve with the seismic-hoop
# detailing kind (135° 6db, the §25.3.4 hook) and are OPTIONAL (dropped
# with a warning when no standard is set, like a stirrup closure) rather
# than mandatory the way a longitudinal-bar development hook is.
_TRANSVERSE_ROLES = frozenset({"tie", "crosstie", "hoop", "stirrup"})

if TYPE_CHECKING:
    from .._core import _ApeGmshSession


# ── resolution-side records (not L1 specs) ───────────────────────────

@dataclass(frozen=True)
class RebarMember:
    """A placed bar/stirrup: the curve physical group + everything the
    bridge needs to realise a Truss/CorotTruss/DispBeamColumn on it
    (diameter + area resolved for BOTH couplings)."""
    pg: str
    role: str
    db: float | str
    diameter: float
    area: float
    material: str
    element: str
    coupling: str
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
        self._place_seq = 0          # per-session counter → unique default PG base
        self.placements: list[RebarPlacement] = []

    # ---- detailing standard (used at resolve time, P3) --------------
    def use_standard(self, standard: Any) -> None:
        """Set the default :class:`DetailingStandard` for this session's
        cages (resolves ``"<k>db"`` tokens + hook factories at bind)."""
        self._standard = standard

    # ---- L1 spec emitters (thin) / L3 fluent ------------------------
    def bar(self, points: Iterable[Vec3] | None = None, *, db, material,
            role: str = "longitudinal", element: str = "truss",
            start_hook: Hook | None = None, end_hook: Hook | None = None,
            corner_radius=METADATA, name: str | None = None):
        """With ``points`` → an L1 :class:`Bar`. Without ``points`` → an L3
        fluent :class:`BarBuilder` (``.through(...).hook_end(...).as_(...)``)."""
        if points is None:
            if (start_hook is not None or end_hook is not None
                    or corner_radius != METADATA or name is not None):
                raise ValueError(
                    "g.rebar.bar(): on the fluent path (no points) set hooks/"
                    "corner_radius/name via the chain (.hook_end(...), "
                    ".corner_radius(...), .as_(name)), not as bar() kwargs.")
            return BarBuilder(db=db, material=material, role=role,
                              element=element)
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

    # ---- standardized members (L2 generators) -----------------------
    def column(self, *, section, height: float, cover: float,
               longitudinal: BarLayout, ties: TieLayout, base_z: float = 0.0,
               origin: tuple[float, float] = (0.0, 0.0), standard=None,
               top_hook: Hook | None = None, bottom_hook: Hook | None = None,
               end_cover: float | None = None, crossties: bool = True,
               confinement_style: str = "crossties") -> Cage:
        """Build a rectangular RC column cage (vertical, z-axis): perimeter
        longitudinal bars + tie rings (densified in the end hinge zones) +
        ACI 318 §25.7.2.3 cross-ties laterally supporting the intermediate
        (``n>2`` per face) longitudinal bars.

        Bars/ties are inset from the section faces (``cover + tie + db/2``)
        AND from the top/bottom faces (``end_cover``, default ``cover``) so
        the cage is interior to the host — both couplings mesh, and
        conformal embedding does not trip a boundary-facet PLC error.

        ``crossties=True`` (default) emits one cross-tie per intermediate
        bar at every tie level: a straight transverse leg spanning the
        section between the two opposite-face longitudinal bars it engages,
        with a 135° seismic hook on one end and a 90° hook on the other,
        alternated end-for-end on consecutive levels (ACI 318 §18.7.5.2).
        The cross-ties use the tie bar size/material and carry ``role=
        "crosstie"`` (so ``per_member_coupling`` can route them). Set
        ``crossties=False`` to emit the single perimeter hoop only (the
        intermediate bars are then laterally unsupported). Conformal
        coupling of cross-ties forms bar/tie T-junctions that need
        ``g.mesh.editing.make_conformal`` — prefer ``coupling="embedded"``.

        **Seismic confinement zone (ACI 318 §18.7.5).** When the active
        standard is ``ACI318_seismic`` and the :class:`TieLayout` leaves
        ``hinge_spacing``/``hinge_length`` unset, the confined-end length
        ``l_o`` and dense spacing ``s_o`` are auto-derived from the section
        geometry (a warning reports the values); ``ties.spacing`` is used
        outside the hinge zones. An explicit hinge layout overrides the code
        rule, and a non-seismic standard leaves the spacing uniform.

        ``confinement_style`` selects how the intermediate bars are laterally
        supported (only when ``crossties=True`` and ``n>2``): ``"crossties"``
        (default) adds straight cross-tie legs as above; ``"overlapping_hoops"``
        instead tiles the core with a grid of closed, overlapping rectangular
        cell-hoops whose corners sit on the longitudinal bar centerlines, so
        every bar is held at a hoop corner (the wide-section detail when the
        bar support spacing would otherwise be cross-tie heavy). The outer
        perimeter hoop is emitted in both styles.
        """
        kind, bx, by = self._rect_section(section, "column")
        self._require_positive(height, "height", "column")
        if longitudinal.n_x < 2 or longitudinal.n_y < 2:
            raise ValueError(
                "g.rebar.column: longitudinal n_x and n_y must be ≥ 2 (a "
                "rectangular perimeter cage); author a single bar line "
                "directly for a wall curtain.")
        if confinement_style not in ("crossties", "overlapping_hoops"):
            raise ValueError(
                "g.rebar.column: confinement_style must be 'crossties' or "
                f"'overlapping_hoops', got {confinement_style!r}.")
        std = standard if standard is not None else self._standard
        need_ct = longitudinal.n_x > 2 or longitudinal.n_y > 2
        if need_ct and not crossties:
            warnings.warn(
                "g.rebar.column: intermediate longitudinal bars (n>2) are "
                "generated but crossties=False, so only a single perimeter "
                "hoop is emitted; the intermediate bars are NOT laterally "
                "supported (ACI 318 §25.7.2.3).", stacklevel=2)
        dia = self._dia(std, longitudinal.db)
        dia_tie = (ties.db_value if ties.db_value is not None
                   else self._dia(std, ties.db))
        ox, oy = origin
        inset = cover + dia_tie + dia / 2.0       # bars nest INSIDE the ties
        if 2.0 * inset >= min(bx, by):
            raise ValueError(
                f"g.rebar.column: cover+tie+db/2={inset} too large for "
                f"section {bx}x{by} (longitudinal bars would cross).")
        ec = cover if end_cover is None else end_cover
        z0, z1 = base_z + ec, base_z + height - ec
        if z1 <= z0:
            raise ValueError(
                f"g.rebar.column: end_cover {ec} too large for height {height}.")
        xs = self._linspace(inset, bx - inset, longitudinal.n_x)
        ys = self._linspace(inset, by - inset, longitudinal.n_y)
        bars = tuple(
            Bar(path=Path(((ox + x, oy + y, z0), (ox + x, oy + y, z1))),
                db=longitudinal.db, material=longitudinal.material,
                start_hook=bottom_hook, end_hook=top_hook)
            for x, y in self._perimeter(xs, ys))
        ties = self._seismic_confinement(
            ties, std, bx=bx, by=by, height=height, inset=inset, dia=dia,
            n_x=longitudinal.n_x, n_y=longitudinal.n_y,
            supported_all=(crossties or not need_ct))
        levels = self._tie_levels(z0, z1, ties.spacing, ties.hinge_spacing,
                                  ties.hinge_length)
        stirrups = tuple(
            Stirrup.rect(bx, by, cover, db=ties.db, material=ties.material,
                         z=z, plane="xy", origin=origin, db_value=dia_tie,
                         closure_hook=ties.hook)
            for z in levels)
        if need_ct and crossties:
            if confinement_style == "overlapping_hoops":
                stirrups = stirrups + tuple(self._column_overlapping_hoops(
                    xs, ys, ox, oy, levels, db=ties.db, material=ties.material))
            else:                                  # straight cross-tie legs
                bars = bars + tuple(self._column_crossties(
                    xs, ys, ox, oy, levels, db=ties.db, material=ties.material))
        return Cage(bars=bars, stirrups=stirrups, standard=std)

    def beam(self, *, section, length: float, cover: float, top: BarLayout,
             bottom: BarLayout, stirrups: TieLayout, base_x: float = 0.0,
             origin: tuple[float, float] = (0.0, 0.0), standard=None,
             end_cover: float | None = None, crossties: bool = True) -> Cage:
        """Build a rectangular RC beam cage (horizontal, x-axis): top +
        bottom longitudinal bars (count = ``BarLayout.n_x``; ``n_y`` is
        ignored) + vertical stirrups (y-z plane) densified in the end hinge
        zones. Bars/stirrups are inset interior (see :meth:`column`).

        ``crossties=True`` (default) adds a vertical supplementary leg at
        every stirrup station for each intermediate (``n>2``) bar pair —
        connecting an interior top bar to the interior bottom bar at the
        same position so both are laterally supported (ACI 318 §25.7.2.3).
        Legs are emitted only for index-aligned interior pairs; if the top
        and bottom bar counts differ, the unpaired interior bars are
        skipped with a warning. Set ``crossties=False`` for the bare
        perimeter stirrup.

        **Seismic confinement zone (ACI 318 §18.6.4).** When the active
        standard is ``ACI318_seismic`` and the ``stirrups`` :class:`TieLayout`
        leaves ``hinge_spacing``/``hinge_length`` unset, the hoop zone length
        ``2h`` and dense spacing min(d/4, 6·d_b, 6 in) are auto-derived from
        the section (a warning reports the values); ``stirrups.spacing`` is
        used outside the hinge zones. An explicit hinge layout overrides, and
        a non-seismic standard leaves the spacing uniform."""
        kind, width, height = self._rect_section(section, "beam")
        self._require_positive(length, "length", "beam")
        std = standard if standard is not None else self._standard
        dia_st = (stirrups.db_value if stirrups.db_value is not None
                  else self._dia(std, stirrups.db))
        oy, oz = origin
        ec = cover if end_cover is None else end_cover
        x0, x1 = base_x + ec, base_x + length - ec
        if x1 <= x0:
            raise ValueError(
                f"g.rebar.beam: end_cover {ec} too large for length {length}.")
        bars: list[Bar] = []
        face_ys: dict[bool, list[float]] = {}     # at_top -> bar y positions
        face_z: dict[bool, float] = {}            # at_top -> bar z
        face_dia: dict[bool, float] = {}          # at_top -> bar diameter
        for layout, at_top in ((top, True), (bottom, False)):
            dia = self._dia(std, layout.db)
            inset_y = cover + dia_st + dia / 2.0      # bars nest inside stirrups
            if 2.0 * inset_y >= width:
                raise ValueError(
                    f"g.rebar.beam: cover+tie+db/2={inset_y} too large for "
                    f"width {width} ({'top' if at_top else 'bottom'} bars "
                    f"would cross).")
            ys = self._linspace(oy + inset_y, oy + width - inset_y, layout.n_x)
            z = (oz + height - cover - dia_st - dia / 2.0 if at_top
                 else oz + cover + dia_st + dia / 2.0)
            face_ys[at_top], face_z[at_top] = ys, z
            face_dia[at_top] = dia
            for y in ys:
                bars.append(Bar(
                    path=Path(((x0, y, z), (x1, y, z))),
                    db=layout.db, material=layout.material,
                    role="top" if at_top else "bottom"))
        stirrups = self._seismic_confinement_beam(
            stirrups, std, height=height, cover=cover, dia_st=dia_st,
            dia_top=face_dia[True], dia_bottom=face_dia[False])
        stations = self._tie_levels(x0, x1, stirrups.spacing,
                                    stirrups.hinge_spacing, stirrups.hinge_length)
        sts = tuple(
            Stirrup.rect(width, height, cover, db=stirrups.db,
                         material=stirrups.material, z=x, plane="yz",
                         origin=origin, db_value=dia_st,
                         closure_hook=stirrups.hook)
            for x in stations)
        if crossties and (top.n_x > 2 or bottom.n_x > 2):
            bars.extend(self._beam_crossties(
                top, bottom, face_ys, face_z, stations,
                db=stirrups.db, material=stirrups.material))
        return Cage(bars=tuple(bars), stirrups=sts, standard=std)

    def circular_column(self, *, diameter: float, height: float, cover: float,
                        n_bars: int, bar_db, bar_material: str = "rebar",
                        ties: TieLayout, base_z: float = 0.0,
                        origin: tuple[float, float] = (0.0, 0.0), standard=None,
                        top_hook: Hook | None = None,
                        bottom_hook: Hook | None = None,
                        end_cover: float | None = None, spiral: bool = False,
                        n_segments: int = 24) -> Cage:
        """Build a circular RC column cage (vertical, z-axis): ``n_bars``
        longitudinal bars evenly spaced on a circle + circular confinement.

        Transverse reinforcement is either discrete **circular hoops**
        (``spiral=False``, default — one closed ring per tie level, densified
        in the hinge zones like the rectangular column) or a continuous
        **spiral** (``spiral=True`` — one helix from end to end at pitch
        ``ties.spacing``). Rings/helix are polygon-approximated with
        ``n_segments`` sides per turn. Circular confinement laterally
        supports every bar, so there are no cross-ties.

        Bars sit on radius ``D/2 − cover − tie − db/2`` and the hoop
        centerline on ``D/2 − cover − tie/2``; the cage is interior to the
        host (both couplings mesh). When the standard is ``ACI318_seismic``
        and ``ties`` omits the hinge fields, the §18.7.5 confinement zone is
        auto-derived (``h_x`` = the bar chord spacing on the circle).
        """
        self._require_positive(diameter, "diameter", "circular_column")
        self._require_positive(height, "height", "circular_column")
        if not isinstance(n_bars, int) or isinstance(n_bars, bool) or n_bars < 3:
            raise ValueError(
                f"g.rebar.circular_column: n_bars must be an int ≥ 3, "
                f"got {n_bars!r}.")
        if not isinstance(n_segments, int) or n_segments < 8:
            raise ValueError(
                f"g.rebar.circular_column: n_segments must be an int ≥ 8, "
                f"got {n_segments!r}.")
        std = standard if standard is not None else self._standard
        dia = self._dia(std, bar_db)
        dia_tie = (ties.db_value if ties.db_value is not None
                   else self._dia(std, ties.db))
        ox, oy = origin
        r_long = diameter / 2.0 - cover - dia_tie - dia / 2.0
        r_tie = diameter / 2.0 - cover - dia_tie / 2.0
        if r_long <= 0.0:
            raise ValueError(
                f"g.rebar.circular_column: cover+tie+db/2 too large for "
                f"diameter {diameter} (bar circle radius {r_long} ≤ 0).")
        ec = cover if end_cover is None else end_cover
        z0, z1 = base_z + ec, base_z + height - ec
        if z1 <= z0:
            raise ValueError(
                f"g.rebar.circular_column: end_cover {ec} too large for "
                f"height {height}.")
        bars = tuple(
            Bar(path=Path(((ox + r_long * math.cos(t), oy + r_long * math.sin(t),
                            z0),
                           (ox + r_long * math.cos(t), oy + r_long * math.sin(t),
                            z1))),
                db=bar_db, material=bar_material,
                start_hook=bottom_hook, end_hook=top_hook)
            for t in (2.0 * math.pi * k / n_bars for k in range(n_bars)))
        # seismic confinement zone (hx = chord between adjacent bars)
        ties = self._seismic_confinement_circular(
            ties, std, diameter=diameter, height=height, r_long=r_long,
            n_bars=n_bars, dia=dia)
        if spiral:
            helix = self._helix_points(ox, oy, r_tie, z0, z1, ties.spacing,
                                       n_segments)
            stirrups = (Bar(path=Path(tuple(helix)), db=ties.db,
                            material=ties.material, role="spiral",
                            element="truss"),)
            return Cage(bars=bars + stirrups, standard=std)
        levels = self._tie_levels(z0, z1, ties.spacing, ties.hinge_spacing,
                                  ties.hinge_length)
        hoops = tuple(
            Stirrup(path=Path(self._circle_points(ox, oy, r_tie, z, n_segments)),
                    db=ties.db, material=ties.material, role="tie",
                    closure_hook=ties.hook or Hook.seismic_135())
            for z in levels)
        return Cage(bars=bars, stirrups=hoops, standard=std)

    @staticmethod
    def _circle_points(cx: float, cy: float, r: float, z: float,
                       n: int) -> tuple[Vec3, ...]:
        """Closed ring of ``n`` points on a circle (first == last so the loop
        welds into one node ring)."""
        pts = [(cx + r * math.cos(2.0 * math.pi * k / n),
                cy + r * math.sin(2.0 * math.pi * k / n), z)
               for k in range(n)]
        pts.append(pts[0])
        return tuple(pts)

    @staticmethod
    def _helix_points(cx: float, cy: float, r: float, z0: float, z1: float,
                      pitch: float, n_seg: int) -> list[Vec3]:
        """Open helix from ``z0`` to ``z1`` at ``pitch`` per turn, ``n_seg``
        sides per turn — the spiral centerline as a polyline."""
        if pitch <= 0:
            raise ValueError("g.rebar.circular_column: spiral pitch (ties."
                             "spacing) must be > 0.")
        dz = pitch / n_seg
        n_steps = max(n_seg, int(math.ceil((z1 - z0) / dz)))
        pts: list[Vec3] = []
        for i in range(n_steps + 1):
            z = min(z0 + i * dz, z1)
            t = 2.0 * math.pi * i / n_seg
            pts.append((cx + r * math.cos(t), cy + r * math.sin(t), z))
            if z >= z1:
                break
        return pts

    def _seismic_confinement_circular(self, ties: TieLayout, std, *,
                                      diameter: float, height: float,
                                      r_long: float, n_bars: int,
                                      dia: float) -> TieLayout:
        """§18.7.5 confinement zone for a circular column (h_x = the chord
        spacing between adjacent bars on the circle). No-op without a seismic
        standard or when the hinge layout is explicit."""
        if not hasattr(std, "confinement_length"):
            return ties
        if ties.hinge_spacing is not None and ties.hinge_length is not None:
            return ties
        hx = 2.0 * r_long * math.sin(math.pi / n_bars)
        l_o = std.confinement_length(member_depth=diameter, clear_span=height)
        s_o = std.confinement_spacing(min_member_dim=diameter, db_long=dia,
                                      hx=hx)
        warnings.warn(
            f"g.rebar.circular_column: ACI318_seismic confinement zone "
            f"auto-derived — l_o={l_o:.4g}, s_o={s_o:.4g} (ACI 318 §18.7.5, "
            f"h_x={hx:.4g}). Pass TieLayout(hinge_length=, hinge_spacing=) to "
            f"override.", stacklevel=3)
        return replace(ties, hinge_length=l_o, hinge_spacing=s_o)

    def _beam_crossties(self, top: BarLayout, bottom: BarLayout,
                        face_ys: dict, face_z: dict, stations,
                        *, db, material) -> list[Bar]:
        """Vertical supplementary legs for the intermediate beam bars. Each
        leg connects an index-aligned interior top bar to the interior
        bottom bar (engaging both) at a stirrup station. When the top and
        bottom counts differ, only the inner ``min(n)-2`` interior pairs are
        supported and the rest are warned about."""
        n_pair = min(top.n_x, bottom.n_x) - 2
        if top.n_x != bottom.n_x:
            warnings.warn(
                "g.rebar.beam: top and bottom bar counts differ; vertical "
                "cross-ties are generated only for the inner "
                f"{max(n_pair, 0)} index-aligned interior bar pair(s) — "
                "verify supplementary legs for the unpaired bars (ACI 318 "
                "§25.7.2.3).", stacklevel=3)
        if n_pair <= 0:
            return []
        top_ys, bot_ys = face_ys[True], face_ys[False]
        top_z, bot_z = face_z[True], face_z[False]
        out: list[Bar] = []
        for s_idx, x in enumerate(stations):
            for k in range(n_pair):
                out.append(self._crosstie_bar(
                    (x, bot_ys[1 + k], bot_z), (x, top_ys[1 + k], top_z),
                    db=db, material=material, level_idx=s_idx, leg_idx=k))
        return out

    @staticmethod
    def _require_positive(v, nm: str, who: str) -> None:
        if (not isinstance(v, (int, float)) or isinstance(v, bool)
                or not v > 0):
            raise ValueError(f"g.rebar.{who}: {nm} must be > 0, got {v!r}.")

    # ---- layout helpers ---------------------------------------------
    @staticmethod
    def _rect_section(section, who: str):
        if (not isinstance(section, (tuple, list)) or len(section) != 3
                or section[0] != "rect"):
            raise ValueError(
                f"g.rebar.{who}: section must be ('rect', b1, b2), "
                f"got {section!r}.")
        return section[0], float(section[1]), float(section[2])

    @staticmethod
    def _linspace(a: float, b: float, n: int) -> list[float]:
        if n < 1:
            raise ValueError("g.rebar: bar count must be ≥ 1.")
        if n == 1:
            return [(a + b) / 2.0]
        return [a + i * (b - a) / (n - 1) for i in range(n)]

    @staticmethod
    def _perimeter(xs: list[float], ys: list[float]):
        pts: list[tuple[float, float]] = []
        seen: set = set()

        def add(x: float, y: float) -> None:
            k = (round(x, 9), round(y, 9))
            if k not in seen:
                seen.add(k)
                pts.append((x, y))

        for x in xs:                       # bottom + top faces
            add(x, ys[0])
            add(x, ys[-1])
        for y in ys[1:-1]:                 # left + right faces (interior)
            add(xs[0], y)
            add(xs[-1], y)
        return pts

    @staticmethod
    def _tie_levels(a: float, b: float, spacing: float,
                    hinge_spacing: float | None,
                    hinge_length: float | None) -> list[float]:
        span = b - a
        if span <= 0:
            raise ValueError(f"g.rebar: tie span must be > 0, got {span}.")
        if spacing <= 0:
            raise ValueError("g.rebar: tie spacing must be > 0.")
        if hinge_spacing is None or hinge_length is None:
            n = max(1, round(span / spacing))
            return [a + i * span / n for i in range(n + 1)]
        if 2.0 * hinge_length >= span:
            # hinge zones cover the whole member → fully confined (one dense
            # zone, no rings outside the span).
            n = max(1, round(span / hinge_spacing))
            return [a + i * span / n for i in range(n + 1)]
        levels: list[float] = []
        z = a
        while z < a + hinge_length - 1e-9:        # bottom hinge (dense)
            levels.append(z)
            z += hinge_spacing
        z = a + hinge_length
        while z < b - hinge_length - 1e-9:        # middle (regular)
            levels.append(z)
            z += spacing
        z = b - hinge_length
        while z <= b + 1e-9:                       # top hinge (dense)
            levels.append(z)
            z += hinge_spacing
        # clamp: never emit a ring outside the member span [a, b]
        return sorted({round(v, 9) for v in levels if a - 1e-9 <= v <= b + 1e-9})

    # ---- seismic confinement zone (ACI 318 §18.7.5) ----------------
    @staticmethod
    def _seismic_confinement(ties: TieLayout, std, *, bx: float, by: float,
                             height: float, inset: float, dia: float,
                             n_x: int, n_y: int,
                             supported_all: bool) -> TieLayout:
        """Auto-derive the §18.7.5 confined-end hinge length ``l_o`` and
        dense tie spacing ``s_o`` from the column geometry when a seismic
        standard is set and the :class:`TieLayout` left them unspecified.
        A standard that does not expose ``confinement_length`` (e.g. plain
        ``ACI318``/``Raw``) is a no-op, and an explicit hinge layout is
        respected verbatim — so the user can always override the code rule.

        ``h_x`` (the §18.7.5.3 bar support spacing) is the maximum bar
        centre-to-centre spacing when every bar is laterally supported
        (cross-ties present, or a corner-only 2×2 cage); with intermediate
        bars left unsupported (``crossties=False``) only the corners count,
        so ``h_x`` is the full clear distance — a larger, more demanding
        spacing."""
        if not hasattr(std, "confinement_length"):
            return ties
        if ties.hinge_spacing is not None and ties.hinge_length is not None:
            return ties
        clear_x, clear_y = bx - 2.0 * inset, by - 2.0 * inset
        hx = (max(clear_x / (n_x - 1), clear_y / (n_y - 1)) if supported_all
              else max(clear_x, clear_y))
        l_o = std.confinement_length(member_depth=max(bx, by),
                                     clear_span=height)
        s_o = std.confinement_spacing(min_member_dim=min(bx, by),
                                      db_long=dia, hx=hx)
        warnings.warn(
            f"g.rebar.column: ACI318_seismic confinement zone auto-derived — "
            f"l_o={l_o:.4g}, s_o={s_o:.4g} (ACI 318 §18.7.5.2/.3, h_x="
            f"{hx:.4g}). Pass TieLayout(hinge_length=, hinge_spacing=) to "
            f"override.", stacklevel=3)
        return replace(ties, hinge_length=l_o, hinge_spacing=s_o)

    @staticmethod
    def _seismic_confinement_beam(stirrups: TieLayout, std, *, height: float,
                                  cover: float, dia_st: float, dia_top: float,
                                  dia_bottom: float) -> TieLayout:
        """Auto-derive the §18.6.4 special-beam hoop zone (length ``2h``,
        spacing min(d/4, 6·d_b, 6 in)) when a seismic standard is set and the
        :class:`TieLayout` left the hinge fields unset. A standard without
        ``beam_confinement_length`` (plain ``ACI318``/``Raw``) is a no-op, and
        an explicit hinge layout is respected. ``d_b`` is the smallest
        longitudinal bar; the effective depth ``d`` is taken to the tension
        (bottom) bar centroid."""
        if not hasattr(std, "beam_confinement_length"):
            return stirrups
        if (stirrups.hinge_spacing is not None
                and stirrups.hinge_length is not None):
            return stirrups
        eff_depth = height - cover - dia_st - dia_bottom / 2.0
        l_h = std.beam_confinement_length(member_depth=height)
        s_h = std.beam_confinement_spacing(
            eff_depth=eff_depth, db_long=min(dia_top, dia_bottom))
        warnings.warn(
            f"g.rebar.beam: ACI318_seismic hoop confinement zone auto-derived "
            f"— length=2h={l_h:.4g}, spacing={s_h:.4g} (ACI 318 §18.6.4). Pass "
            f"TieLayout(hinge_length=, hinge_spacing=) to override.",
            stacklevel=3)
        return replace(stirrups, hinge_length=l_h, hinge_spacing=s_h)

    # ---- cross-ties / supplementary legs (ACI 318 §25.7.2.3) --------
    @staticmethod
    def _crosstie_bar(p0, p1, *, db, material, level_idx: int, leg_idx: int):
        """A straight transverse cross-tie leg between two longitudinal-bar
        positions ``p0``→``p1``, hooked at both ends. One end gets a 135°
        seismic hook, the other a 90° hook; which end carries which is
        flipped on consecutive levels (``level_idx``) and adjacent legs
        (``leg_idx``) so consecutive cross-ties engaging the same bar
        alternate end-for-end (ACI 318 §18.7.5.2). The tails resolve from
        the cage's DetailingStandard at ``place`` time (seismic-hoop kind);
        with no standard they are dropped (warned), leaving the bare leg."""
        flip = (level_idx + leg_idx) % 2 == 1
        h_seismic, h_90 = Hook.standard_135(), Hook.standard_90()
        start_hook, end_hook = (h_90, h_seismic) if flip else (h_seismic, h_90)
        return Bar(
            path=Path((tuple(float(c) for c in p0),
                       tuple(float(c) for c in p1))),
            db=db, material=material, role="crosstie", element="truss",
            start_hook=start_hook, end_hook=end_hook)

    def _column_crossties(self, xs, ys, ox: float, oy: float, levels,
                          *, db, material) -> list[Bar]:
        """Cross-ties for every intermediate perimeter bar, at every tie
        level. An intermediate top/bottom bar (interior ``x``) is engaged by
        a leg spanning the section in ``y`` (bottom↔top); an intermediate
        left/right bar (interior ``y``) by a leg spanning in ``x``
        (left↔right). Each leg engages the two opposite-face bars it
        connects, so supporting every intermediate bar trivially satisfies
        the §25.7.2.3 "every corner and alternate bar" rule."""
        interior_xs, interior_ys = xs[1:-1], ys[1:-1]
        out: list[Bar] = []
        for z_idx, z in enumerate(levels):
            leg = 0
            for x in interior_xs:                  # bottom↔top (spans y)
                out.append(self._crosstie_bar(
                    (ox + x, oy + ys[0], z), (ox + x, oy + ys[-1], z),
                    db=db, material=material, level_idx=z_idx, leg_idx=leg))
                leg += 1
            for y in interior_ys:                  # left↔right (spans x)
                out.append(self._crosstie_bar(
                    (ox + xs[0], oy + y, z), (ox + xs[-1], oy + y, z),
                    db=db, material=material, level_idx=z_idx, leg_idx=leg))
                leg += 1
        return out

    def _column_overlapping_hoops(self, xs, ys, ox: float, oy: float, levels,
                                  *, db, material) -> list[Stirrup]:
        """Closed overlapping cell-hoops tiling the core, at every tie level.
        One rectangular hoop per adjacent 2×2 bar block — corners on the
        longitudinal bar centerlines — so neighbouring cells share an edge
        (overlap) and every bar sits at a hoop corner. The wide-section
        alternative to straight cross-ties; emitted alongside the outer
        perimeter hoop."""
        out: list[Stirrup] = []
        for z in levels:
            for i in range(len(xs) - 1):
                for j in range(len(ys) - 1):
                    corners = (
                        (ox + xs[i],     oy + ys[j],     z),
                        (ox + xs[i + 1], oy + ys[j],     z),
                        (ox + xs[i + 1], oy + ys[j + 1], z),
                        (ox + xs[i],     oy + ys[j + 1], z),
                        (ox + xs[i],     oy + ys[j],     z),
                    )
                    out.append(Stirrup(path=Path(corners), db=db,
                                       material=material, role="tie"))
        return out

    # ---- placement / coupling router --------------------------------
    def place(self, cage: Cage, into: str, *, coupling: str = "conformal",
              per_member_coupling: dict[str, str] | None = None,
              bond: str | None = None, perfect: float | None = None,
              kt=None, kt_alpha=None, enforce: str = "penalty",
              bipenalty: bool = False, dtcr=None, tolerance: float = 1.0e-6,
              snap: bool = False, host_dim: int | None = None,
              true_arc: bool = False, on_conformal_infeasible: str = "fail",
              twin_tail: bool = True, name: str | None = None) -> RebarPlacement:
        """Emit the cage geometry and couple each member to host ``into``.

        ``coupling="conformal"`` embeds the bar curves into the host so the
        mesh conforms (shared nodes, perfect bond). ``coupling="embedded"``
        meshes the bars independently and forwards to ``g.reinforce`` (→
        ``LadrunoEmbeddedRebar``); it needs ``bond=`` (a ``LadrunoBondSlip``
        material name) **or** ``perfect=`` (a perfect-bond axial penalty).
        ``per_member_coupling={role: coupling}`` overrides per role for
        **mixed** cages (e.g. longitudinal conformal + ties embedded).

        ``twin_tail=True`` (default) emits a real hoop/tie seam: both free
        ends of a closed stirrup carry the closure hook (two tails overlap
        at the seam corner). Set ``twin_tail=False`` for the simplified
        single closure hook. A stirrup with an explicit start hook, or one
        whose closure was dropped (no standard), is unaffected.
        """
        chain_phase_guard(self._parent, "g.rebar.place")
        if not isinstance(cage, Cage):
            raise TypeError(
                f"g.rebar.place: cage must be a Cage, got {type(cage).__name__}."
            )
        if coupling not in ("conformal", "embedded"):
            raise ValueError(
                f"g.rebar.place: coupling must be 'conformal' or 'embedded', "
                f"got {coupling!r}."
            )
        if on_conformal_infeasible not in ("fail", "embedded"):
            raise ValueError(
                f"g.rebar.place: on_conformal_infeasible must be 'fail' or "
                f"'embedded', got {on_conformal_infeasible!r}."
            )
        pmc = per_member_coupling or {}
        std = cage.standard if cage.standard is not None else self._standard
        rein_kw = dict(bond=bond, perfect=perfect, kt=kt, kt_alpha=kt_alpha,
                       enforce=enforce, bipenalty=bipenalty, dtcr=dtcr,
                       tolerance=tolerance, snap=snap)
        # Pass 0 — validate EVERYTHING (cage + host) before mutating gmsh, so a
        # bad cage never leaves the model half-emitted.
        plan = self._plan(cage, into, default_coupling=coupling,
                          per_member_coupling=pmc, std=std, rein_kw=rein_kw,
                          on_conformal_infeasible=on_conformal_infeasible,
                          host_dim=host_dim, name=name, true_arc=true_arc,
                          twin_tail=twin_tail)
        return self._emit_plan(plan, into, rein_kw=rein_kw,
                               on_conformal_infeasible=on_conformal_infeasible)

    # ---- Pass 0: validation + planning (no gmsh mutation) -----------
    def _plan(self, cage: Cage, into: str, *, default_coupling: str,
              per_member_coupling: dict[str, str], std, rein_kw: dict,
              on_conformal_infeasible: str, host_dim: int | None,
              name: str | None, true_arc: bool, twin_tail: bool = True) -> dict:
        in_dim = host_dim if host_dim is not None else self._detect_host_dim(into)
        host_tags = resolve_to_tags(into, dim=in_dim, session=self._parent)
        base = name or f"rebar{self._place_seq}"

        planned: list = []
        roles_seen: set[str] = set()
        names_seen: set[str] = set()
        has_conf = has_emb = False
        idx = 0
        for default_role, items, is_stirrup in (
                ("longitudinal", cage.bars, False),
                ("tie", cage.stirrups, True)):
            for m in items:
                role = getattr(m, "role", default_role)
                roles_seen.add(role)
                eff = per_member_coupling.get(role, default_coupling)
                if eff not in ("conformal", "embedded"):
                    raise ValueError(
                        f"g.rebar.place: per_member_coupling[{role!r}]={eff!r} "
                        f"must be 'conformal' or 'embedded'."
                    )
                key = m.name or f"{role}_{idx}"
                if key in names_seen:
                    raise ValueError(
                        f"g.rebar.place: duplicate member identity {key!r}; "
                        f"member names must be unique within a cage."
                    )
                names_seen.add(key)
                pg = f"{base}.{key}"
                if self._is_physical_group(pg):
                    raise ValueError(
                        f"g.rebar.place: physical group {pg!r} already exists "
                        f"(name collision across placements); pass a distinct "
                        f"name= or member name."
                    )
                elem = getattr(m, "element", "truss")
                if elem == "beam" and (
                        len(m.path.points) > 2 or m.start_hook is not None
                        or m.end_hook is not None
                        or getattr(m, "closure_hook", None) is not None):
                    raise NotImplementedError(
                        "g.rebar: element='beam' on a curved/hooked bar needs "
                        "the ADR-0010 Phase-4 orientation fan-out (not yet "
                        "wired); use element='truss' or a straight bar."
                    )
                if is_stirrup:
                    pts = m.path.points
                    distinct = pts[:-1] if pts[0] == pts[-1] else pts
                    if len(set(distinct)) < 3:
                        raise ValueError(
                            f"g.rebar: stirrup {key!r} closed loop needs ≥3 "
                            f"distinct corners, got {len(set(distinct))}."
                        )
                if eff == "embedded":
                    self._check_embedded_args(rein_kw["bond"], rein_kw["perfect"],
                                              member=key)
                    has_emb = True
                else:
                    has_conf = True
                dia = self._dia(std, m.db)
                # Transverse members (ties, cross-ties) detail their end
                # hooks as seismic hoops and tolerate a missing standard;
                # longitudinal-bar development hooks stay primary + required.
                transverse = is_stirrup or role in _TRANSVERSE_ROLES
                hk_kind = "seismic_hoop" if transverse else "primary"
                hk_required = not transverse
                closure = (self._resolve_hook(
                    std, getattr(m, "closure_hook", None), dia,
                    "seismic_hoop", required=False, true_arc=true_arc)
                    if is_stirrup else None)
                start = self._resolve_hook(
                    std, getattr(m, "start_hook", None), dia, hk_kind,
                    required=hk_required, true_arc=true_arc)
                # Twin-tail: a real hoop/tie is bent from straight stock, so
                # BOTH free ends carry the closure hook (the two tails overlap
                # at the seam corner). When a stirrup leaves its start end free
                # (no explicit start_hook) and twin_tail is on, mirror the
                # closure detail onto the start so the seam shows two tails,
                # not one.
                if is_stirrup and twin_tail and start is None:
                    start = closure
                hooks = {
                    "start": start,
                    "end": self._resolve_hook(
                        std, getattr(m, "end_hook", None), dia, hk_kind,
                        required=hk_required, true_arc=true_arc),
                    "closure": closure,
                }
                planned.append((role, eff, m, pg, elem, dia,
                                self._area(std, m.db), hooks))
                idx += 1

        for k in per_member_coupling:
            if k not in roles_seen:
                warnings.warn(
                    f"g.rebar.place: per_member_coupling key {k!r} matches no "
                    f"member role {sorted(roles_seen)}; ignored.", stacklevel=3)

        host_tag = host_tags[0] if host_tags else None
        if has_conf:
            if len(host_tags) != 1:
                raise ValueError(
                    f"g.rebar.place: conformal coupling needs a single host "
                    f"volume; {into!r} resolved to {len(host_tags)} entities. "
                    f"Name one volume or use coupling='embedded'."
                )
            if self._host_is_meshed(in_dim, host_tag):
                raise RuntimeError(
                    "g.rebar.place: conformal coupling must run BEFORE "
                    "g.mesh.generation.generate() — embedding into an already-"
                    "meshed host is a silent no-op."
                )
            self._reject_foreign_part(into)
            if on_conformal_infeasible == "embedded":
                self._check_embedded_args(rein_kw["bond"], rein_kw["perfect"],
                                          member="conformal-fallback")
        if has_emb:
            if not self._is_physical_group(into):
                raise ValueError(
                    f"g.rebar.place: embedded coupling needs host {into!r} to "
                    f"be a physical group (e.g. g.physical.add_volume(...)); a "
                    f"bare geometry label is not resolvable by g.reinforce."
                )
            warnings.warn(
                "g.rebar.place: embedded coupling uses LadrunoEmbeddedRebar, "
                "which is single-process today; partitioned/MPI models must "
                "use coupling='conformal'.", stacklevel=3)

        # Centroid for "centroid"/"in"/"out" hook turn directions — the host
        # volume's centre of mass (hooks bend toward the section core).
        centroid = None
        any_hook = any(any(v is not None for v in hk.values())
                       for *_rest, hk in planned)
        if any_hook and host_tag is not None:
            try:
                com = self._parent.model.queries.center_of_mass(
                    host_tag, dim=in_dim)
                centroid = tuple(float(c) for c in com)
            except Exception:
                centroid = None

        self._place_seq += 1
        return dict(base=base, in_dim=in_dim, host_tag=host_tag,
                    planned=planned, centroid=centroid)

    def _resolve_hook(self, std, hook, dia, kind: str, *, required: bool,
                      true_arc: bool):
        """Resolve a hook to numeric tail+bend_radius at bind time. Returns
        None for an absent hook (or a defaulted stirrup closure when no
        standard is available). A global ``true_arc`` forces arc geometry."""
        if hook is None:
            return None
        if true_arc and not hook.true_arc:
            hook = replace(hook, true_arc=True)
        if std is not None:
            return std.resolve_hook(hook, dia, kind=kind)
        # No standard: only a fully-numeric hook is self-resolving.
        numeric = (isinstance(hook.tail, (int, float))
                   and not isinstance(hook.tail, bool)
                   and isinstance(hook.bend_radius, (int, float))
                   and not isinstance(hook.bend_radius, bool))
        if numeric:
            return hook
        if required:
            raise ValueError(
                "g.rebar: a hook needs a DetailingStandard "
                "(g.rebar.use_standard(...) or Cage(standard=...)) or a "
                "fully-numeric tail + bend_radius."
            )
        warnings.warn(
            "g.rebar: a transverse hook (stirrup closure / cross-tie) was "
            "dropped — no DetailingStandard to resolve its tail, so the tie is "
            "emitted un-anchored. Set g.rebar.use_standard(...) for a seismic "
            "135° hook.", stacklevel=4)
        return None                      # defaulted transverse hook, no std

    # ---- emit (mutates gmsh; all inputs pre-validated) --------------
    def _emit_plan(self, plan: dict, into: str, *, rein_kw: dict,
                   on_conformal_infeasible: str) -> RebarPlacement:
        g = self._parent
        geom = g.model.geometry
        base, in_dim, host_tag = plan["base"], plan["in_dim"], plan["host_tag"]
        centroid = plan["centroid"]

        # Pass 1 — emit all curve geometry (no PGs yet), including hooks.
        emitted: list = []
        arc_centers: list[int] = []
        for role, eff, m, pg, elem, dia, area, hooks in plan["planned"]:
            lts, pts = self._emit_polyline(geom, m.path.points)
            pts_xyz = m.path.points
            for slot, anchor_tag, anchor_xyz, at_start in (
                    ("start", pts[0], pts_xyz[0], True),
                    ("end", pts[-1], pts_xyz[-1], False),
                    ("closure", pts[-1], pts_xyz[-1], False)):
                if hooks[slot] is None:
                    continue
                t, centers = self._emit_hook(
                    geom, anchor_tag, anchor_xyz,
                    outward_tangent(pts_xyz, at_start=at_start),
                    hooks[slot], centroid)
                lts += t
                arc_centers += centers
            emitted.append((role, eff, m, pg, elem, dia, area, lts))
        # Drop the arc-center construction points so they don't survive as
        # stray meshed nodes inside the host (occ bakes the arc geometry).
        if arc_centers:
            gmsh.model.occ.remove([(0, c) for c in arc_centers], recursive=False)
        # Sync once so the curve entities exist before we wrap them in PGs.
        g.model.sync()

        # Pass 2 — physical groups + coupling registration.
        members: list[RebarMember] = []
        conformal_tags: list[int] = []
        conformal_specs: list = []
        for role, eff, m, pg, elem, dia, area, lts in emitted:
            g.physical.add_curve(lts, name=pg)
            member = RebarMember(
                pg=pg, role=role, db=m.db, diameter=dia, area=area,
                material=m.material, element=elem, coupling=eff,
                line_tags=tuple(lts),
            )
            members.append(member)
            if eff == "conformal":
                conformal_tags.extend(lts)
                conformal_specs.append((member, dia, area))
            else:
                self._register_embedded(into, pg, dia, area, **rein_kw)

        if conformal_tags:
            try:
                g.mesh.editing.embed(conformal_tags, host_tag, dim=1, in_dim=in_dim)
            except Exception as exc:                       # embed-time failure
                if on_conformal_infeasible != "embedded":
                    raise
                warnings.warn(
                    f"g.rebar.place: conformal embed failed ({exc}); falling "
                    f"back to embedded coupling for {len(conformal_specs)} "
                    f"member(s).", stacklevel=2,
                )
                members = [mm if mm.coupling == "embedded"
                           else replace(mm, coupling="embedded")
                           for mm in members]
                for member, dia, area in conformal_specs:
                    self._register_embedded(into, member.pg, dia, area, **rein_kw)

        couplings = {mm.coupling for mm in members}
        placement = RebarPlacement(
            name=base, host=into,
            coupling=next(iter(couplings)) if len(couplings) == 1 else "mixed",
            members=tuple(members),
        )
        self.placements.append(placement)
        return placement

    def _register_embedded(self, into: str, pg: str, diameter: float,
                           area: float, *, bond, perfect, kt, kt_alpha,
                           enforce, bipenalty, dtcr, tolerance, snap) -> None:
        """Forward one embedded member to the shipped ``g.reinforce`` binding
        composite (→ ``LadrunoEmbeddedRebar``), then invalidate the FEMData
        cache (ADR §9: a def-append is a broker mutation)."""
        self._parent.reinforce.reinforce(
            host=into, bars=pg, bond=bond, perfect=perfect,
            bar_diameter=diameter, bar_area=area,
            kt=kt, kt_alpha=kt_alpha, enforce=enforce, bipenalty=bipenalty,
            dtcr=dtcr, tolerance=tolerance, snap=snap, name=pg,
        )
        bump = getattr(self._parent, "_bump_fem_counter", None)
        if bump is not None:
            bump()

    # ---- small resolvers / host checks ------------------------------
    @staticmethod
    def _check_embedded_args(bond, perfect, *, member: str) -> None:
        if (bond is None) == (perfect is None):
            raise ValueError(
                f"g.rebar.place: embedded coupling for {member!r} needs "
                f"exactly one of bond=<LadrunoBondSlip name> or "
                f"perfect=<axial penalty>."
            )

    @staticmethod
    def _dia(std, db) -> float:
        if isinstance(db, (int, float)) and not isinstance(db, bool):
            return float(db)
        if std is not None:
            return float(std.bar_diameter(db))
        raise ValueError(
            f"g.rebar: db {db!r} is a designation but no DetailingStandard is "
            f"set; pass a numeric db, a Cage(standard=...), or call "
            f"g.rebar.use_standard(ACI318())."
        )

    @staticmethod
    def _area(std, db) -> float:
        if isinstance(db, (int, float)) and not isinstance(db, bool):
            return math.pi * float(db) ** 2 / 4.0
        if std is not None:
            return float(std.bar_area(db))
        raise ValueError(
            f"g.rebar: db {db!r} is a designation but no DetailingStandard is "
            f"set; pass a numeric db, a Cage(standard=...), or call "
            f"g.rebar.use_standard(ACI318())."
        )

    @staticmethod
    def _is_physical_group(name: str) -> bool:
        for d, t in gmsh.model.getPhysicalGroups():
            try:
                if gmsh.model.getPhysicalName(int(d), int(t)) == name:
                    return True
            except Exception:
                continue
        return False

    @staticmethod
    def _host_is_meshed(in_dim: int, host_tag) -> bool:
        if host_tag is None:
            return False
        try:
            _types, etags, _ = gmsh.model.mesh.getElements(in_dim, host_tag)
            return any(len(t) > 0 for t in etags)
        except Exception:
            return False

    def _reject_foreign_part(self, into: str) -> None:
        parts = getattr(self._parent, "parts", None)
        try:
            labels = parts.labels() if parts is not None else []
        except Exception:
            labels = []
        if into in labels:
            raise ValueError(
                f"g.rebar.place: conformal coupling requires same-session "
                f"authoring, but host {into!r} is a composed Part — use "
                f"coupling='embedded' (ADR 0066 §6.4)."
            )

    # ---- geometry helpers -------------------------------------------
    def _emit_polyline(self, geom, points: tuple[Vec3, ...]):
        """Emit a polyline as gmsh points + line segments. Returns
        ``(line_tags, point_tags)``. A closed loop (first == last) reuses
        the first point so the loop welds into one node ring."""
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
        line_tags = [geom.add_line(pt_tags[i], pt_tags[i + 1], sync=False)
                     for i in range(len(pt_tags) - 1)]
        return line_tags, pt_tags

    def _emit_hook(self, geom, anchor_tag: int, anchor_xyz, tangent,
                   hook: Hook, centroid) -> tuple[list[int], list[int]]:
        """Realise a resolved hook as gmsh curves appended at ``anchor_tag``,
        reusing point tags between primitives so the chain welds without
        make_conformal. Returns ``(curve_tags, arc_center_point_tags)`` — the
        caller deletes the centers (stray construction points)."""
        turn_dir = self._turn_dir(hook.turn, anchor_xyz, centroid)
        prims, fell_back = hook_primitives(
            anchor_xyz, tangent, turn_dir, hook.angle, hook.tail,
            hook.bend_radius, hook.true_arc)
        if fell_back:
            warnings.warn(
                "g.rebar: hook turn direction is collinear with the bar; "
                "picked a deterministic seed bend plane.", stacklevel=3)
        tags: list[int] = []
        centers: list[int] = []
        prev = anchor_tag
        for prim in prims:
            if prim[0] == "line":
                _, _p0, p1 = prim
                end = geom.add_point(float(p1[0]), float(p1[1]), float(p1[2]),
                                     sync=False)
                tags.append(geom.add_line(prev, end, sync=False))
                prev = end
            else:                                    # ("arc", p_start, center, p_end)
                _, _ps, center, p_end = prim
                ct = geom.add_point(float(center[0]), float(center[1]),
                                    float(center[2]), sync=False)
                end = geom.add_point(float(p_end[0]), float(p_end[1]),
                                     float(p_end[2]), sync=False)
                tags.append(geom.add_arc(prev, ct, end, sync=False))
                centers.append(ct)
                prev = end
        return tags, centers

    @staticmethod
    def _turn_dir(turn, anchor, centroid):
        anchor = np.asarray(anchor, dtype=float)
        if isinstance(turn, str):
            tl = turn.lower()
            if tl in ("centroid", "in"):
                return (np.asarray(centroid, float) - anchor
                        if centroid is not None else np.zeros(3))
            if tl == "out":
                return (anchor - np.asarray(centroid, float)
                        if centroid is not None else np.zeros(3))
            axis = _AXIS_TOKENS.get(tl)
            if axis is None:
                raise ValueError(f"g.rebar: unknown turn token {turn!r}.")
            return np.asarray(axis, float)
        return np.asarray(turn, dtype=float)

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
