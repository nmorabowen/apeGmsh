"""
Reinforcement detailing standards + bar catalogue (ADR 0066 §4).

A :class:`DetailingStandard` resolves the *code-aware* numbers a cage
needs — bar diameter/area from a designation, minimum bend diameters,
standard-hook tail lengths — and resolves ``"<k>db"`` length tokens
against a bar diameter. It is bound to a cage at ``place`` time, so the
L1 specs stay unitless, serialisable data (the standard is never baked
into the spec).

Three implementations:

* :class:`Raw` — explicit-only. Delegates diameter/area to the
  :class:`BarCatalog` but raises :class:`DetailingError` on every
  code-derived method (no ACI tables). The escape hatch for "I'll give
  every number myself".
* :class:`ACI318` — ACI 318-19 Table 25.3.1 / 25.3.2 minimum bend
  diameters and standard-hook tail extensions.
* :class:`ACI318_seismic` — adds the seismic 135° hook
  (§18.8.5 / §25.3.4): tail = max(6·d_b, 3 in).

Units: apeGmsh is unit-agnostic. The single unit knob lives on
:class:`BarCatalog` (``unit_length`` = model length units per canonical
inch/mm). The only absolute imperial constants in this module are the
2.5 in / 3 in hook-tail floors, scaled by ``catalog.unit_per_inch``.
Bend-diameter buckets are keyed off the bar diameter **converted to
inches** (unit-safe), never the raw model-unit magnitude.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from .._kernel.defs.rebar import Hook, _parse_db_token

# ── errors ───────────────────────────────────────────────────────────


class DetailingError(ValueError):
    """A detailing rule could not be resolved (unknown designation, a
    code method on :class:`Raw`, an unsupported angle/kind)."""


# ── bar catalogue ────────────────────────────────────────────────────

# ASTM A615 imperial bars: designation number -> (diameter [in], area [in^2]).
_IMPERIAL: dict[int, tuple[float, float]] = {
    3: (0.375, 0.11), 4: (0.500, 0.20), 5: (0.625, 0.31),
    6: (0.750, 0.44), 7: (0.875, 0.60), 8: (1.000, 0.79),
    9: (1.128, 1.00), 10: (1.270, 1.27), 11: (1.410, 1.56),
    14: (1.693, 2.25), 18: (2.257, 4.00),
}
_MM_IN = 25.4
_HASH = re.compile(r"^\s*#\s*(\d+)\s*$")
_MM = re.compile(r"^\s*(\d+(?:\.\d+)?)\s*mm\s*$", re.IGNORECASE)


@dataclass(frozen=True)
class BarCatalog:
    """Maps a bar designation to a (diameter, area) pair in model units.

    ``unit_length`` is the model length per canonical unit:
      * ``base="imperial"`` ⇒ model units per **inch** (model in inches
        ⇒ 1.0; mm ⇒ 25.4; m ⇒ 0.0254).
      * ``base="metric"``   ⇒ model units per **mm** (model in mm ⇒ 1.0;
        m ⇒ 0.001).

    Designations: ``"#8"`` (imperial table), ``"20mm"`` (metric, any
    base), or a raw positive number (already in model units).
    """

    unit_length: float = 1.0
    base: str = "imperial"

    def __post_init__(self) -> None:
        if self.base not in ("imperial", "metric"):
            raise DetailingError(
                f"BarCatalog: base must be 'imperial' or 'metric', "
                f"got {self.base!r}."
            )
        if not isinstance(self.unit_length, (int, float)) or self.unit_length <= 0:
            raise DetailingError(
                f"BarCatalog: unit_length must be > 0, got {self.unit_length!r}."
            )

    @property
    def unit_per_inch(self) -> float:
        return float(self.unit_length) if self.base == "imperial" \
            else float(self.unit_length) * _MM_IN

    @property
    def unit_per_mm(self) -> float:
        return self.unit_per_inch / _MM_IN

    def bar_diameter(self, designation: float | str) -> float:
        """Diameter in model units."""
        if isinstance(designation, str):
            m = _HASH.match(designation)
            if m:
                n = int(m.group(1))
                if n not in _IMPERIAL:
                    raise DetailingError(
                        f"BarCatalog: unknown imperial bar #{n}; valid: "
                        f"{sorted(_IMPERIAL)}."
                    )
                return _IMPERIAL[n][0] * self.unit_per_inch
            m = _MM.match(designation)
            if m:
                return float(m.group(1)) * self.unit_per_mm
            raise DetailingError(
                f"BarCatalog: unrecognised designation {designation!r}; "
                f'use "#N", "<N>mm", or a raw number.'
            )
        if isinstance(designation, bool) or not isinstance(designation, (int, float)):
            raise DetailingError(
                f"BarCatalog: db must be a number or designation string, "
                f"got {type(designation).__name__}."
            )
        if designation <= 0:
            raise DetailingError(f"BarCatalog: db must be > 0, got {designation}.")
        return float(designation)

    def bar_area(self, designation: float | str) -> float:
        """Cross-section area in model units².

        Convention: an imperial ``"#N"`` designation returns the ASTM
        A615 **nominal** area (the design area engineers expect, e.g.
        #8 → 0.79 in²), which differs from π·d²/4 by the nominal-diameter
        rounding (~0.6%). A metric ``"<N>mm"`` designation or a raw-float
        diameter returns the geometric π·d²/4. Both are passed to
        ``ReinforceDef.bar_area`` (which stores diameter and area
        independently), so the (db, As) pair stays ASTM-consistent.
        """
        if isinstance(designation, str):
            m = _HASH.match(designation)
            if m:
                n = int(m.group(1))
                if n not in _IMPERIAL:
                    raise DetailingError(
                        f"BarCatalog: unknown imperial bar #{n}; valid: "
                        f"{sorted(_IMPERIAL)}."
                    )
                return _IMPERIAL[n][1] * self.unit_per_inch ** 2
            # metric / raw: area derived from the resolved diameter
        d = self.bar_diameter(designation)
        return math.pi * d * d / 4.0

    def to_inches(self, db_model: float) -> float:
        """A model-unit diameter expressed in inches (for ACI bucketing)."""
        return db_model / self.unit_per_inch


# ── standard protocol ────────────────────────────────────────────────

_PRIMARY = "primary"
_STIRRUP = "stirrup_tie"
_SEISMIC = "seismic_hoop"
_KINDS = frozenset({_PRIMARY, _STIRRUP, _SEISMIC})


@runtime_checkable
class DetailingStandard(Protocol):
    name: str
    def bar_diameter(self, designation: float | str) -> float: ...
    def bar_area(self, designation: float | str) -> float: ...
    def min_bend_diameter(self, db: float, *, kind: str = _PRIMARY) -> float: ...
    def hook_tail(self, angle: float, db: float, *, kind: str = _PRIMARY) -> float: ...
    def default_corner_radius(self, db: float, *, kind: str = _PRIMARY) -> float: ...
    def resolve_length(self, spec: float | str, db: float) -> float: ...
    def resolve_hook(self, hook: Hook, db: float, *, kind: str = _PRIMARY) -> Hook: ...
    def make_hook(self, kind: str, db: float, *, angle: float) -> Hook: ...


def _check_kind(kind: str, owner: str) -> None:
    if kind not in _KINDS:
        raise DetailingError(
            f"{owner}: kind must be one of {sorted(_KINDS)}, got {kind!r}."
        )


class _BaseStandard:
    """Shared diameter/area (catalogue-delegating) + ``"<k>db"`` resolver."""

    name = "base"

    def __init__(self, catalog: BarCatalog | None = None) -> None:
        self.catalog = catalog if catalog is not None else BarCatalog()

    def bar_diameter(self, designation: float | str) -> float:
        return self.catalog.bar_diameter(designation)

    def bar_area(self, designation: float | str) -> float:
        return self.catalog.bar_area(designation)

    def resolve_length(self, spec: float | str, db: float) -> float:
        k = _parse_db_token(spec)        # shared parser → never drifts from L1
        if k is not None:
            return k * db
        if isinstance(spec, bool) or not isinstance(spec, (int, float)):
            raise DetailingError(
                f"{self.name}: cannot resolve length {spec!r}; expected a "
                f'number or "<k>db" token.'
            )
        return float(spec)


class Raw(_BaseStandard):
    """Explicit-only: diameter/area from the catalogue, but no code-derived
    bend/hook geometry — every such call raises :class:`DetailingError`."""

    name = "Raw"

    def _no(self, what: str):
        raise DetailingError(
            f"Raw: {what} requires a code standard (e.g. ACI318); Raw is "
            f"explicit-only. Supply the number yourself or pick ACI318()."
        )

    def min_bend_diameter(self, db: float, *, kind: str = _PRIMARY) -> float:
        self._no("min_bend_diameter")

    def hook_tail(self, angle: float, db: float, *, kind: str = _PRIMARY) -> float:
        self._no("hook_tail")

    def default_corner_radius(self, db: float, *, kind: str = _PRIMARY) -> float:
        self._no("default_corner_radius")

    def resolve_hook(self, hook: Hook, db: float, *, kind: str = _PRIMARY) -> Hook:
        # Raw can still resolve a hook IF every field is already explicit.
        if hook.tail is None:
            self._no("resolve_hook (tail=None)")
        if hook.bend_radius is None:
            self._no("resolve_hook (bend_radius=None)")
        tail = self.resolve_length(hook.tail, db)
        radius = self.resolve_length(hook.bend_radius, db)
        return Hook(angle=hook.angle, tail=tail, bend_radius=radius,
                    turn=hook.turn, true_arc=hook.true_arc, name=hook.name)

    def make_hook(self, kind: str, db: float, *, angle: float) -> Hook:
        self._no("make_hook")


class ACI318(_BaseStandard):
    """ACI 318-19 minimum bend diameters (Table 25.3.1 / 25.3.2) and
    standard-hook tail extensions (§25.3.1 / §25.3.2)."""

    name = "ACI318"

    def min_bend_diameter(self, db: float, *, kind: str = _PRIMARY) -> float:
        _check_kind(kind, self.name)
        d_in = self.catalog.to_inches(db)
        if kind in (_STIRRUP, _SEISMIC):
            # Table 25.3.2 — stirrups/ties/hoops
            if d_in <= 0.625 + 1e-9:        # #3–#5
                return 4.0 * db
            if d_in <= 1.000 + 1e-9:        # #6–#8
                return 6.0 * db
            # Table 25.3.2 only tabulates transverse reinforcement up to
            # #8; for a tie/hoop larger than #8 fall through to the
            # primary Table 25.3.1 minimum bend diameters (8db/10db).
        # Table 25.3.1 — primary bars
        if d_in <= 1.000 + 1e-9:            # #3–#8
            return 6.0 * db
        if d_in <= 1.410 + 1e-9:            # #9–#11
            return 8.0 * db
        return 10.0 * db                    # #14, #18

    def default_corner_radius(self, db: float, *, kind: str = _PRIMARY) -> float:
        # inside bend radius (scalar; the geometry builder adds db/2 for the
        # centerline when it places the fillet)
        return self.min_bend_diameter(db, kind=kind) / 2.0

    def _tail_floor(self, angle: float, kind: str) -> float:
        """Absolute hook-tail floor (model units): 2.5 in for 180° hooks."""
        upi = self.catalog.unit_per_inch
        if int(round(angle)) == 180:
            return 2.5 * upi
        return 0.0

    @staticmethod
    def _stirrup_tail_multiple(d_in: float) -> float:
        # Table 25.3.2: the stirrup/tie/hoop standard-hook tail extension
        # (BOTH the 90° and 135° rows) is bar-size dependent — 6db for
        # #3–#5, 12db for #6–#8. #6–#8 column ties are common, so the flat
        # 6db used previously under-detailed them by ~2x.
        return 6.0 if d_in <= 0.625 + 1e-9 else 12.0

    def hook_tail(self, angle: float, db: float, *, kind: str = _PRIMARY) -> float:
        _check_kind(kind, self.name)
        a = int(round(angle))
        if kind == _PRIMARY:
            # Table 25.3.1 standard development hooks define only 90° and
            # 180°; there is no 135° standard development hook.
            nominal = {90: 12.0, 180: 4.0}.get(a)
        elif kind == _SEISMIC:
            # §25.3.4 seismic hook: a FLAT 6db extension (not the Table
            # 25.3.2 size split) for the 135° (and 90° circular-hoop)
            # hook; the "not less than 3 in" floor comes from
            # ACI318_seismic._tail_floor.
            nominal = 6.0 if a in (90, 135) else (4.0 if a == 180 else None)
        else:  # _STIRRUP — non-seismic standard tie, Table 25.3.2
            if a in (90, 135):
                nominal = self._stirrup_tail_multiple(self.catalog.to_inches(db))
            elif a == 180:
                nominal = 4.0
            else:
                nominal = None
        if nominal is None:
            raise DetailingError(
                f"{self.name}: no standard hook tail for a {angle}° hook "
                f"(kind={kind}); supply an explicit tail length."
            )
        return max(nominal * db, self._tail_floor(angle, kind))

    def resolve_hook(self, hook: Hook, db: float, *, kind: str = _PRIMARY) -> Hook:
        """Return a fully-numeric Hook: tail honours the code floor, and a
        ``None`` bend_radius is filled from ``min_bend_diameter`` (the
        centerline radius = inside_radius + db/2)."""
        _check_kind(kind, self.name)
        if hook.tail is None:
            tail = self.hook_tail(hook.angle, db, kind=kind)
        else:
            tail = max(self.resolve_length(hook.tail, db),
                       self._tail_floor(hook.angle, kind))
        if hook.bend_radius is None:
            radius = self.min_bend_diameter(db, kind=kind) / 2.0 + db / 2.0
        else:
            radius = self.resolve_length(hook.bend_radius, db)
        return Hook(angle=hook.angle, tail=tail, bend_radius=radius,
                    turn=hook.turn, true_arc=hook.true_arc, name=hook.name)

    def make_hook(self, kind: str, db: float, *, angle: float,
                  turn: str = "centroid", true_arc: bool = False,
                  name: str | None = None) -> Hook:
        """Construct a fully-numeric standard Hook for *kind* at bend
        *angle* and diameter *db* (model units) — the code-aware factory
        the L2 column/beam generators (ADR §8) call."""
        base = Hook(angle=angle, turn=turn, true_arc=true_arc, name=name)
        return self.resolve_hook(base, db, kind=kind)


class ACI318_seismic(ACI318):
    """ACI 318-19 seismic detailing: the 135° hook gets a 3 in tail floor
    (§25.3.4 / §18.8.5). Bend diameters inherit the stirrup/tie table.

    Adds the special-moment-frame column confinement-zone rules
    (§18.7.5) — the hinge length ``l_o`` and the dense tie spacing ``s_o``
    — so a column generator can self-detail its confined ends instead of
    falling back to a uniform spacing."""

    name = "ACI318_seismic"

    def _tail_floor(self, angle: float, kind: str) -> float:
        a = int(round(angle))
        upi = self.catalog.unit_per_inch
        if a == 135:
            return 3.0 * upi
        if a == 180:
            return 2.5 * upi
        return 0.0

    def confinement_length(self, *, member_depth: float,
                           clear_span: float) -> float:
        """ACI 318-19 §18.7.5.2 special-column confinement length ``l_o`` =
        max(member depth, clear span / 6, 18 in), measured from each joint
        face. ``member_depth`` is the section depth in the buckling plane
        (use the larger section dimension for a biaxial column);
        ``clear_span`` is the column clear height. Both in model units."""
        for v, nm in ((member_depth, "member_depth"),
                      (clear_span, "clear_span")):
            if not isinstance(v, (int, float)) or isinstance(v, bool) or v <= 0:
                raise DetailingError(
                    f"{self.name}.confinement_length: {nm} must be > 0, "
                    f"got {v!r}.")
        return max(float(member_depth), clear_span / 6.0,
                   18.0 * self.catalog.unit_per_inch)

    def confinement_spacing(self, *, min_member_dim: float, db_long: float,
                            hx: float) -> float:
        """ACI 318-19 §18.7.5.3 confinement-zone tie spacing ``s_o`` =
        min(¼·min member dimension, 6·d_b of the smallest longitudinal bar,
        ``s_o``) with ``s_o`` = 4 + (14 − h_x)/3 inches clamped to
        [4, 6] in. ``h_x`` is the maximum centre-to-centre horizontal
        spacing of laterally supported longitudinal bars (capped at 14 in
        per §18.7.5.2). All lengths in model units; the s_o equation is
        evaluated in inches (unit-safe via the catalogue)."""
        for v, nm in ((min_member_dim, "min_member_dim"), (db_long, "db_long"),
                      (hx, "hx")):
            if not isinstance(v, (int, float)) or isinstance(v, bool) or v <= 0:
                raise DetailingError(
                    f"{self.name}.confinement_spacing: {nm} must be > 0, "
                    f"got {v!r}.")
        upi = self.catalog.unit_per_inch
        hx_in = min(self.catalog.to_inches(hx), 14.0)   # §18.7.5.2 cap
        so_in = min(6.0, max(4.0, 4.0 + (14.0 - hx_in) / 3.0))
        return min(min_member_dim / 4.0, 6.0 * db_long, so_in * upi)


__all__ = [
    "DetailingError", "BarCatalog", "DetailingStandard",
    "Raw", "ACI318", "ACI318_seismic",
]
