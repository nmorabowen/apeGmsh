"""Result-slot records — the closed catalog of ADR 0098 §4.

Seven categories, at most one occupant each per mesh view; different
categories stack; filling an occupied slot **replaces** the occupant.
The records here carry what §4 puts *inside* the slot (quantity /
component / averaging / pattern) and, on the four colour-mapped
records, the two stored style fields of Amendment 7 (A7.3): ``clim``
and ``cmap``. Both default to ``None`` = the kind's existing default,
so no default changed. They are display state: they never enter field
identity (:func:`slot_field`), so two slots of one quantity still share
one scale, and the view refuses two of them that disagree (§5, A7.3).

Tokens are opaque at this layer: ``quantity`` / ``component`` /
``pattern`` are broker vocabulary strings resolved by S1's realize
mapping (contour averaging → ``ContourStyle.averaging``, the vector
quantity → the vector_glyph / principal_glyph resolver, the line
component → the ``LineForceStyle`` axis machinery). S0 stores them;
it does not interpret them.

The catalog is CLOSED (amended ADR 0094 INV-10): a new category is an
ADR 0098 amendment, not a subclass. Fibers, shell layers, isochrones
and spring-as-its-own-kind do not open an eighth slot.
"""
from __future__ import annotations

import math
import numbers
from dataclasses import dataclass
from typing import Optional


def _require_token(value: object, owner: str, field_name: str) -> None:
    if not isinstance(value, str) or not value:
        raise ValueError(
            f"{owner}.{field_name} must be a non-empty string token; "
            f"got {value!r}."
        )


def _check_style(record: object, owner: str) -> None:
    """Validate and normalise A7.3's ``clim`` / ``cmap`` on a
    colour-mapped record, at assignment (fail closed: a bad value never
    reaches a renderer, nor a snapshot).

    ``clim`` is two finite numbers, low < high, stored as a float
    tuple (a JSON list from a snapshot normalises to the same record).
    ``cmap`` is any name matplotlib resolves — ``'jet'`` included; an
    unknown name raises here, not at first paint.
    """
    clim = getattr(record, "clim")
    if clim is not None:
        if isinstance(clim, (str, bytes)) or not hasattr(clim, "__len__"):
            raise ValueError(
                f"{owner}.clim must be (low, high) or None; got {clim!r}."
            )
        if len(clim) != 2:
            raise ValueError(
                f"{owner}.clim must be two numbers (low, high); got "
                f"{len(clim)}: {clim!r}."
            )
        pair = []
        for v in clim:
            if isinstance(v, bool) or not isinstance(v, numbers.Real):
                raise ValueError(
                    f"{owner}.clim values must be numbers; got {v!r}."
                )
            pair.append(float(v))
        lo, hi = pair
        if not (math.isfinite(lo) and math.isfinite(hi)) or not lo < hi:
            raise ValueError(
                f"{owner}.clim must be finite with low < high; got "
                f"({lo!r}, {hi!r})."
            )
        object.__setattr__(record, "clim", (lo, hi))
    cmap = getattr(record, "cmap")
    if cmap is not None:
        _require_token(cmap, owner, "cmap")
        try:
            import matplotlib
        except ImportError:  # pragma: no cover - plot extra missing
            raise ValueError(
                f"{owner}.cmap={cmap!r} cannot be validated: matplotlib "
                f"is not installed (pip install apeGmsh[plot])."
            ) from None
        if cmap not in matplotlib.colormaps:
            raise ValueError(
                f"{owner}.cmap {cmap!r} is not a colormap matplotlib "
                f"knows (e.g. 'viridis', 'jet', 'coolwarm')."
            )


@dataclass(frozen=True)
class Slot:
    """Base for the seven slot records. Empty — categories share no
    fields; the base exists so a mesh view can say "this is a slot
    record" without enumerating the catalog."""


@dataclass(frozen=True)
class Contour(Slot):
    """One heatmap: quantity + averaged | unaveraged (§4). Colour-mapped.

    ``averaging`` uses the ADR vocabulary; S1 maps ``"unaveraged"`` to
    the existing ``ContourStyle.averaging="discrete"`` token. It is
    contour *display* state, not field identity — it never breaks the
    one-scale match with a Gauss slot of the same quantity (§5).
    """

    quantity: str
    averaging: str = "averaged"
    clim: Optional[tuple[float, float]] = None
    cmap: Optional[str] = None

    def __post_init__(self) -> None:
        _require_token(self.quantity, "Contour", "quantity")
        if self.averaging not in ("averaged", "unaveraged"):
            raise ValueError(
                f"Contour.averaging must be 'averaged' or 'unaveraged'; "
                f"got {self.averaging!r}."
            )
        _check_style(self, "Contour")


@dataclass(frozen=True)
class Vector(Slot):
    """Arrows: one quantity token spanning nodal vector fields AND
    Gauss principal families (§4 — ``vector_glyph`` and
    ``principal_glyph`` collapse to this one slot). Colour-mapped.
    S1's resolver routes the token to the right emit path."""

    quantity: str
    clim: Optional[tuple[float, float]] = None
    cmap: Optional[str] = None

    def __post_init__(self) -> None:
        _require_token(self.quantity, "Vector", "quantity")
        _check_style(self, "Vector")


@dataclass(frozen=True)
class Gauss(Slot):
    """Values at integration points (§4). Colour-mapped; shares the
    contour's scale when the field matches (§5 — same quantity token).
    Gauss values are unaveraged by nature, so there is no averaging
    field here."""

    quantity: str
    clim: Optional[tuple[float, float]] = None
    cmap: Optional[str] = None

    def __post_init__(self) -> None:
        _require_token(self.quantity, "Gauss", "quantity")
        _check_style(self, "Gauss")


@dataclass(frozen=True)
class Line(Slot):
    """Member diagrams — amplitude fill, NOT colour-mapped (§4/§5).

    One component in v1 (N / V / M / torsion); a later widening may put
    several *inside* this one slot — never a second category."""

    component: str

    def __post_init__(self) -> None:
        _require_token(self.component, "Line", "component")


@dataclass(frozen=True)
class Sand(Slot):
    """Sand field of a quantity (§4). Colour-mapped."""

    quantity: str
    clim: Optional[tuple[float, float]] = None
    cmap: Optional[str] = None

    def __post_init__(self) -> None:
        _require_token(self.quantity, "Sand", "quantity")
        _check_style(self, "Sand")


@dataclass(frozen=True)
class Loads(Slot):
    """Applied loads — uniform glyphs, NOT colour-mapped (§4).
    ``pattern`` selects the pattern / stage; ``None`` = the default the
    realize layer resolves."""

    pattern: Optional[str] = None

    def __post_init__(self) -> None:
        if self.pattern is not None:
            _require_token(self.pattern, "Loads", "pattern")


@dataclass(frozen=True)
class Reactions(Slot):
    """Support reactions — uniform glyphs, NOT colour-mapped (§4).
    Nothing inside the slot."""


#: Category name → record type, in the §4 table's row order. This IS
#: the closed catalog: the mesh view derives its slot surface from it.
SLOT_CATALOG: dict[str, type] = {
    "contour": Contour,
    "vector": Vector,
    "gauss": Gauss,
    "line": Line,
    "sand": Sand,
    "loads": Loads,
    "reactions": Reactions,
}

#: The §4 "colour-mapped" *yes* column — the legend law's domain (§5):
#: only these categories emit a colour scale.
COLOUR_MAPPED: frozenset = frozenset({"contour", "vector", "gauss", "sand"})


def slot_field(category: str, record: Slot) -> Optional[str]:
    """The field identity a slot contributes to the legend law (§5).

    "Field matches" = the same quantity token; averaging is display
    state and does not participate. Non-colour-mapped categories emit
    nothing and have no field identity.
    """
    if category not in COLOUR_MAPPED:
        return None
    return record.quantity  # type: ignore[attr-defined]


def slot_style(record: Slot) -> tuple[
    Optional[tuple[float, float]], Optional[str],
]:
    """``(clim, cmap)`` a slot sets explicitly (A7.3); ``(None, None)``
    for a record with no style fields or none set."""
    return (getattr(record, "clim", None), getattr(record, "cmap", None))


__all__ = [
    "Slot",
    "Contour",
    "Vector",
    "Gauss",
    "Line",
    "Sand",
    "Loads",
    "Reactions",
    "SLOT_CATALOG",
    "COLOUR_MAPPED",
    "slot_field",
    "slot_style",
]
