"""Result-slot records — the closed catalog of ADR 0098 §4.

Seven categories, at most one occupant each per mesh view; different
categories stack; filling an occupied slot **replaces** the occupant.
The records here carry only what §4 puts *inside* the slot (quantity /
component / averaging / pattern) — render style (cmap, clim, scale …)
is not slot identity and is a later, additive widening.

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

from dataclasses import dataclass
from typing import Optional


def _require_token(value: object, owner: str, field_name: str) -> None:
    if not isinstance(value, str) or not value:
        raise ValueError(
            f"{owner}.{field_name} must be a non-empty string token; "
            f"got {value!r}."
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

    def __post_init__(self) -> None:
        _require_token(self.quantity, "Contour", "quantity")
        if self.averaging not in ("averaged", "unaveraged"):
            raise ValueError(
                f"Contour.averaging must be 'averaged' or 'unaveraged'; "
                f"got {self.averaging!r}."
            )


@dataclass(frozen=True)
class Vector(Slot):
    """Arrows: one quantity token spanning nodal vector fields AND
    Gauss principal families (§4 — ``vector_glyph`` and
    ``principal_glyph`` collapse to this one slot). Colour-mapped.
    S1's resolver routes the token to the right emit path."""

    quantity: str

    def __post_init__(self) -> None:
        _require_token(self.quantity, "Vector", "quantity")


@dataclass(frozen=True)
class Gauss(Slot):
    """Values at integration points (§4). Colour-mapped; shares the
    contour's scale when the field matches (§5 — same quantity token).
    Gauss values are unaveraged by nature, so there is no averaging
    field here."""

    quantity: str

    def __post_init__(self) -> None:
        _require_token(self.quantity, "Gauss", "quantity")


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

    def __post_init__(self) -> None:
        _require_token(self.quantity, "Sand", "quantity")


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
]
