"""The combination descriptor: columns, rows, per-row refusal (ADR 0130 A3).

This module reads and writes the table of one generation and decides which
rows stand, through the row rules ``model.h5``'s reader shares
(``opensees/_internal/_combination_rules.py``). It evaluates nothing: that is
:mod:`._evaluate`. Whether a ``stage``, ``mode``, ``spectrum`` or ``reference``
source exists needs the set's other files, so the evaluator refuses a missing
one; a ``combination`` source is resolved here, because the rows it names are
all in hand.

Sign is derived from the rule and the terms and never stored.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Optional, Sequence

import h5py
import numpy as np

from ...opensees._internal import _combination_rules as rules
from ._errors import ZoneRefused

__all__ = [
    "ENVELOPES",
    "KINDS",
    "REASONS",
    "RULES",
    "Combination",
    "CombinationTable",
    "Refusal",
    "StateAddress",
    "Term",
    "address_problem",
    "decode_cell",
    "read_generation",
    "resolve",
    "write_generation",
]

RULES = rules.RULES
ENVELOPES = rules.ENVELOPES
KINDS = rules.KINDS
REASONS = rules.REASONS

_STR = h5py.string_dtype(encoding="utf-8")


@dataclass(frozen=True)
class StateAddress:
    """The one source address of ``terms`` and ``STATES`` (A3)."""

    kind: str
    source: str
    step: int = -1
    mode: int = 0


@dataclass(frozen=True)
class Term:
    """One ``(source, factor)`` of a combination."""

    kind: str
    source: str
    step: int = -1
    mode: int = 0
    factor: float = 1.0

    @property
    def address(self) -> StateAddress:
        return StateAddress(self.kind, self.source, self.step, self.mode)


@dataclass(frozen=True)
class Combination:
    """One row. ``None`` here is NaN in the file (``""`` for ``envelope``)."""

    name: str
    rule: str
    terms: tuple[Term, ...]
    envelope: str = ""
    fraction: Optional[float] = None
    damping: Optional[float] = None


@dataclass(frozen=True)
class Refusal:
    """A row that does not stand, and why. ``origin`` is ``user`` or ``model``."""

    name: str
    reason: str
    detail: str
    origin: str = "user"
    row: int = -1


@dataclass(frozen=True)
class CombinationTable:
    """The rows in force, plus each refused row with its reason."""

    rows: tuple[Combination, ...]
    refused: tuple[Refusal, ...]
    model_rows: tuple[Combination, ...] = ()

    def by_name(self, name: str) -> Combination:
        for row in (*self.rows, *self.model_rows):
            if row.name == name:
                return row
        for ref in self.refused:
            if ref.name == name:
                raise KeyError(f"combination {name!r} is refused: {ref.reason} ({ref.detail})")
        raise KeyError(f"no combination named {name!r}")


# ---------------------------------------------------------------------------
# Validation: the rules live in ``opensees/_internal/_combination_rules.py``
# ---------------------------------------------------------------------------


def address_problem(a: StateAddress) -> Optional[str]:
    """Why an address breaks the source-address table, or ``None``."""
    return rules.address_problem(a.kind, a.source, a.step, a.mode)


def resolve(
    user_rows: Sequence[Combination],
    model_rows: Sequence[Combination] = (),
) -> CombinationTable:
    """Decide which rows stand. A bad row refuses alone (A3).

    The verdicts are :func:`~apeGmsh.opensees._internal._combination_rules.judge`'s,
    the one implementation ``model.h5``'s reader shares. A ``.user.h5`` row may
    name a ``model.h5`` row, never the reverse.
    """
    rows: list[tuple[str, int, Combination]] = [("user", i, r) for i, r in enumerate(user_rows)]
    rows += [("model", i, r) for i, r in enumerate(model_rows)]
    verdict = rules.judge([(origin, row) for origin, _, row in rows])
    refused = tuple(
        Refusal(rows[at][2].name, *verdict[at], origin=rows[at][0], row=rows[at][1])
        for at in sorted(verdict)
    )
    standing = [at for at in range(len(rows)) if at not in verdict]
    return CombinationTable(
        rows=tuple(rows[at][2] for at in standing if rows[at][0] == "user"),
        refused=refused,
        model_rows=tuple(rows[at][2] for at in standing if rows[at][0] == "model"),
    )


# ---------------------------------------------------------------------------
# Columns
# ---------------------------------------------------------------------------


def decode_cell(raw: object, zone: str, where: str) -> str:
    """A table cell as ``str``. A cell that is not UTF-8 refuses the zone."""
    if isinstance(raw, bytes):
        try:
            return raw.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ZoneRefused(zone, f"{where} holds a cell that is not UTF-8: {raw!r}") from exc
    if isinstance(raw, str):
        try:
            raw.encode("utf-8")
        except UnicodeEncodeError as exc:
            raise ZoneRefused(zone, f"{where} holds a cell that is not UTF-8: {raw!r}") from exc
        return raw
    raise ZoneRefused(zone, f"{where} holds a cell that is not a string: {raw!r}")


def _column(group: h5py.Group, name: str, code: str, zone: str, length: Optional[int]) -> np.ndarray:
    """A column, refusing the zone when it is absent or of the wrong type."""
    if name not in group:
        raise ZoneRefused(zone, f"{group.name} has no {name} column")
    ds = group[name]
    if not isinstance(ds, h5py.Dataset) or ds.ndim != 1:
        raise ZoneRefused(zone, f"{ds.name} is not a one-dimensional dataset")
    if code == "str":
        if h5py.check_string_dtype(ds.dtype) is None:
            raise ZoneRefused(zone, f"{ds.name} is not a string column ({ds.dtype})")
    elif (ds.dtype.kind, ds.dtype.itemsize) != (code[0], int(code[1:])):
        raise ZoneRefused(zone, f"{ds.name} is {ds.dtype}, expected {code}")
    if length is not None and ds.shape[0] != length:
        raise ZoneRefused(zone, f"{ds.name} has {ds.shape[0]} rows, expected {length}")
    return ds[...]


def read_generation(group: h5py.Group, zone: str = "/combinations") -> list[Combination]:
    """The rows of one ``g<N>``. Malformed columns refuse the zone (A3)."""
    names = _column(group, "name", "str", zone, None)
    k = names.shape[0]
    rules = _column(group, "rule", "str", zone, k)
    envelopes = _column(group, "envelope", "str", zone, k)
    fractions = _column(group, "fraction", "f8", zone, k)
    dampings = _column(group, "damping", "f8", zone, k)
    offsets = _column(group, "term_offsets", "i4", zone, k + 1)
    if "terms" not in group or not isinstance(group["terms"], h5py.Group):
        raise ZoneRefused(zone, f"{group.name} has no terms group")
    terms = group["terms"]
    kinds = _column(terms, "kind", "str", zone, None)
    t = kinds.shape[0]
    sources = _column(terms, "source", "str", zone, t)
    steps = _column(terms, "step", "i4", zone, t)
    modes = _column(terms, "mode", "i4", zone, t)
    factors = _column(terms, "factor", "f8", zone, t)
    if int(offsets[0]) != 0 or int(offsets[-1]) != t or np.any(np.diff(offsets) < 0):
        raise ZoneRefused(zone, f"{group.name}/term_offsets does not partition {t} terms")

    def cell(raw: object) -> str:
        return decode_cell(raw, zone, group.name)

    out = []
    for i in range(k):
        row_terms = tuple(
            Term(cell(kinds[j]), cell(sources[j]), int(steps[j]),
                 int(modes[j]), float(factors[j]))
            for j in range(int(offsets[i]), int(offsets[i + 1]))
        )
        out.append(Combination(
            name=cell(names[i]),
            rule=cell(rules[i]),
            terms=row_terms,
            envelope=cell(envelopes[i]),
            fraction=None if math.isnan(fractions[i]) else float(fractions[i]),
            damping=None if math.isnan(dampings[i]) else float(dampings[i]),
        ))
    return out


def _text_column(group: h5py.Group, name: str, values: Iterable[str]) -> None:
    data = list(values)
    if data:
        group.create_dataset(name, data=data, dtype=_STR)
    else:
        group.create_dataset(name, shape=(0,), dtype=_STR)


def write_generation(group: h5py.Group, rows: Sequence[Combination]) -> None:
    """Write ``rows`` as the columns of one ``g<N>``. Callers validated them."""
    terms = [t for row in rows for t in row.terms]
    _text_column(group, "name", [r.name for r in rows])
    _text_column(group, "rule", [r.rule for r in rows])
    _text_column(group, "envelope", [r.envelope for r in rows])
    nan = float("nan")
    group.create_dataset("fraction", data=np.array(
        [nan if r.fraction is None else r.fraction for r in rows], dtype="f8"))
    group.create_dataset("damping", data=np.array(
        [nan if r.damping is None else r.damping for r in rows], dtype="f8"))
    group.create_dataset("term_offsets", data=np.cumsum(
        [0, *(len(r.terms) for r in rows)], dtype="i4"))
    out = group.create_group("terms")
    _text_column(out, "kind", [t.kind for t in terms])
    _text_column(out, "source", [t.source for t in terms])
    out.create_dataset("step", data=np.array([t.step for t in terms], dtype="i4"))
    out.create_dataset("mode", data=np.array([t.mode for t in terms], dtype="i4"))
    out.create_dataset("factor", data=np.array([t.factor for t in terms], dtype="f8"))
