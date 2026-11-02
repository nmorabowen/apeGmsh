"""``/opensees/combinations``: combinations declared on the bridge (ADR 0130).

A combination is a named descriptor, evaluated when a result is read and
never stored as a stage (ADR 0130 D4, Amendment 1 A2 and A3). This module
holds the descriptor, its validation, and the zone's writer and reader.

Layout (only written when at least one combination is declared)::

    /opensees/combinations
        @current             i4; always 0 in model.h5
        g0/
            name, rule, envelope   str (K,)
            fraction, damping      f8 (K,)    NaN = not defined
            term_offsets           i4 (K+1,)
            terms/
                kind, source       str (T,)
                step, mode         i4 (T,)
                factor             f8 (T,)

``combinations`` is in
:data:`apeGmsh.opensees._internal.lineage.MODEL_HASH_EXCLUDED_CHILDREN`, so
adding one leaves ``model_hash`` unchanged. No combinations means no group, so
a file without any has the bytes it had before the zone existed, and a file
below opensees 2.39.0 reads as empty.

A term resolves within ``model.h5`` only. ``spectrum`` and ``reference`` terms
are refused when declared and when read, because a case lives in ``.user.h5``
and a reference to it would invert the direction D4 fixes (A7). A stage's name
is stamped by the run, so ``stage`` and ``mode`` sources are not resolved here:
the evaluator does that (VU2).
"""
from __future__ import annotations

import math
import unicodedata
import warnings
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Iterable, Sequence

if TYPE_CHECKING:
    import h5py


__all__ = [
    "COMBINATIONS_GROUP",
    "Combination",
    "CombinationError",
    "CombinationRefusal",
    "CombinationTerm",
    "CombinationWarning",
    "CombinationsZone",
    "declare_combination",
    "name_clashes",
    "read_combinations",
    "read_combinations_group",
    "write_combinations_into",
]


#: Group name under ``/opensees``.
COMBINATIONS_GROUP: str = "combinations"

_RULES = frozenset({"linear", "srss", "cqc", "abs_sum", "directional", "envelope"})
_ENVELOPES = frozenset({"max", "min", "absmax"})
#: The term kinds that resolve within ``model.h5``.
_TERM_KINDS = frozenset({"stage", "mode", "combination"})
_I4_MIN, _I4_MAX = -(2 ** 31), 2 ** 31 - 1


class CombinationError(ValueError):
    """A declaration refused, with its reason from ADR 0130 Amendment 1 A3."""

    def __init__(self, reason: str, message: str) -> None:
        super().__init__(f"{message} [{reason}]")
        self.reason = reason


class CombinationWarning(UserWarning):
    """A ``/opensees/combinations`` row or zone was refused at read."""


@dataclass(frozen=True)
class CombinationTerm:
    """One source address and its factor (A3).

    ``step`` is a time index for a stage and a ``STEP_<s>`` id for a mode;
    ``-1`` means the last step or the only set. ``mode`` is 1-based for a mode
    and ``0`` otherwise.
    """

    kind: str
    source: str
    factor: float = 1.0
    step: int = -1
    mode: int = 0


@dataclass(frozen=True)
class Combination:
    """A named combination; sign is derived from the rule and never stored."""

    name: str
    rule: str
    terms: tuple[CombinationTerm, ...]
    envelope: str = ""
    fraction: float | None = None
    damping: float | None = None


@dataclass(frozen=True)
class CombinationRefusal:
    """Row ``index`` of the table was refused at read, with its reason."""

    index: int
    name: str
    reason: str


@dataclass(frozen=True)
class CombinationsZone:
    """What :func:`read_combinations` found.

    ``rows`` holds every row in file order, a refused one included, so that a
    rewrite echoes the table verbatim. ``error`` is set when malformed columns
    refused the whole zone; ``rows`` is then empty.
    """

    rows: tuple[Combination, ...] = ()
    refused: tuple[CombinationRefusal, ...] = ()
    error: str | None = None

    @property
    def valid(self) -> tuple[Combination, ...]:
        """The rows that were not refused, in file order."""
        bad = {r.index for r in self.refused}
        return tuple(c for i, c in enumerate(self.rows) if i not in bad)

    def rows_for_rewrite(self) -> tuple[Combination, ...]:
        """The rows a rewrite stores; raises rather than drop a refused zone."""
        if self.error is not None:
            raise ValueError(
                f"model.h5 rewrite: the source's /opensees/{COMBINATIONS_GROUP} "
                f"was refused ({self.error}), so a rewrite would lose it. "
                "Repair the source file first."
            )
        return self.rows


# ---------------------------------------------------------------------
# Validation, shared by the declaration and the reader
# ---------------------------------------------------------------------

def _name_problem(name: object) -> str | None:
    """Why ``name`` breaks the rules of A2, or ``None`` when it is good."""
    if not isinstance(name, str):
        return f"is a {type(name).__name__}, not a string"
    if not name:
        return "is empty"
    if not unicodedata.is_normalized("NFC", name):
        return "is not NFC"
    # ``Cs``: a lone surrogate has no UTF-8 form, so it could never be written.
    if any(unicodedata.category(c) in ("Cc", "Cs") for c in name):
        return "has a control character"
    if name != name.strip():
        return "has leading or trailing whitespace"
    return None


def _is_i4(value: object) -> bool:
    return (
        isinstance(value, int) and not isinstance(value, bool)
        and _I4_MIN <= value <= _I4_MAX
    )


def _term_problem(term: CombinationTerm) -> str | None:
    """Why ``term`` cannot be a ``model.h5`` term, or ``None``."""
    if term.kind in ("spectrum", "reference"):
        return (
            f"a {term.kind!r} term cannot be archived in model.h5: its case "
            "lives in .user.h5. Declare this combination in .user.h5 after "
            "the run"
        )
    if term.kind not in _TERM_KINDS:
        return f"term kind {term.kind!r} is not one of {sorted(_TERM_KINDS)}"
    problem = _name_problem(term.source)
    if problem is not None:
        return f"term source {term.source!r} {problem}"
    if not _is_i4(term.step) or not _is_i4(term.mode):
        return f"step {term.step!r} and mode {term.mode!r} must be 32-bit integers"
    if isinstance(term.factor, bool) or not isinstance(term.factor, (int, float)) \
            or not math.isfinite(term.factor):
        return f"factor {term.factor!r} must be a finite number"
    if term.kind == "stage" and (term.step < -1 or term.mode != 0):
        return f"a stage term needs step >= -1 and mode 0, got {term!r}"
    if term.kind == "mode" and (term.step < -1 or term.mode < 1):
        return f"a mode term needs step >= -1 and a 1-based mode, got {term!r}"
    if term.kind == "combination" and (term.step != -1 or term.mode != 0):
        return f"a combination term needs step -1 and mode 0, got {term!r}"
    return None


def _param_problem(c: Combination) -> str | None:
    """Why the rule's parameters are wrong, or ``None``."""
    if (c.rule == "envelope") != (c.envelope != ""):
        return "envelope is set if and only if the rule is 'envelope'"
    if c.rule == "envelope" and c.envelope not in _ENVELOPES:
        return f"envelope {c.envelope!r} is not one of {sorted(_ENVELOPES)}"
    if c.rule == "directional":
        if c.fraction is None or not 0.0 <= c.fraction <= 1.0:
            return f"'directional' needs a fraction in [0, 1], got {c.fraction!r}"
    elif c.fraction is not None:
        return "fraction applies only to the 'directional' rule"
    if c.rule != "cqc" and c.damping is not None:
        return "damping applies only to the 'cqc' rule"
    if c.damping is not None and not 0.0 < c.damping < 1.0:
        return f"'cqc' damping must be a ratio in (0, 1), got {c.damping!r}"
    return None


def _row_problem(c: Combination) -> tuple[str, str] | None:
    """``(reason, detail)`` for a row that is bad on its own, or ``None``."""
    problem = _name_problem(c.name)
    if problem is not None:
        return "empty-name", f"name {c.name!r} {problem}"
    if c.rule not in _RULES:
        return "unknown-rule", f"rule {c.rule!r} is not one of {sorted(_RULES)}"
    if not c.terms:
        return "no-terms", "it has no terms"
    problem = _param_problem(c)
    if problem is not None:
        return "bad-param", problem
    for term in c.terms:
        problem = _term_problem(term)
        if problem is not None:
            return "bad-term", problem
    if c.rule == "cqc" and (
        any(t.kind != "mode" for t in c.terms)
        or len({(t.source, t.step) for t in c.terms}) != 1
    ):
        return "bad-term", "in model.h5 'cqc' takes the modes of one set"
    return None


def _as_term(term: object) -> CombinationTerm:
    """Normalise one ``terms`` entry; a shape it does not know refuses."""
    if isinstance(term, CombinationTerm):
        return term
    if isinstance(term, (tuple, list)) and len(term) == 2:
        source, factor = term
        if isinstance(source, Combination):
            return CombinationTerm("combination", source.name, factor)
        if isinstance(source, str):
            return CombinationTerm("stage", source, factor)
    raise CombinationError(
        "bad-term",
        f"apeSees.combination: term {term!r} must be a CombinationTerm, or "
        "(stage name, factor), or (Combination, factor)",
    )


def declare_combination(
    name: str,
    terms: Iterable[object],
    *,
    rule: str,
    envelope: str,
    fraction: float | None,
    damping: float | None,
    known: "Sequence[str] | dict[str, Combination]",
) -> Combination:
    """Build a combination, or raise :class:`CombinationError`.

    ``known`` holds the names already declared on the bridge. A ``combination``
    term must name one of them, so a declared table has no cycle.
    """
    combo = Combination(
        name=name,
        rule=rule,
        terms=tuple(_as_term(t) for t in terms),
        envelope=envelope,
        fraction=None if fraction is None else float(fraction),
        damping=None if damping is None else float(damping),
    )
    found = _row_problem(combo)
    if found is not None:
        raise CombinationError(found[0], f"apeSees.combination({name!r}): {found[1]}")
    if name in known:
        raise CombinationError(
            "duplicate",
            f"apeSees.combination({name!r}): this model already declares a "
            "combination of that name; names are unique across model.h5 and "
            ".user.h5",
        )
    for t in combo.terms:
        if t.kind == "combination" and t.source not in known:
            raise CombinationError(
                "unresolved",
                f"apeSees.combination({name!r}): term names combination "
                f"{t.source!r}, which this model has not declared. A model.h5 "
                "combination resolves only within model.h5",
            )
    return combo


def _refuse_rows(rows: Sequence[Combination]) -> tuple[CombinationRefusal, ...]:
    """The per-row refusals of a table, in row order (A3)."""
    reasons: dict[int, str] = {}
    for i, c in enumerate(rows):
        found = _row_problem(c)
        if found is not None:
            reasons[i] = found[0]
    counts: dict[str, int] = {}
    for c in rows:
        counts[c.name] = counts.get(c.name, 0) + 1
    for i, c in enumerate(rows):
        if i not in reasons and counts[c.name] > 1:
            reasons[i] = "duplicate"

    # Name -> the combinations it references, over the rows still standing.
    def edges() -> dict[str, set[str]]:
        return {
            c.name: {t.source for t in c.terms if t.kind == "combination"}
            for i, c in enumerate(rows) if i not in reasons
        }

    graph = edges()
    for i, c in enumerate(rows):
        if i not in reasons and _reaches_itself(c.name, graph):
            reasons[i] = "cycle"
    # A term on an absent or refused row is unresolved, and that refusal
    # carries to the rows that reference this one.
    changed = True
    while changed:
        changed = False
        graph = edges()
        for i, c in enumerate(rows):
            if i not in reasons and not graph[c.name] <= graph.keys():
                reasons[i] = "unresolved"
                changed = True
    return tuple(
        CombinationRefusal(i, str(rows[i].name), reasons[i]) for i in sorted(reasons)
    )


def _reaches_itself(name: str, graph: dict[str, set[str]]) -> bool:
    seen: set[str] = set()
    stack = list(graph[name])
    while stack:
        current = stack.pop()
        if current == name:
            return True
        if current not in seen:
            seen.add(current)
            stack.extend(graph.get(current, ()))
    return False


def name_clashes(
    model_rows: Iterable[Combination], user_rows: Iterable[Combination],
) -> frozenset[str]:
    """The names both files declare: each such row refuses with ``clash`` (A2).

    Comparison is byte-equal. The join itself belongs to the evaluator, which
    reads both files.
    """
    return frozenset(c.name for c in model_rows) & frozenset(c.name for c in user_rows)


# ---------------------------------------------------------------------
# Writer
# ---------------------------------------------------------------------

def write_combinations_into(
    f: "h5py.File",
    rows: Sequence[Combination],
    *,
    opensees_root: str = "opensees",
) -> None:
    """Write ``rows`` under ``/opensees/combinations`` at ``g0``.

    No-op when ``rows`` is empty. The rows are not validated here: a
    declaration was validated when it was made, and a rewrite echoes the
    source's table verbatim, a refused row included.
    """
    if not rows:
        return

    import h5py
    import numpy as np

    str_dt = h5py.string_dtype(encoding="utf-8")

    def strings(g: "h5py.Group", key: str, values: Sequence[str]) -> None:
        g.create_dataset(key, data=np.array(values, dtype=object), dtype=str_dt)

    def nan_for_none(values: Iterable[float | None]) -> "np.ndarray":
        return np.array(
            [math.nan if v is None else v for v in values], dtype="f8")

    grp = f.require_group(opensees_root)
    if COMBINATIONS_GROUP in grp:
        del grp[COMBINATIONS_GROUP]
    zone = grp.create_group(COMBINATIONS_GROUP)
    zone.attrs["current"] = np.int32(0)
    g = zone.create_group("g0")
    strings(g, "name", [c.name for c in rows])
    strings(g, "rule", [c.rule for c in rows])
    strings(g, "envelope", [c.envelope for c in rows])
    g.create_dataset("fraction", data=nan_for_none(c.fraction for c in rows))
    g.create_dataset("damping", data=nan_for_none(c.damping for c in rows))
    counts = [len(c.terms) for c in rows]
    g.create_dataset(
        "term_offsets", data=np.cumsum([0, *counts], dtype="i4"), dtype="i4")
    terms = [t for c in rows for t in c.terms]
    t_grp = g.create_group("terms")
    strings(t_grp, "kind", [t.kind for t in terms])
    strings(t_grp, "source", [t.source for t in terms])
    t_grp.create_dataset("step", data=np.array([t.step for t in terms], dtype="i4"))
    t_grp.create_dataset("mode", data=np.array([t.mode for t in terms], dtype="i4"))
    t_grp.create_dataset(
        "factor", data=np.array([t.factor for t in terms], dtype="f8"))


# ---------------------------------------------------------------------
# Reader
# ---------------------------------------------------------------------

class _Malformed(Exception):
    """The zone's columns cannot be read as the layout says."""


def _column(g: "h5py.Group", key: str, kinds: str, length: int | None) -> Any:
    """One 1-D column of ``g``; refuses the zone when it is not as specified."""
    import h5py

    if key not in g or not isinstance(g[key], h5py.Dataset):
        raise _Malformed(f"column {g.name}/{key} is missing")
    ds = g[key]
    if ds.ndim != 1 or (length is not None and ds.shape[0] != length):
        raise _Malformed(f"column {ds.name} has shape {ds.shape}")
    if kinds == "str":
        if h5py.check_string_dtype(ds.dtype) is None:
            raise _Malformed(f"column {ds.name} is not a string column")
        try:
            return [str(v) for v in ds.asstr()[()]]
        except UnicodeDecodeError as exc:
            raise _Malformed(f"column {ds.name} is not UTF-8") from exc
    if ds.dtype.kind not in kinds:
        raise _Malformed(f"column {ds.name} has dtype {ds.dtype}")
    return ds[()]


def _read_rows(zone: "h5py.Group") -> tuple[Combination, ...]:
    import h5py

    if "current" not in zone.attrs:
        raise _Malformed("@current is missing")
    current = zone.attrs["current"]
    if getattr(current, "ndim", 1) != 0 or getattr(current, "dtype", None) is None \
            or current.dtype.kind not in "iu":
        raise _Malformed(f"@current is {current!r}, not an integer")
    key = f"g{int(current)}"
    if key not in zone or not isinstance(zone[key], h5py.Group):
        raise _Malformed(f"@current names {key}, which the zone does not have")
    g = zone[key]
    names = _column(g, "name", "str", None)
    k = len(names)
    rules = _column(g, "rule", "str", k)
    envelopes = _column(g, "envelope", "str", k)
    fractions = _column(g, "fraction", "f", k)
    dampings = _column(g, "damping", "f", k)
    offsets = _column(g, "term_offsets", "iu", k + 1)
    if "terms" not in g or not isinstance(g["terms"], h5py.Group):
        raise _Malformed(f"{g.name}/terms is missing")
    t_grp = g["terms"]
    kinds = _column(t_grp, "kind", "str", None)
    n = len(kinds)
    sources = _column(t_grp, "source", "str", n)
    steps = _column(t_grp, "step", "iu", n)
    modes = _column(t_grp, "mode", "iu", n)
    factors = _column(t_grp, "factor", "f", n)
    if int(offsets[0]) != 0 or int(offsets[-1]) != n or any(
        int(a) > int(b) for a, b in zip(offsets, offsets[1:])
    ):
        raise _Malformed(
            f"term_offsets {offsets.tolist()} does not tile {n} term(s)")

    def optional(value: Any) -> float | None:
        return None if math.isnan(value) else float(value)

    terms = [
        CombinationTerm(kinds[i], sources[i], float(factors[i]),
                        int(steps[i]), int(modes[i]))
        for i in range(n)
    ]
    return tuple(
        Combination(
            name=names[i],
            rule=rules[i],
            terms=tuple(terms[int(offsets[i]):int(offsets[i + 1])]),
            envelope=envelopes[i],
            fraction=optional(fractions[i]),
            damping=optional(dampings[i]),
        )
        for i in range(k)
    )


def read_combinations_group(ops_group: "h5py.Group") -> CombinationsZone:
    """Read the zone from an open ``/opensees`` group.

    Never raises on the zone's content: a bad row refuses alone and malformed
    columns refuse the zone, both with a :class:`CombinationWarning`, and the
    file still opens (A3). Absent means no combinations.
    """
    if COMBINATIONS_GROUP not in ops_group:
        return CombinationsZone()
    zone = ops_group[COMBINATIONS_GROUP]
    try:
        import h5py

        if not isinstance(zone, h5py.Group):
            raise _Malformed(f"{zone.name} is not a group")
        rows = _read_rows(zone)
    except _Malformed as exc:
        warnings.warn(
            f"/opensees/{COMBINATIONS_GROUP} refused: {exc}.",
            CombinationWarning, stacklevel=2,
        )
        return CombinationsZone(error=str(exc))
    refused = _refuse_rows(rows)
    if refused:
        listed = ", ".join(f"{r.name!r} ({r.reason})" for r in refused)
        warnings.warn(
            f"/opensees/{COMBINATIONS_GROUP}: row(s) refused: {listed}.",
            CombinationWarning, stacklevel=2,
        )
    return CombinationsZone(rows=rows, refused=refused)


def read_combinations(
    path: str, *, opensees_root: str = "/opensees",
) -> CombinationsZone:
    """Read ``/opensees/combinations`` from the file at ``path``."""
    import h5py

    root = opensees_root.strip("/")
    with h5py.File(path, "r") as f:
        if root not in f:
            return CombinationsZone()
        return read_combinations_group(f[root])
