"""Evaluate a combination per quantity, on read (ADR 0130 D4, Amendment 1 A3, A5).

The rules table of ``architecture/h5-schema.md`` (``<stem>.user.h5``, "Rules"),
with ``E = factor · r`` for each term and ``p`` the row's fraction:

==============  =======================  ====================================
``rule``        terms                    value
==============  =======================  ====================================
``linear``      list                     ``ΣE``
``srss``        list, or one whole case  ``√ΣE²``
``abs_sum``     list, or one whole case  ``Σ|E|``
``cqc``         one whole case, or the   ``√max(0, EᵀρE)``
                modes of one set
``directional`` list                     ``max_a(|E_a| + p Σ_{b≠a} |E_b|)``
``envelope``    list                     ``max``, ``min`` or ``absmax``
==============  =======================  ====================================

A whole case first sums, algebraically, the modes that share a ``sa_period``
(``0`` means the mode's own period), into the slot of the group's first mode at
``ω = 2π/T`` and with that mode's damping. ``ρ`` is Der Kiureghian's, with
``ω = √λ``; ``λ ≤ 0`` is refused. A finite row ``damping`` overrides the case's
column. Terms join on the component label and the row key (the ID, and the
Gauss, station or node slot), and a row missing from a term is NaN. NaN
propagates through every rule: ``max`` is ``numpy.max``, never ``nanmax``.

Only ``linear`` can be signed, and only if all its terms are; sign is never
stored. A ``linear`` with a ``stage`` term carries the badge "superposition
assumes linearity" (D4).

A user result (A5): a stored row for the combination wins. Otherwise
``envelope`` reads the terms' stored rows, and every other rule needs
``@linear=1``; so does a term whose factor is not 1, because a factor scales
the quantity. A rule that cannot be applied refuses; it never sums. One state
read alone (:func:`read_state`) has no rule, so it is read as stored.

Combinations are evaluated in dependency order without recursion, so a deep
chain or a wide diamond of references costs one evaluation per combination.

The evaluator reads values only through a :class:`Source`, so the same code
serves the recorded results of a :class:`~apeGmsh.results.Results`, the user
results of a ``.user.h5``, and the conformance fixture
(``tests/fixtures/user_results/``) that the app's TypeScript evaluator shares.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Protocol, Sequence

import numpy as np

from ._descriptor import Combination, CombinationTable, StateAddress, Term, address_problem
from ._errors import UserFileError

__all__ = [
    "EVAL_REASONS",
    "SUPERPOSITION_BADGE",
    "Case",
    "Combined",
    "CombinationRefused",
    "Field",
    "Source",
    "cqc_rho",
    "evaluate",
    "is_signed",
    "read_state",
]

#: Why an evaluation refuses. A row the table refused reads ``refused``.
EVAL_REASONS = (
    "refused", "unresolved", "bad-param", "bad-term", "not-linear",
    "state-free", "stale", "foreign-handle",
)
SUPERPOSITION_BADGE = "superposition assumes linearity"


class CombinationRefused(UserFileError):
    """One combination cannot be evaluated for one quantity, and why."""

    def __init__(self, name: str, reason: str, detail: str) -> None:
        if reason not in EVAL_REASONS:
            raise ValueError(f"unknown evaluation reason {reason!r}")
        super().__init__(f"combination {name!r} refused ({reason}): {detail}")
        self.name = name
        self.reason = reason
        self.detail = detail


@dataclass(frozen=True, eq=False)
class Field:
    """One quantity on one state.

    ``keys`` is a structured array, one row per value row, whose fields name
    what a row is (``id``, and ``sub`` for a Gauss point, station or element
    node; ``kind`` and ``row`` for an entity). ``values`` is ``(n, c)``,
    one column per label of ``components``.
    """

    keys: np.ndarray
    components: tuple[str, ...]
    values: np.ndarray

    def __post_init__(self) -> None:
        keys = np.asarray(self.keys)
        values = np.asarray(self.values, dtype="f8")
        if keys.dtype.names is None or keys.ndim != 1:
            raise ValueError("Field.keys must be a one-dimensional structured array")
        if values.shape != (keys.shape[0], len(self.components)):
            raise ValueError(
                f"Field.values has shape {values.shape}, expected "
                f"({keys.shape[0]}, {len(self.components)})")
        if len(set(self.components)) != len(self.components):
            raise ValueError(f"Field.components repeats a label: {self.components}")
        if np.unique(keys).shape[0] != keys.shape[0]:
            raise ValueError("Field.keys repeats a row")
        object.__setattr__(self, "keys", keys)
        object.__setattr__(self, "values", values)

    def scaled(self, factor: float) -> "Field":
        return Field(self.keys, self.components, factor * self.values)


@dataclass(frozen=True, eq=False)
class Case:
    """One whole spectrum case for one quantity: per mode, the signed increment.

    Row ``i`` of every column, and ``increments[i]``, is one mode of the case
    (``/spectra/s<k>/modes``). ``sa_period`` is ``0`` for a mode read at its own
    period, and NaN in ``damping`` means the case gives none for that mode.
    """

    mode: np.ndarray
    lam: np.ndarray
    sa_period: np.ndarray
    damping: np.ndarray
    increments: tuple[Field, ...]

    def __post_init__(self) -> None:
        m = len(self.increments)
        for what in ("mode", "lam", "sa_period", "damping"):
            column = np.asarray(getattr(self, what)).ravel()
            if column.shape != (m,):
                raise ValueError(f"Case.{what} has {column.shape[0]} rows for {m} increments")
            object.__setattr__(self, what, column)
        if m == 0:
            raise ValueError("a case has at least one mode")


@dataclass(frozen=True, eq=False)
class Combined:
    """A combination's value for one quantity."""

    name: str
    field: Field
    signed: bool
    badges: tuple[str, ...] = ()


class Source(Protocol):
    """Where an evaluation reads one quantity's values.

    ``linear`` says whether a summing rule may combine this quantity (A5):
    true for a recorded response (in a native ``results.h5``, only a recorder
    spelling: :func:`~._states.recorded_is_linear`), the ``@linear`` flag for a
    user result.
    ``stored`` is a user result's stored row for a combination, else ``None``.
    ``field`` raises :class:`CombinationRefused` for a state it cannot read.
    """

    @property
    def linear(self) -> bool: ...

    def stored(self, name: str) -> Optional[Field]: ...

    def field(self, address: StateAddress) -> Field: ...

    def eigenvalue(self, address: StateAddress) -> float: ...

    def case(self, name: str) -> Case: ...


# ---------------------------------------------------------------------------
# Sign, from the descriptors alone
# ---------------------------------------------------------------------------


def _row(table: CombinationTable, name: str, caller: str) -> Combination:
    try:
        return table.by_name(name)
    except KeyError as exc:
        reason = "refused" if table.refusal(name) is not None else "unresolved"
        raise CombinationRefused(caller, reason, str(exc.args[0])) from exc


def _post_order(table: CombinationTable, name: str, leaf=lambda n: False) -> list[str]:
    """``name`` and every combination it reaches, each after the ones it names.

    Iterative, so depth is not bounded by the interpreter's stack. ``leaf``
    stops the walk at a combination whose terms are not needed (a stored row).
    The table is acyclic: :func:`resolve` refused every row in or reaching a cycle.
    """
    order: list[str] = []
    done: set[str] = set()
    stack: list[tuple[str, bool]] = [(name, False)]
    while stack:
        current, expanded = stack.pop()
        if current in done:
            continue
        if expanded:
            done.add(current)
            order.append(current)
            continue
        row = _row(table, current, name)
        stack.append((current, True))
        if not leaf(current):
            stack.extend((t.source, False) for t in row.terms
                         if t.kind == "combination" and t.source not in done)
    return order


def _signs(table: CombinationTable, order: Sequence[str], memo: dict[str, bool]) -> None:
    for current in order:
        if current not in memo:
            row = _row(table, current, current)
            memo[current] = row.rule == "linear" and all(
                memo[t.source] if t.kind == "combination" else True for t in row.terms)


def is_signed(table: CombinationTable, name: str) -> bool:
    """Whether combination ``name`` is signed: a ``linear`` whose terms all are.

    Only a signed result draws as a deformed shape (A3).
    """
    memo: dict[str, bool] = {}
    _signs(table, _post_order(table, name), memo)
    return memo[name]


# ---------------------------------------------------------------------------
# Joining terms
# ---------------------------------------------------------------------------


def _align(fields: Sequence[Field], name: str) -> tuple[np.ndarray, tuple[str, ...], np.ndarray]:
    """``(keys, components, stack (k, n, c))`` over the union of rows and labels.

    A row or a label a term lacks is NaN in that term.
    """
    first = fields[0].keys.dtype
    for f in fields[1:]:
        if f.keys.dtype.names != first.names:
            raise CombinationRefused(
                name, "bad-term",
                f"its terms are on different targets ({first} and {f.keys.dtype})")
    # One key type for all: a text key (an entity kind) is as wide as its widest.
    assert first.names is not None
    dtype = np.dtype([
        (n, np.result_type(*(f.keys.dtype[n] for f in fields))) for n in first.names])
    fields = [f if f.keys.dtype == dtype else Field(f.keys.astype(dtype), f.components, f.values)
              for f in fields]
    components: list[str] = []
    for f in fields:
        components.extend(c for c in f.components if c not in components)
    same_rows = all(
        f.keys.shape == fields[0].keys.shape and bool(np.all(f.keys == fields[0].keys))
        for f in fields[1:]
    )
    keys = fields[0].keys if same_rows else np.unique(np.concatenate([f.keys for f in fields]))
    stack = np.full((len(fields), keys.shape[0], len(components)), np.nan)
    for i, f in enumerate(fields):
        rows = np.arange(keys.shape[0]) if same_rows else np.searchsorted(keys, f.keys)
        cols = [components.index(c) for c in f.components]
        stack[i][np.ix_(rows, cols)] = f.values
    return keys, tuple(components), stack


# ---------------------------------------------------------------------------
# The kernels
# ---------------------------------------------------------------------------


def cqc_rho(omega: np.ndarray, damping: np.ndarray) -> np.ndarray:
    """Der Kiureghian's modal correlation, ``r = ω_j/ω_i``, ``ρ_ii = 1``.

    ``ρ_ij = 8 √(ξ_i ξ_j) (ξ_i + r ξ_j) r^1.5 /
    [(1 − r²)² + 4 ξ_i ξ_j r (1 + r²) + 4 (ξ_i² + ξ_j²) r²]``.
    """
    w = np.asarray(omega, dtype="f8")
    xi = np.asarray(damping, dtype="f8")
    r = w[None, :] / w[:, None]
    xi_i, xi_j = xi[:, None], xi[None, :]
    num = 8.0 * np.sqrt(xi_i * xi_j) * (xi_i + r * xi_j) * r ** 1.5
    den = ((1.0 - r * r) ** 2 + 4.0 * xi_i * xi_j * r * (1.0 + r * r)
           + 4.0 * (xi_i ** 2 + xi_j ** 2) * r * r)
    with np.errstate(divide="ignore", invalid="ignore"):
        rho = np.where(den > 0.0, num / den, 1.0)
    np.fill_diagonal(rho, 1.0)
    return rho


def _cqc(stack: np.ndarray, omega: np.ndarray, damping: np.ndarray) -> np.ndarray:
    rho = cqc_rho(omega, damping)
    total = np.einsum("i...,ij,j...->...", stack, rho, stack)
    return np.sqrt(np.maximum(total, 0.0))  # np.maximum keeps a NaN


def _modal(rule: str, stack: np.ndarray, omega: Optional[np.ndarray],
           damping: Optional[np.ndarray]) -> np.ndarray:
    if rule == "srss":
        return np.sqrt(np.sum(stack * stack, axis=0))
    if rule == "abs_sum":
        return np.sum(np.abs(stack), axis=0)
    assert omega is not None and damping is not None
    return _cqc(stack, omega, damping)


def _check_lambda(lam: np.ndarray, name: str, what: str) -> np.ndarray:
    lam = np.asarray(lam, dtype="f8")
    if not np.all(np.isfinite(lam)) or np.any(lam <= 0.0):
        raise CombinationRefused(
            name, "bad-param", f"{what} has an eigenvalue <= 0 or not finite: {lam.tolist()}")
    return np.sqrt(lam)


def _check_damping(xi: np.ndarray, name: str, what: str) -> np.ndarray:
    xi = np.asarray(xi, dtype="f8")
    if not np.all(np.isfinite(xi)) or np.any(xi < 0.0) or np.any(xi >= 1.0):
        raise CombinationRefused(
            name, "bad-param", f"{what} has a damping ratio outside [0, 1): {xi.tolist()}")
    return xi


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


class _Run:
    """One evaluation: the table, the source, and what was read already."""

    def __init__(self, table: CombinationTable, source: Source) -> None:
        self.table = table
        self.source = source
        self.memo: dict[str, Combined] = {}
        self.signs: dict[str, bool] = {}
        self.fields: dict[StateAddress, Field] = {}

    def field(self, address: StateAddress) -> Field:
        if address not in self.fields:
            self.fields[address] = self.source.field(address)
        return self.fields[address]

    def combination(self, name: str) -> Combined:
        """Evaluate ``name``: the combinations it needs first, in dependency order."""
        stored: dict[str, Optional[Field]] = {}

        def leaf(current: str) -> bool:
            if current not in stored:
                stored[current] = self.source.stored(current)
            return stored[current] is not None

        order = _post_order(self.table, name, leaf)
        # A stored row is a leaf, but its sign still follows its descriptor.
        _signs(self.table, _post_order(self.table, name), self.signs)
        if not self.source.linear:
            # A5 over every combination the value needs, before any value is read,
            # so the refusal does not depend on the order the terms are read in.
            for current in order:
                if stored.get(current) is None:
                    self._check_a5(_row(self.table, current, current))
        for current in order:
            if current not in self.memo:
                self.memo[current] = self._one(current, stored.get(current))
        return self.memo[name]

    def _check_a5(self, row: Combination) -> None:
        """A rule or a factor that would sum a quantity that is not linear refuses (A5)."""
        if self.source.linear:
            return
        if row.rule != "envelope":
            raise CombinationRefused(
                row.name, "not-linear",
                f"{row.rule} sums its terms, and this quantity is not linear (a user "
                "result without @linear=1, or a native results.h5 component that is not "
                "a recorder spelling): a sum of it is not its value under the sum. Store "
                "a row for this combination, or use envelope (ADR 0130 A5)")
        for term in row.terms:
            if term.factor != 1.0:
                # 2 r(D) is not r(2 D) for a quantity that is not linear.
                raise CombinationRefused(
                    row.name, "not-linear",
                    f"a term of factor {term.factor} scales this quantity, which is not "
                    "linear; an envelope of it takes factor-1 terms (ADR 0130 A5)")

    def _one(self, name: str, stored: Optional[Field]) -> Combined:
        row = _row(self.table, name, name)
        badges = (SUPERPOSITION_BADGE,) if (
            row.rule == "linear" and any(t.kind == "stage" for t in row.terms)) else ()
        if stored is not None:
            return Combined(name, stored, self.signs[name], badges)
        self._check_a5(row)
        return Combined(name, self._apply(row), self.signs[name], badges)

    def term(self, term: Term, name: str) -> Field:
        if term.kind == "combination":
            return self.memo[term.source].field.scaled(term.factor)
        return self.field(term.address).scaled(term.factor)

    def _apply(self, row: Combination) -> Field:
        whole = [t for t in row.terms if t.kind == "spectrum" and t.mode == 0]
        if whole:
            return self._whole_case(row, whole[0])
        fields = [self.term(t, row.name) for t in row.terms]
        keys, comps, stack = _align(fields, row.name)
        rule = row.rule
        if rule == "linear":
            value = np.sum(stack, axis=0)
        elif rule in ("srss", "abs_sum"):
            value = _modal(rule, stack, None, None)
        elif rule == "cqc":
            omega, damping = self._term_modes(row)
            value = _cqc(stack, omega, damping)
        elif rule == "directional":
            assert row.fraction is not None
            e = np.abs(stack)
            total = np.sum(e, axis=0)
            value = np.max(e + row.fraction * (total - e), axis=0)
        elif rule == "envelope":
            if row.envelope == "max":
                value = np.max(stack, axis=0)
            elif row.envelope == "min":
                value = np.min(stack, axis=0)
            else:
                value = np.max(np.abs(stack), axis=0)
        else:  # the table refused any other rule
            raise CombinationRefused(row.name, "refused", f"unknown rule {rule!r}")
        return Field(keys, comps, value)

    def _term_modes(self, row: Combination) -> tuple[np.ndarray, np.ndarray]:
        """``ω`` and ``ξ`` of ``cqc`` over the modes of one set, or one case's increments."""
        first = row.terms[0]
        if first.kind == "mode":
            lam = np.array([self.source.eigenvalue(t.address) for t in row.terms])
            if row.damping is None:  # the table refuses this row; never guess
                raise CombinationRefused(row.name, "bad-param", "cqc over modes needs a damping")
            xi = np.full(lam.shape, row.damping)
        else:  # signed increments of one case
            case = self.source.case(first.source)
            at = [_case_row(case, t.mode, row.name) for t in row.terms]
            lam = case.lam[at]
            xi = np.full(lam.shape, row.damping) if row.damping is not None else case.damping[at]
        omega = _check_lambda(lam, row.name, "a cqc term")
        return omega, _check_damping(xi, row.name, "a cqc term")

    def _whole_case(self, row: Combination, term: Term) -> Field:
        case = self.source.case(term.source)
        keys, comps, stack = _align(list(case.increments), row.name)
        try:
            slot, keep = _sa_groups(case.sa_period)
        except ValueError as exc:
            raise CombinationRefused(row.name, "bad-param", f"case {term.source!r}: {exc}") from exc
        grouped = np.zeros((len(keep),) + stack.shape[1:])
        np.add.at(grouped, slot, stack)
        omega = damping = None
        if row.rule == "cqc":
            lam = case.lam[keep]
            omega = _check_lambda(lam, row.name, f"case {term.source!r}")
            t = case.sa_period[keep]
            over = t > 0.0
            omega = np.where(over, 2.0 * np.pi / np.where(over, t, 1.0), omega)
            xi = (np.full(len(keep), row.damping) if row.damping is not None
                  else case.damping[keep])
            damping = _check_damping(xi, row.name, f"case {term.source!r}")
        value = abs(term.factor) * _modal(row.rule, grouped, omega, damping)
        return Field(keys, comps, value)


def _case_row(case: Case, mode: int, name: str) -> int:
    hits = np.flatnonzero(case.mode == mode)
    if hits.size != 1:
        raise CombinationRefused(
            name, "unresolved", f"the case has {hits.size} rows for mode {mode}")
    return int(hits[0])


def _sa_groups(sa_period: np.ndarray) -> tuple[np.ndarray, list[int]]:
    """Each mode's slot, and the first mode of each slot (A3, "a whole case").

    Modes sharing one ``sa_period > 0`` are one term, in the slot of the first
    of them; a mode at its own period (``0``) is a term alone.
    """
    t = np.asarray(sa_period, dtype="f8")
    slot = np.empty(t.shape[0], dtype=np.intp)
    keep: list[int] = []
    group: dict[float, int] = {}
    for a, period in enumerate(t):
        if not math.isfinite(period) or period < 0.0:
            raise ValueError(f"sa_period {period} is not 0 or a period > 0")
        if period > 0.0 and float(period) in group:
            slot[a] = group[float(period)]
            continue
        slot[a] = len(keep)
        if period > 0.0:
            group[float(period)] = len(keep)
        keep.append(a)
    return slot, keep


def evaluate(table: CombinationTable, name: str, source: Source) -> Combined:
    """Evaluate combination ``name`` of ``table`` for the quantity ``source`` reads.

    Raises :class:`CombinationRefused` with one of :data:`EVAL_REASONS`.
    """
    return _Run(table, source).combination(name)


def read_state(address: StateAddress, source: Source) -> Combined:
    """One state alone, read as it is: no rule, so no ``@linear`` check and no badge.

    A recorded or user result at one stage, mode or increment is signed. A whole
    case (``spectrum``, ``mode=0``) is many states, so it is refused here.
    """
    name = f"<{address.kind} {address.source}>"
    problem = address_problem(address)
    if problem is None and address.kind == "spectrum" and address.mode == 0:
        problem = "a whole case is not one state; combine it with a rule"
    if problem is None and address.kind == "combination":
        problem = "a combination is evaluated by name, not read as a state"
    if problem is not None:
        raise CombinationRefused(name, "bad-term", problem)
    return Combined(name, source.field(address), True, ())
