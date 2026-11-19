"""The row rules of a combination table: the one implementation (ADR 0130 A2, A3).

Both readers of a combination table judge their rows here: ``/opensees/combinations``
in ``model.h5`` (:mod:`._combinations_h5`) and ``/combinations`` in ``<stem>.user.h5``
(``apeGmsh.results.user._descriptor``). They differ only in the dataclasses that
hold a row, so a row is anything with the descriptor's attributes
(:class:`RowLike`), and the file it came from is its ``origin``.

A bad row refuses alone, with one of :data:`REASONS`, and the file still opens.
:func:`row_problem` judges a row by itself; :func:`judge` adds what needs the
whole table: ``duplicate`` and ``clash``, ``cycle``, then ``unresolved``. A
``.user.h5`` row may name a ``model.h5`` row, never the reverse, so a ``model``
row sees only ``model`` rows.

The cycle pass is a topological peel (Kahn), and ``unresolved`` spreads over the
reverse references once, so :func:`judge` is linear in rows plus references. A
chain of a thousand rows read once per ``from_h5``, so a cubic search cost
seconds there.

This module lives in ``opensees`` because ``results`` already depends on
``opensees``, and the dependency must not run the other way.
"""
from __future__ import annotations

import math
import numbers
import unicodedata
from collections import deque
from typing import Any, Optional, Protocol, Sequence

__all__ = [
    "ENVELOPES",
    "KINDS",
    "MODEL_KINDS",
    "REASONS",
    "RULES",
    "RowLike",
    "TermLike",
    "address_problem",
    "judge",
    "name_problem",
    "row_problem",
]

RULES = ("linear", "srss", "cqc", "abs_sum", "directional", "envelope")
ENVELOPES = ("max", "min", "absmax")
KINDS = ("stage", "mode", "spectrum", "reference", "combination")
#: The kinds a ``model.h5`` term may name: a case lives in ``.user.h5`` (A7).
MODEL_KINDS = ("stage", "mode", "combination")
REASONS = (
    "empty-name", "duplicate", "clash", "unknown-rule", "no-terms",
    "bad-param", "bad-term", "unresolved", "cycle",
)
_ORIGINS = ("user", "model")
_I4_MIN, _I4_MAX = -(2 ** 31), 2 ** 31 - 1


class TermLike(Protocol):
    """One ``(source, factor)``: a source address and its factor."""

    @property
    def kind(self) -> str: ...
    @property
    def source(self) -> str: ...
    @property
    def step(self) -> int: ...
    @property
    def mode(self) -> int: ...
    @property
    def factor(self) -> float: ...


class RowLike(Protocol):
    """One combination row. ``None`` is NaN in the file (``""`` for ``envelope``)."""

    @property
    def name(self) -> str: ...
    @property
    def rule(self) -> str: ...
    @property
    def terms(self) -> Sequence[TermLike]: ...
    @property
    def envelope(self) -> str: ...
    @property
    def fraction(self) -> Optional[float]: ...
    @property
    def damping(self) -> Optional[float]: ...


# ---------------------------------------------------------------------------
# One row
# ---------------------------------------------------------------------------


def name_problem(name: Any) -> Optional[str]:
    """Why ``name`` breaks the name rule of A2, or ``None`` when it holds.

    A name is non-empty NFC UTF-8 with no control character and no leading or
    trailing whitespace. A lone surrogate has no UTF-8 form, so it is refused.
    """
    if not isinstance(name, str):
        return f"not a string: {name!r}"
    if not name:
        return "empty"
    try:
        name.encode("utf-8")
    except UnicodeEncodeError:
        return "not valid UTF-8"
    if not unicodedata.is_normalized("NFC", name):
        return "not NFC"
    if any(unicodedata.category(c) == "Cc" for c in name):
        return "has a control character"
    if name != name.strip():
        return "has leading or trailing whitespace"
    return None


def _is_i4(value: object) -> bool:
    return (
        isinstance(value, numbers.Integral) and not isinstance(value, bool)
        and _I4_MIN <= int(value) <= _I4_MAX
    )


def address_problem(kind: object, source: object, step: object, mode: object) -> Optional[str]:
    """Why ``(kind, source, step, mode)`` breaks the source-address table, or ``None``."""
    if kind not in KINDS:
        return f"unknown kind {kind!r}"
    problem = name_problem(source)
    if problem is not None:
        return f"source {source!r}: {problem}"
    if not (_is_i4(step) and _is_i4(mode)):
        return f"step {step!r} and mode {mode!r} must be 32-bit integers"
    s, m = int(step), int(mode)  # type: ignore[call-overload]
    if kind == "stage":
        ok = s >= -1 and m == 0
    elif kind == "mode":
        ok = s >= -1 and m >= 1
    elif kind == "spectrum":
        ok = s == -1 and m >= 0
    else:  # reference, combination
        ok = s == -1 and m == 0
    if not ok:
        return f"a {kind} term with step {s}, mode {m}"
    return None


def _term_problem(term: TermLike) -> Optional[str]:
    problem = address_problem(term.kind, term.source, term.step, term.mode)
    if problem is not None:
        return problem
    f = term.factor
    if isinstance(f, bool) or not isinstance(f, numbers.Real) or not math.isfinite(f):
        return f"factor {f!r} must be a finite number"
    return None


def _shape_problem(row: RowLike) -> Optional[str]:
    """The ``Terms`` column of the rules table, or ``None`` when it holds."""
    terms = row.terms
    whole = [t for t in terms if t.kind == "spectrum" and t.mode == 0]
    if row.rule in ("srss", "abs_sum"):
        if whole and len(terms) != 1:
            return f"{row.rule} takes a list or one whole case, not both"
    elif row.rule == "cqc":
        first = terms[0]
        if whole:
            if len(terms) != 1:
                return "cqc takes one whole case, not a list with one"
        elif first.kind not in ("mode", "spectrum") or any(
            (t.kind, t.source, t.step) != (first.kind, first.source, first.step)
            for t in terms
        ):
            return "cqc takes one whole case, or the modes of one set"
    elif whole:
        return f"{row.rule} takes a list, and a list excludes whole cases"
    return None


def _param_problem(row: RowLike, origin: str) -> Optional[str]:
    if row.rule == "envelope":
        if row.envelope not in ENVELOPES:
            return f"envelope {row.envelope!r} is not one of {ENVELOPES}"
    elif row.envelope:
        return f"{row.rule} takes no envelope, got {row.envelope!r}"
    if row.rule == "directional":
        if row.fraction is None or not 0.0 <= row.fraction <= 1.0:
            return f"directional fraction {row.fraction!r} is not in [0, 1]"
    elif row.fraction is not None:
        return f"{row.rule} takes no fraction, got {row.fraction!r}"
    if row.damping is not None:
        if row.rule != "cqc":
            return f"{row.rule} takes no damping, got {row.damping!r}"
        if not 0.0 < row.damping < 1.0:
            return f"cqc damping {row.damping!r} is not in (0, 1)"
    elif origin == "model" and row.rule == "cqc":
        # A case's per-mode damping lives in .user.h5, so model.h5 has no source.
        return "cqc in model.h5 needs a damping ratio in (0, 1)"
    return None


def row_problem(row: RowLike, origin: str) -> Optional[tuple[str, str]]:
    """``(reason, detail)`` for a row that is bad by itself, or ``None``.

    The checks run in this order, and the first that fails names the reason:
    the name, the rule, the terms being there, a ``model.h5`` term naming a case,
    the rule's parameters, each term's address and factor, the term shape the
    rule takes, and a damping source for ``cqc`` over ``mode`` terms. A case term
    in ``model.h5`` is judged before the parameters, so ``cqc`` over one reads
    ``bad-term`` and points at the term, not at the damping it could not use.
    """
    if origin not in _ORIGINS:
        raise ValueError(f"origin must be one of {_ORIGINS}, got {origin!r}")
    problem = name_problem(row.name)
    if problem is not None:
        return "empty-name", f"name {row.name!r}: {problem}"
    if row.rule not in RULES:
        return "unknown-rule", f"rule {row.rule!r} is not one of {RULES}"
    if not row.terms:
        return "no-terms", "a combination needs at least one term"
    if origin == "model":
        for term in row.terms:
            if term.kind in ("spectrum", "reference"):
                return "bad-term", (
                    f"a model.h5 combination cannot reference a {term.kind!r} "
                    "source: its case lives in .user.h5. Declare this "
                    "combination in .user.h5 after the run (ADR 0130 A7)"
                )
    problem = _param_problem(row, origin)
    if problem is not None:
        return "bad-param", problem
    for term in row.terms:
        problem = _term_problem(term)
        if problem is not None:
            return "bad-term", problem
    problem = _shape_problem(row)
    if problem is not None:
        return "bad-term", problem
    if row.rule == "cqc" and row.damping is None and row.terms[0].kind == "mode":
        return "bad-param", "cqc over mode terms needs a damping: modes have no damping source"
    return None


# ---------------------------------------------------------------------------
# The table
# ---------------------------------------------------------------------------


def judge(rows: Sequence[tuple[str, RowLike]]) -> dict[int, tuple[str, str]]:
    """The verdict on every refused row of a table, keyed by position.

    ``rows`` holds ``(origin, row)`` pairs, ``origin`` being ``user`` or
    ``model``. Names are unique among combinations across both files: two rows
    of one file refuse as ``duplicate``, rows of both files as ``clash``, and
    every row of the name goes. A row already refused keeps its own reason. A
    row in a reference cycle, or reaching one at any depth, reads ``cycle``,
    even beside a term that is merely absent. Then a row naming a combination
    that is absent, refused or not visible to it reads ``unresolved``, and so
    does every row that reaches it.
    """
    verdict: dict[int, tuple[str, str]] = {}
    for at, (origin, row) in enumerate(rows):
        problem = row_problem(row, origin)
        if problem is not None:
            verdict[at] = problem

    holders: dict[str, list[int]] = {}
    for at, (_, row) in enumerate(rows):
        if name_problem(row.name) is None:
            holders.setdefault(row.name, []).append(at)
    for name, ats in holders.items():
        if len(ats) < 2:
            continue
        reason = "clash" if len({rows[at][0] for at in ats}) > 1 else "duplicate"
        detail = f"{len(ats)} combinations are named {name!r}"
        for at in ats:
            verdict.setdefault(at, (reason, detail))

    # Names are unique among the rows still standing, so a name is one row.
    standing = {rows[at][1].name: at for at in range(len(rows)) if at not in verdict}

    def target(at: int, source: str) -> Optional[int]:
        """The standing row a reference of row ``at`` reaches, or ``None``."""
        hit = standing.get(source)
        if hit is None or (rows[at][0] == "model" and rows[hit][0] != "model"):
            return None
        return hit

    refs: dict[int, list[int]] = {}
    dangling: set[int] = set()
    for at in standing.values():
        found = []
        for term in rows[at][1].terms:
            if term.kind != "combination":
                continue
            hit = target(at, term.source)
            if hit is None:
                dangling.add(at)
            else:
                found.append(hit)
        refs[at] = found
    users: dict[int, list[int]] = {at: [] for at in refs}
    for at, found in refs.items():
        for hit in found:
            users[hit].append(at)

    # Kahn's peel: a row settles once every row it references has settled; what
    # never settles is in a cycle or reaches one.
    waiting = {at: len(found) for at, found in refs.items()}
    queue = deque(at for at, n in waiting.items() if n == 0)
    settled: set[int] = set()
    while queue:
        at = queue.popleft()
        settled.add(at)
        for user in users[at]:
            waiting[user] -= 1
            if waiting[user] == 0:
                queue.append(user)
    for at in refs:
        if at not in settled:
            verdict[at] = ("cycle", "this combination is in a reference cycle or reaches one")

    # ``unresolved`` from each dangling reference, carried to every row reaching it.
    spread = deque(at for at in dangling if at in settled)
    while spread:
        at = spread.popleft()
        if at in verdict:
            continue
        missing = [
            t.source for t in rows[at][1].terms
            if t.kind == "combination" and (
                target(at, t.source) is None or target(at, t.source) in verdict)
        ]
        verdict[at] = ("unresolved", f"combination {missing[0]!r} does not stand")
        spread.extend(u for u in users[at] if u not in verdict)
    return verdict
