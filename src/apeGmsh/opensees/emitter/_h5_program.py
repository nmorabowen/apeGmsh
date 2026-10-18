"""Emit-order program tape for :mod:`apeGmsh.opensees.emitter.h5`."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

#: ``row`` / ``store`` of a call that wrote nothing (a ``ledger`` verb).
_NO_ROW = -1
#: A call's ``/opensees/program`` ``decl`` when its declaration is not
#: proved at emit time (A7, schema 2.31.0): a call no declaration opens
#: (a node, a part row, the analysis chain), a neutral-declared row, a
#: tagged call whose ``(kind, tag)`` no ``/opensees/decls/tags`` run
#: holds, and a tagless call with no declaration open (a ghost replay,
#: or a side-channel store's call before 2.32.0). Never a guessed id.
#: Also the "no declaration open" state of :meth:`H5Emitter.set_declaration`.
_NO_DECL = -1


#: The magnitude an ``int`` command argument may have and still survive
#: the ``f8`` ``args`` column exactly.
_EXACT_INT = 2 ** 53
#: ``/opensees/commands/arg_kinds`` codes.
_KIND_INT, _KIND_FLOAT, _KIND_STR = 0, 1, 2


@dataclass(frozen=True, slots=True)
class _CommandRecord:
    """One ``/opensees/commands`` row: a call with no typed store.

    ``args`` holds every argument in call order; ``names[i]`` is ``""``
    for a positional argument and the keyword otherwise, so a replay is
    ``getattr(emitter, method)(*positional, **keywords)`` (``command``
    rows prepend ``token``). The values keep their Python type (``int``,
    ``float`` or ``str``) so a replayed line is byte-identical.
    """

    method: str
    token: str
    stage: int
    args: tuple[int | float | str, ...]
    names: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class _EmbeddedRebarRecord:
    """One ``/opensees/constraints/embeddedRebar`` row (K2b, 2.29.0).

    ``args`` are the tie's positional arguments after the tag, as
    ``embedded_rebar_args`` built them, each keeping its Python type
    (``int``, ``float`` or ``str``) so a replayed line is byte-identical.
    """

    ele_tag: int
    args: tuple[int | float | str, ...]
    name: str


def _command_kind(method: str, value: object) -> int:
    """The ``arg_kinds`` code of one command argument; refuse the rest."""
    if isinstance(value, bool) or not isinstance(value, (int, float, str)):
        raise TypeError(
            f"H5Emitter.{method}: argument {value!r} is a "
            f"{type(value).__name__}; a /opensees/commands row holds int, "
            "float or str values only."
        )
    if isinstance(value, int):
        if abs(value) >= _EXACT_INT:
            raise ValueError(
                f"H5Emitter.{method}: int argument {value} does not fit "
                "the f8 args column exactly."
            )
        return _KIND_INT
    return _KIND_FLOAT if isinstance(value, float) else _KIND_STR


class _ProgramTape:
    """The run-length emit order written as ``/opensees/program``.

    Every Protocol call ``H5Emitter`` completes is one emit index
    (1-based, ``count`` of them in all). A run is a maximal stretch of
    consecutive calls with equal ``(method, store, stage, decl)`` whose
    rows are consecutive (or all :data:`_NO_ROW`). ``row`` is the
    record's ordinal among the records ``method`` wrote to ``store`` in
    that ``stage``, in emit order; a call that a partition bracket
    replicates repeats the row of its first capture, so the run points
    at a row already written. ``decl`` is the ``/opensees/decls`` row of
    the declaration that made the call, proved at emit time, else
    :data:`_NO_DECL` (A7, schema 2.31.0); a body sets :attr:`decl` for
    the call in flight.
    """

    __slots__ = (
        "runs", "methods", "stores", "count", "depth", "pending",
        "override", "ordinal", "_method_ix", "_store_ix", "_first_rows",
        "_last", "_last_method", "_last_store", "_last_stage", "decl",
    )

    def __init__(self) -> None:
        # ``[first, count, method, store, row, stage, decl]`` per run.
        self.runs: list[list[int]] = []
        self.methods: list[str] = []
        self.stores: list[str] = []
        self.count = 0
        # Nesting depth: a Protocol method calling another counts once.
        self.depth = 0
        # The row the call in flight takes unless a body overrides it.
        self.pending = _NO_ROW
        self.override: int | None = None
        # Next row per ``(method, stage)``.
        self.ordinal: dict[tuple[str, int], int] = {}
        self._method_ix: dict[str, int] = {}
        self._store_ix: dict[str, int] = {}
        self._first_rows: dict[tuple[Any, ...], int] = {}
        # The open run and its key, so a repeated call extends it without
        # interning (the per-node / per-element hot path).
        self._last: list[int] | None = None
        self._last_method = ""
        self._last_store = ""
        self._last_stage = _NO_ROW
        # The declaration of the call in flight (A7): reset by the
        # wrapper, set by a body that proves it.
        self.decl = _NO_DECL

    def note_plain(
        self, method: str, store: str, stage: int, decl: int,
    ) -> None:
        """Note a call that always appends a fresh row and is never
        replicated: the per-node / per-element hot path, inlined into
        ``node`` and ``element`` instead of the generic wrapper."""
        key = (method, stage)
        ordinal = self.ordinal
        row = ordinal.get(key, 0)
        ordinal[key] = row + 1
        self.count += 1
        last = self._last
        if (last is not None and method == self._last_method
                and store == self._last_store and stage == self._last_stage
                and row == last[4] + last[1] and decl == last[6]):
            last[1] += 1
            return
        self.pending, self.override, self.decl = row, None, decl
        self.note_open(method, store, stage)

    def note_open(self, method: str, store: str, stage: int) -> None:
        """Open a run for a call already counted, at ``self.pending``,
        declared by ``self.decl``."""
        m = self._method_ix.get(method)
        if m is None:
            m = self._method_ix[method] = len(self.methods)
            self.methods.append(method)
        if store:
            st = self._store_ix.get(store)
            if st is None:
                st = self._store_ix[store] = len(self.stores)
                self.stores.append(store)
        else:
            st = _NO_ROW
        row = self.pending if self.override is None else self.override
        run = [self.count, 1, m, st, row, stage, self.decl]
        self.runs.append(run)
        self._last = run
        self._last_method, self._last_store = method, store
        self._last_stage = stage

    def remember(self, key: "tuple[Any, ...]") -> None:
        """Record that ``key``'s first capture is the call in flight."""
        self._first_rows[key] = self.pending

    def repeat(self, key: "tuple[Any, ...]") -> None:
        """The call in flight replicates ``key``'s first capture."""
        self.override = self._first_rows[key]

    def note(
        self, method: str, store: str, stage: int,
        key: "tuple[str, int]",
    ) -> None:
        """Append the completed call to the open run, or open a new one."""
        self.count += 1
        row = self.override
        if row is None:
            row = self.pending
            if row != _NO_ROW:
                self.ordinal[key] = row + 1
        last = self._last
        if (last is not None and method == self._last_method
                and store == self._last_store and stage == self._last_stage
                and self.decl == last[6]
                and (row == last[4] + last[1] if row != _NO_ROW
                     else last[4] == _NO_ROW)):
            last[1] += 1
            return
        self.note_open(method, store, stage)
