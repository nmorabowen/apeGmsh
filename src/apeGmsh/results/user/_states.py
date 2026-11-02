"""The states a combination's terms name, read from a set (ADR 0130 A3, A5, A7).

:class:`UserCombinations` is what ``Results.combinations()`` returns: the set's
combination table (``.user.h5`` ``/combinations`` joined with ``model.h5``
``/opensees/combinations``) and the evaluator, bound to one ``Results``.

A term's source address resolves here:

=============  ===========================================================
``kind``       read from
=============  ===========================================================
``stage``      the ``Results`` stage whose canonical name is the ``source``:
               its stamped name, or ``stage_<i>`` when the run stamped none
               (A3). ``stage_<i>`` of a stamped stage is refused. ``step`` is
               a time index, ``-1`` the last
``mode``       a mode of that host stage, named the same way. ``step`` is the
               ``STEP_<s>`` id of its modal set, ``-1`` the only set;
               ``mode`` is 1-based
``spectrum``,  the ``spectra`` provider: ``Results.combinations()`` passes a
``reference``  :class:`~._spectra_source.UserSpectra` over the set's ``/spectra``
=============  ===========================================================

A quantity is ``(level, component)``, ``level`` one of :data:`LEVELS`. The
recorded levels read ``Results``; ``domain`` (a whole-model value such as
``base_shear``, named by its label) exists only in ``/spectra``, so a ``stage``
or ``mode`` term of a domain quantity refuses ``unresolved``.

A recorded quantity is linear, so a summing rule may combine it (A5), when the
reader translated it from a recorder: every component of a ``.ladruno`` or an
``.mpco``. A native ``results.h5`` also stores what a writer put beside the
recorded responses (the YOO slab's ``wood_armer_*`` and ``rib_shear_*``), so
there a component is linear only if it is a recorder spelling
(:func:`apeGmsh._vocabulary.is_canonical`). Any other name, the empty one
included, refuses ``not-linear`` under a summing rule or a factor; ``envelope``
of factor-1 terms and a lone state read still read it (owner ruling, VU2c).
A derived scalar (von Mises, a principal value) is not linear on any source.

A user result reads its ``STATES`` rows instead, by exact address, after the
same canonical check, so a row stored as ``("stage", "gravity")`` is the one a
term naming ``gravity`` reads. One state read alone has no rule
(:func:`~._evaluate.read_state`): no ``@linear`` check, no badge. A term holds
no path, stem or hash, so a cross-variant term cannot be written, and a state
handle of another ``Results`` is refused (``foreign-handle``).
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Optional, Protocol

import numpy as np

from ..._vocabulary import is_canonical
from .._derived import is_derived, is_shell_derived
from ._descriptor import CombinationTable, StateAddress, resolve
from ._evaluate import (
    Case,
    Combined,
    CombinationRefused,
    Field,
    evaluate,
    is_signed,
    read_state,
)
from ._fields import UserResult
from ._reader import UserFile, read_model_combinations
from ._staleness import staleness

if TYPE_CHECKING:
    from ..Results import Results

__all__ = [
    "CombinationState",
    "LEVELS",
    "RECORDED_LEVELS",
    "SpectraProvider",
    "UserCombinations",
    "canonical_stage_name",
    "domain_keys",
    "entity_keys",
    "node_keys",
    "sub_keys",
    "user_file_beside",
]

#: The levels a recorded result is read on.
RECORDED_LEVELS = ("nodes", "elements", "gauss", "line_stations")
#: The levels a combination is evaluated on: the recorded ones, and ``domain``,
#: which only ``/spectra`` holds (ADR 0130 D7 as A7 amends it).
LEVELS = (*RECORDED_LEVELS, "domain")
_ID = np.dtype([("id", "i8")])
_SUB = np.dtype([("id", "i8"), ("sub", "i8")])
_MODE_STAGE = re.compile(r"(?P<host>.+?)(?:_step_(?P<step>\d+))?_mode_(?P<mode>\d+)\Z")
#: The name a recorder gives a stage the run did not stamp.
_UNSTAMPED = re.compile(r"MODEL_STAGE\[\d+\]\Z")
_PART = re.compile(r"\.part-\d+\Z")
_MPCO_SUFFIX = ".mpco"


def recorded_is_linear(results_file: Optional[Path], level: str, component: str) -> bool:
    """Whether a summing rule may combine recorded ``(level, component)`` (A5).

    A derived scalar (:func:`apeGmsh.results._derived.is_derived` and
    ``is_shell_derived``: von Mises, the principal values, the invariants) is
    computed on read from a tensor, and a sum of it is not its value under the
    sum, so it is not linear on any source. The families are the vocabulary's
    ``DERIVED_*_SCALARS``; the few of them that are linear in the tensor
    (``mean_stress``, ``volumetric_strain``) are refused with the rest, since the
    vocabulary does not mark them apart.

    A ``domain`` value is read from ``/spectra`` only, never from the results
    file, so the file does not judge it. A ``.ladruno`` or ``.mpco`` serves only
    what it recorded. Any other file, and a ``Results`` with no file, is judged
    by the one recorder vocabulary, :func:`apeGmsh._vocabulary.is_canonical`: a
    native ``results.h5`` stores design fields as ordinary components. The
    suffixes are the readers' dispatch (:func:`apeGmsh.results._open.open_results`):
    those two open with a recorder's reader, any other file as a native one.
    """
    if is_derived(component) or is_shell_derived(component):
        return False
    if level == "domain":
        return True
    if results_file is not None:
        from ..readers._ladruno_live import FORK_SUFFIX

        if results_file.suffix.lower() in (FORK_SUFFIX, _MPCO_SUFFIX):
            return True
    return is_canonical(component)


def canonical_stage_name(info: Any) -> str:
    """A stage's source in a term (A3): its stamped name, else ``stage_<i>``."""
    name = info.name
    if not name or name == info.id or _UNSTAMPED.match(name):
        return str(info.id)
    return str(name)


def user_file_beside(results_file: Optional[Path], model_h5: Optional[Path]) -> Optional[Path]:
    """The set's ``<stem>.user.h5``: the first candidate that exists, else ``None``.

    Beside the results file, in order: ``<model stem>.user.h5`` (the model
    archive is ``<stem>.h5``), then the results file's stem without
    ``.part-<N>`` and ``.results``, so ``truss2d.part-0.ladruno`` and
    ``yoo_slab.results.h5`` find their set's file. A model in another folder or
    under another stem (``from_fem``'s ``<snap>.model.h5``) therefore never
    hides the file beside the results. With no results file, the candidate is
    beside the model. An explicit path is the caller's and is never looked up.
    """
    candidates: list[Path] = []
    folder = results_file.parent if results_file is not None else None
    if model_h5 is not None:
        name = model_h5.name
        stem = name[: -len(".h5")] if name.endswith(".h5") else model_h5.stem
        candidates.append((folder or model_h5.parent) / (stem + ".user.h5"))
    if results_file is not None:
        stem = _PART.sub("", results_file.stem)
        stem = stem[: -len(".results")] if stem.endswith(".results") else stem
        candidates.append(results_file.with_name(stem + ".user.h5"))
    return next((c for c in candidates if c.is_file()), None)


def node_keys(ids: Any) -> np.ndarray:
    """Row keys for a nodal field: the node ID."""
    ids = np.asarray(ids, dtype="i8").ravel()
    keys = np.empty(ids.shape[0], dtype=_ID)
    keys["id"] = ids
    return keys


def sub_keys(ids: Any, sub: Any) -> np.ndarray:
    """Row keys for an element field: the element ID and its Gauss, station or node slot."""
    ids = np.asarray(ids, dtype="i8").ravel()
    keys = np.empty(ids.shape[0], dtype=_SUB)
    keys["id"] = ids
    keys["sub"] = np.asarray(sub, dtype="i8").ravel()
    return keys


def entity_keys(kinds: Any, rows: Any) -> np.ndarray:
    """Row keys for an entity field: ``ENTITY_KIND`` and ``ENTITY_ROW``."""
    rows = np.asarray(rows, dtype="i8").ravel()
    kinds = list(kinds)
    width = max([1, *(len(k) for k in kinds)])
    keys = np.empty(rows.shape[0], dtype=[("kind", f"U{width}"), ("row", "i8")])
    keys["kind"] = kinds
    keys["row"] = rows
    return keys


def domain_keys() -> np.ndarray:
    """The one row of a domain value."""
    return node_keys([0])


class SpectraProvider(Protocol):
    """``/spectra`` values for one quantity (:class:`~._spectra_source.UserSpectra`).

    ``allow_stale`` is the caller's ``evaluate(allow_stale=...)``: a provider
    refuses a stale zone's terms ``stale`` unless it is true (A6).
    """

    def field(self, address: StateAddress, quantity: tuple[str, str], *,
              allow_stale: bool = False) -> Field: ...

    def case(self, name: str, quantity: tuple[str, str], *,
             allow_stale: bool = False) -> Case: ...


@dataclass(frozen=True, eq=False)
class CombinationState:
    """A handle on one combination of one :class:`UserCombinations`."""

    name: str
    signed: bool
    owner: "UserCombinations"


# ---------------------------------------------------------------------------
# Recorded results
# ---------------------------------------------------------------------------


def _ordinal_within(ids: np.ndarray) -> np.ndarray:
    """``0, 1, ...`` within each run of one element id: a row's Gauss or station slot."""
    out = np.zeros(ids.shape[0], dtype="i8")
    seen: dict[int, int] = {}
    for i, eid in enumerate(ids.tolist()):
        out[i] = seen.get(eid, 0)
        seen[eid] = out[i] + 1
    return out


class _RecordedSource:
    """One recorded quantity of a ``Results``: ``(level, component)``.

    ``linear`` is :func:`recorded_is_linear` of ``results_file``, the file the
    ``Results`` was opened on; with none given the component is judged by name.
    """

    def __init__(self, results: "Results", level: str, component: str,
                 ids: Optional[Any], spectra: Optional[SpectraProvider], name: str,
                 unbound_stages: Optional[str] = None, allow_stale: bool = False,
                 results_file: Optional[Path] = None) -> None:
        if level not in LEVELS:
            raise ValueError(f"level {level!r} is not one of {LEVELS}")
        if level == "domain" and ids is not None:
            raise ValueError(
                f"ids={ids!r} selects nodes or elements; a domain value ({component!r}) "
                "has no ids")
        self.results = results
        self.level = level
        self.component = component
        self.ids = None if ids is None else np.asarray(ids)
        self.spectra = spectra
        self.name = name
        self.unbound_stages = unbound_stages
        self.allow_stale = bool(allow_stale)
        self.linear = recorded_is_linear(
            None if results_file is None else Path(results_file), level, component)

    def stored(self, name: str) -> Optional[Field]:
        return None  # a recorded file stores no combination

    def _refuse(self, reason: str, detail: str) -> CombinationRefused:
        return CombinationRefused(self.name, reason, detail)

    def _recorded(self, address: StateAddress) -> None:
        """A domain quantity has no recorded value at a stage or a mode."""
        if self.level == "domain":
            raise self._refuse(
                "unresolved", f"a {address.kind} term reads Results, which records no domain "
                f"quantity; {self.component!r} on the domain is read from /spectra only")

    def _host(self, source: str) -> Any:
        if self.unbound_stages is not None:
            raise self._refuse("unresolved", self.unbound_stages)
        hosts = [s for s in self.results.stages if s.kind != "mode"]
        hits = [s for s in hosts if canonical_stage_name(s) == source]
        if len(hits) == 1:
            return hits[0]
        if len(hits) > 1:
            raise self._refuse("unresolved", f"{len(hits)} stages are named {source!r}")
        stamped = [s for s in hosts if s.id == source]
        if stamped:
            raise self._refuse(
                "unresolved", f"{source!r} is stamped {stamped[0].name!r}: a term names a "
                "stamped stage by its stamped name (ADR 0130 A3)")
        raise self._refuse(
            "unresolved", f"no stage {source!r} in {[canonical_stage_name(s) for s in hosts]}")

    def check(self, address: StateAddress) -> None:
        """Refuse a ``stage`` or ``mode`` source the stage list does not name canonically.

        A user result's stored row is read, never ``Results``, so a domain user
        result passes here: only the name is checked.
        """
        if address.kind == "stage":
            self._host(address.source)
        elif address.kind == "mode":
            self._mode_stage(address)

    def _mode_stage(self, address: StateAddress) -> Any:
        host = self._host(address.source)
        found = []
        for s in self.results.stages:
            m = _MODE_STAGE.match(s.id) if s.kind == "mode" else None
            if m is not None and m.group("host") == host.id:
                found.append((None if m.group("step") is None else int(m.group("step")),
                              int(m.group("mode")), s))
        sets = sorted({step for step, _, _ in found}, key=lambda v: -1 if v is None else v)
        if not sets:
            raise self._refuse("unresolved", f"stage {address.source!r} holds no modes")
        if address.step == -1:
            if len(sets) != 1:
                raise self._refuse(
                    "unresolved", f"stage {address.source!r} holds {len(sets)} modal sets "
                    f"(STEP ids {sets}); name one by its STEP id")
            want = sets[0]
        else:
            if sets == [None]:
                raise self._refuse(
                    "unresolved", f"stage {address.source!r} holds one modal set, and this "
                    "reader does not expose its STEP id; name it with step -1")
            want = address.step
        hits = [s for step, mode, s in found if step == want and mode == address.mode]
        if len(hits) != 1:
            raise self._refuse(
                "unresolved", f"stage {address.source!r} has no mode {address.mode} "
                f"in the set at STEP {want}")
        return hits[0]

    def _read(self, scoped: "Results", step: Optional[int]) -> Field:
        kw: dict[str, Any] = {"component": self.component}
        if step is not None:
            kw["time"] = step
        if self.ids is not None:
            kw["ids"] = self.ids
        if self.level == "nodes":
            nslab = scoped.nodes.get(**kw)
            values = np.asarray(nslab.values)[-1]
            keys = node_keys(nslab.node_ids)
        elif self.level == "elements":
            eslab = scoped.elements.get(**kw)
            block = np.asarray(eslab.values)[-1]  # (E, npe)
            eids = np.repeat(np.asarray(eslab.element_ids), block.shape[1])
            keys = sub_keys(eids, np.tile(np.arange(block.shape[1]), block.shape[0]))
            values = block.ravel()
        else:
            if self.level not in ("gauss", "line_stations"):  # _recorded refused the domain
                raise ValueError(f"level {self.level!r} is not a recorded level {RECORDED_LEVELS}")
            reader = scoped.elements.gauss if self.level == "gauss" else scoped.elements.line_stations
            slab = reader.get(**kw)
            values = np.asarray(slab.values)[-1]
            eids = np.asarray(slab.element_index)
            keys = sub_keys(eids, _ordinal_within(eids))
        return Field(keys, (self.component,), np.asarray(values, dtype="f8").reshape(-1, 1))

    def field(self, address: StateAddress) -> Field:
        if address.kind in ("stage", "mode"):
            self._recorded(address)
        if address.kind == "stage":
            host = self._host(address.source)
            if not -1 <= address.step < host.n_steps:
                raise self._refuse(
                    "unresolved", f"stage {address.source!r} has {host.n_steps} steps, "
                    f"not step {address.step}")
            return self._read(self.results.stage(host.id), address.step)
        if address.kind == "mode":
            return self._read(self.results.stage(self._mode_stage(address).id), None)
        if address.kind in ("spectrum", "reference"):
            if self.spectra is None:
                raise self._refuse(
                    "unresolved", f"a {address.kind} term reads /spectra, and no spectra "
                    "provider was given (Results.combinations() passes UserSpectra)")
            return self._only_ids(self.spectra.field(
                address, (self.level, self.component), allow_stale=self.allow_stale))
        raise self._refuse("bad-term", f"a {address.kind} term is not a state")

    def eigenvalue(self, address: StateAddress) -> float:
        return float(self.results.stage(self._mode_stage(address).id).eigenvalue)

    def case(self, name: str) -> Case:
        if self.spectra is None:
            raise self._refuse(
                "unresolved", f"case {name!r} is read from /spectra, and no spectra provider "
                "was given (Results.combinations() passes UserSpectra)")
        found = self.spectra.case(name, (self.level, self.component),
                                  allow_stale=self.allow_stale)
        if self.ids is None:
            return found
        return Case(found.mode, found.lam, found.sa_period, found.damping,
                    tuple(self._only_ids(f) for f in found.increments))

    def _only_ids(self, f: Field) -> Field:
        """The rows of a ``/spectra`` field at ``ids``, as a ``Results`` read selects them.

        An id the case does not hold refuses: a NaN row would hide it.
        """
        if self.ids is None:
            return f
        want = np.unique(np.asarray(self.ids, dtype="i8").ravel())
        have = np.asarray(f.keys["id"], dtype="i8")
        missing = np.setdiff1d(want, have)
        if missing.size:
            raise self._refuse(
                "unresolved", f"/spectra holds no row for ids {missing.tolist()} of "
                f"{self.component!r}")
        keep = np.isin(have, want)
        return Field(f.keys[keep], f.components, f.values[keep])


# ---------------------------------------------------------------------------
# User results
# ---------------------------------------------------------------------------


def user_result_rows(result: UserResult) -> tuple[np.ndarray, tuple[str, ...], np.ndarray]:
    """``(keys, components, data (S, n, c))`` of a user result, element buckets flattened.

    An element row is ``(ID, Gauss index)``: GP-major within a block, the
    ``.ladruno`` layout (``parse_blocks``).
    """
    if result.target == "nodes":
        assert result.ids is not None and result.data is not None
        return node_keys(result.ids), result.components, np.asarray(result.data)
    if result.target == "entities":
        assert result.entity_row is not None and result.data is not None
        return (entity_keys(result.entity_kind or (), result.entity_row),
                result.components, np.asarray(result.data))
    if result.target == "domain":
        assert result.data is not None
        return domain_keys(), result.components, np.asarray(result.data)
    components: list[str] = []
    for b in result.buckets:
        for names in b.components:
            components.extend(c for c in names if c not in components)
    rows: dict[tuple[int, int], dict[str, np.ndarray]] = {}
    for b in result.buckets:
        data = np.asarray(b.data)
        col = 0
        for names, gauss_id, mult in zip(b.components, b.gauss_ids, b.multiplicity):
            for gp in range(mult):
                g = gp if mult > 1 else gauss_id
                for k, eid in enumerate(np.asarray(b.ids).tolist()):
                    cell = rows.setdefault((int(eid), int(g)), {})
                    for off, comp in enumerate(names):
                        cell[comp] = data[:, k, col + off]
                col += len(names)
    order = sorted(rows)
    s = next(iter(np.asarray(b.data).shape[0] for b in result.buckets))
    out = np.full((s, len(order), len(components)), np.nan)
    for i, key in enumerate(order):
        for comp, column in rows[key].items():
            out[:, i, components.index(comp)] = column
    keys = sub_keys([k[0] for k in order], [k[1] for k in order])
    return keys, tuple(components), out


class _UserResultSource:
    """One user result: its ``STATES`` rows, and ``@linear`` (A5)."""

    def __init__(self, result: UserResult, recorded: Optional[_RecordedSource], name: str) -> None:
        if result.states is None:
            raise CombinationRefused(
                name, "state-free", f"user result {result.name!r} is state-free: it is "
                "shown as it is, never combined")
        self.result = result
        self.linear = bool(result.linear)
        self.keys, self.components, self.data = user_result_rows(result)
        self.rows = {address: i for i, address in enumerate(result.states)}
        self.recorded = recorded
        self.name = name

    def _row(self, address: StateAddress) -> Optional[Field]:
        i = self.rows.get(address)
        if i is None:
            return None
        return Field(self.keys, self.components, self.data[i])

    def stored(self, name: str) -> Optional[Field]:
        return self._row(StateAddress("combination", name, -1, 0))

    def field(self, address: StateAddress) -> Field:
        if self.recorded is not None:
            self.recorded.check(address)
        found = self._row(address)
        if found is None:
            raise CombinationRefused(
                self.name, "unresolved", f"user result {self.result.name!r} stores no row "
                f"for {address}")
        return found

    def eigenvalue(self, address: StateAddress) -> float:
        if self.recorded is None:
            raise CombinationRefused(self.name, "unresolved", "no Results to read the eigenvalue from")
        return self.recorded.eigenvalue(address)

    def case(self, name: str) -> Case:
        raise CombinationRefused(
            self.name, "unresolved", f"case {name!r}: a user result is combined over a case "
            "only through a stored row (ADR 0130 A5)")


# ---------------------------------------------------------------------------
# The set
# ---------------------------------------------------------------------------


class UserCombinations:
    """A set's User Combinations, evaluated per quantity on read (ADR 0130 D4).

    Built by :meth:`apeGmsh.results.Results.combinations`. ``handle`` maps a
    state handle to its stage id: ``None`` when the object is not a
    ``Results``, and :class:`CombinationRefused` (``foreign-handle``) when it is
    one of another run.
    """

    def __init__(
        self,
        results: "Results",
        *,
        user_h5: Optional[Path],
        model_h5: Optional[Path],
        results_file: Optional[Path],
        handle: Callable[[Any], Optional[str]],
        spectra: Optional[SpectraProvider] = None,
        unbound_stages: Optional[str] = None,
    ) -> None:
        self._results = results
        self._unbound_stages = unbound_stages
        self._handle = handle
        self.user_h5 = user_h5
        self.model_h5 = model_h5
        self.results_file = results_file
        self.spectra = spectra
        self._user = UserFile(user_h5) if user_h5 is not None else None
        if self._user is not None:
            self.table: CombinationTable = self._user.combinations(model_h5=model_h5)
        else:
            self.table = resolve([], read_model_combinations(model_h5) if model_h5 else [])

    @property
    def names(self) -> tuple[str, ...]:
        """The combinations in force: ``.user.h5`` rows, then ``model.h5`` rows."""
        return tuple(r.name for r in (*self.table.rows, *self.table.model_rows))

    def state(self, name: str) -> CombinationState:
        """A handle on combination ``name``; ``signed`` says if it draws as a shape."""
        return CombinationState(name, is_signed(self.table, name), self)

    def _target(self, state: Any) -> tuple[Optional[str], Optional[StateAddress]]:
        """``(combination name, None)``, or ``(None, address)`` for one state alone."""
        if isinstance(state, str):
            return state, None
        if isinstance(state, CombinationState):
            if state.owner is not self:
                raise CombinationRefused(
                    state.name, "foreign-handle", "this handle belongs to another set's "
                    "combinations; a combination is valid within one variant (ADR 0130 D4)")
            return state.name, None
        if isinstance(state, StateAddress):
            return None, state
        stage_id = self._handle(state)
        if stage_id is None:
            raise TypeError(
                f"state must be a combination name, a CombinationState, a StateAddress "
                f"or a stage-scoped Results of this run, got {type(state).__name__}")
        return None, self._address_of(stage_id)

    def _address_of(self, stage_id: str) -> StateAddress:
        stages = {s.id: s for s in self._results.stages}
        info = stages[stage_id]
        if info.kind != "mode":
            return StateAddress("stage", canonical_stage_name(info), -1, 0)
        m = _MODE_STAGE.match(info.id)
        if m is None or m.group("host") not in stages:
            raise CombinationRefused("", "unresolved", f"mode stage {info.id!r} has no host")
        step = -1 if m.group("step") is None else int(m.group("step"))
        host = canonical_stage_name(stages[m.group("host")])
        return StateAddress("mode", host, step, int(m.group("mode")))

    def evaluate(
        self,
        state: Any,
        component: Optional[str] = None,
        *,
        level: str = "nodes",
        ids: Optional[Any] = None,
        user_result: Optional[str] = None,
        allow_stale: bool = False,
    ) -> Combined:
        """The value of ``state`` for one quantity.

        Name the quantity with ``component`` (and ``level``, one of
        :data:`LEVELS`) for a recorded result or a ``/spectra`` value, or with
        ``user_result`` for a ``.user.h5`` result. ``level="domain"`` names a
        whole-model value by its label (``base_shear``, an extractor's name),
        which only ``/spectra`` holds. A stale user result or ``/spectra``
        zone is refused unless ``allow_stale=True`` (A6). Raises
        :class:`CombinationRefused`.
        """
        name, address = self._target(state)
        label = name if name is not None else f"<{address}>"
        if (component is None) == (user_result is None):
            raise ValueError("pass exactly one of component= and user_result=")
        recorded = _RecordedSource(
            self._results, level, component or "", ids, self.spectra, label,
            unbound_stages=self._unbound_stages, allow_stale=allow_stale,
            results_file=self.results_file)
        source: Any = recorded
        if user_result is not None:
            if self._user is None:
                raise CombinationRefused(label, "unresolved", "this set has no .user.h5")
            if not allow_stale:
                state_of = staleness(
                    self._user, model_h5=self.model_h5, results_file=self.results_file,
                ).results.get(user_result)
                if state_of is not None and state_of.stale:
                    raise CombinationRefused(
                        label, "stale", f"user result {user_result!r} is stale "
                        f"({state_of.reason}); pass allow_stale=True to read it")
            try:
                result = self._user.read_result(user_result)
            except KeyError as exc:
                raise CombinationRefused(label, "unresolved", str(exc.args[0])) from exc
            source = _UserResultSource(result, recorded, label)
        if address is not None:
            return read_state(address, source)
        assert name is not None
        return evaluate(self.table, name, source)
