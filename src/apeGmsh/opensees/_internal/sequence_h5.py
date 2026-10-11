"""``/sequence``: the readable-stages zone of ``model.h5`` (ADR 0112 D2b).

The zone is a projection, never a second registry: :func:`derive_sequence`
reads the ``/opensees`` zone the composer has just written back through
:class:`~apeGmsh.opensees.emitter.h5_reader.H5Model`, and
:func:`write_sequence_into` stores the result as two flat tables.  Its
only input is the file, so the zone is single-sourced by construction; an
AST lock (``tests/opensees/test_sequence_h5.py``) bars this module from
importing the record or primitive modules and from reaching past the
reader's public methods.

Layout (``architecture/h5-schema.md``, "/sequence")::

    /sequence  @derived_from @model_hash @staged i1
      stages/   order i4, name, analysis, integrator, algorithm, test,
                system, numberer, handler, integrator_args, algorithm_args,
                test_args, system_args, handler_args (JSON),
                steps i4 (-1 = not archived), dt f8 (NaN = static or not
                archived), set_time f8 (NaN = not set), domain_change i1,
                source
      changes/  stage i4, op, what, name ("" = unnamed), count i4, source

A staged archive gives one ``stages`` row per ``/opensees/stages/stage_NNN``
(``@staged`` 1).  A vanilla deck gives one implicit stage from
``/opensees/analysis`` (``@staged`` 0); a model-only file (neither) gives
none.  Global declarations (``/opensees/patterns``, ``recorders``,
``regions``, ``initial_stress``, ``bcs``, ``constraints`` and the global
``rayleigh`` command) are in force from the first analysis, so their
change rows carry stage 0.  Every row's ``source`` names the archive path it was read from;
``<path>[i]`` is the ``i``-th record of that container in reader order.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import Any, Final

import numpy as np

from ..emitter.h5_reader import H5Model
from ..emitter.verbs import VERBS
from .schema_version import SEQUENCE_KEY, SEQUENCE_SCHEMA_VERSION

__all__ = [
    "ChangeRow",
    "Sequence",
    "SequenceError",
    "SequenceOverflowError",
    "StageRow",
    "derive_sequence",
    "write_sequence_into",
]

#: The analysis-chain tokens, one ``stages`` column each.
CHAIN_TOKENS: Final[tuple[str, ...]] = (
    "analysis", "integrator", "algorithm", "test", "system", "numberer",
    "handler",
)
#: The chain tokens' positional-argument attrs, one JSON column each.
CHAIN_ARGS: Final[tuple[str, ...]] = (
    "integrator_args", "algorithm_args", "test_args", "system_args",
    "handler_args",
)
#: Attrs only the global ``/opensees/analysis`` group carries.
_GLOBAL_ANALYZE: Final[tuple[str, ...]] = ("analyze_steps", "analyze_dt")
#: Chain attrs the zone shows no column for, on purpose: a partitioned
#: deck's runtime fallback (ADR 0027 INV-5).  The token column holds the
#: primary, which is the canonical choice; the fallback only replaces it
#: on a build that lacks the primary.
_NOT_PROJECTED: Final[tuple[str, ...]] = (
    "numberer_runtime_fallback", "system_runtime_fallback",
)

#: Every ``(op, what)`` pair a ``changes`` row may carry.  The V3b lock
#: (``test_sequence_h5.py``) holds it equal to the ``seq`` pairs of
#: ``emitter/verbs.py::VERBS`` plus ``("activate", "group")``, which has
#: no verb of its own (a stage's ``activated_pgs`` list).
CHANGE_KINDS: Final[frozenset[tuple[str, str]]] = frozenset({
    ("add", "fix"), ("add", "mass"), ("add", "constraint"),
    ("add", "pattern"), ("add", "region"), ("add", "recorder"),
    ("add", "initial_stress"), ("activate", "element"),
    ("activate", "group"), ("activate", "absorbing"), ("set", "rayleigh"),
    ("set", "material_stage"), ("remove", "fix"), ("remove", "element"),
})

#: The MP-constraint kinds a stage archives (``stage_NNN/constraints``).
_CONSTRAINT_KINDS: Final[tuple[tuple[str, str], ...]] = (
    ("equalDOF", "equal_dofs"),
    ("rigidLink", "rigid_links"),
    ("rigidDiaphragm", "rigid_diaphragms"),
    ("embeddedNode", "embedded_nodes"),
)

#: The store prefix of an archived MP constraint's program run.
_CONSTRAINT_STORE: Final[str] = "{scope}/constraints/"

_INT32_MAX: Final[int] = 2**31 - 1
_INT32_MIN: Final[int] = -(2**31)
_GROUP: Final[str] = "sequence"
_TMP_GROUP: Final[str] = "_sequence_partial"


class SequenceError(ValueError):
    """The ``/opensees`` zone cannot be projected into ``/sequence``."""


class SequenceOverflowError(SequenceError, OverflowError):
    """An integer column value does not fit int32 (V0 decision 3)."""


@dataclass(frozen=True, slots=True)
class StageRow:
    """One ``/sequence/stages`` row."""

    order: int
    name: str
    chain: tuple[str, ...]      # aligned to CHAIN_TOKENS, "" when unset
    args: tuple[str, ...]       # aligned to CHAIN_ARGS, JSON text
    steps: int
    dt: float
    set_time: float
    domain_change: int
    source: str


@dataclass(frozen=True, slots=True)
class ChangeRow:
    """One ``/sequence/changes`` row."""

    stage: int
    op: str
    what: str
    name: str
    count: int
    source: str


@dataclass(frozen=True, slots=True)
class Sequence:
    """The derived zone: ``staged`` is ``@staged``."""

    staged: bool
    stages: tuple[StageRow, ...]
    changes: tuple[ChangeRow, ...]


# ---------------------------------------------------------------------------
# Derivation (reads only through H5Model)
# ---------------------------------------------------------------------------


def _int32(value: int, where: str) -> int:
    v = int(value)
    if not _INT32_MIN <= v <= _INT32_MAX:
        raise SequenceOverflowError(
            f"/sequence: {where} = {v} does not fit int32; the zone stores "
            "int32 only (ADR 0112 D2b, V0 decision 3)."
        )
    return v


def _plain(value: Any) -> Any:
    """A chain argument as a JSON-ready Python value (NaN/inf -> null)."""
    if isinstance(value, (np.ndarray, np.generic)):
        value = value.tolist()
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _chain_columns(
    attrs: dict[str, Any], where: str, *, allowed_extra: tuple[str, ...],
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    unknown = sorted(
        k for k in attrs
        if k not in CHAIN_TOKENS and k not in CHAIN_ARGS
        and k not in _NOT_PROJECTED and k not in allowed_extra
    )
    if unknown:
        # Fail closed: a chain attr this zone has no column for would
        # vanish from the readable view without a trace.
        raise SequenceError(
            f"{where}: attrs {unknown} have no /sequence column; extend "
            "_internal/sequence_h5.py and the /sequence schema."
        )
    chain = tuple(str(attrs.get(k, "")) for k in CHAIN_TOKENS)
    args = tuple(
        json.dumps(_plain(attrs[k]) if k in attrs else [], allow_nan=False)
        for k in CHAIN_ARGS
    )
    return chain, args


class _Names:
    """Declaration names from ``/opensees/decls`` (K1-6).

    A file without the table (below opensees 2.25.0, or one no bridge emit
    wrote) names nothing: every row is ``""``.  With the table, a record
    it does not cover is a malformed archive and refuses.
    """

    def __init__(self, table: Any) -> None:
        self._t = table

    def tag(self, kind: str, tag: int, where: str) -> str:
        if self._t is None:
            return ""
        try:
            return str(self._t.for_tag(kind, int(tag)).name)
        except KeyError as exc:
            raise SequenceError(
                f"{where}: /opensees/decls has no {kind} tag {tag}."
            ) from exc

    def rows(self, store: str, n: int, where: str) -> list[str]:
        """One declaration name per row of tagless ``store``."""
        if n == 0:
            return []
        if self._t is None:
            return [""] * n
        decl_rows = self._t.rows.get(store)
        if decl_rows is None or len(decl_rows) != n:
            raise SequenceError(
                f"{where}: /opensees/decls/rows/{store} does not cover its "
                f"{n} record(s)."
            )
        return [str(self._t.decls[i].name) for i in decl_rows]

    def grouped(
        self, store: str, n: int, where: str,
    ) -> list[tuple[str, int]]:
        """``(name, count)`` per declaration of ``store``'s rows, in
        first-appearance order (one row per declaration, not per name)."""
        if n == 0:
            return []
        if self._t is None:
            return [("", n)]
        names = self.rows(store, n, where)
        idx = self._t.rows[store]
        order: dict[int, list[Any]] = {}
        for i, nm in zip(idx, names):
            if i in order:
                order[i][1] += 1
            else:
                order[i] = [nm, 1]
        return [(str(nm), int(c)) for nm, c in order.values()]

    def global_rows(self, store: str) -> list[tuple[str, int]]:
        """``(name, count)`` per declaration of a global tagless store
        that the reader exposes only through the table (``bcs/fix``,
        ``bcs/mass``); nothing without the table or the store."""
        if self._t is None or store not in self._t.rows:
            return []
        n = len(self._t.rows[store])
        return self.grouped(store, n, f"/opensees/{store}")


def _by_name(names: list[str]) -> list[tuple[str, int]]:
    out: dict[str, int] = {}
    for nm in names:
        out[nm] = out.get(nm, 0) + 1
    return list(out.items())


def _global_changes(model: H5Model, names: _Names) -> list[ChangeRow]:
    """Stage-0 rows for the declarations outside any stage."""
    rows: list[ChangeRow] = []

    def add(op: str, what: str, name: str, count: int, source: str) -> None:
        rows.append(ChangeRow(0, op, what, name, count, source))

    for store, what in (("bcs/fix", "fix"), ("bcs/mass", "mass")):
        for nm, c in names.global_rows(store):
            add("add", what, nm, c, f"/opensees/{store}")
    rows.extend(_global_constraint_changes(model, names))
    for i, reg in enumerate(model.regions()):
        add("add", "region",
            names.tag("region", reg.tag, f"/opensees/regions[{i}]"),
            1, f"/opensees/regions[{i}]")
    cmds = model.commands()
    ray = [j for j, c in enumerate(cmds)
           if c.method == "rayleigh" and c.stage == -1]
    cmd_names = (names.rows("commands", len(cmds), "/opensees/commands")
                 if ray else [])
    for j in ray:
        add("set", "rayleigh", cmd_names[j], 1, f"/opensees/commands[{j}]")
    for p in model.patterns():
        src = f"/opensees/patterns/{p.type_token}_{p.tag}"
        add("add", "pattern", names.tag("pattern", p.tag, src), 1, src)
    recs = model.recorders()
    rec_names = names.rows("recorders", len(recs), "/opensees/recorders")
    for i, nm in enumerate(rec_names):
        add("add", "recorder", nm, 1, f"/opensees/recorders[{i}]")
    for i, st in enumerate(model.initial_stress()):
        add("add", "initial_stress", str(st.name), 1,
            f"/opensees/initial_stress[{i}]")
    return rows


def _global_constraint_changes(
    model: H5Model, names: _Names,
) -> list[ChangeRow]:
    """Stage-0 rows for the global MP constraints.

    The reader exposes ``/opensees/constraints/<kind>`` only through
    ``/opensees/program``: one run per ``add constraint`` call (the
    ``seq`` of :data:`VERBS`), with its store and count.  An archived
    kind is one row per declaration when ``/opensees/decls`` covers its
    store, else one unnamed row (a constraint the mesh carried, not a
    bridge declaration); a ledger verb, which the archive does not carry,
    is one row naming its program run.  An ``add constraint`` run whose
    store is neither refuses.
    """
    counts: dict[str, int] = {}
    ledger: list[ChangeRow] = []
    for i, run in enumerate(model.program()):
        verb = VERBS.get(run.method)
        if run.stage != -1 or verb is None or (
                verb.seq_op, verb.seq_what) != ("add", "constraint"):
            continue
        if run.store == "":
            ledger.append(ChangeRow(
                0, "add", "constraint", "", run.count,
                f"/opensees/program[{i}]"))
            continue
        if not run.store.startswith(_CONSTRAINT_STORE):
            raise SequenceError(
                f"/opensees/program[{i}]: {run.method} writes "
                f"{run.store!r}, not a constraints/<kind> store."
            )
        kind = run.store[len(_CONSTRAINT_STORE):]
        counts[kind] = counts.get(kind, 0) + run.count
    rows: list[ChangeRow] = []
    for kind, n in counts.items():
        store = f"constraints/{kind}"
        src = f"/opensees/{store}"
        named = names.global_rows(store)
        if not named:
            rows.append(ChangeRow(0, "add", "constraint", "", n, src))
            continue
        if sum(c for _nm, c in named) != n:
            raise SequenceError(
                f"{src}: /opensees/decls covers "
                f"{sum(c for _nm, c in named)} of its {n} record(s)."
            )
        rows.extend(ChangeRow(0, "add", "constraint", nm, c, src)
                    for nm, c in named)
    return rows + ledger


def _stage_changes(k: int, st: Any, names: _Names) -> list[ChangeRow]:
    """The rows of stage ``k`` (a ``StageRecordRO`` from the reader)."""
    base = f"stages/stage_{k:03d}"
    pre = f"/opensees/{base}"
    rows: list[ChangeRow] = []

    def add(op: str, what: str, name: str, count: int, source: str) -> None:
        rows.append(ChangeRow(k, op, what, name, count, source))

    def grouped(op: str, what: str, store: str, n: int) -> None:
        for nm, c in names.grouped(f"{base}/{store}", n, f"{pre}/{store}"):
            add(op, what, nm, c, f"{pre}/{store}")

    if st.owned_element_ids:
        add("activate", "element", "", len(st.owned_element_ids),
            f"{pre}/owned_element_ids")
    for pg in st.activated_pgs:
        add("activate", "group", str(pg), 1, f"{pre}/activated_pgs")
    grouped("add", "fix", "bcs/fix", len(st.fixes))
    grouped("add", "mass", "bcs/mass", len(st.masses))
    for kind, attr in _CONSTRAINT_KINDS:
        recs = getattr(st, attr)
        for nm, c in _by_name([str(r.name) for r in recs]):
            add("add", "constraint", nm, c, f"{pre}/constraints/{kind}")
    for i, reg in enumerate(st.regions):
        src = f"{pre}/regions[{i}]"
        add("add", "region", names.tag("region", reg.tag, src), 1, src)
    grouped("set", "rayleigh", "rayleigh", len(st.rayleighs))
    for p in st.patterns:
        src = f"{pre}/patterns/{p.type_token}_{p.tag}"
        add("add", "pattern", names.tag("pattern", p.tag, src), 1, src)
    rec_names = names.rows(
        f"{base}/recorders", len(st.recorders), f"{pre}/recorders")
    for i, nm in enumerate(rec_names):
        add("add", "recorder", nm, 1, f"{pre}/recorders[{i}]")
    for i, rec in enumerate(st.initial_stress):
        add("add", "initial_stress", str(rec.name), 1,
            f"{pre}/initial_stress[{i}]")
    ab_names = names.rows(
        f"{base}/activate_absorbing", len(st.activate_absorbing),
        f"{pre}/activate_absorbing")
    for i, ((_pg, elems), nm) in enumerate(
            zip(st.activate_absorbing, ab_names)):
        add("activate", "absorbing", nm,
            len(elems) if elems is not None else 1,
            f"{pre}/activate_absorbing[{i}]")
    grouped("remove", "fix", "remove_sp", len(st.remove_sps))
    grouped("remove", "element", "remove_element", len(st.remove_elements))
    grouped("set", "material_stage", "update_material_stage",
            len(st.update_material_stages))
    return rows


def _f64(value: Any) -> float:
    return math.nan if value is None else float(value)


def derive_sequence(model: H5Model) -> Sequence:
    """Project ``model``'s ``/opensees`` zone into the ``/sequence`` rows.

    Reads only through ``model``'s public methods.  Raises
    :class:`SequenceError` on an archive this projection cannot represent
    and :class:`SequenceOverflowError` on an int32 overflow, before
    anything is written.
    """
    names = _Names(model.declarations())
    stages: list[StageRow] = []
    changes: list[ChangeRow] = []
    records = model.stages()
    staged = bool(records)
    if staged:
        for k, st in enumerate(records):
            src = f"/opensees/stages/stage_{k:03d}"
            chain, args = _chain_columns(
                dict(st.chain_attrs), f"{src}/analysis", allowed_extra=())
            stages.append(StageRow(
                order=_int32(k, f"{src} order"),
                name=str(st.name),
                chain=chain, args=args,
                steps=_int32(st.analyze_steps, f"{src}@analyze_steps"),
                dt=_f64(st.analyze_dt),
                set_time=_f64(st.set_time),
                domain_change=int(bool(st.domain_changed)),
                source=src,
            ))
    else:
        attrs = model.analysis()
        if attrs is not None:
            src = "/opensees/analysis"
            chain, args = _chain_columns(
                attrs, src, allowed_extra=_GLOBAL_ANALYZE)
            steps = attrs.get("analyze_steps")
            stages.append(StageRow(
                order=0, name="", chain=chain, args=args,
                steps=(_int32(steps, f"{src}@analyze_steps")
                       if steps is not None else -1),
                dt=_f64(attrs.get("analyze_dt")),
                set_time=math.nan,
                domain_change=0,
                source=src,
            ))
    if stages:
        changes.extend(_global_changes(model, names))
        for k, st in enumerate(records):
            changes.extend(_stage_changes(k, st, names))
    for row in changes:
        if (row.op, row.what) not in CHANGE_KINDS:  # pragma: no cover
            raise SequenceError(f"/sequence: unknown change {row!r}.")
        _int32(row.count, f"{row.source} count")
    return Sequence(staged=staged, stages=tuple(stages), changes=tuple(changes))


# ---------------------------------------------------------------------------
# Writing
# ---------------------------------------------------------------------------


def _write_tables(g: Any, seq: Sequence) -> None:
    import h5py

    s = h5py.string_dtype(encoding="utf-8")

    def strs(grp: Any, name: str, values: list[str]) -> None:
        grp.create_dataset(
            name, data=np.array(values, dtype=object).reshape(-1), dtype=s)

    def ints(grp: Any, name: str, values: list[int], dt: Any) -> None:
        grp.create_dataset(name, data=np.asarray(values, dtype=dt).reshape(-1))

    st = g.create_group("stages")
    rows = seq.stages
    ints(st, "order", [r.order for r in rows], np.int32)
    strs(st, "name", [r.name for r in rows])
    for i, tok in enumerate(CHAIN_TOKENS):
        strs(st, tok, [r.chain[i] for r in rows])
    for i, key in enumerate(CHAIN_ARGS):
        strs(st, key, [r.args[i] for r in rows])
    ints(st, "steps", [r.steps for r in rows], np.int32)
    st.create_dataset(
        "dt", data=np.asarray([r.dt for r in rows], dtype=np.float64))
    st.create_dataset(
        "set_time",
        data=np.asarray([r.set_time for r in rows], dtype=np.float64))
    ints(st, "domain_change", [r.domain_change for r in rows], np.int8)
    strs(st, "source", [r.source for r in rows])

    ch = g.create_group("changes")
    crow = seq.changes
    ints(ch, "stage", [r.stage for r in crow], np.int32)
    strs(ch, "op", [r.op for r in crow])
    strs(ch, "what", [r.what for r in crow])
    strs(ch, "name", [r.name for r in crow])
    ints(ch, "count", [r.count for r in crow], np.int32)
    strs(ch, "source", [r.source for r in crow])


def write_sequence_into(f: Any) -> None:
    """Derive ``/sequence`` from ``f``'s ``/opensees`` and write it.

    ``f`` is an h5py file open for writing whose ``/opensees`` (and
    ``/meta/lineage``) is complete.  An existing ``/sequence`` and its
    ``/meta`` key are dropped first, so a rewrite re-derives (V0 Q7) and
    a failure leaves neither a stale nor a half-written zone: the tables
    are built in memory, written to a scratch group, then moved into
    place, and the version key is stamped last.  A file without
    ``/opensees`` gets no zone.
    """
    from .lineage import read_stored_lineage

    if "meta" not in f:
        raise SequenceError("/sequence: the file has no /meta group.")
    meta = f["meta"]
    if SEQUENCE_KEY in meta.attrs:
        del meta.attrs[SEQUENCE_KEY]
    for name in (_GROUP, _TMP_GROUP):
        if name in f:
            del f[name]
    if "opensees" not in f:
        return
    seq = derive_sequence(H5Model(f, path=str(f.filename)))
    _fem, model_hash, _res = read_stored_lineage(meta)
    try:
        g = f.create_group(_TMP_GROUP)
        g.attrs["derived_from"] = "/opensees"
        g.attrs["model_hash"] = model_hash or ""
        g.attrs["staged"] = np.int8(1 if seq.staged else 0)
        _write_tables(g, seq)
        f.move(_TMP_GROUP, _GROUP)
    except BaseException:
        if _TMP_GROUP in f:
            del f[_TMP_GROUP]
        raise
    meta.attrs[SEQUENCE_KEY] = SEQUENCE_SCHEMA_VERSION
