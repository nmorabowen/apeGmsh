"""History read after a deck run (ADR 0115 D3): ``ops.results(path)``.

``ops.py(path, run=True)`` and ``ops.tcl(path, run=True)`` solve the
model out of process, so the in-process domain holds nothing to read.
The deck run leaves a :class:`DeckRun` on the bridge (the deck path, the
working directory the subprocess inherited, the frozen
:class:`BuiltModel` it emitted, a digest of the declarations and whether
the run finished). :func:`open_results` turns it into the existing
:class:`~apeGmsh.results.Results`:

- one ``recorder.Ladruno`` (the fork's file) →
  :meth:`Results.from_ladruno`;
- one ``recorder.MPCO`` → converted to a ``.ladruno`` beside the
  ``.mpco`` and read with :meth:`Results.from_ladruno`, through
  :func:`~apeGmsh.results.live.read_deck_mpco` (ADR 0131 D11). It
  converts only when the stage kinds are facts: the program declares
  ``ops.stage`` blocks, or the unstaged deck was emitted with
  ``analyze_steps`` and left its run record ``<file>.run.json``, which
  must pass the checks in the module docstring of
  ``results/readers/_ladruno_from_mpco.py``. Otherwise the read refuses,
  naming those two remedies. The ``.mpco`` is never read with
  ``Results.from_mpco``;
- ``recorder.declare(nodes=...)`` and typed ``recorder.Node`` text
  records → one recorder spec synthesised from the records, read by
  :meth:`Results.from_recorders` (Q3: one history reader);
- otherwise a :class:`BridgeError` naming ``recorder.Ladruno``,
  ``recorder.MPCO`` under declared stages, ``recorder.declare`` and
  ``np.loadtxt``.

Every refusal is decided before the first file write, except the
converter's own: the run record's checks read the model archive, so they
come after it is written, and a refused conversion writes no
``.ladruno``. A refusal names its cause:

- no deck run on this bridge, or a live ``analyze`` since (``.read``
  answers there, Q4);
- ``path`` is not the deck the run wrote (Q5: the path is explicit and
  must match);
- the run did not finish (a non-zero exit or an interrupt), so its
  files may be partial;
- the bridge's declarations changed after the run;
- a record this route cannot read (a typed ``recorder.Element``, a
  ``recorder.declare`` element category or ``raw_*`` token, a
  ``Monitor`` recorder, an unknown recorder class): the route refuses the
  whole read rather than return a ``Results`` that silently lacks it;
- a history recorder (``MPCO`` or ``Ladruno``) with no ``file=``, more
  than one of them, or one beside text records;
- an ``MPCO`` recorder whose stage kinds are not facts (above), or whose
  ``.ladruno`` target beside it is a file the converter did not write
  (the fork's own, say), which the read never replaces;
- a text record on a staged or partitioned deck (per-stage and per-rank
  files are not one history);
- a typed record without a time column (``time_format="step"``), or a
  DOF whose meaning the bridge's DOF layout does not establish on a node;
- a missing, empty or malformed output file, or records whose time
  columns disagree.

What a successful read writes sits beside the deck, under
``<deck>.results/`` (the way ``<deck>.log`` sits beside it): the model
archive ``model.h5`` the readers bind, the text records transcribed to
the layout :meth:`Results.from_recorders` reads (``out/``), and that
reader's cache. A converted ``.mpco`` is the exception: its ``.ladruno``
is the converter's target, beside the ``.mpco``. A later read of the
same deck run (the same :class:`DeckRun` object, held in memory) reuses
that conversion while the ``.mpco``, its run record and the ``.ladruno``
are unchanged, even while an earlier ``Results`` holds it open. Any other
read converts afresh, replacing the file; an open ``Results`` on it then
refuses the replace by name.

This route imports the build package's names from their defining
submodules (``_internal/build/<module>.py``) here only.
"""
from __future__ import annotations

import os
import warnings
import weakref
from dataclasses import dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any, Sequence

import numpy as np

from ._internal.artifact_write import _declaration_digest
from ._internal.build.errors import BridgeError
from ._internal.build.ndf import (
    _load_dof_layout,
    infer_node_ndf,
    resolve_ndf_overlay,
)
from ._internal.build.ownership import is_partitioned
from ._internal.build.pg_expand import expand_pg_to_nodes
from ._internal.build.recorders import (
    _node_record_layout,
    _recorder_file_path,
    _resolve_node_targets,
)
from ._internal.tag_plan import emit_mode
from ._internal.types import Element, Recorder
from ._readback import ReadSession, RotationRoles, rotation_roles
from ._recorder_translate import (
    group_node_components_by_ops_token,
    node_component_to_ops,
)
from .recorder import MPCO, Ladruno, Monitor, RecorderDeclaration
from .recorder import Element as ElementRecorder
from .recorder import Node as NodeRecorder

if TYPE_CHECKING:
    from ..results.Results import Results
    from ._internal.build.pg_expand import ElementPlanRows
    from .apesees import BuiltModel, apeSees
    from .emitter.base import Emitter


__all__ = [
    "DeckRun",
    "RUN_COMPLETE_MARKER",
    "begin_deck_run",
    "log_proves_completion",
    "open_results",
    "write_tcl_driver",
]

_WHERE = "ops.results"

#: What to use instead, named by every "nothing this route can read"
#: refusal: routes that read (ADR 0131 D11), never one that refuses too.
_ALTERNATIVES = (
    "Record the history with ops.recorder.Ladruno(...) on the Ladruno fork "
    "build, with ops.recorder.MPCO(...) under declared ops.stage(...) "
    "blocks on stock OpenSees, or with ops.recorder.declare(nodes=...); or "
    "read a typed recorder's file yourself with np.loadtxt"
)

#: What makes an MPCO recorder's stage kinds facts, so its ``.mpco``
#: converts (ADR 0131 D5, D11). It never names ``convert_mpco``, which
#: refuses an unstaged program.
_MPCO_REMEDIES = (
    "Declare the stages with ops.stage(...), or emit and run the deck with "
    "analyze_steps= (ops.py(path, run=True, analyze_steps=n), or ops.tcl), "
    "which writes the run record; on the Ladruno fork build, record with "
    "ops.recorder.Ladruno(...) instead"
)

#: Typed ``recorder.Node`` response token -> (translational prefix,
#: rotational prefix) of the canonical component a DOF slot carries.
#: ``None``: the token has no canonical name for that kind of slot.
_TYPED_RESPONSES: dict[str, tuple[str, str | None]] = {
    "disp": ("displacement", "rotation"),
    "vel": ("velocity", "angular_velocity"),
    "accel": ("acceleration", "angular_acceleration"),
    "incrDisp": ("displacement_increment", None),
    "reaction": ("reaction_force", "reaction_moment"),
    "unbalance": ("force", "moment"),
}

#: The DOF counts ``_load_dof_layout`` names exactly.
_LAID_OUT_NDF: frozenset[int] = frozenset({1, 2, 3, 6})

_AXES = "xyz"

#: The reader's file convention (``results.spec._emit.emit_logical``): DOFs
#: 1-6 are ``(x, y, z, rx, ry, rz)`` whatever the model's ``(ndm, ndf)``.
#: The transcription writes each deck column at its component's slot here.
_READER_LAYOUT: tuple[int, ...] = _load_dof_layout(6, 3)

#: Canonical node-component prefix of ``recorder.declare`` -> whether it
#: names a rotational slot. Anything else (``pore_pressure``) is refused.
_DECLARED_PREFIXES: dict[str, bool] = {
    "displacement": False, "rotation": True,
    "velocity": False, "angular_velocity": True,
    "acceleration": False, "angular_acceleration": True,
    "displacement_increment": False,
    "reaction_force": False, "reaction_moment": True,
    "force": False, "moment": True,
}


# ---------------------------------------------------------------------------
# Proof that a Tcl deck ran to its end
# ---------------------------------------------------------------------------

#: Printed by the Tcl run driver after the deck returns. ``OpenSees.exe``
#: exits 0 after a Tcl ``error`` (the deck's aborted-analyze guard
#: included), so only this line proves the deck ran to its end.
RUN_COMPLETE_MARKER = "APEGMSH_RUN_COMPLETE"


def write_tcl_driver(deck: str) -> str:
    """Write ``<deck>.run.tcl``, which sources ``deck`` (its bytes
    unchanged) and then prints :data:`RUN_COMPLETE_MARKER`; return its
    path."""
    driver = os.path.abspath(deck) + ".run.tcl"
    target = Path(os.path.abspath(deck)).as_posix()
    with open(driver, "w", encoding="utf-8") as f:
        f.write(
            "# apeGmsh run driver (ADR 0115 D3): source the deck, then\n"
            "# prove it ran to its end.\n"
            f"source {{{target}}}\n"
            f'puts "{RUN_COMPLETE_MARKER}"\n'
        )
    return driver


def log_proves_completion(log_path: str) -> bool:
    """True iff the run's log holds the completion marker on a line of
    its own. No log is no proof."""
    if not os.path.isfile(log_path):
        return False
    with open(log_path, encoding="utf-8", errors="replace") as f:
        return any(line.strip() == RUN_COMPLETE_MARKER for line in f)


# ---------------------------------------------------------------------------
# The run record
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DeckRun:
    """What a ``py``/``tcl(run=True)`` leaves for :func:`open_results`."""

    verb: str
    deck: str                      # normcased absolute deck path
    deck_as_given: str
    cwd: Path                      # the subprocess's working directory
    built: "BuiltModel"
    supports_partitions: bool      # the deck emitter's capability
    partitioned: bool
    digest: str | None             # None: the declarations would not digest
    finished: bool = False

    def finish(self) -> "DeckRun":
        """The same run, marked as having exited cleanly."""
        return replace(self, finished=True)


def _norm(path: str | os.PathLike[str]) -> str:
    return os.path.normcase(os.path.abspath(os.fspath(path)))


def _digest(bridge: "apeSees") -> str | None:
    try:
        return _declaration_digest(bridge, _WHERE)
    except Exception:  # an unpicklable declaration: refused at read time
        return None


def begin_deck_run(
    bridge: "apeSees", built: "BuiltModel", emitter: "Emitter", path: str,
    *, verb: str,
) -> DeckRun:
    """The record of a deck run that is about to start (not finished)."""
    caps = bool(emitter.caps.supports_partitions)
    return DeckRun(
        verb=verb,
        deck=_norm(path),
        deck_as_given=str(path),
        cwd=Path(os.getcwd()),
        built=built,
        supports_partitions=caps,
        partitioned=bool(is_partitioned(built.fem) and caps),
        digest=_digest(bridge),
    )


# ---------------------------------------------------------------------------
# The read
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _TextSource:
    """One text ``.out`` file a deck run wrote, and what its columns are."""

    origin: str                        # the record, for messages
    path: Path
    node_ids: tuple[int, ...]
    components: tuple[str, ...]        # canonical name of each DOF column


def open_results(bridge: "apeSees", path: str) -> "Results":
    """``bridge``'s last deck run at ``path``, as a :class:`Results`."""
    run = _the_run(bridge, path)
    history, sources = _classify(bridge, run)
    if isinstance(history, Ladruno):
        return _write_and_read_ladruno(
            bridge, run, _history_files(run, history))
    if isinstance(history, MPCO):
        files = _history_files(run, history)
        overwrite = _plan_conversion(run, history, files)
        return _write_and_read_mpco(bridge, run, history, files, overwrite)
    tables = [_read_source(s) for s in sources]
    _check_time_columns(sources, tables)
    return _write_and_read_text(bridge, run, sources, tables)


def _the_run(bridge: "apeSees", path: str) -> DeckRun:
    if isinstance(bridge._read_state, ReadSession):
        raise BridgeError(
            f"{_WHERE}: the last solve on this bridge was a live "
            "ops.analyze(...); read it with .read (ops.nodes.get|one(...)"
            ".read, ops.elements.get|one(...).read). ops.results reads a "
            "deck run (ops.py / ops.tcl with run=True); on the live route "
            "it is deferred (ADR 0115 Q4)."
        )
    run = bridge._deck_run
    if run is None:
        raise BridgeError(
            f"{_WHERE}({path!r}): no deck run happened on this bridge. Run "
            "the deck first with ops.py(path, run=True) or "
            "ops.tcl(path, run=True), then pass the same path."
        )
    if _norm(path) != run.deck:
        raise BridgeError(
            f"{_WHERE}({path!r}): this bridge's last deck run wrote "
            f"{run.deck_as_given!r} ({run.verb}); {path!r} is not that deck. "
            "ops.results reads only the deck this bridge ran."
        )
    if not run.finished:
        raise BridgeError(
            f"{_WHERE}({path!r}): the deck run ({run.verb}) did not finish "
            "(a non-zero exit, an interrupt, or a Tcl deck that stopped "
            f"before its end: no {RUN_COMPLETE_MARKER} line in its log), so "
            "its output files may be partial. Fix the run and run the deck "
            "again."
        )
    if run.digest is None:
        raise BridgeError(
            f"{_WHERE}({path!r}): the bridge's declarations could not be "
            "digested at the deck run, so the read cannot prove they are "
            "the ones the deck ran."
        )
    if _digest(bridge) != run.digest:
        raise BridgeError(
            f"{_WHERE}({path!r}): the bridge's declarations changed after "
            f"the deck run ({run.verb}); the deck's files describe the model "
            "as it was. Run the deck again before reading."
        )
    return run


def _classify(
    bridge: "apeSees", run: DeckRun,
) -> "tuple[MPCO | Ladruno | None, list[_TextSource]]":
    """The one history recorder (MPCO or Ladruno), or the text sources;
    every record this route cannot read refuses here, all of them named
    at once."""
    recorders = [p for p in run.built.primitives if isinstance(p, Recorder)]
    if not recorders:
        raise BridgeError(
            f"{_WHERE}: the deck run ({run.verb}) declared no recorder, so it "
            f"wrote no history to read. {_ALTERNATIVES}."
        )
    histories: list[MPCO | Ladruno] = []
    typed: list[NodeRecorder] = []
    declared: list[RecorderDeclaration] = []
    unreadable: list[str] = []
    for rec in recorders:
        if isinstance(rec, (MPCO, Ladruno)):
            histories.append(rec)
        elif isinstance(rec, NodeRecorder):
            typed.append(rec)
        elif isinstance(rec, RecorderDeclaration):
            declared.append(rec)
        else:
            unreadable.append(_unreadable(rec))
    for decl in declared:
        for record in decl.records:
            if record.category != "nodes":
                unreadable.append(
                    f"recorder.declare(name={decl.name!r}) {record.category}= "
                    "record (element-level histories are read from "
                    "recorder.Ladruno, or from recorder.MPCO under declared "
                    "stages)"
                )
            elif record.raw:
                unreadable.append(
                    f"recorder.declare(name={decl.name!r}) raw_nodes="
                    f"{list(record.raw)!r} (a raw token has no canonical "
                    "component)"
                )
    if unreadable:
        raise BridgeError(
            f"{_WHERE}: the deck run ({run.verb}) holds records this route "
            "cannot read, and it does not return a partial Results: "
            + "; ".join(unreadable) + f". {_ALTERNATIVES}."
        )
    text = bool(typed or declared)
    if histories and text:
        kind = type(histories[0]).__name__
        raise BridgeError(
            f"{_WHERE}: the deck run ({run.verb}) holds "
            f"{'an' if kind == 'MPCO' else 'a'} {kind} recorder and "
            f"{len(typed) + len(declared)} text recorder(s); one Results "
            f"reads one of the two. Keep the {kind} recorder, or the text "
            "ones."
        )
    if len(histories) > 1:
        raise BridgeError(
            f"{_WHERE}: the deck run ({run.verb}) holds {len(histories)} "
            "history recorders ("
            + ", ".join(_history_origin(h) for h in histories)
            + "); one Results reads one file. Declare one MPCO or Ladruno "
            "recorder."
        )
    if histories:
        history = histories[0]
        if not str(history.file).strip():
            raise BridgeError(
                f"{_WHERE}: {_history_origin(history)} names no file, so "
                "ops.results cannot find the history the run wrote. "
                "Declare it with file=<path>."
            )
        return history, []
    if run.built.stage_records:
        raise BridgeError(
            f"{_WHERE}: the deck run ({run.verb}) is staged "
            f"({len(run.built.stage_records)} stage(s)); its text recorders "
            "write one file across every stage, which is not one history "
            "per stage. Record a staged run with ops.recorder.Ladruno on "
            "the Ladruno fork build, or with ops.recorder.MPCO on stock "
            "OpenSees (its stages are declared, so its .mpco converts)."
        )
    if run.partitioned:
        raise BridgeError(
            f"{_WHERE}: the deck run ({run.verb}) is partitioned; its text "
            "recorders are written per rank. Record a partitioned run with "
            "ops.recorder.Ladruno on the Ladruno fork build, or with "
            "ops.recorder.MPCO under declared ops.stage(...) blocks on "
            "stock OpenSees, or emit it with flat=True."
        )
    roles = _NodeRoles(bridge, run)
    sources: list[_TextSource] = []
    for node_rec in typed:
        sources.append(_typed_source(run, roles, node_rec))
    for decl in declared:
        sources.extend(_declared_sources(run, roles, decl))
    return None, sources


def _unreadable(rec: Recorder) -> str:
    if isinstance(rec, ElementRecorder):
        return (
            f"typed recorder.Element(file={rec.file!r}) (element records "
            "are read from recorder.Ladruno, from recorder.MPCO under "
            "declared stages, or with np.loadtxt)"
        )
    if isinstance(rec, Monitor):
        return "recorder.Monitor (not a history ops.results reads)"
    return f"a {type(rec).__name__} recorder (unknown to ops.results)"


def _history_origin(rec: "MPCO | Ladruno") -> str:
    return f"recorder.{type(rec).__name__}(file={rec.file!r})"


class _NodeRoles:
    """The emitted ndf of each node and the element families on it."""

    def __init__(self, bridge: "apeSees", run: DeckRun) -> None:
        built = run.built
        elements = [p for p in built.primitives if isinstance(p, Element)]
        inferred = infer_node_ndf(built.fem, elements, built.ndm)
        overlay = resolve_ndf_overlay(
            built.fem, built.ndf_records, inferred, built.ndm)
        self.ndm = int(built.ndm)
        self.ndf = {int(k): int(v) for k, v in {**inferred, **overlay}.items()}
        self.envelope = int(built.ndf)
        self._fem_nodes = frozenset(int(n) for n in built.fem.nodes.ids)
        self._run = run
        self._rot: "RotationRoles | None" = None

    def node_ndf(self, tag: int, origin: str) -> int:
        """The ndf the deck gave node ``tag``: inferred, ``ops.ndf``, or
        (a node only adaptive elements touch, or none) the ``ops.model``
        envelope the emit writes it at."""
        if tag in self.ndf:
            ndf = self.ndf[tag]
        elif tag in self._fem_nodes:
            ndf = self.envelope
        else:
            raise BridgeError(
                f"{_WHERE}: {origin} records node {tag}, which is not a node "
                "of the model the deck ran."
            )
        if ndf not in _LAID_OUT_NDF:
            raise BridgeError(
                f"{_WHERE}: {origin} records node {tag} with ndf={ndf}; the "
                "bridge's DOF layout names components only for ndf in "
                f"{sorted(_LAID_OUT_NDF)}. {_ALTERNATIVES}."
            )
        return ndf

    def _element_plan(
        self,
    ) -> "tuple[tuple[Element, ElementPlanRows], ...]":
        """The element plan the deck emit read: memoised by ``bm.emit``
        for the run's emit mode (a text read reaches here only on an
        unpartitioned run), so this mints nothing."""
        built = self._run.built
        plan = built._tag_plan(emit_mode(
            built, split=False,
            supports_partitions=self._run.supports_partitions))
        return tuple(plan.elements.specs)

    def rotation_refusal(self, tag: int) -> str | None:
        """Why node ``tag``'s rotational DOF slots are not rotations, or
        ``None`` (:meth:`RotationRoles.refusal`: the table ``.read`` and
        the ``recorder.declare`` emit use)."""
        if self._rot is None:
            self._rot = rotation_roles(self._element_plan())
        return self._rot.refusal(int(tag))

    def component(
        self, tag: int, dof: int, prefixes: tuple[str, str | None],
        origin: str,
    ) -> str:
        """The canonical component DOF ``dof`` of node ``tag`` carries."""
        ndf = self.node_ndf(tag, origin)
        if not 1 <= dof <= ndf:
            raise BridgeError(
                f"{_WHERE}: {origin} records DOF {dof} of node {tag}, which "
                f"has ndf={ndf}."
            )
        spatial = _load_dof_layout(ndf, self.ndm)[dof - 1]
        trans, rot = prefixes
        if spatial < 3:
            return f"{trans}_{_AXES[spatial]}"
        why = self.rotation_refusal(tag)
        if rot is None or why is not None:
            reason = (
                f"{trans!r} has no rotational component" if rot is None
                else f"DOF {dof} is a rotation only where the node's "
                f"elements establish one, and {why}"
            )
            raise BridgeError(
                f"{_WHERE}: {origin} records DOF {dof} of node {tag} "
                f"(ndf={ndf}, ndm={self.ndm}), which has no established "
                f"component: {reason}. {_ALTERNATIVES}."
            )
        return f"{rot}_{_AXES[spatial - 3]}"


def _resolve_file(run: DeckRun, file: str) -> Path:
    p = Path(file)
    return p if p.is_absolute() else run.cwd / p


def _typed_source(
    run: DeckRun, roles: _NodeRoles, rec: NodeRecorder,
) -> _TextSource:
    origin = f"recorder.Node(file={rec.file!r})"
    if rec.time_format != "dt":
        raise BridgeError(
            f"{_WHERE}: {origin} writes no time column "
            f"(time_format={rec.time_format!r}), so its rows have no time "
            "to read. Declare it with time_format=\"dt\"."
        )
    prefixes = _TYPED_RESPONSES.get(rec.response)
    if prefixes is None:
        raise BridgeError(
            f"{_WHERE}: {origin} records response {rec.response!r}, which "
            f"has no canonical component; ops.results reads "
            f"{', '.join(sorted(_TYPED_RESPONSES))}. {_ALTERNATIVES}."
        )
    if rec.nodes is not None:
        nodes = tuple(int(n) for n in rec.nodes)
    else:
        assert rec.pg is not None   # Node.__post_init__: exactly one
        nodes = tuple(int(n) for n in expand_pg_to_nodes(run.built.fem, rec.pg))
    if not nodes:
        raise BridgeError(f"{_WHERE}: {origin} records no node.")
    components: list[str] = []
    for dof in rec.dofs:
        names = {roles.component(n, int(dof), prefixes, origin) for n in nodes}
        if len(names) != 1:
            raise BridgeError(
                f"{_WHERE}: {origin}: DOF {dof} carries {sorted(names)} on "
                "different nodes of one record; one column must be one "
                "component."
            )
        components.append(names.pop())
    return _TextSource(
        origin=origin, path=_resolve_file(run, rec.file), node_ids=nodes,
        components=tuple(components),
    )


def _declared_sources(
    run: DeckRun, roles: _NodeRoles, decl: RecorderDeclaration,
) -> list[_TextSource]:
    """One source per ``recorder Node`` line the declaration emitted
    (the build's own file naming and DOF grouping, re-derived)."""
    out: list[_TextSource] = []
    for record in decl.records:
        origin = f"recorder.declare(name={decl.name!r})"
        nodes = tuple(int(n) for n in _resolve_node_targets(
            record, run.built.fem))
        if not nodes:
            raise BridgeError(
                f"{_WHERE}: {origin} nodes= record selects no node, so the "
                "deck emitted no recorder for it."
            )
        for n in nodes:
            roles.node_ndf(n, origin)
        for comp in record.components:
            prefix, _, axis = comp.rpartition("_")
            if prefix not in _DECLARED_PREFIXES or axis not in _AXES:
                raise BridgeError(
                    f"{_WHERE}: {origin} records {comp!r}, which the "
                    "node DOF layout does not name. "
                    f"{_ALTERNATIVES}."
                )
        # The deck's own DOF mapping, re-derived by the emit's function
        # over this run's per-node ndf (inferred plus ops.ndf): it raises,
        # naming the node, wherever a column is not that component.
        try:
            layout = _node_record_layout(
                record, decl, nodes, roles._element_plan(),
                node_ndf=roles.ndf)
        except BridgeError as e:
            raise BridgeError(f"{_WHERE}: {e} {_ALTERNATIVES}.") from e
        grouped = group_node_components_by_ops_token(
            record.components, layout)
        for token, dofs in grouped.items():
            columns = [
                next(c for c in record.components
                     if node_component_to_ops(c, layout) == (token, dof))
                for dof in dofs
            ]
            file = _recorder_file_path(
                decl.file_root, decl.name, record.name or "default", token)
            out.append(_TextSource(
                origin=origin, path=_resolve_file(run, file),
                node_ids=nodes, components=tuple(columns),
            ))
    return out


def _read_source(src: _TextSource) -> np.ndarray:
    """The source's rows: time, then node-major DOF columns."""
    if not src.path.is_file():
        raise BridgeError(
            f"{_WHERE}: {src.origin}'s output file {str(src.path)!r} does "
            "not exist; the run wrote no history there."
        )
    if src.path.stat().st_size == 0:
        raise BridgeError(
            f"{_WHERE}: {src.origin}'s output file {str(src.path)!r} is "
            "empty; the run recorded no step."
        )
    try:
        table = np.loadtxt(src.path, dtype=np.float64, ndmin=2)
    except ValueError as e:
        raise BridgeError(
            f"{_WHERE}: {src.origin}'s output file {str(src.path)!r} is "
            f"malformed (a partial write?): {e}"
        ) from e
    expected = 1 + len(src.node_ids) * len(src.components)
    if table.shape[1] != expected:
        raise BridgeError(
            f"{_WHERE}: {src.origin}'s output file {str(src.path)!r} has "
            f"{table.shape[1]} columns; the record writes {expected} (time, "
            f"then {len(src.node_ids)} node(s) x {len(src.components)} "
            "DOF(s))."
        )
    return table


def _check_time_columns(
    sources: Sequence[_TextSource], tables: Sequence[np.ndarray],
) -> None:
    first = tables[0][:, 0]
    for src, table in zip(sources[1:], tables[1:]):
        if not np.array_equal(table[:, 0], first):
            raise BridgeError(
                f"{_WHERE}: {src.origin} recorded {table.shape[0]} rows at "
                f"other times than {sources[0].origin} ({first.size} rows); "
                "one Results holds one time axis. Give the records one "
                "cadence."
            )


def _history_files(run: DeckRun, rec: "MPCO | Ladruno") -> list[Path]:
    """The file(s) a history recorder wrote: one, or one per rank."""
    origin = _history_origin(rec)
    path = _resolve_file(run, rec.file)
    if run.partitioned:
        # Exactly one file per rank of this run: a missing rank is a
        # partial run, an extra one is stale (another run's).
        n_ranks = len(run.built.fem.partitions)
        files = [path.with_name(f"{path.stem}.part-{k}{path.suffix}")
                 for k in range(n_ranks)]
        found = set(path.parent.glob(f"{path.stem}.part-*{path.suffix}"))
        stale = sorted(str(p.name) for p in found - set(files))
        if stale:
            raise BridgeError(
                f"{_WHERE}: {origin}: the run has "
                f"{n_ranks} rank(s), but {str(path.parent)!r} also holds "
                f"{stale}, which this run did not write. Remove them and "
                "run the deck again."
            )
    else:
        files = [path]
    for f in files:
        if not f.is_file() or f.stat().st_size == 0:
            hint = ""
            if (isinstance(rec, MPCO) and not run.built.stage_records
                    and not _run_record(run, rec).is_file()):
                # A deck emitted without analyze_steps drives no analysis,
                # so it records nothing; name what makes an MPCO read.
                hint = (
                    " The deck run declares no ops.stage block and left no "
                    f"run record {str(_run_record(run, rec))!r}. "
                    f"{_MPCO_REMEDIES}."
                )
            raise BridgeError(
                f"{_WHERE}: {origin}'s output "
                f"{str(f)!r} does not exist or is empty; the run wrote no "
                f"history there.{hint}"
            )
    return files


def _run_record(run: DeckRun, rec: MPCO) -> Path:
    """``<file>.run.json``: where a deck emitted with ``analyze_steps``
    writes its run record (for a set, beside ``<stem>.mpco``)."""
    from .emitter.py import DECK_RUN_SUFFIX

    return Path(str(_resolve_file(run, rec.file)) + DECK_RUN_SUFFIX)


def _plan_conversion(run: DeckRun, rec: MPCO, files: list[Path]) -> bool:
    """Decide, before any write, whether ``rec``'s ``.mpco`` converts
    (ADR 0131 D11); return whether its target replaces an earlier read's.

    The stage kinds are facts when the program declares ``ops.stage``
    blocks, or, unstaged, when the deck left its run record beside the
    ``.mpco`` (``read_deck_mpco`` then runs the record's checks). No
    record refuses here, naming the remedies. A ``.ladruno`` target that
    the converter did not write (the fork's own file) is never replaced.
    """
    from ..results.readers._ladruno_live import (
        existing_fork_files,
        fork_target_of,
    )

    origin = _history_origin(rec)
    if not run.built.stage_records:
        record = _run_record(run, rec)
        if not record.is_file():
            raise BridgeError(
                f"{_WHERE}: {origin}: the deck run ({run.verb}) declares no "
                f"ops.stage block and left no run record {str(record)!r}, so "
                "the kind of its one stage is not a fact and its .mpco is "
                f"not converted (ADR 0131 D11). {_MPCO_REMEDIES}."
            )
    existing = existing_fork_files(fork_target_of(files[0]))
    foreign = [str(p) for p in existing if not _converted(p)]
    if foreign:
        raise BridgeError(
            f"{_WHERE}: {origin}: its .mpco converts to {foreign}, which "
            "already exist and were not written by the converter (a "
            "Ladruno recorder's file, say); the read never replaces them. "
            "Move them aside, or record to another file."
        )
    return bool(existing)


def _converted(path: Path) -> bool:
    """``path`` is a ``.ladruno`` the MPCO converter wrote: its
    ``INFO/PROVENANCE@writer`` carries the converter's mark. Anything
    unreadable is not."""
    import h5py

    from ..results.readers._ladruno_provenance import _CONVERTED_MARK

    try:
        with h5py.File(path, "r") as f:
            writer = f["INFO/PROVENANCE"].attrs.get("writer")
    except (OSError, KeyError):
        return False
    if isinstance(writer, bytes):
        writer = writer.decode("utf-8", errors="replace")
    return isinstance(writer, str) and _CONVERTED_MARK in writer


# ---------------------------------------------------------------------------
# Writes: the only part that touches the disk
# ---------------------------------------------------------------------------


def _results_dir(bridge: "apeSees", run: DeckRun) -> tuple[Path, Path]:
    """``<deck>.results/`` and the model archive written in it."""
    root = Path(os.path.abspath(run.deck_as_given) + ".results")
    root.mkdir(exist_ok=True)
    archive = root / "model.h5"
    # The archive of the model the deck ran, from the frozen BuiltModel:
    # a read builds nothing (Invariant 2).
    bridge._write_h5(str(archive), run.built)
    return root, archive


def _write_and_read_ladruno(
    bridge: "apeSees", run: DeckRun, files: list[Path],
) -> "Results":
    from ..results.Results import Results

    _root, archive = _results_dir(bridge, run)
    target: "Path | list[str | Path]" = (
        files[0] if len(files) == 1 else list[str | Path](files))
    return Results.from_ladruno(target, fem=bridge.fem, model_h5=archive)


def _write_and_read_mpco(
    bridge: "apeSees", run: DeckRun, rec: MPCO, files: list[Path],
    overwrite: bool,
) -> "Results":
    """Convert the ``.mpco`` beside itself and read the ``.ladruno``,
    through ``read_deck_mpco`` (the one caller of the converter's
    run-record path). A conversion refusal raises here, by name: the
    read never falls back to ``Results.from_mpco``.

    A conversion an earlier read of this same deck run made is reused
    while its files are unchanged (:func:`_reusable_conversion`), so a
    second read never replaces a file an open :class:`Results` holds."""
    from ..results import MPCOConversionRefused
    from ..results.Results import Results
    from ..results.live import DefaultRouteWarning, read_deck_mpco
    from ..results.readers._ladruno_from_mpco import StageKindRefused

    _root, archive = _results_dir(bridge, run)
    origin = _history_origin(rec)
    reuse, why_not = _reusable_conversion(run, rec, files)
    if reuse is not None:
        target: "str | list[str | Path]" = (
            str(reuse[0]) if len(reuse) == 1 else list[str | Path](reuse))
        return Results.from_ladruno(target, model_h5=str(archive))
    try:
        with warnings.catch_warnings():
            # read_deck_mpco warns this just before it would read the
            # .mpco with Results.from_mpco: as an error, it never does.
            warnings.simplefilter("error", DefaultRouteWarning)
            results = read_deck_mpco(files[0], model_h5=archive,
                                     overwrite=overwrite)
    except (StageKindRefused, DefaultRouteWarning) as e:
        why = str(e).partition("Results.from_mpco: ")[2] or str(e)
        raise BridgeError(
            f"{_WHERE}: {origin}: the stage kinds of the deck run "
            f"({run.verb}) are not facts, so its .mpco is not converted "
            f"(ADR 0131 D11): {why} {_MPCO_REMEDIES}."
        ) from e
    except MPCOConversionRefused as e:
        raise BridgeError(
            f"{_WHERE}: {origin}: its .mpco does not convert: {e}"
        ) from e
    except OSError as e:
        raise BridgeError(
            f"{_WHERE}: {origin}: converting its .mpco to a .ladruno failed "
            f"({type(e).__name__}: {e}). This read converts afresh, "
            f"replacing the .ladruno already there, because {why_not}; a "
            "Results still open on that file (an earlier ops.results or "
            "Results.from_ladruno) blocks the replace. Close it "
            "(results.close()) and read again."
        ) from e
    _remember_conversion(run, rec, files)
    return results


#: The last conversion this route made for a deck run, keyed by the
#: :class:`DeckRun` object's id (its weak reference proves it is the same
#: object): the facts of the files it read and wrote. In memory only, so
#: nothing on disk can claim a conversion pairs; a new deck run on the
#: bridge is a new object and converts again.
_CONVERSIONS: "dict[int, tuple[weakref.ref[DeckRun], tuple[Any, ...]]]" = {}


def _file_facts(paths: Sequence[Path]) -> tuple[tuple[str, int, int], ...]:
    out = []
    for p in paths:
        st = p.stat()
        out.append((os.path.normcase(str(p)), st.st_size, st.st_mtime_ns))
    return tuple(out)


def _conversion_facts(
    run: DeckRun, rec: MPCO, files: Sequence[Path],
) -> "tuple[Any, ...] | None":
    """The size and mtime of the ``.mpco`` files, the run record and the
    ``.ladruno`` files of ``rec``, or ``None`` when one cannot be read."""
    from ..results.readers._ladruno_live import (
        existing_fork_files,
        fork_target_of,
    )

    record = _run_record(run, rec)
    try:
        targets = existing_fork_files(fork_target_of(files[0]))
        return (
            _file_facts(files),
            _file_facts([record]) if record.is_file() else (),
            _file_facts(targets),
        )
    except OSError:
        return None


def _reusable_conversion(
    run: DeckRun, rec: MPCO, files: Sequence[Path],
) -> "tuple[list[Path] | None, str]":
    """The ``.ladruno`` file(s) an earlier read of this same deck run
    converted, or ``None`` and why not.

    Reused only when this :class:`DeckRun` object made the conversion and
    the ``.mpco`` files, the run record and the ``.ladruno`` files are as
    that conversion left them (size and mtime): the record's checks then
    passed on exactly these files.
    """
    for key in [k for k, (ref, _) in _CONVERSIONS.items() if ref() is None]:
        del _CONVERSIONS[key]
    entry = _CONVERSIONS.get(id(run))
    if entry is None or entry[0]() is not run:
        return None, "no earlier read of this deck run converted it"
    now = _conversion_facts(run, rec, files)
    if now is None or now != entry[1] or not now[2]:
        return None, (
            "its .mpco, its run record or the .ladruno changed on disk "
            "since this deck run's earlier read converted it")
    return [Path(row[0]) for row in now[2]], ""


def _remember_conversion(
    run: DeckRun, rec: MPCO, files: Sequence[Path],
) -> None:
    facts = _conversion_facts(run, rec, files)
    if facts is None:
        _CONVERSIONS.pop(id(run), None)
    else:
        _CONVERSIONS[id(run)] = (weakref.ref(run), facts)


def _write_and_read_text(
    bridge: "apeSees", run: DeckRun, sources: Sequence[_TextSource],
    tables: Sequence[np.ndarray],
) -> "Results":
    from ..results.Results import Results
    from ..results.spec._emit import emit_logical
    from ..results.spec._resolved import (
        ResolvedRecorderRecord,
        ResolvedRecorderSpec,
    )
    from .opensees_model import OpenSeesModel

    root, archive = _results_dir(bridge, run)
    out_dir = root / "out"
    out_dir.mkdir(exist_ok=True)
    records: list[ResolvedRecorderRecord] = []
    for i, (src, table) in enumerate(zip(sources, tables)):
        comps = tuple(dict.fromkeys(src.components))
        rec = ResolvedRecorderRecord(
            category="nodes", name=f"r{i}", components=comps,
            dt=None, n_steps=None,
            node_ids=np.asarray(src.node_ids, dtype=np.int64),
        )
        logicals = emit_logical(rec, output_dir=str(out_dir))
        if len(logicals) != 1:
            raise BridgeError(
                f"{_WHERE}: {src.origin}: one file maps to "
                f"{len(logicals)} reader files; expected one."
            )
        logical = logicals[0]
        token = logical.response_tokens[0]
        # Each reader column (node, dof) is the source column of the
        # component the reader maps to that dof.
        by_dof: dict[int, int] = {}
        for k, comp in enumerate(src.components):
            pair = node_component_to_ops(comp, _READER_LAYOUT)
            if pair is None or pair[0] != token:
                raise BridgeError(
                    f"{_WHERE}: {src.origin}: component {comp!r} does not "
                    f"map to the reader's {token!r} file."
                )
            by_dof[pair[1]] = k
        assert logical.dofs is not None
        if set(by_dof) != set(logical.dofs):
            raise BridgeError(
                f"{_WHERE}: {src.origin}: the reader expects DOFs "
                f"{sorted(logical.dofs)}, the record maps to {sorted(by_dof)}."
            )
        k_src = len(src.components)
        cols = [0]
        for j in range(len(src.node_ids)):
            cols.extend(1 + j * k_src + by_dof[d] for d in logical.dofs)
        staged = Path(logical.file_path)
        np.savetxt(staged, table[:, cols], fmt="%.17g")
        st = src.path.stat()
        os.utime(staged, ns=(st.st_atime_ns, st.st_mtime_ns))
        records.append(rec)
    spec = ResolvedRecorderSpec(
        fem_snapshot_id=str(bridge.fem.snapshot_id), records=tuple(records),
    )
    model = OpenSeesModel.from_h5(archive)
    return Results.from_recorders(
        spec, out_dir, fem=bridge.fem, cache_root=root, model=model,
    )
