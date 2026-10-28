"""
PyEmitter — accumulates an openseespy Python deck as a list of strings.

Each Protocol method appends one ``ops.X(...)`` call to ``_lines``.
The Tcl block dialect (``\\{`` / ``\\}``) becomes openseespy's
"current section / current pattern" implicit state; patches, fibers,
layers, and loads following an open call attach to that state by
convention. ``section_close`` and ``pattern_close`` are no-ops in
the Python output (the next non-block-bodied call ends the implicit
scope).

The generated script imports openseespy and calls ``ops.wipe()`` once
at the top so re-running the script in a stateful Python session does
not pollute prior state.
"""
from __future__ import annotations

from .._internal.analyze_rc import COMMIT_ABORT_MESSAGE, COMMIT_ABORT_RC
from .._internal.build.errors import BridgeError
from .._internal.build.stage_emit import (
    ASDSHELL_PDELTA_GUARD,
    MEMBRANE_WP208_GUARD,
    Beam3dPartialLoadTargets,
    ForkBuildGuard,
    MaterialStageCarriers,
    fork_guard_lines,
    is_asdshell_p_delta,
    is_membrane_wp208,
)
from .._internal.build.stage_validate import StageZeroLedger
from .._internal.build.validators.solver import stage_marker_name
from .._rc_c2_flags import warn_rc_c2_deck
from .._target import BackendInfo

import io
import math
import os
import re
from typing import Any, Callable, Literal, Sequence

from .caps import TargetCaps
from .base import SUBDIVIDE_TOKEN
from .base import (
    NUMPY_VALUE_TYPES,
    DroppedAxisGuard,
    StrategySpec,
    command_row,
    plain_scalar,
)
# One stamp format for both decks; the Tcl module owns it.
from .tcl import DECK_BANNER, provenance_stamp


__all__ = [
    "DECK_RUN_SCHEMA", "DECK_RUN_SUFFIX", "PyEmitter", "deck_run_body",
    "deck_run_fields", "deck_run_lineage", "deck_run_unlink",
]

#: The deck run record (the ratified spec of the slice). A deck the
#: bridge emits with ``analyze_steps`` writes ``<recorder file>.run.json``
#: beside each MPCO recorder's output once its analyze loop has returned
#: without error, and never otherwise. ``convert_mpco``'s implicit-stage
#: path reads it. This module owns the format; the Tcl deck renders the
#: same fields from :func:`deck_run_fields`.
DECK_RUN_SCHEMA = "apegmsh.deck_run/2"
DECK_RUN_SUFFIX = ".run.json"


class _MemoryArchive(io.BytesIO):
    """An in-memory ``model.h5``: h5py writes it as a file object, and the
    composition's path helpers (the provenance base directory, the stem)
    read the name ``__fspath__`` gives. Nothing reaches the disk."""

    def __fspath__(self) -> str:
        return os.path.join(os.getcwd(), "deck_run_lineage.h5")


def _lineage_of(f: Any) -> "tuple[str | None, str | None]":
    from .._internal.lineage import read_stored_lineage

    fem_hash, model_hash, _ = read_stored_lineage(f["meta"])
    return model_hash, fem_hash


def _no_key_cause(bridge: Any) -> str:
    """Why ``bridge``'s declarations have no lineage key: a snapshot that
    is not a :class:`FEMData`, or the digest's own error, by type."""
    from ...mesh.FEMData import FEMData
    from .._internal.artifact_write import _declaration_digest

    if bridge is None or not isinstance(bridge.fem, FEMData):
        return "its snapshot is not a FEMData"
    try:
        _declaration_digest(bridge, bridge.fem.model_name or "")
    except Exception as exc:  # noqa: BLE001 - its type is the answer
        return f"a declaration cannot be digested ({type(exc).__name__})"
    return "a declaration cannot be digested"


def deck_run_lineage(
    archive: "Callable[[Any], None]", bridge: Any = None, *,
    compose: bool = True, deck: str = "",
) -> "tuple[str, str] | None":
    """``(model_hash, fem_hash)`` of the archive ``archive(path)`` writes.

    The record names the ``model.h5`` the deck was emitted from. Every
    archive the bridge composes (an explicit ``ops.h5(path)``, the
    automatic ``model.h5`` the deck emit writes first, an earlier deck's
    record) memoises its lineage by the digest of the declarations it
    holds (:func:`~apeGmsh.opensees._internal.artifact_write.lineage_key`),
    and the record reuses it: no archive is composed. Otherwise the
    bridge composes the archive (``apeSees._write_h5`` over the deck's own
    build) in memory, once per declaration set. No file is written and
    ``model_hash`` is unchanged. With ``compose=False`` (a partitioned
    deck: its archive is too costly to compose at every export, and P3
    never writes it automatically) nothing is composed: with no memoised
    lineage it warns once and gets no record. So does a model whose archive
    cannot be composed; the deck still emits, and ``read_deck_mpco`` of its
    ``.mpco`` refuses to convert (check 1). The warning names ``deck`` and
    points at the user's export call.
    """
    import warnings

    import h5py

    from .._internal._stacklevel import _stacklevel_outside_package
    from .._internal.artifact_write import _LINEAGE_BY_DIGEST, lineage_key
    from .h5 import H5FeatureDeferredWarning

    key = None if bridge is None else lineage_key(bridge)
    if key is not None and key in _LINEAGE_BY_DIGEST:
        return _LINEAGE_BY_DIGEST[key]
    why = ""
    try:
        if not compose:
            why = ("the model is partitioned and no lineage of these "
                   "declarations is memoised (an export composes no archive "
                   "for a partitioned model); " + (
                       "call ops.h5(path) before the export to record it"
                       if key is not None
                       else f"{_no_key_cause(bridge)}, so none can be"))
            found: "tuple[str | None, str | None]" = (None, None)
        else:
            buf = _MemoryArchive()
            with warnings.catch_warnings():
                # A deferred feature is the explicit save's warning.
                warnings.simplefilter("ignore", H5FeatureDeferredWarning)
                archive(buf)
            with h5py.File(buf, "r") as f:
                found = _lineage_of(f)
        model_hash, fem_hash = found
        if model_hash is None and not why:
            why = "the archive carries no /meta/lineage@model_hash"
    except Exception as exc:  # noqa: BLE001 - named in the warning below
        why = f"its model archive cannot be composed ({exc!r})"
    if why:
        warnings.warn(
            f"The deck {deck or '(unnamed)'} writes no run record: {why}. "
            "read_deck_mpco of its .mpco will read it with "
            "Results.from_mpco.",
            RuntimeWarning, stacklevel=_stacklevel_outside_package(),
        )
        return None
    assert model_hash is not None
    lineage = (model_hash, fem_hash or "")
    if key is not None:
        _LINEAGE_BY_DIGEST[key] = lineage
    return lineage


def deck_run_fields(
    lineage: "tuple[str, str]", recorder_file: str, analysis: str,
    steps: int, dt: "float | None",
) -> "dict[str, Any]":
    """The record's 14 fields, in the spec's order. The six the deck fills
    at run time (``recorder_size``, ``recorder_mtime``, ``steps_completed``,
    ``opensees_backend``, ``opensees_build_sha``, ``completed_iso``) are
    ``None`` here."""
    from apeGmsh import __version__

    return {
        "schema": DECK_RUN_SCHEMA,
        "model_hash": lineage[0],
        "fem_hash": lineage[1],
        "recorder_file": re.split(r"[\\/]", recorder_file)[-1],
        "recorder_size": None,
        "recorder_mtime": None,
        "analysis": analysis,
        "analyze_steps": int(steps),
        "steps_completed": None,
        "dt": None if dt is None else float(dt),
        "opensees_backend": None,
        "opensees_build_sha": None,
        "writer": f"apeGmsh deck {__version__}",
        "completed_iso": None,
    }


def _tcl_quote(text: str) -> str:
    """``text`` as one double-quoted Tcl word, with every character Tcl
    would substitute or count (``\\ [ ] $ { } "``) backslash-escaped."""
    return '"' + re.sub(r'([\\\[\]${}"])', r"\\\1", text) + '"'


def deck_run_unlink(dialect: 'Literal["py", "tcl"]', file: str) -> "list[str]":
    """The lines a record-writing deck runs just before it opens its MPCO
    recorder on ``file``: an earlier ``<file>.run.json`` is removed, so a
    run that then fails, and any later writer of the same ``.mpco``, never
    sits beside an old record."""
    final = file + DECK_RUN_SUFFIX
    if dialect == "py":
        return ["try:", f"    __import__('os').remove({final!r})",
                "except OSError:", "    pass"]
    return [f"file delete -force -- {_tcl_quote(final)}"]


#: The fields a Tcl deck fills at run time: the ``format`` conversion and
#: the variable that holds the value.
_TCL_RUN_TIME = {
    "recorder_size": ("%s", "$_apesees_size"),
    "recorder_mtime": ("%s", "$_apesees_mtime"),
    "steps_completed": ("%d", "$_apesees_i"),
    "opensees_backend": ('"%s"', "$_apesees_backend"),
    "opensees_build_sha": ('"%s"', "$_apesees_sha"),
    "completed_iso": ('"%s"', "$_apesees_iso"),
}


#: The record's final rename is retried while another process (Defender,
#: the indexer) holds the fresh file: waits from 50 ms, doubling, capped at
#: 1 s each, within a 30 s budget of waits; then the deck gives up on it.
_RENAME_FIRST_MS, _RENAME_CAP_MS, _RENAME_BUDGET_MS = 50, 1000, 30000
_NOT_WRITTEN = "apeGmsh: the deck run record was not written"


def _py_record(fields: "dict[str, Any]", file: str, done: str,
               partitioned: bool) -> "list[str]":
    """One file's record, in Python. Any ``OSError`` on the way (a file
    another process briefly holds refuses the rename, a full disk) prints
    one line and leaves no record: a finished run never fails here."""
    final = file + DECK_RUN_SUFFIX
    tmp = final + ".tmp"
    stem = file[: -len(".mpco")] if file.endswith(".mpco") else file
    parts = (f"['%s.part-%d.mpco' % ({stem!r}, _k) for _k in "
             "range(_apesees_np)]" if partitioned else f"[{file!r}]")
    one = [] if partitioned else [
        "        _apesees_size, _apesees_mtime = _apesees_size[0], "
        "_apesees_mtime[0]"]
    return [
        f"_apesees_parts = {parts}",
        "try:",
        "    if all(_apesees_os.path.isfile(_p) for _p in _apesees_parts):",
        "        _apesees_st = [_apesees_os.stat(_p) for _p in _apesees_parts]",
        "        _apesees_size = [_s.st_size for _s in _apesees_st]",
        "        _apesees_mtime = [_s.st_mtime for _s in _apesees_st]",
        *one,
        f"        _apesees_run = {fields!r}",
        "        _apesees_run.update(recorder_size=_apesees_size, "
        f"recorder_mtime=_apesees_mtime, steps_completed={done}, "
        "opensees_backend='stock' if _apesees_sha == 'unknown' "
        "else 'ladruno', opensees_build_sha=_apesees_sha, "
        "completed_iso=_apesees_time.strftime("
        "'%Y-%m-%dT%H:%M:%SZ', _apesees_time.gmtime()))",
        f"        with open({tmp!r}, 'w', encoding='utf-8') as _apesees_f:",
        "            _apesees_json.dump(_apesees_run, _apesees_f)",
        f"        _apesees_wait, _apesees_left = {_RENAME_FIRST_MS}, "
        f"{_RENAME_BUDGET_MS}",
        "        while True:",
        "            try:",
        f"                _apesees_os.replace({tmp!r}, {final!r})",
        "                break",
        "            except OSError:",
        "                if _apesees_left <= 0:",
        "                    raise",
        "                _apesees_time.sleep(_apesees_wait / 1000.0)",
        "                _apesees_left -= _apesees_wait",
        f"                _apesees_wait = min({_RENAME_CAP_MS}, 2 * _apesees_wait)",
        "except OSError as _apesees_e:",
        f"    print('{_NOT_WRITTEN}: %s' % _apesees_e, flush=True)",
    ]


def _tcl_record(fields: "dict[str, Any]", file: str,
                partitioned: bool) -> "list[str]":
    """One file's record, in Tcl, inside a ``catch``: any error on the way
    prints one line and leaves no record, never failing a finished run."""
    import json

    q = _tcl_quote
    final = file + DECK_RUN_SUFFIX
    tmp = final + ".tmp"
    template = "{" + ", ".join(
        f"{json.dumps(k)}: " + (_TCL_RUN_TIME[k][0] if k in _TCL_RUN_TIME
                                else json.dumps(v).replace("%", "%%"))
        for k, v in fields.items()) + "}"
    args = " ".join(_TCL_RUN_TIME[k][1] for k in fields if k in _TCL_RUN_TIME)
    write = [
        f"        set _apesees_f [open {q(tmp)} w]",
        "        fconfigure $_apesees_f -encoding utf-8",
        f"        puts -nonewline $_apesees_f [format {q(template)} {args}]",
        "        close $_apesees_f",
        f"        set _apesees_wait {_RENAME_FIRST_MS}; "
        f"set _apesees_left {_RENAME_BUDGET_MS}",
        f"        while {{[catch {{file rename -force {q(tmp)} {q(final)}}} "
        "_apesees_rerr]} {",
        "            if {$_apesees_left <= 0} { error $_apesees_rerr }",
        "            after $_apesees_wait",
        "            incr _apesees_left [expr {-$_apesees_wait}]",
        "            set _apesees_wait [expr {2 * $_apesees_wait > "
        f"{_RENAME_CAP_MS} ? {_RENAME_CAP_MS} : 2 * $_apesees_wait}}]",
        "        }",
        "    }",
    ]
    if not partitioned:
        body = [
            f"    if {{[file isfile {q(file)}]}} {{",
            f"        set _apesees_size [file size {q(file)}]",
            f"        set _apesees_mtime [file mtime {q(file)}]",
            *write,
        ]
    else:
        stem = file[: -len(".mpco")] if file.endswith(".mpco") else file
        body = [
            "    set _apesees_sizes {}; set _apesees_mtimes {}; "
            "set _apesees_ok 1",
            "    for {set _apesees_k 0} {$_apesees_k < $_apesees_np} "
            "{incr _apesees_k} {",
            f"        set _apesees_p [format {{%s.part-%d.mpco}} {q(stem)} "
            "$_apesees_k]",
            "        if {![file isfile $_apesees_p]} "
            "{ set _apesees_ok 0; break }",
            "        lappend _apesees_sizes [file size $_apesees_p]",
            "        lappend _apesees_mtimes [file mtime $_apesees_p]",
            "    }",
            "    if {$_apesees_ok} {",
            '        set _apesees_size "\\[[join $_apesees_sizes {, }]\\]"',
            '        set _apesees_mtime "\\[[join $_apesees_mtimes {, }]\\]"',
            *write,
        ]
    return ["if {[catch {", *body,
            f'}} _apesees_err]}} {{ puts "{_NOT_WRITTEN}: $_apesees_err" }}']


def deck_run_body(
    dialect: 'Literal["py", "tcl"]', archive: "Callable[[Any], None]",
    files: "Sequence[str]", analysis: "str | None", steps: int,
    dt: "float | None", *, partitioned: bool, bridge: Any = None,
    deck: str = "",
) -> "list[str]":
    """The deck lines that write the run record of each MPCO ``files``
    entry (``<file>.run.json``), in ``dialect``; ``[]`` with no file or no
    ``analysis`` token (nothing to describe).

    The lines close every recorder first (``remove recorders``), so each
    ``.mpco`` is final. That closes every recorder the deck opened at its
    end, the text ``.out`` recorders included, as process exit would,
    so nothing after the record writes to them. The record names each
    file's size and modification time as the deck's runtime reports them
    (a list, one per part in rank
    order, for a partitioned deck; no record when a file is missing). The
    engine's backend and build come from ``ladrunoBuild`` (the G10 rule: a
    40-hex answer is the fork, anything else is stock with an ``unknown``
    build); the steps completed from the loop's own counter."""
    if not files or analysis is None:
        return []
    lineage = deck_run_lineage(archive, bridge, compose=not partitioned,
                               deck=deck)
    if lineage is None:
        return []
    head = ["# apeGmsh deck run record: the analyze loop returned; the "
            "recorders close first, so each .mpco is final."]
    done = "_apesees_i + 1" if int(steps) > 0 else "0"
    if dialect == "py":
        head.append("ops.remove('recorders')")
        body = [
            "import json as _apesees_json, os as _apesees_os",
            "import re as _apesees_re, time as _apesees_time",
            "try:",
            "    _apesees_sha = str(ops.ladrunoBuild()).strip()",
            "except Exception:",
            "    _apesees_sha = ''",
            "if not _apesees_re.fullmatch('[0-9a-f]{40}', _apesees_sha):",
            "    _apesees_sha = 'unknown'",
        ]
        if partitioned:
            body += ["try:", "    _apesees_np = int(ops.getNP())",
                     "except Exception:", "    _apesees_np = 1"]
    else:
        head.append("remove recorders")
        body = [
            "if {[catch {ladrunoBuild} _apesees_sha] || ![regexp "
            "{^[0-9a-f]{40}$} [string trim $_apesees_sha]]} {",
            "    set _apesees_sha unknown; set _apesees_backend stock",
            "} else {",
            "    set _apesees_sha [string trim $_apesees_sha]; "
            "set _apesees_backend ladruno",
            "}",
            # UTC from the epoch in plain expr (days-from-civil inverted):
            # ``clock format`` needs Tcl's library scripts, which an
            # ``OpenSees.exe`` may not ship.
            "set _apesees_s [clock seconds]",
            "set _apesees_z [expr {$_apesees_s / 86400 + 719468}]",
            "set _apesees_e [expr {$_apesees_z / 146097}]",
            "set _apesees_z [expr {$_apesees_z - $_apesees_e * 146097}]",
            "set _apesees_y [expr {($_apesees_z - $_apesees_z / 1460 + "
            "$_apesees_z / 36524 - $_apesees_z / 146096) / 365}]",
            "set _apesees_z [expr {$_apesees_z - (365 * $_apesees_y + "
            "$_apesees_y / 4 - $_apesees_y / 100)}]",
            "set _apesees_m [expr {(5 * $_apesees_z + 2) / 153}]",
            "set _apesees_d [expr {$_apesees_z - (153 * $_apesees_m + 2) / 5 + 1}]",
            "set _apesees_m [expr {$_apesees_m < 10 ? $_apesees_m + 3 : "
            "$_apesees_m - 9}]",
            "set _apesees_y [expr {$_apesees_y + $_apesees_e * 400 + "
            "($_apesees_m <= 2)}]",
            "set _apesees_t [expr {$_apesees_s % 86400}]",
            "set _apesees_iso [format {%04d-%02d-%02dT%02d:%02d:%02dZ} "
            "$_apesees_y $_apesees_m $_apesees_d [expr {$_apesees_t / 3600}] "
            "[expr {$_apesees_t / 60 % 60}] [expr {$_apesees_t % 60}]]",
        ]
        if partitioned:
            body.append("if {[catch {getNP} _apesees_np]} { set _apesees_np 1 }")
    for file in files:
        fields = deck_run_fields(lineage, file, analysis, steps, dt)
        body += (_py_record(fields, file, done, partitioned) if dialect == "py"
                 else _tcl_record(fields, file, partitioned))
    if not partitioned:
        return head + body
    # Every rank closes its recorders and finishes its loop first; one
    # record, from rank 0.
    indented = ["    " + ln for ln in body]
    if dialect == "py":
        return [*head, "if hasattr(ops, 'barrier'):", "    ops.barrier()",
                "if getPID() == 0:", *indented]
    return [*head, 'if {[info commands barrier] ne ""} { barrier }',
            "if {[getPID] == 0} {", *indented, "}"]


class _LineBuf(list[str]):
    """Line buffer with an optional partition-indent prefix.

    Behaves as a plain ``list[str]`` for ``insert``, iteration, slicing,
    and the ``lines()`` copy-out. When ``indent`` is non-empty, every
    ``append(line)`` prepends ``indent`` to ``line`` before storing.
    This keeps the existing ``self._lines.append(...)`` call sites in
    :class:`PyEmitter` working unchanged while
    ``partition_open(K)`` / ``partition_close()`` toggle the indent.
    """

    __slots__ = ("indent",)

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.indent: str = ""

    def append(self, line: str) -> None:
        if self.indent:
            super().append(self.indent + line)
        else:
            super().append(line)


def _fmt_value(v: Any) -> str:
    """Render one positional argument for a Python source line.

    Strings are wrapped in single quotes (Python source). Booleans
    coerce to ``1`` / ``0`` (openseespy does not accept Python bools).
    Integers and floats use their ``repr`` (round-trip-safe). A numpy
    scalar or 0-d array renders as the plain Python number (see
    :func:`~apeGmsh.opensees.emitter.base.plain_scalar`, #1336), and a
    subclass of ``int`` / ``float`` renders as its base value, never
    through its own ``repr``.
    """
    if isinstance(v, bool):
        return "1" if v else "0"
    if isinstance(v, str):
        # Single-quote Python string literal; escape embedded single
        # quotes. OpenSees vocabulary doesn't typically include
        # quotes, but defensively handle the case.
        escaped = v.replace("\\", "\\\\").replace("'", "\\'")
        return f"'{escaped}'"
    if isinstance(v, NUMPY_VALUE_TYPES):
        return _fmt_value(plain_scalar(v))
    if isinstance(v, int):
        return str(int(v))
    if isinstance(v, float):
        f = float(v)
        if not math.isfinite(f):
            raise _NonFiniteArg(f)
        return repr(f)
    _refuse_nonfinite_items(v)
    return repr(v)


class _NonFiniteArg(BridgeError):
    """A ``nan`` / ``inf`` reached the formatter (#1356).

    :func:`_fmt_value` has no command context, so it raises this; the
    caller that knows the command re-raises a :class:`BridgeError` that
    names the command and the argument (:func:`_nonfinite_error`).
    """

    def __init__(self, value: float) -> None:
        super().__init__(f"non-finite float {value!r}")
        self.value = value


def _refuse_nonfinite_items(v: Any) -> None:
    """Raise :class:`_NonFiniteArg` for a ``nan`` / ``inf`` inside a
    list or tuple argument: the fallback renders it with ``repr``, which
    would print a bare ``nan`` token. Fallback path only."""
    if isinstance(v, (list, tuple)):
        for x in v:
            _fmt_value(x)


def _is_nonfinite_token(a: Any) -> bool:
    try:
        _fmt_value(a)
    except _NonFiniteArg:
        return True
    return False


def _nonfinite_error(command: str, args: Sequence[Any], value: float) -> BridgeError:
    """The emit-time refusal of a non-finite float (#1356).

    OpenSees cannot read ``nan`` / ``inf``: a py deck dies with
    ``NameError`` and Tcl's ``Tcl_GetDouble`` rejects the token. The
    message names the command and the offending argument (0-based
    position among the command's positional arguments). Error path
    only, so it re-formats the arguments to find the offender.
    """
    where = f"an argument is {value!r}"
    for i, a in enumerate(args):
        if _is_nonfinite_token(a):
            where = f"argument {i} is {a!r}"
            break
    return BridgeError(
        f"{command}: {where}; OpenSees cannot parse a non-finite float "
        "(nan/inf), so the deck would fail at run time. Fix the value at "
        "its source (#1356)."
    )


def _refuse_nonfinite_ramp(
    name: str, targets: tuple[tuple[int, float], ...], n_steps_to_full: float,
) -> None:
    """Refuse a non-finite ramp constant before any hook line is emitted.

    The ramp body interpolates ``n_steps_to_full`` and each target into
    an expression, not through the line formatter, so it is checked
    here (#1356).
    """
    bad: list[str] = []
    if not math.isfinite(float(n_steps_to_full)):
        bad.append(f"n_steps_to_full is {float(n_steps_to_full)!r}")
    bad += [
        f"target of parameter {int(tag)} is {float(t)!r}"
        for tag, t in targets if not math.isfinite(float(t))
    ]
    if bad:
        raise BridgeError(
            f"step hook ramp {name!r}: {'; '.join(bad)}; OpenSees cannot "
            "parse a non-finite float (nan/inf), so the deck would fail at "
            "run time. Fix the value at its source (#1356)."
        )


def _ops_call(method: str, *args: Any) -> str:
    """Render an ``ops.<method>(...)`` source line.

    The int / float dispatch is inlined (one function call per LINE
    instead of per token — measured flat-emit hotspot); strings and odd
    types still route through :func:`_fmt_value`, so output is
    identical.

    A non-finite float raises :class:`BridgeError` naming the command
    and the argument (#1356). The inlined float check is ``a - a``:
    ``0.0`` (falsy) for every finite float, ``nan`` (truthy) for
    ``nan`` and ``+-inf`` -- one float op, no attribute lookup or call.
    """
    out: list[str] = []
    append = out.append
    try:
        for a in args:
            c = a.__class__
            if c is int:
                append(str(a))
            elif c is float:
                if a - a:
                    raise _NonFiniteArg(a)
                append(repr(a))
            else:
                append(_fmt_value(a))
    except _NonFiniteArg as exc:
        head = f"ops.{method}({_fmt_value(args[0])}, ...)" if (
            args and isinstance(args[0], str)) else f"ops.{method}(...)"
        raise _nonfinite_error(head, args, exc.value) from None
    return f"ops.{method}({', '.join(out)})"


class PyEmitter:
    """Accumulates an openseespy Python deck.

    The output is a self-contained Python script: the user can write
    it to disk, ``python script.py`` it, and OpenSees will execute the
    deck. Top-level statements; no function wrapper.
    """

    #: ADR 0114 D6: a py deck consumes partition brackets; a
    #: ``model BasicBuilder`` re-issue purges nothing on openseespy.
    caps: TargetCaps = TargetCaps(
        archival=False,
        supports_partitions=True,
        per_rank_fragments=False,
        suppress_analysis_chain_auto_emit=False,
        model_reissue_purges=False,
        emit_stage_markers=False,
    )

    def __init__(self, *, backend: BackendInfo | None = None) -> None:
        # ``_lines`` is a :class:`_LineBuf` so ``partition_open`` can
        # toggle a per-rank indent without every ``append`` call site
        # needing to know. All existing primitives keep calling
        # ``self._lines.append(...)`` unchanged.
        # ``backend=None`` keeps the header to banner, import and wipe.
        header = [DECK_BANNER]
        if backend is not None:
            header.append(provenance_stamp(backend))
        header += ["import openseespy.opensees as ops", "ops.wipe()"]
        self._lines: _LineBuf = _LineBuf(header)
        # The elements each swept material is on, for its plastic flip.
        self._stage_carriers = MaterialStageCarriers()
        # The elements a 3-D partial -beamUniform may load on any build.
        self._beam3d_targets = Beam3dPartialLoadTargets()
        # The fork build floors this deck already checks, and
        # where their checks go: after the import, before ``ops.wipe()``.
        self._fork_floors: set[ForkBuildGuard] = set()
        self._fork_floor_at = len(header) - 1
        # Partition-emission state (ADR 0027 / P4). On the first
        # ``partition_open`` call across the emitter's lifetime, emit a
        # one-shot runtime shim that pulls ``getPID`` from openseespy
        # but falls back to a 0-returning stub so single-process
        # Python still runs the deck.
        self._partition_shim_emitted: bool = False
        # Step-hook state (Phase SSI-1).  Mirror of the TclEmitter
        # flags: ``_step_hooks_registered`` flips on the first
        # ``step_hook_ramp`` and is checked by ``analyze`` to wrap the
        # analyze loop with dispatcher calls.
        # ``_hook_dispatcher_emitted`` ensures the dispatcher
        # boilerplate (module-level lists + dispatcher functions) is
        # emitted exactly once.
        self._step_hooks_registered: bool = False
        self._hook_dispatcher_emitted: bool = False
        # Progress-marker injection (opt-in via apeSees.py(progress=)).
        # When True, ``analyze`` drops a throttled ``print(
        # APEGMSH_PROGRESS ...)`` in the loop so the run=True streamer
        # can render a live step counter. Default off keeps decks clean.
        self._emit_progress: bool = False
        # ADR 0106 D2 — stage-marker injection: ``BuiltModel.emit`` sets
        # ``caps.emit_stage_markers`` from ``deck_requests_solver_stats``,
        # and ``stage_open`` / ``stage_close`` then drop a runtime
        # ``print(APEGMSH_STAGE open|close <name>)`` so a solver-stats
        # block on stderr can be attributed to the stage that paid for
        # it. Tracks the name of the currently-open stage so
        # ``stage_close`` (which takes no argument) can name it too.
        self._current_stage_name: str | None = None
        # The deck run record: every MPCO recorder's file, in
        # emit order, and the last ``analysis`` token the deck emitted.
        self._mpco_files: list[str] = []
        self._analysis_token: str | None = None
        # Set by the bridge when the deck will write a record.
        self._run_record: bool = False
        # The ManzariDafalias ``-stage 0`` lines owed before a solve verb.
        self._stage0 = StageZeroLedger()

    # -- Output --------------------------------------------------------------

    def lines(self) -> list[str]:
        """Return a copy of the accumulated Python lines."""
        return list(self._lines)

    def line_count(self) -> int:
        """Number of accumulated lines, without copying the buffer.

        ``len(emitter.lines())`` clones the whole line list (multi-M
        entries on large models) just to read a length — span/module
        recording must use this instead (ADR 0065 A0).
        """
        return len(self._lines)

    def line_buffer(self) -> "list[str]":
        """The internal line buffer — READ-ONLY by contract.

        For O(1)-extra-memory consumers that only iterate or slice
        (the per-rank / split deck writers). Mutating the returned
        list corrupts the emitter; use :meth:`lines` for a defensive
        copy (ADR 0065 A0).
        """
        return self._lines

    def write_to(self, f: "Any") -> None:
        """Stream the accumulated deck to an open text handle.

        Writes the internal buffer line-by-line, so neither the
        ``list(self._lines)`` copy of :meth:`lines` nor a single joined
        deck-sized string is materialized — peak write-time memory drops
        to the OS write buffer (ADR 0065 Tier 1). Output is byte-identical
        to ``f.write("\\n".join(self.lines()) + "\\n")``.
        """
        write = f.write
        for line in self._lines:
            write(line)
            write("\n")

    # -- Model ---------------------------------------------------------------

    #: Model ``ndm``, learned from :meth:`model`; ``None`` until then.
    #: Node coordinates are trimmed to it — see ``trim_coords_to_ndm``.
    _model_ndm: "int | None" = None
    #: Trims nodes to ``_model_ndm`` and refuses a lossy trim (#1337).
    _dropped_axes: DroppedAxisGuard = DroppedAxisGuard.BEFORE_MODEL

    def model(self, *, ndm: int, ndf: int) -> None:
        # openseespy.model takes positional + flag-style args:
        # ops.model('basic', '-ndm', 3, '-ndf', 6).
        self._model_ndm = ndm
        self._dropped_axes = DroppedAxisGuard(ndm)
        self._lines.append(
            _ops_call("model", "basic", "-ndm", ndm, "-ndf", ndf)
        )

    def node(
        self, tag: int, *coords: float, ndf: int | None = None,
    ) -> None:
        # A padded coordinate would swallow the -ndf flag below (and
        # -mass) in a 2-D deck — see trim_coords_to_ndm.
        coords = self._dropped_axes.trim(coords, tag)
        # Fast path for the dominant deck band — mirrors the
        # TclEmitter's: plain-int tag + plain-float coords render via a
        # single f-string, byte-identical to the generic path.  The
        # 2-coord arm is the 2-D deck band (the build layer trims to
        # ``ndm`` coordinates).
        if ndf is None:
            if tag.__class__ is int:
                if len(coords) == 3:
                    x, y, z = coords
                    if (
                        x.__class__ is float
                        and y.__class__ is float
                        and z.__class__ is float
                        # nan/inf fall through to the general path,
                        # which refuses them naming ``node`` and the argument.
                        and not (x - x or y - y or z - z)
                    ):
                        self._lines.append(
                            f"ops.node({tag}, {x!r}, {y!r}, {z!r})"
                        )
                        return
                elif len(coords) == 2:
                    x, y = coords
                    if (x.__class__ is float and y.__class__ is float
                            and not (x - x or y - y)):
                        self._lines.append(f"ops.node({tag}, {x!r}, {y!r})")
                        return
            self._lines.append(_ops_call("node", tag, *coords))
        else:
            # Per-node ``-ndf`` override for mixed-ndf models (e.g. 6-DOF
            # phantom nodes in an ndf=3 solid mesh; ADR 0022 INV-3).
            self._lines.append(
                _ops_call("node", tag, *coords, "-ndf", ndf)
            )

    def fix(self, tag: int, *dofs: int) -> None:
        self._lines.append(_ops_call("fix", tag, *dofs))

    def mass(self, tag: int, *values: float) -> None:
        self._lines.append(_ops_call("mass", tag, *values))

    # -- MP constraints (ADR 0022, Phase 7b) -----------------------------

    def equalDOF(self, master: int, slave: int, *dofs: int) -> None:
        self._lines.append(_ops_call("equalDOF", master, slave, *dofs))

    def equalDOF_mixed(
        self, master: int, slave: int,
        dof_pairs: "Sequence[tuple[int, int]]",
    ) -> None:
        flat = [int(d) for pair in dof_pairs for d in pair]
        self._lines.append(
            _ops_call("equalDOF_Mixed", master, slave, len(dof_pairs), *flat)
        )

    def rigidLink(self, kind: str, master: int, slave: int) -> None:
        self._lines.append(_ops_call("rigidLink", kind, master, slave))

    def rigidDiaphragm(
        self, perp_dir: int, master: int, *slaves: int,
    ) -> None:
        self._lines.append(
            _ops_call("rigidDiaphragm", perp_dir, master, *slaves)
        )

    def embeddedNode(
        self, ele_tag: int, cnode: int, *master_nodes: int,
        stiffness: float = 1.0e18,
        stiffness_p: float | None = None,
        rotational: bool = False,
        pressure: bool = False,
    ) -> None:
        from .base import (
            _build_embedded_flag_args, _embedded_retained_args,
        )
        flag_args = _build_embedded_flag_args(
            stiffness, stiffness_p, rotational, pressure,
        )
        # The 4th retained node goes as a string: openseespy's parser
        # reads it with OPS_GetString and drops an int.
        self._lines.append(
            _ops_call(
                "element", "ASDEmbeddedNodeElement",
                ele_tag, cnode, *_embedded_retained_args(master_nodes),
                *flag_args,
            )
        )

    def embedded_rebar(
        self, ele_tag: int, *args: int | float | str,
    ) -> None:
        # LadrunoEmbeddedRebar coupling (g.reinforce, ADR 20). Args
        # pre-built by ``embedded_rebar_args``; fork-only at run time.
        self._lines.append(
            _ops_call("element", "LadrunoEmbeddedRebar", ele_tag, *args))

    def equationConstraint(
        self, cnode: int, cdof: int, ccoef: float,
        retained: "Sequence[tuple[int, int, float]]",
    ) -> None:
        # EQ_Constraint via openseespy (ADR 0068): one call per tied DOF,
        # flat varargs ops.equationConstraint(cNode, cDOF, cCoef, rn, rd,
        # rc, ...).
        flat: list[int | float] = []
        for rn, rd, rc in retained:
            flat += [int(rn), int(rd), float(rc)]
        self._lines.append(
            _ops_call("equationConstraint", int(cnode), int(cdof),
                      float(ccoef), *flat)
        )

    def embedded_node(
        self, ele_tag: int, *args: int | float | str,
    ) -> None:
        # LadrunoEmbeddedNode coupling (g.embed). Args pre-built by
        # ``embedded_node_args``; fork-only at run time.
        self._lines.append(
            _ops_call("element", "LadrunoEmbeddedNode", ele_tag, *args))

    def contact_surface(
        self, tag: int, *args: int | float | str,
    ) -> None:
        # Fork contact surface (g.constraints.contact); fork-only at run time.
        self._lines.append(_ops_call("contactSurface", tag, *args))

    def contact(
        self, tag: int, *args: int | float | str,
    ) -> None:
        # Fork contact verb; fork-only at run time.
        self._lines.append(_ops_call("contact", tag, *args))

    def contact_plane(
        self, tag: int, *args: int | float | str,
    ) -> None:
        # Fork rigid-plane contact (g.constraints.contact_plane); fork-only.
        self._lines.append(_ops_call("contactPlane", tag, *args))

    def mp_constraint_comment(self, name: str) -> None:
        # Hash-comment line, matching Tcl's ``# {name}`` convention.
        self._lines.append(f"# {name}")

    # -- Regions -------------------------------------------------------------

    def region(self, tag: int, *args: int | float | str) -> None:
        self._lines.append(_ops_call("region", tag, *args))

    # -- Damping (ADR 0053) --------------------------------------------------

    def rayleigh(
        self,
        alpha_m: float,
        beta_k: float,
        beta_k_init: float,
        beta_k_comm: float,
    ) -> None:
        self._lines.append(
            _ops_call("rayleigh", alpha_m, beta_k, beta_k_init, beta_k_comm),
        )

    def damping(
        self, damp_type: str, tag: int, *args: int | float | str,
    ) -> None:
        self._lines.append(_ops_call("damping", damp_type, tag, *args))

    def modal_damping(self, *factors: float) -> None:
        self._lines.append(_ops_call("modalDamping", *factors))

    # -- Constitutive --------------------------------------------------------

    def uniaxialMaterial(
        self, mat_type: str, tag: int, *params: float | str,
    ) -> None:
        self._lines.append(
            _ops_call("uniaxialMaterial", mat_type, tag, *params)
        )

    def nDMaterial(
        self, mat_type: str, tag: int, *params: float | str,
    ) -> None:
        warn_rc_c2_deck(mat_type, params)
        self._stage0.material(mat_type, tag)
        self._stage_carriers.note_material(mat_type, tag, params)
        self._lines.append(_ops_call("nDMaterial", mat_type, tag, *params))

    def section(
        self, sec_type: str, tag: int, *params: float | str,
    ) -> None:
        self._lines.append(_ops_call("section", sec_type, tag, *params))

    def geomTransf(
        self, t_type: str, tag: int, *vec: float,
        jnt_offset: tuple[float, ...] = (),
    ) -> None:
        jnt = ("-jntOffset", *jnt_offset) if jnt_offset else ()
        self._lines.append(_ops_call("geomTransf", t_type, tag, *vec, *jnt))

    # -- Sections that take blocks (Fiber) ----------------------------------

    def section_open(
        self, sec_type: str, tag: int, *params: float | str,
    ) -> None:
        # openseespy's stateful "current section" is set by the same
        # ops.section() call that defines the section's parameters.
        self._lines.append(_ops_call("section", sec_type, tag, *params))

    def section_close(self) -> None:
        # No-op — openseespy doesn't have an explicit close. The
        # current-section context ends implicitly when a non-patch /
        # non-fiber / non-layer command runs next.
        pass

    def patch(self, kind: str, *args: int | float) -> None:
        self._lines.append(_ops_call("patch", kind, *args))

    def fiber(
        self, y: float, z: float, area: float, mat_tag: int,
    ) -> None:
        self._lines.append(_ops_call("fiber", y, z, area, mat_tag))

    def layer(self, kind: str, *args: int | float) -> None:
        self._lines.append(_ops_call("layer", kind, *args))

    # -- Beam integration rules ---------------------------------------------

    def beamIntegration(
        self, rule_type: str, tag: int, *args: int | float | str,
    ) -> None:
        self._lines.append(_ops_call("beamIntegration", rule_type, tag, *args))

    # -- Topology ------------------------------------------------------------

    def element(
        self, ele_type: str, tag: int, *args: int | float | str,
    ) -> None:
        self._stage0.owed = True
        self._stage_carriers.note_element(ele_type, tag, args)
        self._beam3d_targets.note_element(ele_type, tag)
        if is_asdshell_p_delta(ele_type, args):
            self._require_fork_floors((ASDSHELL_PDELTA_GUARD,))
        if is_membrane_wp208(ele_type, args):
            self._require_fork_floors((MEMBRANE_WP208_GUARD,))
        self._lines.append(_ops_call("element", ele_type, tag, *args))

    # -- Time series --------------------------------------------------------

    def timeSeries(
        self, ts_type: str, tag: int, *args: int | float | str,
    ) -> None:
        self._lines.append(_ops_call("timeSeries", ts_type, tag, *args))

    # -- Patterns -----------------------------------------------------------

    def pattern_open(
        self, p_type: str, tag: int, *args: int | float | str,
    ) -> None:
        # Same shape as section_open — openseespy's current-pattern
        # state is opened by ops.pattern(...).
        self._lines.append(_ops_call("pattern", p_type, tag, *args))

    def pattern_close(self) -> None:
        # No-op — see section_close.
        pass

    def load(self, tag: int, *forces: float) -> None:
        self._lines.append(_ops_call("load", tag, *forces))

    def eleLoad(self, *args: int | float | str) -> None:
        beam3d_guard = self._beam3d_targets.guard_for(
            args, self._model_ndm if self._model_ndm is not None else 3)
        if beam3d_guard is not None:
            self._require_fork_floors((beam3d_guard,))
        self._lines.append(_ops_call("eleLoad", *args))

    def sp(self, tag: int, dof: int, value: float) -> None:
        self._lines.append(_ops_call("sp", tag, dof, value))

    def sp_hold(self, node: int, dof: int) -> None:
        # HOLD support (ADR 0052): pin the DOF at its current deformed
        # displacement, captured at runtime via ``ops.nodeDisp``, with
        # ``-const`` so the value is never scaled by the load factor.
        # Emitted as a raw line because the value is a runtime expression
        # (``ops.nodeDisp(...)``), not a literal that ``_ops_call`` formats.
        self._lines.append(
            f"ops.sp({node}, {dof}, ops.nodeDisp({node}, {dof}), '-const')"
        )

    # -- Recorders ----------------------------------------------------------

    def recorder(self, kind: str, *args: int | float | str) -> None:
        if kind == "mpco" and args and str(args[0]) not in self._mpco_files:
            self._mpco_files.append(str(args[0]))
        if kind == "mpco" and args and self._run_record:
            for line in deck_run_unlink("py", str(args[0])):
                self._lines.append(line)
        self._lines.append(_ops_call("recorder", kind, *args))

    def recorder_declaration_begin(
        self,
        *,
        declaration_name: str,
        record_name: str | None,
        category: str,
        components: tuple[str, ...],
        raw: tuple[str, ...] = (),
        pg: tuple[str, ...] = (),
        label: tuple[str, ...] = (),
        selection: tuple[str, ...] = (),
        ids: tuple[int, ...] | None = None,
        dt: float | None = None,
        n_steps: int | None = None,
        file_root: str = ".",
    ) -> None:
        """Phase 9 schema 2.3.0 declaration metadata — no-op for openseespy."""
        del (
            declaration_name, record_name, category, components, raw,
            pg, label, selection, ids, dt, n_steps, file_root,
        )

    def recorder_declaration_end(self) -> None:
        """Phase 9 schema 2.3.0 declaration metadata — no-op for openseespy."""

    # -- Analysis chain -----------------------------------------------------

    def constraints(self, c_type: str, *args: int | float | str) -> None:
        self._lines.append(_ops_call("constraints", c_type, *args))

    def numberer(self, n_type: str) -> None:
        self._lines.append(_ops_call("numberer", n_type))

    def system(self, s_type: str, *args: int | float | str) -> None:
        self._lines.append(_ops_call("system", s_type, *args))

    def test(self, t_type: str, *args: int | float | str) -> None:
        self._lines.append(_ops_call("test", t_type, *args))

    def algorithm(self, a_type: str, *args: int | float | str) -> None:
        self._lines.append(_ops_call("algorithm", a_type, *args))

    def integrator(self, i_type: str, *args: int | float | str) -> None:
        self._lines.append(_ops_call("integrator", i_type, *args))

    def analysis(self, a_type: str) -> None:
        self._analysis_token = str(a_type)
        self._lines.append(_ops_call("analysis", a_type))

    def _deck_run_record(
        self, archive: "Callable[[Any], None]", steps: int,
        dt: "float | None", bridge: Any = None, deck: str = "",
    ) -> None:
        """Make the deck write its run record (#2210) as its last action.

        ``apeSees.py`` calls this after everything else, and only with
        ``analyze_steps``: the deck then writes ``<file>.run.json`` beside
        each MPCO recorder's ``file`` (resolved as the recorder resolves
        it), through ``<name>.tmp`` and a rename. A partitioned deck writes
        it on rank 0 after a barrier. A failed increment raises
        (``SystemExit``) before these lines, so a failed or killed run
        writes nothing. With no MPCO recorder, or no ``analysis``
        declaration (the loop cannot run), there is nothing to describe
        and nothing is emitted. ``archive`` writes the deck's ``model.h5``
        to a path (:func:`deck_run_lineage`).
        """
        for line in deck_run_body(
            "py", archive, self._mpco_files, self._analysis_token, steps,
            dt, partitioned=self._partition_shim_emitted, bridge=bridge,
            deck=deck,
        ):
            self._lines.append(line)

    def analyze(
        self, *, steps: int, dt: float | None = None,
        label: str | None = None,
        strategy: StrategySpec | None = None,
    ) -> int:
        self._stage_zero()
        # Fail-loud per-increment loop (see the Emitter Protocol note):
        # a batched ``ops.analyze(N)`` short-circuits internally on the
        # first failed increment and the deck would silently run on
        # with the stage partial (or not applied at all).  Every
        # increment is checked; the first failure aborts the deck with
        # a banner naming the loop, increment, and pseudo-time.
        #
        # ADR 0057 Phase A: with ``strategy``, the loop walks the rung
        # ladder on a failed increment (rung 0 = the chain's own
        # algorithm) and restores rung 0 after a rescue; exhaustion
        # aborts with the same banner naming the ladder.
        n = int(steps)
        # ADR 0106 D2's stage_marker_name closes the same quoting hole
        # here: a raw label could carry ``\`` or a newline into the
        # generated ``raise SystemExit("...")`` banner below (the old
        # ``"`` -> ``'`` swap alone left those open).
        where = f" of stage '{stage_marker_name(label)}'" if label else ""
        call = (
            _ops_call("analyze", 1) if dt is None
            else _ops_call("analyze", 1, dt)
        )
        if strategy is None:
            self._lines.append(f"for _apesees_i in range({n}):")
            prev_indent = self._lines.indent
            self._lines.indent = prev_indent + "    "
            if self._step_hooks_registered:
                self._lines.append("_apesees_call_before_step()")
            self._lines.append(f"if {call} != 0:")
            self._lines.indent = prev_indent + "        "
            self._lines.append(
                'raise SystemExit("apeGmsh: analyze FAILED at increment '
                f'%d/{n}{where} (pseudo-time %g) -- aborting, the remaining '
                'deck would run on a partial state" % '
                "(_apesees_i + 1, ops.getTime()))"
            )
            self._lines.indent = prev_indent + "    "
            self._emit_progress_marker(n, prev_indent)
            if self._step_hooks_registered:
                self._lines.append("_apesees_call_after_step()")
            self._lines.indent = prev_indent
            return 0

        try:
            rungs_literal = "[" + ", ".join(
                "(" + ", ".join(_fmt_value(a) for a in rung) + ",)"
                for rung in strategy.rungs
            ) + "]"
        except _NonFiniteArg as exc:
            k, bad = next((k, r) for k, r in enumerate(strategy.rungs)
                          if any(_is_nonfinite_token(a) for a in r))
            raise _nonfinite_error(
                f"analyze strategy {strategy.name!r} rung {k}", bad, exc.value,
            ) from None
        sname = stage_marker_name(strategy.name)
        if any(r and r[0] == SUBDIVIDE_TOKEN for r in strategy.rungs):
            return self._analyze_subdivided(
                n=n, dt=dt, where=where, sname=sname,
                rungs_literal=rungs_literal, n_rungs=len(strategy.rungs))
        self._lines.append(f"_apesees_rungs = {rungs_literal}")
        self._lines.append(f"for _apesees_i in range({n}):")
        prev_indent = self._lines.indent
        self._lines.indent = prev_indent + "    "
        if self._step_hooks_registered:
            self._lines.append("_apesees_call_before_step()")
        self._lines.append("for _apesees_r in range(len(_apesees_rungs)):")
        self._lines.indent = prev_indent + "        "
        self._lines.append("if _apesees_r:")
        self._lines.indent = prev_indent + "            "
        self._lines.append(
            f'print("apeGmsh strategy \'{sname}\': increment '
            f'%d/{n}{where} -> rung %d %r" % '
            "(_apesees_i + 1, _apesees_r, _apesees_rungs[_apesees_r]))"
        )
        self._lines.append("ops.algorithm(*_apesees_rungs[_apesees_r])")
        self._lines.indent = prev_indent + "        "
        self._lines.append(f"_apesees_rc = {call}")
        self._lines.append("if _apesees_rc == 0:")
        self._lines.indent = prev_indent + "            "
        self._lines.append("break")
        self._lines.indent = prev_indent + "        "
        # A refused COMMIT is not non-convergence: walking the rest of
        # the ladder would re-analyze a partially committed domain.
        # Same verdict the live emitter reaches through
        # ``_internal.analyze_rc.check_analyze_rc``.
        self._lines.append(f"if _apesees_rc == {COMMIT_ABORT_RC}:")
        self._lines.indent = prev_indent + "            "
        self._lines.append(
            'raise RuntimeError("apeGmsh: analyze aborted at increment '
            f'%d/{n}{where} (pseudo-time %g) -- {COMMIT_ABORT_MESSAGE}" '
            "% (_apesees_i + 1, ops.getTime()))"
        )
        self._lines.indent = prev_indent + "    "
        self._lines.append("else:")
        self._lines.indent = prev_indent + "        "
        self._lines.append(
            'raise SystemExit("apeGmsh: analyze FAILED at increment '
            f'%d/{n}{where} (pseudo-time %g) -- aborting after '
            f"exhausting strategy ladder '{sname}' "
            f'({len(strategy.rungs)} rungs); the remaining deck would '
            'run on a partial state" % (_apesees_i + 1, ops.getTime()))'
        )
        self._lines.indent = prev_indent + "    "
        self._lines.append("if _apesees_r:")
        self._lines.indent = prev_indent + "        "
        self._lines.append("ops.algorithm(*_apesees_rungs[0])")
        self._lines.indent = prev_indent + "    "
        self._emit_progress_marker(n, prev_indent)
        if self._step_hooks_registered:
            self._lines.append("_apesees_call_after_step()")
        self._lines.indent = prev_indent
        return 0

    def _analyze_subdivided(
        self, *, n: int, dt: float | None, where: str, sname: str,
        rungs_literal: str, n_rungs: int,
    ) -> int:
        """The ladder loop with :class:`Subdivide` rungs (ADR 0057
        Amendment 1): each increment ends at ``t_target = t + dt``; an
        algorithm rung retries the remaining time, a ``Subdivide(n)`` rung
        re-issues rung 0's algorithm and covers it in ``n`` pieces."""
        if dt is None:
            raise BridgeError(
                f"analyze strategy '{sname}'{where}: a Subdivide rung cuts "
                "the time step, so it needs a transient stage with dt "
                "(s.run(n_increments=..., dt=...)); a static stage uses "
                "Substep (ADR 0057 Amendment 1)."
            )
        add = self._lines.append
        add(f"_apesees_rungs = {rungs_literal}")
        add(f"for _apesees_i in range({n}):")
        prev = self._lines.indent

        def ind(k: int) -> str:
            return prev + "    " * k

        self._lines.indent = ind(1)
        if self._step_hooks_registered:
            add("_apesees_call_before_step()")
        add(f"_apesees_t1 = ops.getTime() + {_fmt_value(float(dt))}")
        add("for _apesees_r in range(len(_apesees_rungs)):")
        self._lines.indent = ind(2)
        add("_apesees_g = _apesees_rungs[_apesees_r]")
        add("if _apesees_r:")
        self._lines.indent = ind(3)
        add(
            f'print("apeGmsh strategy \'{sname}\': increment '
            f'%d/{n}{where} -> rung %d %r" % '
            "(_apesees_i + 1, _apesees_r, _apesees_g))"
        )
        self._lines.indent = ind(2)
        add(f"if _apesees_g[0] == {SUBDIVIDE_TOKEN!r}:")
        self._lines.indent = ind(3)
        add("ops.algorithm(*_apesees_rungs[0])")
        add("_apesees_h = (_apesees_t1 - ops.getTime()) / _apesees_g[1]")
        add("_apesees_rc = 0")
        add("for _apesees_k in range(_apesees_g[1]):")
        self._lines.indent = ind(4)
        add("_apesees_rc = ops.analyze(1, _apesees_h)")
        add("if _apesees_rc != 0:")
        self._lines.indent = ind(5)
        add("break")
        self._lines.indent = ind(3)
        add(
            f'print("apeGmsh strategy \'{sname}\': increment '
            f'%d/{n}{where} Subdivide(%d) committed %d/%d pieces" % '
            "(_apesees_i + 1, _apesees_g[1], _apesees_k + (_apesees_rc == 0),"
            " _apesees_g[1]))"
        )
        self._lines.indent = ind(2)
        add("else:")
        self._lines.indent = ind(3)
        add("if _apesees_r:")
        self._lines.indent = ind(4)
        add("ops.algorithm(*_apesees_g)")
        self._lines.indent = ind(3)
        add("_apesees_rc = ops.analyze(1, _apesees_t1 - ops.getTime())")
        self._lines.indent = ind(2)
        add("if _apesees_rc == 0:")
        self._lines.indent = ind(3)
        add("break")
        self._lines.indent = ind(2)
        add(f"if _apesees_rc == {COMMIT_ABORT_RC}:")
        self._lines.indent = ind(3)
        add(
            'raise RuntimeError("apeGmsh: analyze aborted at increment '
            f'%d/{n}{where} (pseudo-time %g) -- {COMMIT_ABORT_MESSAGE}" '
            "% (_apesees_i + 1, ops.getTime()))"
        )
        self._lines.indent = ind(1)
        add("else:")
        self._lines.indent = ind(2)
        add(
            'raise SystemExit("apeGmsh: analyze FAILED at increment '
            f'%d/{n}{where} (pseudo-time %g) -- aborting after '
            f"exhausting strategy ladder '{sname}' "
            f'({n_rungs} rungs); the remaining deck would '
            'run on a partial state" % (_apesees_i + 1, ops.getTime()))'
        )
        self._lines.indent = ind(1)
        add("if _apesees_r:")
        self._lines.indent = ind(2)
        add("ops.algorithm(*_apesees_rungs[0])")
        self._lines.indent = ind(1)
        self._emit_progress_marker(n, prev)
        if self._step_hooks_registered:
            add("_apesees_call_after_step()")
        self._lines.indent = prev
        return 0

    def _emit_progress_marker(self, n: int, prev_indent: str) -> None:
        """Emit a throttled ``print(APEGMSH_PROGRESS ...)`` inside the
        analyze loop (~20 samples) when progress markers are on.

        Gated by ``_emit_progress`` (set by ``apeSees.py(progress=)``).
        ``flush=True`` streams the marker live through the pipe. Callers
        must have the loop body indent (``prev_indent + 4``) active;
        this restores it on exit.
        """
        if not self._emit_progress:
            return
        every = max(1, n // 20)
        self._lines.append(
            f"if (_apesees_i + 1) % {every} == 0 or _apesees_i + 1 == {n}:"
        )
        self._lines.indent = prev_indent + "        "
        self._lines.append(
            f'print("APEGMSH_PROGRESS i=%d n={n} t=%g" '
            "% (_apesees_i + 1, ops.getTime()), flush=True)"
        )
        self._lines.indent = prev_indent + "    "

    def _emit_stage_marker(self, phase: str, name: str) -> None:
        """ADR 0106 D2 — ``print(APEGMSH_STAGE open|close <name>)`` at
        the current (outer) indent, name LAST so a name with spaces
        survives the S1 parser's ``r"APEGMSH_STAGE (open|close) (.+)$"``.

        Quoting normalised the same way ``analyze`` already normalises a
        strategy name: ``"`` -> ``'`` so the name cannot close the
        enclosing Python string literal early.
        """
        sname = stage_marker_name(name)
        self._lines.append(f'print("APEGMSH_STAGE {phase} {sname}", flush=True)')

    def eigen(
        self, num_modes: int, *, solver: str = "-genBandArpack",
    ) -> list[float]:
        self._stage_zero()
        self._lines.append(_ops_call("eigen", solver, num_modes))
        return []

    def modal_properties(
        self, *, unorm: bool = False, out: str | None = None,
    ) -> dict[str, list[float]]:
        args: list[str] = []
        if unorm:
            args.append("-unorm")
        if out is not None:
            args.extend(("-file", out))
        self._lines.append(_ops_call("modalProperties", *args))
        return {}

    def eigen_feast(
        self, f_min: float, f_max: float, *, certify: bool = False,
        trace_stop: bool = False,
    ) -> list[float]:
        self._stage_zero()
        args: list[float | str] = ["-feast", float(f_min), float(f_max)]
        if certify:
            args.append("-certify")
        if trace_stop:
            args.append("-traceStop")
        self._lines.append(_ops_call("eigen", *args))
        return []

    def modal_response_history(
        self, *args: int | float | str,
    ) -> None:
        self._lines.append(_ops_call("modalResponseHistory", *args))

    def response_spectrum_analysis(
        self, direction: int, *args: int | float | str,
    ) -> None:
        self._lines.append(
            _ops_call("responseSpectrumAnalysis", direction, *args)
        )

    def profiler(self, *args: int | float | str) -> None:
        self._lines.append(_ops_call("profiler", *args))

    # -- Stress control (Phase SSI-1: initial_stress + ramping hooks) -------

    def addToParameter(
        self, tag: int, ele_tag: int, response: str,
    ) -> None:
        self._lines.append(
            _ops_call("addToParameter", tag, "element", ele_tag, response)
        )

    def flip_element_stage(
        self, pid: int, ele_tags: tuple[int, ...],
    ) -> None:
        self._lines.append(_ops_call("parameter", int(pid)))
        for et in ele_tags:
            self._lines.append(
                _ops_call("addToParameter", int(pid), "element", int(et), "stage")
            )
        self._lines.append(_ops_call("updateParameter", int(pid), 1))
        self._lines.append(_ops_call("remove", "parameter", int(pid)))

    def update_parameter(
        self,
        pid: int,
        ele_tags: tuple[int, ...],
        args: tuple[str | int, ...],
        value: float,
    ) -> None:
        self._stage0.parameter(args)
        self._stage_carriers.note_parameter(args, value)
        if args and args[0] == "materialState":
            self._require_fork_floors(
                self._stage_carriers.fork_floors(ele_tags))
        self._lines.append(_ops_call("parameter", int(pid)))
        for et in ele_tags:
            self._lines.append(
                _ops_call(
                    "addToParameter", int(pid), "element", int(et), *args,
                )
            )
        self._lines.append(
            _ops_call("updateParameter", int(pid), float(value))
        )
        self._lines.append(_ops_call("remove", "parameter", int(pid)))

    def step_hook_ramp(
        self,
        name: str,
        *,
        targets: tuple[tuple[int, float], ...],
        n_steps_to_full: float,
        phase: Literal["before", "after"] = "before",
    ) -> None:
        _refuse_nonfinite_ramp(name, targets, n_steps_to_full)
        # 1. Dispatcher boilerplate (idempotent).
        if not self._hook_dispatcher_emitted:
            self._emit_hook_dispatcher_boilerplate()
            self._hook_dispatcher_emitted = True
        # 2. parameter declarations.
        for tag, _target in targets:
            self._lines.append(_ops_call("parameter", int(tag)))
        # 3. The per-step function body.
        self._emit_hook_ramp_function(name, targets, n_steps_to_full)
        # 4. Register with the appropriate dispatcher list.
        list_var = (
            "_apesees_before_step_hooks"
            if phase == "before"
            else "_apesees_after_step_hooks"
        )
        self._lines.append(f"{list_var}.append({name})")
        # 5. Flip flag so analyze() wraps.
        self._step_hooks_registered = True

    def _emit_hook_dispatcher_boilerplate(self) -> None:
        """Emit the once-per-deck hook dispatcher infrastructure."""
        prev_indent = self._lines.indent
        self._lines.indent = ""
        self._lines.append("# apeSees per-step hook dispatcher (Phase SSI-1)")
        self._lines.append("_apesees_before_step_hooks: list = []")
        self._lines.append("_apesees_after_step_hooks: list = []")
        self._lines.append("def _apesees_call_before_step() -> None:")
        self._lines.append("    for _f in _apesees_before_step_hooks:")
        self._lines.append("        _f()")
        self._lines.append("def _apesees_call_after_step() -> None:")
        self._lines.append("    for _f in _apesees_after_step_hooks:")
        self._lines.append("        _f()")
        self._lines.indent = prev_indent

    # -- Staged analysis (Phase SSI-2.A) ------------------------------------

    def stage_open(self, name: str) -> None:
        prev_indent = self._lines.indent
        self._lines.indent = ""
        self._lines.append(f"# === Stage: {name} ===")
        if self.caps.emit_stage_markers:
            self._current_stage_name = name
            self._emit_stage_marker("open", name)
        self._lines.indent = prev_indent

    def domain_change(self) -> None:
        prev_indent = self._lines.indent
        self._lines.indent = ""
        self._lines.append(_ops_call("domainChange"))
        self._lines.indent = prev_indent

    def stage_close(self) -> None:
        prev_indent = self._lines.indent
        self._lines.indent = ""
        if self.caps.emit_stage_markers and self._current_stage_name is not None:
            self._emit_stage_marker("close", self._current_stage_name)
            self._current_stage_name = None
        # openseespy: ``ops.loadConst('-time', 0.0)`` matches the Tcl
        # ``loadConst -time 0.0`` semantics.
        self._lines.append(_ops_call("loadConst", "-time", 0.0))
        self._lines.append(_ops_call("wipeAnalysis"))
        if self._step_hooks_registered:
            self._lines.append("_apesees_before_step_hooks.clear()")
            self._lines.append("_apesees_after_step_hooks.clear()")
            self._step_hooks_registered = False
        self._lines.indent = prev_indent

    # -- Staged-analysis mutators (Phase SSI-2.E) ---------------------------

    def set_time(self, t: float) -> None:
        prev_indent = self._lines.indent
        self._lines.indent = ""
        self._lines.append(_ops_call("setTime", float(t)))
        self._lines.indent = prev_indent

    def set_creep(self, on: bool) -> None:
        prev_indent = self._lines.indent
        self._lines.indent = ""
        self._lines.append(_ops_call("setCreep", 1 if on else 0))
        self._lines.indent = prev_indent

    def reset(self) -> None:
        prev_indent = self._lines.indent
        self._lines.indent = ""
        self._lines.append(_ops_call("reset"))
        self._lines.indent = prev_indent

    def set_node_vel(self, node: int, dof: int, value: float) -> None:
        prev_indent = self._lines.indent
        self._lines.indent = ""
        self._lines.append(_ops_call(
            "setNodeVel", int(node), int(dof), float(value), "-commit",
        ))
        self._lines.indent = prev_indent

    def set_node_accel(self, node: int, dof: int, value: float) -> None:
        prev_indent = self._lines.indent
        self._lines.indent = ""
        self._lines.append(_ops_call(
            "setNodeAccel", int(node), int(dof), float(value), "-commit",
        ))
        self._lines.indent = prev_indent

    def remove_sp(self, node: int, dof: int) -> None:
        prev_indent = self._lines.indent
        self._lines.indent = ""
        self._lines.append(_ops_call("remove", "sp", int(node), int(dof)))
        self._lines.indent = prev_indent

    def remove_element(self, tag: int) -> None:
        self._stage_carriers.note_removed(tag)
        self._beam3d_targets.note_removed(tag)
        prev_indent = self._lines.indent
        self._lines.indent = ""
        self._lines.append(_ops_call("remove", "element", int(tag)))
        self._lines.indent = prev_indent

    def update_material_stage(self, mat_tag: int, stage: int) -> None:
        self._stage0.user_flip(mat_tag)
        self._require_fork_floors(
            self._stage_carriers.flip_fork_floors(mat_tag, stage))
        self._stage_carriers.note_flip(mat_tag, stage)
        # A plastic flip of a swept material also updates each of its
        # elements; MaterialStageCarriers says why.
        prev_indent = self._lines.indent
        self._lines.indent = ""
        for command, argv in self._stage_carriers.flip_commands(
                mat_tag, stage):
            self._lines.append(_ops_call(command, *argv))
        self._lines.indent = prev_indent

    def _require_fork_floors(
        self, floors: "Sequence[ForkBuildGuard]",
    ) -> None:
        """Make the deck check each new fork build guard at its start.

        The check (:func:`fork_guard_lines`) goes right after the
        ``openseespy`` import, before ``ops.wipe()``, and stops the deck
        when the running build fails it (#2504, #2516).  A deck that
        needs no guard is unchanged.
        """
        for floor in floors:
            if floor in self._fork_floors:
                continue
            self._fork_floors.add(floor)
            guard = fork_guard_lines(floor, "py")
            for line in guard:
                self._lines.insert(self._fork_floor_at, line)
                self._fork_floor_at += 1

    def _stage_zero(self) -> None:
        """Write the owed fresh-stage lines of the class-wide groups
        (:class:`~apeGmsh.opensees._internal.build.stage_validate.
        StageZeroLedger`), at a solve verb, and guard a solve the fork's
        constructor reset would change
        (:meth:`MaterialStageCarriers.solve_fork_guards`)."""
        self._require_fork_floors(self._stage_carriers.solve_fork_guards())
        for tag, stage in self._stage0.due():
            self._lines.append(_ops_call(
                "updateMaterialStage", "-material", tag, "-stage", stage))

    def _emit_hook_ramp_function(
        self,
        name: str,
        targets: tuple[tuple[int, float], ...],
        n_steps_to_full: float,
    ) -> None:
        """Emit the per-step linear ramp function for one hook.

        Mirrors the Tcl proc: a module-level ``{name}_state`` dict
        carries the step counter and cumulative-value per parameter;
        each call advances the counter, recomputes the capped factor,
        and emits one ``ops.updateParameter`` per target.
        """
        prev_indent = self._lines.indent
        self._lines.indent = ""
        # State dict — declared at module level so the closure can
        # mutate it across calls.  Initial cumulative is 0.0 for every
        # parameter; the counter starts at 0.
        state_items = ", ".join(
            f"'cum_{int(tag)}': 0.0" for tag, _ in targets
        )
        self._lines.append(
            f"{name}_state: dict = {{'count': 0, {state_items}}}"
        )
        self._lines.append(f"def {name}() -> None:")
        self._lines.append(f"    {name}_state['count'] += 1")
        self._lines.append(
            f"    _factor = min({name}_state['count'] / "
            f"{repr(float(n_steps_to_full))}, 1.0)"
        )
        for tag, target in targets:
            self._lines.append(
                f"    _cur = {repr(float(target))} * _factor"
            )
            self._lines.append(
                f"    _delta = _cur - {name}_state['cum_{int(tag)}']"
            )
            self._lines.append(
                f"    ops.updateParameter({int(tag)}, _delta)"
            )
            self._lines.append(f"    {name}_state['cum_{int(tag)}'] = _cur")
        self._lines.indent = prev_indent

    # -- Partition emission scoping (ADR 0027, P4) --------------------------

    def partition_open(self, rank: int) -> None:
        """Open ``if getPID() == K:`` block; indent body 4 spaces.

        On the **first** call across the emitter's lifetime, emit the
        one-shot runtime shim wrapping
        ``from openseespy.opensees import getPID`` in a ``try /
        except ImportError`` with a 0-returning fallback — so a
        single-process Python (no OpenSeesMP) still runs the deck.
        """
        self._stage_carriers.partition_open(rank)
        if not self._partition_shim_emitted:
            # Emit at indent 0 (the shim is global, not per-rank).
            prev_indent = self._lines.indent
            self._lines.indent = ""
            self._lines.append("try:")
            self._lines.append(
                "    from openseespy.opensees import getPID"
            )
            self._lines.append("except ImportError:")
            self._lines.append("    def getPID() -> int:")
            self._lines.append("        return 0")
            self._lines.indent = prev_indent
            self._partition_shim_emitted = True
        # Header line emitted at the outer indent.
        self._lines.append(f"if getPID() == {rank}:")
        # Indent subsequent emit calls one level deeper.
        self._lines.indent = self._lines.indent + "    "

    def partition_close(self) -> None:
        """Close the current per-rank block; emit trailing blank line.

        Python uses indentation, not a closing brace — ``partition_close``
        simply dedents and emits a blank separator line at the outer
        indent for readability.
        """
        self._stage_carriers.partition_close()
        if self._lines.indent.endswith("    "):
            self._lines.indent = self._lines.indent[:-4]
        self._lines.append("")

    # -- Partition runtime-conditional fallback (ADR 0027 INV-5) ------------

    def parallel_runtime_fallback_numberer(
        self, primary: str, fallback: str,
    ) -> None:
        """Emit a ``try / except`` wrapping ``ops.numberer(primary)``
        with a fallback ``ops.numberer(fallback)``.

        ``ParallelPlain`` only exists in OpenSeesMP; under single-process
        openseespy the call raises and the ``except`` branch falls back
        to ``RCM`` so the deck still runs.  Restores shim-consistency
        with :meth:`partition_open`'s ``getPID`` fallback (ADR 0027
        INV-5 amendment 2026-05-23).
        """
        self._lines.append(
            f"# {primary} only exists in OpenSeesMP; "
            f"fall back to {fallback} under single-process OpenSees."
        )
        self._lines.append("try:")
        self._lines.append(f"    {_ops_call('numberer', primary)}")
        self._lines.append("except Exception:")
        self._lines.append(f"    {_ops_call('numberer', fallback)}")

    def parallel_runtime_fallback_system(
        self, primary: str, fallback: str,
    ) -> None:
        """Emit a ``try / except`` wrapping ``ops.system(primary)``
        with a fallback ``ops.system(fallback)``.

        Mirror of :meth:`parallel_runtime_fallback_numberer`.
        """
        self._lines.append(
            f"# {primary} requires OpenSeesMP + MPI; "
            f"fall back to {fallback} under single-process OpenSees."
        )
        self._lines.append("try:")
        self._lines.append(f"    {_ops_call('system', primary)}")
        self._lines.append("except Exception:")
        self._lines.append(f"    {_ops_call('system', fallback)}")

    # -- Command channel (ADR 0114 D2/D3) ---------------------------------

    def command(self, verb: str, *args: int | float | str) -> None:
        command_row(verb)
        self._lines.append(_ops_call(verb, *args))
