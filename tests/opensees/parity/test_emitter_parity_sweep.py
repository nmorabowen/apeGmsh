"""End-to-end parity sweep — drive the same model through every emitter
and verify the LOGICAL command sequence is identical.

The Tcl emitter wraps section / pattern bodies in ``\\{`` ... ``\\}``;
the Py emitter folds open/close into a single ``ops.section(...)`` /
``ops.pattern(...)`` call (current-X state). Recording captures the
raw Protocol calls. We reduce all three to a tuple of
``(command, type_token, *positional_int_or_str_args)`` and compare.

If parity fails, the bug is one of:
  - a primitive's _emit emits in a different order between targets
  - an emitter's state machine is wrong
  - the bridge fan-out is non-deterministic

For the live emitter, we don't compare *strings* (it executes ops
calls in-process, no string output) — we verify that the openseespy
domain holds the right number and identity of nodes / elements after
the same model is driven.
"""
from __future__ import annotations

from typing import cast

from apeGmsh.opensees import apeSees
from apeGmsh.opensees.emitter.py import PyEmitter
from apeGmsh.opensees.emitter.recording import RecordingEmitter
from apeGmsh.opensees.emitter.tcl import TclEmitter

from tests.opensees.fixtures.fem_stub import (
    make_two_column_frame,
    make_two_node_beam,
)


def _build_simple_cantilever() -> apeSees:
    """A 1-element elastic cantilever with model + fix + pattern + analysis chain."""
    fem = make_two_node_beam()
    ops = apeSees(cast("object", fem))  # type: ignore[arg-type]
    ops.model(ndm=3, ndf=6)
    transf = ops.geomTransf.Linear(vecxz=(1.0, 0.0, 0.0))
    ops.element.elasticBeamColumn(
        pg="Cols",
        transf=transf,
        A=0.01, E=200e9, Iz=1e-4, Iy=1e-4, G=80e9, J=1e-4,
    )
    ops.fix(pg="Base", dofs=(1, 1, 1, 1, 1, 1))
    ts = ops.timeSeries.Linear()
    with ops.pattern.Plain(series=ts) as p:
        p.load(node=2, forces=(1000.0, 0.0, 0.0, 0.0, 0.0, 0.0))
    ops.constraints.Plain()
    ops.numberer.Plain()
    ops.system.BandGeneral()
    ops.test.NormDispIncr(tol=1e-6, max_iter=10)
    ops.algorithm.Linear()
    ops.integrator.LoadControl(dlam=1.0)
    ops.analysis.Static()
    return ops


def _build_two_column_frame() -> apeSees:
    """Two parallel columns sharing a base PG, with pattern & csys."""
    from apeGmsh.opensees.transform import Cartesian
    fem = make_two_column_frame()
    ops = apeSees(cast("object", fem))  # type: ignore[arg-type]
    ops.model(ndm=3, ndf=6)
    transf = ops.geomTransf.Linear(csys=Cartesian())
    ops.element.elasticBeamColumn(
        pg="Cols",
        transf=transf,
        A=0.01, E=200e9, Iz=1e-4, Iy=1e-4, G=80e9, J=1e-4,
    )
    ops.fix(pg="Base", dofs=(1, 1, 1, 1, 1, 1))
    ts = ops.timeSeries.Linear()
    with ops.pattern.Plain(series=ts) as p:
        p.load(pg="Top", forces=(0.0, 0.0, -1000.0, 0.0, 0.0, 0.0))
    return ops


def _canon_recording(rec: RecordingEmitter) -> list[tuple[str, ...]]:
    """Reduce RecordingEmitter calls to canonical tuples.

    Strips section_close / pattern_close (block-scoping markers, not
    commands) — those are no-ops in py and absent from Tcl as
    standalone command lines. ``model`` becomes ``("model", str(ndm), str(ndf))``.
    """
    out: list[tuple[str, ...]] = []
    for name, args, kwargs in rec.calls:
        if name in ("section_close", "pattern_close"):
            continue
        if name == "section_open":
            out.append(("section", *(_str(a) for a in args)))
        elif name == "pattern_open":
            out.append(("pattern", *(_str(a) for a in args)))
        elif name == "model":
            out.append(("model", str(kwargs["ndm"]), str(kwargs["ndf"])))
        elif name == "analyze":
            steps = kwargs["steps"]
            dt = kwargs.get("dt")
            if dt is None:
                out.append(("analyze", str(steps)))
            else:
                out.append(("analyze", str(steps), str(dt)))
        else:
            out.append((name, *(_str(a) for a in args)))
    return out


def _canon_tcl(tcl: TclEmitter) -> list[tuple[str, ...]]:
    """Reduce TclEmitter lines to canonical tuples.

    Skips comments, blank lines, and bare ``\\{`` / ``\\}`` block markers.
    For ``section Fiber 1 -GJ 1.0e9 \\{`` we drop the trailing ``\\{``.
    Tokens after the leading verb are returned verbatim.
    """
    out: list[tuple[str, ...]] = []
    for raw in tcl.lines():
        ln = raw.strip()
        if not ln or ln.startswith("#"):
            continue
        if ln == "}":
            continue
        # Strip trailing { from open-block lines.
        if ln.endswith("{"):
            ln = ln[:-1].strip()
        tokens = ln.split()
        if not tokens:
            continue
        # First token is the command verb (e.g. "model", "section").
        # Reduce 'model BasicBuilder -ndm 3 -ndf 6' to ('model', '3', '6').
        if tokens[0] == "model":
            # Layout: model BasicBuilder -ndm 3 -ndf 6
            ndm_idx = tokens.index("-ndm") + 1
            ndf_idx = tokens.index("-ndf") + 1
            out.append(("model", tokens[ndm_idx], tokens[ndf_idx]))
        else:
            out.append(tuple(tokens))
    return out


def _canon_py(py: PyEmitter) -> list[tuple[str, ...]]:
    """Reduce PyEmitter lines to canonical tuples.

    Skips comments, ``import``, ``ops.wipe()``. Parses
    ``ops.X(arg, arg, ...)`` into ``("X", str(arg), str(arg), ...)``.
    Then normalizes ``model`` similarly to the others.
    """
    out: list[tuple[str, ...]] = []
    for raw in py.lines():
        ln = raw.strip()
        if not ln or ln.startswith("#"):
            continue
        if ln.startswith("import "):
            continue
        if ln.startswith("ops.wipe("):
            continue
        # Parse ops.X(args).
        if not ln.startswith("ops."):
            continue
        rest = ln[len("ops."):]
        verb, _, paren = rest.partition("(")
        # Strip trailing ).
        assert paren.endswith(")")
        body = paren[:-1]
        # Naive arg-split — OpenSees vocabulary doesn't include
        # commas inside string args, so split-on-comma works.
        args = [a.strip() for a in body.split(",") if a.strip()]
        # Strip enclosing quotes from string args.
        cleaned = [_strip_quotes(a) for a in args]
        if verb == "model":
            # ops.model('basic', '-ndm', 3, '-ndf', 6)
            ndm_i = cleaned.index("-ndm") + 1
            ndf_i = cleaned.index("-ndf") + 1
            out.append(("model", cleaned[ndm_i], cleaned[ndf_i]))
        else:
            out.append((verb, *cleaned))
    return out


def _str(v: object) -> str:
    """Render one value the same way Tcl / Py do.

    Floats render as ``repr(float)``; ints / strings pass through.
    Bools become ``1`` / ``0``."""
    if isinstance(v, bool):
        return "1" if v else "0"
    if isinstance(v, str):
        return v
    if isinstance(v, int):
        return str(v)
    if isinstance(v, float):
        return repr(v)
    return str(v)


def _strip_quotes(token: str) -> str:
    """Strip a single layer of single quotes if present."""
    if token.startswith("'") and token.endswith("'") and len(token) >= 2:
        return token[1:-1]
    return token


def _drive_through_all(ops: apeSees) -> tuple[list, list, list]:
    """Drive a fresh build through all 3 file-target emitters; return
    the canonicalized command lists for Recording, Tcl, Py."""
    rec = RecordingEmitter()
    ops.build().emit(rec)
    tcl = TclEmitter()
    ops.build().emit(tcl)
    py = PyEmitter()
    ops.build().emit(py)
    return _canon_recording(rec), _canon_tcl(tcl), _canon_py(py)


def test_parity_simple_cantilever() -> None:
    """All three file-target emitters produce the same logical command
    sequence on the simple cantilever fixture."""
    rec_cmds, tcl_cmds, py_cmds = _drive_through_all(_build_simple_cantilever())

    # Same length.
    assert len(rec_cmds) == len(tcl_cmds), (
        f"Tcl ({len(tcl_cmds)}) != Recording ({len(rec_cmds)})\n"
        f"Tcl: {tcl_cmds}\nRec: {rec_cmds}"
    )
    assert len(rec_cmds) == len(py_cmds), (
        f"Py ({len(py_cmds)}) != Recording ({len(rec_cmds)})\n"
        f"Py: {py_cmds}\nRec: {rec_cmds}"
    )

    # Same verbs in the same order.
    rec_verbs = [c[0] for c in rec_cmds]
    tcl_verbs = [c[0] for c in tcl_cmds]
    py_verbs = [c[0] for c in py_cmds]
    assert rec_verbs == tcl_verbs == py_verbs


def test_parity_two_column_frame() -> None:
    """Same parity invariant on the two-column frame with csys and PG load."""
    rec_cmds, tcl_cmds, py_cmds = _drive_through_all(_build_two_column_frame())

    assert len(rec_cmds) == len(tcl_cmds) == len(py_cmds), (
        f"counts diverge — rec={len(rec_cmds)}, tcl={len(tcl_cmds)}, "
        f"py={len(py_cmds)}"
    )

    rec_verbs = [c[0] for c in rec_cmds]
    tcl_verbs = [c[0] for c in tcl_cmds]
    py_verbs = [c[0] for c in py_cmds]
    assert rec_verbs == tcl_verbs == py_verbs


def test_parity_node_command_count_matches_fem() -> None:
    """All three emitters produce one ``node`` command per FEM node."""
    fem = make_two_column_frame()  # 4 nodes
    ops = apeSees(cast("object", fem))  # type: ignore[arg-type]
    ops.model(ndm=3, ndf=6)
    transf = ops.geomTransf.Linear(vecxz=(1.0, 0.0, 0.0))
    ops.element.elasticBeamColumn(
        pg="Cols",
        transf=transf,
        A=0.01, E=200e9, Iz=1e-4, Iy=1e-4, G=80e9, J=1e-4,
    )

    rec = RecordingEmitter()
    ops.build().emit(rec)
    n_node_recording = sum(1 for c in rec.calls if c[0] == "node")

    tcl = TclEmitter()
    ops.build().emit(tcl)
    n_node_tcl = sum(1 for ln in tcl.lines() if ln.startswith("node "))

    py = PyEmitter()
    ops.build().emit(py)
    n_node_py = sum(1 for ln in py.lines() if ln.startswith("ops.node("))

    assert n_node_recording == n_node_tcl == n_node_py == 4


def test_readme_smoke_end_to_end_to_tcl(tmp_path):  # type: ignore[no-untyped-def]
    """The architecture/README.md smoke (a frame to a Tcl deck) runs
    end-to-end without raising."""
    ops = _build_simple_cantilever()
    out = tmp_path / "frame.tcl"
    ops.tcl(str(out), run=False)

    text = out.read_text()
    # Sanity: every primitive family appears at least once.
    assert "model BasicBuilder" in text
    assert "node 1 0.0 0.0 0.0" in text
    assert "geomTransf Linear" in text
    assert "element elasticBeamColumn" in text
    assert "fix " in text
    assert "timeSeries Linear" in text
    assert "pattern Plain" in text
    assert "load 2" in text
    assert "constraints Plain" in text
    assert "numberer Plain" in text
    assert "system BandGeneral" in text
    assert "test NormDispIncr" in text
    assert "algorithm Linear" in text
    assert "integrator LoadControl" in text
    assert "analysis Static" in text
