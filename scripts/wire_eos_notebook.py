"""Smart auto-wirer for EOS notebooks.

Inserts a "Record results" section after the analysis of an EOS notebook
that builds its OpenSees model by hand from a FEMData broker (nodes and
elements tagged by their FEM ids). The section re-runs the notebook's
build + analysis and records it on a route chosen by the build the deck
ran on (``backend_info_of(ops)``):

- **Ladruno fork:** ``ResolvedRecorderSpec.emit_default`` records the run
  with the fork's own recorder (``.ladruno``).
- **Stock openseespy:** the cell samples ``ops.nodeDisp`` after every
  ``ops.analyze`` and writes a native file with ``NativeWriter``.
  ``emit_default`` cannot record this deck on stock without MPCO: its
  ``.out`` route needs a bridge archive that declares ``ndm``
  (``apeSees(fem).h5``), and a hand-built deck has none.

Both files are read with ``Results.from_fem(fem, path)`` (the deck tags by
FEM id, so the FEMData snapshot is the model). The cell then checks the
recorded last step against the live domain, node by node, and opens the
run in the apeGmsh ResultsViewer (``Results.viewer`` honours
``APEGMSH_SKIP_VIEWER`` for headless runs).

Algorithm:
  1. Read notebook with nbformat.
  2. Drop an earlier wiring (the V1 ``DomainCapture`` cells); the new
     section goes where it was.
  3. Find first code cell containing ``ops.wipe()`` AND ``ops.model``
     -- the start of model build.
  4. Find last code cell from that point containing ``ops.analyze``
     or ``ops.eigen`` -- end of build+analyze.
  5. Detect ``-ndm N`` from the cell sources (the recorded axes).
  6. Concatenate cells [build_start .. build_end] verbatim and split
     them before the first top-level statement that calls
     ``ops.analyze``: the build runs before the recorder opens, the
     analysis inside it. Call ``_record_step()`` after every
     ``ops.analyze`` line; refuse a call the hook cannot follow.
  7. Refuse an eigen-only build: no default route records modes.
  8. Insert the cells where the V1 wiring was, else before the first
     top-level ``ops.wipe()`` / ``g.end()`` after the analysis (the live
     domain must still be up) and the section heading above it, else
     append them.

Idempotent: skips if the V2 marker is already present.
"""
from __future__ import annotations

import ast
import io
import re
import sys
import textwrap
from pathlib import Path

import nbformat

MARKER = "# --- EOS-WIRING-V2 ---"
#: The marker the DomainCapture wiring (V1) wrote; ``wire`` replaces it.
_V1_MARKER = "# --- EOS-WIRING-V1 ---"
_V1_HEADING = "Capture results — manual + DomainCapture paths"


def _new_md(src: str):
    return nbformat.v4.new_markdown_cell(src)


def _new_code(src: str):
    return nbformat.v4.new_code_cell(src)


def _is_marker_present(nb) -> bool:
    return any(
        c.cell_type == "code" and MARKER in c.source for c in nb.cells
    )


def _is_v1_cell(cell) -> bool:
    """A cell the V1 (DomainCapture) wiring wrote, or hand-tuned from it."""
    src = cell.source
    if cell.cell_type == "markdown":
        return _V1_HEADING in src
    return (
        _V1_MARKER in src
        or "DomainCapture(" in src
        or "Results.from_native(results_capture)" in src
    )


def _drop_v1_cells(nb) -> int | None:
    """Remove the V1 wiring; return the index it started at, or None."""
    hits = [i for i, c in enumerate(nb.cells) if _is_v1_cell(c)]
    if not hits:
        return None
    nb.cells = [c for i, c in enumerate(nb.cells) if i not in set(hits)]
    return hits[0]


def _find_build_range(nb) -> tuple[int, int] | None:
    """Return (start_idx, end_idx) of the build+analyze block, or None."""
    start = None
    for i, c in enumerate(nb.cells):
        if c.cell_type != "code":
            continue
        if re.search(r"\bops\.wipe\(\)", c.source) and "ops.model" in c.source:
            start = i
            break
    if start is None:
        return None
    end = start
    for i in range(start, len(nb.cells)):
        c = nb.cells[i]
        if c.cell_type != "code":
            continue
        if re.search(r"\bops\.(analyze|eigen)\(", c.source):
            end = i
    return (start, end)


def _detect_ndm(code: str) -> int:
    m = re.search(
        r'ops\.model\(\s*[\'"]basic[\'"]\s*,\s*[\'"]-ndm[\'"]\s*,\s*(\d+)',
        code,
    )
    return int(m.group(1)) if m else 3


def _has_eigen(code: str) -> bool:
    return bool(re.search(r"\bops\.eigen\(", code))


def _has_analyze(code: str) -> bool:
    return bool(re.search(r"\bops\.analyze\(", code))


def _calls_analyze(node: ast.AST) -> bool:
    return any(
        isinstance(n, ast.Call)
        and isinstance(n.func, ast.Attribute)
        and n.func.attr == "analyze"
        and isinstance(n.func.value, ast.Name)
        and n.func.value.id == "ops"
        for n in ast.walk(node)
    )


def _split_at_first_analyze(code: str) -> tuple[str, str]:
    """Split ``code`` before its first top-level statement that calls
    ``ops.analyze``: ``(build, analysis)``.

    Raises :class:`ValueError` when the code does not parse or calls no
    ``ops.analyze`` at top level.
    """
    tree = ast.parse(code)
    for stmt in tree.body:
        if _calls_analyze(stmt):
            lines = code.splitlines()
            cut = stmt.lineno - 1
            return "\n".join(lines[:cut]).rstrip(), "\n".join(lines[cut:])
    raise ValueError("no top-level statement calls ops.analyze")


def _strip_ops_wipe_before_end(code: str) -> str:
    """Drop a stray trailing ``ops.wipe()``: the live domain must stay up
    until the recorded run is checked against it."""
    return re.sub(r"\nops\.wipe\(\)\s*$", "", code)


def _is_transient(code: str) -> bool:
    """Detect ``ops.analyze(N, dt)`` two-arg form -> transient."""
    return bool(re.search(r"ops\.analyze\(\s*[^,)]+\s*,\s*[^,)]+\s*\)", code))


_ANALYZE_LINE_RE = re.compile(
    r"^(?P<indent>[ \t]*)(?:[\w, ]+=\s*)?ops\.analyze\([^\n#]*\)\s*(?:#[^\n]*)?$",
    re.MULTILINE,
)


def _hook_each_analyze(code: str) -> str:
    """After every ``ops.analyze(...)`` statement line, call
    ``_record_step()`` at the same indentation (the stock route samples the
    live domain there).

    Raises :class:`ValueError` when an ``ops.analyze`` call sits in a form
    the hook cannot follow (inside an expression, split over lines): a
    step that is not sampled would read back as a missing step.
    """
    calls = sum(
        1 for n in ast.walk(ast.parse(code))
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
        and n.func.attr == "analyze" and isinstance(n.func.value, ast.Name)
        and n.func.value.id == "ops"
    )
    hooked, n = _ANALYZE_LINE_RE.subn(
        lambda m: f"{m.group(0)}\n{m.group('indent')}_record_step()", code)
    if n != calls:
        raise ValueError(
            f"{calls} ops.analyze call(s) but {n} on a line of their own")
    return hooked


# ---------------------------------------------------------------------------
# Cell construction
# ---------------------------------------------------------------------------

_MD_INTRO = """\
## Record results

The cell below re-runs the analysis and records it, choosing the route by
the OpenSees build the deck ran on:

- **Ladruno fork:** ``ResolvedRecorderSpec.emit_default`` records the run
  with the fork's own recorder (a ``.ladruno`` file).
- **Stock openseespy:** the cell samples the live domain after every step
  and writes a native results file with ``NativeWriter``. ``emit_default``
  cannot record this deck on a stock build without MPCO: its ``.out``
  route needs a bridge model archive that declares ``ndm``, and this deck
  was built by hand.

Either file is read back with ``Results.from_fem(fem, path)``: the deck
tags nodes and elements by their FEM ids, so the FEMData snapshot is the
model. The cell then checks the recorded last step against the live
domain. ``Results.viewer`` honours ``APEGMSH_SKIP_VIEWER``, so this
notebook is safe to run under nbconvert / CI without changes.
"""


def _capture_cell(ndm: int, build_code: str, analysis_code: str, *,
                  transient: bool = False) -> str:
    """Compose the record cell around the notebook's own code.

    ``analysis_code`` already calls ``_record_step()`` after each
    ``ops.analyze`` (:func:`_hook_each_analyze`).
    """
    body = textwrap.indent(_strip_ops_wipe_before_end(analysis_code), "    ")
    stage_kind = "transient" if transient else "static"
    return f"""\
{MARKER}
# Record the run: the fork's own recorder on the Ladruno fork, a native
# file sampled from the live domain on stock openseespy.
import contextlib
import numpy as np
from apeGmsh import Results, workdir
from apeGmsh.opensees._target import backend_info_of
from apeGmsh.results.spec import ResolvedRecorderRecord, ResolvedRecorderSpec
from apeGmsh.results.writers import NativeWriter

OUT = workdir()

{build_code}

# Record every FEM node the deck built.
_axes = ("x", "y", "z")[:{ndm}]
_built = set(int(n) for n in ops.getNodeTags())
_node_ids = np.asarray(
    [int(n) for n in fem.nodes.ids if int(n) in _built], dtype=np.int64)
_backend = backend_info_of(ops)
_fork = _backend.kind == "fork"
print(f"OpenSees build: {{_backend.kind}} ({{_backend.source}})")

if _fork:
    spec = ResolvedRecorderSpec(
        fem_snapshot_id=fem.snapshot_id,
        records=tuple(
            ResolvedRecorderRecord(
                category="nodes", name=kind,
                components=tuple(f"{{kind}}_{{a}}" for a in _axes),
                dt=None, n_steps=None, node_ids=_node_ids,
            )
            for kind in ("displacement", "reaction_force")
        ),
    )
    # emit_default needs a model archive on disk; the deck tags nodes and
    # elements by FEM id, so the FEMData archive is that model.
    model_h5 = OUT / "model.h5"
    fem.to_h5(str(model_h5))
    _recording = spec.emit_default(OUT / "run", model_h5=model_h5, ops=ops,
                                   overwrite=True)
else:
    _recording = contextlib.nullcontext()

_steps = []   # stock: (time, displacement) after every step


def _record_step():
    if not _fork:
        _steps.append((ops.getTime(), np.array(
            [ops.nodeDisp(int(n))[:{ndm}] for n in _node_ids])))


with _recording as run:
{body}

if _fork:
    results_path = run.path
    results = Results.from_fem(fem, results_path, cache_root=OUT)
else:
    results_path = OUT / "run_native.h5"
    if results_path.exists():
        results_path.unlink()
    _u = np.stack([u for _, u in _steps])          # (T, N, ndm)
    with NativeWriter(results_path) as _nw:
        _nw.open(fem=fem, source_type="native", opensees_backend=_backend.kind)
        _sid = _nw.begin_stage(name="analysis", kind="{stage_kind}",
                               time=np.array([t for t, _ in _steps]))
        _nw.write_nodes(_sid, "partition_0", node_ids=_node_ids, components={{
            f"displacement_{{a}}": _u[:, :, k] for k, a in enumerate(_axes)}})
        _nw.end_stage()
    results = Results.from_fem(fem, results_path, kind="native", cache_root=OUT)
print(f"recorded -> {{results_path}}")

# The recorded last step is the live domain's displacement field.
_slabs = [results.nodes.get(ids=_node_ids, component=f"displacement_{{a}}")
          for a in _axes]
assert sorted(int(n) for n in _slabs[0].node_ids) == sorted(_node_ids.tolist())
_rec = np.column_stack([s.values[-1] for s in _slabs])
_live = np.array([ops.nodeDisp(int(n))[:{ndm}] for n in _slabs[0].node_ids])
_scale = max(float(np.abs(_live).max()), 1e-300)
assert np.allclose(_rec, _live, rtol=1e-6, atol=1e-9 * _scale), (
    "the recorded displacement disagrees with the live domain")
print(f"recorded last step matches the live domain at {{len(_node_ids)}} nodes")
"""


def _viewer_cell() -> str:
    return """\
# Open the recorded run in the apeGmsh ResultsViewer (subprocess, non-blocking).
# Set APEGMSH_SKIP_VIEWER=1 in the environment to skip the GUI under nbconvert / CI.
results.viewer(blocking=False)
"""


# ---------------------------------------------------------------------------
# Top-level wire(): apply to a notebook
# ---------------------------------------------------------------------------


def _teardown_index(nb, after: int) -> int | None:
    """First cell after ``after`` with a top-level ``ops.wipe()`` /
    ``g.end()`` / ``g_ctx.__exit__`` (the live domain or the Gmsh session
    goes away there), or None."""
    for i in range(after + 1, len(nb.cells)):
        c = nb.cells[i]
        if c.cell_type != "code":
            continue
        for line in c.source.splitlines():
            stripped = line.strip()
            if stripped.startswith("#"):
                continue
            if (
                stripped.startswith("ops.wipe()")
                or stripped.startswith("g.end()")
                or stripped.startswith("g_ctx.__exit__")
            ):
                return i
    return None


def wire(nb_path: Path) -> str:
    nb = nbformat.read(nb_path, as_version=4)
    if _is_marker_present(nb):
        return f"[skip] {nb_path.name}: already wired"
    v1_at = _drop_v1_cells(nb)

    rng = _find_build_range(nb)
    if rng is None:
        return f"[skip] {nb_path.name}: no ops.wipe()+ops.model() build cell found"
    bs, be = rng

    code_text = "\n".join(
        c.source for c in nb.cells[bs : be + 1] if c.cell_type == "code"
    )
    if "fem." not in code_text:
        return f"[skip] {nb_path.name}: build code does not use FEMData (fem.*) — manual notebook"
    if _has_eigen(code_text) and not _has_analyze(code_text):
        return (f"[skip] {nb_path.name}: eigen-only build; no default results "
                "route records modes")
    ndm = _detect_ndm(code_text)

    # Concatenate build cells; mark cell boundaries as comments to keep diff clear.
    parts = []
    for i in range(bs, be + 1):
        c = nb.cells[i]
        if c.cell_type != "code":
            continue
        parts.append(f"# --- copied from cell {i} ---\n" + c.source.rstrip())
    try:
        build_code, analysis_code = _split_at_first_analyze("\n".join(parts))
        analysis_code = _hook_each_analyze(analysis_code)
    except (SyntaxError, ValueError) as exc:
        return f"[skip] {nb_path.name}: cannot hook the analysis ({exc})"

    new_cells = [
        _new_md(_MD_INTRO),
        _new_code(_capture_cell(ndm, build_code, analysis_code,
                                transient=_is_transient(code_text))),
        _new_code(_viewer_cell()),
    ]

    insert_at = v1_at if v1_at is not None else _teardown_index(nb, be)
    # Keep a section heading with the code it introduces.
    while (v1_at is None and insert_at is not None and insert_at - 1 > be
           and nb.cells[insert_at - 1].cell_type == "markdown"):
        insert_at -= 1
    if insert_at is None:
        nb.cells.extend(new_cells)
        where = "appended at end"
    else:
        nb.cells = nb.cells[:insert_at] + new_cells + nb.cells[insert_at:]
        where = f"inserted at cell {insert_at}"

    nbformat.write(nb, nb_path)
    replaced = " replacing the V1 wiring" if v1_at is not None else ""
    return (
        f"[wired] {nb_path.name}: ndm={ndm} ({where}{replaced}, "
        f"build cells {bs}..{be})"
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    if len(sys.argv) < 2:
        print("Usage: python scripts/wire_eos_notebook.py <notebook> [<notebook> ...]")
        print("       python scripts/wire_eos_notebook.py --all")
        sys.exit(1)
    eos = Path("examples/EOS Examples")
    if sys.argv[1] == "--all":
        # Notebooks now live in per-example folders, recurse to find them.
        targets = sorted(p for p in eos.rglob("*.ipynb")
                         if "checkpoint" not in str(p))
    else:
        # Accept either bare name (resolved to <eos>/<stem>/<stem>.ipynb)
        # or an explicit relative/absolute path.
        def _resolve(arg: str) -> Path:
            p = Path(arg)
            if p.exists():
                return p
            stem = p.stem
            candidate = eos / stem / p.name
            return candidate if candidate.exists() else eos / arg
        targets = [_resolve(a) for a in sys.argv[1:]]
    for p in targets:
        print(wire(p))
