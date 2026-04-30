"""Smart auto-wirer for EOS notebooks.

Inserts a "Capture results" section before the final ``g.end()`` of an
EOS notebook, providing two paths to a native-HDF5 results file:

1. **Manual** — query the live OpenSees domain post-analysis, write
   nodal displacements to h5 via ``NativeWriter``.
2. **DomainCapture** — declare ``Recorders`` and run the analysis
   inside a ``DomainCapture`` context. Scales to multi-stage / modal
   / transient runs.

Then opens the captured file in the apeGmsh ResultsViewer (subprocess,
non-blocking; gated on env var ``APEGMSH_SKIP_VIEWER`` for headless
verification).

Algorithm:
  1. Read notebook with nbformat.
  2. Find first code cell containing ``ops.wipe()`` AND ``ops.model``
     — the start of model build.
  3. Find last code cell from that point containing ``ops.analyze``
     or ``ops.eigen`` — end of build+analyze.
  4. Detect ``-ndm N -ndf M`` from the cell sources.
  5. Concatenate cells [build_start .. build_end] verbatim, then
     inject ``cap.step(t=ops.getTime())`` after each ``ops.analyze``
     line (preserving indentation, so loop bodies are handled too).
  6. Detect ``ops.eigen`` → routes through ``cap.capture_modes(...)``
     instead of step-wise capture.
  7. Detect ``ops.analyze(.., dt)`` (two-arg form) → stage kind
     ``"transient"``; otherwise ``"static"``.
  8. Insert the manual cell BEFORE any ``ops.wipe()`` that lives
     between the analysis and ``g.end()`` (so live-domain queries
     still work). Insert capture + viewer cells immediately after.

Idempotent: skips if the marker comment is already present.
"""
from __future__ import annotations

import io
import re
import sys
import textwrap
from pathlib import Path

import nbformat

MARKER = "# --- EOS-WIRING-V1 ---"


def _new_md(src: str):
    return nbformat.v4.new_markdown_cell(src)


def _new_code(src: str):
    return nbformat.v4.new_code_cell(src)


def _is_marker_present(nb) -> bool:
    return any(
        c.cell_type == "code" and MARKER in c.source for c in nb.cells
    )


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


def _detect_ndm_ndf(code: str) -> tuple[int, int]:
    m = re.search(
        r'ops\.model\(\s*[\'"]basic[\'"]\s*,\s*[\'"]-ndm[\'"]\s*,\s*(\d+)\s*,\s*[\'"]-ndf[\'"]\s*,\s*(\d+)',
        code,
    )
    if m:
        return int(m.group(1)), int(m.group(2))
    return 3, 6


def _is_transient(code: str) -> bool:
    """Detect ``ops.analyze(N, dt)`` two-arg form -> transient."""
    return bool(re.search(r"ops\.analyze\(\s*[^,)]+\s*,\s*[^,)]+\s*\)", code))


def _has_eigen(code: str) -> bool:
    return bool(re.search(r"\bops\.eigen\(", code))


def _has_analyze(code: str) -> bool:
    return bool(re.search(r"\bops\.analyze\(", code))


_ANALYZE_RE = re.compile(
    r"^(?P<indent>[ \t]*)(?P<lhs>(?:[\w, ]+=\s*)?)(?P<call>ops\.analyze\([^\n#]*\))\s*(?:#[^\n]*)?$",
    re.MULTILINE,
)

_EIGEN_RE = re.compile(
    r"^(?P<indent>[ \t]*)(?P<lhs>(?:[\w, ]+=\s*)?)(?P<call>ops\.eigen\([^\n#]*\))\s*(?:#[^\n]*)?$",
    re.MULTILINE,
)


def _inject_step_after_analyze(code: str) -> str:
    """After every ``ops.analyze(...)`` line, append ``cap.step(...)``."""

    def _sub(m: re.Match) -> str:
        indent = m.group("indent")
        return f"{m.group(0)}\n{indent}cap.step(t=ops.getTime())"

    return _ANALYZE_RE.sub(_sub, code)


def _strip_ops_wipe_before_end(code: str) -> str:
    """If the build code ends with a stray ``ops.wipe()`` after analyze,
    drop it — DomainCapture's exit and the test harness will clean up."""
    return re.sub(r"\nops\.wipe\(\)\s*$", "", code)


# ---------------------------------------------------------------------------
# Cell construction
# ---------------------------------------------------------------------------

_MD_INTRO = """\
## 9. Capture results — manual + DomainCapture paths

Two ways to produce a native-HDF5 results file consumable by the
apeGmsh ``ResultsViewer``:

1. **Manual path** — query the live OpenSees domain post-analysis,
   open a ``NativeWriter``, and write nodal displacements yourself.
   Good for one-shot snapshots and post-hoc diagnostics.
2. **DomainCapture path** — declare what to capture with
   ``Recorders().nodes(...)``, hand the spec to a ``DomainCapture``
   context, and call ``cap.step(t=...)`` after each ``ops.analyze``
   (the helper does it for you). Scales to multi-stage, transient,
   modal, and multi-recorder runs.

Both produce a file that ``Results.from_native(path).viewer()`` can
open. The viewer launch is gated on ``APEGMSH_SKIP_VIEWER`` so this
notebook is safe to run under nbconvert / CI.
"""


def _manual_cell(basename: str, ndm: int, ndf: int) -> str:
    # ndm controls how many spatial DOFs to query.
    # ndf<ndm would be unusual (e.g. ndf=1 1D bar with ndm=1).
    n_query = min(ndm, ndf)
    queries = []
    for i in range(1, n_query + 1):
        queries.append(
            f'_u{["x","y","z"][i-1]} = np.array([ops.nodeDisp(int(nid), {i}) for nid in fem.nodes.ids])'
        )
    comp_lines = []
    for i, axis in enumerate(["x", "y", "z"], start=1):
        if i <= n_query:
            comp_lines.append(f'    "displacement_{axis}": _u{axis}.reshape(1, _n),')
        else:
            comp_lines.append(f'    "displacement_{axis}": np.zeros((1, _n)),')
    comp_setup = (
        "_n = len(fem.nodes.ids)\n"
        + "\n".join(queries)
        + "\n_components = {\n"
        + "\n".join(comp_lines)
        + "\n}"
    )
    return f"""\
{MARKER}
# Manual path: pull displacements off the live domain, write h5 yourself.
from pathlib import Path
import numpy as np
from apeGmsh.results.writers import NativeWriter

results_manual = Path("{basename}_manual.h5")
if results_manual.exists():
    results_manual.unlink()

{comp_setup}

with NativeWriter(results_manual) as _nw:
    _nw.open(fem=fem)
    _sid = _nw.begin_stage(name="static", kind="static", time=np.array([1.0]))
    _nw.write_nodes(
        _sid, "partition_0",
        node_ids=np.asarray(fem.nodes.ids, dtype=np.int64),
        components=_components,
    )
    _nw.end_stage()

print(f"manual -> {{results_manual}} ({{results_manual.stat().st_size/1024:.1f}} KB)")
"""


def _capture_cell(
    basename: str,
    ndm: int,
    ndf: int,
    build_code: str,
    *,
    eigen_only: bool,
    transient: bool,
    n_modes_default: int = 3,
) -> str:
    """Compose the DomainCapture cell.

    ``build_code`` is the user's original build+analyze code, already
    transformed: ``cap.step(...)`` injected after each ``ops.analyze``.
    For eigen-only, no transformation is applied; we replace the
    eigen call site with ``cap.capture_modes(...)`` after the model
    is built.
    """
    stage_kind = "transient" if transient else "static"
    if eigen_only:
        # Keep the user's ops.eigen(...) call (it computes & prints values),
        # then ``ops.wipeAnalysis()`` to clear OpenSees' implicit analysis
        # state so capture_modes can install its own EigenSOE, then
        # ``cap.capture_modes(...)`` to write one stage per mode.
        body = textwrap.indent(build_code, "    ")
        capture_call = (
            f"    ops.wipeAnalysis()    # clear implicit Analysis from earlier ops.eigen call\n"
            f"    # capture_modes runs ops.eigen(N) and writes one stage per mode\n"
            f"    cap.capture_modes(n_modes={n_modes_default})"
        )
        with_block = (
            f"with DomainCapture(spec, results_capture, fem, ndm={ndm}, ndf={ndf}) as cap:\n"
            f"{body}\n"
            f"{capture_call}"
        )
        recorder_decl = 'recs.nodes(components="displacement")'
    else:
        body = textwrap.indent(_strip_ops_wipe_before_end(build_code), "    ")
        with_block = (
            f"with DomainCapture(spec, results_capture, fem, ndm={ndm}, ndf={ndf}) as cap:\n"
            f"    cap.begin_stage(\"run\", kind=\"{stage_kind}\")\n"
            f"{body}\n"
            f"    cap.end_stage()"
        )
        recorder_decl = (
            'recs.nodes(components="displacement")\n'
            'recs.nodes(components="reaction_force")'
        )
    return f"""\
# DomainCapture path: declarative recorders, capture during analyze.
from apeGmsh.solvers.Recorders import Recorders
from apeGmsh.results.capture._domain import DomainCapture

recs = Recorders()
{recorder_decl}
spec = recs.resolve(fem, ndm={ndm}, ndf={ndf})

results_capture = Path("{basename}_capture.h5")
if results_capture.exists():
    results_capture.unlink()

{with_block}

print(f"capture -> {{results_capture}} ({{results_capture.stat().st_size/1024:.1f}} KB)")
"""


def _viewer_cell() -> str:
    return """\
# Open the captured run in the apeGmsh ResultsViewer (subprocess,
# non-blocking). Set APEGMSH_SKIP_VIEWER=1 to skip in headless / CI.
import os
from apeGmsh.results import Results
results = Results.from_native(results_capture)
if os.environ.get("APEGMSH_SKIP_VIEWER"):
    print("[skip viewer] APEGMSH_SKIP_VIEWER set")
else:
    handle = results.viewer(blocking=False)
    print(f"viewer pid: {handle.pid}  -- close window to exit.")
"""


# ---------------------------------------------------------------------------
# Top-level wire(): apply to a notebook
# ---------------------------------------------------------------------------


def wire(nb_path: Path) -> str:
    nb = nbformat.read(nb_path, as_version=4)
    if _is_marker_present(nb):
        return f"[skip] {nb_path.name}: already wired"

    rng = _find_build_range(nb)
    if rng is None:
        return f"[skip] {nb_path.name}: no ops.wipe()+ops.model() build cell found"
    bs, be = rng

    code_text = "\n".join(
        c.source for c in nb.cells[bs : be + 1] if c.cell_type == "code"
    )
    if "fem." not in code_text:
        return f"[skip] {nb_path.name}: build code does not use FEMData (fem.*) — manual notebook"
    ndm, ndf = _detect_ndm_ndf(code_text)
    eigen_only = _has_eigen(code_text) and not _has_analyze(code_text)
    transient = _is_transient(code_text)

    # Concatenate build cells; mark cell boundaries as comments to keep diff clear.
    parts = []
    for i in range(bs, be + 1):
        c = nb.cells[i]
        if c.cell_type != "code":
            continue
        parts.append(f"# --- copied from cell {i} ---\n" + c.source.rstrip())
    build_code = "\n".join(parts)

    if not eigen_only:
        build_code = _inject_step_after_analyze(build_code)

    basename = nb_path.stem

    new_cells = [
        _new_md(_MD_INTRO),
        _new_code(_manual_cell(basename, ndm, ndf)),
        _new_code(
            _capture_cell(
                basename, ndm, ndf, build_code,
                eigen_only=eigen_only, transient=transient,
            )
        ),
        _new_code(_viewer_cell()),
    ]

    # Insert position: scan cells AFTER the analyze block for the FIRST of:
    #   - top-level ops.wipe()    -> manual cell needs live domain
    #   - g.end() / g_ctx.__exit__ -> Gmsh teardown closes the session
    # Whichever comes first. If neither, append.
    insert_at = None
    teardown_re = re.compile(
        r"^[ \t]*(ops\.wipe\(\)|g\.end\(\)|g_ctx\.__exit__)",
        re.M,
    )
    for i in range(be + 1, len(nb.cells)):
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
                insert_at = i
                break
        if insert_at is not None:
            break

    if insert_at is None:
        nb.cells.extend(new_cells)
        where = "appended at end"
    else:
        nb.cells = nb.cells[:insert_at] + new_cells + nb.cells[insert_at:]
        where = f"inserted at cell {insert_at}"

    nbformat.write(nb, nb_path)
    return (
        f"[wired] {nb_path.name}: ndm={ndm} ndf={ndf} "
        f"eigen={eigen_only} transient={transient} ({where}, "
        f"build cells {bs}..{be})"
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    if len(sys.argv) < 2:
        print("Usage: python _wire_eos.py <notebook_filename> [<notebook_filename> ...]")
        print("       python _wire_eos.py --all")
        sys.exit(1)
    eos = Path("examples/EOS Examples")
    if sys.argv[1] == "--all":
        targets = sorted(p for p in eos.glob("*.ipynb"))
    else:
        targets = [eos / a if not Path(a).exists() else Path(a) for a in sys.argv[1:]]
    for p in targets:
        print(wire(p))
