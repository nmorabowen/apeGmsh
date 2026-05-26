"""Cross-rank constraint cost microbenchmark (ADR 0038 §"v1 scope gate").

Measures the per-rank cost of emitting + parsing a partitioned Tcl/Py
deck containing ``ASDEmbeddedNodeElement`` constraints at SSI-class
scale. Drives the v1 scope decision per ADR 0038's three-branch
thresholds at ``interface_size=10_000`` × ``ranks=4``:

  * ``deck_emit_sec  < 5.0``
  * ``deck_parse_py_sec < 2.0``
  * ``deck_lines    < 500_000``
  * ``peak_rss_mb   < 1_500``

The matrix sweeps ``interface_size ∈ {100, 1_000, 10_000, 100_000}`` ×
``ranks ∈ {2, 4, 8}`` × ``element_kind ∈ {"tet_host_line_embed",
"hex_host_line_embed"}`` = 24 cells. The 10k × 4 cell is the gate;
the smaller cells provide a scaling trace and the larger cells inform
the 100k-fail fallback branch.

Fixture choice — **stub-based, not real Gmsh + METIS.** The benchmark
measures emit / parse cost on a given ``FEMData`` broker; that cost is
downstream of meshing and partitioning. Stubs let us hit exact
interface sizes deterministically and decouple measurement from
Gmsh-side wall-clock variance (which is itself a known cost surfaced
elsewhere). One rebar node per host element; partitions stripe element
ids modulo ``ranks``.

Output — aggregated rows are written to
``docs/benchmarks/cross_rank_constraint_cost.md`` at session teardown
when running under the nightly ``benchmarks.yml`` workflow (the
finalizer always overwrites the markdown table when at least one
benchmark case ran).

Skip behaviour — ``psutil`` is import-skipped via
:func:`pytest.importorskip`; the cells skip cleanly when not installed
(``peak_rss_mb`` is the load-bearing metric and is not optional).
"""
from __future__ import annotations

import os
import time
from pathlib import Path
from typing import cast

import numpy as np
import pytest

# psutil drives ``peak_rss_mb``. Skip the entire module when absent so
# the cells don't fire half-measured (rather than burying gaps in the
# table); local sanity runs without it fall through cleanly.
psutil = pytest.importorskip("psutil")

# Suppress OpenSees binary banner before any apeGmsh import. The
# importorskip above runs first so this doesn't redirect a missing-
# module error.
os.environ.setdefault("LADRUNO_OPENSEES_QUIET", "1")

from apeGmsh._kernel.records._constraints import InterpolationRecord
from apeGmsh._kernel.records._kinds import ConstraintKind
from apeGmsh.opensees import apeSees

from tests.opensees.fixtures.fem_stub import (
    FEMStub,
    _ElementGroupView,
    _ElementsStub,
    _NodesStub,
)


# ---------------------------------------------------------------------
# Constants — matrix axes per ADR 0038 §"v1 scope gate"
# ---------------------------------------------------------------------

INTERFACE_SIZES = (100, 1_000, 10_000, 100_000)
RANK_COUNTS = (2, 4, 8)
ELEMENT_KINDS = ("tet_host_line_embed", "hex_host_line_embed")

# ADR 0038 gate cell.
GATE_INTERFACE = 10_000
GATE_RANKS = 4

# Decision-gate thresholds (per ADR 0038 §"v1 scope gate").
GATE_EMIT_SEC = 5.0
GATE_PARSE_SEC = 2.0
GATE_DECK_LINES = 500_000
GATE_PEAK_RSS_MB = 1_500.0

# Where to write the populated table when running under CI.
RESULTS_MD_PATH = (
    Path(__file__).resolve().parents[2]
    / "docs"
    / "benchmarks"
    / "cross_rank_constraint_cost.md"
)


# ---------------------------------------------------------------------
# Fixture builder — stub-based, deterministic
# ---------------------------------------------------------------------


def _build_embedded_fem(
    interface_size: int,
    ranks: int,
    element_kind: str,
) -> tuple[FEMStub, str, str]:
    """Build an FEMStub with ``interface_size`` embedded line nodes.

    Returns ``(fem, host_pg, rebar_pg)``. Each rebar node is matched
    to one host element (4-node tet or 8-node hex). Partitions stripe
    elements by id modulo ``ranks``; host and rebar elements interleave
    so every rank carries both. Each rebar node becomes one
    :class:`InterpolationRecord` (kind=EMBEDDED, slave=rebar, masters=
    host corners).

    Node layout (sequential ids starting at 1):

      * host nodes:  ids 1 .. n_host       (n_host = interface_size * corners_per_elem)
      * rebar nodes: ids n_host+1 .. n_host+interface_size

    Element layout (sequential ids starting at 1):

      * host elems:  ids 1 .. interface_size
      * rebar elems: ids interface_size+1 .. 2*interface_size - 1 (line2)

    Total wire-cost per cell scales linearly with ``interface_size``.
    """
    if element_kind == "tet_host_line_embed":
        corners = 4
    elif element_kind == "hex_host_line_embed":
        corners = 8
    else:
        raise ValueError(f"unknown element_kind: {element_kind!r}")

    n_host_nodes = interface_size * corners
    n_rebar_nodes = interface_size
    n_host_elems = interface_size
    n_rebar_elems = max(interface_size - 1, 0)
    n_total_nodes = n_host_nodes + n_rebar_nodes

    # Coords — deterministic but spread enough to keep partitions visually
    # distinct in any downstream visualisation. The benchmark only cares
    # about emit cost, not geometry sanity.
    host_coords = np.zeros((n_host_nodes, 3), dtype=np.float64)
    for i in range(interface_size):
        x = float(i)
        for c in range(corners):
            host_coords[i * corners + c] = (x, float(c % 2), float(c // 2))
    rebar_coords = np.zeros((n_rebar_nodes, 3), dtype=np.float64)
    for j in range(n_rebar_nodes):
        rebar_coords[j] = (float(j), 0.5, 0.5)
    coords = np.vstack([host_coords, rebar_coords])

    node_ids = list(range(1, n_total_nodes + 1))
    host_node_ids = node_ids[:n_host_nodes]
    rebar_node_ids = node_ids[n_host_nodes:]

    nodes = _NodesStub(
        ids=node_ids,
        coords=[(float(x), float(y), float(z)) for (x, y, z) in coords],
        node_pgs={"Host": host_node_ids, "Rebar": rebar_node_ids},
    )

    # Host element connectivities: each element's corners are the
    # `corners` consecutive host node ids.
    host_elem_ids = tuple(range(1, n_host_elems + 1))
    host_conn = tuple(
        tuple(host_node_ids[i * corners + c] for c in range(corners))
        for i in range(n_host_elems)
    )

    # Rebar elements — pairs of consecutive rebar nodes.
    rebar_elem_ids = tuple(
        range(n_host_elems + 1, n_host_elems + 1 + n_rebar_elems)
    )
    rebar_conn = tuple(
        (rebar_node_ids[j], rebar_node_ids[j + 1])
        for j in range(n_rebar_elems)
    )

    elements = _ElementsStub(
        elem_pgs={
            "Host": _ElementGroupView(ids=host_elem_ids, connectivity=host_conn),
            "Rebar": _ElementGroupView(ids=rebar_elem_ids, connectivity=rebar_conn),
        },
    )

    fem = FEMStub(nodes=nodes, elements=elements)

    # Partition striping. Each rank claims every K-th host element and
    # every K-th rebar element (interleaved → host + rebar on every
    # rank, exercising the cross-rank emit path on every rank).
    # Node ownership follows: each partition owns its host element's
    # corner nodes plus its rebar element's two endpoint nodes (the
    # union is non-overlapping by construction because host element ``i``
    # owns nodes ``[i*corners : (i+1)*corners]`` and rebar element
    # ``n_host_elems + j`` owns ``[rebar_node_ids[j], rebar_node_ids[j+1]]``).
    parts: list[tuple[int, list[int], list[int]]] = []
    for r in range(ranks):
        owned_host_ids = host_elem_ids[r::ranks]
        owned_rebar_ids = rebar_elem_ids[r::ranks]
        owned_nodes: set[int] = set()
        for hid in owned_host_ids:
            i = hid - 1  # host elem index
            for c in range(corners):
                owned_nodes.add(host_node_ids[i * corners + c])
        for rid in owned_rebar_ids:
            j = rid - n_host_elems - 1  # rebar elem index
            owned_nodes.add(rebar_node_ids[j])
            owned_nodes.add(rebar_node_ids[j + 1])
        parts.append((
            r,
            sorted(owned_nodes),
            sorted(list(owned_host_ids) + list(owned_rebar_ids)),
        ))
    fem.set_partitions(parts)

    # Embedded interpolation records — one per rebar node bound to its
    # paired host element's master tuple.  The ASDEmbeddedNodeElement
    # C++ parser only accepts 3 (tri3) or 4 (tet4 / quad4) master nodes;
    # the brick host pins to a 4-node face (the first four corners of
    # the brick connectivity) rather than the full 8-node hex.  This
    # mirrors the way an embedded rebar pins to a single face of a
    # surrounding brick at the meshed interface.
    n_masters = 4 if corners >= 4 else corners
    weights = np.full(n_masters, 1.0 / n_masters, dtype=np.float64)
    surface_records = []
    for j in range(interface_size):
        master_corners = list(host_conn[j][:n_masters])
        surface_records.append(
            InterpolationRecord(
                kind=ConstraintKind.EMBEDDED,
                slave_node=rebar_node_ids[j],
                master_nodes=master_corners,
                weights=weights.copy(),
                dofs=[1, 2, 3],
                name=f"embed_{j}",
            ),
        )
    fem.add_surface_constraints(surface_records)

    return fem, "Host", "Rebar"


# ---------------------------------------------------------------------
# Aggregator — collects per-cell rows; finalizer writes markdown
# ---------------------------------------------------------------------


_RESULTS: list[dict] = []


@pytest.fixture(scope="session", autouse=True)
def _benchmark_writer():
    """Session finalizer — write the markdown table when any cell ran.

    Local sanity subset runs (2 cells) still produce a markdown file
    with the rows that fired, leaving the rest as ``— (not run)``.
    Nightly CI runs the full 24-cell matrix and the table is complete.
    """
    yield
    if not _RESULTS:
        return
    _write_markdown_table(_RESULTS, RESULTS_MD_PATH)


def _write_markdown_table(rows: list[dict], out_path: Path) -> None:
    """Render the populated table into ``cross_rank_constraint_cost.md``.

    Preserves the docstring header (title + thresholds quote + decision
    gate section). Overwrites everything from the ``## Results`` heading
    downward.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # Build the row index keyed by (interface_size, ranks, element_kind)
    # so we can render the full matrix with placeholders for missing
    # cells.
    by_key = {
        (int(r["interface_size"]), int(r["ranks"]), str(r["element_kind"])): r
        for r in rows
    }
    from datetime import datetime, timezone
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

    lines: list[str] = [
        "# Cross-rank constraint cost — ADR 0038 §\"v1 scope gate\"",
        "",
        f"Last run: {ts}",
        "",
        "## Thresholds (ADR 0038 §\"v1 scope gate\", 10k × 4 ranks)",
        "",
        f"- `deck_emit_sec     < {GATE_EMIT_SEC}`",
        f"- `deck_parse_py_sec < {GATE_PARSE_SEC}`",
        f"- `deck_lines        < {GATE_DECK_LINES:_}`",
        f"- `peak_rss_mb       < {GATE_PEAK_RSS_MB}`",
        "",
        "## Results",
        "",
        (
            "| interface_size | ranks | element_kind | deck_lines | "
            "deck_emit_sec | deck_parse_py_sec | peak_rss_mb | "
            "pass_at_10k×4 |"
        ),
        (
            "|---:|---:|---|---:|---:|---:|---:|:---:|"
        ),
    ]
    for sz in INTERFACE_SIZES:
        for nk in RANK_COUNTS:
            for ek in ELEMENT_KINDS:
                r = by_key.get((sz, nk, ek))
                if r is None:
                    lines.append(
                        f"| {sz:_} | {nk} | {ek} | — | — | — | — | — |"
                    )
                    continue
                gate_str = "—"
                if sz == GATE_INTERFACE and nk == GATE_RANKS:
                    gate_str = "PASS" if bool(r.get("pass_at_gate")) else "FAIL"
                lines.append(
                    f"| {sz:_} | {nk} | {ek} | "
                    f"{int(r['deck_lines']):_} | "
                    f"{float(r['deck_emit_sec']):.3f} | "
                    f"{float(r['deck_parse_py_sec']):.3f} | "
                    f"{float(r['peak_rss_mb']):.1f} | "
                    f"{gate_str} |"
                )
    lines += [
        "",
        "## Decision gate status",
        "",
    ]
    gate_row = by_key.get((GATE_INTERFACE, GATE_RANKS, "tet_host_line_embed"))
    if gate_row is None:
        lines.append(
            "PENDING — gate cell (10k × 4 × tet_host_line_embed) did not "
            "run in this invocation."
        )
    else:
        pass_emit = float(gate_row["deck_emit_sec"]) < GATE_EMIT_SEC
        pass_parse = float(gate_row["deck_parse_py_sec"]) < GATE_PARSE_SEC
        pass_lines = int(gate_row["deck_lines"]) < GATE_DECK_LINES
        pass_rss = float(gate_row["peak_rss_mb"]) < GATE_PEAK_RSS_MB
        all_pass = pass_emit and pass_parse and pass_lines and pass_rss
        lines.append(f"- `deck_emit_sec`     pass: **{pass_emit}**")
        lines.append(f"- `deck_parse_py_sec` pass: **{pass_parse}**")
        lines.append(f"- `deck_lines`        pass: **{pass_lines}**")
        lines.append(f"- `peak_rss_mb`       pass: **{pass_rss}**")
        lines.append("")
        if all_pass:
            lines.append("**Overall: PASS** — proceed to Phase 2 (full feature).")
        else:
            lines.append(
                "**Overall: FAIL** — per ADR 0038 fallback rules, ship "
                "mesh-cache-only `g.compose()` with `ComposeUnsupportedError` "
                "on cross-module MP-constraints."
            )
    lines.append("")
    out_path.write_text("\n".join(lines), encoding="utf-8")


# ---------------------------------------------------------------------
# Helpers — psutil peak-RSS shim (Windows / Linux)
# ---------------------------------------------------------------------


def _peak_rss_mb_now() -> float:
    """Return the current peak RSS in MiB across platforms.

    Windows: ``peak_wset`` is the high-water-mark working set.
    Linux: ``peak_rss`` (added in psutil 5.7) when present; else ``rss``.
    Other / fallback: ``rss``.
    """
    mi = psutil.Process().memory_info()
    for attr in ("peak_wset", "peak_rss"):
        val = getattr(mi, attr, None)
        if val is not None:
            return float(val) / (1024.0 * 1024.0)
    return float(mi.rss) / (1024.0 * 1024.0)


# ---------------------------------------------------------------------
# Parametrized benchmark — 24 cells
# ---------------------------------------------------------------------


@pytest.mark.bench
@pytest.mark.parametrize("element_kind", ELEMENT_KINDS)
@pytest.mark.parametrize("ranks", RANK_COUNTS, ids=lambda r: f"ranks_{r}")
@pytest.mark.parametrize(
    "interface_size", INTERFACE_SIZES, ids=lambda s: f"size_{s}",
)
def test_cross_rank_constraint_cost(
    interface_size: int, ranks: int, element_kind: str, tmp_path,
) -> None:
    """One cell of the ADR 0038 v1 scope-gate microbenchmark.

    Each cell:
      1. Builds the FEMStub at the configured ``interface_size`` with
         the requested host element kind and partition count.
      2. Emits the Tcl deck via ``apeSees(fem).tcl(...)`` and records
         ``deck_emit_sec`` + ``deck_lines``.
      3. Emits the Py deck and parses it via :func:`compile` to
         measure ``deck_parse_py_sec`` (pure Python parse cost; no
         openseespy runtime needed).
      4. Captures peak RSS observed during the cell's emit + parse work.

    Per-cell result dict is appended to ``_RESULTS``; the session
    finalizer writes the markdown table.
    """
    fem, host_pg, rebar_pg = _build_embedded_fem(
        interface_size, ranks, element_kind,
    )

    # Tcl emit — wall clock around `ops.tcl(path)`.
    tcl_path = tmp_path / f"deck_{interface_size}_{ranks}_{element_kind}.tcl"
    ops_tcl = apeSees(cast("object", fem))
    ops_tcl.model(ndm=3, ndf=3)
    # Material + host element + rebar element setup. ND elastic for
    # the host; uniaxial elastic for the rebar truss.
    from apeGmsh.opensees.material.nd import ElasticIsotropic
    from apeGmsh.opensees.material.uniaxial import ElasticMaterial
    nd_mat = ElasticIsotropic(E=2.0e10, nu=0.2)
    ux_mat = ElasticMaterial(E=2.0e11)
    ops_tcl.register(nd_mat)
    ops_tcl.register(ux_mat)
    if element_kind == "tet_host_line_embed":
        ops_tcl.element.FourNodeTetrahedron(pg=host_pg, material=nd_mat)
    else:
        ops_tcl.element.stdBrick(pg=host_pg, material=nd_mat)
    ops_tcl.element.Truss(pg=rebar_pg, A=1e-4, material=ux_mat)

    t0 = time.perf_counter()
    ops_tcl.tcl(str(tcl_path))
    deck_emit_sec = time.perf_counter() - t0

    tcl_text = tcl_path.read_text()
    deck_lines = tcl_text.count("\n") + (0 if tcl_text.endswith("\n") else 1)

    # Py emit + parse — separate emit (not timed; parse is the gate
    # metric per the ADR) followed by `compile()` to measure pure
    # parse cost. Using the standard ast/compile pipeline isolates
    # interpreter parse time from any openseespy runtime cost.
    py_path = tmp_path / f"deck_{interface_size}_{ranks}_{element_kind}.py"
    ops_py = apeSees(cast("object", fem))
    ops_py.model(ndm=3, ndf=3)
    ops_py.register(nd_mat)
    ops_py.register(ux_mat)
    if element_kind == "tet_host_line_embed":
        ops_py.element.FourNodeTetrahedron(pg=host_pg, material=nd_mat)
    else:
        ops_py.element.stdBrick(pg=host_pg, material=nd_mat)
    ops_py.element.Truss(pg=rebar_pg, A=1e-4, material=ux_mat)
    ops_py.py(str(py_path))

    py_src = py_path.read_text()
    t1 = time.perf_counter()
    compile(py_src, str(py_path), "exec")
    deck_parse_py_sec = time.perf_counter() - t1

    peak_rss_mb = _peak_rss_mb_now()

    is_gate = interface_size == GATE_INTERFACE and ranks == GATE_RANKS
    pass_at_gate = (
        deck_emit_sec < GATE_EMIT_SEC
        and deck_parse_py_sec < GATE_PARSE_SEC
        and deck_lines < GATE_DECK_LINES
        and peak_rss_mb < GATE_PEAK_RSS_MB
    ) if is_gate else False

    _RESULTS.append({
        "interface_size": interface_size,
        "ranks": ranks,
        "element_kind": element_kind,
        "deck_lines": deck_lines,
        "deck_emit_sec": deck_emit_sec,
        "deck_parse_py_sec": deck_parse_py_sec,
        "peak_rss_mb": peak_rss_mb,
        "pass_at_gate": pass_at_gate,
    })

    # Print one-line summary so `pytest -v` shows progress per cell.
    print(
        f"\n[bench] size={interface_size} ranks={ranks} kind={element_kind} "
        f"lines={deck_lines} emit={deck_emit_sec:.3f}s "
        f"parse_py={deck_parse_py_sec:.3f}s peak_rss={peak_rss_mb:.1f}MB",
    )
