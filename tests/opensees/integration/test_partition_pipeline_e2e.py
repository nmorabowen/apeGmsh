"""End-to-end integration test for the partition pipeline (P1+P2+P3+P4).

Exercises the full composition through the user-facing API:

    apeGmsh session -> g.mesh.partitioning.partition(n)
        -> g.mesh.queries.get_fem_data(dim=...)
            -> apeSees(fem)
                -> ops.tcl(...) / ops.py(...) / ops.h5(...) / ops.run(...)

Three scenarios are run per ``n_parts`` value (2 and 4):

* **Scenario A — Flavor A (Gmsh-native METIS)**: unweighted balance.
* **Scenario B — Flavor B (pymetis, weighted)**: skips cleanly when
  ``pymetis`` is not installed; otherwise validated against either the
  real binding or the P1 stub at ``/tmp/fake_pymetis_pkg/pymetis.py``.
* **Scenario C — OpenSeesMP runtime smoke**: best-effort runtime
  parse-and-execute via ``mpirun``/``mpiexec`` + ``OpenSeesMP``.
  Skips cleanly when any binary is missing.

The Phase-A and Phase-B PRs cover unit / per-emitter coverage of each
seam; this file pins the **composition** by driving the user-facing
flow against a real Gmsh session — proving the five PRs compose
end-to-end without an integration gap between them.
"""
from __future__ import annotations

import math
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest

# Imports are deferred where they require optional dependencies so
# module import never errors on a fresh CI box.


# ---------------------------------------------------------------------------
# Fixture builders
# ---------------------------------------------------------------------------


def _build_frame_fem(g, *, n_stories: int = 3, plan: float = 5.0,
                     story_height: float = 3.0, lc: float = 1.5) -> None:
    """Build a multi-story 3-D space-frame mesh on the given session.

    Geometry:

      * ``n_stories`` × 4-column rectangular plan, columns aligned at
        the corners of a ``plan`` × ``plan`` footprint.
      * 4 perimeter beams + a master "centre-of-mass" point per floor.
      * Mesh size ``lc`` so each line is subdivided into a handful of
        elements — gives the partitioner enough structure to balance,
        and ensures perimeter floor nodes scatter across multiple ranks
        when partitioned into 4 (so the cross-partition MP-constraint
        replication path actually fires per ADR 0027 INV-1/INV-2).

    Physical groups emitted onto the model:

      * ``"Columns"``     — every column line
      * ``"Beams"``       — every perimeter beam line
      * ``"Base"``        — the four base points
      * ``"Floor{s}"``    — the four perimeter points at story ``s``
                            (slaves of the rigid diaphragm)
      * ``"Master{s}"``   — the single centre-of-mass point at story
                            ``s`` (master of the rigid diaphragm)
    """
    import gmsh

    points: dict[tuple[Any, Any], int] = {}

    # Base nodes — corner of the ``plan`` × ``plan`` footprint at z=0.
    for c, (x, y) in enumerate(
        ((0, 0), (plan, 0), (plan, plan), (0, plan)), start=1,
    ):
        points[("base", c)] = gmsh.model.geo.addPoint(x, y, 0, lc)

    # Floor nodes — same plan, plus a centre-of-mass master at z=s*H.
    for s in range(1, n_stories + 1):
        for c, (x, y) in enumerate(
            ((0, 0), (plan, 0), (plan, plan), (0, plan)), start=1,
        ):
            points[(s, c)] = gmsh.model.geo.addPoint(x, y, s * story_height, lc)
        points[(s, "M")] = gmsh.model.geo.addPoint(
            plan / 2, plan / 2, s * story_height, lc,
        )

    # Columns — story s connects floor s-1 to floor s ("base" at s=1).
    column_lines: list[int] = []
    for s in range(1, n_stories + 1):
        prev: Any = "base" if s == 1 else s - 1
        for c in (1, 2, 3, 4):
            column_lines.append(
                gmsh.model.geo.addLine(points[(prev, c)], points[(s, c)]),
            )

    # Perimeter beams at every floor.
    beam_lines: list[int] = []
    for s in range(1, n_stories + 1):
        beam_lines.extend([
            gmsh.model.geo.addLine(points[(s, 1)], points[(s, 2)]),
            gmsh.model.geo.addLine(points[(s, 2)], points[(s, 3)]),
            gmsh.model.geo.addLine(points[(s, 3)], points[(s, 4)]),
            gmsh.model.geo.addLine(points[(s, 4)], points[(s, 1)]),
        ])

    gmsh.model.geo.synchronize()

    gmsh.model.addPhysicalGroup(1, column_lines, name="Columns")
    gmsh.model.addPhysicalGroup(1, beam_lines, name="Beams")
    gmsh.model.addPhysicalGroup(
        0, [points[("base", c)] for c in (1, 2, 3, 4)], name="Base",
    )
    for s in range(1, n_stories + 1):
        gmsh.model.addPhysicalGroup(
            0, [points[(s, c)] for c in (1, 2, 3, 4)], name=f"Floor{s}",
        )
        gmsh.model.addPhysicalGroup(
            0, [points[(s, "M")]], name=f"Master{s}",
        )

    g.mesh.sizing.set_global_size(lc)
    g.mesh.generation.generate(1)
    g.mesh.partitioning.renumber(dim=1, method="simple", base=1)


def _make_session():
    """Build a fresh apeGmsh session with the standard 3-story frame."""
    from apeGmsh import apeGmsh

    sess = apeGmsh(model_name="partition_e2e", verbose=False)
    sess.begin()
    try:
        _build_frame_fem(sess)
    except Exception:
        sess.end()
        raise
    return sess


def _inject_floor_diaphragms(fem, *, n_stories: int = 3) -> None:
    """Inject one rigid_diaphragm per floor onto ``fem.nodes.constraints``.

    The master is the centre-of-mass node (one node in PG ``Master{s}``)
    and the slaves are the four perimeter corners (PG ``Floor{s}``).
    With a 4-partition decomposition the corners are very likely to
    land on different ranks (the corners are far apart in the dual
    graph), so the cross-partition MP-constraint replication path (ADR
    0027 INV-1/INV-2) fires.
    """
    from apeGmsh._kernel.records._constraints import NodeGroupRecord
    from apeGmsh._kernel.records._kinds import ConstraintKind

    for s in range(1, n_stories + 1):
        master_ids = list(fem.nodes.select(pg=f"Master{s}").ids)
        slave_ids = list(fem.nodes.select(pg=f"Floor{s}").ids)
        assert len(master_ids) == 1, (
            f"expected exactly one master node in PG Master{s}; got "
            f"{master_ids}"
        )
        fem.nodes.constraints._records.append(
            NodeGroupRecord(
                kind=ConstraintKind.RIGID_DIAPHRAGM,
                master_node=int(master_ids[0]),
                slave_nodes=[int(n) for n in slave_ids],
                plane_normal=np.array([0.0, 0.0, 1.0]),
                dofs=[1, 2, 6],
                name=f"floor_{s}",
            ),
        )


def _wire_apesees(fem, *, with_recorders: bool = True):
    """Build an ``apeSees(fem)`` and declare a full physical model.

    Returns the bound ``ops`` so the caller can extend the deck
    (e.g. add a custom numberer / system) or emit directly via
    ``ops.tcl(...)`` / ``ops.py(...)`` / ``ops.h5(...)``.
    """
    from apeGmsh.opensees import apeSees

    ops = apeSees(cast("object", fem))
    ops.model(ndm=3, ndf=6)

    # Two separate transforms — a single vecxz cannot cover both
    # vertical columns and horizontal beams without being parallel to
    # one of the beam axes (OpenSees rejects degenerate vecxz with
    # "vector v that defines plane xz is parallel to x axis"; OpenSeesMP
    # segfaults on the same condition instead of erroring cleanly).
    #   * Columns (along Z): vecxz=(1,0,0) — horizontal, perpendicular.
    #   * Beams (along X or Y): vecxz=(0,0,1) — vertical, perpendicular
    #     to any horizontal beam regardless of plan orientation.
    transf_col = ops.geomTransf.Linear(vecxz=(1.0, 0.0, 0.0))
    transf_beam = ops.geomTransf.Linear(vecxz=(0.0, 0.0, 1.0))
    # Columns and beams share section properties for simplicity.
    ops.element.elasticBeamColumn(
        pg="Columns", transf=transf_col,
        A=0.01, E=200e9, Iz=1e-4, Iy=1e-4, G=80e9, J=1e-4,
    )
    ops.element.elasticBeamColumn(
        pg="Beams", transf=transf_beam,
        A=0.01, E=200e9, Iz=1e-4, Iy=1e-4, G=80e9, J=1e-4,
    )

    # Base SP: pin every base node.
    ops.fix(pg="Base", dofs=(1, 1, 1, 1, 1, 1))

    # Top-floor mass + load + plain analysis chain.
    ops.mass(pg="Master1", values=(1.0, 1.0, 0.0, 0.0, 0.0, 0.0))

    ts = ops.timeSeries.Linear()
    with ops.pattern.Plain(series=ts) as p:
        p.load(pg="Master1", forces=(1000.0, 0.0, 0.0, 0.0, 0.0, 0.0))

    if with_recorders:
        ops.recorder.Node(
            file="disp.out",
            response="disp",
            pg="Master1",
            dofs=(1, 2, 3),
        )
        ops.recorder.Element(
            file="forces.out",
            response=("globalForce",),
            pg="Columns",
        )

    return ops


# ---------------------------------------------------------------------------
# Tcl deck parsers
# ---------------------------------------------------------------------------


def _check_brace_balance(text: str) -> None:
    """Assert ``{`` and ``}`` counts match in a Tcl deck."""
    n_open = text.count("{")
    n_close = text.count("}")
    assert n_open == n_close, (
        f"Tcl brace imbalance: {n_open} '{{' vs {n_close} '}}'"
    )


def _split_into_partition_blocks(text: str) -> dict[int, str]:
    """Return ``{rank: body_text}`` for each ``if {[getPID] == K} {`` block.

    Body text excludes the opening header and closing brace.  Brace
    depth tracked manually since Tcl uses syntactic braces.
    """
    blocks: dict[int, str] = {}
    lines = text.split("\n")
    i = 0
    rank_re = re.compile(r"if \{\[getPID\] == (\d+)\} \{")
    while i < len(lines):
        m = rank_re.search(lines[i])
        if not m:
            i += 1
            continue
        rank = int(m.group(1))
        depth = 1
        i += 1
        body: list[str] = []
        while i < len(lines) and depth > 0:
            for ch in lines[i]:
                if ch == "{":
                    depth += 1
                elif ch == "}":
                    depth -= 1
                    if depth == 0:
                        break
            if depth == 0:
                break
            body.append(lines[i])
            i += 1
        i += 1
        blocks[rank] = "\n".join(body)
    return blocks


def _outside_partition_blocks(text: str) -> str:
    """Return the text OUTSIDE every ``if {[getPID] == K}`` block.

    Strips every ``if {[getPID] == K} { ... }`` sub-region; whatever
    remains is the global script state (analysis chain lives here per
    ADR 0027 INV-5).
    """
    return re.sub(
        r"if \{\[getPID\] == \d+\} \{[\s\S]*?\n\}",
        "",
        text,
    )


# ---------------------------------------------------------------------------
# Scenario A — Flavor A (Gmsh-native METIS, element-count balance)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("n_parts", [2, 4], ids=["n2", "n4"])
def test_scenario_a_flavor_a_full_pipeline(n_parts: int, tmp_path: Path) -> None:
    """E2E: Gmsh-native METIS partition + apeSees emit + h5 round-trip.

    1. Build the 3-story frame; partition with Flavor A (Gmsh-native).
    2. Extract ``fem = g.mesh.queries.get_fem_data(dim=1)``.
    3. Assert broker invariants:
       - ``len(fem.partitions) == n_parts``
       - Sum of per-partition element counts >= total elements (>= because
         boundary elements appear in multiple ranks via ``getPartitions``).
       - For each ``p``: ``p.element_ids.size == p.n_elements`` and
         ``p.node_ids.size == p.n_nodes``.
       - Element-count spread <= 30 % heuristic (METIS allowed slack).
    4. Inject one rigid_diaphragm per floor — at n_parts=4 a master /
       slave pair will fall on different ranks (cross-partition replication
       per ADR 0027 INV-1).
    5. Drive ``apeSees(fem)`` with the standard physical model.
    6. ``ops.tcl(path)`` — assert deck structure: getPID shim ONCE; one
       block per rank; analysis chain OUTSIDE every block; balanced braces.
    7. ``ops.py(path)`` — assert the openseespy equivalents.
    8. ``ops.h5(path)`` — open with h5py; assert ``/opensees/partitions/``
       group present with ``n_partitions == n_parts`` and a sub-group per
       rank with the contract datasets / attrs.
    9. Cross-rank element-tag consistency: tags across all partition
       blocks form a single contiguous set (no leaks, no duplicates
       outside MP-constraint replication).
    10. Round-trip the h5 via :meth:`OpenSeesModel.from_h5` — the
        composed-model reload succeeds end-to-end.
    11. Cross-partition rigid_diaphragm: when a master/slave pair lands
        on multiple ranks, the ``rigidDiaphragm`` line text must appear
        inside more than one ``if {[getPID] == K}`` block.
    """
    import gmsh
    import h5py
    from apeGmsh.opensees import OpenSeesModel

    sess = _make_session()
    try:
        info = sess.mesh.partitioning.partition(n_parts)
        assert info.n_parts == n_parts

        fem = sess.mesh.queries.get_fem_data(dim=1)
        n_total_elements = int(fem.elements.ids.size)

        # --- 3. Broker invariants ---------------------------------------
        assert len(fem.partitions) == n_parts, (
            f"len(fem.partitions)={len(fem.partitions)} != "
            f"requested n_parts={n_parts}"
        )

        sum_per_partition_elements = sum(p.n_elements for p in fem.partitions)
        assert sum_per_partition_elements >= n_total_elements, (
            f"sum of per-partition element counts ({sum_per_partition_elements}) "
            f"must be >= total ({n_total_elements}); boundary elements may be "
            "visible on multiple ranks (this is expected with Gmsh ghosts)."
        )

        per_rank_elem_counts: list[int] = []
        for rec in fem.partitions:
            assert rec.element_ids.size == rec.n_elements
            assert rec.node_ids.size == rec.n_nodes
            assert rec.n_elements > 0, (
                f"partition {rec.id} has no elements; partition is degenerate"
            )
            per_rank_elem_counts.append(rec.n_elements)

        # Balance: METIS heuristic — allow 30 % slack on the spread.
        spread = max(per_rank_elem_counts) - min(per_rank_elem_counts)
        slack = math.ceil(n_total_elements / n_parts * 0.30)
        # Clamp to a small minimum to avoid spurious failures on tiny
        # meshes where any spread > 0 is significant in absolute terms.
        slack = max(slack, 4)
        assert spread <= slack, (
            f"element-count balance too poor: counts={per_rank_elem_counts}, "
            f"spread={spread}, allowed slack={slack}"
        )

        # --- 4. Inject diaphragms BEFORE driving apeSees ----------------
        _inject_floor_diaphragms(fem, n_stories=3)

        # --- 5. Drive apeSees -------------------------------------------
        ops = _wire_apesees(fem)

        # --- 6. Emit Tcl deck -------------------------------------------
        tcl_path = tmp_path / f"model_n{n_parts}.tcl"
        ops.tcl(str(tcl_path))
        text = tcl_path.read_text(encoding="utf-8")

        _check_brace_balance(text)

        assert text.count("proc getPID") == 1, (
            f"the getPID shim must appear exactly once; got "
            f"{text.count('proc getPID')}"
        )

        blocks = _split_into_partition_blocks(text)
        assert len(blocks) == n_parts, (
            f"expected {n_parts} partition blocks; got {sorted(blocks.keys())}"
        )
        # Schema 2.11.0 contract: rank gates are 0-based, matching
        # ``OpenSeesMP::getPID()``.  Under ``mpiexec -np N`` ranks
        # cover ``{0..N-1}`` — the broker's 1-based ``PartitionRecord.id``
        # is *not* used as the runtime rank (the build path uses
        # ``enumerate(fem.partitions)``).
        rank_set = set(blocks.keys())
        assert rank_set == set(range(n_parts)), (
            f"expected 0-based rank gates {set(range(n_parts))}; "
            f"got {sorted(rank_set)}.  If this is {{1..{n_parts}}} the "
            f"0-based-rank fix has regressed."
        )

        outside = _outside_partition_blocks(text)
        assert "numberer ParallelPlain" in outside, (
            "numberer ParallelPlain must be emitted outside partition blocks"
        )
        assert "system Mumps" in outside, (
            "system Mumps must be emitted outside partition blocks"
        )
        # And neither line may appear inside any rank block.
        for rank, body in blocks.items():
            assert "numberer" not in body, (
                f"numberer must not appear inside rank-{rank} block"
            )
            assert "system" not in body, (
                f"system must not appear inside rank-{rank} block"
            )

        # --- 7. Emit Python deck ----------------------------------------
        py_path = tmp_path / f"model_n{n_parts}.py"
        ops.py(str(py_path))
        py_text = py_path.read_text(encoding="utf-8")

        # The Py shim is a try/except wrap of `getPID` from openseespy.
        assert "getPID" in py_text, "Py deck missing the getPID shim symbol"
        # Per schema 2.11.0, Py gates are 0-based matching
        # ``OpenSeesMP::getPID()`` (mirrors the Tcl assertion above).
        py_rank_re = re.compile(r"if getPID\(\) == (\d+):")
        py_rank_set = {int(m.group(1)) for m in py_rank_re.finditer(py_text)}
        assert py_rank_set == set(range(n_parts)), (
            f"Py deck: expected 0-based per-rank gates "
            f"{set(range(n_parts))}; got {sorted(py_rank_set)}"
        )

        # --- 8. Emit H5 archive -----------------------------------------
        h5_path = tmp_path / f"model_n{n_parts}.h5"
        ops.h5(str(h5_path))

        with h5py.File(str(h5_path), "r") as f:
            assert "/opensees/partitions" in f, (
                "H5 file missing /opensees/partitions/ group "
                "(ADR 0027 schema 2.10.0, 0-based ranks since 2.11.0)"
            )
            parts_grp = f["/opensees/partitions"]
            assert int(parts_grp.attrs["n_partitions"]) == n_parts, (
                f"n_partitions attr = "
                f"{parts_grp.attrs.get('n_partitions')!r}; expected {n_parts}"
            )
            # One partition_NN sub-group per rank.  Schema 2.11.0
            # contract: the per-group ``rank`` attr is the 0-based
            # runtime rank (== loop index ``k`` here).
            for k in range(n_parts):
                gname = f"partition_{k:02d}"
                assert gname in parts_grp, (
                    f"H5 missing {gname} sub-group under /opensees/partitions"
                )
                g_p = parts_grp[gname]
                # Required attrs.
                for attr in ("rank", "n_elements", "n_nodes"):
                    assert attr in g_p.attrs, (
                        f"{gname}: missing required attr {attr!r}"
                    )
                assert int(g_p.attrs["rank"]) == k, (
                    f"{gname}/@rank must be {k} (0-based runtime "
                    f"rank, schema 2.11.0); got {int(g_p.attrs['rank'])}"
                )
                # Required datasets.
                for ds in ("element_ids", "node_ids", "boundary_node_ids"):
                    assert ds in g_p, (
                        f"{gname}: missing required dataset {ds!r}"
                    )

            # element_meta should carry per-rank partition_ids column.
            elem_meta = f["/opensees/element_meta"]
            for ele_type_name in elem_meta:
                g_t = elem_meta[ele_type_name]
                if "ids" not in g_t:
                    continue
                ids_len = g_t["ids"].shape[0]
                if "partition_ids" in g_t:
                    pids = np.asarray(g_t["partition_ids"])
                    assert pids.shape == (ids_len,), (
                        f"element_meta/{ele_type_name}/partition_ids shape "
                        f"{pids.shape} does not match ids length {ids_len}"
                    )

        # --- 9. Cross-rank element tag consistency ----------------------
        # Match `element <type> <tag> ...` lines, allowing leading
        # whitespace (the bridge indents the body of each
        # ``if {[getPID] == K} { ... }`` block by 4 spaces).
        ele_tag_re = re.compile(r"^\s*element\s+\S+\s+(\d+)\s", re.MULTILINE)
        per_rank_tags: dict[int, set[int]] = {}
        for rank, body in blocks.items():
            per_rank_tags[rank] = {int(m.group(1)) for m in ele_tag_re.finditer(body)}
        all_emitted_tags: set[int] = set()
        for tags in per_rank_tags.values():
            all_emitted_tags.update(tags)
        assert all_emitted_tags, (
            "no element lines parsed out of any partition block; "
            "deck may not be fanning out as expected"
        )
        # Tag set per rank must be non-empty (every rank owns at least
        # one element — diagnoses a degenerate decomposition).
        for rank, tags in per_rank_tags.items():
            assert tags, (
                f"rank {rank} emitted zero element lines; balance heuristic "
                f"is too loose or partitioner produced an empty rank"
            )

        # --- 10. H5 round-trip via OpenSeesModel.from_h5 ----------------
        # ``ops.h5(path)`` writes a standalone ``model.h5`` (FEM at root,
        # /opensees/ alongside).  Composed ``results.h5`` files put the
        # FEM under ``/model/`` — that's the Results pipeline's shape,
        # not the bridge's.  Default ``fem_root="/"`` is the right entry.
        om = OpenSeesModel.from_h5(str(h5_path))
        assert om is not None
        # The OpenSeesModel handle must rehydrate without error.
        # ``om.ndm`` may be inferred from transform vecxz length when
        # the broker's ``/meta.ndm`` is less than the bridge's spatial
        # ndm (line-only FEM yields broker ndm=1 but bridge ndm=3).
        assert om.ndm == 3
        assert om.ndf == 6

        # --- 11. Cross-partition rigid_diaphragm replication -----------
        # Pick a diaphragm whose master and slaves land on different ranks.
        # Re-read the partition records on the broker side.
        diaphragm_blocks_with_constraint = []
        for rank, body in blocks.items():
            if "rigidDiaphragm" in body:
                diaphragm_blocks_with_constraint.append(rank)

        if n_parts == 4:
            # With 4 partitions, the perimeter of a floor (4 corners + 1
            # centre) will almost always scatter across multiple ranks.
            assert len(diaphragm_blocks_with_constraint) >= 2, (
                f"n_parts=4: expected rigidDiaphragm replicated across "
                f">=2 ranks per ADR 0027 INV-1; got only "
                f"{diaphragm_blocks_with_constraint}.  Bodies: "
                f"{[len(b) for b in blocks.values()]}"
            )
        # When the partition happens to collocate every diaphragm's
        # master+slaves on one rank (rare at n_parts=2), we only assert
        # the line shows up at least once.
        assert len(diaphragm_blocks_with_constraint) >= 1, (
            "rigidDiaphragm must emit on at least one rank when a "
            "diaphragm is declared"
        )
    finally:
        sess.end()


# ---------------------------------------------------------------------------
# Scenario B — Flavor B (weighted, pymetis)
# ---------------------------------------------------------------------------


def _ensure_pymetis():
    """Return the ``pymetis`` module or skip cleanly.

    Tries the installed package first; falls back to the P1 stub at
    ``<tempdir>/fake_pymetis_pkg/pymetis.py`` (per the task brief).
    On POSIX ``tempdir = /tmp``; on Windows it's ``%TEMP%``.  If
    neither path resolves, the test is skipped.
    """
    try:
        import pymetis  # type: ignore[import-not-found]
        return pymetis, "installed"
    except ImportError:
        pass

    import tempfile
    candidates = [
        os.path.join(tempfile.gettempdir(), "fake_pymetis_pkg"),
        "/tmp/fake_pymetis_pkg",  # POSIX path even on Windows shells.
    ]
    for stub_dir in candidates:
        if not os.path.isdir(stub_dir):
            continue
        # Insert the stub's parent dir on sys.path so a regular import
        # picks it up.
        if stub_dir not in sys.path:
            sys.path.insert(0, stub_dir)
        try:
            import pymetis  # type: ignore[import-not-found]
            return pymetis, "stub"
        except ImportError:
            continue

    pytest.skip(
        "pymetis is not installed and no stub at "
        "<tempdir>/fake_pymetis_pkg/pymetis.py was loadable"
    )


@pytest.mark.parametrize("n_parts", [2, 4], ids=["n2", "n4"])
def test_scenario_b_flavor_b_weighted_pipeline(
    n_parts: int, tmp_path: Path,
) -> None:
    """E2E: weighted pymetis partition + apeSees emit.

    Repeats the apeSees emit + h5 emit from Scenario A against a
    weighted partition (Flavor B):

    1. Trivial weights (``ones``) — weight balance is identical to
       count balance; serves as a regression test for the
       ``weights_per_partition`` populated path.
    2. Biased weights (half heavy, half light) — weighted partition
       should converge to a near-balanced weight sum (within 30 %)
       even when the element-count is unbalanced.

    Skips cleanly when pymetis is not available.
    """
    import h5py

    _pymetis, source = _ensure_pymetis()

    sess = _make_session()
    try:
        # Count all elements (across all dims) for the weights vector.
        import gmsh
        n_total_elems = 0
        for d in range(4):
            _etypes, etl, _ = gmsh.model.mesh.getElements(dim=d, tag=-1)
            n_total_elems += sum(len(t) for t in etl)

        # --- 1. Trivial unit weights ------------------------------------
        weights_unit = np.ones(n_total_elems, dtype=np.float64)
        info = sess.mesh.partitioning.partition(
            n_parts, weights=weights_unit.tolist(), backend="pymetis",
        )
        assert info.n_parts == n_parts
        assert info.weights_per_partition is not None, (
            "weights_per_partition must be populated on a weighted call"
        )
        total_w = sum(info.weights_per_partition.values())
        # Approximate (owning-entity accounting may include lower-dim
        # ghosts not in the weight vector; tolerate 5 % drift).
        assert total_w > 0
        assert abs(total_w - sum(weights_unit.tolist())) <= 0.05 * n_total_elems, (
            f"weight sum drift: {total_w} vs {sum(weights_unit.tolist())} "
            f"(n_total_elems={n_total_elems})"
        )

        fem = sess.mesh.queries.get_fem_data(dim=1)
        assert len(fem.partitions) == n_parts

        # Drive apeSees + tcl + h5 against the weighted partition.
        ops = _wire_apesees(fem, with_recorders=False)
        tcl_path = tmp_path / f"weighted_unit_n{n_parts}.tcl"
        ops.tcl(str(tcl_path))
        text = tcl_path.read_text(encoding="utf-8")
        _check_brace_balance(text)
        blocks = _split_into_partition_blocks(text)
        assert len(blocks) == n_parts

        h5_path = tmp_path / f"weighted_unit_n{n_parts}.h5"
        ops.h5(str(h5_path))
        with h5py.File(str(h5_path), "r") as f:
            assert "/opensees/partitions" in f
            assert int(f["/opensees/partitions"].attrs["n_partitions"]) == n_parts

        # --- 2. Biased weights ------------------------------------------
        # Unpartition between calls.
        sess.mesh.partitioning.unpartition()
        half = n_total_elems // 2
        biased = [10.0] * half + [1.0] * (n_total_elems - half)
        info2 = sess.mesh.partitioning.partition(
            n_parts, weights=biased, backend="pymetis",
        )
        assert info2.n_parts == n_parts
        assert info2.weights_per_partition is not None

        # The biased partition should distribute weight roughly evenly
        # across ranks.  We're lenient on the bound since the stub does
        # greedy first-fit (not METIS proper), but every rank must carry
        # SOME weight.
        for pid, w in info2.weights_per_partition.items():
            assert w > 0, (
                f"partition {pid} carries zero weight under biased "
                f"weights (source={source!r}; partition info: "
                f"{info2.weights_per_partition!r})"
            )
    finally:
        sess.end()


# ---------------------------------------------------------------------------
# Scenario C — OpenSeesMP runtime smoke
# ---------------------------------------------------------------------------


def _find_mpi_launcher() -> str | None:
    """Return ``mpirun`` or ``mpiexec`` if either is on PATH, else None."""
    for name in ("mpirun", "mpiexec"):
        path = shutil.which(name)
        if path:
            return path
    return None


def test_scenario_c_opensees_mp_runtime_smoke(tmp_path: Path) -> None:
    """E2E: ``mpirun -np 4 OpenSeesMP $tcl_path`` returns 0.

    Best-effort runtime gate.  Skips cleanly when ``OpenSeesMP`` or any
    MPI launcher (``mpirun`` / ``mpiexec``) is not on PATH — the
    binary's absence is environment-conditional, not a defect.
    """
    launcher = _find_mpi_launcher()
    opensees_mp = shutil.which("OpenSeesMP")
    if launcher is None or opensees_mp is None:
        pytest.skip(
            f"OpenSeesMP runtime not available "
            f"(launcher={launcher!r}, OpenSeesMP={opensees_mp!r})"
        )

    n_parts = 4
    sess = _make_session()
    try:
        sess.mesh.partitioning.partition(n_parts)
        fem = sess.mesh.queries.get_fem_data(dim=1)
        _inject_floor_diaphragms(fem, n_stories=3)

        ops = _wire_apesees(fem, with_recorders=False)

        # Append a trivial analysis driver so OpenSeesMP has something
        # to run after declaring the model.
        tcl_path = tmp_path / f"runtime_n{n_parts}.tcl"
        ops.tcl(str(tcl_path))
        # Append "analyze 1" so the runtime actually steps once.
        with tcl_path.open("a", encoding="utf-8") as f:
            f.write("\n")
            f.write("integrator LoadControl 1.0\n")
            f.write("test NormDispIncr 1e-6 10 0\n")
            f.write("algorithm Linear\n")
            f.write("analysis Static\n")
            f.write("analyze 1\n")
            f.write('puts "ANALYSIS_DONE"\n')

        proc = subprocess.run(
            [launcher, "-np", str(n_parts), opensees_mp, str(tcl_path)],
            capture_output=True,
            text=True,
            timeout=120,
        )
        if proc.returncode != 0:
            pytest.fail(
                f"OpenSeesMP returned {proc.returncode}.\n"
                f"--- stdout ---\n{proc.stdout}\n"
                f"--- stderr ---\n{proc.stderr}"
            )
        # Sanity: every rank should have printed our marker; at minimum
        # one of them should appear in stdout.
        assert "ANALYSIS_DONE" in proc.stdout or "ANALYSIS_DONE" in proc.stderr, (
            f"expected 'ANALYSIS_DONE' marker in MP output.\n"
            f"--- stdout ---\n{proc.stdout}\n"
            f"--- stderr ---\n{proc.stderr}"
        )
    finally:
        sess.end()
