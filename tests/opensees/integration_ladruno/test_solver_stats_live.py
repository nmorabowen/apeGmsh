"""ADR 0106 S5 — live acceptance of ``system Pardiso -stats`` against the
real Ladruno fork binary.

S1-S4 proved the parser and the wiring against fake children (a canned
string, or ``sys.executable`` running a throwaway script). This is the
first test to run the real ``OpenSees.exe`` and check apeGmsh's parsed
numbers against what the fork's own solver actually prints.

**Measured reference** (2026-09-08), fork build ``b52f8d83b`` (PR #814)::

    C:\\Users\\nmb\\Documents\\Github\\OpenSees\\.claude\\worktrees\\
        ladruno-sanisand-psi-yield-cd584c\\dist\\bin\\OpenSees.exe

Deck: the fork's own ``tests/test_pardiso_stats.py`` cantilever, node for
node — a 2x2x2 ``stdBrick`` mesh (``NX=2``, element edge ``LEL=100``),
``ElasticIsotropic(E=200000, nu=0.3)``, the bottom face (``z=0``, 9 nodes)
fully fixed, the top face (``z=200``, 9 nodes) loaded ``(1e4, 0, -1e4)``
per node, ``constraints Plain``, ``numberer RCM`` — giving the fork's own
``n=54`` free-DOF equation count. The fork's suite only asserts a
*positive integer* for ``factor entries iparm(18)``
(``test_stats_block_present_with_flag``); the full block below is
MEASURED against the exe above, not asserted anywhere upstream:

    PARDISO stats: n=54 nnz(A)=1764 matrixType=11 threads=16
      factor entries iparm(18)  = 1836
      peak memory KB iparm(15)  = 252
      perm memory KB iparm(16)  = 215
      fact memory KB iparm(17)  = 31
      factor Mflops  iparm(19)  = 0

Test 1 reproduces the fork's own analysis chain exactly (``algorithm
Newton``, one ``LoadControl`` step) to pin that block. Also MEASURED: this
exact linear-elastic model takes exactly 2 refactorisations per step under
Newton to satisfy ``NormDispIncr 1e-10`` (the tangent never changes, but
OpenSees still re-solves once more to confirm the residual) — a second
magic number, on top of 1836, that a reader would have to trust. Tests 2-4
sidestep that by using ``algorithm Linear`` instead: mathematically
equivalent on this elastic material (no residual iteration at all), and
it refactorises exactly once per ``analyze(1)`` call BY CONSTRUCTION, not
by a second measurement. This was checked, not assumed: 3 Linear steps
against the same exe printed exactly 3 blocks, each reporting the same
n=54 / entries=1836 as the Newton run.

Gate: ``pytest.mark.ladruno_fork`` (root ``conftest.py`` auto-skips unless
the live backend resolves to the fork). The subprocess lane additionally
needs the Tcl ``OpenSees(.exe)`` binary, resolved from the SAME
``APEGMSH_OPENSEES_BIN`` directory the live backend uses (the fork's
``dist\\bin`` co-locates ``opensees.pyd`` and ``OpenSees.exe``) — see
:func:`_resolve_fork_exe`.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import cast

import pytest

from apeGmsh.opensees import apeSees
from apeGmsh.opensees._solver_stats import RunSolverStats

from tests.opensees.fixtures.fem_stub import (
    FEMStub,
    _ElementGroupView,
    _ElementsStub,
    _NodesStub,
)

pytestmark = pytest.mark.ladruno_fork

NX = 2
LEL = 100.0
E = 200000.0
NU = 0.3


def _nid(i: int, j: int, k: int) -> int:
    return 1 + i + (NX + 1) * (j + (NX + 1) * k)


def _build_54dof_brick() -> tuple[FEMStub, list[int], list[int]]:
    """The fork's ``tests/test_pardiso_stats.py`` cantilever, node for node.

    Returns ``(fem, bottom_node_ids, top_node_ids)`` — bottom (``z=0``) is
    fully fixed, top (``z=NX*LEL``) is where the fork applies its nodal
    load.
    """
    ids: list[int] = []
    coords: list[tuple[float, float, float]] = []
    for k in range(NX + 1):
        for j in range(NX + 1):
            for i in range(NX + 1):
                ids.append(_nid(i, j, k))
                coords.append((i * LEL, j * LEL, k * LEL))

    conn: list[tuple[int, ...]] = []
    elem_ids: list[int] = []
    tag = 1
    for k in range(NX):
        for j in range(NX):
            for i in range(NX):
                conn.append((
                    _nid(i, j, k), _nid(i + 1, j, k),
                    _nid(i + 1, j + 1, k), _nid(i, j + 1, k),
                    _nid(i, j, k + 1), _nid(i + 1, j, k + 1),
                    _nid(i + 1, j + 1, k + 1), _nid(i, j + 1, k + 1),
                ))
                elem_ids.append(tag)
                tag += 1

    nodes = _NodesStub(ids=ids, coords=coords, node_pgs={})
    elements = _ElementsStub(elem_pgs={
        "Body": _ElementGroupView(
            ids=tuple(elem_ids), connectivity=tuple(conn),
        ),
    })
    fem = FEMStub(nodes=nodes, elements=elements)

    bottom = [_nid(i, j, 0) for j in range(NX + 1) for i in range(NX + 1)]
    top = [_nid(i, j, NX) for j in range(NX + 1) for i in range(NX + 1)]
    return fem, bottom, top


def _new_bridge() -> tuple[apeSees, list[int]]:
    """A fresh bridge over the 54-DOF brick: model, material, elements,
    fixities declared; caller adds the load pattern + analysis chain."""
    fem, bottom, top = _build_54dof_brick()
    ops = apeSees(cast("object", fem))  # type: ignore[arg-type]
    ops.model(ndm=3, ndf=3)
    mat = ops.nDMaterial.ElasticIsotropic(E=E, nu=NU)
    ops.element.stdBrick(pg="Body", material=mat)
    ops.fix(nodes=bottom, dofs=(1, 1, 1))
    return ops, top


def _resolve_fork_exe() -> str:
    """The Tcl subprocess binary alongside the fork's ``opensees.pyd``.

    Same env var the ``ladruno_fork`` marker's ``conftest.py`` gate uses
    to resolve the LIVE backend (``APEGMSH_OPENSEES_BIN`` names the
    fork's ``dist\\bin``, which co-locates ``opensees.pyd`` and
    ``OpenSees.exe``). That gate can pass without this variable set (the
    fork on ``PYTHONPATH`` also resolves it) — this test additionally
    needs the EXE for the subprocess lane (ADR 0106 D3), so it skips on
    its own when the variable is unset or does not name a directory
    holding the binary.
    """
    bin_dir = os.environ.get("APEGMSH_OPENSEES_BIN")
    if bin_dir:
        exe_name = "OpenSees.exe" if os.name == "nt" else "OpenSees"
        candidate = os.path.join(bin_dir, exe_name)
        if os.path.isfile(candidate):
            return candidate
    pytest.skip(
        "APEGMSH_OPENSEES_BIN is not set to a dist/bin containing the "
        "OpenSees Tcl binary -- ADR 0106 S5 needs it for the subprocess "
        "lane (D3), separately from the live backend the ladruno_fork "
        "marker gates on."
    )
    raise AssertionError("unreachable")  # pragma: no cover


def _log_path(deck_path: str) -> Path:
    return Path(os.path.splitext(deck_path)[0] + ".log")


# ---------------------------------------------------------------------------
# 1. First factorisation matches the fork's own reference model
# ---------------------------------------------------------------------------


def test_first_factorisation_matches_fork_reference(tmp_path: Path) -> None:
    """The fork's exact chain (Newton, one LoadControl step): n=54,
    factor entries iparm(18)=1836, plus the rest of the measured block."""
    ops, top = _new_bridge()
    ts = ops.timeSeries.Linear()
    with ops.pattern.Plain(series=ts) as p:
        for nid in top:
            p.load(node=nid, forces=(1.0e4, 0.0, -1.0e4))
    ops.constraints.Plain()
    ops.numberer.RCM()
    ops.system.Pardiso(stats=True)
    ops.test.NormDispIncr(tol=1e-10, max_iter=10)
    ops.algorithm.Newton()
    ops.integrator.LoadControl(dlam=1.0)
    ops.analysis.Static()

    rec = ops.tcl(
        str(tmp_path / "brick54_newton.tcl"),
        run=True, bin=_resolve_fork_exe(), analyze_steps=1,
    )
    assert isinstance(rec, RunSolverStats)
    assert rec.stages == ()  # flat deck: everything lands in run_level

    block = rec.run_level
    assert block.n == 54
    assert block.max_factor_entries == 1836
    # The rest of the measured block (see module docstring) -- pinned so
    # a silent fork-side regression in the OTHER four numbers is caught
    # too, not just the fill.
    assert block.nnz_a == 1764
    assert block.matrix_type == 11
    assert block.max_peak_memory_kb == 252
    assert block.max_perm_memory_kb == 215
    assert block.max_fact_memory_kb == 31


# ---------------------------------------------------------------------------
# 2. factorisations is exact, not >= 1
# ---------------------------------------------------------------------------


def test_factorisations_counts_exactly_one_per_linear_step(
    tmp_path: Path,
) -> None:
    """``algorithm Linear`` refactorises exactly once per ``analyze(1)``
    call by construction (no residual iteration) -- 3 steps, 3 blocks."""
    n_steps = 3
    ops, top = _new_bridge()
    ts = ops.timeSeries.Linear()
    with ops.pattern.Plain(series=ts) as p:
        for nid in top:
            p.load(node=nid, forces=(1.0e4, 0.0, -1.0e4))
    ops.constraints.Plain()
    ops.numberer.RCM()
    ops.system.Pardiso(stats=True)
    ops.test.NormDispIncr(tol=1e-10, max_iter=10)
    ops.algorithm.Linear()
    ops.integrator.LoadControl(dlam=1.0 / n_steps)
    ops.analysis.Static()

    rec = ops.tcl(
        str(tmp_path / "brick54_linear.tcl"),
        run=True, bin=_resolve_fork_exe(), analyze_steps=n_steps,
    )
    assert isinstance(rec, RunSolverStats)
    assert rec.factorisations == n_steps
    assert rec.run_level.factorisations == n_steps
    assert rec.malformed_blocks == 0


# ---------------------------------------------------------------------------
# 3. Two-stage split sums to the total; run_level is empty
# ---------------------------------------------------------------------------


def test_two_stage_split_sums_to_the_total(tmp_path: Path) -> None:
    """A ``gravity`` stage (1 step) + a ``push`` stage (2 steps), both
    ``algorithm Linear`` -- per-stage counts are exact and every block
    is attributed to a stage (INV-4): ``run_level.factorisations == 0``.
    """
    ops, top = _new_bridge()

    with ops.stage(name="gravity") as s:
        with s.pattern(series=ops.timeSeries.Linear()) as p:
            for nid in top:
                p.load(node=nid, forces=(0.0, 0.0, -1.0e4))
        s.analysis(
            test=ops.test.NormDispIncr(tol=1e-10, max_iter=10),
            algorithm=ops.algorithm.Linear(),
            integrator=ops.integrator.LoadControl(dlam=1.0),
            constraints=ops.constraints.Plain(),
            numberer=ops.numberer.RCM(),
            system=ops.system.Pardiso(stats=True),
            analysis=ops.analysis.Static(),
        )
        s.run(n_increments=1)

    with ops.stage(name="push") as s:
        with s.pattern(series=ops.timeSeries.Linear()) as p:
            for nid in top:
                p.load(node=nid, forces=(5.0e3, 0.0, 0.0))
        s.analysis(
            test=ops.test.NormDispIncr(tol=1e-10, max_iter=10),
            algorithm=ops.algorithm.Linear(),
            integrator=ops.integrator.LoadControl(dlam=0.5),
            constraints=ops.constraints.Plain(),
            numberer=ops.numberer.RCM(),
            system=ops.system.Pardiso(stats=True),
            analysis=ops.analysis.Static(),
        )
        s.run(n_increments=2)

    rec = ops.tcl(
        str(tmp_path / "brick54_staged.tcl"),
        run=True, bin=_resolve_fork_exe(),
    )
    assert isinstance(rec, RunSolverStats)

    by_name = {stage.name: stage for stage in rec.stages}
    assert set(by_name) == {"gravity", "push"}
    assert by_name["gravity"].factorisations == 1
    assert by_name["push"].factorisations == 2

    assert rec.run_level.factorisations == 0
    assert (
        sum(stage.factorisations for stage in rec.stages)
        + rec.run_level.factorisations
        == rec.factorisations
    )
    assert rec.factorisations == 3


# ---------------------------------------------------------------------------
# 4. INV-1 at the live level: no request, no block, ``None`` back
# ---------------------------------------------------------------------------


def test_without_stats_returns_none_and_log_has_no_block(
    tmp_path: Path,
) -> None:
    ops, top = _new_bridge()
    ts = ops.timeSeries.Linear()
    with ops.pattern.Plain(series=ts) as p:
        for nid in top:
            p.load(node=nid, forces=(1.0e4, 0.0, -1.0e4))
    ops.constraints.Plain()
    ops.numberer.RCM()
    ops.system.Pardiso()  # stats defaults to False
    ops.test.NormDispIncr(tol=1e-10, max_iter=10)
    ops.algorithm.Newton()
    ops.integrator.LoadControl(dlam=1.0)
    ops.analysis.Static()

    deck = str(tmp_path / "brick54_nostats.tcl")
    rec = ops.tcl(deck, run=True, bin=_resolve_fork_exe(), analyze_steps=1)
    assert rec is None

    log_text = _log_path(deck).read_text(encoding="utf-8", errors="replace")
    assert "PARDISO stats:" not in log_text
