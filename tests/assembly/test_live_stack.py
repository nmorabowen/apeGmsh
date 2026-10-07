"""ADR 0117 INV-8 (AS1): two stacked instances of one block file, solved live.

Two identical blocks (``nu = 0``) stacked by an assembly-level
``tie(enforce="equation")`` are two springs in series, each ``EA/H``, so

    K = EA / (2 H)

exactly: uniform axial strain is in every mesh's span and the interface
meshes coincide, so the only tolerance is solver precision. The tie is
also checked row by row: every equation row holds to round-off in the
solved state. A tie on the wrong face or instance reads K off by a whole
block; a penalty default sneaking in reads soft.

Each solve runs in a fresh interpreter: stock ``wipe()`` keeps
``equationConstraint`` rows (see ``tests/test_meshable_part_route.py``),
so a tied model in the shared pytest process would poison later live tests.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from tests.assembly.test_two_instances_one_tie import (
    E,
    H,
    SIDE,
    block_fem,
    write_instance,
)

DELTA = 0.01                         # prescribed shortening, mm
K_EXACT = E * SIDE * SIDE / (2 * H)  # two blocks of EA/H in series


def _declare(ops) -> None:
    """The block's model content with ``nu = 0``: the 1-D closed form."""
    ops.model(ndm=3, ndf=3)
    steel = ops.nDMaterial.ElasticIsotropic(E=E, nu=0.0, name="steel")
    ops.element.stdBrick(pg="Vol", material=steel)


def _solve(workdir: Path) -> dict:
    """Build the two-instance stack and solve it IN THIS PROCESS."""
    import numpy as np

    from apeGmsh.assembly import Assembly
    from apeGmsh.opensees.emitter.live import LiveOpsEmitter

    block = write_instance(workdir / "block.h5", block_fem(workdir), _declare)
    ops = (
        Assembly("stack")
        .instance("pier_1", block)
        .instance("pier_2", block, translate=(0.0, 0.0, H))
        .tie("pier_1.top", "pier_2.bot", enforce="equation", dofs=[1, 2, 3])
        .bridge(ndm=3, ndf=3)
    )
    # Analysis content is assembly-level (ADR 0117 D4).
    ops.fix(pg="pier_1.bot", dofs=(1, 1, 1))
    ts = ops.timeSeries.Linear()
    with ops.pattern.Plain(series=ts) as pat:
        pat.sp(pg="pier_2.top", dof=3, value=-DELTA)
    ops.constraints.Lagrange()
    ops.numberer.Plain()
    ops.system.FullGeneral()
    ops.test.NormDispIncr(tol=1e-10, max_iter=10)
    ops.algorithm.Linear()
    ops.integrator.LoadControl(dlam=1.0)
    ops.analysis.Static()

    emitter = LiveOpsEmitter(wipe=True)
    ops.build().emit(emitter)
    assert emitter.analyze(steps=1) == 0

    live = emitter.ops
    live.reactions()
    fem = ops.fem
    base = [int(t) for t in fem.nodes.select(pg="pier_1.bot").ids]
    r_z = sum(live.nodeReaction(t, 3) for t in base)
    residual, n_rows = 0.0, 0
    for rec in fem.elements.constraints:
        if rec.enforce != "equation":
            continue
        weights = np.asarray(rec.weights, dtype=float)
        masters = [int(m) for m in rec.master_nodes]
        for d in rec.dofs:
            n_rows += 1
            u_s = live.nodeDisp(int(rec.slave_node), int(d))
            u_m = sum(w * live.nodeDisp(m, int(d)) for w, m in zip(weights, masters))
            residual = max(residual, abs(u_s - u_m))
    return {"k": abs(r_z) / DELTA, "residual": residual, "rows": n_rows}


def _main() -> None:
    """``python -c`` entry: argv[1] is the work directory."""
    print("RESULT " + json.dumps(_solve(Path(sys.argv[1]))))


def _live_ops():
    from apeGmsh.opensees.emitter.live import _get_ops
    try:
        return _get_ops()
    except ImportError as e:
        pytest.skip(f"no OpenSees backend: {e}")


@pytest.mark.live
def test_two_stacked_instances_match_series_closed_form(tmp_path: Path):
    if not hasattr(_live_ops(), "equationConstraint"):
        pytest.skip("this OpenSees build predates equationConstraint "
                    "(upstream 2025-05-10, openseespy >= 3.8.0)")
    root = Path(__file__).resolve().parents[2]
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        [str(root), str(root / "src"), env.get("PYTHONPATH", "")])
    env.setdefault("LADRUNO_OPENSEES_QUIET", "1")
    proc = subprocess.run(
        [sys.executable, "-c",
         "from tests.assembly.test_live_stack import _main; _main()",
         str(tmp_path)],
        env=env, capture_output=True, text=True, encoding="utf-8",
        errors="replace", timeout=600,
    )
    lines = [ln for ln in proc.stdout.splitlines() if ln.startswith("RESULT ")]
    assert proc.returncode == 0 and lines, (
        f"stack subprocess failed (rc={proc.returncode}):\n"
        f"{proc.stdout[-2000:]}\n{proc.stderr[-2000:]}")
    res = json.loads(lines[-1][len("RESULT "):])
    assert res["rows"] == 27, res          # 9 interface nodes x 3 dofs
    assert res["residual"] < 1e-12, res
    assert res["k"] == pytest.approx(K_EXACT, rel=1e-9), (
        f"K = {res['k']!r} vs EA/(2H) = {K_EXACT!r}")
