"""ADR 0117 D3 (AS4-a): the assembly couplings, solved live against closed forms.

* ``equal_dof`` — two stacked ``nu = 0`` blocks of ``EA/H`` each, joined by
  ``equal_dof`` on their coincident interface nodes, are two springs in
  series: ``K = EA / (2H)``, the value ``tie(enforce="equation")`` gives in
  ``test_live_stack.py``.
* ``rigid_diaphragm`` — two plate walls stacked in the plane ``y = 0`` with a
  gap, each tied by its own diaphragm to one reference node, move as one
  in-plane rigid body when the reference moves by ``(dx, dz, theta)``:
  ``u = dx + theta (z - z_cm)``, ``w = dz - theta (x - x_cm)``,
  ``ry = theta`` at every node of both walls; out of plane nothing moves.
* ``couple(kind="kinematic")`` (RBE2) — a load ``P`` on a reference node
  above a ``nu = 0`` block fixed at its base moves the reference and every
  top node by the rigid-body value ``P H / (E A)``.
* ``couple(kind="distributing")`` (RBE3) — the base reactions of the block
  sum to minus the load applied on the reference.

RBE2 and RBE3 are Ladruno-fork elements: those two run on the fork
(``ladruno_fork``) and skip on stock. Each solve runs in a fresh
interpreter, as in ``test_live_stack.py``.
"""
from __future__ import annotations

import json
import math
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
    plate_fem,
    write_instance,
)

DELTA = 0.01                          # prescribed shortening, mm
K_EXACT = E * SIDE * SIDE / (2 * H)   # two blocks of EA/H in series
P = 1000.0                            # N, on the reference node
U_RIGID = P * H / (E * SIDE * SIDE)   # one block of EA/H
#: Reference-node motion of the diaphragm rig: (dx, dz, theta_y).
MOTION = (0.01, -0.02, 0.001)
WALL_DZ = SIDE + 2.0
CM = (5.0, 0.0, SIDE + 1.0)
REF = (SIDE / 2, SIDE / 2, H + 5.0)


def _declare_block(ops) -> None:
    """The block with ``nu = 0``: the 1-D closed form."""
    ops.model(ndm=3, ndf=3)
    steel = ops.nDMaterial.ElasticIsotropic(E=E, nu=0.0, name="steel")
    ops.element.stdBrick(pg="Vol", material=steel)


def _declare_wall(ops) -> None:
    ops.model(ndm=3, ndf=6)
    sec = ops.section.ElasticMembranePlateSection(E=E, nu=0.2, h=0.5, name="slab")
    ops.element.ShellMITC4(pg="Slab", section=sec)


def _static(ops, *, handler: str) -> None:
    getattr(ops.constraints, handler)()
    ops.numberer.Plain()
    ops.system.FullGeneral()
    ops.test.NormDispIncr(tol=1e-10, max_iter=10)
    ops.algorithm.Linear()
    ops.integrator.LoadControl(dlam=1.0)
    ops.analysis.Static()


def _run(ops):
    from apeGmsh.opensees.emitter.live import LiveOpsEmitter

    emitter = LiveOpsEmitter(wipe=True)
    ops.build().emit(emitter)
    assert emitter.analyze(steps=1) == 0
    return emitter.ops


def _ids(fem, **sel) -> list[int]:
    return sorted(int(i) for i in fem.nodes.select(**sel).ids)


def _solve_equal_dof(workdir: Path) -> dict:
    from apeGmsh.assembly import Assembly

    block = write_instance(workdir / "block.h5", block_fem(workdir), _declare_block)
    ops = (Assembly("stack")
           .instance("pier_1", block)
           .instance("pier_2", block, translate=(0.0, 0.0, H))
           .equal_dof("pier_1.top", "pier_2.bot", dofs=[1, 2, 3])
           .bridge(ndm=3, ndf=3))
    ops.fix(pg="pier_1.bot", dofs=(1, 1, 1))
    ts = ops.timeSeries.Linear()
    with ops.pattern.Plain(series=ts) as pat:
        pat.sp(pg="pier_2.top", dof=3, value=-DELTA)
    _static(ops, handler="Transformation")
    live = _run(ops)
    live.reactions()
    r_z = sum(live.nodeReaction(t, 3) for t in _ids(ops.fem, pg="pier_1.bot"))
    return {"k": abs(r_z) / DELTA}


def _solve_rigid_diaphragm(workdir: Path) -> dict:
    import numpy as np

    from apeGmsh.assembly import Assembly

    plate = write_instance(workdir / "plate.h5", plate_fem(workdir), _declare_wall)
    stand = ((1.0, 0.0, 0.0), math.pi / 2)
    ops = (Assembly("walls")
           .instance("w1", plate, rotate=stand)
           .instance("w2", plate, rotate=stand, translate=(0.0, 0.0, WALL_DZ))
           .node("cm", CM)
           .rigid_diaphragm("cm", "w1.Slab", plane_normal=(0, 1, 0),
                            constrained_dofs=(1, 3, 5))
           .rigid_diaphragm("cm", "w2.Slab", plane_normal=(0, 1, 0),
                            constrained_dofs=(1, 3, 5))
           .bridge(ndm=3, ndf=6))
    fem = ops.fem
    (cm,) = _ids(fem, label="cm")
    coord = dict(zip((int(i) for i in fem.nodes.ids),
                     np.asarray(fem.nodes.coords, dtype=float)))
    walls = _ids(fem, pg="w1.Slab") + _ids(fem, pg="w2.Slab")
    # Out of plane each wall is a cantilever from its lower edge.
    for pg in ("w1.Slab", "w2.Slab"):
        ids = _ids(fem, pg=pg)
        z0 = min(coord[t][2] for t in ids)
        ops.fix(nodes=[t for t in ids if abs(coord[t][2] - z0) < 1e-9],
                dofs=(0, 1, 0, 1, 0, 1))
    ops.fix(nodes=[cm], dofs=(0, 1, 0, 1, 0, 1))
    dx, dz, theta = MOTION
    ts = ops.timeSeries.Linear()
    with ops.pattern.Plain(series=ts) as pat:
        pat.sp(node=cm, dof=1, value=dx)
        pat.sp(node=cm, dof=3, value=dz)
        pat.sp(node=cm, dof=5, value=theta)
    _static(ops, handler="Transformation")
    live = _run(ops)
    err = 0.0
    for t in walls:
        x, _, z = coord[t]
        want = {1: dx + theta * (z - CM[2]), 2: 0.0,
                3: dz - theta * (x - CM[0]), 5: theta}
        for dof, value in want.items():
            err = max(err, abs(live.nodeDisp(t, dof) - value))
    return {"err": err, "n": len(walls)}


def _block_with_reference(workdir: Path, kind: str, ref_load):
    from apeGmsh.assembly import Assembly

    block = write_instance(workdir / "block.h5", block_fem(workdir), _declare_block)
    ops = (Assembly("col")
           .instance("col", block)
           .node("ref", REF)
           .couple("col.top", kind=kind, reference="ref")
           .bridge(ndm=3, ndf=3))
    (ref,) = _ids(ops.fem, label="ref")
    ops.ndf(ref, ndf=6)
    ops.fix(pg="col.bot", dofs=(1, 1, 1))
    ts = ops.timeSeries.Linear()
    with ops.pattern.Plain(series=ts) as pat:
        pat.load(node=ref, forces=tuple(ref_load))
    _static(ops, handler="Plain")
    return ops, ref, _run(ops)


def _solve_kinematic(workdir: Path) -> dict:
    ops, ref, live = _block_with_reference(
        workdir, "kinematic", (0.0, 0.0, -P, 0.0, 0.0, 0.0))
    top = _ids(ops.fem, pg="col.top")
    return {"u_ref": live.nodeDisp(ref, 3),
            "u_top": [live.nodeDisp(t, 3) for t in top],
            "lateral": max(abs(live.nodeDisp(t, d)) for t in top for d in (1, 2))}


def _solve_distributing(workdir: Path) -> dict:
    load = (0.3 * P, 0.0, -P, 0.0, 0.0, 0.0)
    ops, _, live = _block_with_reference(workdir, "distributing", load)
    live.reactions()
    base = _ids(ops.fem, pg="col.bot")
    return {"sum": [sum(live.nodeReaction(t, d) for t in base) for d in (1, 2, 3)],
            "load": list(load[:3])}


_CASES = {
    "equal_dof": _solve_equal_dof,
    "rigid_diaphragm": _solve_rigid_diaphragm,
    "kinematic": _solve_kinematic,
    "distributing": _solve_distributing,
}


def _main() -> None:
    """``python -c`` entry: argv[1] is the case, argv[2] the work directory."""
    print("RESULT " + json.dumps(_CASES[sys.argv[1]](Path(sys.argv[2]))))


def _solve(case: str, workdir: Path) -> dict:
    root = Path(__file__).resolve().parents[2]
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        [str(root), str(root / "src"), env.get("PYTHONPATH", "")])
    env.setdefault("LADRUNO_OPENSEES_QUIET", "1")
    proc = subprocess.run(
        [sys.executable, "-W", "ignore::UserWarning", "-c",
         "from tests.assembly.test_live_couplings import _main; _main()",
         case, str(workdir)],
        env=env, capture_output=True, text=True, encoding="utf-8",
        errors="replace", timeout=600,
    )
    lines = [ln for ln in proc.stdout.splitlines() if ln.startswith("RESULT ")]
    assert proc.returncode == 0 and lines, (
        f"{case} subprocess failed (rc={proc.returncode}):\n"
        f"{proc.stdout[-2000:]}\n{proc.stderr[-2000:]}")
    return json.loads(lines[-1][len("RESULT "):])


def _live_ops():
    from apeGmsh.opensees.emitter.live import _get_ops
    try:
        return _get_ops()
    except ImportError as e:
        pytest.skip(f"no OpenSees backend: {e}")


@pytest.mark.live
def test_equal_dof_stack_matches_the_series_closed_form(tmp_path: Path):
    _live_ops()
    res = _solve("equal_dof", tmp_path)
    assert res["k"] == pytest.approx(K_EXACT, rel=1e-9), (
        f"K = {res['k']!r} vs EA/(2H) = {K_EXACT!r}")


@pytest.mark.live
def test_rigid_diaphragm_moves_two_walls_as_one_in_plane(tmp_path: Path):
    _live_ops()
    res = _solve("rigid_diaphragm", tmp_path)
    assert res["n"] == 18, res
    assert res["err"] < 1e-12, res


@pytest.mark.live
@pytest.mark.ladruno_fork
def test_kinematic_coupling_gives_the_rigid_body_displacement(tmp_path: Path):
    _live_ops()
    res = _solve("kinematic", tmp_path)
    assert res["u_ref"] == pytest.approx(-U_RIGID, rel=1e-6), res
    for u in res["u_top"]:
        assert u == pytest.approx(-U_RIGID, rel=1e-6), res
    assert res["lateral"] < 1e-9 * U_RIGID, res


@pytest.mark.live
@pytest.mark.ladruno_fork
def test_distributing_coupling_reactions_sum_to_the_load(tmp_path: Path):
    _live_ops()
    res = _solve("distributing", tmp_path)
    for r, f in zip(res["sum"], res["load"]):
        assert r == pytest.approx(-f, abs=1e-8 * P), res
