"""#2502: finite-strain wrappers over a SANISAND inner.

WIP probe (live-fork lane): which (wrapper chain x inner) pairs end the
fork process at FORK_PIN?  Every case runs in a child interpreter, so an
abort fails one case instead of ending the lane's pytest process.  The
child prints a marker after each stage (declared / element / analyzed),
so the parent sees where the process ended.
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

_ROOT = Path(__file__).resolve().parents[3]
_STAGE = "FWMD_STAGE:"

_GORINI: dict[str, float] = {
    "G0": 264.32, "nu": 0.3129, "e_init": 0.6944, "Mc": 1.33090, "c": 0.71,
    "lambda_c": 0.027, "e0": 0.83, "ksi": 0.45, "P_atm": 101.0, "m": 0.005,
    "h0": 1.3, "Ch": 0.968, "nb": 3.5, "A0": 0.05, "nd": 5.75,
    "rho": 2.0,
}

_QUAD4 = [(0, 0), (1, 0), (1, 1), (0, 1)]
_HEX8 = [(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0),
         (0, 0, 1), (1, 0, 1), (1, 1, 1), (0, 1, 1)]

#: chain id -> (host token, ndm, host argv tail, wrapper tokens applied
#: innermost first; the element carries the outermost).
_CHAINS: dict[str, tuple[str, int, list[Any], list[str]]] = {
    "LogStrain": ("LadrunoBrick", 3, ["-geom", "finite"], ["LogStrain"]),
    "LogStrain2D": ("LadrunoQuad", 2, ["-geom", "finite", "-thick", 1.0],
                    ["LogStrain2D"]),
    "InitDefGrad-LogStrain": ("LadrunoBrick", 3, ["-geom", "finite"],
                              ["LogStrain", "InitDefGrad"]),
    "InitDefGrad": ("LadrunoBrick", 3, ["-geom", "finite"], ["InitDefGrad"]),
    "LogStrain-StagedStrain": ("LadrunoBrick", 3, ["-geom", "finite"],
                               ["StagedStrain", "LogStrain"]),
    "LogStrain2D-StagedStrain": ("LadrunoQuad", 2,
                                 ["-geom", "finite", "-thick", 1.0],
                                 ["StagedStrain", "LogStrain2D"]),
    "LogStrain2D-PlaneStrain": ("LadrunoQuad", 2,
                                ["-geom", "finite", "-thick", 1.0],
                                ["PlaneStrain", "LogStrain2D"]),
}

_LEAVES = ("ManzariDafalias", "LadrunoSANISAND", "SAniSandMS",
           "ElasticIsotropic")


def _leaf_argv(leaf: str) -> tuple[str, list[Any]]:
    """The token and argv the bridge emits for ``leaf`` (its own _emit)."""
    from apeGmsh.opensees.material import nd

    if leaf == "ManzariDafalias":
        mat: Any = nd.ManzariDafalias(**_GORINI, z_max=12.5, cz=1100.0)
    elif leaf == "LadrunoSANISAND":
        mat = nd.LadrunoSANISAND(**_GORINI, z_max=12.5, cz=1100.0)
    elif leaf == "SAniSandMS":
        mat = nd.SAniSandMS(**_GORINI, zeta=0.0005, mu0=0.26, beta=1.0)
    else:
        mat = nd.ElasticIsotropic(E=2.0e5, nu=0.25)
    seen: list[tuple[str, list[Any]]] = []

    class _Rec:
        def nDMaterial(self, token: str, tag: int, *args: Any) -> None:
            seen.append((token, list(args)))

    mat._emit(_Rec(), 1)
    assert len(seen) == 1, seen
    return seen[0]


def child_main(leaf: str, chain_id: str) -> None:
    """Declare the chain, build one element, run it; print each stage."""
    from apeGmsh.opensees.emitter.live import get_backend_info, get_ops

    def stage(name: str) -> None:
        print(_STAGE + name, flush=True)
        sys.stdout.flush()

    ops = get_ops()
    stage("build=" + str(get_backend_info().build))
    token, ndm, tail, chain = _CHAINS[chain_id]
    leaf_token, leaf_args = _leaf_argv(leaf)
    ops.wipe()
    ops.model("basic", "-ndm", ndm, "-ndf", ndm)
    ops.nDMaterial(leaf_token, 1, *leaf_args)
    stage("leaf")
    inner = 1
    for depth, wtoken in enumerate(chain, start=1):
        outer = 10 * depth + 1
        ops.nDMaterial(wtoken, outer, inner)
        inner = outer
        stage("declared:" + wtoken)
    stage("declared")
    coords = _HEX8 if ndm == 3 else _QUAD4
    eps = (-1e-3, -1.5e-3, 0.0)[:ndm]
    ops.timeSeries("Linear", 1)
    ops.pattern("Plain", 1, 1)
    nodes = []
    for i, c in enumerate(coords):
        xyz = [float(v) for v in c]
        ops.node(i + 1, *xyz)
        nodes.append(i + 1)
        for d in range(ndm):
            ops.sp(i + 1, d + 1, eps[d] * xyz[d])
    ops.element(token, 1, *nodes, inner, *tail)
    stage("element")
    ops.constraints("Penalty", 1.0e12, 1.0e12)
    ops.numberer("Plain")
    ops.system("FullGeneral")
    ops.test("NormDispIncr", 1e-10, 25, 0)
    ops.algorithm("Newton")
    ops.integrator("LoadControl", 0.25)
    ops.analysis("Static")
    rc = ops.analyze(4)
    stage(f"analyzed:rc={rc}")
    ops.wipe()


def _in_child(leaf: str, chain_id: str) -> tuple[int, list[str], str]:
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        [str(_ROOT / "src"), str(_ROOT)]
        + ([env["PYTHONPATH"]] if env.get("PYTHONPATH") else []))
    env.setdefault("LADRUNO_OPENSEES_QUIET", "1")
    code = (
        "from tests.opensees.unit import test_finite_wrapper_over_md_refused as m\n"
        f"m.child_main({leaf!r}, {chain_id!r})\n"
    )
    done = subprocess.run(
        [sys.executable, "-c", code], cwd=str(_ROOT), env=env,
        capture_output=True, text=True, timeout=300,
    )
    stages = [ln[len(_STAGE):] for ln in done.stdout.splitlines()
              if ln.startswith(_STAGE)]
    tail = done.stderr.strip().splitlines()[-3:] if done.stderr else []
    return done.returncode, stages, " | ".join(tail)[-400:]


@pytest.mark.live
@pytest.mark.ladruno_fork
def test_probe_finite_wrapper_over_sanisand_at_pin() -> None:
    """WIP: one table of outcomes; fails on purpose so the lane prints it."""
    rows = []
    for leaf in _LEAVES:
        for chain_id in _CHAINS:
            rc, stages, tail = _in_child(leaf, chain_id)
            rows.append(f"{leaf:17s} {chain_id:26s} rc={rc:<4d} "
                        f"last={stages[-1] if stages else '-'!s:22s} "
                        f"stages={stages} err={tail}")
    pytest.fail("PROBE #2502\n" + "\n".join(rows))
