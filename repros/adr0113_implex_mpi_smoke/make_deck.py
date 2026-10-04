"""ADR 0113 -- 2-rank OpenSeesMP smoke deck for the IMPL-EX time driver.

Writes ``implex_smoke.tcl`` next to this file: a partitioned, staged
bridge deck (``ops.implex_time()``) of six fiber columns, three per rank,
each column two forceBeamColumn elements with fixed base; per rank two
columns are IMPL-EX ASDConcrete1D (targets) and one is elastic (not a
target).  Stages: gravity (LoadControl 0.1 x 10), hold (LoadControl
0.02 x 5), transient (Newmark, dt 0.01 x 5).

The ONLY text added to the bridge's deck is a probe after each stage's
analyze loop (before its ``loadConst``): every rank prints, for each
IMPL-EX target it holds, ``eleResponse $e section 1 fiber 0 time`` --
ASDConcrete1D's ``dTime dTimeCommit dTimeInitial`` (response 4000).  After
the stage's steps all three must equal the stage's increment.
``check_output.py`` reads the run's stdout and checks that.

Run from the worktree root (no OpenSees needed to generate):

    PYTHONPATH=src:. python repros/adr0113_implex_mpi_smoke/make_deck.py
"""
from __future__ import annotations

import json
from pathlib import Path

from apeGmsh.opensees.apesees import apeSees
from apeGmsh.opensees.emitter.tcl import TclEmitter
from apeGmsh.opensees.section.fiber import FiberPoint

from tests.opensees.fixtures.fem_stub import (
    FEMStub,
    _ElementGroupView,
    _ElementsStub,
    _NodesStub,
)

HERE = Path(__file__).resolve().parent
STAGES = (("gravity", 0.1), ("hold", 0.02), ("transient", 0.01))

# columns: x position -> (rank, implex?)
COLUMNS = {0.0: (0, True), 1.0: (0, True), 2.0: (0, False),
           3.0: (1, True), 4.0: (1, True), 5.0: (1, False)}


def build_fem() -> tuple[FEMStub, dict[str, list[int]]]:
    ids, coords, base, top = [], [], [], []
    groups: dict[str, tuple[list[int], list[tuple[int, int]]]] = {
        "implex": ([], []), "elastic": ([], []),
    }
    parts: dict[int, tuple[list[int], list[int]]] = {0: ([], []), 1: ([], [])}
    nid = eid = 0
    for x, (rank, implex) in COLUMNS.items():
        col = []
        for z in (0.0, 1500.0, 3000.0):
            nid += 1
            ids.append(nid)
            coords.append((x * 1000.0, 0.0, z))
            col.append(nid)
            parts[rank][0].append(nid)
        base.append(col[0])
        top.append(col[2])
        g = groups["implex" if implex else "elastic"]
        for a, b in ((col[0], col[1]), (col[1], col[2])):
            eid += 1
            g[0].append(eid)
            g[1].append((a, b))
            parts[rank][1].append(eid)
    fem = FEMStub(
        nodes=_NodesStub(ids=ids, coords=coords,
                         node_pgs={"Base": base, "Top": top}),
        elements=_ElementsStub(elem_pgs={
            k: _ElementGroupView(ids=tuple(v[0]), connectivity=tuple(v[1]))
            for k, v in groups.items()
        }),
    )
    fem.set_partitions([(r + 1, n, e) for r, (n, e) in parts.items()])
    return fem, {"targets": groups["implex"][0], "top": top}


def _chain(ops: apeSees, integrator: object, analysis: object) -> dict:
    return {
        "test": ops.test.NormDispIncr(tol=1e-8, max_iter=50),
        "algorithm": ops.algorithm.Newton(),
        "integrator": integrator,
        "constraints": ops.constraints.Plain(),
        "numberer": ops.numberer.ParallelPlain(),
        "system": ops.system.Mumps(),
        "analysis": analysis,
    }


def build_deck() -> tuple[list[str], dict[str, list[int]]]:
    fem, info = build_fem()
    ops = apeSees(fem, default_orientation=None, element_tags="fem")
    ops.model(ndm=3, ndf=6)
    conc = ops.uniaxialMaterial.ASDConcrete1D(
        E=30000.0, fc=30.0, implex=True, lch_ref=500.0)
    elas = ops.uniaxialMaterial.ElasticMaterial(E=30000.0)
    fibers = [(y, z) for y in (-150.0, 150.0) for z in (-150.0, 150.0)]
    sec_c = ops.section.Fiber(fibers=tuple(
        FiberPoint(material=conc, y=y, z=z, area=22500.0) for y, z in fibers
    ), GJ=1.0e12)
    sec_e = ops.section.Fiber(fibers=tuple(
        FiberPoint(material=elas, y=y, z=z, area=22500.0) for y, z in fibers
    ), GJ=1.0e12)
    tr = ops.geomTransf.Linear(vecxz=(1.0, 0.0, 0.0))
    ops.element.forceBeamColumn(
        pg="implex", transf=tr,
        integration=ops.beamIntegration.Lobatto(section=sec_c, n_ip=3))
    ops.element.forceBeamColumn(
        pg="elastic", transf=tr,
        integration=ops.beamIntegration.Lobatto(section=sec_e, n_ip=3))
    ops.fix(pg="Base", dofs=(1, 1, 1, 1, 1, 1))
    ops.mass(pg="Top", values=(1.0, 1.0, 1.0, 0.0, 0.0, 0.0))
    ops.implex_time()

    with ops.stage(name="gravity") as s:
        with s.pattern(series=ops.timeSeries.Linear()) as p:
            for n in info["top"]:
                p.load(node=n, forces=(0.0, 0.0, -1.0e6, 0.0, 0.0, 0.0))
        s.analysis(**_chain(ops, ops.integrator.LoadControl(dlam=0.1),
                            ops.analysis.Static()))
        s.run(n_increments=10)
    with ops.stage(name="hold") as s:
        s.analysis(**_chain(ops, ops.integrator.LoadControl(dlam=0.02),
                            ops.analysis.Static()))
        s.run(n_increments=5)
    with ops.stage(name="transient") as s:
        s.analysis(**_chain(ops, ops.integrator.Newmark(gamma=0.5, beta=0.25),
                            ops.analysis.Transient()))
        s.run(n_increments=5, dt=0.01)

    em = TclEmitter()
    ops.build().emit(em)
    return em.lines(), info


def add_probes(lines: list[str], targets: list[int]) -> list[str]:
    tl = " ".join(map(str, targets))
    out: list[str] = []
    stage = None
    for s in lines:
        if s.startswith("# === Stage: "):
            stage = s[len("# === Stage: "):].rstrip(" =")
        if s == "loadConst -time 0.0" and stage is not None:
            out += [
                f"# ---- ADR 0113 smoke probe (not bridge output): stage {stage}",
                "foreach _p_e [getEleTags] {",
                f"    if {{[lsearch -exact {{{tl}}} $_p_e] >= 0}} {{",
                f'        puts "IMPLEX_PROBE stage={stage} rank=[getPID] '
                f'ele=$_p_e [eleResponse $_p_e section 1 fiber 0 time]"',
                "    }",
                "}",
            ]
            stage = None
        out.append(s)
    return out


def main() -> None:
    lines, info = build_deck()
    lines = add_probes(lines, info["targets"])
    (HERE / "implex_smoke.tcl").write_text("\n".join(lines) + "\n",
                                           encoding="utf-8")
    expected = {
        "targets": info["targets"],
        "targets_by_rank": {
            str(r): [e for e in info["targets"]
                     if COLUMNS[list(COLUMNS)[(e - 1) // 2]][0] == r]
            for r in (0, 1)
        },
        "stages": dict(STAGES),
    }
    (HERE / "expected.json").write_text(json.dumps(expected, indent=2) + "\n",
                                        encoding="utf-8")
    print(f"wrote {HERE / 'implex_smoke.tcl'} ({len(lines)} lines)")


if __name__ == "__main__":
    main()
