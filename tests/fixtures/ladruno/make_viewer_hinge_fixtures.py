"""Write the Ladruno hinge fixtures and their oracles from a real fork run.

    set PYTHONPATH=<checkout>/src
    python tests/fixtures/ladruno/make_viewer_hinge_fixtures.py

Needs the Ladruno fork (`import opensees`). For each of the 3-D and 2-D models
it writes, under `apeGmshViewer/fixtures/`, `ladruno_hinges<n>d.h5`, `.ladruno` (stamped against the model) and
`.oracle.json`: the live `eleResponse(tag, "hingeSummary")` of every hinged
element at every step, keyed by OpenSees tag (the file's element ids), which the viewer's reader is
checked against (`apeGmshViewer/test/hinges.test.ts`). It lives outside the app:
no Python under `apeGmshViewer/` imports apeGmsh (ADR 0112 D4).
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import gmsh
import h5py

from apeGmsh import apeGmsh
from apeGmsh.opensees import apeSees
from apeGmsh.opensees.emitter.live import LiveOpsEmitter
from apeGmsh.results.readers._ladruno_provenance import stamp_provenance

HERE = Path(__file__).resolve().parents[3] / "apeGmshViewer" / "fixtures"
L = 4.0
E, AREA, IZ, IY, G, J = 2.0e11, 0.04, 1.6e-4, 1.2e-4, 8.0e10, 2.0e-4
KH = 1.0e3 * 4.0 * E * IZ / L
FY, B = 50.0e3, 0.02
STEPS, DU = 20, 3.0e-3


def _fem(name: str, beams: tuple[tuple[str, float], ...]) -> Any:
    """Straight beams from their own fixed bases to one shared tip at ``(L, 0, 0)``."""
    with apeGmsh(model_name=name, verbose=False) as g:
        geo = g.model.geometry
        tip = geo.add_point(L, 0.0, 0.0)
        for pg, offset in beams:
            base = geo.add_point(0.0, offset, 0.0) if name.endswith("2d") else geo.add_point(0.0, 0.0, offset)
            line = geo.add_line(base, tip)
            gmsh.model.mesh.setTransfiniteCurve(line, 2)
            g.physical.add(1, [line], name=pg)
            g.labels.add(0, [base], f"base_{pg}")
        g.labels.add(0, [tip], "tip")
        g.mesh.generation.generate(dim=1)
        fem = g.mesh.queries.get_fem_data(dim=1)
    # The session saves beside this script; only the files under HERE are fixtures.
    for suffix in (".h5", ".geometry.h5"):
        (Path(__file__).resolve().parent / f"{name}{suffix}").unlink(missing_ok=True)
    return fem


def _analyse(name: str, fem: Any, ops: Any, pgs: tuple[str, ...], load: tuple[float, ...], hinged: tuple[str, ...], extra: dict[str, Any]) -> None:
    tip = ops.nodes.one(label="tip")
    with ops.pattern.Plain(series=ops.timeSeries.Linear()) as p:
        p.load(node=tip, forces=load)
    results = HERE / f"{name}.ladruno"
    model = HERE / f"{name}.h5"
    for path in (results, model):
        path.unlink(missing_ok=True)
    ops.recorder.Ladruno(file=results.name, nodal_responses=("displacement",), elem_responses=("hingeSummary", "localForce"))
    ops.constraints.Plain()
    ops.numberer.RCM()
    ops.system.BandGeneral()
    ops.test.NormDispIncr(tol=1e-9, max_iter=80)
    ops.algorithm.Newton()
    ops.integrator.DisplacementControl(node=int(tip.tag), dof=2, dU=DU)
    ops.analysis.Static()
    ops.h5(model.name)
    emitter = LiveOpsEmitter(wipe=True)
    ops.build().emit(emitter)
    with h5py.File(model, "r") as f:
        meta = f["/opensees/element_meta"]
        fem_to_tag = {int(e): int(t) for k in meta for t, e in zip(meta[k]["ids"][()], meta[k]["fem_eids"][()])}
    eids = {pg: int(fem.elements.select(pg=pg).ids[0]) for pg in pgs}
    tags = {pg: fem_to_tag[eids[pg]] for pg in pgs}
    live: dict[str, list[list[float]]] = {str(tags[pg]): [] for pg in hinged}
    for step in range(STEPS):
        assert emitter.analyze(steps=1) == 0, f"{name}: step {step} did not converge"
        for pg in hinged:
            live[str(tags[pg])].append([float(v) for v in emitter.ops.eleResponse(tags[pg], "hingeSummary")])
    emitter.ops.remove("recorders")
    emitter.ops.wipe()
    stamp_provenance(str(results), str(model))
    (HERE / f"{name}.geometry.h5").unlink(missing_ok=True)
    oracle = {"elements": tags, "femIds": eids, "steps": STEPS, "hingeSummary": live, **extra}
    (HERE / f"{name}.oracle.json").write_text(json.dumps(oracle, indent=1) + "\n", encoding="utf-8")


def three_d() -> None:
    beams = (("imk", -1.0), ("fbc", 0.0), ("dbc", 1.0), ("plain", 2.0))
    fem = _fem("ladruno_hinges3d", beams)
    ops = apeSees(fem, default_orientation=None)
    ops.model(ndm=3, ndf=6)
    tr = ops.geomTransf.Linear(vecxz=(0.0, 0.0, 1.0))
    sec = ops.section.Elastic(E=E, A=AREA, Iz=IZ, Iy=IY, G=G, J=J)
    integ = ops.beamIntegration.Lobatto(section=sec, n_ip=4)
    steel = ops.uniaxialMaterial.Steel01(fy=FY, E=KH, b=B)
    ops.element.LadrunoIMKBeam(pg="imk", transf=tr, A=AREA, E=E, Iz=IZ, G=G, Jx=J, Iy=IY, mat_z=steel, mat_yj=steel, end_zones=(0.2, 0.4))
    h_i = ops.uniaxialMaterial.LadrunoCohesiveHinge(Mc=45e3, Gf=2500.0, softening="linear")
    h_j = ops.uniaxialMaterial.LadrunoCohesiveHinge(Mc=50e3, Gf=2800.0, softening="linear")
    ops.element.LadrunoForceBeamColumn(pg="fbc", transf=tr, integration=integ, hinge_ij=(h_i, h_j), hinge_loc=(0.05, 0.95))
    b_i = ops.nDMaterial.LadrunoCohesiveHingeBiaxial(Mcz=45e3, Gfz=2500.0, Mcy=35e3, Gfy=2000.0)
    b_j = ops.nDMaterial.LadrunoCohesiveHingeBiaxial(Mcz=50e3, Gfz=2800.0, Mcy=40e3, Gfy=2200.0)
    ops.element.LadrunoDispBeamColumn(pg="dbc", transf=tr, integration=integ, hinge_biaxial_ij=(b_i, b_j), hinge_loc=(0.1, 0.9))
    ops.element.elasticBeamColumn(pg="plain", transf=tr, A=AREA, E=E, G=G, J=J, Iy=IY, Iz=IZ)
    pgs = tuple(pg for pg, _ in beams)
    for pg in pgs:
        ops.fix(label=f"base_{pg}", dofs=(1, 1, 1, 1, 1, 1))
    _analyse("ladruno_hinges3d", fem, ops, pgs, (0.0, 1e3, 200.0, 0.0, 0.0, 0.0), ("imk", "fbc", "dbc"), {"steel": {"fy": FY, "k": KH, "b": B}})


def two_d() -> None:
    beams = (("imk", -1.0), ("fbc", 1.0))
    fem = _fem("ladruno_hinges2d", beams)
    ops = apeSees(fem, default_orientation=None)
    ops.model(ndm=2, ndf=3)
    tr = ops.geomTransf.Linear()
    sec = ops.section.Elastic(E=E, A=AREA, Iz=IZ)
    integ = ops.beamIntegration.Lobatto(section=sec, n_ip=4)
    steel = ops.uniaxialMaterial.Steel01(fy=FY, E=KH, b=B)
    ops.element.LadrunoIMKBeam(pg="imk", transf=tr, A=AREA, E=E, Iz=IZ, ends="i", mat_z=steel)
    hinge = ops.uniaxialMaterial.LadrunoCohesiveHinge(Mc=45e3, Gf=2500.0, softening="linear")
    ops.element.LadrunoForceBeamColumn(pg="fbc", transf=tr, integration=integ, hinge_j=hinge)
    pgs = tuple(pg for pg, _ in beams)
    for pg in pgs:
        ops.fix(label=f"base_{pg}", dofs=(1, 1, 1))
    _analyse("ladruno_hinges2d", fem, ops, pgs, (0.0, 1e3, 0.0), pgs, {"steel": {"fy": FY, "k": KH, "b": B}})


if __name__ == "__main__":
    os.chdir(HERE)  # the recorder path is stored in the model, so keep it relative
    three_d()
    two_d()
