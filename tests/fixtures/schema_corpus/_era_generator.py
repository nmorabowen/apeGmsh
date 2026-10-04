"""Era-stable generator for one schema-corpus file (ADR 0113 D8).

``scripts/build_schema_corpus.py`` runs this file in a subprocess with
``PYTHONPATH=<era worktree>/src``, so ``import apeGmsh`` resolves to the
frozen writer of one schema era.  It writes one ``model.h5``, then opens
it with **that era's own reader** and stores the semantic dump
(``_semantic_dump.py``, beside this file) and, for the opensees zone,
the era's own ``build("tcl")`` deck.  Usage::

    python _era_generator.py --zone neutral  --out F.h5 --dump F.dump.json
    python _era_generator.py --zone opensees --out F.h5 --dump F.dump.json --tcl F.tcl

Only API that exists unchanged from the zone floors on is used (neutral
2.10.0, opensees 2.11.0); where a call was renamed, the generator probes
for the spelling the era knows and records which one it used, so the
*model* is the same in every era.  A failure here is a gap the builder
records; the generator never substitutes a different model.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import gmsh

import apeGmsh
from apeGmsh import apeGmsh as Session

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _semantic_dump import dump_fem, dump_model  # noqa: E402


def _face_at_z(volume_tag: int, z: float, tol: float = 1e-6) -> int:
    for dim, tag in gmsh.model.getBoundary([(3, volume_tag)], oriented=False):
        if dim != 2:
            continue
        com = gmsh.model.occ.getCenterOfMass(2, abs(tag))
        if abs(com[2] - z) < tol:
            return abs(tag)
    raise AssertionError(f"no boundary face of volume {volume_tag} at z={z}")


def neutral_box() -> "object":
    """The deterministic box of ``tests/fixtures/neutral_zone/_generate_fixtures.py``."""
    g = Session(model_name="neutral_box", verbose=False)
    g.begin()
    try:
        vol = g.model.geometry.add_box(0.0, 0.0, 0.0, 1.0, 1.0, 1.0, label="box")
        g.model.sync()
        base = _face_at_z(vol, 0.0)
        g.physical.add_volume("box", name="Body")
        g.physical.add(2, [base], name="Base")
        g.mesh.structured.set_transfinite_box("box", n=3)
        g.mesh.generation.generate(dim=3)
        g.mesh_selection.select().in_box(
            (-0.01, -0.01, -0.01), (1.01, 1.01, 0.01),
        ).save_as("base_nodes")
        fem = g.mesh.queries.get_fem_data(dim=3)
    finally:
        g.end()
    return fem


def _point_load(g: "object", pg: str, force: tuple) -> str:
    """One nodal load on ``pg``; returns the spelling the era knows."""
    point = g.loads.point
    if callable(getattr(point, "force", None)):
        point.force(pg=pg, force=force)
        return "g.loads.point.force(pg=, force=)"
    point(pg=pg, force_xyz=force)
    return "g.loads.point(pg=, force_xyz=)"


def opensees_frame(path: str) -> "tuple[object, dict]":
    """A one-bay 3-D portal frame through ``apeSees(fem).h5()``."""
    from apeGmsh.opensees import apeSees
    from apeGmsh.opensees.section.fiber import FiberPoint

    notes: dict = {}
    g = Session(model_name="corpus_frame", verbose=False)
    g.begin()
    try:
        occ = gmsh.model.occ
        p0 = occ.addPoint(0.0, 0.0, 0.0)
        p1 = occ.addPoint(0.0, 0.0, 3.0)
        p2 = occ.addPoint(4.0, 0.0, 3.0)
        p3 = occ.addPoint(4.0, 0.0, 0.0)
        c1 = occ.addLine(p0, p1)
        bm = occ.addLine(p1, p2)
        c2 = occ.addLine(p3, p2)
        g.model.sync()
        for c in (c1, bm, c2):
            gmsh.model.mesh.setTransfiniteCurve(c, 3)  # two elements per member
        g.physical.add(1, [c1, c2], name="Cols")
        g.physical.add(1, [bm], name="Beam")
        g.physical.add(0, [p0, p3], name="Base")
        g.physical.add(0, [p1], name="Top")
        # ADR 0051 renamed g.loads.pattern -> g.loads.case (2026-05-31).
        if callable(getattr(g.loads, "case", None)):
            grouping, notes["neutral_case_api"] = g.loads.case, "g.loads.case"
        else:
            grouping, notes["neutral_case_api"] = g.loads.pattern, "g.loads.pattern"
        with grouping("Lateral"):
            notes["neutral_load_api"] = _point_load(g, "Top", (10.0e3, 0.0, 0.0))
        g.mesh.generation.generate(dim=1)
        fem = g.mesh.queries.get_fem_data(dim=1)
    finally:
        g.end()

    ops = apeSees(fem)
    ops.model(ndm=3, ndf=6)
    transf = ops.geomTransf.Linear(vecxz=(0.0, 1.0, 0.0))
    ops.element.elasticBeamColumn(
        pg="Cols", transf=transf,
        A=0.01, E=200.0e9, Iz=1.0e-4, Iy=2.0e-4, G=80.0e9, J=3.0e-4,
    )
    steel = ops.uniaxialMaterial.Steel02(fy=420.0e6, E=200.0e9, b=0.01)
    sec = ops.section.Fiber(
        GJ=2.4e7,
        fibers=(
            FiberPoint(material=steel, y=0.1, z=0.0, area=0.005),
            FiberPoint(material=steel, y=-0.1, z=0.0, area=0.005),
        ),
    )
    integ = ops.beamIntegration.Lobatto(section=sec, n_ip=3)
    ops.element.forceBeamColumn(pg="Beam", transf=transf, integration=integ)
    ops.fix(pg="Base", dofs=(1, 1, 1, 1, 1, 1))
    ops.mass(pg="Top", values=(100.0, 100.0, 100.0, 0.0, 0.0, 0.0))
    ts = ops.timeSeries.Linear()
    with ops.pattern.Plain(series=ts) as pat:
        pat.load(pg="Top", forces=(5.0e3, 0.0, 0.0, 0.0, 0.0, 0.0))
    ops.h5(path)
    return fem, notes


def main(argv: "list[str] | None" = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--zone", choices=("neutral", "opensees"), required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--dump", required=True)
    ap.add_argument("--tcl")
    ap.add_argument("--expect-src", required=True,
                    help="the era worktree's src/; apeGmsh must import from it")
    a = ap.parse_args(argv)

    here = os.path.normcase(os.path.abspath(apeGmsh.__file__))
    want = os.path.normcase(os.path.abspath(a.expect_src))
    if not here.startswith(want + os.sep):
        raise SystemExit(f"apeGmsh imported from {here}, not the era tree {want}")

    from apeGmsh.mesh.FEMData import FEMData

    notes: dict = {}
    if a.zone == "neutral":
        neutral_box().to_h5(a.out)
    else:
        _fem, notes = opensees_frame(a.out)

    # The era's OWN reader produces the oracle.
    dump: dict = {"fem": dump_fem(FEMData.from_h5(a.out))}
    if a.zone == "opensees":
        from apeGmsh.opensees.opensees_model import OpenSeesModel

        model = OpenSeesModel.from_h5(a.out)
        dump["model"] = dump_model(model)
        if a.tcl:
            deck = model.build("tcl")
            with open(a.tcl, "w", encoding="utf-8", newline="\n") as fh:
                fh.write(deck)
    dump["generator_notes"] = notes
    with open(a.dump, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(dump, fh, indent=1, sort_keys=True)
        fh.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
