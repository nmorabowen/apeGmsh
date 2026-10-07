"""Era-stable generator for one schema-corpus file (ADR 0113 D8).

``scripts/build_schema_corpus.py`` runs this file in a subprocess with
``PYTHONPATH=<era worktree>/src``, so ``import apeGmsh`` resolves to the
frozen writer of one schema era.  It writes one ``model.h5``, then opens
it with **that era's own reader** and stores the semantic dump
(``_semantic_dump.py``, beside this file) and, for the opensees zone,
the era's own ``build("tcl")`` deck.  Usage::

    python _era_generator.py --zone neutral  --out F.h5 --dump F.dump.json
    python _era_generator.py --zone opensees --out F.h5 --dump F.dump.json --tcl F.tcl
    python _era_generator.py --zone assembly --out F.h5 --dump F.dump.json

Only API that exists unchanged from the zone floors on is used (neutral
2.10.0, opensees 2.12.0); where a call was renamed, the generator probes
for the spelling the era knows and records which one it used, so the
*model* is the same in every era.  A failure here is a gap the builder
records; the generator never substitutes a different model.

``--zone assembly`` (ADR 0117 D5, from ``assembly_schema_version`` 1.0.0)
writes a two-instance stack through ``Assembly.h5`` (:func:`assembly_stack`)
and adds the ``/assembly`` dump of ``Assembly.from_h5`` and the zone's rows.

``--variant`` builds one of the ADR 0113 D4 shim-ledger cases instead of
the plain model, at one chosen era (``scripts/build_schema_corpus.py``
``VARIANTS``):

* ``sp_cases`` (neutral): the box with prescribed displacements under two
  ``g.displacements.case`` names.  A writer before neutral 2.26.1 flattens
  every SP record into ``/loads/sp/default``, so the file reads as one
  ``default`` case (ledger, Q5).
* ``frame2d`` (opensees): the same portal frame declared ``ops.model(ndm=2,
  ndf=3)`` in the XY plane.  A writer before neutral 2.34.0 stamps the mesh
  dimension in ``/meta/ndm``, the case ``read_spatial_ndm`` salvages (#1300).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import tempfile

import gmsh

import apeGmsh
from apeGmsh import apeGmsh as Session

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _semantic_dump import dump_fem, dump_model, dump_stamps  # noqa: E402


def _face_at_z(volume_tag: int, z: float, tol: float = 1e-6) -> int:
    for dim, tag in gmsh.model.getBoundary([(3, volume_tag)], oriented=False):
        if dim != 2:
            continue
        com = gmsh.model.occ.getCenterOfMass(2, abs(tag))
        if abs(com[2] - z) < tol:
            return abs(tag)
    raise AssertionError(f"no boundary face of volume {volume_tag} at z={z}")


#: The ``g.displacements.case`` names the ``sp_cases`` variant authors.
SP_CASES = ("PushA", "PushB")


def neutral_box(*, sp_cases: bool = False) -> "tuple[object, dict]":
    """The deterministic box of ``tests/fixtures/neutral_zone/_generate_fixtures.py``.

    With ``sp_cases`` the base face also carries one prescribed
    displacement per case in :data:`SP_CASES` (``g.displacements``, ADR
    0050, present from 2026-05-31 on).
    """
    notes: dict = {}
    g = Session(model_name="neutral_box", verbose=False)
    g.begin()
    try:
        vol = g.model.geometry.add_box(0.0, 0.0, 0.0, 1.0, 1.0, 1.0, label="box")
        g.model.sync()
        base = _face_at_z(vol, 0.0)
        g.physical.add_volume("box", name="Body")
        g.physical.add(2, [base], name="Base")
        if sp_cases:
            for case, values in zip(SP_CASES, ((0.0, 0.0, -0.01), (0.01, 0.0, 0.0))):
                with g.displacements.case(case):
                    g.displacements.point(pg="Base", dofs=[1, 1, 1], values=values)
            notes["sp_cases"] = list(SP_CASES)
        g.mesh.structured.set_transfinite_box("box", n=3)
        g.mesh.generation.generate(dim=3)
        g.mesh_selection.select().in_box(
            (-0.01, -0.01, -0.01), (1.01, 1.01, 0.01),
        ).save_as("base_nodes")
        fem = g.mesh.queries.get_fem_data(dim=3)
    finally:
        g.end()
    return fem, notes


def _point_load(g: "object", pg: str, force: tuple) -> str:
    """One nodal load on ``pg``; returns the spelling the era knows."""
    point = g.loads.point
    if callable(getattr(point, "force", None)):
        point.force(pg=pg, force=force)
        return "g.loads.point.force(pg=, force=)"
    point(pg=pg, force_xyz=force)
    return "g.loads.point(pg=, force_xyz=)"


def opensees_frame(path: str, *, ndm: int = 3) -> "tuple[object, dict]":
    """A one-bay portal frame through ``apeSees(fem).h5()``.

    ``ndm=3`` is the plain corpus model (XZ plane, 6 dof).  ``ndm=2`` is
    the ``frame2d`` variant: the same frame in the XY plane declared
    ``ops.model(ndm=2, ndf=3)``, with the 2-D member signatures.
    """
    from apeGmsh.opensees import apeSees
    from apeGmsh.opensees.section.fiber import FiberPoint

    if ndm not in (2, 3):
        raise ValueError(f"opensees_frame: ndm must be 2 or 3, got {ndm!r}")
    notes: dict = {"declared_ndm": ndm}
    g = Session(model_name="corpus_frame", verbose=False)
    g.begin()
    try:
        occ = gmsh.model.occ
        p0 = occ.addPoint(0.0, 0.0, 0.0)
        if ndm == 3:
            p1 = occ.addPoint(0.0, 0.0, 3.0)
            p2 = occ.addPoint(4.0, 0.0, 3.0)
        else:
            p1 = occ.addPoint(0.0, 3.0, 0.0)
            p2 = occ.addPoint(4.0, 3.0, 0.0)
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
    steel = ops.uniaxialMaterial.Steel02(fy=420.0e6, E=200.0e9, b=0.01)
    fibers = (
        FiberPoint(material=steel, y=0.1, z=0.0, area=0.005),
        FiberPoint(material=steel, y=-0.1, z=0.0, area=0.005),
    )
    if ndm == 3:
        ops.model(ndm=3, ndf=6)
        transf = ops.geomTransf.Linear(vecxz=(0.0, 1.0, 0.0))
        ops.element.elasticBeamColumn(
            pg="Cols", transf=transf,
            A=0.01, E=200.0e9, Iz=1.0e-4, Iy=2.0e-4, G=80.0e9, J=3.0e-4,
        )
        sec = ops.section.Fiber(GJ=2.4e7, fibers=fibers)
        fix = (1, 1, 1, 1, 1, 1)
        mass = (100.0, 100.0, 100.0, 0.0, 0.0, 0.0)
        forces = (5.0e3, 0.0, 0.0, 0.0, 0.0, 0.0)
    else:
        ops.model(ndm=2, ndf=3)
        transf = ops.geomTransf.Linear()
        ops.element.elasticBeamColumn(
            pg="Cols", transf=transf, A=0.01, E=200.0e9, Iz=1.0e-4,
        )
        sec = ops.section.Fiber(fibers=fibers)
        fix = (1, 1, 1)
        mass = (100.0, 100.0, 0.0)
        forces = (5.0e3, 0.0, 0.0)
    integ = ops.beamIntegration.Lobatto(section=sec, n_ip=3)
    ops.element.forceBeamColumn(pg="Beam", transf=transf, integration=integ)
    ops.fix(pg="Base", dofs=fix)
    ops.mass(pg="Top", values=mass)
    ts = ops.timeSeries.Linear()
    with ops.pattern.Plain(series=ts) as pat:
        pat.load(pg="Top", forces=forces)
    ops.h5(path)
    return fem, notes


def assembly_stack(path: str) -> dict:
    """Two instances of one hex8 block, the second turned half a turn about
    z and stacked on the first, joined by one named ``equation`` tie.

    The block file is written in a temporary directory that is the working
    directory while the assembly is declared, so ``/assembly`` and
    ``/composed_from`` record the relative ``block.h5``.
    """
    from apeGmsh.assembly import Assembly
    from apeGmsh.opensees import apeSees

    side, h, tol = 10.0, 10.0, 0.01
    notes: dict = {"instances": 2, "ties": 1}
    here = os.getcwd()
    work = tempfile.mkdtemp(prefix="assembly_corpus_")
    os.chdir(work)
    try:
        with Session(model_name="block", verbose=False,
                     save_to=os.path.join(work, "block_mesh.h5")) as g:
            g.model.geometry.add_box(0.0, 0.0, 0.0, side, side, h, label="v")
            g.physical.add_volume("v", name="Vol")
            for z, pg in ((0.0, "bot"), (h, "top")):
                faces = g.model.select(None, dim=2).in_box(
                    (-side, -side, z - tol), (2 * side, 2 * side, z + tol),
                ).result().tags()
                g.physical.add_surface(faces, name=pg)
            g.mesh.recipe.structured(size=5.0, fallback="strict")
            fem = g.mesh.queries.get_fem_data(dim=None)
        block = apeSees(fem)
        block.model(ndm=3, ndf=3)
        steel = block.nDMaterial.ElasticIsotropic(
            E=200_000.0, nu=0.3, rho=7.85e-9, name="steel")
        block.element.stdBrick(pg="Vol", material=steel)
        block.h5("block.h5")

        asm = Assembly("stack")
        asm.instance("pier_1", "block.h5")
        asm.instance("pier_2", "block.h5", translate=(side, side, h),
                     rotate=((0.0, 0.0, 1.0), math.pi))
        asm.tie("pier_1.top", "pier_2.bot", enforce="equation",
                dofs=[1, 2, 3], name="t1")
        asm.bridge(ndm=3, ndf=3)
        asm.h5(path)
    finally:
        os.chdir(here)
    return notes


def main(argv: "list[str] | None" = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--zone", choices=("neutral", "opensees", "assembly"), required=True)
    ap.add_argument("--variant", choices=("sp_cases", "frame2d"), default=None,
                    help="an ADR 0113 D4 ledger case instead of the plain model")
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

    variant_zone = {"sp_cases": "neutral", "frame2d": "opensees"}
    if a.variant and variant_zone[a.variant] != a.zone:
        raise SystemExit(
            f"--variant {a.variant} belongs to the {variant_zone[a.variant]} zone"
        )
    if a.zone == "neutral":
        fem, notes = neutral_box(sp_cases=a.variant == "sp_cases")
        fem.to_h5(a.out)
    elif a.zone == "assembly":
        notes = assembly_stack(os.path.abspath(a.out))
    else:
        _fem, notes = opensees_frame(a.out, ndm=2 if a.variant == "frame2d" else 3)

    # The era's OWN reader produces the oracle.
    dump: dict = {
        "fem": dump_fem(FEMData.from_h5(a.out)),
        "meta": dump_stamps(a.out),
    }
    if a.zone in ("opensees", "assembly"):
        from apeGmsh.opensees.opensees_model import OpenSeesModel

        model = OpenSeesModel.from_h5(a.out)
        dump["model"] = dump_model(model)
        if a.tcl:
            deck = model.build("tcl")
            with open(a.tcl, "w", encoding="utf-8", newline="\n") as fh:
                fh.write(deck)
    if a.zone == "assembly":
        from apeGmsh.assembly import Assembly
        from apeGmsh.assembly._h5 import read_assembly_zone

        from _semantic_dump import dump_assembly
        dump["assembly"] = dump_assembly(
            Assembly.from_h5(a.out), read_assembly_zone(a.out))
    dump["generator_notes"] = notes
    with open(a.dump, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(dump, fh, indent=1, sort_keys=True)
        fh.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
