"""ETABS analytical model -> apeGmsh mesh -> OpenSees deck (Phase 2 demo).

Reads the two-story-frame fixture from the apeETABS schema, builds geometry,
meshes the frame members as 1-D beam elements, and emits OpenSees Tcl + Python
decks. Proves the full apeETABS -> apeGmsh pipe with no live ETABS.

Run:
    python examples/etabs_import_demo.py [path/to/two_story_frame.sm.json]
"""
from __future__ import annotations

import sys
from pathlib import Path

from apeGmsh import apeGmsh
from apeGmsh.interop import StructuralModel, build_opensees, import_structural_model

FIXTURE = Path(
    r"C:/Users/nmb/Documents/Github/apeETABS/schema/examples/two_story_frame.sm.json"
)
OUT_TCL = Path("etabs_frame.tcl")
OUT_PY = Path("etabs_frame.py")


def main(fixture: Path) -> None:
    model = StructuralModel.from_json(fixture)

    has_areas = bool(model.areas)
    sess = apeGmsh(model_name="etabs_import", verbose=False)
    sess.begin()
    try:
        result = import_structural_model(sess, model)
        if has_areas:
            sess.mesh.sizing.set_global_size(0.5)
            sess.mesh.generation.generate(dim=2)     # shells + beam edges
            sess.mesh.partitioning.renumber(base=1)
            fem = sess.mesh.queries.get_fem_data(dim=None)
        else:
            sess.mesh.sizing.set_global_size(1.0)    # subdivide members
            sess.mesh.generation.generate(dim=1)
            sess.mesh.partitioning.renumber(dim=1, method="rcm", base=1)
            fem = sess.mesh.queries.get_fem_data(dim=1)
    finally:
        sess.end()

    ops = build_opensees(fem, model, result, ndm=3, ndf=6)
    ops.tcl(str(OUT_TCL))
    ops.py(str(OUT_PY))

    print("=== etabs_import_demo ===")
    print(f"  source:       {fixture.name}")
    print(f"  units:        {model.units}")
    print(f"  nodes (ETABS):{len(model.nodes)}  frames:{len(model.frames)}")
    print(f"  mesh nodes:   {fem.info.n_nodes}")
    print(f"  mesh elems:   {fem.info.n_elems}")
    print(f"  frame groups: {[(fg.pg, fg.orient) for fg in result.frame_groups]}")
    print(f"  area groups:  {[ag.pg for ag in result.area_groups]}")
    print(f"  restraints:   {[rg.pg for rg in result.restraint_groups]}")
    print(f"  load patterns:{result.load_patterns}")
    print(f"  self-mass:    {result.has_masses}")
    print(f"  diaphragms:   {[(d.name, 'shell' if d.shell_backed else 'rigid') for d in result.diaphragms]}")
    for msg in result.skipped:
        print(f"  skipped:      {msg}")
    print(f"  Tcl:          {OUT_TCL.resolve()}")
    print(f"  Py:           {OUT_PY.resolve()}")


if __name__ == "__main__":
    arg = Path(sys.argv[1]) if len(sys.argv) > 1 else FIXTURE
    main(arg)
