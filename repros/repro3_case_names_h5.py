r"""Defect 3 reproduction — displacement case names vs the h5 neutral zone.

Builds one block, applies a displacement under case "push_gap" and a
nodal load under case "live", extracts, prints each record's `pattern`,
saves to h5, reloads with FEMData.from_h5, prints again.

Expected defect: SPRecord.pattern == 'push_gap' before save, 'default'
(or missing) after reload; NodalLoadRecord may or may not round-trip.

Run:
  set PYTHONPATH=<worktree>\src
  C:\Users\nmb\venv\opensees_env\Scripts\python.exe repros\repro3_case_names_h5.py
"""
import tempfile
from pathlib import Path

import apeGmsh as _pkg
print(f"apeGmsh from: {_pkg.__file__}")
from apeGmsh import apeGmsh
from apeGmsh.mesh.FEMData import FEMData

TMP = Path(tempfile.mkdtemp(prefix="apeGmsh_repro3_"))
path = TMP / "part.h5"

with apeGmsh(model_name="part", save_to=str(path)) as g:
    g.model.geometry.add_box(0, 0, 0, 100, 100, 50, label="blk")
    g.model.sync()
    g.model.select("blk").to_physical("Body")
    g.model.select(dim=2).on_plane(
        (0, 0, 50), (0, 0, 1), tol=1e-3).to_physical("TopFace")
    g.model.select(dim=2).on_plane(
        (0, 0, 0), (0, 0, 1), tol=1e-3).to_physical("BaseFace")
    g.constraints.bc("BaseFace", dofs=[1, 1, 1])

    with g.displacements.case("push_gap"):
        g.displacements.surface("TopFace", disp_xyz=(0, 0, -15.0),
                                name="top_push_gap")
    with g.loads.case("live"):
        g.loads.point.force("TopFace", force=(0, 0, -1000.0), name="tip")

    g.mesh.sizing.set_size("blk", 25.0)
    g.mesh.generation.generate(3)
    fem = g.mesh.queries.get_fem_data(dim=None)


def summarize(tag, f):
    sp_patterns = {}
    for r in f.nodes.sp:
        key = (getattr(r, "pattern", "<no attr>"), r.is_homogeneous)
        sp_patterns[key] = sp_patterns.get(key, 0) + 1
    load_patterns = {}
    for r in f.nodes.loads:
        p = getattr(r, "pattern", "<no attr>")
        load_patterns[p] = load_patterns.get(p, 0) + 1
    print(f"\n[{tag}]")
    print(f"  sp records   (pattern, is_homogeneous) -> count: {sp_patterns}")
    print(f"  load records  pattern -> count: {load_patterns}")


summarize("in-session FEMData (before save)", fem)

reloaded = FEMData.from_h5(str(path))
summarize("reloaded via FEMData.from_h5", reloaded)

# The assembly-side consequence: from_model('push_gap') on the reload.
n_match_sp = sum(
    1 for r in reloaded.nodes.sp if getattr(r, "pattern", None) == "push_gap")
n_match_ld = sum(
    1 for r in reloaded.nodes.loads if getattr(r, "pattern", None) == "live")
print(f"\nreloaded records matching case 'push_gap' (sp): {n_match_sp}")
print(f"reloaded records matching case 'live' (loads):  {n_match_ld}")
print(f"\ntmp dir: {TMP}")
