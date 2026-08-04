r"""Defect 1 reproduction — chain-phase silent failures in the def router.

Builds two independently-meshed blocks (module A: z 0..50, module B:
z 50.5..100.5, 0.5 mm gap), saves each to h5, then opens a chain-phase
assembly via apeGmsh.from_h5(A) + g.compose(B) and probes every silent
path:

  1a. tie with a MISSPELLED slave label            -> expected: raise?
  1b. constraints.bc with a misspelled target      -> silent?
  1c. loads.point (in a case) misspelled target    -> silent?
  1d. masses.point with a misspelled target        -> silent?
  1e. displacements.surface on a VALID face        -> silent no-op?
  1f. tie, valid labels, tolerance << gap          -> 0 records, silent?
  1g. gravity load on a valid volume PG            -> silent no-op?

Run:
  set PYTHONPATH=<worktree>\src
  C:\Users\nmb\venv\opensees_env\Scripts\python.exe repros\repro1_silent_router.py
"""
import sys
import tempfile
from pathlib import Path

import apeGmsh as _pkg
print(f"apeGmsh from: {_pkg.__file__}")
from apeGmsh import apeGmsh

TMP = Path(tempfile.mkdtemp(prefix="apeGmsh_repro1_"))


def build_module(fname, name, z0, size):
    path = TMP / fname
    with apeGmsh(model_name=name, save_to=str(path)) as g:
        g.model.geometry.add_box(0, 0, z0, 100, 100, 50, label="blk")
        g.model.sync()
        g.model.select("blk").to_physical(f"{name}_body")
        g.model.select(dim=2).on_plane(
            (0, 0, z0 + 50), (0, 0, 1), tol=1e-3).to_physical(f"{name}_top")
        g.model.select(dim=2).on_plane(
            (0, 0, z0), (0, 0, 1), tol=1e-3).to_physical(f"{name}_bot")
        g.mesh.sizing.set_size("blk", size)
        g.mesh.generation.generate(3)
        g.mesh.queries.get_fem_data(dim=None)
    return path


pa = build_module("mod_a.h5", "A", 0.0, 25.0)
pb = build_module("mod_b.h5", "B", 50.5, 20.0)

g = apeGmsh.from_h5(pa)
g.compose(str(pb), label="B")
fem = g._fem

node_pgs = sorted(e.get("name", "?") for e in fem.nodes.physical._groups.values())
elem_pgs = sorted(e.get("name", "?") for e in fem.elements.physical._groups.values())
print(f"\nnode-side PGs: {node_pgs}")
print(f"elem-side PGs: {elem_pgs}")


def counts():
    f = g._fem
    return dict(
        elem_constraints=len(list(f.elements.constraints)),
        node_constraints=len(list(f.nodes.constraints)),
        sp=len(list(f.nodes.sp)),
        loads=len(list(f.nodes.loads)),
        masses=len(list(f.nodes.masses)) if hasattr(f.nodes, "masses") else -1,
    )


def probe(tag, fn):
    before = counts()
    try:
        fn()
        outcome = "NO EXCEPTION"
    except Exception as exc:  # noqa: BLE001
        outcome = f"raised {type(exc).__name__}: {str(exc)[:100]}"
    after = counts()
    delta = {k: after[k] - before[k] for k in after if after[k] != before[k]}
    print(f"\n[{tag}] {outcome}")
    print(f"      record delta: {delta if delta else 'NONE (broker unchanged)'}")


# Resolve the actual composed names for module B (compose may prefix them).
b_top = next((n for n in node_pgs + elem_pgs if n.endswith("B_top")), "B_top")
b_bot = next((n for n in node_pgs + elem_pgs if n.endswith("B_bot")), "B_bot")
b_body = next((n for n in elem_pgs if n.endswith("B_body")), "B_body")
print(f"\nusing composed names: b_top={b_top!r} b_bot={b_bot!r} b_body={b_body!r}")

probe("1a tie, misspelled slave label",
      lambda: g.constraints.tie("A_top", "B_bott_TYPO", tolerance=1.0))

probe("1b bc, misspelled target",
      lambda: g.constraints.bc("A_bott_TYPO", dofs=[1, 1, 1]))


def _load_case():
    with g.loads.case("push"):
        g.loads.point.force("A_topp_TYPO", force=(0, 0, -1000.0))


probe("1c point load, misspelled target", _load_case)

probe("1d point mass, misspelled target",
      lambda: g.masses.point("A_topp_TYPO", mass=10.0))


def _disp_case():
    with g.displacements.case("push_gap"):
        g.displacements.surface(b_top, disp_xyz=(0, 0, -15.0), name="top_push")


probe("1e displacements.surface, VALID face, chain phase", _disp_case)

probe("1f tie, valid labels, tolerance 0.01 << gap 0.5 (zero projections)",
      lambda: g.constraints.tie("A_top", b_bot, tolerance=0.01))

def _gravity():
    with g.loads.case("dead"):
        g.loads.gravity(b_body, g=(0, 0, -9810.0), density=2.4e-9)


probe("1g gravity load, valid volume PG, chain phase", _gravity)

print(f"\ntmp dir: {TMP}")
