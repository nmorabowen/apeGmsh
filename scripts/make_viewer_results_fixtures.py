"""Write the V4a results.h5 files with apeGmsh's own NativeWriter (#1707).

This generator imports apeGmsh, so it lives here, outside ``apeGmshViewer/``:
the app never imports or runs apeGmsh (ADR 0112 D4,
``scripts/check_app_wall.py``). It writes into ``apeGmshViewer/fixtures/``.
Run it from the repository root with the venv interpreter and the
checkout's ``src``:

    $env:PYTHONPATH = "src"
    python scripts/make_viewer_results_fixtures.py            # results_small.h5
    python scripts/make_viewer_results_fixtures.py --measure apeGmshViewer/out/results

The values are SYNTHETIC: no solve runs. They are closed forms of the node
id, the element id, the Gauss point and the step, so a test can recompute
them without this script (``test/results/results.test.ts``):

    time[k]                      = 0.01 * k
    nodes/<comp c>[k, j]         = id_j + 0.001 * (c + 1) * k
    gauss <comp c>[k, e, g]      = eid_e + 0.1 * g + 0.001 * (c + 1) * k

with ``c`` the component's position in the lists below. Every file is a
``domain_capture`` file with ``/model`` embedded and one partition,
``partition_0``, which is what ``DomainCapture`` writes today.

Files:

* ``small`` (committed, ``fixtures/results_small.h5``): an extruded
  triangle mesh, 2 x 2 cells x 1 layer (18 nodes, 8 wedges), 5 steps,
  ``displacement_x/y/z`` and one gauss group of 6 points with
  ``stress_xx``, ``stress_yy``.
* ``a`` (measured, not committed): the same mesh at 13 x 13 x 21
  (4 312 nodes, 7 098 wedges, the size of the ledger's largest maintainer
  model, 4 254 nodes / 7 104 elements), 100 steps, 3 nodal components and 6
  stress components at 6 Gauss points.
* ``b`` (measured): two tet-meshed blocks, the ledger's ``stress_box`` scale
  probe rebuilt (about 220 900 tets), 100 steps, 3 nodal and 6 stress
  components at 1 Gauss point.
* ``c`` (measured): the ``a`` mesh, 5 000 steps (a ground-motion history at
  dt = 0.01 s), 3 nodal components and ``stress_xx`` only at 6 Gauss points
  (6 stress components would be 10 GB).

``--staged`` (V4b, #1728; committed) writes a staged pair instead:
``fixtures/staged_activation.h5`` (a composed ``model.h5`` with ``/sequence``)
and ``fixtures/staged_activation.results.h5``, written by ``NativeWriter``
with ``model_h5_src=`` so the results carry the model's ``model_hash``. Three
touching tet-meshed boxes, ``Col`` and ``Prop`` global and ``Top`` activated
by the second stage; the third stage removes ``Prop``::

    base      Col + Prop drawn, Top hidden (not activated yet)
    erect     s.activate(pgs=["Top"]): everything drawn
    unprop    s.remove_element(pg="Prop"): Col + Top drawn

The results hold one stage per program stage, by name, plus ``mode_1`` (a
``mode`` stage no program stage names: the viewer's ``absent`` state). The
values are synthetic, ``displacement_x[k, j] = id_j + 0.001 * k`` with the
stage's ``time[k] = (k + 1) / T``; no solve runs.
"""

from __future__ import annotations

import argparse
import tempfile
import time as _time
from pathlib import Path

import gmsh
import numpy as np

from apeGmsh import apeGmsh
from apeGmsh.results.writers._native import NativeWriter

HERE = Path(__file__).resolve().parents[1] / "apeGmshViewer" / "fixtures"
# The session writes its own model.h5 + geometry sibling at end() (ADR 0112 D1);
# send them to a throw-away directory, not beside the fixtures.
SESSION_DIR = Path(tempfile.mkdtemp(prefix="v4a-session-"))
NODAL = ["displacement_x", "displacement_y", "displacement_z"]
STRESS6 = ["stress_xx", "stress_yy", "stress_zz", "stress_xy", "stress_yz", "stress_xz"]
WEDGE_GP = np.array(  # 6-point wedge rule: (xi, eta, zeta)
    [[1 / 6, 1 / 6, -1 / np.sqrt(3)], [2 / 3, 1 / 6, -1 / np.sqrt(3)], [1 / 6, 2 / 3, -1 / np.sqrt(3)],
     [1 / 6, 1 / 6, 1 / np.sqrt(3)], [2 / 3, 1 / 6, 1 / np.sqrt(3)], [1 / 6, 2 / 3, 1 / np.sqrt(3)]]
)
TET_GP = np.array([[0.25, 0.25, 0.25]])


def wedge_mesh(n: int, layers: int, name: str):
    """A unit square, n x n structured triangles (2 per cell), extruded in `layers` wedge layers."""
    with apeGmsh(model_name=name, verbose=False, save_to=SESSION_DIR / f"{name}.h5") as g:
        p = gmsh.model.geo.addPoint(0, 0, 0)
        line = gmsh.model.geo.extrude([(0, p)], 1, 0, 0, numElements=[n])
        surf = gmsh.model.geo.extrude([line[1]], 0, 1, 0, numElements=[n])
        vol = gmsh.model.geo.extrude([surf[1]], 0, 0, 1.5, numElements=[layers], recombine=True)
        gmsh.model.geo.synchronize()
        g.model.sync()
        g.physical.add(3, [vol[1][1]], name="Body")
        g.mesh.generation.generate(3)
        return g.mesh.queries.get_fem_data(dim=3)


def tet_blocks(size: float, name: str):
    """Two touching tet-meshed blocks (1 x 1 x 1 and 1 x 1 x 0.6), the stress_box probe's shape."""
    with apeGmsh(model_name=name, verbose=False, save_to=SESSION_DIR / f"{name}.h5") as g:
        a = g.model.geometry.add_box(0, 0, 0, 1, 1, 1, label="lower")
        b = g.model.geometry.add_box(0, 0, 1, 1, 1, 0.6, label="upper")
        g.model.sync()
        g.physical.add_volume("lower", name="Lower")
        g.physical.add_volume("upper", name="Upper")
        g.mesh.sizing.set_global_size(size)
        g.mesh.generation.generate(dim=3)
        del a, b
        return g.mesh.queries.get_fem_data(dim=3)


def write(path: Path, fem, n_steps: int, stress: list[str], gp: np.ndarray, class_tag: int) -> None:
    node_ids = np.asarray(fem.nodes.ids, dtype=np.int64)
    eids = np.asarray(fem.elements.ids, dtype=np.int64)
    k = np.arange(n_steps, dtype=np.float64)
    t0 = _time.perf_counter()
    with NativeWriter(path) as w:
        w.open(fem=fem, source_type="domain_capture", analysis_label="V4a synthetic (#1707)")
        sid = w.begin_stage(name="history", kind="transient", time=0.01 * k)
        w.write_nodes(
            sid, "partition_0", node_ids=node_ids,
            components={c: node_ids[None, :] + 0.001 * (i + 1) * k[:, None] for i, c in enumerate(NODAL)},
        )
        base = eids[:, None] + 0.1 * np.arange(gp.shape[0])[None, :]  # (E, G)
        w.write_gauss_group(
            sid, "partition_0", "group_0", class_tag=class_tag, int_rule=0,
            element_index=eids, natural_coords=gp,
            components={c: base[None, :, :] + (0.001 * (i + 1) * k)[:, None, None] for i, c in enumerate(stress)},
        )
        w.end_stage()
    mb = path.stat().st_size / 1048576
    print(f"{path.name}: {node_ids.size} nodes, {eids.size} elements, {n_steps} steps, "
          f"{len(NODAL)} nodal + {len(stress)} x {gp.shape[0]}-point gauss components; "
          f"{mb:.2f} MB in {_time.perf_counter() - t0:.1f} s")


STAGED_STEPS = {"base": 3, "erect": 2, "unprop": 4}


def staged_activation(out_dir: Path) -> None:
    """The V4b staged pair: a composed model.h5 and its results.h5 (see the docstring)."""
    from apeGmsh.opensees import apeSees

    with apeGmsh(model_name="staged_activation", verbose=False, save_to=SESSION_DIR / "staged_activation.h5") as g:
        g.model.geometry.add_box(0, 0, 0, 1, 1, 3, label="col")
        g.model.geometry.add_box(1, 0, 0, 1, 1, 1.5, label="prop")
        g.model.geometry.add_box(0, 0, 3, 2, 1, 0.5, label="top")
        g.model.sync()
        g.physical.add_volume("col", name="Col")
        g.physical.add_volume("prop", name="Prop")
        g.physical.add_volume("top", name="Top")
        g.mesh.sizing.set_global_size(0.5)
        g.mesh.generation.generate(dim=3)
        fem = g.mesh.queries.get_fem_data(dim=3)

    ops = apeSees(fem)
    ops.model(ndm=3, ndf=3)
    conc = ops.nDMaterial.ElasticIsotropic(E=30e9, nu=0.2, rho=2400.0, name="concrete")
    for pg in ("Col", "Prop", "Top"):
        ops.element.FourNodeTetrahedron(pg=pg, material=conc)

    def chain(s) -> None:
        s.analysis(
            test=ops.test.NormDispIncr(tol=1e-6, max_iter=10),
            algorithm=ops.algorithm.Newton(),
            integrator=ops.integrator.LoadControl(dlam=0.5),
            constraints=ops.constraints.Plain(),
            numberer=ops.numberer.RCM(),
            system=ops.system.UmfPack(),
            analysis=ops.analysis.Static(),
        )

    with ops.stage(name="base") as s:
        s.support(pg="Col", dofs=(1, 1, 1))
        chain(s)
        s.run(n_increments=STAGED_STEPS["base"])
    with ops.stage(name="erect") as s:
        s.activate(pgs=["Top"])
        chain(s)
        s.run(n_increments=STAGED_STEPS["erect"])
    with ops.stage(name="unprop") as s:
        s.remove_element(pg="Prop")
        chain(s)
        s.run(n_increments=STAGED_STEPS["unprop"])
    model_path = out_dir / "staged_activation.h5"
    ops.h5(str(model_path))

    node_ids = np.asarray(fem.nodes.ids, dtype=np.int64)
    res_path = out_dir / "staged_activation.results.h5"
    with NativeWriter(res_path) as w:
        w.open(fem=fem, source_type="domain_capture", analysis_label="V4b synthetic (#1728)", model_h5_src=model_path)
        for name, n in STAGED_STEPS.items():
            k = np.arange(n, dtype=np.float64)
            sid = w.begin_stage(name=name, kind="static", time=(k + 1) / n)
            w.write_nodes(sid, "partition_0", node_ids=node_ids,
                          components={"displacement_x": node_ids[None, :] + 0.001 * k[:, None]})
            w.end_stage()
        sid = w.begin_stage(name="mode_1", kind="mode", time=np.zeros(1),
                            eigenvalue=4.0, frequency_hz=1 / np.pi, period_s=np.pi, mode_index=1)
        w.write_nodes(sid, "partition_0", node_ids=node_ids, components={"displacement_x": node_ids[None, :] * 0.0})
        w.end_stage()
    # The same stages with no model_h5_src: no /opensees, so no model_hash. It
    # pairs with the model by fem_hash only, which proves the mesh and not the
    # program (the V4b review's reproducer: activation must stay unbound).
    fem_only = out_dir / "staged_activation.femonly.results.h5"
    with NativeWriter(fem_only) as w:
        w.open(fem=fem, source_type="domain_capture", analysis_label="V4b synthetic, fem_hash only (#1728)")
        for name, n in STAGED_STEPS.items():
            k = np.arange(n, dtype=np.float64)
            sid = w.begin_stage(name=name, kind="static", time=(k + 1) / n)
            w.write_nodes(sid, "partition_0", node_ids=node_ids,
                          components={"displacement_x": node_ids[None, :] + 0.001 * k[:, None]})
            w.end_stage()
    print(f"wrote {model_path.name}, {res_path.name} and {fem_only.name}: {node_ids.size} nodes, {len(fem.elements.ids)} elements")
    staged_oracle(model_path, out_dir / "staged_activation.oracle.json")


def staged_oracle(model_path: Path, out: Path) -> None:
    """Python's activation per program stage, the oracle the app's test compares with.

    ``build_from_model`` (``viewers/data/_stage_activation.py``) on the model as
    ``OpenSeesModel.from_h5`` reads it, over a grid whose cell ``i`` is the
    ``i``-th FEM element id: the hidden FEM ids of each stage, by name.
    """
    import json

    from apeGmsh.opensees import OpenSeesModel
    from apeGmsh.viewers.data._stage_activation import build_from_model

    model = OpenSeesModel.from_h5(str(model_path))
    eids = [int(e) for e in model.fem.elements.ids]
    amap = build_from_model(model, {e: i for i, e in enumerate(eids)}, len(eids))
    if amap is None:
        raise SystemExit(f"{model_path}: build_from_model found no program stages")
    hidden = {name: sorted(eids[i] for i in np.flatnonzero(mask)) for name, mask in amap.hidden_by_name.items()}
    out.write_text(json.dumps({"source": "apeGmsh.viewers.data._stage_activation.build_from_model",
                               "model": model_path.name, "elements": len(eids),
                               "unmapped_tags": amap.unmapped_tags, "hidden_fem_ids": hidden}, indent=1) + "\n")
    print(f"wrote {out.name}: " + ", ".join(f"{k} hides {len(v)}" for k, v in hidden.items()))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--measure", type=Path, help="write the measured files a, b, c into this directory")
    ap.add_argument("--only", choices=["a", "b", "c"], action="append")
    ap.add_argument("--tet-size", type=float, default=0.0335)
    ap.add_argument("--staged", action="store_true", help="write the V4b staged pair (staged_activation.h5 + .results.h5)")
    args = ap.parse_args()
    if args.staged:
        staged_activation(HERE)
        return
    if args.measure is None:
        write(HERE / "results_small.h5", wedge_mesh(2, 1, "results_small"), 5, ["stress_xx", "stress_yy"], WEDGE_GP, 0)
        return
    out = args.measure
    out.mkdir(parents=True, exist_ok=True)
    todo = set(args.only or ["a", "b", "c"])
    if todo & {"a", "c"}:
        fem = wedge_mesh(13, 21, "v4a_maintainer")
        if "a" in todo:
            write(out / "a_maintainer_100.results.h5", fem, 100, STRESS6, WEDGE_GP, 0)
        if "c" in todo:
            write(out / "c_maintainer_5000.results.h5", fem, 5000, ["stress_xx"], WEDGE_GP, 0)
    if "b" in todo:
        write(out / "b_stress_box_100.results.h5", tet_blocks(args.tet_size, "v4a_stress_box"), 100, STRESS6, TET_GP, 179)


if __name__ == "__main__":
    main()
