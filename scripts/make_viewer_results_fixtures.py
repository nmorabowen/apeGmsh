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
* ``stamped`` (committed, ``fixtures/results_stamped.h5``, ``--stamped``;
  V4g, #1768): the ``small`` file written at the writer's current results
  version with ``NativeWriter.stamp_element_index_space("fem")``, so its
  element rows are verified FEM ids (h5-schema.md §``element_index_space``).
  ``results_small.h5`` stays unstamped: a ``domain_capture`` file without
  the stamp, which reads as unverified. ``--stamped`` also writes
  ``fixtures/stamp_variants/`` (``stamp_variants``): the stamp in other
  HDF5 encodings, with Python's verdict on each in an oracle JSON.
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

``--contour`` (V4c1, #1756; committed) writes the contour pairs: the same
staged model as ``fixtures/contour_staged.h5`` with ``displacement_x/y/z``
closed forms of the node coordinates (``contour_fixtures``), and
``fixtures/contour_nodata.h5`` (the same model file) whose results record
only the nodes with ``z <= 3``. ``--measure`` also writes each measured
file's model beside it, ``<stem>.h5``, from the same FEMData, so the app
pairs the two by ``fem_hash`` for the contour scrub rows.
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


def write(path: Path, fem, n_steps: int, stress: list[str], gp: np.ndarray, class_tag: int, *, stamp: bool = False) -> None:
    node_ids = np.asarray(fem.nodes.ids, dtype=np.int64)
    eids = np.asarray(fem.elements.ids, dtype=np.int64)
    k = np.arange(n_steps, dtype=np.float64)
    t0 = _time.perf_counter()
    with NativeWriter(path) as w:
        w.open(fem=fem, source_type="domain_capture", analysis_label="V4a synthetic (#1707)")
        if stamp:
            # The element rows are fem.elements.ids by construction (V4g, #1768).
            w.stamp_element_index_space("fem")
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


def staged_model(out_dir: Path, name: str):
    """The three-box staged model (see the docstring), written to ``out_dir/<name>.h5``; returns ``(fem, path)``."""
    from apeGmsh.opensees import apeSees

    with apeGmsh(model_name=name, verbose=False, save_to=SESSION_DIR / f"{name}.h5") as g:
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
    model_path = out_dir / f"{name}.h5"
    ops.h5(str(model_path))
    return fem, model_path


def staged_activation(out_dir: Path) -> None:
    """The V4b staged pair: a composed model.h5 and its results.h5 (see the docstring)."""
    fem, model_path = staged_model(out_dir, "staged_activation")
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


def contour_values(xyz: np.ndarray, stage: int, k: int, n: int) -> dict[str, np.ndarray]:
    """The V4c1 closed form: one step's displacement_x/y/z of nodes at ``xyz`` (N, 3)."""
    s = 0.01 * (stage + 1) * (k + 1) / n
    x, z = xyz[:, 0], xyz[:, 2]
    return {"displacement_x": s * z, "displacement_y": 0.5 * s * x, "displacement_z": -0.2 * s * (x + z)}


def contour_fixtures(out_dir: Path) -> None:
    """The V4c1 contour pairs (#1756): the staged model with a vector field, and a copy whose results miss nodes.

    ``contour_staged.h5`` is the three-box staged model; ``contour_staged.results.h5``
    holds the three program stages with ``displacement_x/y/z`` of every node,
    closed forms of the node coordinates (``contour_values``):

        s = 0.01 * (stage + 1) * (k + 1) / T
        displacement_x = s * z,  displacement_y = 0.5 * s * x,  displacement_z = -0.2 * s * (x + z)

    ``contour_nodata.h5`` is the same model file under another stem (the same
    hashes), and ``contour_nodata.results.h5`` records only the nodes with
    ``z <= 3`` (the ``Top`` box's own nodes are absent: "no data").
    """
    import shutil

    fem, model_path = staged_model(out_dir, "contour_staged")
    node_ids = np.asarray(fem.nodes.ids, dtype=np.int64)
    xyz = np.asarray(fem.nodes.coords, dtype=np.float64).reshape(-1, 3)
    nodata_model = out_dir / "contour_nodata.h5"
    shutil.copyfile(model_path, nodata_model)
    keep = xyz[:, 2] <= 3.0 + 1e-9
    for res_path, ids, coords in (
        (out_dir / "contour_staged.results.h5", node_ids, xyz),
        (out_dir / "contour_nodata.results.h5", node_ids[keep], xyz[keep]),
    ):
        with NativeWriter(res_path) as w:
            w.open(fem=fem, source_type="domain_capture", analysis_label="V4c1 synthetic (#1756)", model_h5_src=model_path)
            for si, (name, n) in enumerate(STAGED_STEPS.items()):
                k = np.arange(n, dtype=np.float64)
                sid = w.begin_stage(name=name, kind="static", time=(k + 1) / n)
                steps = [contour_values(coords, si, kk, n) for kk in range(n)]
                w.write_nodes(sid, "partition_0", node_ids=ids,
                              components={c: np.stack([st[c] for st in steps]) for c in steps[0]})
                w.end_stage()
        print(f"wrote {res_path.name}: {ids.size} of {node_ids.size} nodes")


DEFORM_STEPS = 3


def deform_values(xyz: np.ndarray, k: int) -> dict[str, np.ndarray]:
    """The V4d 2-D closed form: one step's displacement_x/y of nodes at ``xyz`` (N, 3); no z (ndm 2)."""
    s = 0.01 * (k + 1)
    x, y = xyz[:, 0], xyz[:, 1]
    return {"displacement_x": s * y, "displacement_y": s * (0.5 * x + 0.25 * y)}


def deform_fixtures(out_dir: Path) -> None:
    """The V4d 2-D pair (#1765): a plane model declared ``ndm=2``, with quads and truss chords.

    ``deform_2d.h5`` is composed by the bridge with ``ops.model(ndm=2, ndf=2)``,
    so ``/meta/ndm`` is 2 (neutral 2.34.0 on): a 2 x 1 rectangle of 4 x 2
    ``FourNodeQuad`` (``Panel``) at z = 0, and a 2 m tie of 4 ``Truss``
    segments (``Tie``) along y = 1.5. ``deform_2d.results.h5`` holds one
    static stage, ``load``, of ``DEFORM_STEPS`` steps with only
    ``displacement_x`` and ``displacement_y`` (the shorthand clipped to
    ``ndm``, as h5-schema.md says a 2-D file has no ``displacement_z``)::

        s = 0.01 * (k + 1)
        displacement_x = s * y,  displacement_y = s * (0.5 * x + 0.25 * y)
    """
    from apeGmsh.opensees import apeSees

    name = "deform_2d"
    with apeGmsh(model_name=name, verbose=False, save_to=SESSION_DIR / f"{name}.h5") as g:
        p0 = gmsh.model.geo.addPoint(0, 0, 0)
        edge = gmsh.model.geo.extrude([(0, p0)], 2, 0, 0, numElements=[4])
        surf = gmsh.model.geo.extrude([edge[1]], 0, 1, 0, numElements=[2], recombine=True)
        q0 = gmsh.model.geo.addPoint(0, 1.5, 0)
        tie = gmsh.model.geo.extrude([(0, q0)], 2, 0, 0, numElements=[4])
        gmsh.model.geo.synchronize()
        g.model.sync()
        g.physical.add(2, [surf[1][1]], name="Panel")
        g.physical.add(1, [tie[1][1]], name="Tie")
        g.mesh.generation.generate(2)
        fem = g.mesh.queries.get_fem_data(remove_orphans=True)

    ops = apeSees(fem)
    ops.model(ndm=2, ndf=2)
    plate = ops.nDMaterial.ElasticIsotropic(E=30e9, nu=0.2, rho=2400.0, name="plate")
    steel = ops.uniaxialMaterial.ElasticMaterial(E=200e9, name="steel")
    ops.element.FourNodeQuad(pg="Panel", thickness=0.2, material=plate)
    ops.element.Truss(pg="Tie", A=1e-3, material=steel)
    model_path = out_dir / f"{name}.h5"
    ops.h5(str(model_path))

    node_ids = np.asarray(fem.nodes.ids, dtype=np.int64)
    xyz = np.asarray(fem.nodes.coords, dtype=np.float64).reshape(-1, 3)
    res_path = out_dir / f"{name}.results.h5"
    with NativeWriter(res_path) as w:
        w.open(fem=fem, source_type="domain_capture", analysis_label="V4d synthetic 2-D (#1765)", model_h5_src=model_path)
        k = np.arange(DEFORM_STEPS, dtype=np.float64)
        sid = w.begin_stage(name="load", kind="static", time=(k + 1) / DEFORM_STEPS)
        steps = [deform_values(xyz, kk) for kk in range(DEFORM_STEPS)]
        w.write_nodes(sid, "partition_0", node_ids=node_ids,
                      components={c: np.stack([st[c] for st in steps]) for c in steps[0]})
        w.end_stage()
    print(f"wrote {model_path.name} and {res_path.name}: {node_ids.size} nodes, {len(fem.elements.ids)} elements, ndm 2")


def stamp_variants(out_dir: Path) -> None:
    """V4g review (#1810): ``results_stamped.h5`` with the stamp re-encoded, and Python's verdict on each.

    Each file is a copy of ``results_stamped.h5`` whose root
    ``element_index_space`` is rewritten with h5py in one HDF5 encoding.
    ``stamp_variants.oracle.json`` records what ``NativeReader`` makes of
    each (``element_index_verified`` and the ``element_index_space`` string),
    so the app's verdict is checked against Python's, not against itself.
    Only a scalar variable-length UTF-8 string ``"fem"`` is what
    ``NativeWriter.stamp_element_index_space`` writes, and the only form
    that verifies (h5-schema.md §``element_index_space``, #1811).
    """
    import json
    import shutil

    import h5py

    from apeGmsh.results.readers._native import NativeReader

    key = "element_index_space"
    cases = {
        "vlen_str": lambda a: a.create(key, "fem", dtype=h5py.string_dtype("utf-8")),
        "vlen_bytes": lambda a: a.create(key, b"fem", dtype=h5py.string_dtype("ascii")),
        "vlen_ascii_str": lambda a: a.create(key, "fem", dtype=h5py.string_dtype("ascii")),
        "np_bytes_S3": lambda a: a.__setitem__(key, np.bytes_(b"fem")),
        "fixed_S3": lambda a: a.create(key, b"fem", dtype="S3"),
        "fixed_utf8_3": lambda a: a.create(key, "fem".encode(), dtype=h5py.string_dtype("utf-8", length=3)),
        "fixed_utf8_8_nullpad": lambda a: a.create(key, "fem".encode(), dtype=h5py.string_dtype("utf-8", length=8)),
        "fixed_S8_nullpad":lambda a: a.create(key, np.bytes_(b"fem"), dtype="S8"),
        "vlen_array": lambda a: a.create(key, np.array(["fem"], dtype=object), dtype=h5py.string_dtype("utf-8")),
    }
    src = out_dir / "results_stamped.h5"
    vdir = out_dir / "stamp_variants"
    vdir.mkdir(exist_ok=True)
    oracle = {}
    for name, encode in cases.items():
        p = vdir / f"{name}.results.h5"
        shutil.copyfile(src, p)
        with h5py.File(p, "a") as f:
            del f.attrs[key]
            encode(f.attrs)
        with NativeReader(str(p)) as r:
            oracle[name] = {"verified": bool(r.element_index_verified), "element_index_space": r.element_index_space}
    (vdir / "stamp_variants.oracle.json").write_text(
        json.dumps({"source": "apeGmsh.results.readers._native.NativeReader.element_index_verified", "cases": oracle}, indent=1) + "\n"
    )
    print("wrote stamp_variants: " + ", ".join(f"{k}={v['verified']}" for k, v in oracle.items()))


def reinforced_fixture(out_dir: Path) -> None:
    """The V4m reinforced model (#1846): ``reinforced.h5``, its oracle and ``reinforced_unknown.h5``.

    A tet-meshed box with three embedded bars: ``L1`` tied with the bond law
    ``bond1`` (a LadrunoBondSlip no element uses), ``L2`` tied with
    ``bond2`` and its own ``CorotTruss`` cells on the same material, so the
    material has element and tie users, and ``L3`` tied by perfect bond (no
    material). ``reinforced.oracle.json`` is Python's count of ties per bond
    material, from apeGmsh's own reader (``h5_reader.embedded_rebars``), not
    from the app's parse. No results: the window reads the model only.
    """
    import json

    from apeGmsh._kernel.defs.rebar import Cage
    from apeGmsh.opensees import apeSees
    from apeGmsh.opensees.emitter import h5_reader

    name = "reinforced"
    with apeGmsh(model_name=name, verbose=False, save_to=SESSION_DIR / f"{name}.h5") as g:
        g.model.geometry.add_box(0, 0, 0, 0.5, 0.5, 2.0, label="Box")
        g.physical.add_volume(g.model.select(dim=3).tags(), name="Concrete")
        g.physical.add_surface(
            g.model.select(dim=2).in_box((-0.01, -0.01, -0.01), (0.51, 0.51, 0.01)).tags(), name="Base")
        bars = [g.rebar.bar([(x, 0.15, 0.1), (x, 0.15, 1.9)], db=0.0254, material=mat, name=n)
                for x, mat, n in ((0.1, "steel", "L1"), (0.25, "bond2", "L2"), (0.4, "steel", "L3"))]
        g.rebar.place(Cage(bars=(bars[0],)), into="Concrete", coupling="embedded", name="cage1", bond="bond1")
        g.rebar.place(Cage(bars=(bars[1],)), into="Concrete", coupling="embedded", name="cage2", bond="bond2",
                      emit_elements=True)
        g.rebar.place(Cage(bars=(bars[2],)), into="Concrete", coupling="embedded", name="cage3", perfect=1.5e8)
        g.mesh.sizing.set_global_size(0.4)
        g.mesh.generation.generate(dim=3)
        fem = g.mesh.queries.get_fem_data(dim=None)

    ops = apeSees(fem)
    ops.model(ndm=3, ndf=3)
    ops.uniaxialMaterial.LadrunoBondSlip(tau_max=10.5, s1=0.001, s2=0.003, s3=0.01, tau_f=2.5, alpha=0.4, name="bond1")
    ops.uniaxialMaterial.LadrunoBondSlip(tau_max=8.5, s1=0.001, s2=0.003, s3=0.01, tau_f=2.0, alpha=0.4, name="bond2")
    conc = ops.nDMaterial.ElasticIsotropic(E=30000.5, nu=0.2, rho=2.5e-6, name="concrete")
    ops.element.FourNodeTetrahedron(pg="Concrete", material=conc)
    ops.fix(pg="Base", dofs=(1, 1, 1))
    path = out_dir / f"{name}.h5"
    ops.h5(str(path))

    by_tag: dict[int, int] = {}
    unbonded = 0
    with h5_reader.open(str(path)) as m:
        ties = m.embedded_rebars()
    for t in ties:
        if "-bond" in t.args:
            tag = t.args[t.args.index("-bond") + 1]
            assert isinstance(tag, int), t
            by_tag[tag] = by_tag.get(tag, 0) + 1
        else:
            unbonded += 1
    oracle = {"ties": len(ties), "unbonded": unbonded, "bond": {str(k): v for k, v in sorted(by_tag.items())}}
    (out_dir / f"{name}.oracle.json").write_text(json.dumps(oracle, indent=2) + chr(10), encoding="utf-8")
    print(f"wrote {path.name}: {len(ties)} ties, by bond tag {by_tag}, {unbonded} perfect-bond")
    # reinforced_unknown.h5: the same file with the first bonded tie's tag
    # stored as a float (arg_kinds 1), a tag the app must not read: its D7
    # still shows "users unknown", never 0 or "used by nothing".
    import shutil

    import h5py
    unknown = out_dir / f"{name}_unknown.h5"
    shutil.copyfile(path, unknown)
    with h5py.File(unknown, "r+") as f:
        er = f["opensees/constraints/embeddedRebar"]
        strs = [x.decode() if isinstance(x, bytes) else x for x in er["args_str"][()]]
        k = strs.index("-bond") + 1
        kinds = er["arg_kinds"][()]
        kinds[k] = 1
        er["arg_kinds"][...] = kinds
    print(f"wrote {unknown.name}: arg {k} (the first -bond tag) re-kinded as a float")


ELEMENT_STEPS = 3
PLANE_STRESS = ["stress_xx", "stress_yy", "stress_xy"]
HEX_GP = np.array([[a, b, c] for c in (-1, 1) for b in (-1, 1) for a in (-1, 1)]) / np.sqrt(3)  # 2 x 2 x 2


def element_value(eids: np.ndarray, n_gp: int, c: int, k: int) -> np.ndarray:
    """The V4c2a closed form, one step of component ``c`` at every (element, Gauss point): (E, G).

        v[k, e, g] = eid_e + 0.1 * g + 0.25 * (c + 1) * (k + 1)

    With one Gauss point (G == 1) the cell's value is ``eid + 0.25 (c + 1)(k + 1)``;
    the mean of G points adds ``0.05 (G - 1)``.
    """
    return eids[:, None].astype(np.float64) + 0.1 * np.arange(n_gp)[None, :] + 0.25 * (c + 1) * (k + 1)


def element_fixtures(out_dir: Path) -> None:
    """The V4c2a element-contour pairs (#1833), every results file written by ``NativeWriter``.

    * ``elem_mixed.h5`` / ``.results.h5`` (stamped ``element_index_space="fem"``):
      a 2-D model (``ops.model(ndm=2)``), a 2 x 1 panel of 4 x 2 ``FourNodeQuad``
      (``Quads``) beside a 1 x 1 panel of 8 ``Tri31`` (``Tris``). One static stage,
      ``load``, of ``ELEMENT_STEPS`` steps: ``displacement_x/y`` (``deform_values``)
      and ``stress_xx/yy/xy`` in two one-point groups, ``quads_G1`` and ``tris_G1``
      (one field across two element types). The first quad (lowest id) is left
      out of ``quads_G1``: "no data".
    * ``elem_mixed_unstamped.h5`` / ``.results.h5``: the same model file and the
      same results written again without the stamp (unverified).
    * ``elem_badid.h5`` / ``.results.h5`` (stamped): the same, with an
      ``_element_index`` row of id 999999, which the mesh does not have.
    * ``elem_solid.h5`` / ``.results.h5`` (stamped): a 3 x 2 x 2 box of 12 hexahedra
      (FEM-only ``model.h5``, paired by ``fem_hash``), ``displacement_x/y/z``, and
      ``stress_xx``, ``stress_yy`` at 8 Gauss points (``hex_G8``).
    * ``elem_staged.results.h5`` (stamped) for the committed ``contour_staged.h5``
      (``model_h5_src``): the three staged tet stages with ``stress_xx`` at one Gauss
      point of every tet, ``v = eid + 0.25 (k + 1) + stage``.

    Values are ``element_value`` (the closed form the tests restate).
    """
    import shutil

    from apeGmsh.mesh.FEMData import FEMData
    from apeGmsh.opensees import apeSees

    name = "elem_mixed"
    with apeGmsh(model_name=name, verbose=False, save_to=SESSION_DIR / f"{name}.h5") as g:
        p0 = gmsh.model.geo.addPoint(0, 0, 0)
        e0 = gmsh.model.geo.extrude([(0, p0)], 2, 0, 0, numElements=[4])
        quads = gmsh.model.geo.extrude([e0[1]], 0, 1, 0, numElements=[2], recombine=True)
        e1 = gmsh.model.geo.extrude([e0[0]], 1, 0, 0, numElements=[2])
        tris = gmsh.model.geo.extrude([e1[1]], 0, 1, 0, numElements=[2])
        gmsh.model.geo.removeAllDuplicates()
        gmsh.model.geo.synchronize()
        g.model.sync()
        g.physical.add(2, [quads[1][1]], name="Quads")
        g.physical.add(2, [tris[1][1]], name="Tris")
        g.mesh.generation.generate(2)
        fem = g.mesh.queries.get_fem_data(dim=2)
    ops = apeSees(fem)
    ops.model(ndm=2, ndf=2)
    plate = ops.nDMaterial.ElasticIsotropic(E=30e9, nu=0.2, rho=2400.0, name="plate")
    ops.element.FourNodeQuad(pg="Quads", thickness=0.2, material=plate)
    ops.element.Tri31(pg="Tris", thickness=0.2, material=plate)
    model_path = out_dir / f"{name}.h5"
    ops.h5(str(model_path))
    by_npe = {grp.connectivity.shape[1]: np.sort(np.asarray(grp.ids, dtype=np.int64)) for grp in fem.elements}
    quad_ids, tri_ids = by_npe[4], by_npe[3]
    node_ids = np.asarray(fem.nodes.ids, dtype=np.int64)
    xyz = np.asarray(fem.nodes.coords, dtype=np.float64).reshape(-1, 3)
    one_gp = np.array([[0.0, 0.0]])
    tri_gp = np.array([[1 / 3, 1 / 3]])

    def mixed(res: Path, *, stamp: bool, extra_tri: list[int]) -> None:
        with NativeWriter(res) as w:
            w.open(fem=fem, source_type="domain_capture", analysis_label="V4c2a synthetic (#1833)", model_h5_src=model_path)
            if stamp:
                # The element rows are fem.elements ids by construction.
                w.stamp_element_index_space("fem")
            k = np.arange(ELEMENT_STEPS, dtype=np.float64)
            sid = w.begin_stage(name="load", kind="static", time=(k + 1) / ELEMENT_STEPS)
            steps = [deform_values(xyz, kk) for kk in range(ELEMENT_STEPS)]
            w.write_nodes(sid, "partition_0", node_ids=node_ids,
                          components={c: np.stack([st[c] for st in steps]) for c in steps[0]})
            for gid, eids, gp in (("quads_G1", quad_ids[1:], one_gp), ("tris_G1", np.concatenate([tri_ids, np.asarray(extra_tri, dtype=np.int64)]), tri_gp)):
                w.write_gauss_group(
                    sid, "partition_0", gid, class_tag=0, int_rule=0, element_index=eids, natural_coords=gp,
                    components={c: np.stack([element_value(eids, 1, ci, kk) for kk in range(ELEMENT_STEPS)]) for ci, c in enumerate(PLANE_STRESS)},
                )
            w.end_stage()

    mixed(out_dir / "elem_mixed.results.h5", stamp=True, extra_tri=[])
    shutil.copyfile(model_path, out_dir / "elem_mixed_unstamped.h5")
    mixed(out_dir / "elem_mixed_unstamped.results.h5", stamp=False, extra_tri=[])
    shutil.copyfile(model_path, out_dir / "elem_badid.h5")
    mixed(out_dir / "elem_badid.results.h5", stamp=True, extra_tri=[999999])
    print(f"wrote elem_mixed (+ unstamped, badid): {len(quad_ids)} quads, {len(tri_ids)} tris, {node_ids.size} nodes")

    name = "elem_solid"
    with apeGmsh(model_name=name, verbose=False, save_to=SESSION_DIR / f"{name}.h5") as g:
        p = gmsh.model.geo.addPoint(0, 0, 0)
        line = gmsh.model.geo.extrude([(0, p)], 3, 0, 0, numElements=[3])
        surf = gmsh.model.geo.extrude([line[1]], 0, 2, 0, numElements=[2], recombine=True)
        vol = gmsh.model.geo.extrude([surf[1]], 0, 0, 2, numElements=[2], recombine=True)
        gmsh.model.geo.synchronize()
        g.model.sync()
        g.physical.add(3, [vol[1][1]], name="Block")
        g.mesh.generation.generate(3)
        sfem = g.mesh.queries.get_fem_data(dim=3)
    solid_model = out_dir / f"{name}.h5"
    sfem.to_h5(str(solid_model), model_name=name)
    hex_ids = np.sort(np.asarray(sfem.elements.ids, dtype=np.int64))
    snode = np.asarray(sfem.nodes.ids, dtype=np.int64)
    sxyz = np.asarray(sfem.nodes.coords, dtype=np.float64).reshape(-1, 3)
    with NativeWriter(out_dir / f"{name}.results.h5") as w:
        w.open(fem=sfem, source_type="domain_capture", analysis_label="V4c2a synthetic solid (#1833)")
        w.stamp_element_index_space("fem")
        k = np.arange(ELEMENT_STEPS, dtype=np.float64)
        sid = w.begin_stage(name="load", kind="static", time=(k + 1) / ELEMENT_STEPS)
        steps = [contour_values(sxyz, 0, kk, ELEMENT_STEPS) for kk in range(ELEMENT_STEPS)]
        w.write_nodes(sid, "partition_0", node_ids=snode,
                      components={c: np.stack([st[c] for st in steps]) for c in steps[0]})
        w.write_gauss_group(
            sid, "partition_0", "hex_G8", class_tag=0, int_rule=0, element_index=hex_ids, natural_coords=HEX_GP,
            components={c: np.stack([element_value(hex_ids, 8, ci, kk) for kk in range(ELEMENT_STEPS)]) for ci, c in enumerate(["stress_xx", "stress_yy"])},
        )
        w.end_stage()
    print(f"wrote elem_solid: {hex_ids.size} hexahedra, {snode.size} nodes, 8 Gauss points")

    staged_model_path = out_dir / "contour_staged.h5"
    tfem = FEMData.from_h5(str(staged_model_path))
    tet_ids = np.sort(np.asarray(tfem.elements.ids, dtype=np.int64))
    with NativeWriter(out_dir / "elem_staged.results.h5") as w:
        w.open(fem=tfem, source_type="domain_capture", analysis_label="V4c2a synthetic staged (#1833)", model_h5_src=staged_model_path)
        w.stamp_element_index_space("fem")
        for si, (sname, n) in enumerate(STAGED_STEPS.items()):
            k = np.arange(n, dtype=np.float64)
            sid = w.begin_stage(name=sname, kind="static", time=(k + 1) / n)
            w.write_gauss_group(
                sid, "partition_0", "tets_G1", class_tag=0, int_rule=0, element_index=tet_ids, natural_coords=TET_GP,
                components={"stress_xx": np.stack([element_value(tet_ids, 1, 0, kk) + si for kk in range(n)])},
            )
            w.end_stage()
    print(f"wrote elem_staged.results.h5: {tet_ids.size} tets over {len(STAGED_STEPS)} stages")


QUAD_GP = np.array([[a, b] for b in (-1, 1) for a in (-1, 1)]) / np.sqrt(3)  # 2 x 2


def extrap_field(xyz: np.ndarray, c: int, k: int) -> np.ndarray:
    """The V4c2b field at physical points ``xyz`` (n, 3): not linear, so extrapolation is not trivial.

        f = (c + 1) (k + 1) (1 + x^2 - 0.5 x y + 0.3 y^2 + y z - 0.2 z^3)
    """
    x, y, z = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    return (c + 1) * (k + 1) * (1 + x * x - 0.5 * x * y + 0.3 * y * y + y * z - 0.2 * z ** 3)


def extrapolation_fixtures(out_dir: Path) -> None:
    """The V4c2b Gauss-to-node extrapolation pairs (#1875) and Python's answer on them.

    Each pair is a FEM-only ``model.h5`` and a stamped ``results.h5`` written by
    ``NativeWriter``, one static stage ``load`` of ``ELEMENT_STEPS`` steps, whose
    Gauss values are ``extrap_field`` at each Gauss point's physical position:

    * ``extrap_plane``: a 2 x 1 panel of 4 x 2 quads, 4 Gauss points each
      (``quads_G4``), beside a 1 x 1 panel of 8 triangles, one point each
      (``tris_G1``); ``stress_xx/yy/xy``. The first quad is left out: no data.
    * ``extrap_solid``: a 3 x 2 x 2 box of 12 hexahedra, 8 points (``hex_G8``).
    * ``extrap_tet``: a 2 x 2 x 2 structured tetrahedral box, one point per tet (``tets_G1``).

    ``extrap.oracle.json`` holds, per pair, component and step, apeGmsh's own
    ``extrapolate_gauss_slab_to_nodes`` (``gauss_node``),
    ``extrapolate_gauss_slab_per_element`` (``gauss_node_discrete``), and the
    averaged path on each element's mean as one point (``cell_averaged``). The
    slab is read back from the written file, and the Qt fast path
    (``GaussToNodeAccumulator``) is checked equal to ``gauss_node`` here.
    """
    import json

    from apeGmsh.fem._shape_functions import get_shape_functions
    from apeGmsh.mesh.FEMData import FEMData
    from apeGmsh.results._gauss_extrapolation import (
        GaussToNodeAccumulator,
        extrapolate_gauss_slab_per_element,
        extrapolate_gauss_slab_to_nodes,
    )
    from apeGmsh.results._slabs import GaussSlab

    def gp_xyz(fem, eids: np.ndarray, gp: np.ndarray) -> np.ndarray:
        """Physical (E, G, 3) positions of the Gauss points, through the corner shape functions."""
        nid = np.asarray(fem.nodes.ids, dtype=np.int64)
        xyz = np.asarray(fem.nodes.coords, dtype=np.float64).reshape(-1, 3)
        at = {int(n): i for i, n in enumerate(nid)}
        out = np.zeros((eids.size, gp.shape[0], 3))
        for grp in fem.elements:
            N_fn, _, _, nc = get_shape_functions(int(grp.element_type.code))
            N = N_fn(gp)  # (G, nc)
            row = {int(e): r for r, e in enumerate(np.asarray(grp.ids, dtype=np.int64))}
            conn = np.asarray(grp.connectivity, dtype=np.int64)
            for i, e in enumerate(eids):
                r = row.get(int(e))
                if r is not None:
                    out[i] = N @ xyz[[at[int(n)] for n in conn[r, :nc]]]
        return out

    def write_pair(name: str, fem, groups: list[tuple[str, np.ndarray, np.ndarray]], comps: list[str]) -> dict:
        model = out_dir / f"{name}.h5"
        fem.to_h5(str(model), model_name=name)
        res = out_dir / f"{name}.results.h5"
        with NativeWriter(res) as w:
            w.open(fem=fem, source_type="domain_capture", analysis_label="V4c2b synthetic (#1875)")
            w.stamp_element_index_space("fem")
            k = np.arange(ELEMENT_STEPS, dtype=np.float64)
            sid = w.begin_stage(name="load", kind="static", time=(k + 1) / ELEMENT_STEPS)
            for gid, eids, gp in groups:
                pts = gp_xyz(fem, eids, gp).reshape(-1, 3)
                w.write_gauss_group(
                    sid, "partition_0", gid, class_tag=0, int_rule=0, element_index=eids, natural_coords=gp,
                    components={c: np.stack([extrap_field(pts, ci, kk).reshape(eids.size, gp.shape[0]) for kk in range(ELEMENT_STEPS)]) for ci, c in enumerate(comps)},
                )
            w.end_stage()
        # Python's answer, on the slab read back from the file (every group of the stage, element-major).
        import h5py
        fem_back = FEMData.from_h5(str(model))
        oracle: dict = {}
        with h5py.File(res, "r") as f:
            gg = f[f"stages/{list(f['stages'])[0]}/partitions/partition_0/elements/gauss_points"]
            for c in comps:
                vals, eidx, nat = [], [], []
                for gid in gg:
                    g = gg[gid]
                    E, G = g["_element_index"].shape[0], g["_natural_coords"].shape[0]
                    vals.append(np.asarray(g[c][...], dtype=np.float64).reshape(ELEMENT_STEPS, E * G))
                    eidx.append(np.repeat(np.asarray(g["_element_index"][...], dtype=np.int64), G))
                    nat.append(np.tile(np.asarray(g["_natural_coords"][...], dtype=np.float64), (E, 1)))
                slab = GaussSlab(component=c, values=np.concatenate(vals, axis=1), element_index=np.concatenate(eidx),
                                 natural_coords=np.concatenate(nat), local_axes_quaternion=None, time=np.arange(ELEMENT_STEPS, dtype=np.float64))
                nids, nodal = extrapolate_gauss_slab_to_nodes(slab, fem_back)
                per = extrapolate_gauss_slab_per_element(slab, fem_back)
                acc = GaussToNodeAccumulator(slab.element_index, slab.natural_coords, fem_back)
                for kk in range(ELEMENT_STEPS):
                    assert np.array_equal(acc.node_ids, nids)
                    assert np.allclose(acc.accumulate(slab.values[kk]), nodal[kk], rtol=1e-12, atol=0), f"{name} {c}: the Qt accumulator disagrees"
                # cell_averaged: each element's mean as its one point, through the same averaged path.
                ue, inv = np.unique(slab.element_index, return_inverse=True)
                means = np.stack([np.bincount(inv, weights=slab.values[kk]) / np.bincount(inv) for kk in range(ELEMENT_STEPS)])
                one = GaussSlab(component=c, values=means, element_index=ue, natural_coords=np.zeros((ue.size, 3)),
                                local_axes_quaternion=None, time=slab.time)
                cids, cnodal = extrapolate_gauss_slab_to_nodes(one, fem_back)
                oracle[c] = {
                    "gauss_node": {"node_ids": nids.tolist(), "values": nodal.tolist()},
                    "gauss_node_discrete": {
                        "element_ids": per.element_ids.tolist(),
                        "corner_node_ids": [x.tolist() for x in per.corner_node_ids],
                        "values": [x.tolist() for x in per.values],
                    },
                    "cell_averaged": {"node_ids": cids.tolist(), "values": cnodal.tolist()},
                }
        print(f"wrote {name}: {len(fem.elements.ids)} elements, groups {[g[0] for g in groups]}")
        return oracle

    oracle: dict = {}

    name = "extrap_plane"
    with apeGmsh(model_name=name, verbose=False, save_to=SESSION_DIR / f"{name}.h5") as g:
        p0 = gmsh.model.geo.addPoint(0, 0, 0)
        e0 = gmsh.model.geo.extrude([(0, p0)], 2, 0, 0, numElements=[4])
        quads = gmsh.model.geo.extrude([e0[1]], 0, 1, 0, numElements=[2], recombine=True)
        e1 = gmsh.model.geo.extrude([e0[0]], 1, 0, 0, numElements=[2])
        tris = gmsh.model.geo.extrude([e1[1]], 0, 1, 0, numElements=[2])
        gmsh.model.geo.removeAllDuplicates()
        gmsh.model.geo.synchronize()
        g.model.sync()
        g.physical.add(2, [quads[1][1]], name="Quads")
        g.physical.add(2, [tris[1][1]], name="Tris")
        g.mesh.generation.generate(2)
        fem = g.mesh.queries.get_fem_data(dim=2)
    by_npe = {grp.connectivity.shape[1]: np.sort(np.asarray(grp.ids, dtype=np.int64)) for grp in fem.elements}
    oracle[name] = write_pair(name, fem, [("quads_G4", by_npe[4][1:], QUAD_GP), ("tris_G1", by_npe[3], np.array([[1 / 3, 1 / 3]]))], PLANE_STRESS)

    name = "extrap_solid"
    with apeGmsh(model_name=name, verbose=False, save_to=SESSION_DIR / f"{name}.h5") as g:
        p = gmsh.model.geo.addPoint(0, 0, 0)
        line = gmsh.model.geo.extrude([(0, p)], 3, 0, 0, numElements=[3])
        surf = gmsh.model.geo.extrude([line[1]], 0, 2, 0, numElements=[2], recombine=True)
        vol = gmsh.model.geo.extrude([surf[1]], 0, 0, 2, numElements=[2], recombine=True)
        gmsh.model.geo.synchronize()
        g.model.sync()
        g.physical.add(3, [vol[1][1]], name="Block")
        g.mesh.generation.generate(3)
        fem = g.mesh.queries.get_fem_data(dim=3)
    oracle[name] = write_pair(name, fem, [("hex_G8", np.sort(np.asarray(fem.elements.ids, dtype=np.int64)), HEX_GP)], ["stress_xx", "stress_yy"])

    name = "extrap_tet"
    with apeGmsh(model_name=name, verbose=False, save_to=SESSION_DIR / f"{name}.h5") as g:
        p = gmsh.model.geo.addPoint(0, 0, 0)
        line = gmsh.model.geo.extrude([(0, p)], 1, 0, 0, numElements=[2])
        surf = gmsh.model.geo.extrude([line[1]], 0, 1, 0, numElements=[2])
        vol = gmsh.model.geo.extrude([surf[1]], 0, 0, 1, numElements=[2])
        gmsh.model.geo.synchronize()
        g.model.sync()
        g.physical.add(3, [vol[1][1]], name="Box")
        g.mesh.generation.generate(3)
        fem = g.mesh.queries.get_fem_data(dim=3)
    oracle[name] = write_pair(name, fem, [("tets_G1", np.sort(np.asarray(fem.elements.ids, dtype=np.int64)), TET_GP)], ["stress_xx"])

    (out_dir / "extrap.oracle.json").write_text(json.dumps(oracle, separators=(",", ":")), encoding="utf8")
    print("wrote extrap.oracle.json")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--measure", type=Path, help="write the measured files a, b, c into this directory")
    ap.add_argument("--only", choices=["a", "b", "c"], action="append")
    ap.add_argument("--tet-size", type=float, default=0.0335)
    ap.add_argument("--staged", action="store_true", help="write the V4b staged pair (staged_activation.h5 + .results.h5)")
    ap.add_argument("--contour", action="store_true", help="write the V4c1 contour pairs (contour_staged.*, contour_nodata.*)")
    ap.add_argument("--deform", action="store_true", help="write the V4d 2-D pair (deform_2d.h5 + .results.h5)")
    ap.add_argument("--stamped", action="store_true", help="write results_stamped.h5, the small file with element_index_space='fem' (V4g)")
    ap.add_argument("--reinforced", action="store_true", help="write reinforced.h5 + oracle, a 2.31 model with embeddedRebar ties (V4m)")
    ap.add_argument("--elements", action="store_true", help="write the V4c2a element-contour pairs (elem_*.h5 + .results.h5)")
    ap.add_argument("--elements-measure", type=Path, help="write the V4c2a measured solid (stamped, G = 6) into this directory")
    ap.add_argument("--extrapolation", action="store_true", help="write the V4c2b extrapolation pairs (extrap_*.h5 + .results.h5) and Python's oracle")
    args = ap.parse_args()
    if args.extrapolation:
        extrapolation_fixtures(HERE)
        return
    if args.reinforced:
        reinforced_fixture(HERE)
        return
    if args.elements:
        element_fixtures(HERE)
        return
    if args.elements_measure is not None:
        out = args.elements_measure
        out.mkdir(parents=True, exist_ok=True)
        fem = wedge_mesh(13, 21, "v4c2a_solid")
        write(out / "e_solid_100.results.h5", fem, 100, ["stress_xx", "stress_yy"], WEDGE_GP, 0, stamp=True)
        fem.to_h5(str(out / "e_solid_100.h5"), model_name="v4c2a_solid")
        return
    if args.deform:
        deform_fixtures(HERE)
        return
    if args.staged:
        staged_activation(HERE)
        return
    if args.contour:
        contour_fixtures(HERE)
        return
    if args.stamped:
        write(HERE / "results_stamped.h5", wedge_mesh(2, 1, "results_stamped"), 5, ["stress_xx", "stress_yy"], WEDGE_GP, 0, stamp=True)
        stamp_variants(HERE)
        return
    if args.measure is None:
        write(HERE / "results_small.h5", wedge_mesh(2, 1, "results_small"), 5, ["stress_xx", "stress_yy"], WEDGE_GP, 0)
        return
    out = args.measure
    out.mkdir(parents=True, exist_ok=True)
    todo = set(args.only or ["a", "b", "c"])
    if todo & {"a", "c"}:
        fem = wedge_mesh(13, 21, "v4a_maintainer")
        # V4c1: the model of each file beside it, as <stem>.h5, written from the
        # same FEMData, so the app pairs the two by fem_hash (the contour rows).
        if "a" in todo:
            write(out / "a_maintainer_100.results.h5", fem, 100, STRESS6, WEDGE_GP, 0)
            fem.to_h5(str(out / "a_maintainer_100.h5"), model_name="v4a_maintainer")
        if "c" in todo:
            write(out / "c_maintainer_5000.results.h5", fem, 5000, ["stress_xx"], WEDGE_GP, 0)
            fem.to_h5(str(out / "c_maintainer_5000.h5"), model_name="v4a_maintainer")
    if "b" in todo:
        write(out / "b_stress_box_100.results.h5", tet_blocks(args.tet_size, "v4a_stress_box"), 100, STRESS6, TET_GP, 179)


if __name__ == "__main__":
    main()
