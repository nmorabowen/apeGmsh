"""#1971: the ``.out`` transcoder NaNs rotations a node's ndf does not reach.

One ``recorder Node`` line over a mixed-ndf model writes every requested DOF
for every node and pads the DOFs a node lacks. The transcoder reads the
model's ``/opensees/nodes_ndf`` and writes NaN there, so a tet node beside a
shell (3-dof next to 6-dof) or a quad node beside a frame (2-dof next to
3-dof) never reads a fabricated rotation.

The oracle is closed form: synthetic ``.out`` files hold
``value(node, dof)`` in every slot, padded or not, so a kept value must equal
it exactly and a masked one is decided by the model's per-node ndf alone.
The models are real bridge archives (``apeSees(fem).h5``), so the per-node
ndf is the bridge's own inference; no OpenSees process runs.
"""
from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pytest

from apeGmsh.opensees import apeSees
from apeGmsh.results import Results
from apeGmsh.results.spec._resolved import (
    ResolvedRecorderRecord,
    ResolvedRecorderSpec,
)
from apeGmsh.results.transcoders import RecorderTranscoder

from tests.conftest import _open_model_from_h5

TIME = (0.5, 1.0)


def _value(k: int, nid: int, dof: int) -> float:
    """Non-zero in every slot: the mask must not key on the padded 0.0."""
    return 1.0 + 0.01 * nid + 0.001 * dof + 0.0001 * k


def _write_out(path: Path, node_ids: list[int], dofs: list[int]) -> None:
    rows = []
    for k, t in enumerate(TIME):
        cols = [t] + [_value(k, n, d) for n in node_ids for d in dofs]
        rows.append(" ".join(f"{v:.10g}" for v in cols))
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")


def _nodes_ndf(model_h5: Path) -> dict[int, int]:
    with h5py.File(model_h5, "r") as f:
        grp = f["opensees/nodes_ndf"]
        return {int(t): int(n) for t, n in zip(grp["tags"][...], grp["ndf"][...])}


# ---------------------------------------------------------------------------
# Models: 3-D shell beside a solid, 2-D frame beside a quad block
# ---------------------------------------------------------------------------


def _shell_beside_solid(g: Any, tmp_path: Path) -> tuple[Any, Path, list[int], list[int]]:
    import gmsh

    G = g.model.geometry
    G.add_box(0, 0, -1.0, 1.0, 1.0, 1.0)
    p = [G.add_point(*xyz) for xyz in
         ((0, 0, 0.5), (1.0, 0, 0.5), (1.0, 0, 1.5), (0, 0, 1.5))]
    lines = [G.add_line(p[i], p[(i + 1) % 4]) for i in range(4)]
    wall = G.add_plane_surface([G.add_curve_loop(lines)])
    g.model.sync()
    g.physical.add(3, [t for _d, t in gmsh.model.getEntities(3)], name="Solid")
    g.physical.add(2, [wall], name="Wall")
    g.mesh.structured.set_recombine("Wall", dim=2)
    g.mesh.sizing.set_global_size(1.0)
    g.mesh.generation.generate(3)
    g.mesh.partitioning.renumber(base=1)
    fem = g.mesh.queries.get_fem_data()

    ops = apeSees(fem)
    ops.model(ndm=3, ndf=6)
    ops.element.FourNodeTetrahedron(
        pg="Solid", material=ops.nDMaterial.ElasticIsotropic(E=30e9, nu=0.2))
    ops.element.ShellMITC4(
        pg="Wall",
        section=ops.section.ElasticMembranePlateSection(E=30e9, nu=0.0, h=0.1))
    model_h5 = tmp_path / "model.h5"
    ops.h5(str(model_h5))
    solid = sorted({int(n) for n in fem.nodes.select(pg="Solid").ids})
    shell = sorted({int(n) for n in fem.nodes.select(pg="Wall").ids})
    return fem, model_h5, solid, shell


def _frame_beside_quad(g: Any, tmp_path: Path) -> tuple[Any, Path, list[int], list[int]]:
    G = g.model.geometry
    block = G.add_rectangle(0.0, 0.0, 0.0, 1.0, 1.0)
    col = G.add_line(G.add_point(3.0, 0.0, 0.0), G.add_point(3.0, 2.0, 0.0))
    g.model.sync()
    g.physical.add(2, [block], name="Block")
    g.physical.add(1, [col], name="Col")
    g.mesh.structured.set_recombine("Block", dim=2)
    g.mesh.sizing.set_global_size(0.5)
    g.mesh.generation.generate(2)
    g.mesh.partitioning.renumber(base=1)
    fem = g.mesh.queries.get_fem_data()

    ops = apeSees(fem)
    ops.model(ndm=2, ndf=3)
    ops.element.FourNodeQuad(
        pg="Block", thickness=1.0,
        material=ops.nDMaterial.ElasticIsotropic(E=1e6, nu=0.3))
    ops.element.elasticBeamColumn(
        pg="Col", transf=ops.geomTransf.Linear(), A=0.01, E=200e9, Iz=1e-4)
    model_h5 = tmp_path / "model.h5"
    ops.h5(str(model_h5))
    quad = sorted({int(n) for n in fem.nodes.select(pg="Block").ids})
    frame = sorted({int(n) for n in fem.nodes.select(pg="Col").ids})
    return fem, model_h5, quad, frame


def _transcode(
    fem: Any, tmp_path: Path, node_ids: list[int],
    components: tuple[str, ...], ops_type: str, dofs: list[int],
    model_h5: Path | None,
) -> Results:
    out = tmp_path / "out"
    out.mkdir(exist_ok=True)
    spec = ResolvedRecorderSpec(fem_snapshot_id=fem.snapshot_id, records=(
        ResolvedRecorderRecord(
            category="nodes", name="all", components=components,
            dt=None, n_steps=None, node_ids=np.array(node_ids),
        ),
    ))
    _write_out(out / f"all_{ops_type}.out", node_ids, dofs)
    target = tmp_path / ("with_model.h5" if model_h5 else "no_model.h5")
    RecorderTranscoder(spec, out, target, fem, model_h5_src=model_h5).run()
    return Results.from_native(target, fem=fem,
                               model=_open_model_from_h5(target))


def _last(r: Results, comp: str) -> dict[int, float]:
    slab = r.nodes.get(component=comp)
    return {int(i): float(v) for i, v in zip(slab.node_ids,
                                             np.asarray(slab.values)[-1])}


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

#: (translational prefix, rotational prefix, OpenSees recorder type)
FAMILIES = [
    ("displacement", "rotation", "disp"),
    ("reaction_force", "reaction_moment", "reaction"),
]


@pytest.mark.parametrize("trans, rot, ops_type", FAMILIES)
def test_3dof_solid_nodes_read_nan_rotation(
    g: Any, tmp_path: Path, trans: str, rot: str, ops_type: str,
) -> None:
    fem, model_h5, solid, shell = _shell_beside_solid(g, tmp_path)
    ndf = _nodes_ndf(model_h5)
    assert {ndf[n] for n in solid} == {3}      # the mixed-ndf premise
    assert {ndf[n] for n in shell} == {6}
    comps = tuple(f"{p}_{a}" for p in (trans, rot) for a in "xyz")
    r = _transcode(fem, tmp_path, solid + shell, comps, ops_type,
                   [1, 2, 3, 4, 5, 6], model_h5)
    k = len(TIME) - 1
    for d, comp in enumerate(comps, start=1):
        got = _last(r, comp)
        assert sorted(got) == sorted(solid + shell), comp
        for n in shell:
            assert got[n] == pytest.approx(_value(k, n, d), rel=1e-9), comp
        for n in solid:
            if d <= 3:
                assert got[n] == pytest.approx(_value(k, n, d), rel=1e-9)
            else:
                assert np.isnan(got[n]), f"{comp} at 3-dof node {n}"


def test_2dof_quad_nodes_read_nan_rotation_z(g: Any, tmp_path: Path) -> None:
    fem, model_h5, quad, frame = _frame_beside_quad(g, tmp_path)
    ndf = _nodes_ndf(model_h5)
    assert {ndf[n] for n in quad} == {2}
    assert {ndf[n] for n in frame} == {3}
    # The reader's file convention labels rotation_z DOF 6 whatever the
    # ndm (opensees._results_route._READER_LAYOUT), so the mask follows
    # the node's own layout, not the label: a 2-D frame node (ndf 3)
    # carries rz, a quad node (ndf 2) does not.
    r = _transcode(fem, tmp_path, quad + frame,
                   ("displacement_x", "displacement_y", "rotation_z"),
                   "disp", [1, 2, 6], model_h5)
    k = len(TIME) - 1
    rz = _last(r, "rotation_z")
    assert all(np.isnan(rz[n]) for n in quad)
    assert all(rz[n] == pytest.approx(_value(k, n, 6), rel=1e-9)
               for n in frame)
    ux = _last(r, "displacement_x")
    assert all(ux[n] == pytest.approx(_value(k, n, 1), rel=1e-9)
               for n in quad + frame)


def test_without_a_model_the_values_are_kept(g: Any, tmp_path: Path) -> None:
    """No ``model_h5_src``: the transcode is unchanged (no per-node ndf)."""
    fem, _model_h5, solid, shell = _shell_beside_solid(g, tmp_path)
    r = _transcode(fem, tmp_path, solid + shell, ("rotation_x",), "disp",
                   [4], None)
    k = len(TIME) - 1
    got = _last(r, "rotation_x")
    assert all(got[n] == pytest.approx(_value(k, n, 4), rel=1e-9)
               for n in solid + shell)


def test_a_node_missing_from_nodes_ndf_refuses(g: Any, tmp_path: Path) -> None:
    fem, model_h5, solid, shell = _shell_beside_solid(g, tmp_path)
    cut = tmp_path / "cut_model.h5"
    shutil.copy(model_h5, cut)
    with h5py.File(cut, "a") as f:
        grp = f["opensees/nodes_ndf"]
        tags, ndf = grp["tags"][...], grp["ndf"][...]
        keep = tags != solid[0]
        del grp["tags"], grp["ndf"]
        grp.create_dataset("tags", data=tags[keep])
        grp.create_dataset("ndf", data=ndf[keep])
    with pytest.raises(ValueError, match=rf"nodes \[{solid[0]}\]"):
        _transcode(fem, tmp_path, solid + shell, ("rotation_x",), "disp",
                   [4], cut)
