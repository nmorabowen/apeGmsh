"""Stock fallback: truss and shell-on-solid read back from ``.out`` recorders.

Gap G7 of the DomainCapture parity table (#1767, #1945). Pip ``openseespy``
ships no MPCO recorder, so the route a stock user gets is the classic
``.out`` one: ``ResolvedRecorderSpec.emit_recorders`` into the live domain,
then ``Results.from_recorders``. The two families that run on stock are
solved through the bridge (``apeSees(fem)``, sequential element tags) and
checked against the live domain:

* truss axial force, recorded as a Gauss record, against
  ``eleResponse(tag, "axialForce")`` and ``P`` for the determinate bar;
* mixed-ndf shell-on-solid nodes against ``nodeDisp`` (3-dof tet nodes,
  6-dof MITC4 nodes). The tet nodes read rotation NaN, not the 0.0 the
  recorder pads them with (#1971);
* the 2-D twin, a frame beside a quad block (2-dof quad nodes, 3-dof frame
  nodes): the quad nodes read ``rotation_z`` NaN. The frame's own
  ``rotation_z`` is a strict xfail: in 2-D it is recorded at DOF 6.

The tolerance is the format's, not the oracle's: a ``.out`` file holds six
significant digits, so values compare at ``rtol=1e-5``. The fork-only
families (LadrunoBrick20, LadrunoLST) and the ``.ladruno`` / MPCO routes are
``tests/opensees/integration_ladruno/test_element_families_ladruno_live.py``.

Runs in the ``live-stock`` lane (marker ``live``); it needs only a live
OpenSees backend, stock or fork.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pytest

from apeGmsh.opensees import OpenSeesModel, apeSees
from apeGmsh.opensees.emitter.live import LiveOpsEmitter
from apeGmsh.results import Results
from apeGmsh.results.spec._resolved import (
    ResolvedRecorderRecord,
    ResolvedRecorderSpec,
)

pytestmark = pytest.mark.live

#: Six significant digits in a ``.out`` file.
RTOL_OUT = 1e-5
P = 1000.0
DISP = ("displacement_x", "displacement_y", "displacement_z")
ROT = ("rotation_x", "rotation_y", "rotation_z")


@pytest.fixture(autouse=True)
def _backend() -> None:
    from apeGmsh.opensees.emitter.live import get_backend_name

    try:
        get_backend_name()
    except Exception as e:  # no OpenSees backend importable at all
        pytest.skip(f"no live OpenSees backend: {e}")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _fem_to_ops(model_h5: str) -> dict[int, int]:
    """``{fem_eid: ops_tag}`` from the bridge's own ``element_meta``."""
    with h5py.File(model_h5, "r") as f:
        meta = f["/opensees/element_meta"]
        return {
            int(e): int(t)
            for tok in meta
            for t, e in zip(meta[tok]["ids"][...], meta[tok]["fem_eids"][...])
        }


def _build(ops: apeSees, tmp_path: Path) -> tuple[Any, str, OpenSeesModel]:
    """Persist the model, build it into the live domain."""
    model_h5 = str(tmp_path / "model.h5")
    ops.h5(model_h5)
    emitter = LiveOpsEmitter(wipe=True)
    ops.build().emit(emitter)
    return emitter, model_h5, OpenSeesModel.from_h5(model_h5)


def _record(category: str, name: str, components: tuple[str, ...],
            **ids: Any) -> ResolvedRecorderRecord:
    return ResolvedRecorderRecord(category=category, name=name,
                                  components=components, dt=None,
                                  n_steps=None, **ids)


# ---------------------------------------------------------------------------
# Truss: axial force as a Gauss record
# ---------------------------------------------------------------------------


def _truss_fem(g: Any) -> Any:
    """One bar between two tagged points: its fem_eid (3) is not its tag."""
    G = g.model.geometry
    a = G.add_point(0.0, 0.0, 0.0)
    b = G.add_point(2.0, 0.0, 0.0)
    bar = G.add_line(a, b)
    g.model.sync()
    g.physical.add(1, [bar], name="Bar")
    g.physical.add(0, [a], name="A")
    g.physical.add(0, [b], name="B")
    g.mesh.structured.set_transfinite_curve(bar, 2)
    g.mesh.generation.generate(1)
    return g.mesh.queries.get_fem_data(dim=1)


@pytest.mark.parametrize("cls", ["Truss", "CorotTruss"])
def test_truss_axial_force_reads_back_from_out(
    g: Any, tmp_path: Path, cls: str,
) -> None:
    fem = _truss_fem(g)
    (bar_eid,) = [int(e) for e in fem.elements.physical.element_ids("Bar")]

    ops = apeSees(fem)
    ops.model(ndm=3, ndf=3)
    steel = ops.uniaxialMaterial.ElasticMaterial(E=200e9)
    getattr(ops.element, cls)(pg="Bar", A=1e-3, material=steel)
    ops.fix(pg="A", dofs=(1, 1, 1))
    ops.fix(pg="B", dofs=(0, 1, 1))
    with ops.pattern.Plain(series=ops.timeSeries.Linear()) as p:
        p.load(pg="B", forces=(P, 0.0, 0.0))
    ops.constraints.Plain()
    ops.numberer.Plain()
    ops.system.BandGeneral()
    ops.test.NormDispIncr(tol=1e-10, max_iter=10)
    ops.algorithm.Newton()
    ops.integrator.LoadControl(dlam=0.5)
    ops.analysis.Static()

    emitter, model_h5, model = _build(ops, tmp_path)
    tag = _fem_to_ops(model_h5)[bar_eid]
    assert tag != bar_eid          # the translation is exercised
    spec = ResolvedRecorderSpec(fem_snapshot_id=fem.snapshot_id, records=(
        _record("gauss", "bar", ("axial_force",),
                element_ids=np.array([bar_eid]), element_class_name=cls),
    ))
    out = tmp_path / "out"
    live = []
    with spec.emit_recorders(str(out), ops=emitter.ops, model=model) as rec:
        rec.begin_stage("pull", kind="static")
        for _ in range(2):
            assert emitter.ops.analyze(1) == 0
            live.append(float(emitter.ops.eleResponse(tag, "axialForce")[0]))
        rec.end_stage()
    np.testing.assert_allclose(live, [P / 2, P], rtol=1e-10)

    r = Results.from_recorders(spec, out, fem=fem, stage_id="pull",
                               model=model, cache_root=tmp_path / "cache")
    slab = r.elements.gauss.get(component="axial_force")
    assert [int(e) for e in slab.element_index] == [bar_eid]
    np.testing.assert_allclose(np.asarray(slab.values)[:, 0], live,
                               rtol=RTOL_OUT)


# ---------------------------------------------------------------------------
# Mixed-ndf shell-on-solid: nodal round-trip
# ---------------------------------------------------------------------------


def _shell_on_solid_fem(g: Any) -> Any:
    """A tet footing (3-dof nodes) and a detached MITC4 wall (6-dof nodes)."""
    import gmsh

    G = g.model.geometry
    G.add_box(0, 0, -0.5, 1.0, 1.0, 0.5)
    p1 = G.add_point(0, 0, 0.5)
    p2 = G.add_point(1.0, 0, 0.5)
    p3 = G.add_point(1.0, 0, 1.5)
    p4 = G.add_point(0, 0, 1.5)
    l1 = G.add_line(p1, p2)
    l2 = G.add_line(p2, p3)
    l3 = G.add_line(p3, p4)
    l4 = G.add_line(p4, p1)
    wall = G.add_plane_surface([G.add_curve_loop([l1, l2, l3, l4])])
    g.model.sync()
    g.physical.add(3, [t for _d, t in gmsh.model.getEntities(3)], name="Footing")
    g.physical.add(2, [wall], name="Wall")
    sel = g.model.select
    sel(dim=2).on_plane((0, 0, -0.5), (0, 0, 1), tol=1e-4).to_physical("FootBase")
    sel(dim=2).on_plane((0, 0, 0.0), (0, 0, 1), tol=1e-4).to_physical("FootTop")
    g.physical.add(1, [l1], name="WallBase")
    g.physical.add(1, [l3], name="WallTop")
    g.mesh.structured.set_recombine("Wall", dim=2)
    g.mesh.sizing.set_global_size(0.34)
    g.mesh.generation.generate(3)
    g.mesh.partitioning.renumber(base=1)
    return g.mesh.queries.get_fem_data()


def _shell_on_solid_run(
    g: Any, tmp_path: Path,
) -> tuple[Results, list[int], list[int], dict[int, np.ndarray]]:
    fem = _shell_on_solid_fem(g)
    solid = sorted({int(n) for n in fem.nodes.select(pg="Footing").ids})
    shell = sorted({int(n) for n in fem.nodes.select(pg="Wall").ids})
    assert solid and shell and not set(solid) & set(shell)

    ops = apeSees(fem)
    ops.model(ndm=3, ndf=6)
    ops.element.FourNodeTetrahedron(
        pg="Footing", material=ops.nDMaterial.ElasticIsotropic(E=30e9, nu=0.2),
    )
    ops.element.ShellMITC4(
        pg="Wall",
        section=ops.section.ElasticMembranePlateSection(E=30e9, nu=0.0, h=0.1),
    )
    ops.fix(pg="FootBase", dofs=(1, 1, 1))
    ops.fix(pg="WallBase", dofs=(1, 1, 1, 1, 1, 1))
    with ops.pattern.Plain(series=ops.timeSeries.Linear()) as pat:
        for n in fem.nodes.select(pg="WallTop").ids:
            pat.load(node=int(n), forces=(10.0, 50.0, -5.0, 1.0, 0.0, 0.0))
        for n in fem.nodes.select(pg="FootTop").ids:
            pat.load(node=int(n), forces=(1e5, -2e5, -3e5))
    ops.constraints.Plain()
    ops.numberer.RCM()
    ops.system.UmfPack()
    ops.test.NormDispIncr(tol=1e-8, max_iter=50)
    ops.algorithm.Newton()
    ops.integrator.LoadControl(dlam=1.0)
    ops.analysis.Static()

    emitter, _model_h5, model = _build(ops, tmp_path)
    # One record over both node families, as a user would declare it.
    spec = ResolvedRecorderSpec(fem_snapshot_id=fem.snapshot_id, records=(
        _record("nodes", "all", DISP + ROT, node_ids=np.array(solid + shell)),
    ))
    out = tmp_path / "out"
    with spec.emit_recorders(str(out), ops=emitter.ops, model=model) as rec:
        rec.begin_stage("lat", kind="static")
        assert emitter.ops.analyze(1) == 0
        rec.end_stage()
    live = {n: np.asarray(emitter.ops.nodeDisp(n), dtype=float)
            for n in solid + shell}
    # The mixed-ndf premise: 3-dof solid nodes, 6-dof shell nodes.
    assert {live[n].size for n in solid} == {3}
    assert {live[n].size for n in shell} == {6}

    r = Results.from_recorders(spec, out, fem=fem, stage_id="lat",
                               model=model, cache_root=tmp_path / "cache")
    return r, solid, shell, live


def test_shell_on_solid_nodes_read_back_from_out(g: Any, tmp_path: Path) -> None:
    r, solid, shell, live = _shell_on_solid_run(g, tmp_path)
    for k, comp in enumerate(DISP + ROT):
        owners = solid + shell if k < 3 else shell
        slab = r.nodes.get(component=comp, ids=owners)
        got = {int(i): v for i, v in zip(slab.node_ids,
                                         np.asarray(slab.values)[-1])}
        assert sorted(got) == owners, comp
        want = np.array([live[n][k] for n in owners])
        assert np.abs(want).max() > 0.0, comp
        np.testing.assert_allclose(
            [got[n] for n in owners], want, rtol=RTOL_OUT, atol=1e-15,
            err_msg=f"{comp} on .out",
        )


def test_solid_node_rotations_are_absent_from_out(
    g: Any, tmp_path: Path,
) -> None:
    """A tet node has no rotation DOF: absent from the slab, or NaN (#1971).

    One ``.out`` node record over both families pads the tet nodes'
    missing DOFs with zeros; the transcoder reads the model's per-node
    ndf and writes NaN there instead of a rotation of 0.0.
    """
    r, solid, _shell, _live = _shell_on_solid_run(g, tmp_path)
    for comp in ROT:
        slab = r.nodes.get(component=comp)
        vals = np.asarray(slab.values)[-1]
        on_solid = [v for i, v in zip(slab.node_ids, vals) if int(i) in solid]
        assert not np.isfinite(on_solid).any(), (
            f"{comp}: {int(np.isfinite(on_solid).sum())} 3-dof nodes read a "
            "finite rotation"
        )


# ---------------------------------------------------------------------------
# Mixed-ndf frame beside a quad block, 2-D: ndf=2 nodes and rotation_z
# ---------------------------------------------------------------------------


def _frame_beside_quad_fem(g: Any) -> Any:
    """A quad block (2-dof nodes) and a detached column (3-dof nodes)."""
    G = g.model.geometry
    block = G.add_rectangle(0.0, 0.0, 0.0, 1.0, 1.0)
    c0 = G.add_point(3.0, 0.0, 0.0)
    c1 = G.add_point(3.0, 2.0, 0.0)
    col = G.add_line(c0, c1)
    g.model.sync()
    g.physical.add(2, [block], name="Block")
    g.physical.add(1, [col], name="Col")
    g.physical.add(0, [c0], name="ColBase")
    g.physical.add(0, [c1], name="ColTop")
    sel = g.model.select
    sel(dim=1).on_plane((0, 0, 0), (0, 1, 0), tol=1e-4).to_physical("BlockBase")
    sel(dim=1).on_plane((0, 1.0, 0), (0, 1, 0), tol=1e-4).to_physical("BlockTop")
    g.mesh.structured.set_recombine("Block", dim=2)
    g.mesh.sizing.set_global_size(0.5)
    g.mesh.generation.generate(2)
    g.mesh.partitioning.renumber(base=1)
    return g.mesh.queries.get_fem_data()


def _frame_beside_quad_run(
    g: Any, tmp_path: Path,
) -> tuple[Results, list[int], list[int], dict[int, np.ndarray]]:
    fem = _frame_beside_quad_fem(g)
    quad = sorted({int(n) for n in fem.nodes.select(pg="Block").ids})
    frame = sorted({int(n) for n in fem.nodes.select(pg="Col").ids})
    assert quad and frame and not set(quad) & set(frame)

    ops = apeSees(fem)
    ops.model(ndm=2, ndf=3)
    ops.element.FourNodeQuad(
        pg="Block", thickness=1.0,
        material=ops.nDMaterial.ElasticIsotropic(E=1e6, nu=0.3),
    )
    ops.element.elasticBeamColumn(
        pg="Col", transf=ops.geomTransf.Linear(), A=0.01, E=200e9, Iz=1e-4,
    )
    ops.fix(pg="BlockBase", dofs=(1, 1))
    ops.fix(pg="ColBase", dofs=(1, 1, 1))
    with ops.pattern.Plain(series=ops.timeSeries.Linear()) as pat:
        for n in fem.nodes.select(pg="BlockTop").ids:
            pat.load(node=int(n), forces=(100.0, -50.0))
        for n in fem.nodes.select(pg="ColTop").ids:
            pat.load(node=int(n), forces=(1e3, 0.0, 2e2))
    ops.constraints.Plain()
    ops.numberer.RCM()
    ops.system.UmfPack()
    ops.test.NormDispIncr(tol=1e-8, max_iter=50)
    ops.algorithm.Newton()
    ops.integrator.LoadControl(dlam=1.0)
    ops.analysis.Static()

    emitter, _model_h5, model = _build(ops, tmp_path)
    spec = ResolvedRecorderSpec(fem_snapshot_id=fem.snapshot_id, records=(
        _record("nodes", "all", DISP[:2] + ("rotation_z",),
                node_ids=np.array(quad + frame)),
    ))
    out = tmp_path / "out"
    with spec.emit_recorders(str(out), ops=emitter.ops, model=model) as rec:
        rec.begin_stage("lat", kind="static")
        assert emitter.ops.analyze(1) == 0
        rec.end_stage()
    live = {n: np.asarray(emitter.ops.nodeDisp(n), dtype=float)
            for n in quad + frame}
    # The mixed-ndf premise: 2-dof quad nodes, 3-dof frame nodes.
    assert {live[n].size for n in quad} == {2}
    assert {live[n].size for n in frame} == {3}

    r = Results.from_recorders(spec, out, fem=fem, stage_id="lat",
                               model=model, cache_root=tmp_path / "cache")
    return r, quad, frame, live


def _rotation_z(r: Results) -> dict[int, float]:
    slab = r.nodes.get(component="rotation_z")
    return {int(i): float(v) for i, v in zip(slab.node_ids,
                                             np.asarray(slab.values)[-1])}


def test_ndf2_node_rotation_is_nan_from_out(g: Any, tmp_path: Path) -> None:
    """A 2-dof quad node has no rotation: NaN, never a padded 0.0 (#1971).

    No node reads a fabricated rotation either: a 3-dof frame node is NaN
    or its ``nodeDisp`` value. It is NaN today, because a 2-D
    ``rotation_z`` is recorded at DOF 6 (the next test).
    """
    r, quad, frame, live = _frame_beside_quad_run(g, tmp_path)
    got = _rotation_z(r)
    assert not np.isfinite([got[n] for n in quad]).any()
    for n in frame:
        if np.isfinite(got[n]):
            np.testing.assert_allclose(got[n], live[n][2], rtol=RTOL_OUT,
                                       atol=1e-15, err_msg=f"frame node {n}")


@pytest.mark.xfail(
    strict=True, raises=AssertionError,
    reason="2-D rotation_z is emitted as `-dof 6` (spec/_emit.py "
           "_AXIS_TO_ROT_DOF is 3-D only), so a 3-dof node's rotation is "
           "not recorded; follow-up of #1971",
)
def test_frame_rotation_z_reads_back_from_out_2d(
    g: Any, tmp_path: Path,
) -> None:
    r, _quad, frame, live = _frame_beside_quad_run(g, tmp_path)
    got = _rotation_z(r)
    want = np.array([live[n][2] for n in frame])
    assert np.abs(want).max() > 0.0
    np.testing.assert_allclose([got[n] for n in frame], want,
                               rtol=RTOL_OUT, atol=1e-15)
