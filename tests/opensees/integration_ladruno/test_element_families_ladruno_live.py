"""Fork lane: four element families read back from ``.ladruno`` and MPCO.

Gap G7 of the DomainCapture parity table (#1767, #1945). Four families were
proven only through ``DomainCapture``; each is solved here through the bridge
(``apeSees(fem)``, so element ids go through the fem -> ops translation),
recorded by the fork's own recorder and read back by the public factory, and
every value is checked against the live domain at ``rtol=1e-10``:

======================== ============================== =====================
Family                   Oracle                         Read back as
======================== ============================== =====================
Truss axial force        ``eleResponse(t, "axialForce")``  ``.ladruno``: line
                         and ``P`` (determinate bar)    station (see below)
LadrunoBrick20 stress    ``eleResponse(t, "stresses")``  Gauss, 27 GP / elem
                         per GP, and its GP coordinates
LadrunoLST stress        ``eleResponse(t, "stresses")``  Gauss, 3 GP / elem
Shell-on-solid nodes     ``nodeDisp`` (3-dof solid,     nodes
                         6-dof shell)
======================== ============================== =====================

What a route cannot carry is not tested here; it is named in the PR and on
#1767 (build 79e0623670, 2026-10-14):

* the ``.ladruno`` reader files the truss ``axialForce`` record as a *line
  station*, never as a Gauss record (``elements.gauss`` comes back empty);
* MPCO: the truss bucket raises (``META GAUSS_IDS=[-1] is not sequential``),
  and LadrunoBrick20 / LadrunoLST stress buckets come back empty with no
  warning (the fork writes integration rule 0 in the bucket key, so the
  catalog lookup misses);
* ``.ladruno`` LadrunoLST under the ``stresses`` token comes back empty; the
  ``stress`` token (used below) reads.

Gated by the ``ladruno_fork`` marker, so the nightly ``live-fork`` lane runs it
against the pinned build. Fork-only element classes cannot run on stock, and
the stock fallback of the truss and shell-on-solid cases is
``tests/opensees/integration/test_element_families_fallback_live.py``.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import h5py
import numpy as np
import pytest

from apeGmsh import apeGmsh
from apeGmsh.opensees import apeSees
from apeGmsh.opensees.emitter.live import LiveOpsEmitter
from apeGmsh.results import Results

pytestmark = pytest.mark.ladruno_fork

RTOL = 1e-10
P = 1000.0

#: Spelled out, not read from RESPONSE_CATALOG: the reader under test
#: takes its layout from the catalog, so deriving them would assert nothing.
STRESS_3D = ("stress_xx", "stress_yy", "stress_zz",
             "stress_xy", "stress_yz", "stress_xz")
STRESS_2D = ("stress_xx", "stress_yy", "stress_xy")
DISP = ("displacement_x", "displacement_y", "displacement_z")
ROT = ("rotation_x", "rotation_y", "rotation_z")


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


def _nodes_on_plane(fem: Any, axis: int, value: float) -> list[int]:
    ids = np.asarray(fem.nodes.ids)
    xyz = np.asarray(fem.nodes.coords)
    return [int(n) for n, p in zip(ids, xyz) if abs(float(p[axis]) - value) < 1e-9]


def _record(ops: apeSees, kind: str, path: str, **channels: Any) -> None:
    """Declare the fork's ``ladruno`` or the ``mpco`` recorder.

    The nodal channel is always on: the MPCO reader takes a stage's time
    vector from the node results, so an element-only MPCO file reads with
    zero steps.
    """
    channels.setdefault("nodal_responses", ("displacement",))
    getattr(ops.recorder, kind)(file=path, **channels)


def _open(kind: str, path: str, fem: Any, model_h5: str) -> Results:
    factory: Callable[..., Results] = (
        Results.from_ladruno if kind == "Ladruno" else Results.from_mpco
    )
    return factory(path, fem=fem, model_h5=model_h5)


def _run(ops: apeSees, tmp_path: Path, steps: int = 1) -> tuple[Any, str]:
    """Persist the model, build it into the live fork domain, solve."""
    model_h5 = str(tmp_path / "model.h5")
    ops.h5(model_h5)
    emitter = LiveOpsEmitter(wipe=True)
    ops.build().emit(emitter)
    return emitter, model_h5


def _gauss_per_element(
    slab: Any, eid: int,
) -> tuple[np.ndarray, np.ndarray]:
    """``(values at the last step, natural coords)`` of one element's GPs."""
    cols = np.flatnonzero(np.asarray(slab.element_index) == eid)
    return (np.asarray(slab.values)[-1, cols],
            np.asarray(slab.natural_coords)[cols])


# ---------------------------------------------------------------------------
# Truss: axial force (.ladruno only; MPCO is a gap, see the module docstring)
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
def test_truss_axial_force_reads_back_from_ladruno(
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
    path = str(tmp_path / "truss.ladruno")
    _record(ops, "Ladruno", path, elem_responses=("axialForce",))
    ops.constraints.Plain()
    ops.numberer.Plain()
    ops.system.BandGeneral()
    ops.test.NormDispIncr(tol=1e-10, max_iter=10)
    ops.algorithm.Newton()
    ops.integrator.LoadControl(dlam=0.5)
    ops.analysis.Static()

    emitter, model_h5 = _run(ops, tmp_path)
    tag = _fem_to_ops(model_h5)[bar_eid]
    assert tag != bar_eid          # the translation is exercised
    live = []
    for _ in range(2):
        assert emitter.analyze(steps=1) == 0
        live.append(float(emitter.ops.eleResponse(tag, "axialForce")[0]))
    emitter.ops.remove("recorders")
    np.testing.assert_allclose(live, [P / 2, P], rtol=RTOL)

    r = _open("Ladruno", path, fem, model_h5)
    slab = r.elements.line_stations.get(component="axial_force")
    assert [int(e) for e in slab.element_index] == [bar_eid]
    np.testing.assert_allclose(np.asarray(slab.values)[:, 0], live, rtol=RTOL)


# ---------------------------------------------------------------------------
# LadrunoBrick20 and LadrunoLST: Gauss stress (.ladruno; MPCO is a gap)
# ---------------------------------------------------------------------------


def test_ladruno_brick20_gauss_stress_reads_back_from_ladruno(
    tmp_path: Path,
) -> None:
    with apeGmsh(model_name="g7_h20", verbose=False) as g:
        box = g.model.geometry.add_box(0, 0, 0, 1, 1, 2)
        g.model.sync()
        g.mesh.structured.set_transfinite_box(box, n=3)
        g.mesh.generation.generate(3)
        g.mesh.generation.set_order(2, bubble=False)
        g.physical.add(3, [box], name="Body")
        fem = g.mesh.queries.get_fem_data(dim=3)
    assert list(fem.elements)[0].element_type.npe == 20

    ops = apeSees(fem)
    ops.model(ndm=3, ndf=3)
    mat = ops.nDMaterial.ElasticIsotropic(E=2.0e8, nu=0.25)
    ops.element.LadrunoBrick20(pg="Body", material=mat, formulation="std")
    ops.fix(nodes=_nodes_on_plane(fem, 2, 0.0), dofs=(1, 1, 1))
    with ops.pattern.Plain(series=ops.timeSeries.Linear()) as p:
        # A lateral share as well, so every stress component is non-trivial.
        for k, nid in enumerate(_nodes_on_plane(fem, 2, 2.0)):
            p.load(node=nid, forces=(1.0e2 * (k % 3), 50.0, -1.0e3))
    path = str(tmp_path / "h20.ladruno")
    _record(ops, "Ladruno", path, elem_responses=("stress",))
    ops.constraints.Plain()
    ops.numberer.RCM()
    ops.system.BandGeneral()
    ops.test.NormDispIncr(tol=1e-9, max_iter=10)
    ops.algorithm.Linear()
    ops.integrator.LoadControl(dlam=1.0)
    ops.analysis.Static()

    emitter, model_h5 = _run(ops, tmp_path)
    assert emitter.analyze(steps=1) == 0
    fem_to_ops = _fem_to_ops(model_h5)
    eids = [int(e) for grp in fem.elements for e in grp.ids]
    assert any(fem_to_ops[e] != e for e in eids)
    truth = {e: np.asarray(emitter.ops.eleResponse(fem_to_ops[e], "stresses"),
                           dtype=float).reshape(27, 6) for e in eids}
    gp_xi = {e: np.asarray(emitter.ops.eleResponse(fem_to_ops[e],
                                                   "integrationPoints"),
                           dtype=float).reshape(27, 3) for e in eids}
    emitter.ops.remove("recorders")

    r = _open("Ladruno", path, fem, model_h5)
    for k, comp in enumerate(STRESS_3D):
        slab = r.elements.gauss.get(component=comp)
        assert sorted(set(int(e) for e in slab.element_index)) == sorted(eids)
        for eid in eids:
            vals, xi = _gauss_per_element(slab, eid)
            col = truth[eid][:, k]
            np.testing.assert_allclose(
                vals, col, rtol=RTOL, atol=RTOL * np.abs(truth[eid]).max(),
                err_msg=f"{comp} of element {eid} (tag {fem_to_ops[eid]})",
            )
            # GP order, not just count: the read coordinates are the
            # element's own integration points.
            np.testing.assert_allclose(xi, gp_xi[eid], atol=1e-12)


def test_ladruno_lst_gauss_stress_reads_back_from_ladruno(tmp_path: Path) -> None:
    with apeGmsh(model_name="g7_lst", verbose=False) as g:
        rect = g.model.geometry.add_rectangle(0, 0, 0, 4, 1)
        g.model.sync()
        g.mesh.sizing.set_global_size(0.5)
        g.mesh.generation.generate(2)
        g.mesh.generation.set_order(2, bubble=False)
        g.physical.add(2, [rect], name="Plate")
        fem = g.mesh.queries.get_fem_data(dim=2)
    assert list(fem.elements)[0].element_type.npe == 6

    ops = apeSees(fem)
    ops.model(ndm=2, ndf=2)
    mat = ops.nDMaterial.ElasticIsotropic(E=2.0e8, nu=0.25)
    ops.element.LadrunoLST(pg="Plate", material=mat, thickness=0.1,
                           plane_type="PlaneStress", geom="linear")
    ops.fix(nodes=_nodes_on_plane(fem, 0, 0.0), dofs=(1, 1))
    with ops.pattern.Plain(series=ops.timeSeries.Linear()) as p:
        for nid in _nodes_on_plane(fem, 0, 4.0):
            p.load(node=nid, forces=(30.0, -1.0e2))
    path = str(tmp_path / "lst.ladruno")
    _record(ops, "Ladruno", path, elem_responses=("stress",))
    ops.constraints.Plain()
    ops.numberer.RCM()
    ops.system.BandGeneral()
    ops.test.NormDispIncr(tol=1e-9, max_iter=10)
    ops.algorithm.Linear()
    ops.integrator.LoadControl(dlam=1.0)
    ops.analysis.Static()

    emitter, model_h5 = _run(ops, tmp_path)
    assert emitter.analyze(steps=1) == 0
    fem_to_ops = _fem_to_ops(model_h5)
    eids = [int(e) for grp in fem.elements for e in grp.ids]
    assert any(fem_to_ops[e] != e for e in eids)
    truth = {e: np.asarray(emitter.ops.eleResponse(fem_to_ops[e], "stresses"),
                           dtype=float).reshape(3, 3) for e in eids}
    emitter.ops.remove("recorders")

    r = _open("Ladruno", path, fem, model_h5)
    for k, comp in enumerate(STRESS_2D):
        slab = r.elements.gauss.get(component=comp)
        assert sorted(set(int(e) for e in slab.element_index)) == sorted(eids)
        for eid in eids:
            vals, _xi = _gauss_per_element(slab, eid)
            np.testing.assert_allclose(
                vals, truth[eid][:, k],
                rtol=RTOL, atol=RTOL * np.abs(truth[eid]).max(),
                err_msg=f"{comp} of element {eid} (tag {fem_to_ops[eid]})",
            )


# ---------------------------------------------------------------------------
# Mixed-ndf shell-on-solid: nodal round-trip (.ladruno and MPCO)
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


def _shell_on_solid_bridge(fem: Any) -> apeSees:
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
    return ops


def _static_shell_chain(ops: apeSees) -> None:
    ops.constraints.Plain()
    ops.numberer.RCM()
    ops.system.UmfPack()
    ops.test.NormDispIncr(tol=1e-8, max_iter=50)
    ops.algorithm.Newton()
    ops.integrator.LoadControl(dlam=1.0)
    ops.analysis.Static()


@pytest.mark.parametrize("kind, ext", [("Ladruno", "ladruno"), ("MPCO", "mpco")])
def test_shell_on_solid_nodes_read_back(
    g: Any, tmp_path: Path, kind: str, ext: str,
) -> None:
    fem = _shell_on_solid_fem(g)
    solid = sorted({int(n) for n in fem.nodes.select(pg="Footing").ids})
    shell = sorted({int(n) for n in fem.nodes.select(pg="Wall").ids})
    assert solid and shell and not set(solid) & set(shell)

    ops = _shell_on_solid_bridge(fem)
    path = str(tmp_path / f"sos.{ext}")
    _record(ops, kind, path, nodal_responses=("displacement", "rotation"))
    _static_shell_chain(ops)

    emitter, model_h5 = _run(ops, tmp_path)
    assert emitter.analyze(steps=1) == 0
    live = {n: np.asarray(emitter.ops.nodeDisp(n), dtype=float)
            for n in solid + shell}
    # The mixed-ndf premise: 3-dof solid nodes, 6-dof shell nodes.
    assert {live[n].size for n in solid} == {3}
    assert {live[n].size for n in shell} == {6}
    emitter.ops.remove("recorders")

    r = _open(kind, path, fem, model_h5)
    for k, comp in enumerate(DISP + ROT):
        owners = solid + shell if k < 3 else shell
        slab = r.nodes.get(component=comp, ids=owners)
        got = {int(i): v for i, v in zip(slab.node_ids,
                                         np.asarray(slab.values)[-1])}
        assert sorted(got) == owners, comp
        want = np.array([live[n][k] for n in owners])
        scale = np.abs(want).max()
        assert scale > 0.0, comp
        np.testing.assert_allclose(
            [got[n] for n in owners], want, rtol=RTOL, atol=RTOL * scale,
            err_msg=f"{comp} on {kind}",
        )
