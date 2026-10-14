"""#1711 parity: recorder-emitted rows vs the live domain, sequential tags.

A :class:`ResolvedRecorderSpec` names elements by FEM id (the contract
``DomainCapture`` honours too). Under ``element_tags="sequential"`` (the
bridge default) an element's OpenSees tag is not its FEM id, so the
``-ele`` list must be translated at emit. Each test below declares the
element families out of mesh order, so the case is never vacuous, emits
the spec live through ``spec.emit_recorders(..., model=)``, and checks
every recorded row against the live domain itself: ``eleResponse`` on the
OpenSees tag of the FEM id the row is labelled with (``nodeDisp`` for
nodes). Before the fix (origin/main) the ``-ele`` list carried the FEM
ids verbatim, so OpenSees recorded other elements: the portal beam's rows
held column forces (Fy 57034 / 102966 against -2965.7), with no warning.

Covered emit sites, all through ``Results.from_recorders``: ``elements``
(globalForce), ``gauss`` (stresses), ``nodes``, and ``line_stations``
(section force for the force-beam columns, localForce for the elastic
girder; #1741).
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import gmsh
import numpy as np
import pytest

from apeGmsh import apeGmsh
from apeGmsh.opensees.apesees import apeSees

pytestmark = pytest.mark.live

H, L = 3.0, 5.0
#: A ``.out`` file holds 6 significant digits; a wrong element is off by
#: orders of magnitude more than that (see the module docstring).
_TOL = {"rtol": 1e-5, "atol": 1e-6}
NODAL_FORCE = ("nodal_resisting_force_x", "nodal_resisting_force_y",
               "nodal_resisting_moment_z")


@pytest.fixture(autouse=True)
def _backend() -> None:
    pytest.importorskip("openseespy.opensees")


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------


def _portal_fem(n: int = 4) -> Any:
    with apeGmsh(model_name="b5c_portal", verbose=False) as g:
        geo = g.model.geometry
        p0 = geo.add_point(0.0, 0.0, 0.0)
        p1 = geo.add_point(0.0, H, 0.0)
        p2 = geo.add_point(L, H, 0.0)
        p3 = geo.add_point(L, 0.0, 0.0)
        c1 = geo.add_line(p0, p1)
        bm = geo.add_line(p1, p2)
        c2 = geo.add_line(p3, p2)
        for c in (c1, bm, c2):
            gmsh.model.mesh.setTransfiniteCurve(c, n)
        g.physical.add(1, [c1, c2], name="Cols")
        g.physical.add(1, [bm], name="Beam")
        g.labels.add(0, [p0, p3], "base")
        g.labels.add(0, [p1], "tl")
        g.labels.add(0, [p2], "tr")
        g.mesh.generation.generate(dim=1)
        return g.mesh.queries.get_fem_data(dim=1)


def _static_chain(ops: apeSees) -> None:
    ops.constraints.Plain()
    ops.numberer.RCM()
    ops.system.BandGeneral()
    ops.test.NormDispIncr(tol=1e-10, max_iter=10)
    ops.algorithm.Linear()
    ops.integrator.LoadControl(dlam=1.0)
    ops.analysis.Static()


def _portal_bridge(fem: Any) -> apeSees:
    ops = apeSees(fem, default_orientation=None)   # sequential tags
    ops.model(ndm=2, ndf=3)
    tr = ops.geomTransf.Linear()
    # Beam first: its sequential tags are taken before the columns'.
    ops.element.elasticBeamColumn(pg="Beam", transf=tr,
                                  A=0.02, E=2.0e11, Iz=3.0e-4)
    sec = ops.section.Elastic(E=2.0e11, A=0.03, Iz=5.0e-4)
    integ = ops.beamIntegration.Lobatto(section=sec, n_ip=4)
    ops.element.forceBeamColumn(pg="Cols", transf=tr, integration=integ)
    ops.fix(label="base", dofs=(1, 1, 1))
    ts = ops.timeSeries.Linear()
    with ops.pattern.Plain(series=ts) as p:
        p.load(node=ops.nodes.one(label="tl"), forces=(10e3, -30e3, 0.0))
        p.load(node=ops.nodes.one(label="tr"), forces=(0.0, -50e3, 4e3))
    _static_chain(ops)
    return ops


def _curve_at_y(surface: int, y: float, tol: float = 1e-6) -> int:
    for _dim, tag in gmsh.model.getBoundary([(2, surface)], oriented=False):
        bb = gmsh.model.getBoundingBox(1, abs(tag))
        if abs(bb[1] - y) < tol and abs(bb[4] - y) < tol:
            return abs(tag)
    raise AssertionError(f"no boundary curve of surface {surface} at y={y}")


def _two_blocks_fem() -> Any:
    with apeGmsh(model_name="b5c_blocks", verbose=False) as g:
        left = g.model.geometry.add_rectangle(0, 0, 0, 1, 1)
        right = g.model.geometry.add_rectangle(2, 0, 0, 1, 1)
        g.model.sync()
        g.mesh.structured.set_transfinite([(2, left), (2, right)], n=3)
        g.mesh.generation.generate(2)
        g.physical.add(2, [left], name="Left")
        g.physical.add(2, [right], name="Right")
        g.labels.add(1, [_curve_at_y(left, 0.0), _curve_at_y(right, 0.0)],
                     "base")
        g.labels.add(1, [_curve_at_y(left, 1.0), _curve_at_y(right, 1.0)],
                     "top")
        return g.mesh.queries.get_fem_data(dim=2)


def _blocks_bridge(fem: Any) -> apeSees:
    ops = apeSees(fem)   # sequential tags
    ops.model(ndm=2, ndf=2)
    mat = ops.nDMaterial.ElasticIsotropic(E=30e9, nu=0.2)
    # Right first: the left block's sequential tags follow the right's.
    ops.element.FourNodeQuad(pg="Right", thickness=0.5, material=mat)
    ops.element.FourNodeQuad(pg="Left", thickness=0.5, material=mat)
    ops.fix(label="base", dofs=(1, 1))
    ts = ops.timeSeries.Linear()
    top = fem.nodes.select(label="top").ids
    with ops.pattern.Plain(series=ts) as p:
        for k, n in enumerate(top):
            p.load(node=int(n), forces=(1e5 * (k + 1), -2e5 * (k % 3 + 1)))
    _static_chain(ops)
    return ops


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _solve_and_map(ops: apeSees) -> dict[int, int]:
    """Run step 1 through the bridge; return its ``{fem_eid: ops_tag}``."""
    assert ops.analyze(steps=1) == 0
    plan = ops._read_state.element_plan  # type: ignore[union-attr]
    return {int(e): int(t) for _s, sub in plan for e, _c, t in sub}


def _model(ops: apeSees, tmp_path: Path) -> Any:
    from apeGmsh.opensees.opensees_model import OpenSeesModel

    path = tmp_path / "model.h5"
    ops.h5(str(path))
    return OpenSeesModel.from_h5(str(path))


def _record_step(spec: Any, out: Path, model: Any) -> Any:
    """Emit ``spec`` live for one more solve step (stage ``"s"``)."""
    from apeGmsh.opensees.emitter.live import get_ops

    osp = get_ops()
    with spec.emit_recorders(str(out), ops=osp, model=model) as live:
        live.begin_stage("s", kind="static")
        assert osp.analyze(1) == 0
        live.end_stage()
    return osp


def _spec(fem: Any, *records: Any) -> Any:
    from apeGmsh.results.spec._resolved import ResolvedRecorderSpec

    return ResolvedRecorderSpec(fem_snapshot_id=fem.snapshot_id,
                                records=tuple(records))


def _record(category: str, name: str, components: tuple[str, ...],
            **ids: Any) -> Any:
    from apeGmsh.results.spec._resolved import ResolvedRecorderRecord

    return ResolvedRecorderRecord(category=category, name=name,
                                  components=components, dt=None,
                                  n_steps=None, **ids)


# ---------------------------------------------------------------------------
# elements: globalForce
# ---------------------------------------------------------------------------


def test_elements_global_force_rows_are_the_labelled_elements(
    tmp_path: Path,
) -> None:
    from apeGmsh.results import Results

    fem = _portal_fem()
    ops = _portal_bridge(fem)
    fem_to_ops = _solve_and_map(ops)
    beam = [int(e) for e in ops.elements.get(pg="Beam").ids]
    assert any(fem_to_ops[e] != e for e in beam)     # not vacuous
    model = _model(ops, tmp_path)
    spec = _spec(fem, _record(
        "elements", "beam", NODAL_FORCE, element_ids=np.asarray(beam),
        element_class_name="ElasticBeam2d",
    ))
    osp = _record_step(spec, tmp_path / "out", model)
    truth = {e: np.asarray(osp.eleResponse(fem_to_ops[e], "globalForce"))
             for e in beam}

    res = Results.from_recorders(spec, tmp_path / "out", fem=fem,
                                 stage_id="s", model=model,
                                 cache_root=tmp_path / "cache")
    for k, comp in enumerate(NODAL_FORCE):
        slab = res.elements.get(component=comp, ids=beam)
        got_ids = [int(i) for i in np.asarray(slab.element_ids)]
        assert sorted(got_ids) == sorted(beam)
        vals = np.asarray(slab.values)[-1]          # (E, n_nodes)
        for row, eid in enumerate(got_ids):
            np.testing.assert_allclose(
                vals[row], truth[eid][[k, k + 3]], **_TOL,
                err_msg=f"{comp} of element {eid} (ops tag {fem_to_ops[eid]})",
            )
    # The beam carries the shear of its end loads, not a column's.
    slab = res.elements.get(component="nodal_resisting_force_y", ids=beam)
    np.testing.assert_allclose(np.asarray(slab.values)[-1][:, 0], -2965.73,
                               atol=0.01)


# ---------------------------------------------------------------------------
# gauss: stresses
# ---------------------------------------------------------------------------


def test_gauss_stress_rows_are_the_labelled_elements(tmp_path: Path) -> None:
    from apeGmsh.results import Results

    fem = _two_blocks_fem()
    ops = _blocks_bridge(fem)
    fem_to_ops = _solve_and_map(ops)
    left = [int(e) for e in ops.elements.get(pg="Left").ids]
    assert any(fem_to_ops[e] != e for e in left)
    model = _model(ops, tmp_path)
    spec = _spec(fem, _record(
        "gauss", "left", ("stress_xx", "stress_yy", "stress_xy"),
        element_ids=np.asarray(left), element_class_name="FourNodeQuad",
    ))
    osp = _record_step(spec, tmp_path / "out", model)
    truth = {e: np.asarray(osp.eleResponse(fem_to_ops[e], "stresses"))
             for e in left}

    res = Results.from_recorders(spec, tmp_path / "out", fem=fem,
                                 stage_id="s", model=model,
                                 cache_root=tmp_path / "cache")
    for k, comp in enumerate(("stress_xx", "stress_yy", "stress_xy")):
        slab = res.elements.gauss.get(component=comp, ids=left)
        index = [int(i) for i in np.asarray(slab.element_index)]
        vals = np.asarray(slab.values)[-1]
        assert sorted(set(index)) == sorted(left)
        for eid in left:
            rows = [i for i, x in enumerate(index) if x == eid]
            np.testing.assert_allclose(
                vals[rows], truth[eid][k::3], **_TOL,
                err_msg=f"{comp} of element {eid} (ops tag {fem_to_ops[eid]})",
            )


# ---------------------------------------------------------------------------
# line_stations: section force, and the elastic-beam localForce synthesis
# ---------------------------------------------------------------------------


def test_line_station_rows_are_the_labelled_elements(tmp_path: Path) -> None:
    """#1741: ``Results.from_recorders`` on the real FEMData reads both
    line-station routes, each row the live response of its labelled element.

    The force-beam columns record ``section force`` (one row per IP). The
    elastic-beam girder answers no ``section force``, so its record emits
    ``localForce`` and the transcoder synthesises two stations: end i as
    recorded, end j negated (the internal section-force convention that
    DomainCapture and the MPCO reader use). Before the fix the transcode
    raised ``KeyError: 'No group with tag 5'``: the element-length resolver
    selected the element id as a physical-group tag.
    """
    from apeGmsh.results import Results

    fem = _portal_fem()
    ops = _portal_bridge(fem)
    fem_to_ops = _solve_and_map(ops)
    cols = [int(e) for e in ops.elements.get(pg="Cols").ids]
    beam = [int(e) for e in ops.elements.get(pg="Beam").ids]
    assert any(fem_to_ops[e] != e for e in cols)
    assert any(fem_to_ops[e] != e for e in beam)
    model = _model(ops, tmp_path)
    comps = ("axial_force", "shear_y", "bending_moment_z")
    spec = _spec(
        fem,
        _record("line_stations", "cols", comps, element_ids=np.asarray(cols),
                element_class_name="ForceBeamColumn2d"),
        _record("line_stations", "beam", comps, element_ids=np.asarray(beam),
                element_class_name="ElasticBeam2d"),
    )
    osp = _record_step(spec, tmp_path / "out", model)
    n_ip = 4
    col_truth = {      # (n_ip, 2): the elastic section's P, Mz
        e: np.stack([
            np.asarray(osp.eleResponse(fem_to_ops[e], "section", ip, "force"))
            for ip in range(1, n_ip + 1)
        ])
        for e in cols
    }
    beam_truth = {}    # (2, 3): end i as recorded, end j negated
    for e in beam:
        lf = np.asarray(osp.eleResponse(fem_to_ops[e], "localForce"))
        beam_truth[e] = np.stack([lf[0:3], -lf[3:6]])

    res = Results.from_recorders(spec, tmp_path / "out", fem=fem,
                                 stage_id="s", model=model,
                                 cache_root=tmp_path / "cache")
    ls = res.elements.line_stations
    for comp, k in (("axial_force", 0), ("bending_moment_z", 1)):
        slab = ls.get(component=comp, ids=cols)
        index = [int(i) for i in np.asarray(slab.element_index)]
        assert sorted(set(index)) == sorted(cols)
        vals = np.asarray(slab.values)[-1]
        for eid in cols:
            rows = [i for i, x in enumerate(index) if x == eid]
            assert len(rows) == n_ip
            np.testing.assert_allclose(
                vals[rows], col_truth[eid][:, k], **_TOL,
                err_msg=f"{comp} of column {eid} (ops tag {fem_to_ops[eid]})",
            )
    for k, comp in enumerate(comps):
        slab = ls.get(component=comp, ids=beam)
        index = [int(i) for i in np.asarray(slab.element_index)]
        xi = np.asarray(slab.station_natural_coord)
        assert sorted(set(index)) == sorted(beam)
        vals = np.asarray(slab.values)[-1]
        for eid in beam:
            rows = [i for i, x in enumerate(index) if x == eid]
            np.testing.assert_array_equal(xi[rows], [-1.0, 1.0])
            np.testing.assert_allclose(
                vals[rows], beam_truth[eid][:, k], **_TOL,
                err_msg=f"{comp} of girder {eid} (ops tag {fem_to_ops[eid]})",
            )


# ---------------------------------------------------------------------------
# nodes: identity by construction, guarded anyway
# ---------------------------------------------------------------------------


def test_node_displacement_rows_are_the_labelled_nodes(tmp_path: Path) -> None:
    """The bridge emits every node under its FEM id, so node records need
    no map; this pins that, should node tags ever be renumbered."""
    from apeGmsh.results import Results

    fem = _portal_fem()
    ops = _portal_bridge(fem)
    _solve_and_map(ops)
    model = _model(ops, tmp_path)
    nodes = [int(n) for n in fem.nodes.ids]
    spec = _spec(fem, _record(
        "nodes", "all", ("displacement_x", "displacement_y"),
        node_ids=np.asarray(nodes),
    ))
    osp = _record_step(spec, tmp_path / "out", model)
    res = Results.from_recorders(spec, tmp_path / "out", fem=fem,
                                 stage_id="s", model=model,
                                 cache_root=tmp_path / "cache")
    for k, comp in enumerate(("displacement_x", "displacement_y")):
        slab = res.nodes.get(component=comp, ids=nodes)
        got = [int(i) for i in np.asarray(slab.node_ids)]
        vals = np.asarray(slab.values)[-1]
        for col, nid in enumerate(got):
            assert vals[col] == pytest.approx(osp.nodeDisp(nid, k + 1),
                                              rel=1e-5, abs=1e-12), nid
