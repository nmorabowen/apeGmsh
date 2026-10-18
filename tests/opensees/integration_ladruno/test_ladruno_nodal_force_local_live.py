"""Fork lane: ``.ladruno`` local-frame nodal forces equal MPCO's (#2124).

``.ladruno`` read ``nodal_resisting_*`` from ``globalForce`` (#2071), but
the local-frame names ``nodal_resisting_*_local_*`` refused, while MPCO
reads them from ``localForce`` through ``NODAL_FORCE_CATALOG``. The reader
now decodes a ``localForce`` bucket by the catalog layout of its class.

Every class the catalog lists a ``localForce`` layout for and the bridge
builds is solved here once:

* an L frame (a column up +Y, then a beam along +X, two elements each) of
  one class, fixed at the base, with a tip load on every DOF, so the
  column's local axes differ from the global ones;
* solved in two load steps through the bridge with sequential tags (FEM
  ids are not OpenSees tags), recording MPCO and ``.ladruno`` with
  ``localForce`` and ``globalForce`` in the same run.

The oracles: the ``.ladruno`` read of every local and global component
equals the MPCO read of the same run bit for bit, in the same element
order and node order; and both equal ``eleResponse(tag, "localForce" |
"globalForce")`` at ``rtol=1e-10``.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import gmsh
import h5py
import numpy as np
import pytest

from apeGmsh import apeGmsh
from apeGmsh.opensees import apeSees
from apeGmsh.opensees.emitter.live import LiveOpsEmitter
from apeGmsh.results import Results

pytestmark = pytest.mark.ladruno_fork

RTOL = 1e-10
LAMS = (0.5, 1.0)                     # LoadControl dlam = 0.5, two steps
H, L = 3.0, 4.0
#: Tip load per DOF: 2-D (Fx, Fy, Mz); 3-D (Fx, Fy, Fz, Mx, My, Mz).
TIP = {2: (1.0e3, -2.0e3, 3.0e2),
       3: (1.0e3, -2.0e3, 1.5e3, 3.0e2, -4.0e2, 5.0e2)}
E_, G_, A_, IZ, IY, J_ = 2.0e11, 8.0e10, 0.02, 3.0e-4, 2.0e-4, 1.5e-4

LOCAL = {
    2: ("nodal_resisting_force_local_x", "nodal_resisting_force_local_y",
        "nodal_resisting_moment_local_z"),
    3: ("nodal_resisting_force_local_x", "nodal_resisting_force_local_y",
        "nodal_resisting_force_local_z", "nodal_resisting_moment_local_x",
        "nodal_resisting_moment_local_y", "nodal_resisting_moment_local_z"),
}
GLOBAL = {
    2: ("nodal_resisting_force_x", "nodal_resisting_force_y",
        "nodal_resisting_moment_z"),
    3: ("nodal_resisting_force_x", "nodal_resisting_force_y",
        "nodal_resisting_force_z", "nodal_resisting_moment_x",
        "nodal_resisting_moment_y", "nodal_resisting_moment_z"),
}
FRAMES = {"localForce": LOCAL, "globalForce": GLOBAL}


def _sectioned(token: str) -> Callable[[apeSees, Any, int], Any]:
    def build(ops: apeSees, tr: Any, ndm: int) -> Any:
        if ndm == 2:
            sec = ops.section.Elastic(E=E_, A=A_, Iz=IZ)
        else:
            sec = ops.section.Elastic(E=E_, A=A_, Iz=IZ, Iy=IY, G=G_, J=J_)
        integ = ops.beamIntegration.Lobatto(section=sec, n_ip=3)
        make: Callable[..., Any] = getattr(ops.element, token)
        return make(pg="Frame", transf=tr, integration=integ)
    return build


def _elastic(ops: apeSees, tr: Any, ndm: int) -> Any:
    if ndm == 2:
        return ops.element.elasticBeamColumn(pg="Frame", transf=tr, A=A_,
                                             E=E_, Iz=IZ)
    return ops.element.elasticBeamColumn(pg="Frame", transf=tr, A=A_, E=E_,
                                         Iz=IZ, Iy=IY, G=G_, J=J_)


def _timoshenko(ops: apeSees, tr: Any, ndm: int) -> Any:
    if ndm == 2:
        return ops.element.ElasticTimoshenkoBeam(
            pg="Frame", transf=tr, E=E_, G=G_, A=A_, Iz=IZ, Avy=0.75 * A_)
    return ops.element.ElasticTimoshenkoBeam(
        pg="Frame", transf=tr, E=E_, G=G_, A=A_, Iz=IZ, Iy=IY, J=J_,
        Avy=0.75 * A_, Avz=0.75 * A_)


#: OpenSees class -> (bridge builder, ndm): every NODAL_FORCE_CATALOG class
#: with a ``local_force`` layout the bridge builds.
CLASSES: dict[str, tuple[Callable[[apeSees, Any, int], Any], int]] = {
    "ElasticBeam2d": (_elastic, 2),
    "ElasticBeam3d": (_elastic, 3),
    "ElasticTimoshenkoBeam2d": (_timoshenko, 2),
    "ElasticTimoshenkoBeam3d": (_timoshenko, 3),
    "ForceBeamColumn2d": (_sectioned("forceBeamColumn"), 2),
    "ForceBeamColumn3d": (_sectioned("forceBeamColumn"), 3),
    "DispBeamColumn2d": (_sectioned("dispBeamColumn"), 2),
    "DispBeamColumn3d": (_sectioned("dispBeamColumn"), 3),
    "LadrunoDispBeamColumn2d": (_sectioned("LadrunoDispBeamColumn"), 2),
    "LadrunoDispBeamColumn3d": (_sectioned("LadrunoDispBeamColumn"), 3),
}


def _l_frame_fem(name: str) -> Any:
    with apeGmsh(model_name=name, verbose=False) as g:
        geo = g.model.geometry
        p0 = geo.add_point(0.0, 0.0, 0.0)
        p1 = geo.add_point(0.0, H, 0.0)
        p2 = geo.add_point(L, H, 0.0)
        col = geo.add_line(p0, p1)
        beam = geo.add_line(p1, p2)
        for c in (col, beam):
            gmsh.model.mesh.setTransfiniteCurve(c, 3)
        g.physical.add(1, [col, beam], name="Frame")
        g.labels.add(0, [p0], "base")
        g.labels.add(0, [p2], "tip")
        g.mesh.generation.generate(dim=1)
        return g.mesh.queries.get_fem_data(dim=1)


def _fem_to_ops(model_h5: str) -> dict[int, int]:
    """``{fem_eid: ops_tag}`` from the bridge's own ``element_meta``."""
    with h5py.File(model_h5, "r") as f:
        meta = f["/opensees/element_meta"]
        return {
            int(e): int(t)
            for tok in meta
            for t, e in zip(meta[tok]["ids"][...], meta[tok]["fem_eids"][...])
        }


def _solve(tmp: Path, cls: str) -> dict[str, Any]:
    build, ndm = CLASSES[cls]
    fem = _l_frame_fem(f"lad_local_{cls}")
    ops = apeSees(fem, default_orientation=None)       # sequential tags
    if ndm == 2:
        ops.model(ndm=2, ndf=3)
        tr = ops.geomTransf.Linear()
    else:
        ops.model(ndm=3, ndf=6)
        tr = ops.geomTransf.Linear(vecxz=(0.0, 0.0, 1.0))
    build(ops, tr, ndm)
    ops.fix(label="base", dofs=(1,) * (3 if ndm == 2 else 6))
    with ops.pattern.Plain(series=ops.timeSeries.Linear()) as p:
        p.load(node=ops.nodes.one(label="tip"), forces=TIP[ndm])
    paths = {"mpco": str(tmp / "frame.mpco"),
             "ladruno": str(tmp / "frame.ladruno")}
    # A node channel on MPCO too: an element-only MPCO file reads 0 steps
    # (#1970).
    ops.recorder.MPCO(file=paths["mpco"], nodal_responses=("displacement",),
                      elem_responses=tuple(FRAMES))
    ops.recorder.Ladruno(file=paths["ladruno"], elem_responses=tuple(FRAMES))
    ops.constraints.Plain()
    ops.numberer.RCM()
    ops.system.BandGeneral()
    ops.test.NormDispIncr(tol=1e-12, max_iter=20)
    ops.algorithm.Newton()
    ops.integrator.LoadControl(dlam=LAMS[0])
    ops.analysis.Static()
    model_h5 = str(tmp / "model.h5")
    ops.h5(model_h5)
    emitter = LiveOpsEmitter(wipe=True)
    ops.build().emit(emitter)
    o = emitter.ops
    fem_to_ops = _fem_to_ops(model_h5)
    eids = [int(e) for e in fem.elements.select(pg="Frame").ids]
    assert any(fem_to_ops[e] != e for e in eids)       # not vacuous
    assert {o.eleType(fem_to_ops[e]) for e in eids} == {cls}

    live: dict[str, dict[int, list[np.ndarray]]] = {
        kw: {e: [] for e in eids} for kw in FRAMES}
    for _ in LAMS:
        assert emitter.analyze(steps=1) == 0
        for kw in FRAMES:
            for e in eids:
                live[kw][e].append(np.asarray(
                    o.eleResponse(fem_to_ops[e], kw), dtype=float))
    o.remove("recorders")          # flush and close both files
    return {
        "cls": cls, "ndm": ndm, "eids": eids,
        "live": {kw: {e: np.stack(v) for e, v in per.items()}
                 for kw, per in live.items()},
        "results": {
            "mpco": Results.from_mpco(paths["mpco"], fem=fem,
                                      model_h5=model_h5),
            "ladruno": Results.from_ladruno(paths["ladruno"], fem=fem,
                                            model_h5=model_h5),
        },
    }


@pytest.fixture(scope="module", params=sorted(CLASSES))
def frame(request: pytest.FixtureRequest,
          tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    return _solve(tmp_path_factory.mktemp(f"ll_{request.param}"),
                  request.param)


def _slab(r: Results, comp: str) -> tuple[np.ndarray, np.ndarray]:
    slab = r.elements.get(component=comp)
    return np.asarray(slab.element_ids), np.asarray(slab.values)


@pytest.mark.parametrize("kw", list(FRAMES))
def test_ladruno_equals_mpco_bit_for_bit(frame: dict[str, Any],
                                         kw: str) -> None:
    """The ``.ladruno`` read of each component equals MPCO's read of the
    same run: the same element ids in the same order, the same values in
    the same node order, exactly."""
    ndm, eids = frame["ndm"], frame["eids"]
    res = frame["results"]
    for route in ("mpco", "ladruno"):
        assert set(FRAMES[kw][ndm]) <= set(
            res[route].elements.available_components()), (route, kw)
    for comp in FRAMES[kw][ndm]:
        m_ids, m_vals = _slab(res["mpco"], comp)
        l_ids, l_vals = _slab(res["ladruno"], comp)
        assert sorted(m_ids.tolist()) == sorted(eids), comp
        assert l_ids.tolist() == m_ids.tolist(), comp
        assert m_vals.shape == (len(LAMS), len(eids), 2), comp
        np.testing.assert_array_equal(
            l_vals, m_vals, err_msg=f"{frame['cls']} {comp}")


@pytest.mark.parametrize("kw", list(FRAMES))
def test_ladruno_equals_ele_response(frame: dict[str, Any], kw: str) -> None:
    """Every element, component, end and step read from ``.ladruno``
    equals ``eleResponse`` on the paired OpenSees tag."""
    ndm = frame["ndm"]
    comps = FRAMES[kw][ndm]
    n = len(comps)
    for k, comp in enumerate(comps):
        ids, vals = _slab(frame["results"]["ladruno"], comp)
        for i, e in enumerate(ids.tolist()):
            want = frame["live"][kw][e][:, [k, k + n]]
            assert np.abs(frame["live"][kw][e]).max() > 0.0
            np.testing.assert_allclose(
                vals[:, i, :], want, rtol=RTOL,
                atol=RTOL * float(np.abs(frame["live"][kw][e]).max()),
                err_msg=f"{frame['cls']} {comp} of element {e}")


def test_local_frame_is_not_the_global_frame(frame: dict[str, Any]) -> None:
    """The column's local x is global +Y, so the local read is not the
    global one under another name: the parity above is not vacuous."""
    ndm = frame["ndm"]
    _, local_x = _slab(frame["results"]["ladruno"], LOCAL[ndm][0])
    _, global_x = _slab(frame["results"]["ladruno"], GLOBAL[ndm][0])
    assert not np.allclose(local_x, global_x)
