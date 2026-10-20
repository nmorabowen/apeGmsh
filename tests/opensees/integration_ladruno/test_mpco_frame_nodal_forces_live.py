"""Fork lane: frame-beam nodal forces on MPCO, DomainCapture and ``.out`` (#2072).

Before #2072 an MPCO read of ``nodal_resisting_*`` on a force-based beam
returned an empty slab with no warning. The fork's MPCO recorder writes a
frame beam's ``globalForce`` / ``localForce`` under a custom-rule bucket
key (``74-ForceBeamColumn3d[1000:1:0]``, beside ``5-ElasticBeam3d[1:0:0]``)
and the reader skipped every custom-rule bucket, then every class outside
``NODAL_FORCE_CATALOG``, which listed the elastic beams only.

``NODAL_FORCE_CATALOG`` also gates DomainCapture and the ``.out``
transcoder, so every frame-beam class it now lists is solved here once and
read on all three routes:

* an L frame (a column up +Y, then a beam along +X, two elements each) of
  one class, fixed at the base, with a tip load on every DOF;
* solved in two load steps through the bridge with sequential tags (FEM
  ids are not OpenSees tags), recording MPCO, a DomainCapture and ``.out``
  recorders in the same run.

The oracles: every row read back on every route equals
``eleResponse(tag, "globalForce" | "localForce")`` at ``rtol=1e-10``, in
both frames; and the base element's global end forces equal the closed-form
base reaction of the cantilever frame (``-P`` and ``-(M + r x P)``), which
equals ``nodeReaction`` at the base.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import gmsh
import h5py
import numpy as np
import pytest

from apeGmsh import apeGmsh
from apeGmsh.opensees import OpenSeesModel, apeSees
from apeGmsh.opensees.emitter.live import LiveOpsEmitter
from apeGmsh.results import Results
from apeGmsh.results.capture.spec import DomainCaptureSpec

pytestmark = pytest.mark.ladruno_fork

RTOL = 1e-10
LAMS = (0.5, 1.0)                     # LoadControl dlam = 0.5, two steps
H, L = 3.0, 4.0
#: Tip load per DOF: 2-D (Fx, Fy, Mz); 3-D (Fx, Fy, Fz, Mx, My, Mz).
TIP = {2: (1.0e3, -2.0e3, 3.0e2),
       3: (1.0e3, -2.0e3, 1.5e3, 3.0e2, -4.0e2, 5.0e2)}

GLOBAL = {
    2: ("nodal_resisting_force_x", "nodal_resisting_force_y",
        "nodal_resisting_moment_z"),
    3: ("nodal_resisting_force_x", "nodal_resisting_force_y",
        "nodal_resisting_force_z", "nodal_resisting_moment_x",
        "nodal_resisting_moment_y", "nodal_resisting_moment_z"),
}
LOCAL = {
    2: ("nodal_resisting_force_local_x", "nodal_resisting_force_local_y",
        "nodal_resisting_moment_local_z"),
    3: ("nodal_resisting_force_local_x", "nodal_resisting_force_local_y",
        "nodal_resisting_force_local_z", "nodal_resisting_moment_local_x",
        "nodal_resisting_moment_local_y", "nodal_resisting_moment_local_z"),
}
FRAMES = {"globalForce": GLOBAL, "localForce": LOCAL}
ROUTES = ("mpco", "capture", "out")

#: OpenSees class -> (bridge element token, ndm). One class per family the
#: fork records, in 2-D and 3-D: every class #2072 adds to the catalog.
CLASSES: dict[str, tuple[str, int]] = {
    "ForceBeamColumn2d": ("forceBeamColumn", 2),
    "ForceBeamColumn3d": ("forceBeamColumn", 3),
    "DispBeamColumn2d": ("dispBeamColumn", 2),
    "DispBeamColumn3d": ("dispBeamColumn", 3),
    "LadrunoDispBeamColumn2d": ("LadrunoDispBeamColumn", 2),
    "LadrunoDispBeamColumn3d": ("LadrunoDispBeamColumn", 3),
    "LadrunoIMKBeam2d": ("LadrunoIMKBeam", 2),
    "LadrunoIMKBeam": ("LadrunoIMKBeam", 3),
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


def _build(ops: apeSees, token: str, ndm: int) -> None:
    if ndm == 2:
        ops.model(ndm=2, ndf=3)
        tr = ops.geomTransf.Linear()
        if token == "LadrunoIMKBeam":
            ops.element.LadrunoIMKBeam(
                pg="Frame", A=0.02, E=2.0e11, Iz=3.0e-4, transf=tr
            )
        else:
            sec = ops.section.Elastic(E=2.0e11, A=0.02, Iz=3.0e-4)
            integ = ops.beamIntegration.Lobatto(section=sec, n_ip=3)
            make: Callable[..., Any] = getattr(ops.element, token)
            make(pg="Frame", transf=tr, integration=integ)
    else:
        ops.model(ndm=3, ndf=6)
        tr = ops.geomTransf.Linear(vecxz=(0.0, 0.0, 1.0))
        if token == "LadrunoIMKBeam":
            ops.element.LadrunoIMKBeam(
                pg="Frame", A=0.02, E=2.0e11, G=8.0e10, Jx=1.5e-4,
                Iy=2.0e-4, Iz=3.0e-4, transf=tr,
            )
        else:
            sec = ops.section.Elastic(E=2.0e11, A=0.02, Iz=3.0e-4, Iy=2.0e-4,
                                      G=8.0e10, J=1.5e-4)
            integ = ops.beamIntegration.Lobatto(section=sec, n_ip=3)
            make = getattr(ops.element, token)
            make(pg="Frame", transf=tr, integration=integ)
    ops.fix(label="base", dofs=(1,) * (3 if ndm == 2 else 6))
    with ops.pattern.Plain(series=ops.timeSeries.Linear()) as p:
        p.load(node=ops.nodes.one(label="tip"), forces=TIP[ndm])
    ops.constraints.Plain()
    ops.numberer.RCM()
    ops.system.BandGeneral()
    ops.test.NormDispIncr(tol=1e-12, max_iter=20)
    ops.algorithm.Newton()
    ops.integrator.LoadControl(dlam=LAMS[0])
    ops.analysis.Static()


def _fem_to_ops(model_h5: str) -> dict[int, int]:
    """``{fem_eid: ops_tag}`` from the bridge's own ``element_meta``."""
    with h5py.File(model_h5, "r") as f:
        meta = f["/opensees/element_meta"]
        return {
            int(e): int(t)
            for tok in meta
            for t, e in zip(meta[tok]["ids"][...], meta[tok]["fem_eids"][...])
        }


def _recorder_spec(fem: Any, eids: list[int], cls: str, ndm: int) -> Any:
    from apeGmsh.results.spec._resolved import (
        ResolvedRecorderRecord,
        ResolvedRecorderSpec,
    )

    return ResolvedRecorderSpec(fem_snapshot_id=fem.snapshot_id, records=tuple(
        ResolvedRecorderRecord(
            category="elements", name=f"frame_{kw}", components=comps[ndm],
            dt=None, n_steps=None, element_ids=np.asarray(eids),
            element_class_name=cls)
        for kw, comps in FRAMES.items()
    ))


def _solve(tmp: Path, cls: str) -> dict[str, Any]:
    token, ndm = CLASSES[cls]
    fem = _l_frame_fem(f"frame_nf_{cls}")
    ops = apeSees(fem, default_orientation=None)       # sequential tags
    _build(ops, token, ndm)
    mpco = str(tmp / "frame.mpco")
    # A node channel too: an element-only MPCO file reads 0 steps (#1970).
    ops.recorder.MPCO(file=mpco, nodal_responses=("displacement",),
                      elem_responses=tuple(FRAMES))
    model_h5 = str(tmp / "model.h5")
    ops.h5(model_h5)
    emitter = LiveOpsEmitter(wipe=True)
    ops.build().emit(emitter)
    o = emitter.ops
    fem_to_ops = _fem_to_ops(model_h5)
    eids = [int(e) for e in fem.elements.select(pg="Frame").ids]
    assert any(fem_to_ops[e] != e for e in eids)       # not vacuous
    assert {o.eleType(fem_to_ops[e]) for e in eids} == {cls}

    cspec = DomainCaptureSpec(opensees=ops)
    for kw, comps in FRAMES.items():
        cspec.elements(pg="Frame", components=list(comps[ndm]),
                       name=f"cap_{kw}")
    rspec = _recorder_spec(fem, eids, cls, ndm)
    model = OpenSeesModel.from_h5(model_h5)
    cap_path = str(tmp / "capture.h5")
    out_dir = tmp / "out"
    base = int(ops.nodes.one(label="base").tag)
    live: dict[str, dict[int, list[np.ndarray]]] = {
        kw: {e: [] for e in eids} for kw in FRAMES}
    reaction: list[np.ndarray] = []
    with ops.domain_capture(cspec, path=cap_path, ops=o) as cap, \
            rspec.emit_recorders(str(out_dir), ops=o, model=model) as rec:
        rec.begin_stage("s", kind="static")
        cap.begin_stage("s", kind="static")
        for _ in LAMS:
            assert emitter.analyze(steps=1) == 0
            cap.step(t=o.getTime())
            for kw in FRAMES:
                for e in eids:
                    live[kw][e].append(np.asarray(
                        o.eleResponse(fem_to_ops[e], kw), dtype=float))
            o.reactions()
            reaction.append(np.asarray(o.nodeReaction(base), dtype=float))
        cap.end_stage()
        rec.end_stage()
    o.remove("recorders")          # flush and close the MPCO file

    # The base element: the one whose connectivity holds the base node.
    root = [int(e) for grp in fem.elements
            for e, conn in zip(grp.ids, grp.connectivity) if base in conn]
    assert len(root) == 1, root
    om = OpenSeesModel.from_h5(cap_path, fem_root="/model")
    return {
        "cls": cls, "ndm": ndm, "eids": eids, "root": root[0],
        "reaction": np.stack(reaction),
        "live": {kw: {e: np.stack(v) for e, v in per.items()}
                 for kw, per in live.items()},
        "results": {
            "mpco": Results.from_mpco(mpco, fem=fem, model_h5=model_h5),
            "capture": Results.from_native(cap_path, model=om),
            "out": Results.from_recorders(rspec, out_dir, fem=fem,
                                          stage_id="s", model=model,
                                          cache_root=tmp / "cache"),
        },
        "conn": {int(e): [int(n) for n in conn] for grp in fem.elements
                 for e, conn in zip(grp.ids, grp.connectivity)},
        "base": base,
    }


@pytest.fixture(scope="module", params=sorted(CLASSES))
def frame(request: pytest.FixtureRequest,
          tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    return _solve(tmp_path_factory.mktemp(f"nf_{request.param}"),
                  request.param)


def _close(got: np.ndarray, want: np.ndarray, msg: str) -> None:
    np.testing.assert_allclose(
        got, want, rtol=RTOL, atol=RTOL * float(np.abs(want).max()),
        err_msg=msg,
    )


def _read(r: Results, comp: str, ids: list[int] | None = None,
          ) -> dict[int, np.ndarray]:
    """``{eid: (T, 2)}``: the end-i and end-j value of ``comp``."""
    slab = r.elements.get(component=comp, ids=ids)
    vals = np.asarray(slab.values)
    return {int(e): vals[:, i, :]
            for i, e in enumerate(np.asarray(slab.element_ids))}


def _closed_reaction(ndm: int) -> np.ndarray:
    """Base reaction of the cantilever L frame per unit load factor."""
    if ndm == 2:
        fx, fy, mz = TIP[2]
        return -np.array([fx, fy, mz + L * fy - H * fx])
    f = np.array(TIP[3][:3])
    m = np.array(TIP[3][3:])
    r = np.array([L, H, 0.0])
    return -np.concatenate([f, m + np.cross(r, f)])


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_reaction_is_the_closed_form(frame: dict[str, Any]) -> None:
    """The oracle: ``nodeReaction`` at the base is the statics of the frame,
    and equals the base element's global end forces at the base node."""
    ndm = frame["ndm"]
    _close(frame["reaction"], np.outer(LAMS, _closed_reaction(ndm)),
           f"{frame['cls']} base reaction vs statics")
    root = frame["root"]
    end = frame["conn"][root].index(frame["base"])
    n = len(GLOBAL[ndm])
    gf = frame["live"]["globalForce"][root][:, end * n:(end + 1) * n]
    _close(gf, frame["reaction"], f"{frame['cls']} root globalForce")


@pytest.mark.parametrize("route", ROUTES)
@pytest.mark.parametrize("kw", list(FRAMES))
def test_nodal_forces_equal_ele_response(
    frame: dict[str, Any], route: str, kw: str,
) -> None:
    """Every element, component, end and step read back on ``route`` equals
    ``eleResponse`` on the paired OpenSees tag, at ``rtol=1e-10``."""
    ndm, eids = frame["ndm"], frame["eids"]
    comps = FRAMES[kw][ndm]
    n = len(comps)
    r = frame["results"][route]
    assert set(comps) <= set(_available(r)), (route, kw)
    for k, comp in enumerate(comps):
        got = _read(r, comp)
        assert sorted(got) == sorted(eids), (route, comp, sorted(got))
        for e in eids:
            want = frame["live"][kw][e][:, [k, k + n]]
            assert np.abs(frame["live"][kw][e]).max() > 0.0
            _close(got[e], want,
                   f"{route} {frame['cls']} {comp} of element {e}")


@pytest.mark.parametrize("route", ROUTES)
def test_root_base_end_is_the_closed_form_reaction(
    frame: dict[str, Any], route: str,
) -> None:
    """The base element's global end forces at the base node, read back by
    FEM id on ``route``, are the closed-form base reaction."""
    ndm, root = frame["ndm"], frame["root"]
    end = frame["conn"][root].index(frame["base"])
    got = np.stack([_read(frame["results"][route], c, ids=[root])[root][:, end]
                    for c in GLOBAL[ndm]], axis=1)          # (T, n)
    _close(got, np.outer(LAMS, _closed_reaction(ndm)),
           f"{route} {frame['cls']} base end vs statics")


def _available(r: Results) -> list[str]:
    return list(r.elements.available_components())
