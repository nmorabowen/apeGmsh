"""The instance frame rule (ADR 0117, #1593): points move by ``R x + t``,
directions by ``R v`` only.

An instance is authored in its source frame. The merge moves the node table
by ``R x + t``; every point a constraint record caches must move the same
way, and every direction it caches (a normal, an offset, a bar axis, an
angular velocity) must turn by ``R`` and never translate. Before #1593 the
merge copied ``plane_normal``, ``offsets``, ``offset``, ``omega``,
``direction`` and ``projected_point`` from the source, so a rotated
instance's rigid diaphragm kept its source plane with no error.

Oracles, each independent of the code under test:

* closed form: ``R`` is 90 degrees about +x, ``(x, y, z) -> (x, -z, y)``,
  and ``t = T``. One unit test per field family, on records built by hand.
* lock: every :class:`ConstraintRecord` kind has a row in the frame table,
  and every array or tuple field on it is either placed or listed here as
  frame-free with the reason.
* stream routing: a real module composed with ``rotate=`` carries a rotated
  diaphragm normal, tie projection point and bar axis.
* deck: a v2 instance rotated so its floor stands in the plane ``y = 0``
  writes ``rigidDiaphragm 2`` (it wrote ``3``).
* live: that instance, driven through its diaphragm master, moves rigidly
  in its own plane (the closed form of ``test_live_couplings.py``).
"""
from __future__ import annotations

import dataclasses
import json
import math
import os
import subprocess
import sys
from pathlib import Path

import gmsh
import numpy as np
import pytest

from apeGmsh import apeGmsh
from apeGmsh._kernel.records import _constraints as C
from apeGmsh.mesh.FEMData import FEMData
from apeGmsh.mesh._compose import (
    _place_record_geometry,
    _record_geometry_fields,
)

T = (100.0, -50.0, 7.0)
ROT_X_90 = (1.0, 0.0, 0.0, math.pi / 2.0)
SIDE = 10.0
CM_SRC = (5.0, 5.0, 0.0)
#: Reference-node motion of the live rig: (dx, dz, theta_y).
MOTION = (0.01, -0.02, 0.001)


def _r(v) -> np.ndarray:
    """The closed-form ``R v``: 90 degrees about +x."""
    v = np.asarray(v, dtype=np.float64).reshape(-1, 3)
    return np.column_stack([v[:, 0], -v[:, 2], v[:, 1]])


def _place(rec):
    return _place_record_geometry(rec, translate=T, rotate=ROT_X_90)


# ---------------------------------------------------------------------------
# One unit test per field family (closed form)
# ---------------------------------------------------------------------------

def test_node_pair_offset_rotates_and_never_translates():
    rec = C.NodePairRecord(kind="rigid_beam", master_node=1, slave_node=2,
                           dofs=[1, 2, 3], offset=np.array([1.0, 2.0, 3.0]))
    np.testing.assert_allclose(_place(rec).offset, _r(rec.offset)[0], atol=1e-12)


def test_node_group_plane_normal_offsets_and_omega_rotate():
    offsets = np.array([[1.0, 2.0, 0.0], [-3.0, 0.5, 0.0]])
    rec = C.NodeGroupRecord(
        kind="rigid_diaphragm", master_node=1, slave_nodes=[2, 3],
        dofs=[1, 2, 6], offsets=offsets, plane_normal=np.array([0.0, 0.0, 1.0]),
        omega=(0.0, 0.3, 0.4))
    got = _place(rec)
    np.testing.assert_allclose(got.plane_normal, [0.0, -1.0, 0.0], atol=1e-12)
    np.testing.assert_allclose(got.offsets, _r(offsets), atol=1e-12)
    assert got.offsets.shape == offsets.shape
    np.testing.assert_allclose(got.omega, _r(rec.omega)[0], atol=1e-12)
    assert isinstance(got.omega, tuple)


def test_interpolation_projected_point_is_a_point():
    pp = np.array([1.0, 2.0, 3.0])
    rec = C.InterpolationRecord(kind="tie", slave_node=9, master_nodes=[1, 2, 3],
                                weights=np.array([0.2, 0.3, 0.5]), dofs=[1, 2, 3],
                                projected_point=pp,
                                parametric_coords=np.array([0.1, 0.2]))
    got = _place(rec)
    np.testing.assert_allclose(got.projected_point, _r(pp)[0] + T, atol=1e-12)
    # Natural coordinates and weights are frame-free.
    np.testing.assert_array_equal(got.parametric_coords, rec.parametric_coords)
    np.testing.assert_array_equal(got.weights, rec.weights)
    # A coupling's slave rows are walked too.
    cpl = C.SurfaceCouplingRecord(kind="tied_contact", slave_records=[rec],
                                  master_nodes=[1, 2, 3], slave_nodes=[9])
    np.testing.assert_allclose(_place(cpl).slave_records[0].projected_point,
                               _r(pp)[0] + T, atol=1e-12)


def test_node_to_surface_rigid_link_offsets_rotate():
    link = C.NodePairRecord(kind="rigid_beam", master_node=1, slave_node=50,
                            dofs=[1, 2, 3], offset=np.array([0.0, 0.0, 2.0]))
    rec = C.NodeToSurfaceRecord(
        kind="node_to_surface", master_node=1, slave_nodes=[5],
        phantom_nodes=[50], phantom_coords=np.array([[1.0, 1.0, 2.0]]),
        rigid_link_records=[link], dofs=[1, 2, 3])
    got = _place(rec)
    np.testing.assert_allclose(got.rigid_link_records[0].offset,
                               [0.0, -2.0, 0.0], atol=1e-12)
    np.testing.assert_allclose(got.phantom_coords, _r([1.0, 1.0, 2.0]) + T,
                               atol=1e-12)


def test_reinforce_tie_direction_rotates():
    d = np.array([0.0, 0.6, 0.8])
    rec = C.ReinforceTieRecord(kind="reinforce", rebar_node=1, host_nodes=[2, 3],
                               weights=np.array([0.5, 0.5]), direction=d,
                               shape_b=np.array([0.4, 0.6]))
    got = _place(rec)
    np.testing.assert_allclose(got.direction, _r(d)[0], atol=1e-12)
    np.testing.assert_array_equal(got.shape_b, rec.shape_b)


def test_translate_only_leaves_directions_alone():
    rec = C.NodeGroupRecord(kind="rigid_diaphragm", master_node=1,
                            slave_nodes=[2], plane_normal=np.array([0.0, 0.0, 1.0]),
                            offsets=np.array([[1.0, 0.0, 0.0]]))
    got = _place_record_geometry(rec, translate=T, rotate=None)
    np.testing.assert_array_equal(got.plane_normal, rec.plane_normal)
    np.testing.assert_array_equal(got.offsets, rec.offsets)


def test_unknown_record_kind_raises():
    @dataclasses.dataclass
    class Mystery(C.ConstraintRecord):
        anchor: tuple = (0.0, 0.0, 0.0)

    with pytest.raises(TypeError, match="instance frame table"):
        _place(Mystery(kind="mystery"))


# ---------------------------------------------------------------------------
# Lock: every kind and every vector-shaped field is classified
# ---------------------------------------------------------------------------

#: Array/tuple fields no rigid placement changes, with the reason.
FRAME_FREE = {
    "weights": "shape-function weights, a partition of unity",
    "parametric_coords": "natural coordinates on the master face",
    "mortar_operator": "a dimensionless coupling matrix",
    "shape_b": "shape-function weights at the bar's second point",
    "master_faces": "node tags",
    "slave_faces": "node tags",
}


def _vector_fields(cls) -> set[str]:
    out = set()
    for f in dataclasses.fields(cls):
        ann = str(f.type)
        if ("ndarray" in ann or "tuple" in ann) and "ClassVar" not in ann:
            out.add(f.name)
    return out


def test_every_constraint_kind_and_vector_field_is_classified():
    table = _record_geometry_fields()
    kinds = {cls for cls in vars(C).values()
             if isinstance(cls, type) and issubclass(cls, C.ConstraintRecord)
             and cls is not C.ConstraintRecord}
    assert kinds == set(table), "a constraint kind is missing from the frame table"
    for cls, (points, directions, _nested) in table.items():
        placed = set(points) | set(directions)
        unclassified = _vector_fields(cls) - placed - set(FRAME_FREE)
        assert not unclassified, (cls.__name__, unclassified)
        assert placed <= _vector_fields(cls), (cls.__name__, placed)


# ---------------------------------------------------------------------------
# Stream routing: a real rotated compose
# ---------------------------------------------------------------------------

def _slab_module(path: Path) -> Path:
    """A 2x2 quad plate in z = 0 with a rigid diaphragm on its centre point."""
    with apeGmsh(model_name="slab", verbose=False) as g:
        g.model.geometry.add_rectangle(0.0, 0.0, 0.0, SIDE, SIDE, label="p")
        g.model.geometry.add_point(*CM_SRC, label="cm")
        g.model.sync()
        g.physical.add_surface("p", name="Slab")
        g.physical.add_point("cm", name="CM")
        g.constraints.rigid_diaphragm(
            "cm", "Slab", master_point=CM_SRC, plane_normal=(0, 0, 1),
            plane_tolerance=0.01)
        g.mesh.recipe.structured(size=5.0, fallback="strict")
        g.mesh.queries.get_fem_data(dim=None).to_h5(str(path))
    return path


def _tied_module(path: Path) -> Path:
    """Two stacked unit cubes tied across z = 1 (InterpolationRecord rows)."""
    with apeGmsh(model_name="tied", verbose=False) as g:
        g.model.geometry.add_box(0, 0, 0, 1, 1, 1, label="a")
        g.model.geometry.add_box(0, 0, 1, 1, 1, 1, label="b")
        g.model.sync()
        faces = g.model.select(None, dim=2).in_box(
            (-0.1, -0.1, 0.99), (1.1, 1.1, 1.01)).result().tags()
        g.physical.add_surface([faces[0]], name="ta")
        g.physical.add_surface([faces[-1]], name="bb")
        g.constraints.tie("ta", "bb")
        g.mesh.sizing.set_global_size(0.5)
        g.mesh.generation.generate(3)
        g.mesh.queries.get_fem_data(dim=3).to_h5(str(path))
    return path


def _reinforced_module(path: Path) -> Path:
    with apeGmsh(model_name="rc", verbose=False) as g:
        box = g.model.geometry.add_box(0, 0, 0, 1, 1, 1)
        p0 = gmsh.model.occ.addPoint(0.5, 0.2, 0.2)
        p1 = gmsh.model.occ.addPoint(0.5, 0.8, 0.8)
        ln = gmsh.model.occ.addLine(p0, p1)
        g.model.sync()
        g.mesh.sizing.set_global_size(0.4)
        g.mesh.generation.generate(3)
        g.physical.add(3, [box], name="concrete")
        g.physical.add(1, [ln], name="rebar")
        g.reinforce(host="concrete", bars="rebar", perfect=1.0e12,
                    bar_diameter=0.025)
        g.mesh.queries.get_fem_data(dim=3).to_h5(str(path))
    return path


def _composed(path: Path) -> tuple[FEMData, FEMData]:
    src = FEMData.from_h5(str(path))
    out = FEMData.from_h5(str(path)).compose(
        str(path), label="m", translate=T, rotate=ROT_X_90)
    return src, out


def _module_rows(records, src_records) -> list:
    """The composed module's rows, which the merge appends after the host's.

    Host and module are the same file, so the host holds the first half.
    """
    records = list(records)
    n = len(src_records)
    assert len(records) == 2 * n, (len(records), n)
    return records[n:]


def test_rotated_compose_turns_the_diaphragm_normal_and_offsets(tmp_path):
    src, fem = _composed(_slab_module(tmp_path / "slab.h5"))
    (s,) = src.nodes.constraints
    rows = _module_rows(fem.nodes.constraints, [s])
    assert isinstance(rows[0], C.NodeGroupRecord)
    np.testing.assert_allclose(rows[0].plane_normal, [0.0, -1.0, 0.0], atol=1e-12)
    np.testing.assert_allclose(rows[0].offsets, _r(s.offsets), atol=1e-12)
    # The offsets agree with the placed node table.
    row = {int(n): i for i, n in enumerate(fem.nodes.ids)}
    xyz = np.asarray(fem.nodes.coords)
    m = xyz[row[int(rows[0].master_node)]]
    want = np.array([xyz[row[int(n)]] - m for n in rows[0].slave_nodes])
    np.testing.assert_allclose(rows[0].offsets, want, atol=1e-9)


def test_rotated_compose_places_tie_projection_points(tmp_path):
    src, fem = _composed(_tied_module(tmp_path / "tied.h5"))
    src_rows = list(src.elements.constraints)
    assert src_rows and all(isinstance(r, C.InterpolationRecord) for r in src_rows)
    row = {int(n): i for i, n in enumerate(fem.nodes.ids)}
    xyz = np.asarray(fem.nodes.coords)
    for s, got in zip(src_rows, _module_rows(fem.elements.constraints, src_rows)):
        np.testing.assert_allclose(got.projected_point,
                                   _r(s.projected_point)[0] + T, atol=1e-9)
        # A tie's slave sits on the master face: its placed node agrees.
        np.testing.assert_allclose(got.projected_point,
                                   xyz[row[int(got.slave_node)]], atol=1e-6)


def test_rotated_compose_turns_the_bar_axis(tmp_path):
    src, fem = _composed(_reinforced_module(tmp_path / "rc.h5"))
    src_ties = src.elements.reinforce_ties
    assert len(src_ties) >= 2
    ties = _module_rows(fem.elements.reinforce_ties, src_ties)
    # The source bar runs along (0, 1, 1); placed, along R (0, 1, 1).
    want = _r(np.array([0.0, 1.0, 1.0]) / math.sqrt(2.0))[0]
    for s, t in zip(src_ties, ties):
        np.testing.assert_allclose(t.direction, _r(s.direction)[0], atol=1e-12)
        np.testing.assert_allclose(np.abs(t.direction), np.abs(want), atol=1e-9)


# ---------------------------------------------------------------------------
# Deck and live: a v2 instance whose floor stands up
# ---------------------------------------------------------------------------

def _declare_slab(ops) -> None:
    ops.model(ndm=3, ndf=6)
    sec = ops.section.ElasticMembranePlateSection(
        E=200_000.0, nu=0.2, h=0.5, name="slab")
    ops.element.ShellMITC4(pg="Slab", section=sec)


def _standing_slab(workdir: Path):
    from apeGmsh.assembly import Assembly
    from apeGmsh.opensees import apeSees

    fem = FEMData.from_h5(str(_slab_module(workdir / "slab_mesh.h5")))
    ops = apeSees(fem)
    _declare_slab(ops)
    archive = workdir / "slab.h5"
    ops.h5(str(archive))
    return (Assembly("stand")
            .instance("w", archive, rotate=((1.0, 0.0, 0.0), math.pi / 2),
                      translate=T)
            .bridge(ndm=3, ndf=6))


def test_rotated_instance_writes_the_turned_diaphragm(tmp_path):
    ops = _standing_slab(tmp_path)
    deck = tmp_path / "stand.tcl"
    ops.tcl(str(deck), flat=True)
    lines = [ln.split() for ln in deck.read_text(encoding="utf-8").splitlines()
             if ln.startswith("rigidDiaphragm")]
    assert len(lines) == 1, lines
    assert lines[0][1] == "2", "the standing floor's plane is y = const"
    assert len(lines[0]) == 3 + 9, "the master plus the slab's nine nodes"


def _ids(fem, **sel) -> list[int]:
    return sorted(int(i) for i in fem.nodes.select(**sel).ids)


def _solve_standing_slab(workdir: Path) -> dict:
    from apeGmsh.opensees.emitter.live import LiveOpsEmitter

    ops = _standing_slab(workdir)
    fem = ops.fem
    (cm,) = _ids(fem, pg="w.CM")
    coord = dict(zip((int(i) for i in fem.nodes.ids),
                     np.asarray(fem.nodes.coords, dtype=float)))
    slab = [t for t in _ids(fem, pg="w.Slab") if t != cm]
    # Out of plane the slab and the master are held; in plane the
    # diaphragm alone carries the slab.
    ops.fix(nodes=slab + [cm], dofs=(0, 1, 0, 1, 0, 1))
    dx, dz, theta = MOTION
    ts = ops.timeSeries.Linear()
    with ops.pattern.Plain(series=ts) as pat:
        pat.sp(node=cm, dof=1, value=dx)
        pat.sp(node=cm, dof=3, value=dz)
        pat.sp(node=cm, dof=5, value=theta)
    ops.constraints.Transformation()
    ops.numberer.Plain()
    ops.system.FullGeneral()
    ops.test.NormDispIncr(tol=1e-10, max_iter=10)
    ops.algorithm.Linear()
    ops.integrator.LoadControl(dlam=1.0)
    ops.analysis.Static()
    emitter = LiveOpsEmitter(wipe=True)
    ops.build().emit(emitter)
    assert emitter.analyze(steps=1) == 0
    live = emitter.ops
    xc, _, zc = coord[cm]
    err = 0.0
    for t in slab:
        x, _, z = coord[t]
        want = {1: dx + theta * (z - zc), 3: dz - theta * (x - xc), 5: theta}
        for dof, value in want.items():
            err = max(err, abs(live.nodeDisp(t, dof) - value))
    return {"err": err, "n": len(slab)}


def _main() -> None:
    """``python -c`` entry: argv[1] is the work directory."""
    print("RESULT " + json.dumps(_solve_standing_slab(Path(sys.argv[1]))))


@pytest.mark.live
def test_rotated_instance_moves_rigidly_in_its_own_plane(tmp_path):
    from apeGmsh.opensees.emitter.live import _get_ops

    try:
        _get_ops()
    except ImportError as e:
        pytest.skip(f"no OpenSees backend: {e}")
    root = Path(__file__).resolve().parents[2]
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        [str(root), str(root / "src"), env.get("PYTHONPATH", "")])
    env.setdefault("LADRUNO_OPENSEES_QUIET", "1")
    proc = subprocess.run(
        [sys.executable, "-W", "ignore::UserWarning", "-c",
         "from tests.assembly.test_instance_frame_rule import _main; _main()",
         str(tmp_path)],
        env=env, capture_output=True, text=True, encoding="utf-8",
        errors="replace", timeout=600,
    )
    lines = [ln for ln in proc.stdout.splitlines() if ln.startswith("RESULT ")]
    assert proc.returncode == 0 and lines, (
        f"subprocess failed (rc={proc.returncode}):\n"
        f"{proc.stdout[-2000:]}\n{proc.stderr[-2000:]}")
    res = json.loads(lines[-1][len("RESULT "):])
    assert res["n"] == 9, res
    assert res["err"] < 1e-9, res
