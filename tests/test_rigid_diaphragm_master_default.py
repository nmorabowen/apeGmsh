"""R3-S4 (#1786, ruling D3; closes #1778): ``rigid_diaphragm`` with
``master_point`` omitted takes the master from a one-node label.

Oracles, each naming the right answer independently of the code:

* **#1778.** On main, the default ``master_point=(0, 0, 0)`` put the
  diaphragm plane at z = 0, so a floor at z = 3.2 or 6.4 collected no
  node: the record was ``(perp=3, master=0, slaves=[])``, the deck line
  ``rigidDiaphragm 3 0``, and the live ``eigen`` raised ``OpenSeesError``.
  Now each floor's master is the one node of its ``floor_k_master`` label
  and its slaves are exactly the nine ``floor_k_joints`` nodes, by
  construction of the frame.
* **Frame oracle.** The reference frame (``model-scripts.md``,
  ``frame_modal.py``) gives the same diaphragm records, the same deck
  and the same eigen periods with the point omitted as with it given.
  The given point of floor 2 is coincident with beam-column joint
  ``J11_L2``; the one-node rule picks the labelled master, the node the
  script fixes and masses, in both cases (B5, #1804).
* **Refusals.** A master resolving to 0 or to several nodes, and a plane
  through the master that holds no slave, raise naming the count or the
  plane.
* **D4.** ``p.load(node=ops.nodes.one(label="tip"))`` puts the load on
  the labelled node's tag.
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pytest

from apeGmsh import apeGmsh
from apeGmsh._kernel.defs.constraints import RigidDiaphragmDef
from apeGmsh._kernel.resolvers._constraint_resolver import ConstraintResolver

BAY, N_BAYS, STOREY, N_STOREYS = 6.0, 2, 3.2, 2
CENTRE = N_BAYS * BAY / 2
FLOORS = range(1, N_STOREYS + 1)
PLANE_TOL = 0.05


def _frame(*, given: bool):
    """The frame_modal geometry: 3 x 3 column lines, two storeys, a
    labelled master point per floor at the plan centre."""
    with apeGmsh(model_name=f"r3s4_{'given' if given else 'omitted'}") as g:
        joints: dict[int, list[str]] = {}
        for k in range(N_STOREYS + 1):
            joints[k] = []
            for i in range(N_BAYS + 1):
                for j in range(N_BAYS + 1):
                    name = f"J{i}{j}_L{k}"
                    g.model.geometry.add_point(i * BAY, j * BAY, k * STOREY,
                                               label=name)
                    joints[k].append(name)
        for k in FLOORS:
            g.model.geometry.add_point(CENTRE, CENTRE, k * STOREY,
                                       label=f"floor_{k}_master")
        cols, beams = [], []
        for k in FLOORS:
            for i in range(N_BAYS + 1):
                for j in range(N_BAYS + 1):
                    cols.append(f"C{i}{j}_{k}")
                    g.model.geometry.add_line(f"J{i}{j}_L{k - 1}",
                                              f"J{i}{j}_L{k}", label=cols[-1])
            for i in range(N_BAYS):
                for j in range(N_BAYS + 1):
                    beams.append(f"BX{i}{j}_{k}")
                    g.model.geometry.add_line(f"J{i}{j}_L{k}",
                                              f"J{i + 1}{j}_L{k}", label=beams[-1])
            for i in range(N_BAYS + 1):
                for j in range(N_BAYS):
                    beams.append(f"BY{i}{j}_{k}")
                    g.model.geometry.add_line(f"J{i}{j}_L{k}",
                                              f"J{i}{j + 1}_L{k}", label=beams[-1])
        g.physical.add_curve(cols, name="columns")
        g.physical.add_curve(beams, name="beams")
        g.physical.add_point(joints[0], name="column_bases")
        for k in FLOORS:
            g.physical.add_point(joints[k], name=f"floor_{k}_joints")
            g.physical.add_point([f"floor_{k}_master"], name=f"floor_{k}_master")
            kw = {"master_point": (CENTRE, CENTRE, k * STOREY)} if given else {}
            g.constraints.rigid_diaphragm(
                f"floor_{k}_master", f"floor_{k}_joints",
                plane_normal=(0, 0, 1), plane_tolerance=PLANE_TOL, **kw)
        g.mesh.structured.set_transfinite_curve("columns", n_nodes=2)
        g.mesh.structured.set_transfinite_curve("beams", n_nodes=2)
        g.mesh.generation.generate(dim=1)
        return g.mesh.queries.get_fem_data(dim=1)


def _ids(fem, **sel) -> list[int]:
    return sorted(int(i) for i in fem.nodes.select(**sel).ids)


def _diaphragms(fem) -> list[tuple[int, int, list[int]]]:
    return [(int(p), int(m), sorted(int(s) for s in ss))
            for p, m, ss in fem.nodes.constraints.rigid_diaphragms()]


def _bridge(fem, *, master_fix=(0, 0, 1, 1, 1, 0)):
    from apeGmsh.opensees import apeSees

    e, nu = 25e9, 0.2
    side, bw, bd = 0.5, 0.3, 0.6
    plan = N_BAYS * BAY
    mass = 8e3 * plan**2 / 9.81
    rot = mass * 2 * plan**2 / 12
    ops = apeSees(fem)
    ops.model(ndm=3, ndf=6)
    ct = ops.geomTransf.Linear(vecxz=(1, 0, 0))
    bt = ops.geomTransf.Linear(vecxz=(0, 0, 1))
    ops.element.elasticBeamColumn(
        pg="columns", transf=ct, A=side**2, E=e, G=e / (2 * (1 + nu)),
        J=0.141 * side**4, Iy=side**4 / 12, Iz=side**4 / 12)
    ops.element.elasticBeamColumn(
        pg="beams", transf=bt, A=bw * bd, E=e, G=e / (2 * (1 + nu)),
        J=0.229 * bd * bw**3, Iy=bw * bd**3 / 12, Iz=bd * bw**3 / 12)
    ops.fix(pg="column_bases", dofs=(1, 1, 1, 1, 1, 1))
    for k in FLOORS:
        ops.fix(pg=f"floor_{k}_master", dofs=master_fix)
        ops.mass(pg=f"floor_{k}_master", values=(mass, mass, 0.0, 0.0, 0.0, rot))
    ops.constraints.Transformation()
    ops.numberer.RCM()
    ops.system.BandGeneral()
    ops.test.NormDispIncr(tol=1e-10, max_iter=1)
    ops.algorithm.Linear()
    ops.integrator.LoadControl(dlam=1.0)
    ops.analysis.Static()
    return ops


@pytest.fixture(scope="module")
def frames():
    return {"given": _frame(given=True), "omitted": _frame(given=False)}


# ---------------------------------------------------------------------------
# #1778 — an upper floor with the point omitted
# ---------------------------------------------------------------------------

def test_1778_omitted_point_resolves_each_floor_on_its_labelled_master(frames):
    fem = frames["omitted"]
    got = _diaphragms(fem)
    want = []
    for k in FLOORS:
        (master,) = _ids(fem, pg=f"floor_{k}_master")
        want.append((3, master, _ids(fem, pg=f"floor_{k}_joints")))
    # main: [(3, 0, []), (3, 0, [])] -- the plane through the origin
    # collected no floor node.
    assert got == want
    assert all(len(slaves) == 9 for _, _, slaves in got)


def test_1778_the_deck_carries_a_master_and_nine_slaves_per_floor(
        frames, tmp_path: Path):
    deck = tmp_path / "omitted.tcl"
    _bridge(frames["omitted"]).tcl(str(deck))
    lines = [ln.split() for ln in deck.read_text().splitlines()
             if ln.startswith("rigidDiaphragm")]
    # main: two lines "rigidDiaphragm 3 0".
    assert len(lines) == 2
    for k, parts in zip(FLOORS, lines):
        (master,) = _ids(frames["omitted"], pg=f"floor_{k}_master")
        assert parts[:3] == ["rigidDiaphragm", "3", str(master)]
        assert len(parts) == 3 + 9


# ---------------------------------------------------------------------------
# Frame oracle — omitted point vs given point
# ---------------------------------------------------------------------------

def test_frame_without_point_ties_the_same_nodes_as_with_it(frames):
    given, omitted = _diaphragms(frames["given"]), _diaphragms(frames["omitted"])
    assert len(given) == len(omitted) == N_STOREYS
    for (pg, mg, sg), (po, mo, so) in zip(given, omitted):
        assert pg == po == 3
        assert sorted([mg, *sg]) == sorted([mo, *so])
    # Both floors: the one-node master wins with the point given too
    # (B5, #1804), so the records are identical.
    assert given == omitted


def test_frame_decks_are_identical_with_and_without_the_point(frames, tmp_path):
    texts = {}
    for mode, fem in frames.items():
        path = tmp_path / f"{mode}.tcl"
        _bridge(fem).tcl(str(path))
        texts[mode] = path.read_text().splitlines()
    # Before B5 (#1804) floor 2's rigidDiaphragm line named the coincident
    # joint J11_L2 as master when the point was given.
    assert texts["given"] == texts["omitted"]


@pytest.mark.live
def test_frame_without_point_has_the_same_periods(frames):
    pytest.importorskip("openseespy.opensees")
    periods = {}
    for mode, fem in frames.items():
        periods[mode] = np.asarray(list(_bridge(fem).eigen(3).periods), float)
    # main: the omitted run raised OpenSeesError ("rigidDiaphragm 3 0").
    np.testing.assert_allclose(periods["omitted"], periods["given"], rtol=1e-9)
    # Symmetric plan: the X and Y translation modes share T1.
    assert periods["omitted"][0] == pytest.approx(periods["omitted"][1], rel=1e-9)
    assert periods["omitted"][0] == pytest.approx(0.254293, abs=5e-7)


# ---------------------------------------------------------------------------
# Refusals
# ---------------------------------------------------------------------------

def test_multi_node_master_without_point_is_refused():
    with apeGmsh(model_name="r3s4_multi") as g:
        for i, x in enumerate((0.0, 1.0, 2.0)):
            g.model.geometry.add_point(x, 0.0, 3.0, label=f"p{i}")
        g.model.geometry.add_line("p0", "p1", label="a")
        g.model.geometry.add_line("p1", "p2", label="b")
        g.physical.add_point(["p0", "p1"], name="two")
        g.physical.add_point(["p2"], name="one")
        g.constraints.rigid_diaphragm("two", "one")
        g.mesh.generation.generate(dim=1)
        with pytest.raises(ValueError,
                           match=r"must resolve to exactly one node, but it "
                                 r"resolves to 2.*master_point"):
            g.mesh.queries.get_fem_data(dim=1)


def _resolver() -> ConstraintResolver:
    coords = {1: (0.0, 0.0, 3.0), 2: (1.0, 0.0, 3.0), 3: (0.0, 1.0, 3.0),
              4: (0.0, 0.0, 6.0)}
    tags = np.array(sorted(coords), dtype=int)
    return ConstraintResolver(tags, np.array([coords[t] for t in tags]))


def _defn(**kw) -> RigidDiaphragmDef:
    return RigidDiaphragmDef(master_label="m", slave_label="s",
                             plane_tolerance=0.01, **kw)


def test_default_master_point_is_none():
    assert _defn().master_point is None


def test_zero_node_master_without_point_is_refused():
    with pytest.raises(ValueError, match="resolves to 0"):
        _resolver().resolve_rigid_diaphragm(_defn(), {2, 3}, master_nodes=set())


def test_master_whose_plane_holds_no_slave_is_refused():
    # Master 4 at z = 6; the slaves sit at z = 3.
    with pytest.raises(ValueError, match="no node of 's' lies within"):
        _resolver().resolve_rigid_diaphragm(
            _defn(), {2, 3, 4}, master_nodes={4})


def test_one_node_master_sets_the_plane_and_the_offsets():
    rec = _resolver().resolve_rigid_diaphragm(
        _defn(), {1, 2, 3, 4}, master_nodes={1})
    assert rec.master_node == 1
    assert rec.slave_nodes == [2, 3]          # node 4 is off the z = 3 plane
    np.testing.assert_allclose(rec.offsets, [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])


def test_given_point_keeps_a_one_node_master():
    # Point (0.9, 0, 3): the nearest plane node is 2, but the master label
    # names node 1 alone, so node 1 stays the master (B5, #1804).
    rec = _resolver().resolve_rigid_diaphragm(
        _defn(master_point=(0.9, 0.0, 3.0)), {1, 2, 3}, master_nodes={1})
    assert rec.master_node == 1
    assert sorted(rec.slave_nodes) == [2, 3]


def test_given_point_whose_plane_is_empty_is_refused():
    # Plane z = 0 through the point: no node of {1, 2, 3} lies there.
    # main: a record (3, 0, []) and the deck line "rigidDiaphragm 3 0".
    with pytest.raises(ValueError, match=r"master_point=\(0\.0, 0\.0, 0\.0\)"
                                         r".*collects 0 node"):
        _resolver().resolve_rigid_diaphragm(
            _defn(master_point=(0.0, 0.0, 0.0)), {1, 2, 3}, master_nodes={1})


def test_given_point_whose_plane_holds_only_the_master_is_refused():
    # Plane z = 6 holds node 4 alone: a master with nothing to drive.
    # main: a record (3, 4, []) and the deck line "rigidDiaphragm 3 4".
    with pytest.raises(ValueError, match="collects 1 node"):
        _resolver().resolve_rigid_diaphragm(
            _defn(master_point=(0.0, 0.0, 6.0)), {2, 3, 4}, master_nodes={4})


def test_frame_floor_with_a_wrong_given_point_is_refused():
    """#1778 with the point given at z = 0 for an upper floor: the frame
    declaration now refuses instead of emitting ``rigidDiaphragm 3 0``."""
    with apeGmsh(model_name="r3s4_wrong_point") as g:
        g.model.geometry.add_point(0.0, 0.0, 3.0, label="m")
        g.model.geometry.add_point(1.0, 0.0, 3.0, label="s")
        g.model.geometry.add_line("m", "s", label="beam")
        g.physical.add_point(["m"], name="master")
        g.physical.add_point(["s"], name="slave")
        g.constraints.rigid_diaphragm("master", "slave",
                                      master_point=(0.0, 0.0, 0.0),
                                      plane_tolerance=0.05)
        g.mesh.generation.generate(dim=1)
        with pytest.raises(ValueError, match="collects 0 node"):
            g.mesh.queries.get_fem_data(dim=1)


def test_chain_phase_route_applies_the_one_node_rule():
    from apeGmsh._kernel.resolvers._chain_phase_router import route_def_to_fem

    fem = _frame(given=True)   # any FEM carrying the two labels
    defn = RigidDiaphragmDef(master_label="floor_2_master",
                             slave_label="floor_2_joints",
                             plane_tolerance=PLANE_TOL)
    out = route_def_to_fem(fem, defn)
    (master,) = _ids(fem, pg="floor_2_master")
    assert _diaphragms(out)[-1] == (3, master, _ids(fem, pg="floor_2_joints"))
    bad = RigidDiaphragmDef(master_label="floor_2_joints",
                            slave_label="floor_2_master",
                            plane_tolerance=PLANE_TOL)
    with pytest.raises(ValueError, match="resolves to 9"):
        route_def_to_fem(fem, bad)


# ---------------------------------------------------------------------------
# D4 — p.load on a labelled node
# ---------------------------------------------------------------------------

def test_p_load_on_a_labelled_node_via_nodes_one(tmp_path: Path):
    from apeGmsh.opensees import apeSees

    with apeGmsh(model_name="r3s4_tip") as g:
        g.model.geometry.add_point(0.0, 0.0, 0.0, label="base")
        g.model.geometry.add_point(2.0, 0.0, 0.0, label="tip")
        g.model.geometry.add_line("base", "tip", label="beam")
        g.physical.add_curve(["beam"], name="Beam")
        g.physical.add_point(["base"], name="Base")
        g.mesh.structured.set_transfinite_curve("Beam", n_nodes=3)
        g.mesh.generation.generate(dim=1)
        fem = g.mesh.queries.get_fem_data(dim=1)

    (tip,) = _ids(fem, label="tip")
    ops = apeSees(fem)
    ops.model(ndm=2, ndf=3)
    tr = ops.geomTransf.Linear()
    ops.element.elasticBeamColumn(pg="Beam", transf=tr, A=1.0, E=1.0, Iz=1.0)
    ops.fix(pg="Base", dofs=(1, 1, 1))
    with ops.pattern.Plain(series=ops.timeSeries.Linear()) as p:
        p.load(node=ops.nodes.one(label="tip"), forces=(0.0, -5.0, 0.0))
    deck = tmp_path / "tip.tcl"
    ops.tcl(str(deck))
    loads = [ln for ln in deck.read_text().splitlines()
             if re.match(r"\s*load\s", ln)]
    assert [ln.split()[1] for ln in loads] == [str(tip)]
    assert [float(v) for v in loads[0].split()[2:]] == [0.0, -5.0, 0.0]
