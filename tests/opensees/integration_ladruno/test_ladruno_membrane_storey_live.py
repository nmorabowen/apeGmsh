"""A meshed storey of LadrunoMembrane(pg=) panels solves on the fork.

One storey at z = H: three slab panels, each a ``LadrunoMembrane(pg=...)``
with ``frames=("Beams",)`` (the whole floor's beam group), on beams and
columns, with one shear wall (``ASDShellT3``) under the x = 12 edge:

* ``Panel_0`` [0, 6] x [0, 5] - four edge beams;
* ``Panel_1`` [6, 12] x [0, 5] - four edge beams plus an interior beam at
  x = 9 that splits the panel;
* ``Panel_2`` [0, 6] x [5, 10] - four edge beams and a 2 x 2 hole framed
  by four beams, with two trimmer beams tying the frame to the x = 0 and
  x = 6 edge beams.

Before the per-panel frame filter every panel received every beam of the
floor (and the kernel went singular, ``sum Rz = 0``), and before the
interior-node skip the panels' interior mesh nodes were emitted with no
element (K singular, every eigenvalue undetermined).  Here gravity
``q = 5`` kPa must reach the base exactly, ``sum Rz = q * sum(A_i)``, the
gravity sag must stay below 0.05, every period must be below 1 s, and mode 1
must carry more than 90 % of the X mass. ``sum Rz`` alone cannot see a
mechanism, because a singular K still balances the loads.
"""
from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Any

import numpy as np
import pytest

pytestmark = [pytest.mark.live, pytest.mark.ladruno_fork]

H = 3.0
Q = 5.0
PANELS = {
    "Panel_0": ((0.0, 6.0), (0.0, 5.0), None),
    "Panel_1": ((6.0, 12.0), (0.0, 5.0), None),
    "Panel_2": ((0.0, 6.0), (5.0, 10.0), ((2.0, 4.0), (6.5, 8.5))),
}
#: Net slab area: 30 + 30 + (30 - 4).
AREA = 86.0
BEAMS = [
    ((0, 0), (6, 0)), ((6, 0), (12, 0)), ((0, 5), (6, 5)), ((6, 5), (12, 5)),
    ((0, 0), (0, 5)), ((6, 0), (6, 5)), ((12, 0), (12, 5)), ((9, 0), (9, 5)),
    ((0, 10), (6, 10)), ((0, 5), (0, 10)), ((6, 5), (6, 10)),
    # The hole is framed: the fork gives a hole node no stiffness of its
    # own (MembraneHoleNodeFreeWarning).
    ((2, 6.5), (4, 6.5)), ((4, 6.5), (4, 8.5)), ((4, 8.5), (2, 8.5)),
    ((2, 8.5), (2, 6.5)),
    # Trimmers frame the opening into the edge beams. Without them the hole
    # ring reaches no support: the membrane holds it in its plane only, so
    # its uz / rx / ry are three rigid-body modes (T ~ 1e6 s) and gravity
    # solves a singular K (|uz| ~ 1e12 m) with sum Rz still exact.
    ((0, 6.5), (2, 6.5)), ((4, 8.5), (6, 8.5)),
]
COLS = [(0, 0), (6, 0), (12, 0), (0, 5), (6, 5), (0, 10), (6, 10)]


@pytest.fixture(scope="module")
def fem() -> Any:
    from apeGmsh import apeGmsh

    with apeGmsh(model_name="membrane_storey", save_to=None) as g:
        geo = g.model.geometry
        for name, ((x0, x1), (y0, y1), hole) in PANELS.items():
            geo.add_rectangle(x0, y0, H, x1 - x0, y1 - y0, label=name)
            if hole is not None:
                (hx0, hx1), (hy0, hy1) = hole
                geo.add_rectangle(hx0, hy0, H, hx1 - hx0, hy1 - hy0, label="Hole")
                g.model.boolean.cut(name, "Hole", dim=2, label=name)
        for a, b in BEAMS:
            geo.add_polyline([(*a, H), (*b, H)], label="Beams")
        for x, y in COLS:
            geo.add_polyline([(x, y, 0.0), (x, y, H)], label="Cols")
        geo.add_rectangle(12.0, 0.0, 0.0, 5.0, H, plane="yz", label="Wall")
        g.model.boolean.fragment(
            [*PANELS, "Wall"], ["Beams", "Cols"], dim=2, tolerance=1e-3)
        for name in PANELS:
            g.physical.add_surface(g.labels.entities(name, dim=2), name=name)
        g.physical.add_surface(g.labels.entities("Wall", dim=2), name="Wall")
        g.physical.add_curve(g.labels.entities("Beams", dim=1), name="Beams")
        g.physical.add_curve(g.labels.entities("Cols", dim=1), name="Cols")
        base = g.model.select(None, dim=0).in_box(
            (-1, -1, -1e-3), (13, 11, 1e-3))
        base.to_physical("BasePts")
        g.constraints.bc("BasePts", dofs=[1, 1, 1, 1, 1, 1])
        g.mesh.sizing.set_global_size(1.0)
        g.mesh.generation.generate(dim=2)
        return g.mesh.queries.get_fem_data(dim=None)


def _bridge(fem: Any) -> Any:
    from apeGmsh.opensees import apeSees

    ops = apeSees(fem, _artifacts=False)
    ops.model(ndm=3, ndf=6)
    E = 25e6
    bt = ops.geomTransf.Linear(vecxz=(0.0, 0.0, 1.0))
    ct = ops.geomTransf.Linear(vecxz=(1.0, 0.0, 0.0))
    ops.element.elasticBeamColumn(
        pg="Beams", transf=bt, A=0.18, E=E, G=E / 2.4,
        Iy=0.5 * 0.3 * 0.6**3 / 12, Iz=0.5 * 0.6 * 0.3**3 / 12, J=0.003)
    ops.element.elasticBeamColumn(
        pg="Cols", transf=ct, A=0.25, E=E, G=E / 2.4,
        Iy=0.7 * 0.5**4 / 12, Iz=0.7 * 0.5**4 / 12, J=0.008)
    wsec = ops.section.ElasticMembranePlateSection(E=E, nu=0.2, h=0.3, rho=0.0)
    ops.element.ASDShellT3(pg="Wall", section=wsec)
    mems = [
        ops.element.LadrunoMembrane(
            pg=name, frames=("Beams",), method="tributary",
            E=0.5 * E, nu=0.2, thickness=0.25, rho=2.4, pressure=-Q)
        for name in PANELS
    ]
    ops.fix_from_model()
    with ops.pattern.Plain(series=ops.timeSeries.Linear(), name="SD") as p:
        for m in mems:
            p.surface_load(ele=m)
    ops.constraints.Plain()
    ops.numberer.RCM()
    ops.system.UmfPack()
    ops.test.NormDispIncr(tol=1e-8, max_iter=10)
    ops.algorithm.Linear()
    ops.integrator.LoadControl(dlam=1.0)
    ops.analysis.Static()
    return ops


def _deck(ops: Any, tmp_path: Path) -> str:
    path = tmp_path / "storey.tcl"
    ops.tcl(path)
    return path.read_text(encoding="utf-8")


def _in_panel(xyz: np.ndarray, name: str) -> bool:
    (x0, x1), (y0, y1), hole = PANELS[name]
    x, y, z = (float(v) for v in xyz)
    tol = 1e-9
    if abs(z - H) > tol or not (x0 - tol <= x <= x1 + tol and y0 - tol <= y <= y1 + tol):
        return False
    if hole is not None:
        (hx0, hx1), (hy0, hy1) = hole
        if hx0 + tol < x < hx1 - tol and hy0 + tol < y < hy1 - tol:
            return False
    return True


def test_each_panel_keeps_only_its_own_beams(fem: Any, tmp_path: Path) -> None:
    deck = _deck(_bridge(fem), tmp_path)
    xyz = {int(n): np.asarray(c, dtype=float)
           for n, c in zip(fem.nodes.ids, fem.nodes.coords)}
    beams = {}
    for m in re.finditer(r"^element elasticBeamColumn (\d+) (\d+) (\d+) ", deck, re.M):
        beams[int(m.group(1))] = (int(m.group(2)), int(m.group(3)))
    membranes = re.findall(r"^element LadrunoMembrane .*$", deck, re.M)
    assert len(membranes) == 3
    for name, line in zip(PANELS, membranes):
        frames = re.search(r"-frames ([\d ]+?) -", line)
        assert frames is not None, line
        got = {int(t) for t in frames.group(1).split()}
        want = {
            t for t, (i, j) in beams.items()
            if _in_panel(xyz[i], name) and _in_panel(xyz[j], name)
            and _in_panel(0.5 * (xyz[i] + xyz[j]), name)
        }
        assert got == want, name
    # Panel_1 carries its interior beam (x = 9): its frames reach x = 9.
    p1 = re.search(r"-frames ([\d ]+?) -", membranes[1])
    assert p1 is not None
    assert any(abs(xyz[beams[int(t)][0]][0] - 9.0) < 1e-9
               for t in p1.group(1).split())


def test_interior_nodes_are_not_in_the_deck(fem: Any, tmp_path: Path) -> None:
    deck = _deck(_bridge(fem), tmp_path)
    declared = {int(n) for n in re.findall(r"^node (\d+) ", deck, re.M)}
    used: set[int] = set()
    for line in deck.splitlines():
        if line.startswith("element "):
            used.update(int(t) for t in re.findall(r"\b\d+\b", line.split(" -")[0])[1:])
            if "LadrunoMembrane" in line:
                for flag in ("-nodes", "-hole"):
                    for m in re.finditer(rf"{flag} ([\d ]+?) -", line):
                        used.update(int(t) for t in m.group(1).split())
    all_ids = {int(n) for n in fem.nodes.ids}
    assert declared < all_ids
    # Every declared node is on an element; every skipped node is a slab
    # interior node (at z = H, inside a panel).
    assert declared <= used
    xyz = {int(n): c for n, c in zip(fem.nodes.ids, fem.nodes.coords)}
    skipped = all_ids - declared
    assert skipped
    assert all(any(_in_panel(np.asarray(xyz[n]), p) for p in PANELS) for n in skipped)


def test_gravity_reaches_the_base_exactly_and_eigen_is_regular(fem: Any) -> None:
    from apeGmsh.opensees.emitter.live import get_ops

    ops = _bridge(fem)
    assert ops.analyze(steps=1) == 0
    o = get_ops()
    o.reactions()
    rz = sum(o.nodeReaction(int(n), 3) for n in o.getNodeTags())
    assert rz == pytest.approx(Q * AREA, rel=1e-9)
    # sum Rz cannot see a mechanism (a singular K still balances the loads):
    # bound the gravity sag, the periods, and the first mode's X mass.
    uz = max(abs(float(o.nodeDisp(int(n), 3))) for n in o.getNodeTags())
    assert uz < 0.05, f"gravity max |uz| = {uz:.3e}: a mechanism"
    o.loadConst("-time", 0.0)
    res = ops.eigen(4)
    periods = np.asarray(res.periods, dtype=float)
    assert np.all(np.isfinite(periods)) and np.all(periods > 0.0), periods
    assert all(math.isfinite(float(lam)) and lam > 0 for lam in res.eigenvalues)
    assert np.all(periods < 1.0), f"periods {periods}: a mechanism"
    mx = list(o.modalProperties("-return")["partiMassRatiosMX"])
    assert mx[0] > 90.0, f"mode 1 MX = {mx[0]:.3f} %"


def test_archive_replay_skips_the_same_nodes(fem: Any, tmp_path: Path) -> None:
    from apeGmsh.opensees import OpenSeesModel

    ops = _bridge(fem)
    deck = _deck(ops, tmp_path)
    ops.h5(tmp_path / "storey.h5")
    model = OpenSeesModel.from_h5(tmp_path / "storey.h5")
    replay = model.build("tcl")
    assert isinstance(replay, str)
    nodes = lambda text: set(re.findall(r"^node (\d+) ", text, re.M))  # noqa: E731
    assert nodes(replay) == nodes(deck)
