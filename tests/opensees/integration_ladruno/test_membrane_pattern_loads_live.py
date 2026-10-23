"""Per-pattern LadrunoMembrane magnitudes (D, SD, L) solve exactly on the fork.

Two 6 m x 5 m panels side by side (x = 0..6 and 6..12, at z = H), each a
``LadrunoMembrane(nodes=<boundary loop>)`` on its four edge beams (``method="tributary"``),
on six columns. Three patterns D, SD and L give each panel its own
magnitude with ``p.surface_load(ele=panel, pressure=q)``. Each pattern is
switched on alone at one step by a ``Path`` series (step k: pattern k only),
so one linear model answers all three.

Hand reference (tributary method on a rectangle, the 45-degree envelope):
a 5 m edge takes the triangle ``5 * 2.5 / 2 = 6.25`` m2 and a 6 m edge the
trapezoid ``(30 - 2 * 6.25) / 2 = 8.75`` m2, so a beam line carries
``sum_panels q * area`` and the base ``sum Rz = sum_i q_i * A_i``. A beam
line's load is read from its elements' resisting forces (the membrane has no
out-of-plane stiffness, so a line's interior nodes carry no other z force).
Per pattern it equals ``sum_panels q * a`` exactly, with ``a`` the kernel's
own unit tributary area (``eleResponse tributaryAreas``); the kernel's areas
match the hand split to about 1e-5 (its tributary partition is meshed).
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pytest

pytestmark = [pytest.mark.live, pytest.mark.ladruno_fork]

H = 3.0
PANELS = {"Panel_0": (0.0, 6.0), "Panel_1": (6.0, 12.0)}
#: kPa per pattern per panel (negative: along -z, gravity).
Q = {
    "D":  {"Panel_0": -4.0, "Panel_1": -4.5},
    "SD": {"Panel_0": -1.5, "Panel_1": -1.0},
    "L":  {"Panel_0": -2.0, "Panel_1": -3.0},
}
#: Beam lines: name -> ((x0, y0), (x1, y1)).
LINES = {
    "S0": ((0, 0), (6, 0)), "S1": ((6, 0), (12, 0)),
    "N0": ((0, 5), (6, 5)), "N1": ((6, 5), (12, 5)),
    "W": ((0, 0), (0, 5)), "M": ((6, 0), (6, 5)), "E": ((12, 0), (12, 5)),
}
#: Tributary area (m2) each panel gives each line (45-degree envelope).
AREA = {
    "S0": {"Panel_0": 8.75}, "N0": {"Panel_0": 8.75}, "W": {"Panel_0": 6.25},
    "S1": {"Panel_1": 8.75}, "N1": {"Panel_1": 8.75}, "E": {"Panel_1": 6.25},
    "M": {"Panel_0": 6.25, "Panel_1": 6.25},
}
COLS = [(0, 0), (6, 0), (12, 0), (0, 5), (6, 5), (12, 5)]


@pytest.fixture(scope="module")
def fem() -> Any:
    from apeGmsh import apeGmsh

    with apeGmsh(model_name="membrane_patterns", save_to=None) as g:
        geo = g.model.geometry
        for name, (x0, x1) in PANELS.items():
            geo.add_rectangle(x0, 0.0, H, x1 - x0, 5.0, label=name)
        for name, (a, b) in LINES.items():
            geo.add_polyline([(*a, H), (*b, H)], label=f"B_{name}")
        for x, y in COLS:
            geo.add_polyline([(x, y, 0.0), (x, y, H)], label="Cols")
        g.model.boolean.fragment(
            list(PANELS), [*(f"B_{n}" for n in LINES), "Cols"], dim=2,
            tolerance=1e-3)
        for name in LINES:
            g.physical.add_curve(
                g.labels.entities(f"B_{name}", dim=1), name=f"B_{name}")
        g.physical.add_curve(g.labels.entities("Cols", dim=1), name="Cols")
        base = g.model.select(None, dim=0).in_box(
            (-1, -1, -1e-3), (13, 6, 1e-3))
        base.to_physical("BasePts")
        g.mesh.sizing.set_global_size(1.0)
        # Lines only: each membrane is declared on its explicit boundary
        # loop (the edge-beam nodes), so the deck has no slab-interior node.
        g.mesh.generation.generate(dim=1)
        return g.mesh.queries.get_fem_data(dim=None)


def _boundary_loop(fem: Any, x0: float, x1: float) -> tuple[int, ...]:
    """The slab-level nodes on the rectangle [x0, x1] x [0, 5], CCW."""
    tol = 1e-9
    pts = []
    for n, c in zip(fem.nodes.ids, fem.nodes.coords):
        x, y, z = (float(v) for v in c)
        on_edge = (abs(x - x0) < tol or abs(x - x1) < tol
                   or abs(y) < tol or abs(y - 5.0) < tol)
        inside = x0 - tol <= x <= x1 + tol and -tol <= y <= 5.0 + tol
        if abs(z - H) < tol and inside and on_edge:
            pts.append((float(np.arctan2(y - 2.5, x - 0.5 * (x0 + x1))), int(n)))
    return tuple(n for _a, n in sorted(pts))


#: The beam lines on each panel's edges.
EDGES = {"Panel_0": ("S0", "N0", "W", "M"), "Panel_1": ("S1", "N1", "M", "E")}


def _solve(fem: Any) -> "dict[str, Any]":
    from apeGmsh.opensees import apeSees
    from apeGmsh.opensees.emitter.live import get_ops

    ops = apeSees(fem, _artifacts=False)
    ops.model(ndm=3, ndf=6)
    E = 25e6
    bt = ops.geomTransf.Linear(vecxz=(0.0, 0.0, 1.0))
    ct = ops.geomTransf.Linear(vecxz=(1.0, 0.0, 0.0))
    beams = {
        name: ops.element.elasticBeamColumn(
            pg=f"B_{name}", transf=bt, A=0.18, E=E, G=E / 2.4,
            Iy=0.3 * 0.6**3 / 12, Iz=0.6 * 0.3**3 / 12, J=0.003)
        for name in LINES
    }
    ops.element.elasticBeamColumn(
        pg="Cols", transf=ct, A=0.25, E=E, G=E / 2.4,
        Iy=0.5**4 / 12, Iz=0.5**4 / 12, J=0.008)
    mems = {
        name: ops.element.LadrunoMembrane(
            nodes=_boundary_loop(fem, x0, x1),
            frames=tuple(beams[e] for e in EDGES[name]), method="tributary",
            E=0.5 * E, nu=0.2, thickness=0.25)
        for name, (x0, x1) in PANELS.items()
    }
    ops.fix(pg="BasePts", dofs=(1, 1, 1, 1, 1, 1))
    for k, case in enumerate(Q, start=1):
        on = tuple(1.0 if j == k else 0.0 for j in range(len(Q) + 1))
        series = ops.timeSeries.Path(
            time=tuple(float(j) for j in range(len(Q) + 1)), values=on)
        with ops.pattern.Plain(series=series, name=case) as p:
            for name, m in mems.items():
                p.surface_load(ele=m, pressure=Q[case][name])
    ops.constraints.Plain()
    ops.numberer.RCM()
    ops.system.UmfPack()
    ops.test.NormDispIncr(tol=1e-10, max_iter=10)
    ops.algorithm.Linear()
    ops.integrator.LoadControl(dlam=1.0)
    ops.analysis.Static()

    o = get_ops()
    out: dict[str, Any] = {"unit": {}}
    line_tags: dict[str, set[int]] = {}
    for k, case in enumerate(Q):
        # ops.analyze builds the live domain and takes step 1; the next
        # steps continue in that domain.
        assert (ops.analyze(steps=1) if k == 0 else o.analyze(1)) == 0
        o.reactions()
        rz = sum(o.nodeReaction(int(n), 3) for n in o.getNodeTags())
        out[case] = {"sum_rz": rz, "lines": {}}
        if k == 0:
            # The live domain exists now: map every beam line to its tags,
            # and read the kernel's own unit tributary area per (panel, line).
            line_tags = {name: set(_line_tags(o, fem, name)) for name in LINES}
            assert all(line_tags.values())
            mem_tags = sorted(t for t in o.getEleTags() if len(o.eleNodes(t)) > 2)
            for panel, mt in zip(PANELS, mem_tags):
                frames = [int(f) for f in o.eleResponse(mt, "frameTags")]
                areas = list(o.eleResponse(mt, "tributaryAreas"))
                out["unit"][panel] = {
                    name: sum(a for f, a in zip(frames, areas) if f in tags)
                    for name, tags in line_tags.items()
                }
        for name, tags in line_tags.items():
            # The resisting forces of a loaded element sum to minus its
            # applied load: sum Fz = -(q * area) > 0 under gravity.
            w = 0.0
            for t in sorted(tags):
                f = o.eleForce(t)
                w += f[2] + f[8]
            out[case]["lines"][name] = w
    return out


def _line_tags(o: Any, fem: Any, name: str) -> list[int]:
    """Live element tags whose two nodes lie on beam line ``name``."""
    (x0, y0), (x1, y1) = LINES[name]
    a = np.array([x0, y0, H], dtype=float)
    b = np.array([x1, y1, H], dtype=float)
    d = b - a
    tags = []
    for t in o.getEleTags():
        nodes = o.eleNodes(t)
        if len(nodes) != 2:
            continue
        ok = True
        for n in nodes:
            p = np.asarray(o.nodeCoord(n), dtype=float)
            s = float(np.dot(p - a, d) / np.dot(d, d))
            if not (-1e-9 <= s <= 1 + 1e-9) or np.linalg.norm(a + s * d - p) > 1e-9:
                ok = False
                break
        if ok:
            tags.append(int(t))
    return tags


@pytest.fixture(scope="module")
def solved(fem: Any) -> "dict[str, Any]":
    return _solve(fem)


@pytest.mark.parametrize("case", list(Q))
def test_sum_rz_per_pattern_is_q_times_area(solved: Any, case: str) -> None:
    want = -sum(Q[case][p] * 30.0 for p in PANELS)
    assert solved[case]["sum_rz"] == pytest.approx(want, rel=1e-9)


@pytest.mark.parametrize("case", list(Q))
def test_each_pattern_loads_the_beams_with_its_own_magnitude(
    solved: Any, case: str,
) -> None:
    """Exact: a line's load is sum_panels q_panel * (the kernel's unit
    tributary area of that panel on that line)."""
    for name in LINES:
        want = -sum(Q[case][p] * solved["unit"][p][name] for p in PANELS)
        got = solved[case]["lines"][name]
        assert got == pytest.approx(want, rel=1e-9, abs=1e-9), (case, name)


def test_kernel_tributary_areas_are_the_45_degree_envelope(solved: Any) -> None:
    """The kernel's unit areas against the hand 45-degree split (its
    tributary partition is meshed: measured within 1e-5 relative)."""
    for name, by_panel in AREA.items():
        for p in PANELS:
            want = by_panel.get(p, 0.0)
            got = solved["unit"][p][name]
            assert got == pytest.approx(want, rel=1e-4, abs=1e-9), (p, name, got)
