"""A 3-storey frame + wall building built straight into the live domain.

Shared by ``test_building_live.py``. Stock elements only (elasticBeamColumn,
ShellMITC4), so it runs on stock openseespy and on the fork. Everything the
tests check by hand is recorded on :class:`Building`.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from types import ModuleType

import numpy as np

LEVELS = (0.0, 3.0, 6.0, 9.0)
E = 25e6
NU = 0.2
RHO = 2.4          # mass density (t/m3)
T_WALL = 0.2
T_SLAB = 0.15
A_COL, I_COL = 0.16, 0.16**2 / 12.0
A_BM, I_BM = 0.12, 0.3 * 0.4**3 / 12.0
M_COL = 0.4        # column mass per length
M_BM = 0.3         # beam mass per length


@dataclass
class Building:
    coords: dict[int, tuple[float, float, float]] = field(default_factory=dict)
    loads: dict[int, np.ndarray] = field(default_factory=dict)       # node -> 6 forces
    # (element tag, resultant point, force vector) of each element load
    ele_loads: list[tuple[int, np.ndarray, np.ndarray]] = field(default_factory=list)
    wall: list[int] = field(default_factory=list)                    # ShellMITC4 tags
    wall_by_story: dict[int, list[int]] = field(default_factory=dict)
    nodal_mass: dict[int, float] = field(default_factory=dict)
    line_mass: list[tuple[np.ndarray, np.ndarray, float]] = field(default_factory=list)
    area_mass: list[tuple[np.ndarray, float]] = field(default_factory=list)
    base_nodes: list[int] = field(default_factory=list)


def build(ops: ModuleType, *, with_loads: bool = True) -> Building:
    b = Building()
    ops.wipe()
    ops.model("basic", "-ndm", 3, "-ndf", 6)
    tag = [0]

    def node(x: float, y: float, z: float) -> int:
        for t, c in b.coords.items():
            if np.allclose(c, (x, y, z)):
                return t
        tag[0] += 1
        ops.node(tag[0], float(x), float(y), float(z))
        b.coords[tag[0]] = (float(x), float(y), float(z))
        return tag[0]

    ops.geomTransf("Linear", 1, 1.0, 0.0, 0.0)   # columns
    ops.geomTransf("Linear", 2, 0.0, 0.0, 1.0)   # beams (local z = global Z)
    ops.section("ElasticMembranePlateSection", 1, E, NU, T_WALL, RHO)
    ops.section("ElasticMembranePlateSection", 2, E, NU, T_SLAB, RHO)
    G = E / (2 * (1 + NU))
    ele = [0]

    def beam(n1: int, n2: int, transf: int, area: float, inertia: float, m: float) -> int:
        ele[0] += 1
        ops.element("elasticBeamColumn", ele[0], n1, n2, area, E, G,
                    2 * inertia, inertia, inertia, transf, "-mass", m)
        b.line_mass.append((np.array(b.coords[n1]), np.array(b.coords[n2]), m))
        return ele[0]

    def shell(ns: list[int], sec: int, rho_h: float) -> int:
        ele[0] += 1
        ops.element("ShellMITC4", ele[0], *ns, sec)
        b.area_mass.append((np.array([b.coords[n] for n in ns]), rho_h))
        return ele[0]

    cols = [(0.0, 0.0), (6.0, 0.0), (0.0, 4.0), (6.0, 4.0)]
    for k in range(1, 4):
        z0, z1 = LEVELS[k - 1], LEVELS[k]
        for x, y in cols:
            if k == 1 and (x, y) == (0.0, 0.0):        # split: a mid-story node
                mid = node(x, y, 1.5)
                beam(node(x, y, z0), mid, 1, A_COL, I_COL, M_COL)
                beam(mid, node(x, y, z1), 1, A_COL, I_COL, M_COL)
            else:
                beam(node(x, y, z0), node(x, y, z1), 1, A_COL, I_COL, M_COL)
        # wall strip y = 0, x in [2, 4], 2 x 2 elements per story
        b.wall_by_story[k] = []
        for iz in range(2):
            for ix in range(2):
                xa, xb = 2.0 + ix, 3.0 + ix
                za, zb = z0 + 1.5 * iz, z0 + 1.5 * (iz + 1)
                t = shell([node(xa, 0, za), node(xb, 0, za), node(xb, 0, zb),
                           node(xa, 0, zb)], 1, RHO * T_WALL)
                b.wall.append(t)
                b.wall_by_story[k].append(t)
        # beams of the level
        xs = [0.0, 2.0, 3.0, 4.0, 6.0]
        for xa, xb in zip(xs[:-1], xs[1:]):
            beam(node(xa, 0, z1), node(xb, 0, z1), 2, A_BM, I_BM, M_BM)
        beam(node(0, 4, z1), node(6, 4, z1), 2, A_BM, I_BM, M_BM)
        beam(node(0, 0, z1), node(0, 4, z1), 2, A_BM, I_BM, M_BM)
        beam(node(6, 0, z1), node(6, 4, z1), 2, A_BM, I_BM, M_BM)
        # slab (in the level plane): one quad per level
        shell([node(0, 0, z1), node(6, 0, z1), node(6, 4, z1), node(0, 4, z1)],
              2, RHO * T_SLAB)
    for t, (_x, _y, z) in b.coords.items():
        if z == 0.0:
            ops.fix(t, 1, 1, 1, 1, 1, 1)
            b.base_nodes.append(t)
    # nodal masses (a superimposed mass) at two level nodes
    for t, m in ((node(6, 4, 6.0), 1.5), (node(3, 0, 9.0), 0.7)):
        ops.mass(t, m, m, m, 0.0, 0.0, 0.0)
        b.nodal_mass[t] = m
    if not with_loads:
        return b
    ops.timeSeries("Linear", 1)
    ops.pattern("Plain", 1, 1)
    rng = np.random.default_rng(7)
    loaded = [node(0, 0, 1.5)] + [t for t, c in b.coords.items() if c[2] > 0.0][::3]
    for t in loaded:
        f = np.zeros(6)
        f[:3] = rng.uniform(-10.0, 10.0, 3)
        b.loads[t] = f
    b.loads[node(6, 4, 9.0)][3:] = (2.0, -3.0, 4.0)     # one nodal moment
    for t, f in b.loads.items():
        ops.load(t, *f)
    # a uniform gravity load on the y = 4 beam of level 2
    w = 5.0
    n1, n2 = node(0, 4, 6.0), node(6, 4, 6.0)
    beam_tag = next(e for e in ops.getEleTags()
                    if sorted(ops.eleNodes(e)) == sorted((n1, n2)))
    ops.eleLoad("-ele", beam_tag, "-type", "-beamUniform", 0.0, -w)
    b.ele_loads.append((beam_tag, np.array([3.0, 4.0, 6.0]),
                        np.array([0.0, 0.0, -w * 6.0])))
    ops.constraints("Transformation")
    ops.numberer("RCM")
    ops.system("BandGeneral")
    ops.test("NormDispIncr", 1e-12, 10)
    ops.algorithm("Linear")
    ops.integrator("LoadControl", 1.0)
    ops.analysis("Static")
    assert ops.analyze(1) == 0
    return b
