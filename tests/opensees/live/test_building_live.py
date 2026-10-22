"""Live validation of :mod:`apeGmsh.opensees.building` against hand statics.

The 3-storey frame + wall building of ``_building_model.py`` carries random
nodal forces (one nodal moment, one mid-story load) and a uniform load on
one beam. Each story force is checked against the resultant of the loads
above the cut, computed independently here; the pier forces of a
cantilever wall against closed-form statics; drifts against ``nodeDisp``;
story masses against hand sums and ``modalProperties``. Stock elements
only: these run on stock openseespy and on the fork.
"""
from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("openseespy.opensees")

from apeGmsh.opensees import building as B  # noqa: E402
from apeGmsh.opensees._internal.build.errors import BridgeError  # noqa: E402
from apeGmsh.opensees.emitter.live import get_ops  # noqa: E402

from tests.opensees.live._building_model import (  # noqa: E402
    LEVELS,
    M_BM,
    M_COL,
    RHO,
    T_SLAB,
    T_WALL,
    Building,
    build,
)

pytestmark = pytest.mark.live
REL = 1e-10


def _expected(b: Building, z: float, *, upper_inclusive: bool,
              ref_xy: tuple[float, float] = (0.0, 0.0)) -> np.ndarray:
    """Resultant ``[F, M]`` of the loads on the part above a cut at ``z``."""
    ref = np.array([ref_xy[0], ref_xy[1], z])
    out = np.zeros(6)

    def above(zz: float) -> bool:
        return zz >= z - 1e-9 if upper_inclusive else zz > z + 1e-9

    for t, f in b.loads.items():
        c = np.array(b.coords[t])
        if above(c[2]):
            out[:3] += f[:3]
            out[3:] += np.cross(c - ref, f[:3]) + f[3:]
    for _e, p, f in b.ele_loads:
        if above(p[2]):
            out[:3] += f
            out[3:] += np.cross(p - ref, f)
    return out


def _close(got: np.ndarray, want: np.ndarray) -> None:
    scale = max(1.0, float(np.abs(want).max()))
    np.testing.assert_allclose(got, want, rtol=0.0, atol=REL * scale)


def test_story_shear_bottom_equals_loads_above() -> None:
    ops = get_ops()
    b = build(ops)
    ref = (1.25, -0.5)
    sf = B.StoryForces(LEVELS, reference=ref)
    t = sf.table()
    assert t.rows == ("Story1", "Story2", "Story3")
    assert t.columns == ("VX", "VY", "P", "MX", "MY", "MZ")
    for i, story in enumerate(t.rows, start=1):
        _close(t.row(story), _expected(b, LEVELS[i - 1], upper_inclusive=False,
                                       ref_xy=ref))


def test_story_shear_top_equals_loads_at_and_above_level() -> None:
    ops = get_ops()
    b = build(ops)
    t = B.story_forces(LEVELS, location="top")
    for i, story in enumerate(t.rows, start=1):
        _close(t.row(story), _expected(b, LEVELS[i], upper_inclusive=True))
    # the mid-story load of story 1 makes its top and bottom differ
    assert abs(t.row("Story1")[0] - B.story_forces(LEVELS).row("Story1")[0]) > 1e-3


def test_base_story_equals_minus_sum_of_reactions() -> None:
    ops = get_ops()
    build(ops)
    base = B.base_shear_from_reactions(0.0, reference=(2.0, 1.0))
    story1 = B.StoryForces(LEVELS, reference=(2.0, 1.0)).table().row("Story1")
    _close(story1, base)
    # and -sum(R) by hand, forces only
    ops.reactions()
    r = np.sum([ops.nodeReaction(n)[:3] for n in ops.getNodeTags()
                if ops.nodeCoord(n)[2] == 0.0], axis=0)
    _close(base[:3], -r)


def test_extractor_protocol_and_combined_table() -> None:
    ops = get_ops()
    build(ops)
    sf = B.StoryForces(LEVELS)
    v = sf(ops)
    assert v.shape == (len(sf.labels),) == (18,)
    assert sf.labels[:2] == ("Story1:VX", "Story1:VY")
    combo = np.sqrt(v**2 + (0.5 * v) ** 2)     # an SRSS of two "modes"
    np.testing.assert_allclose(sf.table(combo).values.ravel(), combo)
    with pytest.raises(ValueError, match="labels"):
        sf.table(v[:-1])


def test_cut_crossing_an_element_refuses() -> None:
    ops = get_ops()
    build(ops, with_loads=False)
    # z = 4.5: wall nodes lie on it, the story-2 columns do not
    with pytest.raises(BridgeError, match="cross the cut at z=4.5"):
        B.StoryForces((0.0, 4.5, 9.0))


def test_mp_constraint_across_a_cut_refuses() -> None:
    ops = get_ops()
    b = build(ops, with_loads=False)
    lo = next(t for t, c in b.coords.items() if c == (6.0, 4.0, 3.0))
    hi = next(t for t, c in b.coords.items() if c == (6.0, 4.0, 6.0))
    ops.equalDOF(lo, hi, 1)
    with pytest.raises(BridgeError, match="MP constraint"):
        B.StoryForces(LEVELS)


# ---------------------------------------------------------------------------
# Pier forces
# ---------------------------------------------------------------------------


def _cantilever_wall(ops: object, element: str, theta: float) -> tuple[list[int], dict[str, float]]:
    """A ``3 x 6`` wall in plan direction ``theta``, 3 x 4 elements, fixed base."""
    lw, hw, nx, nz = 3.0, 6.0, 3, 4
    H, N, F3 = 40.0, 120.0, 7.0
    e2 = np.array([np.cos(theta), np.sin(theta), 0.0])
    e3 = np.cross([0.0, 0.0, 1.0], e2)
    ops.wipe()
    ops.model("basic", "-ndm", 3, "-ndf", 6)
    ops.section("ElasticMembranePlateSection", 1, 25e6, 0.2, 0.25, 0.0)
    x0 = np.array([10.0, -4.0, 2.0])               # off the origin on purpose

    def tag(i: int, k: int) -> int:
        return 1 + i + k * (nx + 1)

    for k in range(nz + 1):
        for i in range(nx + 1):
            p = x0 + (i * lw / nx - lw / 2) * e2 + np.array([0, 0, k * hw / nz])
            ops.node(tag(i, k), *p)
            if k == 0:
                ops.fix(tag(i, k), 1, 1, 1, 1, 1, 1)
    eles = []
    for k in range(nz):
        for i in range(nx):
            e = 1 + i + k * nx
            ops.element(element, e, tag(i, k), tag(i + 1, k), tag(i + 1, k + 1),
                        tag(i, k + 1), 1)
            eles.append(e)
    ops.timeSeries("Linear", 1)
    ops.pattern("Plain", 1, 1)
    top = [tag(i, nz) for i in range(nx + 1)]
    for t in top:
        f = (H * e2 + F3 * e3) / len(top) + np.array([0.0, 0.0, -N / len(top)])
        ops.load(t, *f, 0.0, 0.0, 0.0)
    ops.constraints("Plain")
    ops.numberer("RCM")
    ops.system("BandGeneral")
    ops.test("NormDispIncr", 1e-12, 10)
    ops.algorithm("Linear")
    ops.integrator("LoadControl", 1.0)
    ops.analysis("Static")
    assert ops.analyze(1) == 0
    return eles, {"H": H, "N": N, "F3": F3, "hw": hw, "z0": x0[2],
                  "theta": theta}


@pytest.mark.parametrize("element", ["ShellMITC4", "ASDShellQ4"])
@pytest.mark.parametrize("theta", [0.0, np.pi / 6])
def test_pier_forces_of_a_cantilever_wall_match_statics(element: str, theta: float) -> None:
    ops = get_ops()
    eles, s = _cantilever_wall(ops, element, theta)
    pf = B.PierForces([B.Pier("W1", tuple(eles))])
    t = pf.table()
    H, N, F3, hw = s["H"], s["N"], s["F3"], s["hw"]
    want_bottom = np.array([-N, H, F3, 0.0, -F3 * hw, H * hw])
    want_top = np.array([-N, H, F3, 0.0, 0.0, 0.0])
    _close(t.row("W1:bottom"), want_bottom)
    _close(t.row("W1:top"), want_top)
    np.testing.assert_allclose(pf.axes("W1")[1],
                               [np.cos(theta), np.sin(theta), 0.0], atol=1e-12)
    np.testing.assert_allclose(pf.centroids().row("W1:bottom"),
                               [10.0, -4.0, s["z0"]], atol=1e-12)


def test_pier_over_every_vertical_element_equals_the_story_force() -> None:
    ops = get_ops()
    build(ops)
    story1 = [e for e in ops.getEleTags()
              if (zs := [ops.nodeCoord(n)[2] for n in ops.eleNodes(e)])
              and max(zs) <= 3.0 and max(zs) > min(zs)]
    pier = B.Pier("all", tuple(story1), bottom=0.0, top=3.0, axis2=(1.0, 0.0))
    row = B.pier_forces([pier]).row("all:bottom")
    sf = B.story_forces(LEVELS).row("Story1")
    _close(row[[1, 2, 0]], sf[:3])          # V2, V3, P == VX, VY, P


def test_piers_by_story_splits_the_wall() -> None:
    ops = get_ops()
    b = build(ops)
    piers = B.piers_by_story(B.Pier("W", tuple(b.wall)), LEVELS)
    assert [p.name for p in piers] == ["W/Story1", "W/Story2", "W/Story3"]
    for k, p in enumerate(piers, start=1):
        assert sorted(p.elements) == sorted(b.wall_by_story[k])
        assert (p.bottom, p.top) == (LEVELS[k - 1], LEVELS[k])
    t = B.pier_forces(piers)
    assert t.rows[0] == "W/Story1:bottom" and t.values.shape == (6, 6)
    with pytest.raises(BridgeError, match="reach outside"):
        B.PierForces([B.Pier("W", tuple(b.wall), bottom=0.0, top=3.0)])


# ---------------------------------------------------------------------------
# Drifts
# ---------------------------------------------------------------------------


def test_joint_drifts_match_nodedisp_differences() -> None:
    ops = get_ops()
    b = build(ops)
    d = B.StoryDrifts(LEVELS)
    v = d.table()
    by_xyz = {c: t for t, c in b.coords.items()}
    want_max = np.zeros((3, 4))           # max |DX|, |DY|, then the means
    n_joints = 0
    for i in range(1, 4):
        h = LEVELS[i] - LEVELS[i - 1]
        story = []
        for (x, y, z), t in by_xyz.items():
            if z != LEVELS[i] or (x, y, LEVELS[i - 1]) not in by_xyz:
                continue
            bot = by_xyz[(x, y, LEVELS[i - 1])]
            want = (np.array(ops.nodeDisp(t)[:2]) - np.array(ops.nodeDisp(bot)[:2])) / h
            np.testing.assert_allclose(v.row(f"Story{i}:{t}"), want, rtol=1e-13)
            story.append(np.abs(want))
            n_joints += 1
        want_max[i - 1, :2] = np.max(story, axis=0)
        want_max[i - 1, 2:] = np.mean(story, axis=0)
    assert len(v.rows) == n_joints
    np.testing.assert_allclose(d.story_max().values, want_max, rtol=1e-13)
    np.testing.assert_allclose(B.story_drifts(LEVELS).values, want_max, rtol=1e-13)


def test_point_drifts_nearest_and_its_offset_guard() -> None:
    ops = get_ops()
    b = build(ops)
    by_xyz = {c: t for t, c in b.coords.items()}
    t = B.story_drifts(LEVELS, points=[(6.0, 4.0)])
    for i in range(1, 4):
        top, bot = by_xyz[(6.0, 4.0, LEVELS[i])], by_xyz[(6.0, 4.0, LEVELS[i - 1])]
        want = (np.array(ops.nodeDisp(top)[:2]) - np.array(ops.nodeDisp(bot)[:2])) / 3.0
        np.testing.assert_allclose(t.row(f"Story{i}:P0"), want, rtol=1e-13)
    with pytest.raises(BridgeError, match="max_offset"):
        B.StoryDrifts(LEVELS, points=[(6.05, 4.0)])
    near = B.story_drifts(LEVELS, points=[(6.05, 4.0)], max_offset=0.1)
    np.testing.assert_allclose(near.values, t.values)


def test_rigid_point_drift_is_exact_for_a_rigid_level_motion() -> None:
    ops = get_ops()
    b = build(ops, with_loads=False)
    a = [0.0, 0.010, 0.025, 0.031]
    bb = [0.0, -0.004, 0.002, 0.012]
    th = [0.0, 0.001, -0.002, 0.0035]
    for t, (x, y, z) in b.coords.items():
        k = LEVELS.index(z) if z in LEVELS else None
        if k is None:
            continue
        ops.setNodeDisp(t, 1, a[k] - th[k] * y, "-commit")
        ops.setNodeDisp(t, 2, bb[k] + th[k] * x, "-commit")
    p = (1.7, 3.3)                                  # not a node
    t = B.story_drifts(LEVELS, points=[p], method="rigid")
    for i in range(1, 4):
        ux = lambda k: a[k] - th[k] * p[1]          # noqa: E731
        uy = lambda k: bb[k] + th[k] * p[0]         # noqa: E731
        want = np.array([ux(i) - ux(i - 1), uy(i) - uy(i - 1)]) / 3.0
        np.testing.assert_allclose(t.row(f"Story{i}:P0"), want, rtol=1e-10)


def test_story_displacements_match_nodedisp() -> None:
    ops = get_ops()
    b = build(ops)
    u = B.StoryDisplacements(LEVELS)
    t = u.table()
    want = np.zeros((3, 4))
    for k in (1, 2, 3):
        level = [t_ for t_, c in b.coords.items() if c[2] == LEVELS[k]]
        disp = np.array([ops.nodeDisp(n)[:2] for n in level])
        for n, d in zip(level, disp):
            np.testing.assert_allclose(t.row(f"Story{k}:{n}"), d, rtol=1e-14)
        want[k - 1, :2] = np.abs(disp).max(axis=0)
        want[k - 1, 2:] = np.abs(disp).mean(axis=0)
    assert len(t.rows) == sum(1 for c in b.coords.values() if c[2] > 0 and c[2] in LEVELS)
    np.testing.assert_allclose(u.story_max().values, want, rtol=1e-14)
    corner = next(n for n, c in b.coords.items() if c == (6.0, 4.0, 9.0))
    p = B.StoryDisplacements(LEVELS, points=[(6.0, 4.0)])
    np.testing.assert_allclose(p.table().row("Story3:P0"), ops.nodeDisp(corner)[:2],
                               rtol=1e-14)


# ---------------------------------------------------------------------------
# Masses
# ---------------------------------------------------------------------------


def _hand_level_masses(b: Building) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Mass and ``(x, y)`` first moments per level: each element's mass at
    its centroid, split between the bounding levels linearly in height."""
    z = np.asarray(LEVELS)
    m = np.zeros(4)
    mx = np.zeros(4)
    my = np.zeros(4)

    def add(mass: float, c: np.ndarray) -> None:
        if c[2] <= z[0]:
            w = {0: 1.0}
        else:
            i = int(np.searchsorted(z, c[2] - 1e-12))
            t = (c[2] - z[i - 1]) / (z[i] - z[i - 1])
            w = {i - 1: 1.0 - t, i: t}
        for k, wk in w.items():
            m[k] += wk * mass
            mx[k] += wk * mass * c[0]
            my[k] += wk * mass * c[1]

    for p1, p2, ml in b.line_mass:
        add(ml * float(np.linalg.norm(p2 - p1)), 0.5 * (p1 + p2))
    for corners, rho_h in b.area_mass:
        d1, d2 = corners[2] - corners[0], corners[3] - corners[1]
        add(rho_h * 0.5 * float(np.linalg.norm(np.cross(d1, d2))), corners.mean(axis=0))
    for t, mn in b.nodal_mass.items():
        add(mn, np.array(b.coords[t]))
    return m, mx, my


def test_story_masses_match_hand_sums_and_modal_properties() -> None:
    ops = get_ops()
    b = build(ops, with_loads=False)
    t = B.story_masses(LEVELS)
    assert t.rows == ("Base", "Story1", "Story2", "Story3")
    m, mx, my = _hand_level_masses(b)
    for col in ("MassX", "MassY", "MassZ"):
        np.testing.assert_allclose(t.column(col), m, rtol=1e-12)
    np.testing.assert_allclose(t.column("XCM"), mx / m, rtol=1e-12)
    np.testing.assert_allclose(t.column("YCM"), my / m, rtol=1e-12)
    # closed form of level 1: slab + its beams + half of the columns and
    # wall of the two stories it bounds
    slab = RHO * T_SLAB * 24.0
    level1 = slab + M_BM * 20.0 + M_COL * 4 * 3.0 + RHO * T_WALL * 6.0
    assert t.row("Story1")[0] == pytest.approx(level1, rel=1e-12)
    # modalProperties: totalMass is every row; totalFreeMass leaves out the
    # mass lumped on the fixed base nodes (the columns' lower halves, 0.3
    # on the split column, and the wall's bottom element row), while the
    # Base row also holds half of the mid-story nodes' mass.
    on_fixed = M_COL * 1.5 * 3 + M_COL * 0.75 + RHO * T_WALL * 2.0 * 1.5 / 2
    ops.constraints("Transformation")
    ops.numberer("RCM")
    ops.system("BandGeneral")
    ops.algorithm("Linear")
    ops.integrator("LoadControl", 1.0)
    ops.analysis("Static")
    ops.eigen(3)
    props = ops.modalProperties("-return")
    total = t.values[:, 0].sum()
    np.testing.assert_allclose(np.asarray(props["totalMass"][:3]), total,
                               rtol=1e-12)
    np.testing.assert_allclose(np.asarray(props["totalFreeMass"][:3]),
                               total - on_fixed, rtol=1e-12)


def test_story_masses_leave_the_accelerations_untouched() -> None:
    ops = get_ops()
    build(ops, with_loads=False)
    B.story_masses(LEVELS)
    assert all(not any(ops.nodeAccel(n)) for n in ops.getNodeTags())
    nodal_only = B.story_masses(LEVELS, include_elements=False)
    assert nodal_only.column("MassX").tolist() == [0.0, 0.0, 1.5, 0.7]


# ---------------------------------------------------------------------------
# Fork: a LadrunoMembrane diaphragm
# ---------------------------------------------------------------------------


@pytest.mark.ladruno_fork
def test_ladruno_membrane_slab_mass_and_story_shear() -> None:
    """A one-story frame with a LadrunoMembrane slab (``rho t A`` of mass,
    in the level plane): the membrane's eleForce / eleDynamicalForce are one
    value per node DOF, its mass reaches the level, and the story shear is
    the applied lateral load."""
    ops = get_ops()
    ops.wipe()
    ops.model("basic", "-ndm", 3, "-ndf", 6)
    L, H = 6.0, 3.0
    corners = [(1, 0.0, 0.0), (2, L, 0.0), (3, L, L), (4, 0.0, L), (5, L / 2, L / 2)]
    for t, x, y in corners:
        ops.node(t, x, y, H)
        ops.node(100 + t, x, y, 0.0)
        ops.fix(100 + t, 1, 1, 1, 1, 1, 1)
    ops.geomTransf("Linear", 1, 1.0, 0.0, 0.0)
    for t, _x, _y in corners:
        ops.element("elasticBeamColumn", t, 100 + t, t, 0.16, 25e6, 25e6 / 2.4,
                    0.003, 0.002, 0.002, 1)
    rho, thick = 2.4, 0.2
    ops.element("LadrunoMembrane", 50, "-nodes", 1, 2, 3, 4, "-stiff", 25e6, 0.2,
                thick, "-rho", rho, "-supportNodes", 1, 2, 3, 4, 5)
    ops.timeSeries("Linear", 1)
    ops.pattern("Plain", 1, 1)
    ops.load(1, 100.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    ops.load(3, 0.0, 40.0, 0.0, 0.0, 0.0, 0.0)
    ops.system("UmfPack")
    ops.numberer("RCM")
    ops.constraints("Transformation")
    ops.integrator("LoadControl", 1.0)
    ops.algorithm("Linear")
    ops.analysis("Static")
    assert ops.analyze(1) == 0
    row = B.story_forces((0.0, H)).row("Story1")
    _close(row, [100.0, 40.0, 0.0, -40.0 * H, 100.0 * H, 40.0 * L])
    _close(B.base_shear_from_reactions(0.0), row)
    m = B.story_masses((0.0, H))
    assert m.row("Story1")[0] == pytest.approx(rho * thick * L * L, rel=1e-12)
    assert m.row("Story1")[3:].tolist() == pytest.approx([L / 2, L / 2], rel=1e-12)
    ops.eigen(3)
    total = ops.modalProperties("-return")["totalMass"][0]
    assert total == pytest.approx(m.column("MassX").sum(), rel=1e-12)
