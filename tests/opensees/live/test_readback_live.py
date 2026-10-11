"""ADR 0115 D2 live oracles (R2-b, #1695), on stock openseespy.

- A Pratt truss: the three bars a section through panel 2 cuts, read by
  label, against the method of sections; the support reactions, read by
  label and summed, against the applied load.
- A 2-D cantilever at ``ndf=3``: tip ``displacement_y`` against
  ``PL^3/3EI`` and tip ``rotation_z`` against ``PL^2/2EI`` (the DOF the
  results capture table gets wrong: it sends ``rotation_z`` to DOF 6).
- The tag map (Invariant 3): element families declared in reverse, so
  sequential tags are not FEM ids; a default build and an
  ``element_tags="fem"`` build read the same values.
- Invariant 2: no tag is allocated, no plan made, no build run, during
  a read.
- Invariant 4: ``eigen``, a second ``analyze``, ``run`` and a second
  bridge each close the session, and the raise names the cause (the
  deck-run refusals need no solver: ``tests/opensees/unit/test_readback.py``).
"""
from __future__ import annotations

import math
from typing import Any

import gmsh
import pytest

from apeGmsh import apeGmsh
from apeGmsh.opensees._internal.build import BridgeError
from apeGmsh.opensees.apesees import apeSees

pytestmark = pytest.mark.live

P = 1000.0          # midspan load (N)
A_PANEL = 2.0       # panel width (m)
H = 1.5             # truss height (m)
E = 2.0e11
A_BAR = 1.0e-3


@pytest.fixture(autouse=True)
def _backend() -> None:
    pytest.importorskip("openseespy.opensees")


# ---------------------------------------------------------------------------
# The Pratt truss
# ---------------------------------------------------------------------------


def build_pratt_fem() -> Any:
    """Four panels, bottom nodes B0..B4 and top nodes T0..T4; Pratt
    diagonals (T0-B1, T1-B2, T3-B2, T4-B3). Labels on the three bars a
    section through panel 2 cuts (``top2`` = T1-T2, ``bot2`` = B1-B2,
    ``diag2`` = T1-B2), on the supports (``supports`` = B0, B4) and the
    loaded node (``mid`` = B2). PGs ``Chords`` and ``Webs``."""
    with apeGmsh(model_name="r2b_pratt", verbose=False) as g:
        geo = g.model.geometry
        b = [geo.add_point(i * A_PANEL, 0.0, 0.0) for i in range(5)]
        t = [geo.add_point(i * A_PANEL, H, 0.0) for i in range(5)]
        chords = [geo.add_line(b[i], b[i + 1]) for i in range(4)]
        chords += [geo.add_line(t[i], t[i + 1]) for i in range(4)]
        webs = [geo.add_line(b[i], t[i]) for i in range(5)]
        diag = {
            "d1": geo.add_line(t[0], b[1]),
            "d2": geo.add_line(t[1], b[2]),
            "d3": geo.add_line(t[3], b[2]),
            "d4": geo.add_line(t[4], b[3]),
        }
        webs += list(diag.values())
        for c in chords + webs:
            gmsh.model.mesh.setTransfiniteCurve(c, 2)
        g.physical.add(1, chords, name="Chords")
        g.physical.add(1, webs, name="Webs")
        g.labels.add(1, [chords[5]], "top2")
        g.labels.add(1, [chords[1]], "bot2")
        g.labels.add(1, [diag["d2"]], "diag2")
        g.labels.add(0, [b[0], b[4]], "supports")
        g.labels.add(0, [b[0]], "pin")
        g.labels.add(0, [b[4]], "roller")
        g.labels.add(0, [b[2]], "mid")
        g.mesh.generation.generate(dim=1)
        return g.mesh.queries.get_fem_data(dim=1)


@pytest.fixture(scope="module")
def pratt() -> Any:
    return build_pratt_fem()


def pratt_bridge(
    fem: Any, *, element_tags: str = "sequential", reverse: bool = False,
) -> apeSees:
    ops = apeSees(fem, default_orientation=None,
                  element_tags=element_tags)  # type: ignore[arg-type]
    ops.model(ndm=2, ndf=2)
    mat = ops.uniaxialMaterial.ElasticMaterial(E=E)
    order = ("Webs", "Chords") if reverse else ("Chords", "Webs")
    for pg in order:
        ops.element.Truss(pg=pg, A=A_BAR, material=mat)
    ops.fix(label="pin", dofs=(1, 1))
    ops.fix(label="roller", dofs=(0, 1))
    ts = ops.timeSeries.Linear()
    with ops.pattern.Plain(series=ts) as p:
        p.load(node=ops.nodes.one(label="mid"), forces=(0.0, -P))
    ops.constraints.Plain()
    ops.numberer.RCM()
    ops.system.BandGeneral()
    ops.test.NormDispIncr(tol=1e-10, max_iter=10)
    ops.algorithm.Linear()
    ops.integrator.LoadControl(dlam=1.0)
    ops.analysis.Static()
    return ops


#: Method of sections through panel 2 (between x = a and x = 2a), the left
#: part, with R = P/2 at the pin:
#: - moments about B2 (2a, 0): R*2a + F_top*H = 0
#: - moments about T1 (a, H):  R*a - F_bot*H = 0
#: - vertical forces:          R - F_diag*H/L = 0, L = sqrt(a^2 + H^2)
F_TOP = -(P / 2.0) * 2.0 * A_PANEL / H
F_BOT = (P / 2.0) * A_PANEL / H
F_DIAG = (P / 2.0) * math.hypot(A_PANEL, H) / H


def test_pratt_bar_forces_by_label_match_the_method_of_sections(
    pratt: Any,
) -> None:
    ops = pratt_bridge(pratt)
    assert ops.analyze(steps=1) == 0
    for label, expected in (("top2", F_TOP), ("bot2", F_BOT),
                            ("diag2", F_DIAG)):
        got = ops.elements.one(label=label).read("axial_force")
        assert got == pytest.approx(expected, rel=1e-9), label
        bars = ops.elements.get(label=label)
        assert bars.read("axial_force").shape == (1,)


def test_pratt_reactions_by_label_sum_to_the_load(pratt: Any) -> None:
    ops = pratt_bridge(pratt)
    ops.analyze(steps=1)
    supports = ops.nodes.get(label="supports")
    ry = supports.read("reaction_force_y")
    assert ry.shape == (2,)
    assert ry.sum() == pytest.approx(P, rel=1e-12)
    # Symmetry: each support carries half.
    assert ry == pytest.approx([P / 2.0, P / 2.0], rel=1e-9)
    # The roller carries no horizontal reaction, and nor does the pin
    # under a vertical load.
    assert supports.read("reaction_force_x") == pytest.approx(
        [0.0, 0.0], abs=1e-9 * P)
    # Midspan deflects down.
    assert ops.nodes.one(label="mid").read("displacement_y") < 0.0


def test_reaction_read_calls_reactions_first(pratt: Any) -> None:
    """Without ``reactions()`` openseespy answers a stale zero; the read
    calls it, so a first read after analyze is already right."""
    ops = pratt_bridge(pratt)
    ops.analyze(steps=1)
    pin = ops.nodes.one(label="pin")
    assert pin.read("reaction_force_y") == pytest.approx(P / 2.0, rel=1e-9)


def test_tag_map_default_and_fem_builds_read_equal_values(
    pratt: Any,
) -> None:
    """Invariant 3: with the webs declared first, sequential tags differ
    from FEM ids, and both builds read every bar by FEM id alike."""
    ids = None
    reads: dict[str, Any] = {}
    for mode in ("sequential", "fem"):
        ops = pratt_bridge(pratt, element_tags=mode, reverse=True)
        ops.analyze(steps=1)
        bars = ops.elements.get()
        ops_ids = bars.ids
        if ids is None:
            ids = ops_ids
        assert ops_ids == ids
        reads[mode] = bars.read("axial_force")
    assert reads["sequential"] == pytest.approx(reads["fem"], rel=1e-12)
    # The section bars, by label, are the closed forms in both builds.
    ops = pratt_bridge(pratt, element_tags="fem", reverse=True)
    ops.analyze(steps=1)
    assert ops.elements.one(label="diag2").read("axial_force") == (
        pytest.approx(F_DIAG, rel=1e-9))
    # Sequential tags really are not FEM ids here (the case is not vacuous).
    seq = pratt_bridge(pratt, reverse=True)
    seq.analyze(steps=1)
    session = seq._read_state
    plan = session.element_plan  # type: ignore[union-attr]
    assert any(
        int(eid) != tag
        for _spec, sub in plan for eid, _conn, tag in sub
    )


def test_a_read_allocates_no_tag_and_builds_nothing(
    pratt: Any, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Invariant 2: a spy on every allocator mint, the planner and the
    build raises if a read reaches one."""
    from apeGmsh.opensees._internal import tag_allocator, tag_plan

    ops = pratt_bridge(pratt)
    ops.analyze(steps=1)
    calls: list[str] = []

    def spy(name: str) -> Any:
        def _refuse(*_a: Any, **_k: Any) -> Any:
            calls.append(name)
            raise AssertionError(f"a read called {name}")
        return _refuse

    for attr in dir(tag_allocator.TagAllocator):
        if attr.startswith("allocate"):
            monkeypatch.setattr(tag_allocator.TagAllocator, attr, spy(attr))
    monkeypatch.setattr(tag_plan, "plan_tags", spy("plan_tags"))
    monkeypatch.setattr(apeSees, "build", spy("build"))
    ops.elements.get().read("axial_force")
    ops.elements.one(label="top2").read("axial_force")
    ops.nodes.get().read("displacement_x")
    ops.nodes.get(label="supports").read("reaction_force_y")
    assert calls == []


# ---------------------------------------------------------------------------
# The 2-D cantilever at ndf=3
# ---------------------------------------------------------------------------

L_BEAM = 3.0
I_BEAM = 8.0e-6
A_BEAM = 1.0e-2
P_TIP = 5.0e3


def build_cantilever_fem() -> Any:
    with apeGmsh(model_name="r2b_cantilever", verbose=False) as g:
        geo = g.model.geometry
        base = geo.add_point(0.0, 0.0, 0.0)
        tip = geo.add_point(L_BEAM, 0.0, 0.0)
        line = geo.add_line(base, tip)
        gmsh.model.mesh.setTransfiniteCurve(line, 5)
        g.physical.add(1, [line], name="Beam")
        g.labels.add(0, [base], "base")
        g.labels.add(0, [tip], "tip")
        g.mesh.generation.generate(dim=1)
        return g.mesh.queries.get_fem_data(dim=1)


@pytest.fixture(scope="module")
def cantilever() -> Any:
    return build_cantilever_fem()


def cantilever_bridge(fem: Any) -> apeSees:
    ops = apeSees(fem, default_orientation=None)
    ops.model(ndm=2, ndf=3)
    t = ops.geomTransf.Linear()
    ops.element.elasticBeamColumn(pg="Beam", transf=t, A=A_BEAM, E=E,
                                  Iz=I_BEAM)
    ops.fix(label="base", dofs=(1, 1, 1))
    ts = ops.timeSeries.Linear()
    with ops.pattern.Plain(series=ts) as p:
        p.load(node=ops.nodes.one(label="tip"), forces=(0.0, -P_TIP, 0.0))
    ops.constraints.Plain()
    ops.numberer.RCM()
    ops.system.BandGeneral()
    ops.test.NormDispIncr(tol=1e-10, max_iter=10)
    ops.algorithm.Linear()
    ops.integrator.LoadControl(dlam=1.0)
    ops.analysis.Static()
    return ops


def test_cantilever_tip_rotation_and_deflection_at_ndf3(
    cantilever: Any,
) -> None:
    ops = cantilever_bridge(cantilever)
    assert ops.analyze(steps=1) == 0
    tip = ops.nodes.one(label="tip")
    ei = E * I_BEAM
    assert tip.read("displacement_y") == pytest.approx(
        -P_TIP * L_BEAM ** 3 / (3.0 * ei), rel=1e-9)
    assert tip.read("rotation_z") == pytest.approx(
        -P_TIP * L_BEAM ** 2 / (2.0 * ei), rel=1e-9)
    base = ops.nodes.one(label="base")
    assert base.read("reaction_force_y") == pytest.approx(P_TIP, rel=1e-9)
    assert base.read("reaction_moment_z") == pytest.approx(
        P_TIP * L_BEAM, rel=1e-9)
    # Invariant 5: an ndf=3 planar node has no z translation, and no
    # rotation about x; the raise names the node and its ndf.
    for missing in ("displacement_z", "rotation_x"):
        with pytest.raises(BridgeError, match=rf"node {tip.tag} has no "
                           rf"'{missing}': it has ndf=3"):
            tip.read(missing)
    # A non-scalar element component raises and points to Results.
    with pytest.raises(BridgeError, match="Truss and CorotTruss"):
        ops.elements.get(pg="Beam").read("axial_force")


# ---------------------------------------------------------------------------
# Staleness (Invariant 4) and route refusals
# ---------------------------------------------------------------------------


def test_eigen_closes_the_session_and_names_it(pratt: Any) -> None:
    ops = pratt_bridge(pratt)
    ops.mass(nodes=tuple(int(n) for n in pratt.nodes.ids), values=(1.0, 1.0))
    ops.analyze(steps=1)
    mid = ops.nodes.one(label="mid")
    mid.read("displacement_y")
    assert ops.eigen(1).eigenvalues[0] > 0.0
    with pytest.raises(BridgeError, match=r"ops\.eigen\(\.\.\.\) wiped"):
        mid.read("displacement_y")


def test_a_second_analyze_replaces_the_session(pratt: Any) -> None:
    ops = pratt_bridge(pratt)
    ops.analyze(steps=1)
    mid = ops.nodes.one(label="mid")
    first = mid.read("displacement_y")
    ops.analyze(steps=1)   # a fresh build, a fresh solve: one step again
    assert mid.read("displacement_y") == pytest.approx(first, rel=1e-12)


def test_a_failed_second_analyze_leaves_a_closed_session(pratt: Any) -> None:
    ops = pratt_bridge(pratt)
    ops.analyze(steps=1)
    mid = ops.nodes.one(label="mid")
    with ops.stage(name="s") as s:   # live analyze refuses staged models
        s.analysis(
            test=ops.test.NormDispIncr(tol=1e-8, max_iter=5),
            algorithm=ops.algorithm.Linear(),
            integrator=ops.integrator.LoadControl(dlam=1.0),
            constraints=ops.constraints.Plain(),
            numberer=ops.numberer.RCM(),
            system=ops.system.BandGeneral(),
            analysis=ops.analysis.Static(),
        )
        s.run(n_increments=1)
    with pytest.raises(NotImplementedError):
        ops.analyze(steps=1)
    with pytest.raises(BridgeError, match="a later ops.analyze"):
        mid.read("displacement_y")


def test_run_closes_the_session(pratt: Any) -> None:
    ops = pratt_bridge(pratt)
    ops.analyze(steps=1)
    ops.run()
    with pytest.raises(BridgeError, match=r"ops\.run\(\) re-emitted"):
        ops.nodes.one(label="mid").read("displacement_y")


def test_a_second_bridge_makes_the_first_stale(pratt: Any) -> None:
    first = pratt_bridge(pratt)
    first.analyze(steps=1)
    mid = first.nodes.one(label="mid")
    mid.read("displacement_y")
    second = pratt_bridge(pratt)
    second.analyze(steps=1)
    with pytest.raises(BridgeError, match="domain was rebuilt"):
        mid.read("displacement_y")
    with pytest.raises(BridgeError, match="domain was rebuilt"):
        first.elements.one(label="top2").read("axial_force")
    # The second bridge answers.
    assert second.elements.one(label="top2").read("axial_force") == (
        pytest.approx(F_TOP, rel=1e-9))


# ---------------------------------------------------------------------------
# A u-p node (the R2-b review blocker on #1701), on a LadrunoUP fork build
# ---------------------------------------------------------------------------


def test_up_node_rotation_read_refuses_live_on_the_fork() -> None:
    """On the fork, a LadrunoUP plate at ndm=2, ndf=3: DOF 3 is pore
    pressure, so ``rotation_z`` refuses, while ``displacement_y`` reads
    the solver's DOF 2. Skips off a LadrunoUP-bearing build."""
    from tests.opensees.integration.test_ladruno_up_emission import (
        _skip_unless_fork_up,
    )
    from tests.opensees.unit.test_label_selector_fix_merge import (
        build_footing_fem,
    )

    _skip_unless_fork_up()
    fem = build_footing_fem()
    ops = apeSees(fem, default_orientation=None)
    ops.model(ndm=2, ndf=3)
    mat = ops.nDMaterial.ElasticIsotropic(E=1e4, nu=0.3, rho=0.0)
    ops.element.LadrunoUP(pg="Plate", material=mat, Kf=2.2e6, poro=0.4,
                          rhoF=1.0, perm=(1e-4, 1e-4), dynSeepage="off")
    ops.fix(label="bottom", dofs=(1, 1, 1))
    top = max(
        (int(n) for n in fem.nodes.ids),
        key=lambda t: tuple(fem.nodes.coords[fem.nodes.index(t)][[1, 0]]),
    )
    ts = ops.timeSeries.Constant()
    with ops.pattern.Plain(series=ts) as p:
        p.load(node=top, forces=(0.0, -1.0, 0.0))
    ops.constraints.Transformation()
    ops.numberer.RCM()
    ops.system.UmfPack()
    ops.test.NormDispIncr(tol=1e-8, max_iter=20)
    ops.algorithm.Newton()
    ops.integrator.Newmark(gamma=0.6, beta=0.3025)
    ops.analysis.Transient()
    try:
        assert ops.analyze(steps=1, dt=0.05) == 0
    except RuntimeError as e:  # pragma: no cover - stock build
        if "Ladruno fork" in str(e):
            pytest.skip(str(e))
        raise
    node = ops.nodes[top]
    with pytest.raises(BridgeError, match=r"has no established 'rotation_z'"
                       r".*LadrunoUP"):
        node.read("rotation_z")
    live = ops._read_state.emitter.ops  # type: ignore[union-attr]
    assert node.read("displacement_y") == live.nodeDisp(top, 2)
    assert node.read("displacement_y") < 0.0
