"""Fork lane: four element families read back from ``.ladruno`` and MPCO.

Gap G7 of the DomainCapture parity table (#1767, #1945). Four families were
proven only through ``DomainCapture``; each is solved here through the bridge
(``apeSees(fem)``, so element ids go through the fem -> ops translation),
recorded by the fork's own recorder and read back by the public factory, and
checked against the live domain at ``rtol=1e-10``.

Every family x route cell is pinned by one of three kinds of test, so a
reader change flips something:

* a passing value test;
* ``xfail(strict=True)`` citing a chain-B issue, asserting the right value,
  where the route reads back a silent wrong answer (none left: the #1971
  3-dof rotation 0.0 now reads NaN with ``model_h5=``);
* ``pytest.raises`` where the route refuses the family loudly (a gap: the
  MPCO truss bucket).

=================== =============================== ========================
Family              ``.ladruno``                    MPCO
=================== =============================== ========================
Truss axial force   line station: pass;             raises (gap)
                    Gauss record: pass
LadrunoBrick20      pass (values + GP coords)       pass (``stress``,
                                                    ``stresses``)
LadrunoLST          ``stress``: values + GP         pass (``stress``,
                    coords pass;                    ``stresses``)
                    ``stresses``: pass
LadrunoBrick +      (not covered here)              pass (damage and
ASDConcrete3D                                       equivalent plastic
material state                                      strain)
Shell-on-solid      pass; 3-dof rotation NaN        pass; 3-dof rotation
                    (#1971)                         NaN (#1971)
=================== =============================== ========================

Every MPCO recorder here records nodal displacement too, except the test
pinning that an element-only MPCO file still reads its steps (#1970).

Gated by the ``ladruno_fork`` marker, so the nightly ``live-fork`` lane runs it
against the pinned build. The stock fallback (``.out``) of the truss and
shell-on-solid cases is
``tests/opensees/integration/test_element_families_fallback_live.py``.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable

import h5py
import numpy as np
import pytest

from apeGmsh import apeGmsh
from apeGmsh.opensees import apeSees
from apeGmsh.opensees.emitter.live import LiveOpsEmitter
from apeGmsh.results import Results

from tests.opensees.integration_ladruno.test_mpco_modes_live import _fresh_env

pytestmark = pytest.mark.ladruno_fork

RTOL = 1e-10
P = 1000.0

#: Spelled out, not read from RESPONSE_CATALOG: the reader under test
#: takes its layout from the catalog, so deriving them would assert nothing.
STRESS_3D = ("stress_xx", "stress_yy", "stress_zz",
             "stress_xy", "stress_yz", "stress_xz")
STRESS_2D = ("stress_xx", "stress_yy", "stress_xy")
DISP = ("displacement_x", "displacement_y", "displacement_z")
ROT = ("rotation_x", "rotation_y", "rotation_z")
#: LadrunoLST's own 3-point interior rule, in the element's GP order
#: (LadrunoLST.cpp, the SixNodeTri anchors). The element answers no
#: ``integrationPoints`` response, so the fork source is the oracle. The
#: source lists (2/3, 1/6), (1/6, 2/3), (1/6, 1/6) node-0-first (L1 = s on
#: node 0); the reader hands them over in gmsh (u, v), the weights of nodes
#: 1 and 2 (#2228; the engine oracle is
#: tests/results/test_gauss_simplex_coords.py).
LST_GP = np.array([[1 / 6, 1 / 6], [2 / 3, 1 / 6], [1 / 6, 2 / 3]])

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _fem_to_ops(model_h5: str) -> dict[int, int]:
    """``{fem_eid: ops_tag}`` from the bridge's own ``element_meta``."""
    with h5py.File(model_h5, "r") as f:
        meta = f["/opensees/element_meta"]
        return {
            int(e): int(t)
            for tok in meta
            for t, e in zip(meta[tok]["ids"][...], meta[tok]["fem_eids"][...])
        }


def _nodes_on_plane(fem: Any, axis: int, value: float) -> list[int]:
    ids = np.asarray(fem.nodes.ids)
    xyz = np.asarray(fem.nodes.coords)
    return [int(n) for n, p in zip(ids, xyz) if abs(float(p[axis]) - value) < 1e-9]


def _record(ops: apeSees, kind: str, path: str, *, nodal: bool = True,
            **channels: Any) -> None:
    """Declare the fork's ``ladruno`` or the ``mpco`` recorder.

    The nodal channel is on by default; ``nodal=False`` leaves an
    element-only file, whose steps come from its element buckets (#1970).
    """
    if nodal:
        channels.setdefault("nodal_responses", ("displacement",))
    getattr(ops.recorder, kind)(file=path, **channels)


def _open(kind: str, path: str, fem: Any, model_h5: str) -> Results:
    factory: Callable[..., Results] = (
        Results.from_ladruno if kind == "Ladruno" else Results.from_mpco
    )
    return factory(path, fem=fem, model_h5=model_h5)


def _run(ops: apeSees, tmp_path: Path) -> tuple[Any, str]:
    """Persist the model and build it into the live fork domain."""
    model_h5 = str(tmp_path / "model.h5")
    ops.h5(model_h5)
    emitter = LiveOpsEmitter(wipe=True)
    ops.build().emit(emitter)
    return emitter, model_h5


# ---------------------------------------------------------------------------
# Truss: axial force
# ---------------------------------------------------------------------------


def _truss_fem(g: Any) -> Any:
    """One bar between two tagged points: its fem_eid (3) is not its tag."""
    G = g.model.geometry
    a = G.add_point(0.0, 0.0, 0.0)
    b = G.add_point(2.0, 0.0, 0.0)
    bar = G.add_line(a, b)
    g.model.sync()
    g.physical.add(1, [bar], name="Bar")
    g.physical.add(0, [a], name="A")
    g.physical.add(0, [b], name="B")
    g.mesh.structured.set_transfinite_curve(bar, 2)
    g.mesh.generation.generate(1)
    return g.mesh.queries.get_fem_data(dim=1)


def _truss_run(
    g: Any, tmp_path: Path, cls: str, kind: str,
) -> tuple[Results, int, list[float]]:
    """Pull the bar to ``P`` in two steps; return results, fem_eid, live N."""
    fem = _truss_fem(g)
    (bar_eid,) = [int(e) for e in fem.elements.physical.element_ids("Bar")]

    ops = apeSees(fem)
    ops.model(ndm=3, ndf=3)
    steel = ops.uniaxialMaterial.ElasticMaterial(E=200e9)
    getattr(ops.element, cls)(pg="Bar", A=1e-3, material=steel)
    ops.fix(pg="A", dofs=(1, 1, 1))
    ops.fix(pg="B", dofs=(0, 1, 1))
    with ops.pattern.Plain(series=ops.timeSeries.Linear()) as p:
        p.load(pg="B", forces=(P, 0.0, 0.0))
    path = str(tmp_path / ("truss.ladruno" if kind == "Ladruno" else "truss.mpco"))
    _record(ops, kind, path, elem_responses=("axialForce",))
    ops.constraints.Plain()
    ops.numberer.Plain()
    ops.system.BandGeneral()
    ops.test.NormDispIncr(tol=1e-10, max_iter=10)
    ops.algorithm.Newton()
    ops.integrator.LoadControl(dlam=0.5)
    ops.analysis.Static()

    emitter, model_h5 = _run(ops, tmp_path)
    tag = _fem_to_ops(model_h5)[bar_eid]
    assert tag != bar_eid          # the translation is exercised
    live = []
    for _ in range(2):
        assert emitter.analyze(steps=1) == 0
        live.append(float(emitter.ops.eleResponse(tag, "axialForce")[0]))
    emitter.ops.remove("recorders")
    np.testing.assert_allclose(live, [P / 2, P], rtol=RTOL)
    return _open(kind, path, fem, model_h5), bar_eid, live


@pytest.mark.parametrize("cls", ["Truss", "CorotTruss"])
def test_truss_axial_force_reads_back_from_ladruno(
    g: Any, tmp_path: Path, cls: str,
) -> None:
    """``.ladruno`` carries the truss force as a line station."""
    r, bar_eid, live = _truss_run(g, tmp_path, cls, "Ladruno")
    slab = r.elements.line_stations.get(component="axial_force")
    assert [int(e) for e in slab.element_index] == [bar_eid]
    np.testing.assert_allclose(np.asarray(slab.values)[:, 0], live, rtol=RTOL)


@pytest.mark.parametrize("cls", ["Truss", "CorotTruss"])
def test_truss_axial_force_gauss_record_from_ladruno(
    g: Any, tmp_path: Path, cls: str,
) -> None:
    """The Gauss record DomainCapture and ``.out`` give."""
    r, bar_eid, live = _truss_run(g, tmp_path, cls, "Ladruno")
    slab = r.elements.gauss.get(component="axial_force")
    assert [int(e) for e in slab.element_index] == [bar_eid]
    np.testing.assert_allclose(np.asarray(slab.values)[:, 0], live, rtol=RTOL)


@pytest.mark.parametrize("cls", ["Truss", "CorotTruss"])
def test_truss_axial_force_from_mpco_raises(
    g: Any, tmp_path: Path, cls: str,
) -> None:
    """Gap: MPCO cannot carry the truss force; the read refuses loudly."""
    r, _bar_eid, _live = _truss_run(g, tmp_path, cls, "MPCO")
    with pytest.raises(ValueError, match="GAUSS_IDS"):
        r.elements.gauss.get(component="axial_force")
    # Nor does it surface as a line station.
    assert np.asarray(
        r.elements.line_stations.get(component="axial_force").values
    ).size == 0


# ---------------------------------------------------------------------------
# LadrunoBrick20 and LadrunoLST: Gauss stress
# ---------------------------------------------------------------------------


def _solid_run(
    family: str, kind: str, token: str, tmp_path: Path, *, nodal: bool = True,
) -> dict[str, Any]:
    """Solve one fork-solid model, record ``token`` on ``kind``, read back."""
    with apeGmsh(model_name=f"g7_{family}", verbose=False) as g:
        if family == "h20":
            box = g.model.geometry.add_box(0, 0, 0, 1, 1, 2)
            g.model.sync()
            g.mesh.structured.set_transfinite_box(box, n=3)
            g.mesh.generation.generate(3)
            g.mesh.generation.set_order(2, bubble=False)
            g.physical.add(3, [box], name="Body")
            fem = g.mesh.queries.get_fem_data(dim=3)
        else:
            rect = g.model.geometry.add_rectangle(0, 0, 0, 4, 1)
            g.model.sync()
            g.mesh.sizing.set_global_size(0.5)
            g.mesh.generation.generate(2)
            g.mesh.generation.set_order(2, bubble=False)
            g.physical.add(2, [rect], name="Plate")
            fem = g.mesh.queries.get_fem_data(dim=2)

    ops = apeSees(fem)
    mat_args = dict(E=2.0e8, nu=0.25)
    if family == "h20":
        assert list(fem.elements)[0].element_type.npe == 20
        ops.model(ndm=3, ndf=3)
        mat = ops.nDMaterial.ElasticIsotropic(**mat_args)
        ops.element.LadrunoBrick20(pg="Body", material=mat, formulation="std")
        ops.fix(nodes=_nodes_on_plane(fem, 2, 0.0), dofs=(1, 1, 1))
        with ops.pattern.Plain(series=ops.timeSeries.Linear()) as p:
            # A lateral share too, so every stress component is non-trivial.
            for k, nid in enumerate(_nodes_on_plane(fem, 2, 2.0)):
                p.load(node=nid, forces=(1.0e2 * (k % 3), 50.0, -1.0e3))
        n_gp, n_comp = 27, 6
    else:
        assert list(fem.elements)[0].element_type.npe == 6
        ops.model(ndm=2, ndf=2)
        mat = ops.nDMaterial.ElasticIsotropic(**mat_args)
        ops.element.LadrunoLST(pg="Plate", material=mat, thickness=0.1,
                               plane_type="PlaneStress", geom="linear")
        ops.fix(nodes=_nodes_on_plane(fem, 0, 0.0), dofs=(1, 1))
        with ops.pattern.Plain(series=ops.timeSeries.Linear()) as p:
            for nid in _nodes_on_plane(fem, 0, 4.0):
                p.load(node=nid, forces=(30.0, -1.0e2))
        n_gp, n_comp = 3, 3
    ext = "ladruno" if kind == "Ladruno" else "mpco"
    path = str(tmp_path / f"{family}.{ext}")
    _record(ops, kind, path, nodal=nodal, elem_responses=(token,))
    ops.constraints.Plain()
    ops.numberer.RCM()
    ops.system.BandGeneral()
    ops.test.NormDispIncr(tol=1e-9, max_iter=10)
    ops.algorithm.Linear()
    ops.integrator.LoadControl(dlam=1.0)
    ops.analysis.Static()

    emitter, model_h5 = _run(ops, tmp_path)
    assert emitter.analyze(steps=1) == 0
    fem_to_ops = _fem_to_ops(model_h5)
    eids = [int(e) for grp in fem.elements for e in grp.ids]
    assert any(fem_to_ops[e] != e for e in eids)
    live = emitter.ops
    truth = {e: np.asarray(live.eleResponse(fem_to_ops[e], "stresses"),
                           dtype=float).reshape(n_gp, n_comp) for e in eids}
    if family == "h20":
        gp_xi = {e: np.asarray(live.eleResponse(fem_to_ops[e],
                                                "integrationPoints"),
                               dtype=float).reshape(27, 3) for e in eids}
    else:
        gp_xi = {e: LST_GP for e in eids}
    live.remove("recorders")
    return dict(r=_open(kind, path, fem, model_h5), eids=eids, truth=truth,
                gp_xi=gp_xi, fem_to_ops=fem_to_ops,
                comps=STRESS_3D if family == "h20" else STRESS_2D)


def _assert_gauss_stress(run: dict[str, Any], *, values: bool = True,
                         coords: bool = True) -> None:
    """Every GP value and/or GP coordinate of every element, at ``RTOL``."""
    r, eids, truth = run["r"], run["eids"], run["truth"]
    for k, comp in enumerate(run["comps"]):
        slab = r.elements.gauss.get(component=comp)
        index = np.asarray(slab.element_index)
        assert sorted(set(int(e) for e in index)) == sorted(eids), comp
        for eid in eids:
            cols = np.flatnonzero(index == eid)
            want = truth[eid]
            if values:
                np.testing.assert_allclose(
                    np.asarray(slab.values)[-1, cols], want[:, k],
                    rtol=RTOL, atol=RTOL * np.abs(want).max(),
                    err_msg=f"{comp} of element {eid} "
                            f"(tag {run['fem_to_ops'][eid]})",
                )
            if coords:
                # GP order, not just count: the read coordinates are the
                # element's own integration points.
                xi = np.asarray(slab.natural_coords)[cols]
                gp = run["gp_xi"][eid]
                np.testing.assert_allclose(
                    xi[:, :gp.shape[1]], gp, atol=1e-12,
                    err_msg=f"GP coords of {eid}",
                )


def test_ladruno_brick20_gauss_stress_reads_back_from_ladruno(
    tmp_path: Path,
) -> None:
    _assert_gauss_stress(_solid_run("h20", "Ladruno", "stress", tmp_path))


def test_ladruno_lst_gauss_stress_reads_back_from_ladruno(
    tmp_path: Path,
) -> None:
    _assert_gauss_stress(_solid_run("lst", "Ladruno", "stress", tmp_path),
                         coords=False)


def test_ladruno_lst_gauss_coords_read_back_from_ladruno(
    tmp_path: Path,
) -> None:
    """Each GP comes back at the element's own rule (no GP_PARAM in the file)."""
    _assert_gauss_stress(_solid_run("lst", "Ladruno", "stress", tmp_path),
                         values=False)


def test_ladruno_lst_stresses_token_reads_back_from_ladruno(
    tmp_path: Path,
) -> None:
    _assert_gauss_stress(_solid_run("lst", "Ladruno", "stresses", tmp_path))


@pytest.mark.parametrize("token", ["stress", "stresses"])
@pytest.mark.parametrize("family", ["h20", "lst"])
def test_fork_solid_gauss_stress_reads_back_from_mpco(
    tmp_path: Path, family: str, token: str,
) -> None:
    _assert_gauss_stress(_solid_run(family, "MPCO", token, tmp_path))


def test_element_only_mpco_file_has_its_steps(tmp_path: Path) -> None:
    run = _solid_run("h20", "MPCO", "stress", tmp_path, nodal=False)
    assert [s.n_steps for s in run["r"].stages] == [1]
    _assert_gauss_stress(run)       # the element reads have that step too


@pytest.mark.parametrize("token", [
    "material.equivalentPlasticStrain", "material.damage",
])
def test_ladruno_brick_material_state_reads_back_from_mpco(
    tmp_path: Path, token: str,
) -> None:
    """The material-state path resolves the rule-0 bracket too (#1970).

    LadrunoBrick + ASDConcrete3D shortened under displacement control:
    every GP's ``(+, -)`` pair against ``eleResponse(tag, "material", gp,
    key)``. The fixed base restrains the Poisson expansion, so four GPs
    crack in tension (d+ ~ 1, d- = 0) while every GP gathers compressive
    plastic strain: the two columns differ, so a swapped column fails.

    Displacement control, not load control (#2032): the tension crack
    drops the load, which LoadControl cannot follow at small steps. (The
    LoadControl divergence once blamed on the fork was test-order
    contamination, #2044; see the LoadControl tests below.)
    """
    du, steps = -1.0e-4, 20          # the top shortens 2e-3 (strain 1e-3)
    with apeGmsh(model_name="g7_mat", verbose=False) as g:
        box = g.model.geometry.add_box(0, 0, 0, 1, 1, 2)
        g.model.sync()
        g.mesh.structured.set_transfinite_box(box, n=2)
        g.mesh.generation.generate(3)
        g.physical.add(3, [box], name="Body")
        fem = g.mesh.queries.get_fem_data(dim=3)
    ops = apeSees(fem)
    ops.model(ndm=3, ndf=3)
    mat = ops.nDMaterial.ASDConcrete3D(E=30e6, v=0.2, fc=30e3, ft=3e3)
    ops.element.LadrunoBrick(pg="Body", material=mat)
    ops.fix(nodes=_nodes_on_plane(fem, 2, 0.0), dofs=(1, 1, 1))
    top = _nodes_on_plane(fem, 2, 2.0)
    with ops.pattern.Plain(series=ops.timeSeries.Linear()) as p:
        for nid in top:
            p.load(node=nid, forces=(0.0, 0.0, -1.0))
    path = str(tmp_path / "mat.mpco")
    _record(ops, "MPCO", path, elem_responses=(token,))
    ops.constraints.Plain()
    ops.numberer.RCM()
    ops.system.BandGeneral()
    ops.test.NormDispIncr(tol=1e-8, max_iter=50)
    ops.algorithm.Newton()
    ops.integrator.DisplacementControl(node=top[0], dof=3, dU=du)
    ops.analysis.Static()

    emitter, model_h5 = _run(ops, tmp_path)
    for step in range(steps):
        assert emitter.analyze(steps=1) == 0, f"step {step + 1}"
    # The controlled DOF followed the path (the right node was driven).
    np.testing.assert_allclose(
        emitter.ops.nodeDisp(top[0], 3), steps * du, rtol=1e-9)
    (eid,) = [int(e) for grp in fem.elements for e in grp.ids]
    tag = _fem_to_ops(model_h5)[eid]
    key = token.split(".", 1)[1]
    truth = np.array([
        np.asarray(emitter.ops.eleResponse(tag, "material", gp, key), float)
        for gp in range(1, 9)
    ])                                   # (8 GP, 2)
    emitter.ops.remove("recorders")
    # A non-zero oracle: four GPs cracked, and compressive plasticity.
    if key == "damage":
        assert truth[:, 0].max() > 0.5, truth
    else:
        assert truth[:, 0].max() > 0.0 and truth[:, 1].min() > 0.0, truth
    assert not np.allclose(truth[:, 0], truth[:, 1])   # a swap would show
    r = _open("MPCO", path, fem, model_h5)
    parent = ("equivalent_plastic_strain" if "Plastic" in token else "damage")
    for k, suffix in enumerate(("tension", "compression")):
        slab = r.elements.gauss.get(component=f"{parent}_{suffix}")
        assert [int(e) for e in slab.element_index] == [eid] * 8
        np.testing.assert_allclose(
            np.asarray(slab.values)[-1], truth[:, k], rtol=RTOL, atol=1e-14,
        )


def _concrete(E: float, fc: float, ft: float) -> Callable[[apeSees], Any]:
    return lambda ops: ops.nDMaterial.ASDConcrete3D(E=E, v=0.2, fc=fc, ft=ft)


def _load_control_brick(
    material: Callable[[apeSees], Any],
) -> tuple[apeSees, list[int]]:
    """The #1999 deck: one LadrunoBrick on a fixed base, -7.2e3 per top node
    in LoadControl steps of 0.25. Returns the bridge and the top nodes."""
    with apeGmsh(model_name="g7_lc", verbose=False) as g:
        box = g.model.geometry.add_box(0, 0, 0, 1, 1, 2)
        g.model.sync()
        g.mesh.structured.set_transfinite_box(box, n=2)
        g.mesh.generation.generate(3)
        g.physical.add(3, [box], name="Body")
        fem = g.mesh.queries.get_fem_data(dim=3)
    ops = apeSees(fem)
    ops.model(ndm=3, ndf=3)
    ops.element.LadrunoBrick(pg="Body", material=material(ops))
    ops.fix(nodes=_nodes_on_plane(fem, 2, 0.0), dofs=(1, 1, 1))
    top = _nodes_on_plane(fem, 2, 2.0)
    with ops.pattern.Plain(series=ops.timeSeries.Linear()) as p:
        for nid in top:
            p.load(node=nid, forces=(0.0, 0.0, -7.2e3))
    ops.constraints.Plain()
    ops.numberer.RCM()
    ops.system.BandGeneral()
    ops.test.NormDispIncr(tol=1e-8, max_iter=50)
    ops.algorithm.Newton()
    ops.integrator.LoadControl(dlam=0.25)
    ops.analysis.Static()
    return ops, top


def _fork_deck(ops: apeSees, path: Path) -> str:
    """The bridge's openseespy deck, on the fork module instead of pip's.

    The deck opens with ``ops.wipe()``, so decks concatenate into one
    process as a sequence of models.
    """
    ops.py(str(path), progress=False)
    text = path.read_text(encoding="utf-8")
    stock_import = "import openseespy.opensees as ops"
    assert stock_import in text
    return text.replace(
        stock_import,
        "from apeGmsh.opensees.emitter.live import get_ops\nops = get_ops()",
    )


def _run_fresh(tmp_path: Path, script: str) -> Any:
    """Run ``script`` in a fresh process and return the JSON it writes to
    ``OUT`` (the name is defined for it)."""
    out, run = tmp_path / "out.json", tmp_path / "run.py"
    run.write_text(f"OUT = {str(out)!r}\n{script}", encoding="utf-8")
    proc = subprocess.run(
        [sys.executable, str(run)], cwd=tmp_path, env=_fresh_env(),
        capture_output=True, text=True, check=False,
    )
    assert proc.returncode == 0, proc.stdout[-2000:] + proc.stderr[-2000:]
    return json.loads(out.read_text(encoding="utf-8"))


def test_ladruno_brick_asdconcrete_converges_under_load_control(
    tmp_path: Path,
) -> None:
    """LadrunoBrick + ASDConcrete3D under LoadControl, the #1999 case.

    The run converges in all four steps, and four GPs crack in tension
    (d+ ~ 1), so the passing case carries a non-zero answer.

    It runs in a fresh process (#2044). The divergence this test was once
    pinned to was not a fork regression: ASDConcrete3D kept a process-wide
    hardening law per material tag that ``wipe`` did not clear, so an earlier
    model in the same pytest process that reused tag 1 handed this one its E.
    A fresh process makes the case independent of test order. The cache
    itself is covered by the next test.
    """
    ops, _top = _load_control_brick(_concrete(30e6, 30e3, 3e3))
    result = _run_fresh(tmp_path, _fork_deck(ops, tmp_path / "lc.py") + (
        "\nimport json\n"
        "codes = [ops.analyze(1) for _ in range(4)]\n"
        "(tag,) = ops.getEleTags()\n"
        "d_plus = [float(ops.eleResponse(tag, 'material', gp, 'damage')[0])\n"
        "          for gp in range(1, 9)]\n"
        "open(OUT, 'w').write(json.dumps({'codes': codes, 'd_plus': d_plus}))\n"
    ))
    assert result["codes"] == [0, 0, 0, 0], result
    assert max(result["d_plus"]) > 0.5, result


def test_asdconcrete_hardening_law_does_not_survive_wipe(
    tmp_path: Path,
) -> None:
    """A wiped ASDConcrete3D does not hand its law to the next model (#2044).

    One fresh process runs two decks: an earlier model with ASDConcrete3D on
    the same tag (1) at a tenth of the stiffness and strength, which takes
    one step; then, after the second deck's ``wipe``, the #1999 brick. Its
    first step is elastic, so its top displacements equal the same mesh with
    ElasticIsotropic (E=30e6, v=0.2). Before nmorabowen/OpenSees#990
    (``ladruno-v20261015``) the first law stored for a tag survived ``wipe``:
    the brick ran on the earlier E and Newton diverged (the F-lane probe on
    #2044, Linux and Windows alike).

    The process is fresh because a broken build keeps the *first* law stored
    in the process: in the shared pytest process an earlier test's identical
    law would win, and this test would pass on a broken build.
    """
    ref, top = _load_control_brick(
        lambda ops: ops.nDMaterial.ElasticIsotropic(E=30e6, nu=0.2))
    emitter = LiveOpsEmitter(wipe=True)
    ref.build().emit(emitter)
    assert emitter.analyze(steps=1) == 0
    elastic = np.array([emitter.ops.nodeDisp(n, 3) for n in top])
    assert elastic.max() < -1e-4, elastic          # a non-zero oracle

    soft, _ = _load_control_brick(_concrete(3e6, 3e3, 3e2))
    brick, top2 = _load_control_brick(_concrete(30e6, 30e3, 3e3))
    assert top2 == top
    result = _run_fresh(tmp_path, (
        _fork_deck(soft, tmp_path / "soft.py")
        # The earlier model's own outcome is not under test; the step only
        # has to reach setTrialStrain, where the law is stored.
        + "\nops.analyze(1)\n"
        + _fork_deck(brick, tmp_path / "brick.py")
        + "\nimport json\n"
        "code = ops.analyze(1)\n"
        f"uz = [ops.nodeDisp(n, 3) for n in {top!r}]\n"
        "open(OUT, 'w').write(json.dumps({'code': code, 'uz': uz}))\n"
    ))
    assert result["code"] == 0, result
    np.testing.assert_allclose(result["uz"], elastic, rtol=1e-6)


# ---------------------------------------------------------------------------
# Mixed-ndf shell-on-solid: nodal round-trip (.ladruno and MPCO)
# ---------------------------------------------------------------------------


def _shell_on_solid_fem(g: Any) -> Any:
    """A tet footing (3-dof nodes) and a detached MITC4 wall (6-dof nodes)."""
    import gmsh

    G = g.model.geometry
    G.add_box(0, 0, -0.5, 1.0, 1.0, 0.5)
    p1 = G.add_point(0, 0, 0.5)
    p2 = G.add_point(1.0, 0, 0.5)
    p3 = G.add_point(1.0, 0, 1.5)
    p4 = G.add_point(0, 0, 1.5)
    l1 = G.add_line(p1, p2)
    l2 = G.add_line(p2, p3)
    l3 = G.add_line(p3, p4)
    l4 = G.add_line(p4, p1)
    wall = G.add_plane_surface([G.add_curve_loop([l1, l2, l3, l4])])
    g.model.sync()
    g.physical.add(3, [t for _d, t in gmsh.model.getEntities(3)], name="Footing")
    g.physical.add(2, [wall], name="Wall")
    sel = g.model.select
    sel(dim=2).on_plane((0, 0, -0.5), (0, 0, 1), tol=1e-4).to_physical("FootBase")
    sel(dim=2).on_plane((0, 0, 0.0), (0, 0, 1), tol=1e-4).to_physical("FootTop")
    g.physical.add(1, [l1], name="WallBase")
    g.physical.add(1, [l3], name="WallTop")
    g.mesh.structured.set_recombine("Wall", dim=2)
    g.mesh.sizing.set_global_size(0.34)
    g.mesh.generation.generate(3)
    g.mesh.partitioning.renumber(base=1)
    return g.mesh.queries.get_fem_data()


def _shell_on_solid_run(
    g: Any, tmp_path: Path, kind: str,
) -> tuple[Results, list[int], list[int], dict[int, np.ndarray]]:
    fem = _shell_on_solid_fem(g)
    solid = sorted({int(n) for n in fem.nodes.select(pg="Footing").ids})
    shell = sorted({int(n) for n in fem.nodes.select(pg="Wall").ids})
    assert solid and shell and not set(solid) & set(shell)

    ops = apeSees(fem)
    ops.model(ndm=3, ndf=6)
    ops.element.FourNodeTetrahedron(
        pg="Footing", material=ops.nDMaterial.ElasticIsotropic(E=30e9, nu=0.2),
    )
    ops.element.ShellMITC4(
        pg="Wall",
        section=ops.section.ElasticMembranePlateSection(E=30e9, nu=0.0, h=0.1),
    )
    ops.fix(pg="FootBase", dofs=(1, 1, 1))
    ops.fix(pg="WallBase", dofs=(1, 1, 1, 1, 1, 1))
    with ops.pattern.Plain(series=ops.timeSeries.Linear()) as pat:
        for n in fem.nodes.select(pg="WallTop").ids:
            pat.load(node=int(n), forces=(10.0, 50.0, -5.0, 1.0, 0.0, 0.0))
        for n in fem.nodes.select(pg="FootTop").ids:
            pat.load(node=int(n), forces=(1e5, -2e5, -3e5))
    ext = "ladruno" if kind == "Ladruno" else "mpco"
    path = str(tmp_path / f"sos.{ext}")
    _record(ops, kind, path, nodal_responses=("displacement", "rotation"))
    ops.constraints.Plain()
    ops.numberer.RCM()
    ops.system.UmfPack()
    ops.test.NormDispIncr(tol=1e-8, max_iter=50)
    ops.algorithm.Newton()
    ops.integrator.LoadControl(dlam=1.0)
    ops.analysis.Static()

    emitter, model_h5 = _run(ops, tmp_path)
    assert emitter.analyze(steps=1) == 0
    live = {n: np.asarray(emitter.ops.nodeDisp(n), dtype=float)
            for n in solid + shell}
    # The mixed-ndf premise: 3-dof solid nodes, 6-dof shell nodes.
    assert {live[n].size for n in solid} == {3}
    assert {live[n].size for n in shell} == {6}
    emitter.ops.remove("recorders")
    return _open(kind, path, fem, model_h5), solid, shell, live


@pytest.mark.parametrize("kind", ["Ladruno", "MPCO"])
def test_shell_on_solid_nodes_read_back(
    g: Any, tmp_path: Path, kind: str,
) -> None:
    r, solid, shell, live = _shell_on_solid_run(g, tmp_path, kind)
    for k, comp in enumerate(DISP + ROT):
        owners = solid + shell if k < 3 else shell
        slab = r.nodes.get(component=comp, ids=owners)
        got = {int(i): v for i, v in zip(slab.node_ids,
                                         np.asarray(slab.values)[-1])}
        assert sorted(got) == owners, comp
        want = np.array([live[n][k] for n in owners])
        scale = np.abs(want).max()
        assert scale > 0.0, comp
        np.testing.assert_allclose(
            [got[n] for n in owners], want, rtol=RTOL, atol=RTOL * scale,
            err_msg=f"{comp} on {kind}",
        )


@pytest.mark.parametrize("kind", ["Ladruno", "MPCO"])
def test_solid_node_rotations_are_absent(
    g: Any, tmp_path: Path, kind: str,
) -> None:
    """A tet node has no rotation DOF: NaN, never the recorder's 0.0.

    The file records no per-node ndf; ``model_h5=`` supplies it (#1971).
    """
    r, solid, _shell, _live = _shell_on_solid_run(g, tmp_path, kind)
    for comp in ROT:
        slab = r.nodes.get(component=comp)
        vals = np.asarray(slab.values)[-1]
        on_solid = [v for i, v in zip(slab.node_ids, vals) if int(i) in solid]
        assert not np.isfinite(on_solid).any(), (
            f"{comp}: {int(np.isfinite(on_solid).sum())} 3-dof nodes read a "
            "finite rotation"
        )
