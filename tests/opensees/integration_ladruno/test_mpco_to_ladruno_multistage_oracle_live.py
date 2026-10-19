"""#2177 — ADR 0122 D8: the MPCO -> ``.ladruno`` oracle across stage boundaries.

Slice 6 (#2120) proved :func:`~apeGmsh.results.convert_mpco` on one
program stage plus modes. Stage boundaries are where the converter re-maps
structure (``MODEL_STAGE[k]``, the per-stage ``KIND``, each stage's time
origin, the stage-name binding in ``INFO/PROVENANCE@stage_names``), so this
oracle runs a three-stage program on the fork, in a fresh process, with an
``.mpco`` and a ``.ladruno`` recorded in the same run:

=========  =========  =====================================  =====
stage      KIND       chain                                  steps
=========  =========  =====================================  =====
gravity    static     ``LoadControl`` (dlam 0.5), loadConst  2
(modes)    eigen      ``ops.eigen(3)`` then one ``record()``  --
push       static     ``DisplacementControl`` on ``top`` x    3
shake      transient  ``Newmark``, ``dt`` 0.01               5
=========  =========  =====================================  =====

The converted file must pass the fork's ``validate()`` and equal the fork's
file under the slice-6 comparator (:mod:`._mpco_ladruno_compare`). Beside
the comparator, absolute oracles hold the program's own answer: each
stage's step count, its time origin (``loadConst -time 0`` restarts every
stage at zero, so ``gravity`` steps at the load factors and ``shake`` at
``k * dt``), its last time against ``ops.getTime()`` read in the deck, and
its name. Four mutations of the converted file (stages swapped, a stage
dropped, a wrong ``KIND`` on stage 2, a shifted time origin on stage 3)
each fail the comparator.

Gated by the ``ladruno_fork`` marker (root conftest auto-skips off-fork).
"""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path
from types import ModuleType
from typing import Any, Iterator

import h5py
import numpy as np
import pytest

from apeGmsh.results import convert_mpco

from tests.opensees.integration_ladruno import _fork_spec
from tests.opensees.integration_ladruno._mpco_ladruno_compare import (
    attr,
    compare,
    stages,
)
from tests.opensees.integration_ladruno.test_mpco_to_ladruno_oracle_live import (
    A,
    E,
    ELEMENTS,
    I_BEAM,
    MASS,
    N_MODES,
    NODAL,
    P,
    _fresh_env,
)

pytestmark = pytest.mark.ladruno_fork


def _portal_fem() -> Any:
    """Slice 6's portal, under a model name of its own (one artifact each)."""
    import gmsh

    from apeGmsh import apeGmsh as Session

    with Session(model_name="mpco_to_ladruno_multistage_portal",
                 verbose=False) as g:
        geo = g.model.geometry
        base = geo.add_point(0.0, 0.0, 0.0)
        top = geo.add_point(0.0, 3.0, 0.0)
        corner = geo.add_point(4.0, 3.0, 0.0)
        col = geo.add_line(base, top)
        beam = geo.add_line(top, corner)
        for line in (col, beam):
            gmsh.model.mesh.setTransfiniteCurve(line, 3)
        g.physical.add(1, [col], name="Col")
        g.physical.add(1, [beam], name="Beam")
        g.labels.add(0, [base], "base")
        g.labels.add(0, [top], "top")
        g.labels.add(0, [corner], "corner")
        g.mesh.generation.generate(dim=1)
        return g.mesh.queries.get_fem_data(dim=1)

N_GRAV, DLAM = 2, 0.5
N_PUSH, DU = 3, 1.0e-4
N_SHAKE, DT = 5, 0.01
#: The program: (name, KIND, steps), in run order.
PROGRAM = (("gravity", "static", N_GRAV), ("push", "static", N_PUSH),
           ("shake", "transient", N_SHAKE))
_LOAD_CONST = "ops.loadConst('-time', 0.0)\n"


def _chain(ops: Any, integrator: Any, analysis: Any) -> "dict[str, Any]":
    return dict(
        test=ops.test.NormDispIncr(tol=1e-12, max_iter=10),
        algorithm=ops.algorithm.Newton(), integrator=integrator,
        constraints=ops.constraints.Plain(), numberer=ops.numberer.RCM(),
        system=ops.system.BandGeneral(), analysis=analysis,
    )


@pytest.fixture(scope="module")
def run(tmp_path_factory: pytest.TempPathFactory) -> "dict[str, Any]":
    from apeGmsh.opensees.apesees import apeSees
    from apeGmsh.opensees.section.fiber import FiberPoint

    tmp = tmp_path_factory.mktemp("mpco_to_ladruno_multistage")
    mpco, fork = tmp / "r.mpco", tmp / "r.ladruno"
    model_h5, deck = tmp / "model.h5", tmp / "deck.py"
    fem = _portal_fem()
    ops = apeSees(fem, default_orientation=None)
    ops.model(ndm=2, ndf=3)
    t = ops.geomTransf.Linear()
    steel = ops.uniaxialMaterial.ElasticMaterial(E=E)
    sec = ops.section.Fiber(fibers=tuple(
        FiberPoint(material=steel, y=y, z=0.0, area=A / 4)
        for y in (-0.15, -0.05, 0.05, 0.15)))
    ops.element.forceBeamColumn(
        pg="Col", transf=t,
        integration=ops.beamIntegration.Lobatto(section=sec, n_ip=3))
    ops.element.elasticBeamColumn(pg="Beam", transf=t, A=A, E=E, Iz=I_BEAM)
    ops.fix(label="base", dofs=(1, 1, 1))
    ids = np.asarray(fem.nodes.ids, dtype=np.int64)
    xyz = np.asarray(fem.nodes.coords, dtype=np.float64)
    free = ids[np.linalg.norm(xyz, axis=1) > 0.0]  # every node but the base
    ops.mass(nodes=[int(n) for n in free], values=(MASS, MASS, 0.0))
    ops.recorder.MPCO(file=str(mpco), nodal_responses=NODAL,
                      elem_responses=ELEMENTS)
    ops.recorder.Ladruno(file=str(fork), nodal_responses=NODAL,
                         elem_responses=ELEMENTS)
    top = ops.nodes.one(label="top")
    corner = ops.nodes.one(label="corner")
    with ops.stage(name="gravity") as s:
        with s.pattern(series=ops.timeSeries.Linear()) as p:
            p.load(node=top, forces=(P, -P, 0.0))
            p.load(node=corner, forces=(0.0, -P, 0.0))
        s.analysis(**_chain(ops, ops.integrator.LoadControl(dlam=DLAM),
                            ops.analysis.Static()))
        s.run(n_increments=N_GRAV)
    with ops.stage(name="push") as s:
        with s.pattern(series=ops.timeSeries.Linear()) as p:
            p.load(node=top, forces=(1.0, 0.0, 0.0))
        s.analysis(**_chain(
            ops, ops.integrator.DisplacementControl(node=int(top), dof=1, dU=DU),
            ops.analysis.Static()))
        s.run(n_increments=N_PUSH)
    with ops.stage(name="shake") as s:
        with s.pattern(series=ops.timeSeries.Constant()) as p:
            p.load(node=corner, forces=(P, 0.0, 0.0))
        s.analysis(**_chain(ops, ops.integrator.Newmark(gamma=0.5, beta=0.25),
                            ops.analysis.Transient()))
        s.run(n_increments=N_SHAKE, dt=DT)
    ops.h5(str(model_h5))
    ops.py(str(deck), progress=False)

    text = deck.read_text(encoding="utf-8")
    stock_import = "import openseespy.opensees as ops"
    assert stock_import in text
    assert text.count(_LOAD_CONST) == len(PROGRAM), text.count(_LOAD_CONST)
    text = text.replace(
        stock_import,
        "from apeGmsh.opensees.emitter.live import get_ops\nops = get_ops()\n"
        "TIMES = []\n",
    )
    # Each stage's last time, read where the stage ends (before loadConst
    # restarts the clock); the eigen solution once, between gravity and
    # push.
    head, sep, rest = text.partition(_LOAD_CONST)
    rest = rest.replace(_LOAD_CONST, "TIMES.append(ops.getTime())\n" + _LOAD_CONST)
    text = (head + "TIMES.append(ops.getTime())\n" + sep
            + f"ops.eigen({N_MODES})\nops.record()\n" + rest)
    out = tmp / "times.json"
    text += ("\nimport json\nops.remove('recorders')\n"
             f"open({str(out)!r}, 'w').write(json.dumps(TIMES))\n"
             "print('DECK-DONE')\n")
    deck.write_text(text, encoding="utf-8")
    proc = subprocess.run(
        [sys.executable, str(deck)], cwd=tmp, env=_fresh_env(),
        capture_output=True, text=True, check=False,
    )
    assert proc.returncode == 0 and "DECK-DONE" in proc.stdout, (
        proc.stdout[-2000:] + proc.stderr[-2000:])
    converted = convert_mpco(mpco, tmp / "converted.ladruno",
                             model_h5=model_h5)
    assert converted == [tmp / "converted.ladruno"]
    return {"tmp": tmp, "mpco": mpco, "fork": fork, "model_h5": model_h5,
            "converted": converted[0],
            "times": json.loads(out.read_text(encoding="utf-8")),
            "top": int(top)}


@pytest.fixture(scope="module")
def lf(run: "dict[str, Any]") -> Iterator[ModuleType]:
    spec = _fork_spec.fetch(run["tmp"] / "fork_spec")
    with _fork_spec.ladruno_format(spec) as module:
        yield module


# ---------------------------------------------------------------------------
# The program's own answer
# ---------------------------------------------------------------------------

#: Every MODEL_STAGE of the run: gravity, push and shake, in order.
_ALL = ("MODEL_STAGE[1]", "MODEL_STAGE[2]", "MODEL_STAGE[3]")
#: The stage whose last record is the eigen solution. It is taken after the
#: gravity ``loadConst`` and before push adds its pattern, and only a domain
#: change opens a stage, so the modes ride in gravity's stage.
_MODAL_HOST = "MODEL_STAGE[1]"
_NAMES = [name for name, _, _ in PROGRAM]
_KINDS = [kind for _, kind, _ in PROGRAM]
#: Recorded steps per stage: gravity also holds the eigen ``record()``.
_COUNTS = [N_GRAV + 1, N_PUSH, N_SHAKE]


def _series(f: "h5py.File", stage: str) -> "list[h5py.Group]":
    """Every step series of a ``.ladruno`` stage: nodal results and buckets."""
    on_nodes = f"{stage}/RESULTS/ON_NODES"
    on_el = f"{stage}/RESULTS/ON_ELEMENTS"
    out = [f[on_nodes][n] for n in f[on_nodes]
           if not n.startswith("MODES_OF_VIBRATION")]
    out += [f[on_el][t][b] for t in f[on_el] for b in f[on_el][t]]
    assert out, f"{stage} holds no step series"
    return out


#: Each stage's TIME axis where it has a closed form. ``loadConst -time 0``
#: ends every stage, so each one starts its clock at zero: gravity steps at
#: the load factor ``k * DLAM`` and then holds the eigen ``record()`` at
#: time 0 (after its ``loadConst``); push steps at the
#: ``DisplacementControl`` load factor, which only the run knows (None);
#: shake at ``k * DT``.
_TIMES: "list[np.ndarray | None]" = [
    np.r_[DLAM * np.arange(1, N_GRAV + 1), 0.0], None,
    DT * np.arange(1, N_SHAKE + 1)]


def test_every_file_splits_the_run_into_the_program_stages(run) -> None:
    """All three files hold one MODEL_STAGE per program stage, and only the
    gravity stage holds modes."""
    for key in ("mpco", "fork", "converted"):
        with h5py.File(run[key], "r") as f:
            assert tuple(stages(f)) == _ALL, key
            for stage in _ALL:
                on = f"{stage}/RESULTS/ON_NODES"
                modal = [n for n in f[on] if n.startswith("MODES_OF_VIBRATION")]
                assert bool(modal) == (stage == _MODAL_HOST), (key, stage, modal)


def test_converted_file_passes_the_fork_validator(run, lf) -> None:
    assert lf.validate(str(run["converted"])) == []
    assert lf.validate(str(run["fork"])) == []  # the reference conforms too


def test_converted_file_equals_the_fork_file(run, lf) -> None:
    problems, lossy = compare(run["fork"], run["converted"], lf, _KINDS,
                              kind_checked_elsewhere=frozenset(_ALL))
    assert problems == []
    with h5py.File(run["converted"], "r") as f:
        prov = f["INFO/PROVENANCE"].attrs
        # What the comparator had to leave out is exactly what was declared.
        assert set(json.loads(prov["unsupported"])) == lossy
        assert json.loads(prov["stage_names"]) == _NAMES


@pytest.mark.parametrize("key", ["fork", "converted"])
def test_step_counts_and_times_are_the_programs(run, key: str) -> None:
    """Each stage's step count, STEP ids and TIME axis against the program,
    and each stage's last analysis time against ``ops.getTime()`` read in
    the deck where the stage ended."""
    first = np.cumsum([0, *_COUNTS[:-1]])
    with h5py.File(run[key], "r") as f:
        for k, stage in enumerate(_ALL):
            series = _series(f, stage)
            for grp in series:
                step, time = np.asarray(grp["STEP"]), np.asarray(grp["TIME"])
                where = f"{key} {grp.name}"
                np.testing.assert_array_equal(
                    step, first[k] + np.arange(_COUNTS[k]), err_msg=where)
                if _TIMES[k] is not None:
                    np.testing.assert_allclose(time, _TIMES[k], rtol=1e-12,
                                               atol=1e-15, err_msg=where)
                end = time[-2] if stage == _MODAL_HOST else time[-1]
                np.testing.assert_allclose(end, run["times"][k], rtol=1e-12,
                                           err_msg=where)
            assert int(attr(f[stage], "STEP")) == first[k], (key, stage)
            first_time = float(np.asarray(series[0]["TIME"])[0])
            assert float(attr(f[stage], "TIME")) == pytest.approx(
                first_time, rel=1e-12), (key, stage)


def test_push_is_the_displacement_controlled_stage(run) -> None:
    """In the converted file the ``push`` stage moves ``top`` by ``DU`` in
    x per step (the controlled DOF): that stage's data is push's."""
    from apeGmsh.results import Results

    res = Results.from_ladruno(str(run["converted"]),
                               model_h5=str(run["model_h5"]))
    slab = res.stage("push").nodes.get(component="displacement_x",
                                       ids=[run["top"]])
    assert [int(i) for i in np.asarray(slab.node_ids)] == [run["top"]]
    ux = np.asarray(slab.values)[:, 0]
    assert ux.shape == (N_PUSH,)
    np.testing.assert_allclose(np.diff(ux), DU, rtol=1e-9, atol=0.0)


@pytest.mark.parametrize("key", ["fork", "converted"])
def test_stages_take_the_program_names_and_kinds(run, key: str) -> None:
    from apeGmsh.results import Results

    res = Results.from_ladruno(str(run[key]), model_h5=str(run["model_h5"]))
    hosts = [s for s in res.stages if s.kind != "mode"]
    assert [s.name for s in hosts] == _NAMES
    assert [s.kind for s in hosts] == _KINDS


def test_mpco_stages_take_the_program_names_and_kinds(run) -> None:
    """The MPCO reader pairs the source's stages with the program alike."""
    from apeGmsh.results import Results

    res = Results.from_mpco(str(run["mpco"]), model_h5=str(run["model_h5"]))
    hosts = [s for s in res.stages if s.kind != "mode"]
    assert [s.name for s in hosts] == _NAMES
    assert [s.kind for s in hosts] == _KINDS


def _kinds(path: Path) -> "list[Any]":
    with h5py.File(path, "r") as f:
        return [attr(f[s], "KIND") for s in _ALL]


def test_converted_kinds_are_the_programs(run) -> None:
    """The converter writes the program's kind on each stage."""
    assert _kinds(run["converted"]) == _KINDS


@pytest.mark.xfail(
    strict=True, raises=AssertionError,
    reason="nmorabowen/apeGmsh#2103: a fork recorder with a modal channel "
           "stamps every stage KIND=eigen",
)
def test_every_stage_kind_equals_the_fork_kind(run) -> None:
    """The fork labels every stage of a recorder with a modal channel
    ``eigen`` (#2103). Only that exact shape is the expected failure: a
    full fork fix XPASSes (strict), and any other mismatch, a partial fix
    included, fails outright (``pytest.fail`` is not an AssertionError)."""
    fork, converted = _kinds(run["fork"]), _kinds(run["converted"])
    assert fork != ["eigen"] * len(_ALL), f"#2103 shape: fork {fork}"
    if fork != converted:
        pytest.fail(f"fork KIND {fork} is neither #2103's all-eigen shape "
                    f"nor the converted {converted}")


# ---------------------------------------------------------------------------
# Mutations: each one a converter that got a stage boundary wrong
# ---------------------------------------------------------------------------


def _mutant(run: "dict[str, Any]", name: str) -> Path:
    out = run["tmp"] / f"mutant_{name}.ladruno"
    shutil.copy(run["converted"], out)
    return out


def _compare(run: "dict[str, Any]", lf: ModuleType, path: Path) -> list[str]:
    problems, _ = compare(run["fork"], path, lf, _KINDS,
                          kind_checked_elsewhere=frozenset(_ALL))
    return problems


def test_swapped_stage_order_fails_the_comparator(run, lf) -> None:
    """push and shake swapped: stage 2 holds shake's transient steps."""
    path = _mutant(run, "swap")
    with h5py.File(path, "r+") as f:
        f.move("MODEL_STAGE[2]", "MODEL_STAGE[9]")
        f.move("MODEL_STAGE[3]", "MODEL_STAGE[2]")
        f.move("MODEL_STAGE[9]", "MODEL_STAGE[3]")
    problems = _compare(run, lf, path)
    assert ("MODEL_STAGE[2]|MODEL_STAGE[2] @KIND: converted 'transient', "
            "program 'static'") in problems, problems
    assert any(p.startswith("MODEL_STAGE[2]|MODEL_STAGE[2] @STEP")
               for p in problems), problems
    assert any(p.startswith("MODEL_STAGE[2]|MODEL_STAGE[2] nodal")
               for p in problems), problems


def test_dropped_stage_fails_the_comparator(run, lf) -> None:
    path = _mutant(run, "drop")
    with h5py.File(path, "r+") as f:
        del f["MODEL_STAGE[3]"]
    problems = _compare(run, lf, path)
    assert problems == ["stage count: fork 3, converted 2"], problems


def test_wrong_kind_on_stage_2_fails_the_comparator(run, lf) -> None:
    path = _mutant(run, "kind")
    with h5py.File(path, "r+") as f:
        f["MODEL_STAGE[2]"].attrs["KIND"] = np.array([b"transient"])
    assert lf.validate(str(path)) == []  # a known word: validate() passes it
    problems = _compare(run, lf, path)
    assert problems == [
        "MODEL_STAGE[2]|MODEL_STAGE[2] @KIND: converted 'transient', "
        "program 'static'"], problems


def test_shifted_time_origin_on_stage_3_fails_the_comparator(run, lf) -> None:
    """A converter that lost the ``loadConst -time 0`` restart: shake's
    clock continues from push's last time instead of from zero."""
    path = _mutant(run, "time")
    shift = run["times"][1]
    with h5py.File(path, "r+") as f:
        stage = f["MODEL_STAGE[3]"]
        stage.attrs["TIME"] = np.asarray(stage.attrs["TIME"]) + shift
        for grp in _series(f, "MODEL_STAGE[3]"):
            grp["TIME"][...] = np.asarray(grp["TIME"]) + shift
    problems = _compare(run, lf, path)
    assert any(p.startswith("MODEL_STAGE[3]|MODEL_STAGE[3] @TIME")
               for p in problems), problems
    assert any(p.endswith(" TIME: (5,) vs (5,) or values differ")
               for p in problems), problems
    assert all(p.startswith("MODEL_STAGE[3]|") for p in problems), problems
