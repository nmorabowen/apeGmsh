"""#2120 — ADR 0122 D8: the MPCO -> ``.ladruno`` conformance oracle.

One bridge model (a 2-D portal: a fiber ``forceBeamColumn`` column and an
``elasticBeamColumn`` beam, lumped masses) runs one static stage
(``gravity``) and then ``eigen``; the deck records an ``.mpco`` and a
``.ladruno`` in the same run, on the fork, in a fresh process.
``convert_mpco(.mpco)`` must then:

1. pass the fork's own ``ladruno_format.validate()`` (fetched at
   ``FORK_PIN``), returning ``[]``;
2. equal the fork's ``.ladruno`` under :func:`compare`, an
   information-equality comparator: the fork's ``normalize_nodal`` and
   ``normalize_element`` (values keyed by id, gauss point, section, fiber,
   component and step), the modal groups, ``NODES``, ``CONNECTIVITY``,
   ``SECTION_ASSIGNMENTS``, the result and bucket names (in order), the
   ``STEP`` / ``TIME`` axes and the BASIS of every group the converter
   describes; what it leaves out must equal the declared
   ``PROVENANCE@unsupported``;
3. fail :func:`compare` after either mutation, a swapped column or a
   dropped bucket, which ``validate()`` alone passes (it checks shapes).

Gated by the ``ladruno_fork`` marker (root conftest auto-skips off-fork).
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from types import ModuleType
from typing import Any, Iterator

import h5py
import numpy as np
import pytest

import apeGmsh
from apeGmsh.results import convert_mpco

from tests.opensees.integration_ladruno import _fork_spec
from tests.opensees.integration_ladruno._mpco_ladruno_compare import (
    attr as _attr,
    compare,
    stages as _stages,
)

pytestmark = pytest.mark.ladruno_fork

E, A, I_BEAM, MASS, P = 2.0e11, 0.02, 2.0e-4, 50.0, 1.0e4
N_MODES = 3
NODAL = ("displacement", "reactionForce", "modesOfVibration",
         "modesOfVibrationRotational")
ELEMENTS = ("force", "localForce", "basicForce", "section.force",
            "section.fiber.stress")
PROGRAM_KINDS = ["static"]  # the one program stage, gravity
#: The stage that holds the static steps and the modes.
_MODAL_STAGE = frozenset({"MODEL_STAGE[1]"})


def _portal_fem() -> Any:
    import gmsh

    from apeGmsh import apeGmsh as Session

    with Session(model_name="mpco_to_ladruno_portal", verbose=False) as g:
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


def _fresh_env() -> "dict[str, str]":
    src = str(Path(apeGmsh.__file__).resolve().parents[1])
    env = dict(os.environ, LADRUNO_OPENSEES_QUIET="1")
    env["PYTHONPATH"] = os.pathsep.join(
        [src, *filter(None, [os.environ.get("PYTHONPATH")])],
    )
    return env


@pytest.fixture(scope="module")
def run(tmp_path_factory: pytest.TempPathFactory) -> "dict[str, Path]":
    from apeGmsh.opensees.apesees import apeSees
    from apeGmsh.opensees.section.fiber import FiberPoint

    tmp = tmp_path_factory.mktemp("mpco_to_ladruno_oracle")
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
    with ops.stage(name="gravity") as s:
        with s.pattern(series=ops.timeSeries.Linear()) as p:
            p.load(node=ops.nodes.one(label="top"), forces=(P, -P, 0.0))
            p.load(node=ops.nodes.one(label="corner"), forces=(0.0, -P, 0.0))
        s.analysis(
            test=ops.test.NormDispIncr(tol=1e-12, max_iter=10),
            algorithm=ops.algorithm.Newton(),
            integrator=ops.integrator.LoadControl(dlam=0.5),
            constraints=ops.constraints.Plain(), numberer=ops.numberer.RCM(),
            system=ops.system.BandGeneral(), analysis=ops.analysis.Static(),
        )
        s.run(n_increments=2)
    ops.h5(str(model_h5))
    ops.py(str(deck), progress=False)

    text = deck.read_text(encoding="utf-8")
    stock_import = "import openseespy.opensees as ops"
    assert stock_import in text
    text = text.replace(
        stock_import,
        "from apeGmsh.opensees.emitter.live import get_ops\nops = get_ops()\n",
    )
    # The eigen solution after the stage, recorded once; then close the
    # recorders and prove the deck ran to its end.
    text += (f"\nops.eigen({N_MODES})\nops.record()\n"
             "ops.remove('recorders')\nprint('DECK-DONE')\n")
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
            "converted": converted[0]}


@pytest.fixture(scope="module")
def lf(run: "dict[str, Path]") -> Iterator[ModuleType]:
    spec = _fork_spec.fetch(run["tmp"] / "fork_spec")
    with _fork_spec.ladruno_format(spec) as module:
        yield module


# ---------------------------------------------------------------------------
# The oracle
# ---------------------------------------------------------------------------


def test_the_fork_and_mpco_stage_counts_agree(run) -> None:
    """Card stop condition: the two recorders split the run alike."""
    with h5py.File(run["fork"], "r") as a, h5py.File(run["mpco"], "r") as b:
        assert len(_stages(a)) == len(_stages(b)) == 1


def test_converted_file_passes_the_fork_validator(run, lf) -> None:
    assert lf.validate(str(run["converted"])) == []
    assert lf.validate(str(run["fork"])) == []  # the reference conforms too


def test_converted_file_equals_the_fork_file(run, lf) -> None:
    problems, lossy = compare(run["fork"], run["converted"], lf,
                              PROGRAM_KINDS, kind_checked_elsewhere=_MODAL_STAGE)
    assert problems == []
    with h5py.File(run["converted"], "r") as f:
        unsupported = json.loads(f["INFO/PROVENANCE"].attrs["unsupported"])
        assert f["INFO/PROVENANCE"].attrs["writer"].startswith(
            "apeGmsh mpco->ladruno ")
    # What the comparator had to leave out is exactly what was declared.
    assert set(unsupported) == lossy
    assert "quadrature:3-ElasticBeam2d[1:0]" in lossy  # the oracle bites


def _mutant(run: "dict[str, Path]", name: str) -> Path:
    out = run["tmp"] / f"mutant_{name}.ladruno"
    shutil.copy(run["converted"], out)
    return out


_BUCKET = "RESULTS/ON_ELEMENTS/localForce/3-ElasticBeam2d[1:0:0]"


def test_a_swapped_column_fails_the_comparator(run, lf) -> None:
    path = _mutant(run, "swap")
    with h5py.File(path, "r+") as f:
        data = f[f"{_stages(f)[0]}/{_BUCKET}/DATA"]
        values = data[...]
        # The first two columns that hold different values (N_1 and V_1 of
        # the loaded beam), so the swap changes the information.
        i, j = next((i, j) for i in range(values.shape[2])
                    for j in range(i + 1, values.shape[2])
                    if not np.allclose(values[..., i], values[..., j]))
        values[..., [i, j]] = values[..., [j, i]]
        data[...] = values
    assert lf.validate(str(path)) == []  # shapes only: validate() passes it
    problems, _ = compare(run["fork"], path, lf, PROGRAM_KINDS,
                          kind_checked_elsewhere=_MODAL_STAGE)
    assert any("element" in p for p in problems), problems


def test_a_dropped_bucket_fails_the_comparator(run, lf) -> None:
    path = _mutant(run, "drop")
    with h5py.File(path, "r+") as f:
        del f[f"{_stages(f)[0]}/{_BUCKET}"]
    assert lf.validate(str(path)) == []  # shapes only: validate() passes it
    problems, _ = compare(run["fork"], path, lf, PROGRAM_KINDS,
                          kind_checked_elsewhere=_MODAL_STAGE)
    assert any("buckets" in p for p in problems), problems
    assert any("keys differ" in p for p in problems), problems


def test_a_wrong_kind_fails_the_comparator(run, lf) -> None:
    path = _mutant(run, "kind")
    with h5py.File(path, "r+") as f:
        f[_stages(f)[0]].attrs["KIND"] = np.array([b"transient"])
    assert lf.validate(str(path)) == []  # a known word: validate() passes it
    problems, _ = compare(run["fork"], path, lf, PROGRAM_KINDS,
                          kind_checked_elsewhere=_MODAL_STAGE)
    assert any("@KIND" in p for p in problems), problems


def test_a_wrong_stage_time_fails_the_comparator(run, lf) -> None:
    path = _mutant(run, "time")
    with h5py.File(path, "r+") as f:
        f[_stages(f)[0]].attrs["TIME"] = np.array([99.0])
    problems, _ = compare(run["fork"], path, lf, PROGRAM_KINDS,
                          kind_checked_elsewhere=_MODAL_STAGE)
    assert any("@TIME" in p for p in problems), problems


@pytest.mark.xfail(
    strict=True, raises=AssertionError,
    reason="fork nmorabowen/OpenSees#2103: a modal channel stamps every "
           "stage KIND=eigen",
)
def test_modal_stage_kind_equals_the_fork_kind(run) -> None:
    """The gravity stage also holds the modes. The converter writes the
    program's ``static``; the fork labels the stage ``eigen`` (#2103)."""
    with h5py.File(run["fork"], "r") as a, h5py.File(run["converted"], "r") as b:
        (stage,) = _MODAL_STAGE
        ka, kb = _attr(a[stage], "KIND"), _attr(b[stage], "KIND")
    assert ka == kb, (
        f"{stage} (gravity + modes) KIND: fork {ka!r}, converted {kb!r} "
        f"(program {PROGRAM_KINDS[0]!r})")
