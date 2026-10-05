"""Live replay of #1324 / #1325: a staged quad plate through ``ops.py(run=True)``.

The reproduction from both issues, end to end: gmsh mesh with
physical groups, two named ``ops.stage`` blocks, the MPCO recorder,
``ops.h5`` for the sibling archive, then ``Results.from_mpco(path,
model_h5=...)`` with no ``fem=``.  Pins that the stage names and the
``pg=`` queries come from the archive, and that the values read by
program name equal the values read by the ``MODEL_STAGE[<k>]`` alias.
"""
from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pytest

pytestmark = pytest.mark.live

openseespy = pytest.importorskip(
    "openseespy.opensees", reason="openseespy required",
)


def _openseespy_has_mpco() -> bool:
    ops = openseespy
    ops.wipe()
    ops.model("basic", "-ndm", 1, "-ndf", 1)
    ops.node(1, 0.0)
    try:
        ops.recorder("mpco", "_probe_1324.mpco", "-N", "displacement")
        return True
    except Exception:
        return False
    finally:
        ops.wipe()
        Path("_probe_1324.mpco").unlink(missing_ok=True)


STAGE_NAMES = ("elastic_50pct", "plastic_100pct")


def _run_staged_plate(
    tmp_path: Path, *, size: float = 0.25, tag: str = "", run: bool = True,
) -> "tuple[Path, Path]":
    """Mesh the plate at ``size``, write ``model{tag}.h5``; run the deck if asked."""
    from apeGmsh import apeGmsh
    from apeGmsh.opensees import apeSees

    with apeGmsh(model_name=f"plate_1324{tag}", verbose=False) as g:
        g.model.geometry.add_rectangle(0, 0, 0, 2.0, 1.0, label="plate")
        g.physical.add_surface("plate", name="Plate")
        e = 1e-6
        g.model.select(None, dim=1).in_box(
            (-e, -e, -e), (e, 1 + e, e),
        ).to_physical("Left")
        g.model.select(None, dim=1).in_box(
            (2 - e, -e, -e), (2 + e, 1 + e, e),
        ).to_physical("Right")
        with g.loads.case("tension"):
            g.loads.line("Right", magnitude=1e5, direction=(1.0, 0, 0))
        g.mesh.sizing.set_global_size(size)
        g.mesh.structured.set_recombine("plate")
        g.mesh.generation.generate(dim=2)
        fem = g.mesh.queries.get_fem_data(dim=2)

    mpco = tmp_path / f"r{tag}.mpco"
    model_h5 = tmp_path / f"model{tag}.h5"
    ops = apeSees(fem)
    ops.model(ndm=2, ndf=2)
    m = ops.nDMaterial.ElasticIsotropic(E=200e9, nu=0.3)
    ops.element.FourNodeQuad(
        pg="Plate", thickness=0.01, material=m, plane_type="PlaneStress",
    )
    ops.fix(pg="Left", dofs=(1, 1))
    ops.recorder.MPCO(
        file=str(mpco), nodal_responses=("displacement",),
        elem_responses=("material.stress",),
    )
    for name in STAGE_NAMES:
        with ops.stage(name=name) as s:
            with s.pattern(series=ops.timeSeries.Linear(factor=0.5)) as p:
                p.from_model("tension")
            s.analysis(
                test=ops.test.NormDispIncr(tol=1e-8, max_iter=10),
                algorithm=ops.algorithm.Newton(),
                integrator=ops.integrator.LoadControl(dlam=0.5),
                constraints=ops.constraints.Plain(),
                numberer=ops.numberer.RCM(),
                system=ops.system.UmfPack(),
                analysis=ops.analysis.Static(),
            )
            s.run(n_increments=2, dt=0.5)
    ops.h5(str(model_h5))
    if run:
        ops.py(str(tmp_path / f"deck{tag}.py"), run=True)
    return mpco, model_h5


def test_from_mpco_binds_archive_stage_names_and_pgs(tmp_path: Path) -> None:
    if not _openseespy_has_mpco():
        pytest.skip("active openseespy build has no MPCO recorder")
    from apeGmsh.results import Results
    from apeGmsh.results._bind import (
        ModelFemMismatchWarning,
        StageCountMismatchWarning,
    )

    mpco, model_h5 = _run_staged_plate(tmp_path)
    with warnings.catch_warnings():
        warnings.simplefilter("error", ModelFemMismatchWarning)
        warnings.simplefilter("error", StageCountMismatchWarning)
        r = Results.from_mpco(mpco, model_h5=model_h5)
    with r:
        # #1324 — program names, with the file's names as aliases.
        assert [s.name for s in r.stages] == list(STAGE_NAMES)
        assert [s.aliases for s in r.stages] == [
            ("MODEL_STAGE[1]",), ("MODEL_STAGE[2]",),
        ]
        # #1325 — the archive's physical groups, from files alone.
        assert sorted(r.fem.nodes.physical.names()) == [
            "Left", "Plate", "Right",
        ]
        by_name = r.stage("plastic_100pct").nodes.get(
            pg="Right", component="displacement_x",
        )
        by_alias = r.stage("MODEL_STAGE[2]").nodes.get(
            pg="Right", component="displacement_x",
        )
        assert by_name.values.shape[0] == 2          # two increments
        right_nodes = {int(n) for n in r.fem.nodes.physical.node_ids("Right")}
        assert {int(n) for n in by_name.node_ids} == right_nodes
        np.testing.assert_array_equal(by_name.node_ids, by_alias.node_ids)
        np.testing.assert_allclose(by_name.values, by_alias.values)
        # Tension on the free edge: the whole edge moves +x.
        assert np.all(by_name.values[-1] > 0.0)


def test_from_mpco_refuses_a_finer_mesh_archive(tmp_path: Path) -> None:
    """The reviewer's reproducer for #1393: a run beside a finer mesh's archive.

    Both meshes number their nodes from 1, so the finer archive's ids
    cover the run's; only the coordinates tell them apart.  The bind
    must warn and fall back to the MPCO ``MODEL/`` geometry instead of
    answering ``pg="Right"`` with the finer mesh's edge.
    """
    if not _openseespy_has_mpco():
        pytest.skip("active openseespy build has no MPCO recorder")
    from apeGmsh.results import Results
    from apeGmsh.results._bind import ModelFemMismatchWarning
    from apeGmsh.results.readers._mpco import MPCOReader

    mpco, _coarse_h5 = _run_staged_plate(tmp_path, size=0.25, tag="A")
    _unused, fine_h5 = _run_staged_plate(
        tmp_path, size=0.1, tag="B", run=False,
    )
    with pytest.warns(ModelFemMismatchWarning, match="different mesh"):
        r = Results.from_mpco(mpco, model_h5=fine_h5)
    with r, MPCOReader(mpco) as raw:
        captured = raw.fem()
        assert captured is not None
        np.testing.assert_array_equal(r.fem.nodes.ids, captured.nodes.ids)
        np.testing.assert_allclose(r.fem.nodes.coords, captured.nodes.coords)
        assert r.fem.nodes.physical.names() == []
        assert [s.name for s in r.stages] == list(STAGE_NAMES)
