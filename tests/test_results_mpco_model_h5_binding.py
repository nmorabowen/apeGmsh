"""``Results.from_mpco(model_h5=...)`` binds what the archive carries (#1324, #1325).

An ``.mpco`` carries only ``MODEL_STAGE[<k>]`` groups and a bare
``MODEL/`` (no physical groups).  The sibling ``model.h5`` that
``from_mpco`` requires carries both the user's stage names
(``/opensees/stages/stage_NNN@name``) and the neutral FEMData with
``/physical_groups``.  These tests pin that ``from_mpco`` uses them:

* #1324 — program stage names map onto the capture stages when the
  counts agree (``MODEL_STAGE[k]`` stays reachable as an alias); a
  count mismatch warns and keeps the raw names; a vanilla model is
  silent.
* #1325 — without ``fem=``, the FEMData stored in ``model_h5`` is the
  bound fem when it covers the capture's node ids, so ``pg=`` queries
  work from files alone.  An explicit ``fem=`` keeps priority.  A
  ``model_h5`` whose fem does not cover the capture's nodes (the test
  suite's stub model) warns and falls back to the MPCO synthesis.

The ``.mpco`` here is synthetic (h5py); the ``model.h5`` is written by
the real bridge (``apeSees(...).h5``) from a real :class:`FEMData`.
"""
from __future__ import annotations

import warnings
from pathlib import Path

import h5py
import numpy as np
import pytest

from apeGmsh.mesh._element_types import ElementGroup, make_type_info
from apeGmsh.mesh._group_set import LabelSet, PhysicalGroupSet
from apeGmsh.mesh.FEMData import (
    ElementComposite,
    FEMData,
    MeshInfo,
    NodeComposite,
)
from apeGmsh.results import Results
from apeGmsh.results.readers._mpco_multi import MPCOMultiPartitionReader

from tests.conftest import _stub_model_h5_path

STAGE_NAMES = ("elastic_50pct", "plastic_100pct")
NODE_IDS = np.array([1, 2, 3, 4], dtype=np.int64)
COORDS = np.array(
    [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [2.0, 1.0, 0.0], [0.0, 1.0, 0.0]],
    dtype=np.float64,
)


# ---------------------------------------------------------------------------
# model.h5 — one quad, three physical groups, two named stages
# ---------------------------------------------------------------------------


def _plate_fem() -> FEMData:
    quad_info = make_type_info(
        code=3, gmsh_name="Quadrangle 4", dim=2, order=1, npe=4, count=1,
    )
    quad_group = ElementGroup(
        element_type=quad_info,
        ids=np.array([1], dtype=np.int64),
        connectivity=np.array([[1, 2, 3, 4]], dtype=np.int64),
    )
    pg = {
        (2, 1): {
            "name": "Plate",
            "node_ids": NODE_IDS,
            "node_coords": COORDS,
            "element_ids": np.array([1], dtype=np.int64),
        },
        (1, 2): {
            "name": "Left",
            "node_ids": np.array([1, 4], dtype=np.int64),
            "node_coords": COORDS[[0, 3]],
        },
        (1, 3): {
            "name": "Right",
            "node_ids": np.array([2, 3], dtype=np.int64),
            "node_coords": COORDS[[1, 2]],
        },
    }
    nodes = NodeComposite(
        node_ids=NODE_IDS, node_coords=COORDS,
        physical=PhysicalGroupSet(pg), labels=LabelSet({}),
    )
    elements = ElementComposite(
        groups={3: quad_group},
        physical=PhysicalGroupSet(pg), labels=LabelSet({}),
    )
    info = MeshInfo(n_nodes=4, n_elems=1, bandwidth=3, types=[quad_info])
    return FEMData(nodes=nodes, elements=elements, info=info)


def _write_model_h5(
    path: Path, *, stage_names: "tuple[str, ...]" = STAGE_NAMES,
) -> "tuple[Path, FEMData]":
    """Bridge-written ``model.h5``; ``stage_names=()`` gives a vanilla model."""
    from apeGmsh.opensees import apeSees

    fem = _plate_fem()
    ops = apeSees(fem, default_orientation=None)
    ops.model(ndm=2, ndf=2)
    mat = ops.nDMaterial.ElasticIsotropic(E=200e9, nu=0.3)
    ops.element.FourNodeQuad(
        pg="Plate", thickness=0.01, material=mat, plane_type="PlaneStress",
    )
    ops.fix(pg="Left", dofs=(1, 1))
    for name in stage_names:
        with ops.stage(name=name) as s:
            with s.pattern(series=ops.timeSeries.Linear(factor=0.5)) as p:
                p.load(pg="Right", forces=(1.0e3, 0.0))
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
    ops.h5(str(path))
    return path, fem


# ---------------------------------------------------------------------------
# .mpco — synthetic STKO layout, ``n_stages`` MODEL_STAGE groups
# ---------------------------------------------------------------------------


def _ux(stage_k: int, step: int, nid: int) -> float:
    """Closed-form value written to the file: identifies stage/step/node."""
    return 100.0 * stage_k + 10.0 * step + float(nid)


def _write_mpco(
    path: Path, *, n_stages: int,
    node_ids: np.ndarray = NODE_IDS, coords: np.ndarray = COORDS,
    n_steps: int = 2,
) -> Path:
    with h5py.File(path, "w") as f:
        info = f.create_group("INFO")
        info.create_dataset("SPATIAL_DIM", data=3)
        info.create_dataset("SOLVER_NAME", data=np.bytes_(b"OpenSees"))
        info.create_dataset("SOLVER_VERSION", data=np.array([3, 7, 1]))
        for k in range(1, n_stages + 1):
            stage = f.create_group(f"MODEL_STAGE[{k}]")
            stage.attrs["STEP"] = 0
            stage.attrs["TIME"] = 0.0
            model = stage.create_group("MODEL")
            nodes = model.create_group("NODES")
            nodes.create_dataset(
                "ID", data=node_ids.reshape(-1, 1).astype(np.int32),
            )
            nodes.create_dataset("COORDINATES", data=coords)
            model.create_group("ELEMENTS")
            results = stage.create_group("RESULTS")
            disp = results.create_group("ON_NODES").create_group(
                "DISPLACEMENT",
            )
            disp.attrs["DISPLAY_NAME"] = np.bytes_(b"Displacement")
            disp.attrs["COMPONENTS"] = np.array([np.bytes_(b"Ux,Uy,Uz")])
            disp.create_dataset(
                "ID", data=node_ids.reshape(-1, 1).astype(np.int32),
            )
            data = disp.create_group("DATA")
            for step in range(n_steps):
                vals = np.zeros((node_ids.size, 3), dtype=np.float64)
                vals[:, 0] = [_ux(k, step, int(n)) for n in node_ids]
                ds = data.create_dataset(f"STEP_{step}", data=vals)
                ds.attrs["STEP"] = step
                ds.attrs["TIME"] = 0.5 * (step + 1)
            results.create_group("ON_ELEMENTS")
    return path


@pytest.fixture
def staged_model_h5(tmp_path: Path) -> Path:
    path, _fem = _write_model_h5(tmp_path / "model.h5")
    return path


@pytest.fixture
def mpco_two_stages(tmp_path: Path) -> Path:
    return _write_mpco(tmp_path / "r.mpco", n_stages=2)


# ---------------------------------------------------------------------------
# #1324 — stage names
# ---------------------------------------------------------------------------


def test_stage_names_come_from_model_h5(
    mpco_two_stages: Path, staged_model_h5: Path,
) -> None:
    with Results.from_mpco(mpco_two_stages, model_h5=staged_model_h5) as r:
        assert [s.name for s in r.stages] == list(STAGE_NAMES)
        # Reader ids are untouched — scoped reads still key on them.
        assert [s.id for s in r.stages] == ["stage_0", "stage_1"]


def test_stage_lookup_by_program_name_reads_that_stage(
    mpco_two_stages: Path, staged_model_h5: Path,
) -> None:
    """``results.stage("plastic_100pct")`` reads MODEL_STAGE[2]'s data."""
    with Results.from_mpco(mpco_two_stages, model_h5=staged_model_h5) as r:
        slab = r.stage("plastic_100pct").nodes.get(
            component="displacement_x", ids=NODE_IDS,
        )
        expected = np.array(
            [[_ux(2, step, int(n)) for n in slab.node_ids] for step in (0, 1)],
        )
        np.testing.assert_allclose(slab.values, expected)


def test_model_stage_name_stays_an_alias(
    mpco_two_stages: Path, staged_model_h5: Path,
) -> None:
    """Backward compatibility: ``MODEL_STAGE[k]`` still resolves."""
    with Results.from_mpco(mpco_two_stages, model_h5=staged_model_h5) as r:
        by_alias = r.stage("MODEL_STAGE[1]")
        by_name = r.stage("elastic_50pct")
        assert by_alias.name == by_name.name == "elastic_50pct"
        assert r.stages[0].aliases == ("MODEL_STAGE[1]",)
        with pytest.raises(KeyError, match="No stage matches"):
            r.stage("MODEL_STAGE[3]")


def test_stage_count_mismatch_warns_and_keeps_raw_names(
    tmp_path: Path, staged_model_h5: Path,
) -> None:
    """Three capture stages, two program stages: no positional guess."""
    from apeGmsh.results._bind import StageCountMismatchWarning

    mpco = _write_mpco(tmp_path / "three.mpco", n_stages=3)
    with pytest.warns(StageCountMismatchWarning, match="2 .* 3 "):
        r = Results.from_mpco(mpco, model_h5=staged_model_h5)
    with r:
        assert [s.name for s in r.stages] == [
            "MODEL_STAGE[1]", "MODEL_STAGE[2]", "MODEL_STAGE[3]",
        ]
        assert all(s.aliases == () for s in r.stages)


def test_vanilla_model_h5_keeps_raw_names_silently(
    tmp_path: Path, mpco_two_stages: Path,
) -> None:
    """No ``/opensees/stages`` in the archive: nothing to map, no warning."""
    from apeGmsh.results._bind import StageCountMismatchWarning

    vanilla, _fem = _write_model_h5(tmp_path / "vanilla.h5", stage_names=())
    with warnings.catch_warnings():
        warnings.simplefilter("error", StageCountMismatchWarning)
        with Results.from_mpco(mpco_two_stages, model_h5=vanilla) as r:
            assert [s.name for s in r.stages] == [
                "MODEL_STAGE[1]", "MODEL_STAGE[2]",
            ]


# ---------------------------------------------------------------------------
# #1325 — physical groups
# ---------------------------------------------------------------------------


def test_fem_comes_from_model_h5_so_pg_queries_work(
    mpco_two_stages: Path, staged_model_h5: Path,
) -> None:
    with Results.from_mpco(mpco_two_stages, model_h5=staged_model_h5) as r:
        assert sorted(r.fem.nodes.physical.names()) == [
            "Left", "Plate", "Right",
        ]
        slab = r.stage("elastic_50pct").nodes.get(
            pg="Right", component="displacement_x",
        )
        assert sorted(int(n) for n in slab.node_ids) == [2, 3]
        assert slab.values.shape == (2, 2)
        expected = np.array(
            [[_ux(1, step, int(n)) for n in slab.node_ids] for step in (0, 1)],
        )
        np.testing.assert_allclose(slab.values, expected)


def test_explicit_fem_keeps_priority(
    mpco_two_stages: Path, staged_model_h5: Path,
) -> None:
    explicit = _plate_fem()
    with Results.from_mpco(
        mpco_two_stages, model_h5=staged_model_h5, fem=explicit,
    ) as r:
        assert r.fem is explicit


def test_unrelated_model_h5_warns_and_falls_back_to_mpco_synthesis(
    mpco_two_stages: Path,
) -> None:
    """The stub model (nodes 1-2) does not cover the capture (nodes 1-4)."""
    from apeGmsh.results._bind import ModelFemMismatchWarning

    with pytest.warns(ModelFemMismatchWarning, match="does not cover"):
        r = Results.from_mpco(mpco_two_stages, model_h5=_stub_model_h5_path())
    with r:
        assert sorted(int(n) for n in r.fem.nodes.ids) == [1, 2, 3, 4]
        assert r.fem.nodes.physical.names() == []


def test_unrelated_model_h5_with_explicit_fem_is_silent(
    mpco_two_stages: Path,
) -> None:
    """``fem=`` is the user's pairing: the gate does not second-guess it."""
    from apeGmsh.results._bind import ModelFemMismatchWarning

    explicit = _plate_fem()
    with warnings.catch_warnings():
        warnings.simplefilter("error", ModelFemMismatchWarning)
        with Results.from_mpco(
            mpco_two_stages, model_h5=_stub_model_h5_path(), fem=explicit,
        ) as r:
            assert r.fem is explicit


# ---------------------------------------------------------------------------
# Multi-partition facade — same contract over ``.part-N.mpco`` siblings
# ---------------------------------------------------------------------------


def test_multi_partition_binds_names_and_fem(
    tmp_path: Path, staged_model_h5: Path,
) -> None:
    """Node union {1,2,3} ∪ {2,3,4} is covered by the model's fem."""
    p0 = _write_mpco(
        tmp_path / "r.part-0.mpco", n_stages=2,
        node_ids=NODE_IDS[:3], coords=COORDS[:3],
    )
    _write_mpco(
        tmp_path / "r.part-1.mpco", n_stages=2,
        node_ids=NODE_IDS[1:], coords=COORDS[1:],
    )
    with Results.from_mpco(p0, model_h5=staged_model_h5) as r:
        assert isinstance(r._reader, MPCOMultiPartitionReader)
        assert [s.name for s in r.stages] == list(STAGE_NAMES)
        for child in r._reader._readers:
            assert [s.name for s in child.stages()] == list(STAGE_NAMES)
        slab = r.stage("plastic_100pct").nodes.get(
            pg="Right", component="displacement_x",
        )
        assert sorted(int(n) for n in slab.node_ids) == [2, 3]
