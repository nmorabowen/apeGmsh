"""``Results.from_mpco(model_h5=...)`` binds what the archive carries (#1324, #1325).

An ``.mpco`` carries only ``MODEL_STAGE[<k>]`` groups and a bare
``MODEL/`` (no physical groups).  The sibling ``model.h5`` that
``from_mpco`` requires carries both the user's stage names
(``/opensees/stages/stage_NNN@name``) and the neutral FEMData with
``/physical_groups``.  These tests pin that ``from_mpco`` uses them:

* #1324 — program stage names map onto the capture stages in order
  (``MODEL_STAGE[k]`` stays reachable as an alias); a partial run
  (fewer capture stages) names the prefix and warns; more capture
  stages than program stages warns and keeps the raw names; a vanilla
  model is silent; duplicate program names warn.
* #1325 — without ``fem=``, the FEMData stored in ``model_h5`` is the
  bound fem when it covers the capture's node ids at the capture's
  coordinates, so ``pg=`` queries work from files alone.  An explicit
  ``fem=`` keeps priority.  A ``model_h5`` whose fem does not cover the
  capture's nodes (the test suite's stub model), or covers the ids at
  other coordinates (a finer mesh of the same part, #1393), warns and
  falls back to the MPCO synthesis.
* #1393 — ``results.stage(x)`` resolves by exact id before name before
  alias, so the ids the viewers hand back (``stage_<k>``) never land on
  a program stage that happens to be *named* ``stage_<k>``.

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


def _quad_plate(
    node_ids: np.ndarray, coords: np.ndarray, connectivity: np.ndarray,
    left: np.ndarray, right: np.ndarray,
) -> FEMData:
    """A quad mesh of the 2 x 1 plate with ``Plate`` / ``Left`` / ``Right``."""
    n_elems = int(connectivity.shape[0])
    quad_info = make_type_info(
        code=3, gmsh_name="Quadrangle 4", dim=2, order=1, npe=4,
        count=n_elems,
    )
    quad_group = ElementGroup(
        element_type=quad_info,
        ids=np.arange(1, n_elems + 1, dtype=np.int64),
        connectivity=connectivity,
    )
    index = {int(n): i for i, n in enumerate(node_ids)}

    def _at(ids: np.ndarray) -> np.ndarray:
        return coords[[index[int(n)] for n in ids]]

    pg = {
        (2, 1): {
            "name": "Plate",
            "node_ids": node_ids,
            "node_coords": coords,
            "element_ids": quad_group.ids,
        },
        (1, 2): {"name": "Left", "node_ids": left, "node_coords": _at(left)},
        (1, 3): {
            "name": "Right", "node_ids": right, "node_coords": _at(right),
        },
    }
    nodes = NodeComposite(
        node_ids=node_ids, node_coords=coords,
        physical=PhysicalGroupSet(pg), labels=LabelSet({}),
    )
    elements = ElementComposite(
        groups={3: quad_group},
        physical=PhysicalGroupSet(pg), labels=LabelSet({}),
    )
    info = MeshInfo(
        n_nodes=int(node_ids.size), n_elems=n_elems, bandwidth=3,
        types=[quad_info],
    )
    return FEMData(nodes=nodes, elements=elements, info=info)


def _plate_fem(z: float = 0.0) -> FEMData:
    """One quad: the mesh the synthetic ``.mpco`` was recorded from.

    ``z`` lifts the whole plate onto an offset plane: still a legal
    ``ndm=2`` model since #1346 (the dropped axis is uniform).
    """
    coords = COORDS.copy()
    coords[:, 2] = z
    return _quad_plate(
        NODE_IDS, coords, np.array([[1, 2, 3, 4]], dtype=np.int64),
        left=np.array([1, 4], dtype=np.int64),
        right=np.array([2, 3], dtype=np.int64),
    )


def _refined_plate_fem() -> FEMData:
    """The same plate meshed 2 x 2: nine nodes numbered 1..9 row by row.

    Its ids are a superset of the capture's ``1..4``, but ids 2 and 3
    sit at ``(1, 0)`` / ``(2, 0)`` instead of ``(2, 0)`` / ``(2, 1)``:
    an id-only coverage check pairs it, and ``Right`` would then be
    ``{3, 6, 9}`` (#1393).
    """
    ids = np.arange(1, 10, dtype=np.int64)
    xs = np.array([0.0, 1.0, 2.0])
    ys = np.array([0.0, 0.5, 1.0])
    coords = np.array(
        [[x, y, 0.0] for y in ys for x in xs], dtype=np.float64,
    )
    connectivity = np.array(
        [[1, 2, 5, 4], [2, 3, 6, 5], [4, 5, 8, 7], [5, 6, 9, 8]],
        dtype=np.int64,
    )
    return _quad_plate(
        ids, coords, connectivity,
        left=np.array([1, 4, 7], dtype=np.int64),
        right=np.array([3, 6, 9], dtype=np.int64),
    )


def _write_model_h5(
    path: Path, *, stage_names: "tuple[str, ...]" = STAGE_NAMES,
    fem: "FEMData | None" = None,
) -> "tuple[Path, FEMData]":
    """Bridge-written ``model.h5``; ``stage_names=()`` gives a vanilla model."""
    from apeGmsh.opensees import apeSees

    if fem is None:
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
    n_steps: int = 2, ndm: int = 3,
    elements: "list[tuple[int, list[int]]] | None" = None,
) -> Path:
    """``ndm=2`` stores two COORDINATES columns, as a 2-D deck's recorder does.

    ``elements`` are ``(ops_tag, node_tags)`` rows of one FourNodeQuad
    bucket, in the recorder's ``<class_tag>-<Class>[<rule>:<custom>]``
    layout; ``None`` leaves ``MODEL/ELEMENTS`` empty.
    """
    coords = np.asarray(coords, dtype=np.float64)[:, :ndm]
    with h5py.File(path, "w") as f:
        info = f.create_group("INFO")
        info.create_dataset("SPATIAL_DIM", data=ndm)
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
            elements_grp = model.create_group("ELEMENTS")
            if elements:
                elements_grp.create_dataset(
                    "3-FourNodeQuad[1:0]",
                    data=np.array(
                        [[tag, *conn] for tag, conn in elements],
                        dtype=np.int32,
                    ),
                )
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


def test_more_capture_stages_than_program_warns_and_keeps_raw_names(
    tmp_path: Path, staged_model_h5: Path,
) -> None:
    """Three capture stages, two program stages: no positional guess."""
    from apeGmsh.results._bind import StageCountMismatchWarning

    mpco = _write_mpco(tmp_path / "three.mpco", n_stages=3)
    with pytest.warns(
        StageCountMismatchWarning, match=r"2 .* 3 .*by id .*stage_<k>",
    ):
        r = Results.from_mpco(mpco, model_h5=staged_model_h5)
    with r:
        assert [s.name for s in r.stages] == [
            "MODEL_STAGE[1]", "MODEL_STAGE[2]", "MODEL_STAGE[3]",
        ]
        assert all(s.aliases == () for s in r.stages)


def test_partial_run_pairs_the_prefix_and_warns(
    tmp_path: Path, staged_model_h5: Path,
) -> None:
    """One capture stage, two program stages: the first name is paired.

    The bridge emits one ``domainChange`` per stage, so a run that
    stopped after stage one holds exactly ``MODEL_STAGE[1]``; its name
    is known, and the rest of the program has no capture (#1393).
    """
    from apeGmsh.results._bind import StageCountMismatchWarning

    mpco = _write_mpco(tmp_path / "one.mpco", n_stages=1)
    with pytest.warns(
        StageCountMismatchWarning,
        match=r"partial run.*\['plastic_100pct'\] have no capture",
    ):
        r = Results.from_mpco(mpco, model_h5=staged_model_h5)
    with r:
        assert [(s.id, s.name, s.aliases) for s in r.stages] == [
            ("stage_0", "elastic_50pct", ("MODEL_STAGE[1]",)),
        ]
        slab = r.stage("elastic_50pct").nodes.get(
            component="displacement_x", ids=NODE_IDS,
        )
        expected = np.array(
            [[_ux(1, step, int(n)) for n in slab.node_ids] for step in (0, 1)],
        )
        np.testing.assert_allclose(slab.values, expected)
        with pytest.raises(KeyError, match="No stage matches"):
            r.stage("plastic_100pct")


def test_duplicate_program_names_warn_and_ids_stay_unique(
    tmp_path: Path, mpco_two_stages: Path,
) -> None:
    """``ops.stage(name="load")`` twice: named lookup is ambiguous, ids are not."""
    from apeGmsh.results._bind import DuplicateStageNameWarning

    model_h5, _fem = _write_model_h5(
        tmp_path / "dup.h5", stage_names=("load", "load"),
    )
    with pytest.warns(DuplicateStageNameWarning, match=r"\['load'\]"):
        r = Results.from_mpco(mpco_two_stages, model_h5=model_h5)
    with r:
        assert [s.name for s in r.stages] == ["load", "load"]
        assert r.stage("load")._stage_id == "stage_0"
        assert r.stage("stage_1")._stage_id == "stage_1"
        assert r.stage("MODEL_STAGE[2]")._stage_id == "stage_1"


# ---------------------------------------------------------------------------
# #1393 — stage lookup order: exact id, then name, then alias
# ---------------------------------------------------------------------------


@pytest.fixture
def id_like_names_model_h5(tmp_path: Path) -> Path:
    """A program whose stage *names* collide with the reader's *ids*.

    Opening it warns ``ShadowedStageNameWarning`` by design
    (``test_program_name_that_matches_another_stage_id_warns``); the
    lookup tests are about where the lookup lands, so they ignore it.
    """
    path, _fem = _write_model_h5(
        tmp_path / "idlike.h5", stage_names=("stage_1", "stage_2"),
    )
    return path


def test_program_name_that_matches_another_stage_id_warns(
    mpco_two_stages: Path, tmp_path: Path,
) -> None:
    """``stage_1`` at id ``stage_0`` is unreachable by name: say so."""
    from apeGmsh.results._bind import ShadowedStageNameWarning

    model_h5, _fem = _write_model_h5(
        tmp_path / "idlike.h5", stage_names=("stage_1", "stage_2"),
    )
    with pytest.warns(
        ShadowedStageNameWarning, match=r"'stage_1' \(id 'stage_0'",
    ):
        r = Results.from_mpco(mpco_two_stages, model_h5=model_h5)
    r.close()
    # A name that equals its OWN id is not shadowed.
    own, _fem = _write_model_h5(
        tmp_path / "own.h5", stage_names=("stage_0", "stage_1"),
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error", ShadowedStageNameWarning)
        with Results.from_mpco(mpco_two_stages, model_h5=own) as r:
            assert [s.name for s in r.stages] == ["stage_0", "stage_1"]
    # A name like ANOTHER stage's MODEL_STAGE alias shadows that alias.
    alias, _fem = _write_model_h5(
        tmp_path / "alias.h5", stage_names=("MODEL_STAGE[2]", "second"),
    )
    with pytest.warns(
        ShadowedStageNameWarning,
        match=r"'MODEL_STAGE\[2\]' \(id 'stage_0'\) by the alias of stage 1",
    ):
        r = Results.from_mpco(mpco_two_stages, model_h5=alias)
    with r:
        # Names resolve before aliases: the name wins, as the warning says.
        assert r.stage("MODEL_STAGE[2]")._stage_id == "stage_0"
        assert r.stage("stage_1")._stage_id == "stage_1"


@pytest.mark.filterwarnings(
    "ignore::apeGmsh.results._bind.ShadowedStageNameWarning",
)
def test_stage_lookup_prefers_the_exact_id_over_a_name(
    mpco_two_stages: Path, id_like_names_model_h5: Path,
) -> None:
    """``stage("stage_1")`` is the stage with id ``stage_1`` (MODEL_STAGE[2]).

    Before #1393 the first match on id *or* name won, and the program
    stage named ``stage_1`` (id ``stage_0``) answered instead.
    """
    with Results.from_mpco(
        mpco_two_stages, model_h5=id_like_names_model_h5,
    ) as r:
        assert [(s.id, s.name) for s in r.stages] == [
            ("stage_0", "stage_1"), ("stage_1", "stage_2"),
        ]
        by_id = r.stage("stage_1")
        assert by_id._stage_id == "stage_1"
        assert by_id.name == "stage_2"
        slab = by_id.nodes.get(component="displacement_x", ids=NODE_IDS)
        expected = np.array(
            [[_ux(2, step, int(n)) for n in slab.node_ids] for step in (0, 1)],
        )
        np.testing.assert_allclose(slab.values, expected)
        # The name still resolves when no id claims it, and the alias too.
        assert r.stage("stage_2")._stage_id == "stage_1"
        assert r.stage("MODEL_STAGE[1]")._stage_id == "stage_0"


@pytest.mark.filterwarnings(
    "ignore::apeGmsh.results._bind.ShadowedStageNameWarning",
)
def test_viewer_id_lookup_reads_the_stage_it_names(
    mpco_two_stages: Path, id_like_names_model_h5: Path,
) -> None:
    """The viewers scope by ``StageInfo.id`` and must land on that stage.

    ``viewers/session/_realize.py`` scopes ``results.stage(stages[-1].id)``,
    ``viewers/diagrams/_director.py`` and ``viewers/session/_scrubber.py``
    scope ``results.stage(stage_id)`` with the id they stored: every
    ``StageInfo`` must round-trip through its own id, whatever the
    program named its stages.
    """
    with Results.from_mpco(
        mpco_two_stages, model_h5=id_like_names_model_h5,
    ) as r:
        for info in r.stages:
            scoped = r.stage(info.id)
            assert scoped._stage_id == info.id
            assert scoped.name == info.name
            assert scoped.n_steps == info.n_steps == 2
        last = r.stages[-1]
        slab = r.stage(last.id).nodes.get(
            component="displacement_x", ids=NODE_IDS,
        )
        np.testing.assert_allclose(
            slab.values[-1], [_ux(2, 1, int(n)) for n in slab.node_ids],
        )


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


def test_superset_mesh_model_h5_warns_and_falls_back_to_mpco_synthesis(
    tmp_path: Path, mpco_two_stages: Path,
) -> None:
    """A finer mesh of the same part covers the ids at other coordinates.

    Ids alone would pair it (1..4 is inside 1..9) and ``Right`` would
    come back as nodes ``{3, 6, 9}`` over geometry off by up to 1.0
    (#1393, the reviewer's 86-node run beside a 272-node archive).
    The coordinates at the capture's ids decide: warn, bind the MPCO
    synthesis (ADR 0021: warn, do not raise).
    """
    from apeGmsh.results._bind import ModelFemMismatchWarning

    refined, _fem = _write_model_h5(
        tmp_path / "refined.h5", fem=_refined_plate_fem(),
    )
    with pytest.warns(
        ModelFemMismatchWarning, match=r"different mesh.*differ by up to 1",
    ):
        r = Results.from_mpco(mpco_two_stages, model_h5=refined)
    with r:
        assert sorted(int(n) for n in r.fem.nodes.ids) == [1, 2, 3, 4]
        np.testing.assert_allclose(r.fem.nodes.coords, COORDS)
        assert r.fem.nodes.physical.names() == []
        # The archive's stage names still apply: the program ran this
        # capture even though its archived mesh is not this mesh.
        assert [s.name for s in r.stages] == list(STAGE_NAMES)


def test_offset_plane_2d_model_binds_without_a_warning(tmp_path: Path) -> None:
    """``ndm=2`` at ``z = 5``: the recorder stores ``x, y``; the archive keeps ``z``.

    The MPCO synthesis pads its two columns with ``z = 0`` while the
    archive's FEMData carries gmsh's ``z = 5``, so a comparison over
    every column reads a 5.0 mismatch on a correct pairing and drops the
    physical groups.  Only the model's ``ndm`` columns are compared
    (#1346 made the offset plane a supported ``ndm=2`` model).
    """
    from apeGmsh.results._bind import ModelFemMismatchWarning

    mpco = _write_mpco(tmp_path / "flat.mpco", n_stages=2, ndm=2)
    lifted, _fem = _write_model_h5(
        tmp_path / "lifted.h5", fem=_plate_fem(z=5.0),
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error", ModelFemMismatchWarning)
        r = Results.from_mpco(mpco, model_h5=lifted)
    with r:
        assert r.model.ndm == 2
        assert sorted(r.fem.nodes.physical.names()) == [
            "Left", "Plate", "Right",
        ]
        assert np.all(r.fem.nodes.coords[:, 2] == 5.0)
        slab = r.stage("elastic_50pct").nodes.get(
            pg="Right", component="displacement_x",
        )
        assert sorted(int(n) for n in slab.node_ids) == [2, 3]


def test_offset_plane_2d_model_still_refuses_an_in_plane_mismatch(
    tmp_path: Path,
) -> None:
    """Dropping ``z`` from the comparison must not drop ``x, y``."""
    from apeGmsh.results._bind import ModelFemMismatchWarning

    mpco = _write_mpco(tmp_path / "flat.mpco", n_stages=2, ndm=2)
    shifted = _plate_fem(z=5.0)
    shifted.nodes.coords[:, 0] += 0.5
    wrong, _fem = _write_model_h5(tmp_path / "shifted.h5", fem=shifted)
    with pytest.warns(ModelFemMismatchWarning, match="differ by up to 0.5"):
        r = Results.from_mpco(mpco, model_h5=wrong)
    with r:
        assert r.fem.nodes.physical.names() == []


# ---------------------------------------------------------------------------
# #1393 — the broker-only route (``fem.to_h5``: /meta/ndm = 0, no element_meta)
# ---------------------------------------------------------------------------


def test_broker_archive_with_renumbered_elements_warns_and_falls_back(
    tmp_path: Path,
) -> None:
    """The reviewer's b0: ops tags are read as fem element ids on this route.

    ``fem.to_h5`` carries no element tag map, and the bridge renumbers
    elements densely, so the capture's tag 7 is not the archive's
    element 1.  Binding the archive would answer ``gauss.get(pg=...)``
    with the wrong elements and no error; the bind must refuse it.
    """
    from apeGmsh.results._bind import ModelFemMismatchWarning

    broker = tmp_path / "broker.h5"
    _plate_fem().to_h5(str(broker))
    mpco = _write_mpco(
        tmp_path / "r.mpco", n_stages=1, elements=[(7, [1, 2, 3, 4])],
    )
    with pytest.warns(
        ModelFemMismatchWarning,
        match=r"no element tag map.*1 absent from its elements \(e\.g\. 7\)",
    ):
        r = Results.from_mpco(mpco, model_h5=broker)
    with r:
        assert r.model.ndm == 0
        assert r.fem.nodes.physical.names() == []
        assert [int(e) for e in r.fem.elements.ids] == [7]
        with pytest.raises(KeyError, match="No group named"):
            r.stage("stage_0").elements.gauss.get(
                pg="Plate", component="stress_xx",
            )


def test_broker_archive_with_matching_elements_binds_on_an_offset_plane(
    tmp_path: Path,
) -> None:
    """The reviewer's b5 with consistent elements: silent, PGs bound.

    ``/meta/ndm`` is 0 on a broker archive, so the capture's own column
    count (two) decides how many coordinates are compared; the padded
    ``z = 0`` against the archive's ``z = 5`` is not a mismatch.  The
    capture's element 1 on nodes 1-4 is the archive's element 1.
    """
    from apeGmsh.results._bind import ModelFemMismatchWarning

    broker = tmp_path / "broker.h5"
    _plate_fem(z=5.0).to_h5(str(broker))
    mpco = _write_mpco(
        tmp_path / "r.mpco", n_stages=1, ndm=2,
        elements=[(1, [1, 2, 3, 4])],
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error", ModelFemMismatchWarning)
        r = Results.from_mpco(mpco, model_h5=broker)
    with r:
        assert r.model.ndm == 0
        assert r.fem is r.model.fem
        assert sorted(r.fem.nodes.physical.names()) == [
            "Left", "Plate", "Right",
        ]
        assert [int(e) for e in r.fem.elements.physical.element_ids("Plate")] == [1]


def test_broker_archive_element_on_other_nodes_warns(tmp_path: Path) -> None:
    """Same element id, other nodes: still not the archive's element."""
    from apeGmsh.results._bind import ModelFemMismatchWarning

    broker = tmp_path / "broker.h5"
    _plate_fem().to_h5(str(broker))
    mpco = _write_mpco(
        tmp_path / "r.mpco", n_stages=1, elements=[(1, [1, 2, 3, 1])],
    )
    with pytest.warns(
        ModelFemMismatchWarning, match=r"1 on other nodes \(e\.g\. 1\)",
    ):
        r = Results.from_mpco(mpco, model_h5=broker)
    with r:
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
