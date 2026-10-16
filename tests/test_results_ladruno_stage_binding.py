"""#2101: ``Results.from_ladruno(model_h5=)`` binds the program's stage names.

ADR 0122 design point 9: a ``.ladruno`` opened beside its ``model.h5``
takes the ``ops.stage(name=...)`` names and each stage's ``analysis``
kind from ``/opensees/stages``, as :meth:`Results.from_mpco` does
(#1324, #1948), through ``results/_bind.py::_bind_stage_names``.  The
file's ``MODEL_STAGE[<k>]`` name stays an alias.

Oracles:

* the committed fork capture ``tests/fixtures/ladruno/truss2d.ladruno``
  (and its two-part split) with a bridge ``model.h5`` naming its one
  static stage: the stage takes the name, and ``stage(<name>)`` reads
  the same slab as ``stage("MODEL_STAGE[1]")``;
* ``Results.from_mpco`` over the same stage layout, written as ``.mpco``:
  where the ``.ladruno`` records no contradicting ``KIND`` both bind the
  same names, kinds and aliases with the same warnings;
* #2079 Q7: a ``.ladruno`` stage records the ``KIND`` it ran, so where it
  disagrees with the program kind nothing is renamed, and
  ``StageKindMismatchWarning`` is the **one** warning.  Before #2101,
  ``_bind_stage_names`` first warned "the first N program names are
  paired" (or a duplicate / shadow warning) and the reader then said
  nothing was renamed: two warnings that contradict each other.
"""
from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any

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
from apeGmsh.results._bind import (
    DuplicateStageNameWarning,
    StageCountMismatchWarning,
)
from apeGmsh.results.readers._ladruno import StageKindMismatchWarning
from apeGmsh.results.readers._ladruno_multi import LadrunoMultiPartitionReader

FIXTURES = Path(__file__).parent / "fixtures" / "ladruno"
TRUSS = FIXTURES / "truss2d.ladruno"
TRUSS_PART0 = FIXTURES / "truss2d.part-0.ladruno"

NODE_IDS = np.array([1, 2, 3], dtype=np.int64)
COORDS = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
_FILE = ["MODEL_STAGE[1]", "MODEL_STAGE[2]"]


# ---------------------------------------------------------------------------
# model.h5 — the truss2d fixture's mesh (3 nodes, 2 trusses), named stages
# ---------------------------------------------------------------------------

def _truss_fem() -> FEMData:
    line = make_type_info(code=1, gmsh_name="Line 2", dim=1, order=1,
                          npe=2, count=2)
    group = ElementGroup(element_type=line,
                         ids=np.array([1, 2], dtype=np.int64),
                         connectivity=np.array([[1, 2], [2, 3]],
                                               dtype=np.int64))
    pg = {
        (1, 1): {"name": "Bars", "node_ids": NODE_IDS, "node_coords": COORDS,
                 "element_ids": group.ids},
        (0, 2): {"name": "Base", "node_ids": NODE_IDS[:1],
                 "node_coords": COORDS[:1]},
        (0, 3): {"name": "Tip", "node_ids": NODE_IDS[2:],
                 "node_coords": COORDS[2:]},
    }
    nodes = NodeComposite(node_ids=NODE_IDS, node_coords=COORDS,
                          physical=PhysicalGroupSet(pg), labels=LabelSet({}))
    elements = ElementComposite(groups={1: group},
                                physical=PhysicalGroupSet(pg),
                                labels=LabelSet({}))
    info = MeshInfo(n_nodes=3, n_elems=2, bandwidth=2, types=[line])
    return FEMData(nodes=nodes, elements=elements, info=info)


def _model_h5(path: Path, *stages: "tuple[str, str]") -> Path:
    """A bridge ``model.h5``; each stage is ``(name, "static"|"transient")``."""
    from apeGmsh.opensees import apeSees

    ops = apeSees(_truss_fem(), default_orientation=None)
    ops.model(ndm=2, ndf=2)
    mat = ops.uniaxialMaterial.ElasticMaterial(E=1000.0)
    ops.element.Truss(pg="Bars", A=1.0, material=mat)
    ops.fix(pg="Base", dofs=(1, 1))
    for name, kind in stages:
        with ops.stage(name=name) as s:
            with s.pattern(series=ops.timeSeries.Linear()) as p:
                p.load(pg="Tip", forces=(10.0, 0.0))
            if kind == "static":
                integrator = ops.integrator.LoadControl(dlam=0.25)
                analysis = ops.analysis.Static()
            else:
                integrator = ops.integrator.Newmark(gamma=0.5, beta=0.25)
                analysis = ops.analysis.Transient()
            s.analysis(
                test=ops.test.NormDispIncr(tol=1e-8, max_iter=10),
                algorithm=ops.algorithm.Linear(), integrator=integrator,
                constraints=ops.constraints.Plain(),
                numberer=ops.numberer.RCM(),
                system=ops.system.BandGeneral(), analysis=analysis,
            )
            if kind == "static":
                s.run(n_increments=2)
            else:
                s.run(n_increments=2, dt=0.5)
    ops.h5(str(path))
    return path


# ---------------------------------------------------------------------------
# A synthetic capture of the same mesh: one DISPLACEMENT per stage
# ---------------------------------------------------------------------------

def _ux(stage_k: int, step: int, nid: int) -> float:
    """Closed-form value written to the file: names its stage, step and node."""
    return 100.0 * stage_k + 10.0 * step + float(nid)


def _capture(
    path: Path, fmt: str, kinds: "list[str | None]",
    part: "tuple[int, int] | None" = None,
) -> Path:
    """``MODEL_STAGE[1..n]``; ``kinds[k]`` is stage k's ``.ladruno`` ``KIND``
    (``None`` writes none; an ``.mpco`` records none)."""
    with h5py.File(path, "w") as f:
        info = f.create_group("INFO")
        if fmt == "ladruno":
            info.attrs["GENERATOR"] = np.bytes_(b"Ladruno")
            info.attrs["FORMAT_VERSION"] = 1
            info.attrs["SPATIAL_DIM"] = 2
            if part is not None:
                info.attrs["PARTITIONED"] = 1
                info.attrs["PARTITION_ID"] = part[0]
                info.attrs["NUM_PARTITIONS"] = part[1]
        else:
            info.create_dataset("SPATIAL_DIM", data=2)
        for k, kind in enumerate(kinds, start=1):
            stage = f.create_group(f"MODEL_STAGE[{k}]")
            if fmt == "ladruno" and kind is not None:
                stage.attrs["KIND"] = np.bytes_(kind.encode())
            nodes = stage.create_group("MODEL/NODES")
            nodes.create_dataset("ID", data=NODE_IDS.reshape(-1, 1).astype(np.int32))
            nodes.create_dataset("COORDINATES", data=COORDS[:, :2])
            stage.create_group("MODEL/ELEMENTS")
            stage.create_group("RESULTS/ON_ELEMENTS")
            disp = stage.create_group("RESULTS/ON_NODES/DISPLACEMENT")
            disp.attrs["COMPONENTS"] = np.array([b"Ux,Uy"])
            disp.create_dataset("ID", data=NODE_IDS.reshape(-1, 1).astype(np.int32))
            vals = np.zeros((2, 3, 2))
            for step in range(2):
                vals[step, :, 0] = [_ux(k, step, int(n)) for n in NODE_IDS]
            if fmt == "ladruno":
                disp.create_dataset("DATA", data=vals)
                disp.create_dataset("TIME", data=np.array([0.5, 1.0]))
                disp.create_dataset("STEP", data=np.array([1, 2]))
            else:
                for step in range(2):
                    ds = disp.create_dataset(f"DATA/STEP_{step}", data=vals[step])
                    ds.attrs["TIME"] = np.array([0.5 * (step + 1)])
    return path


def _ladruno(tmp_path: Path, kinds: "list[str | None]", parts: int) -> Path:
    """One ``r.ladruno``, or ``parts`` siblings ``r.part-<i>.ladruno``."""
    if parts == 1:
        return _capture(tmp_path / "r.ladruno", "ladruno", kinds)
    for i in range(parts):
        _capture(tmp_path / f"r.part-{i}.ladruno", "ladruno", kinds,
                 part=(i, parts))
    return tmp_path / "r.part-0.ladruno"


def _hosts(r: Results) -> "list[tuple[str, str, str, tuple[str, ...]]]":
    return [(s.id, s.name, s.kind, s.aliases) for s in r.stages
            if s.kind != "mode"]


def _open(path: Path, model_h5: Path, fmt: str = "ladruno") -> "tuple[Results, list[Any]]":
    """Open ``path`` and return every warning the open raised."""
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        if fmt == "ladruno":
            r = Results.from_ladruno(path, model_h5=model_h5)
        else:
            r = Results.from_mpco(path, model_h5=model_h5)
    return r, [w.category for w in rec]


def _ux_last(r: Results, stage: str) -> "list[float]":
    slab = r.stage(stage).nodes.get(component="displacement_x")
    return [float(v) for v in np.asarray(slab.values)[-1]]


# =====================================================================
# The committed fork capture
# =====================================================================

@pytest.mark.parametrize("path", [TRUSS, TRUSS_PART0], ids=["file", "parts"])
def test_committed_capture_takes_the_program_name(
    tmp_path: Path, path: Path,
) -> None:
    model_h5 = _model_h5(tmp_path / "model.h5", ("pull", "static"))
    with Results.from_ladruno(path, model_h5=model_h5) as r:
        if path is TRUSS_PART0:
            assert isinstance(r._reader, LadrunoMultiPartitionReader)
        assert _hosts(r) == [("stage_0", "pull", "static", ("MODEL_STAGE[1]",))]
        assert _ux_last(r, "pull") == _ux_last(r, "MODEL_STAGE[1]")
        assert _ux_last(r, "pull")[-1] != 0.0       # the loaded tip moved


def test_without_model_h5_the_file_names_stand() -> None:
    with Results.from_ladruno(TRUSS) as r:
        assert _hosts(r) == [("stage_0", "MODEL_STAGE[1]", "static", ())]


# =====================================================================
# Static then transient: names and kinds come from the program
# =====================================================================

@pytest.mark.parametrize("parts", [1, 2])
def test_static_then_transient_bind_names_and_kinds(
    tmp_path: Path, parts: int,
) -> None:
    model_h5 = _model_h5(tmp_path / "model.h5",
                         ("gravity", "static"), ("shake", "transient"))
    path = _ladruno(tmp_path, ["static", "transient"], parts)
    r, raised = _open(path, model_h5)
    with r:
        assert raised == []
        assert _hosts(r) == [
            ("stage_0", "gravity", "static", ("MODEL_STAGE[1]",)),
            ("stage_1", "shake", "transient", ("MODEL_STAGE[2]",)),
        ]
        # Each name reads its own stage: the closed form names the stage.
        assert _ux_last(r, "shake") == [_ux(2, 1, n) for n in (1, 2, 3)]
        assert _ux_last(r, "gravity") == [_ux(1, 1, n) for n in (1, 2, 3)]


def test_same_layout_binds_as_from_mpco(tmp_path: Path) -> None:
    """Where the recorded KINDs agree, ``.ladruno`` and ``.mpco`` bind alike."""
    model_h5 = _model_h5(tmp_path / "model.h5",
                         ("gravity", "static"), ("shake", "transient"))
    lad, lad_raised = _open(
        _capture(tmp_path / "r.ladruno", "ladruno", ["static", "transient"]),
        model_h5)
    mp, mp_raised = _open(
        _capture(tmp_path / "r.mpco", "mpco", [None, None]), model_h5, "mpco")
    with lad, mp:
        assert _hosts(lad) == _hosts(mp)
        assert lad_raised == mp_raised == []


def test_eigen_kind_is_no_evidence(tmp_path: Path) -> None:
    """A recorder that asks for modes stamps every stage ``eigen`` (fork
    #2103): the program kind fills in and the names bind, silently."""
    model_h5 = _model_h5(tmp_path / "model.h5",
                         ("gravity", "static"), ("shake", "transient"))
    r, raised = _open(_ladruno(tmp_path, ["eigen", "eigen"], 1), model_h5)
    with r:
        assert raised == []
        assert [(n, k) for _, n, k, _ in _hosts(r)] == [
            ("gravity", "static"), ("shake", "transient")]


# =====================================================================
# #2079 Q7 / #2101: a KIND conflict renames nothing and warns once
# =====================================================================

@pytest.mark.parametrize("parts", [1, 2])
def test_misordered_program_warns_once_and_keeps_the_file_names(
    tmp_path: Path, parts: int,
) -> None:
    model_h5 = _model_h5(tmp_path / "model.h5",
                         ("shake", "transient"), ("gravity", "static"))
    r, raised = _open(_ladruno(tmp_path, ["static", "transient"], parts),
                      model_h5)
    with r:
        assert raised == [StageKindMismatchWarning]
        assert _hosts(r) == [
            ("stage_0", _FILE[0], "static", ()),
            ("stage_1", _FILE[1], "transient", ()),
        ]


@pytest.mark.parametrize("parts", [1, 2])
def test_partial_run_in_conflict_warns_once(tmp_path: Path, parts: int) -> None:
    """One capture stage recorded ``transient``; the program's first stage
    is static.  Before #2101 this warned StageCountMismatchWarning ("the
    first 1 program names are paired") and then StageKindMismatchWarning
    ("no stage is renamed")."""
    model_h5 = _model_h5(tmp_path / "model.h5",
                         ("gravity", "static"), ("shake", "transient"))
    r, raised = _open(_ladruno(tmp_path, ["transient"], parts), model_h5)
    with r:
        assert raised == [StageKindMismatchWarning]
        assert _hosts(r) == [("stage_0", _FILE[0], "transient", ())]


def test_duplicate_names_in_conflict_warn_once(tmp_path: Path) -> None:
    model_h5 = _model_h5(tmp_path / "model.h5",
                         ("load", "static"), ("load", "static"))
    r, raised = _open(_ladruno(tmp_path, ["static", "transient"], 1), model_h5)
    with r:
        assert raised == [StageKindMismatchWarning]
        assert [n for _, n, _, _ in _hosts(r)] == _FILE


def test_shadowing_name_in_conflict_warns_once(tmp_path: Path) -> None:
    """A program name like another stage's alias warns only on a rename."""
    model_h5 = _model_h5(tmp_path / "model.h5",
                         ("MODEL_STAGE[2]", "transient"), ("b", "static"))
    r, raised = _open(_ladruno(tmp_path, ["static", "transient"], 1), model_h5)
    with r:
        assert raised == [StageKindMismatchWarning]
        assert [n for _, n, _, _ in _hosts(r)] == _FILE


# =====================================================================
# Without a conflict the count and duplicate rules are from_mpco's
# =====================================================================

@pytest.mark.parametrize("fmt", ["ladruno", "mpco"])
def test_partial_run_without_conflict_pairs_the_prefix(
    tmp_path: Path, fmt: str,
) -> None:
    model_h5 = _model_h5(tmp_path / "model.h5",
                         ("gravity", "static"), ("shake", "transient"))
    path = _capture(tmp_path / f"r.{fmt}", fmt, ["static"])
    r, raised = _open(path, model_h5, fmt)
    with r:
        assert raised == [StageCountMismatchWarning]
        assert _hosts(r) == [("stage_0", "gravity", "static", (_FILE[0],))]


def test_mpco_pairs_where_a_ladruno_would_refuse(tmp_path: Path) -> None:
    """MPCO is unchanged: an ``.mpco`` records no KIND, so the misordered
    program of the conflict cases above renames its stages, silently."""
    model_h5 = _model_h5(tmp_path / "model.h5",
                         ("shake", "transient"), ("gravity", "static"))
    r, raised = _open(_capture(tmp_path / "r.mpco", "mpco", [None, None]),
                      model_h5, "mpco")
    with r:
        assert raised == []
        assert [(n, k) for _, n, k, _ in _hosts(r)] == [
            ("shake", "transient"), ("gravity", "static")]


def test_duplicate_names_without_conflict_warn(tmp_path: Path) -> None:
    model_h5 = _model_h5(tmp_path / "model.h5",
                         ("load", "static"), ("load", "static"))
    r, raised = _open(_ladruno(tmp_path, ["static", "static"], 1), model_h5)
    with r:
        assert raised == [DuplicateStageNameWarning]
        assert [n for _, n, _, _ in _hosts(r)] == ["load", "load"]
        assert r.stage("load")._stage_id == "stage_0"
