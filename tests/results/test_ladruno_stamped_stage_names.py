"""#2823 (W4c): a ``.ladruno``'s stages bind by stamped name, never by position.

Audit #2824 found three producers whose files reopened under the wrong stage
name with ``Results.from_ladruno(model_h5=)``: nothing read
``INFO/PROVENANCE@stage_names``, ``_bind_stage_names`` paired the program's
first stages onto the file by position, and a modes-only (``KIND="eigen"``)
stage counted as a static host. W4a's ``.out`` transcode had the same
exposure. One reproducer per producer, each failing at the W4a merge head
d4d6f2e4a:

* ``capture_live_modes`` over a one-stage archive: the modes stage came back
  named ``pushover``, silently;
* ``ParallelModalResult.to_ladruno``, through the same ``write_modes``;
* a fork file stamped by ``stamp_fork_file`` whose recorder was bound to the
  program's second stage: it came back named ``gravity``;
* the ``.out`` transcode with ``stages=`` reordered against the program: a
  ``.ladruno`` written by the transcoder reopened with the names swapped.

Oracles: the values are closed forms that name their stage (``100 * stage +
10 * step + node``, the eigenvalues and shapes the fakes return), so a stage
read by its name either reads its own values or is not named at all. The
other contracts: an eigen-only stage is no static host, a stamp that
contradicts the program refuses by name, a prefix-plan transcode opens with
no warning, and a transcode whose binding fails is removed from the cache.
"""
from __future__ import annotations

import json
import shutil
import warnings
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import h5py
import numpy as np
import pytest

from apeGmsh.results import Results
from apeGmsh.results.readers._ladruno import StageStampRefused
from apeGmsh.results.readers._ladruno_live import stamp_fork_file
from apeGmsh.results.readers._ladruno_provenance import stamp_provenance

from tests.test_results_ladruno_stage_binding import (
    COORDS,
    NODE_IDS,
    _model_h5,
    _ux,
)

# =====================================================================
# Helpers
# =====================================================================


def _open(path: "Path | list[Path]", model_h5: Path) -> "tuple[Results, list[Any]]":
    """Open ``path`` beside ``model_h5``; return it and every warning raised."""
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        r = Results.from_ladruno(path, model_h5=model_h5)
    return r, list(rec)


def _hosts(r: Results) -> "list[tuple[str, str]]":
    return [(s.id, s.name) for s in r.stages if s.kind != "mode"]


def _stamp(path: Path) -> "list[str] | None":
    with h5py.File(path, "r") as f:
        raw = f["INFO/PROVENANCE"].attrs.get("stage_names")
    if raw is None:
        return None
    return json.loads(raw.decode() if isinstance(raw, bytes) else str(raw))


def _fork_file(path: Path, stage_ks: "list[int]", kinds: "list[str]",
               part: "tuple[int, int] | None" = None) -> Path:
    """A fork-recorder ``.ladruno`` of the truss mesh: ``MODEL_STAGE[i]``
    holds the closed-form values of program stage ``stage_ks[i - 1]``, with
    ``KIND`` ``kinds[i - 1]`` and an empty ``INFO/PROVENANCE`` to stamp."""
    with h5py.File(path, "w") as f:
        info = f.create_group("INFO")
        info.attrs["GENERATOR"] = np.bytes_(b"Ladruno")
        info.attrs["FORMAT_VERSION"] = 1
        info.attrs["SPATIAL_DIM"] = 2
        if part is not None:
            info.attrs["PARTITIONED"] = 1
            info.attrs["PARTITION_ID"] = part[0]
            info.attrs["NUM_PARTITIONS"] = part[1]
        info.create_group("PROVENANCE")
        for i, (k, kind) in enumerate(zip(stage_ks, kinds), start=1):
            stage = f.create_group(f"MODEL_STAGE[{i}]")
            stage.attrs["KIND"] = np.bytes_(kind.encode())
            nodes = stage.create_group("MODEL/NODES")
            nodes.create_dataset("ID", data=NODE_IDS.reshape(-1, 1).astype(np.int32))
            nodes.create_dataset("COORDINATES", data=COORDS[:, :2])
            stage.create_group("MODEL/ELEMENTS")
            disp = stage.create_group("RESULTS/ON_NODES/DISPLACEMENT")
            disp.attrs["COMPONENTS"] = np.array([b"Ux,Uy"])
            disp.create_dataset("ID", data=NODE_IDS.reshape(-1, 1).astype(np.int32))
            vals = np.zeros((2, 3, 2))
            for step in range(2):
                vals[step, :, 0] = [_ux(k, step, int(n)) for n in NODE_IDS]
            disp.create_dataset("DATA", data=vals)
            disp.create_dataset("TIME", data=np.array([0.5, 1.0]))
            disp.create_dataset("STEP", data=np.array([1, 2]))
    return path


def _ux_last(r: Results, stage: str) -> "list[float]":
    slab = r.stage(stage).nodes.get(component="displacement_x")
    return [float(v) for v in np.asarray(slab.values)[-1]]


# =====================================================================
# Producer 1: capture_live_modes
# =====================================================================


class _TrussOps:
    """The truss mesh as a live 2-D domain (ndf 2), with known modes."""

    LAMS = [16.0, 81.0]

    def getNDM(self) -> list[int]:
        return [2]

    def getNodeTags(self) -> list[int]:
        return [int(n) for n in NODE_IDS]

    def getNDF(self, n: int) -> list[int]:
        return [2]

    def nodeCoord(self, n: int) -> list[float]:
        return [float(c) for c in COORDS[int(n) - 1, :2]]

    def getEleTags(self) -> list[int]:
        return [1, 2]

    def eleType(self, tag: int) -> str:
        return "Truss"

    def eleNodes(self, tag: int) -> list[int]:
        return [tag, tag + 1]

    def eigen(self, *args: Any) -> list[float]:
        return list(self.LAMS[:int(args[-1])])

    def nodeEigenvector(self, n: int, k: int) -> list[float]:
        return [10.0 * k + n, -(10.0 * k + n)]


def test_capture_live_modes_is_not_named_after_the_program(tmp_path: Path) -> None:
    """#2824 reproduced: a one-stage archive labelled the modes stage
    ``pushover`` with no warning. The stage keeps its own name, silently,
    and the modes read back the solve's values."""
    from apeGmsh.results.live import capture_live_modes

    model_h5 = _model_h5(tmp_path / "model.h5", ("pushover", "static"))
    path = capture_live_modes(model_h5, 2, tmp_path / "modes.ladruno", ops=_TrussOps())

    assert _stamp(path) == ["MODEL_STAGE[1]"]
    r, raised = _open(path, model_h5)
    with r:
        assert raised == []
        assert _hosts(r) == [("stage_0", "MODEL_STAGE[1]")]
        assert "pushover" not in [s.name for s in r.stages]
        modes = sorted(r.modes, key=lambda m: m.mode_index)
        assert [m.eigenvalue for m in modes] == _TrussOps.LAMS
        ux = modes[1].nodes.get(component="displacement_x")
        assert ux.values[0].tolist() == [20.0 + n for n in NODE_IDS]


# =====================================================================
# Producer 2: ParallelModalResult.to_ladruno
# =====================================================================


def test_parallel_modal_result_is_not_named_after_the_program(tmp_path: Path) -> None:
    from apeGmsh.opensees.analysis.modal import ParallelModalResult

    model_h5 = _model_h5(tmp_path / "model.h5", ("pushover", "static"))
    job = tmp_path / "job"
    job.mkdir()
    lams = [25.0, 100.0]
    (job / "eigenvalues.out").write_text(" ".join(repr(v) for v in lams))
    (job / "mode_shapes.json").write_text(json.dumps(
        {"nodes": NODE_IDS.tolist(), "ndf": 2, "ndm": 2}))
    for k in (1, 2):
        (job / f"mode_shape_{k}.out").write_text(" ".join(
            repr(float(v)) for n in NODE_IDS for v in (100.0 * k + n, 0.5)))

    path = ParallelModalResult.from_job(job).to_ladruno(
        tmp_path / "modes.ladruno", model_h5=model_h5)

    assert _stamp(path) == ["MODEL_STAGE[1]"]
    r, raised = _open(path, model_h5)
    with r:
        assert raised == []
        assert _hosts(r) == [("stage_0", "MODEL_STAGE[1]")]
        modes = sorted(r.modes, key=lambda m: m.mode_index)
        assert [m.eigenvalue for m in modes] == lams
        ux = modes[0].nodes.get(component="displacement_x")
        assert ux.values[0].tolist() == [100.0 + n for n in NODE_IDS]


def test_an_unstamped_modes_file_is_no_static_host(tmp_path: Path) -> None:
    """A modes file written before the stamp named its stage (no
    ``stage_names``): its eigen-only stage is no evidence of a static
    program stage, so it keeps its name."""
    from apeGmsh.results.live import capture_live_modes

    model_h5 = _model_h5(tmp_path / "model.h5", ("pushover", "static"))
    path = capture_live_modes(model_h5, 1, tmp_path / "modes.ladruno", ops=_TrussOps())
    with h5py.File(path, "r+") as f:
        del f["INFO/PROVENANCE"].attrs["stage_names"]

    r, raised = _open(path, model_h5)
    with r:
        assert raised == []
        assert _hosts(r) == [("stage_0", "MODEL_STAGE[1]")]


# =====================================================================
# Producer 3: a fork file stamped by stamp_fork_file, recorder bound late
# =====================================================================


@pytest.mark.parametrize("parts", [1, 2])
def test_a_fork_file_of_the_second_stage_is_not_named_gravity(
    tmp_path: Path, parts: int,
) -> None:
    """The recorder was bound to ``pushover``, the program's second stage
    (both static): the file holds one stage and the stamp cannot name it.
    The reopen named it ``gravity``; it keeps its own name and the warning
    names it, and it reads pushover's values."""
    from apeGmsh.results.readers._ladruno_provenance import StageNamesOmittedWarning

    model_h5 = _model_h5(tmp_path / "model.h5",
                         ("gravity", "static"), ("pushover", "static"))
    if parts == 1:
        target = _fork_file(tmp_path / "run.ladruno", [2], ["static"])
    else:
        for i in range(parts):
            _fork_file(tmp_path / f"run.part-{i}.ladruno", [2], ["static"],
                       part=(i, parts))
        target = tmp_path / "run.part-0.ladruno"
    with pytest.warns(StageNamesOmittedWarning):
        assert stamp_fork_file(target, model_h5) is None
    assert _stamp(target) is None

    r, raised = _open(target, model_h5)
    with r:
        assert _hosts(r) == [("stage_0", "MODEL_STAGE[1]")]
        assert [w.category.__name__ for w in raised] == ["UnpairedStageWarning"]
        assert "MODEL_STAGE[1]" in str(raised[0].message)
        assert _ux_last(r, "MODEL_STAGE[1]") == [_ux(2, 1, n) for n in NODE_IDS]


@pytest.mark.parametrize("parts", [1, 2])
def test_a_stamp_naming_the_second_stage_binds_it(tmp_path: Path, parts: int) -> None:
    """The writer that knows the stage stamps it (``recorded_stages=``):
    the stage binds ``pushover`` by name, with no partial-run warning."""
    model_h5 = _model_h5(tmp_path / "model.h5",
                         ("gravity", "static"), ("pushover", "static"))
    if parts == 1:
        target = _fork_file(tmp_path / "run.ladruno", [2], ["static"])
    else:
        for i in range(parts):
            _fork_file(tmp_path / f"run.part-{i}.ladruno", [2], ["static"],
                       part=(i, parts))
        target = tmp_path / "run.part-0.ladruno"
    stamp_provenance(target, model_h5, recorded_stages=["pushover"])

    r, raised = _open(target, model_h5)
    with r:
        assert raised == []
        assert [(s.name, s.kind, s.aliases) for s in r.stages] == [
            ("pushover", "static", ("MODEL_STAGE[1]",))]
        assert _ux_last(r, "pushover") == [_ux(2, 1, n) for n in NODE_IDS]


# =====================================================================
# A stamp that contradicts the program refuses by name
# =====================================================================


def _stamped(tmp_path: Path, names: "list[str]", kinds: "list[str]") -> Path:
    path = _fork_file(tmp_path / "run.ladruno", list(range(1, len(names) + 1)), kinds)
    with h5py.File(path, "r+") as f:
        f["INFO/PROVENANCE"].attrs.create(
            "stage_names", data=json.dumps(names), dtype=h5py.string_dtype("utf-8"))
    return path


def test_a_stamped_name_the_program_does_not_declare_refuses(tmp_path: Path) -> None:
    model_h5 = _model_h5(tmp_path / "model.h5",
                         ("gravity", "static"), ("pushover", "static"))
    path = _stamped(tmp_path, ["gravity", "shake"], ["static", "static"])
    with pytest.raises(StageStampRefused, match=r"MODEL_STAGE\[2\] is stamped 'shake'"):
        Results.from_ladruno(path, model_h5=model_h5)


def test_a_stamped_stage_of_another_kind_refuses(tmp_path: Path) -> None:
    model_h5 = _model_h5(tmp_path / "model.h5",
                         ("gravity", "static"), ("shake", "transient"))
    path = _stamped(tmp_path, ["gravity", "shake"], ["static", "static"])
    with pytest.raises(StageStampRefused,
                       match=r"stamped 'shake' and records KIND 'static'"):
        Results.from_ladruno(path, model_h5=model_h5)


def test_a_reordered_stamp_binds_each_stage_by_name(tmp_path: Path) -> None:
    """Stage 1 holds shake, stage 2 gravity: the stamp says so, and each
    name reads its own values (a positional pairing would swap them)."""
    model_h5 = _model_h5(tmp_path / "model.h5",
                         ("gravity", "static"), ("shake", "transient"))
    path = _fork_file(tmp_path / "run.ladruno", [2, 1], ["transient", "static"])
    stamp_provenance(path, model_h5, recorded_stages=["shake", "gravity"])
    r, raised = _open(path, model_h5)
    with r:
        assert raised == []
        assert [(s.name, s.kind) for s in r.stages] == [
            ("shake", "transient"), ("gravity", "static")]
        assert _ux_last(r, "shake") == [_ux(2, 1, n) for n in NODE_IDS]
        assert _ux_last(r, "gravity") == [_ux(1, 1, n) for n in NODE_IDS]


def test_recorded_stages_refuse_a_name_the_program_does_not_declare(tmp_path: Path) -> None:
    model_h5 = _model_h5(tmp_path / "model.h5", ("gravity", "static"))
    path = _fork_file(tmp_path / "run.ladruno", [1], ["static"])
    with pytest.raises(ValueError, match=r"\['shake'\] are not stages"):
        stamp_provenance(path, model_h5, recorded_stages=["shake"])
    assert _stamp(path) is None


# =====================================================================
# Producer 4: the .out transcode with reordered stages=
# =====================================================================

_PROGRAM = (("load", "static"), ("base", "static"))


def _transcode_inputs(tmp_path: Path) -> "tuple[Any, Any, Any, Path]":
    from tests.test_results_out_multistage import _node_spec, _write_named_stages
    from tests.test_results_transcoder import staged_model

    fem, model = staged_model(tmp_path / "m", _PROGRAM)
    out = tmp_path / "out"
    _write_named_stages(out)
    return fem, model, _node_spec(fem), out


#: The ``_NAMED`` stages of ``test_results_out_multistage``, reordered
#: against ``_PROGRAM``: ``base`` (values ``200 + node + 0.5 k``), then
#: ``load`` (``node + 0.5 k``).
_REORDERED = (SimpleNamespace(name="base", kind="static"),
              SimpleNamespace(name="load", kind="static"))
_LOAD, _BASE = [1.5, 2.5], [202.5, 203.5]


def _last(r: Results) -> "dict[str, list[float]]":
    return {s.name: r.stage(s.id).nodes.get(component="displacement_x")
            .values[-1].tolist() for s in r.stages}


def test_a_reordered_transcode_never_reopens_under_swapped_names(tmp_path: Path) -> None:
    """The transcoder wrote the stages unnamed; the reopen paired ``load``
    with the ``base`` values. Each stage is stamped with its own name, so a
    reopen names no stage after the program."""
    from apeGmsh.results.transcoders import RecorderTranscoder

    fem, model, spec, out = _transcode_inputs(tmp_path)
    archive = tmp_path / "archive.model.h5"
    model.to_h5(archive)
    target = RecorderTranscoder(spec, out, tmp_path / "t.ladruno", fem,
                                model_h5_src=archive, stages=_REORDERED).run()

    r, raised = _open(target, archive)
    with r:
        assert raised == []
        assert _last(r) == {"MODEL_STAGE[1]": _BASE, "MODEL_STAGE[2]": _LOAD}


def test_a_reordered_plan_is_a_ladruno_bound_by_name(tmp_path: Path) -> None:
    """Goal 4: ``from_recorders`` caches the reordered plan as a stamped
    ``.ladruno`` (W4a kept it native), bound to the caller's model; it
    reads, and reopens, each name with its own values, with no warning."""
    from apeGmsh.results._open import open_results

    fem, model, spec, out = _transcode_inputs(tmp_path)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        r = Results.from_recorders(spec, out, fem=fem, model=model,
                                   cache_root=tmp_path / "c", stages=_REORDERED)
        with r:
            assert r.path is not None and r.path.suffix == ".ladruno"
            assert r.model is model
            assert _stamp(r.path) == ["base", "load"]
            assert _last(r) == {"base": _BASE, "load": _LOAD}
            assert r._unbound_stage_names(r._model_path) is None
            path, model_path = r.path, r._model_path
        with open_results(path, model_path) as again:
            assert _last(again) == {"base": _BASE, "load": _LOAD}


def test_a_prefix_plan_opens_without_a_count_warning(tmp_path: Path) -> None:
    """Card amendment 1: ``stage_id="load"`` of ``[load, base]`` warned
    ``StageCountMismatchWarning`` on every open; the stamp names the stage,
    so ``-W error`` opens it (and its cached copy)."""
    fem, model, spec, out = _transcode_inputs(tmp_path)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        for _ in range(2):     # the transcode, then the cached file
            with Results.from_recorders(spec, out, fem=fem, model=model,
                                        cache_root=tmp_path / "c", stage_id="load",
                                        stage_kind="static") as r:
                assert r.path is not None and r.path.suffix == ".ladruno"
                assert _last(r) == {"load": _LOAD}


def test_a_transcode_whose_binding_fails_leaves_the_cache(tmp_path: Path) -> None:
    """Card amendment 1: a cached ``.ladruno`` whose plan does not bind is
    removed when ``open_transcode`` refuses, so the next read transcodes
    again instead of refusing forever."""
    from apeGmsh.results.readers._ladruno_writer import open_transcode
    from apeGmsh.results.transcoders import RecorderTranscoder

    fem, model, spec, out = _transcode_inputs(tmp_path)
    archive = tmp_path / "archive.model.h5"
    model.to_h5(archive)
    target = RecorderTranscoder(spec, out, tmp_path / "t.ladruno", fem,
                                model_h5_src=archive, stages=_REORDERED).run()
    kept = tmp_path / "kept.ladruno"
    shutil.copy(target, kept)
    with pytest.raises(ValueError, match=r"\['erect'\] are not stages"):
        open_transcode(target, [("erect", "static", None), ("load", "static", None)],
                       fem=fem, model_h5=archive)
    assert not target.exists()
    assert _stamp(kept) == ["MODEL_STAGE[1]", "MODEL_STAGE[2]"]   # untouched copy
