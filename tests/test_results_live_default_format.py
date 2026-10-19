"""``spec.emit_default``: the default results route, against fake ops (#1950).

Each fake stands for one build and writes, when its recorder is removed,
the file that build's recorder would have written:

* **fork** (``ladrunoBuild`` returns a sha): a ``.ladruno`` converted from
  the committed springs capture;
* **stock with MPCO**: the springs ``.mpco`` itself;
* **pip** (``recorder('mpco', ...)`` raises): nothing for MPCO; the classic
  recorders it accepts are counted, and removed.

The oracle for the reads is the committed capture: every node value the
route returns equals ``Results.from_mpco`` of the same file. Warning counts
are exact (``record=True`` with ``"always"``). The pip read itself, on real
``.out`` files, is the live-stock test in the same file's ``live`` cases.
"""
from __future__ import annotations

import shutil
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from apeGmsh.opensees.opensees_model import OpenSeesModel
from apeGmsh.results import Results, convert_mpco
from apeGmsh.results.live import DefaultRouteWarning, LiveDefault
from apeGmsh.results.readers._ladruno_live import fork_recorder_args
from apeGmsh.results.spec._emit import mpco_ops_args
from apeGmsh.results.spec._resolved import (
    ResolvedRecorderRecord,
    ResolvedRecorderSpec,
)
from tests.test_results_mpco_to_ladruno import (
    SPRINGS,
    _bridge_copy,
    _unstaged_springs_model,
)

_SHA = "0123456789abcdef0123456789abcdef01234567"
_DISP = ("displacement_x", "displacement_y", "displacement_z")


@pytest.fixture(scope="module")
def files(tmp_path_factory: pytest.TempPathFactory) -> "dict[str, Path]":
    tmp = tmp_path_factory.mktemp("emit_default")
    model = _unstaged_springs_model(tmp / "model.h5", ["static"])
    other = _unstaged_springs_model(tmp / "other.h5", ["static", "static"])
    mpco = _bridge_copy(SPRINGS, model, tmp / "src.mpco")
    (fork,) = convert_mpco(mpco, tmp / "src.ladruno", model_h5=model)
    global _SNAPSHOT
    _SNAPSHOT = OpenSeesModel.from_h5(model).lineage.fem_hash
    return {"model": model, "other": other, "mpco": mpco, "fork": fork}


#: The springs FEMData's snapshot id (``model.h5``'s ``fem_hash``): the
#: classic-recorder route translates ids only through a matching model.
_SNAPSHOT = ""


def _spec() -> ResolvedRecorderSpec:
    return ResolvedRecorderSpec(
        fem_snapshot_id=_SNAPSHOT,
        records=(ResolvedRecorderRecord(
            category="nodes", name="all", components=_DISP,
            dt=None, n_steps=None, node_ids=np.array([1, 2, 3, 4]),
        ),),
    )


class FakeOps:
    """A build: ``recorder`` hands out tags, ``remove`` writes ``writes``."""

    def __init__(self, writes: "Path | None" = None, *, mpco: bool = True,
                 mpco_leaves: bool = False) -> None:
        self.writes = writes
        self.mpco = mpco
        self.mpco_leaves = mpco_leaves
        self.recorder_calls: list[tuple] = []
        self.removed: list[int] = []
        self.live: dict[int, tuple] = {}
        self._next = 1

    def recorder(self, *args: Any) -> int:
        self.recorder_calls.append(args)
        if args[0] == "mpco" and not self.mpco:
            if self.mpco_leaves:
                Path(args[1]).write_bytes(b"partial")
            raise RuntimeError("recorder mpco: unknown recorder type")
        tag, self._next = self._next, self._next + 1
        self.live[tag] = args
        return tag

    def remove(self, what: str, tag: int) -> None:
        assert what == "recorder"
        args = self.live.pop(tag)
        self.removed.append(tag)
        if self.writes is not None and args[0] != "Node":
            shutil.copy(self.writes, args[1])


class ForkOps(FakeOps):
    def ladrunoBuild(self) -> str:  # the one fork signal (_target.py)
        return _SHA


def _run(ops: FakeOps, path: Path, model: Path, **kw: Any) -> "tuple[LiveDefault, list]":
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with _spec().emit_default(path, model_h5=model, ops=ops, **kw) as run:
            pass
    return run, caught


def _cats(caught: list) -> list:
    return [w.category for w in caught]


def _assert_nodes_equal(res: Results, src: Path, model: Path) -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        ref = Results.from_mpco(str(src), model_h5=str(model))
    try:
        for name in _DISP:
            a = ref.stage(ref.stages[0].id).nodes.get(component=name)
            b = res.stage(res.stages[0].id).nodes.get(component=name)
            np.testing.assert_allclose(np.asarray(b.values),
                                       np.asarray(a.values), rtol=1e-12)
    finally:
        ref.close()


# ---------------------------------------------------------------------------
# The three routes
# ---------------------------------------------------------------------------


def test_fork_records_its_own_file_stamps_it_and_reads_it(files, tmp_path) -> None:
    ops = ForkOps(files["fork"])
    run, caught = _run(ops, tmp_path / "run", files["model"])
    target = tmp_path / "run.ladruno"
    assert run.route == "fork" and run.path == target and caught == []
    (args,) = ops.recorder_calls
    assert args == fork_recorder_args(_spec().records, target)
    assert args[1:] == mpco_ops_args(_spec().records, output_dir=str(tmp_path),
                                     filename="run.ladruno")[1:]
    assert ops.live == {}
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        res = run.results()
    try:
        _assert_nodes_equal(res, files["mpco"], files["model"])
    finally:
        res.close()
    import h5py

    with h5py.File(target, "r") as f, h5py.File(files["model"], "r") as m:
        assert str(f["INFO/PROVENANCE"].attrs["model_hash"]) == \
            str(m["meta/lineage"].attrs["model_hash"])


def test_a_fork_stamp_refusal_warns_once_and_reads_unstamped(
    files, tmp_path, monkeypatch,
) -> None:
    from apeGmsh.results.readers import _ladruno_live
    from apeGmsh.results.readers._ladruno_provenance import (
        ProvenanceStampRefused,
    )

    def refuse(path: Any, model_h5: Any) -> None:
        raise ProvenanceStampRefused(f"{path}: locked by another process")

    monkeypatch.setattr(_ladruno_live, "stamp_provenance", refuse)
    run, caught = _run(ForkOps(files["fork"]), tmp_path / "run", files["model"])
    assert caught == []
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        res = run.results()
    try:
        assert _cats(caught) == [DefaultRouteWarning]
        assert "unstamped" in str(caught[0].message)
        assert "locked by another process" in str(caught[0].message)
        _assert_nodes_equal(res, files["mpco"], files["model"])
    finally:
        res.close()


def test_a_real_stamp_refusal_is_skipped_too(files, tmp_path) -> None:
    """A file stamped for another archive refuses the stamp (§7b.1 rule 3)."""
    (stale,) = convert_mpco(files["mpco"], tmp_path / "stale.ladruno",
                            model_h5=files["model"])
    run, _ = _run(ForkOps(stale), tmp_path / "run", files["other"])
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        res = run.results()
    res.close()
    ours = [w for w in caught if w.category is DefaultRouteWarning]
    assert len(ours) == 1 and "already stamped" in str(ours[0].message)


def test_stock_with_mpco_reads_the_mpco_after_one_warning(files, tmp_path) -> None:
    ops = FakeOps(files["mpco"])
    run, caught = _run(ops, tmp_path / "run", files["model"])
    target = tmp_path / "run.mpco"
    assert run.route == "mpco" and run.path == target
    assert _cats(caught) == [DefaultRouteWarning]
    assert "not converted" in str(caught[0].message)
    (args,) = ops.recorder_calls
    assert args == mpco_ops_args(_spec().records, output_dir=str(tmp_path),
                                 filename="run.mpco")
    assert ops.live == {} and target.exists()
    assert not (tmp_path / "run.ladruno").exists()
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        res = run.results()
    try:
        _assert_nodes_equal(res, files["mpco"], files["model"])
    finally:
        res.close()


@pytest.mark.parametrize("leaves", [False, True], ids=["clean", "partial-file"])
def test_pip_falls_back_to_classic_recorders_with_one_warning(
    files, tmp_path, leaves,
) -> None:
    ops = FakeOps(mpco=False, mpco_leaves=leaves)
    run, caught = _run(ops, tmp_path / "run", files["model"])
    assert run.route == "recorders" and run.path == tmp_path / "run_out"
    assert _cats(caught) == [DefaultRouteWarning]
    assert "no MPCO recorder" in str(caught[0].message)
    assert not (tmp_path / "run.mpco").exists()       # the probe's file is gone
    assert ops.recorder_calls[0][0] == "mpco"
    assert {a[0] for a in ops.recorder_calls[1:]} == {"Node"}
    assert ops.live == {}                              # nothing left registered
    assert sorted(ops.removed) == sorted(range(1, len(ops.recorder_calls)))


def test_a_probe_that_returns_no_tag_is_a_failed_probe(files, tmp_path) -> None:
    class Odd(FakeOps):
        def recorder(self, *args: Any) -> int:
            if args[0] == "mpco":
                self.recorder_calls.append(args)
                return -1
            return super().recorder(*args)

    ops = Odd()
    run, caught = _run(ops, tmp_path / "run", files["model"])
    assert run.route == "recorders" and _cats(caught) == [DefaultRouteWarning]
    assert "returned -1" in str(caught[0].message)
    assert ops.live == {} and -1 not in ops.removed


# ---------------------------------------------------------------------------
# Refusals and cleanup
# ---------------------------------------------------------------------------


def test_a_fork_rerun_refuses_unless_overwrite(files, tmp_path) -> None:
    run, _ = _run(ForkOps(files["fork"]), tmp_path / "run", files["model"])
    first = run.path
    assert first is not None and first.exists()
    with pytest.raises(FileExistsError, match="overwrite=True"):
        _run(ForkOps(files["fork"]), tmp_path / "run", files["model"])
    run2, caught = _run(ForkOps(files["fork"]), tmp_path / "run.ladruno",
                        files["model"], overwrite=True)
    assert run2.path == first and caught == []


def test_the_fork_refusing_its_recorder_raises(files, tmp_path) -> None:
    class Refuses(ForkOps):
        def recorder(self, *args: Any) -> int:
            raise RuntimeError("bad token -N")

    with pytest.raises(RuntimeError, match="refused its own results recorder"):
        _run(Refuses(), tmp_path / "run", files["model"])


def test_an_exception_in_the_block_still_removes_the_recorder(files, tmp_path) -> None:
    ops = ForkOps(files["fork"])
    with pytest.raises(KeyError, match="boom"):
        with _spec().emit_default(tmp_path / "run", model_h5=files["model"],
                                  ops=ops):
            raise KeyError("boom")
    assert ops.live == {} and ops.removed == [1]


@pytest.mark.parametrize("kw, exc, match", [
    ({"path": "run.out"}, ValueError, "suffix"),
    ({"model_h5": "missing.h5"}, FileNotFoundError, "ops.h5"),
])
def test_bad_arguments_refuse_before_any_recorder(
    files, tmp_path, kw, exc, match,
) -> None:
    ops = ForkOps(files["fork"])
    path = tmp_path / kw.get("path", "run")
    model = tmp_path / kw["model_h5"] if "model_h5" in kw else files["model"]
    with pytest.raises(exc, match=match):
        with _spec().emit_default(path, model_h5=model, ops=ops):
            pass
    assert ops.recorder_calls == []


def test_modal_records_have_no_default_route(files, tmp_path) -> None:
    spec = ResolvedRecorderSpec(fem_snapshot_id="testhash", records=(
        ResolvedRecorderRecord(category="modal", name="m", components=(),
                               dt=None, n_steps=None, n_modes=2),))
    ops = ForkOps(files["fork"])
    with pytest.raises(ValueError, match="modal"):
        with spec.emit_default(tmp_path / "run", model_h5=files["model"], ops=ops):
            pass
    assert ops.recorder_calls == []


def test_model_h5_is_keyword_required(files, tmp_path) -> None:
    with pytest.raises(TypeError):
        _spec().emit_default(tmp_path / "run")  # type: ignore[call-arg]


def test_results_inside_the_block_refuses(files, tmp_path) -> None:
    with _spec().emit_default(tmp_path / "run", model_h5=files["model"],
                              ops=ForkOps(files["fork"])) as run:
        with pytest.raises(RuntimeError, match="after the with block"):
            run.results()
