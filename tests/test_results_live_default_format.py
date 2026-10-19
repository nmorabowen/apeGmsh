"""``spec.emit_default``: the default results route, against fake ops (#1950).

Each fake stands for one build and writes, when its recorder is removed,
the file that build's recorder would have written:

* **fork** (``ladrunoBuild`` returns a sha): a ``.ladruno`` converted from
  the committed springs capture;
* **stock with MPCO**: the springs ``.mpco`` itself;
* **pip** (``recorder('mpco', ...)`` raises): nothing for MPCO; the classic
  recorders it accepts are counted, and removed.

``read_deck_mpco`` (a deck's ``.mpco``) is tested on the same capture: a
staged program converts it, and so does an unstaged one whose archive
records one ``analysis`` and the deck's one ``analyze`` call of the file's
step count (ruling (ii), #2210); with no ``analyze`` call, two ``analysis``
calls or another step count it reads with ``from_mpco`` after one warning.

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

from apeGmsh.mesh._element_types import ElementGroup, make_type_info
from apeGmsh.mesh._group_set import LabelSet, PhysicalGroupSet
from apeGmsh.mesh.FEMData import (
    ElementComposite,
    FEMData,
    MeshInfo,
    NodeComposite,
)
from apeGmsh.opensees.opensees_model import OpenSeesModel
from apeGmsh.results import MPCOConversionRefused, Results, convert_mpco
from apeGmsh.results.readers._ladruno_from_mpco import StageKindRefused
from apeGmsh.results.live import DefaultRouteWarning, LiveDefault, read_deck_mpco
from apeGmsh.results.readers._ladruno_live import fork_recorder_args
from apeGmsh.results.spec._emit import mpco_ops_args
from apeGmsh.results.spec._resolved import (
    ResolvedRecorderRecord,
    ResolvedRecorderSpec,
)
from tests.test_results_mpco_to_ladruno import (
    SPRINGS,
    _bridge_copy,
    _frame_model,
    SPRINGS_STEPS,
    _springs_model,
    _unstaged_springs_model,
)

_SHA = "0123456789abcdef0123456789abcdef01234567"
_DISP = ("displacement_x", "displacement_y", "displacement_z")


@pytest.fixture(scope="module")
def files(tmp_path_factory: pytest.TempPathFactory) -> "dict[str, Path]":
    tmp = tmp_path_factory.mktemp("emit_default")
    model = _springs_model(tmp / "model.h5")          # one static ops.stage
    other = _unstaged_springs_model(tmp / "other.h5", ["static", "static"])
    flat = _unstaged_springs_model(tmp / "flat.h5", ["static"])
    deck = _unstaged_springs_model(tmp / "deck.h5", ["static"],
                                   analyze_steps=SPRINGS_STEPS)
    off = _unstaged_springs_model(tmp / "off.h5", ["static"],
                                  analyze_steps=SPRINGS_STEPS + 1)
    mpco = _bridge_copy(SPRINGS, model, tmp / "src.mpco")
    (fork,) = convert_mpco(mpco, tmp / "src.ladruno", model_h5=model)
    global _SNAPSHOT
    _SNAPSHOT = OpenSeesModel.from_h5(model).lineage.fem_hash
    return {"model": model, "other": other, "flat": flat, "deck": deck,
            "off": off, "mpco": mpco, "fork": fork}


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
        prov = f["INFO/PROVENANCE"].attrs
        assert str(prov["model_hash"]) == str(m["meta/lineage"].attrs["model_hash"])
        # G10 (#2208): the build of the module that ran, from ladrunoBuild.
        assert str(prov["opensees_build_sha"]) == _SHA
        assert str(prov["opensees_backend"]) == "ladruno"


def test_a_fork_stamp_refusal_warns_once_and_reads_unstamped(
    files, tmp_path, monkeypatch,
) -> None:
    from apeGmsh.results.readers import _ladruno_live
    from apeGmsh.results.readers._ladruno_provenance import (
        ProvenanceStampRefused,
    )

    def refuse(path: Any, model_h5: Any, **kw: Any) -> None:
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


def test_pip_names_skipped_fiber_records_in_its_one_warning(files, tmp_path) -> None:
    """The classic route's per-record fiber skip folds into the one warning."""
    spec = ResolvedRecorderSpec(fem_snapshot_id=_SNAPSHOT, records=(
        *_spec().records,
        ResolvedRecorderRecord(category="fibers", name="fib",
                               components=("fiber_stress",), dt=None,
                               n_steps=None, element_ids=np.array([100])),
    ))
    ops = FakeOps(mpco=False)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with spec.emit_default(tmp_path / "run", model_h5=files["model"],
                               ops=ops) as run:
            pass
    assert run.route == "recorders"
    assert _cats(caught) == [DefaultRouteWarning]
    assert "Not recorded on this route: ['fib']" in str(caught[0].message)
    assert ops.live == {}


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
    with pytest.raises(ValueError, match="modal") as e:
        with spec.emit_default(tmp_path / "run", model_h5=files["model"], ops=ops):
            pass
    assert ops.recorder_calls == []
    # The hint names the routes that exist (no DomainCapture).
    assert "ops.eigen(n)" in str(e.value) and "emit_mpco" in str(e.value)
    assert "domain_capture" not in str(e.value)


def test_model_h5_is_keyword_required(files, tmp_path) -> None:
    with pytest.raises(TypeError):
        _spec().emit_default(tmp_path / "run")  # type: ignore[call-arg]


def test_results_inside_the_block_refuses(files, tmp_path) -> None:
    with _spec().emit_default(tmp_path / "run", model_h5=files["model"],
                              ops=ForkOps(files["fork"])) as run:
        with pytest.raises(RuntimeError, match="after the with block"):
            run.results()


# ---------------------------------------------------------------------------
# A deck's .mpco: converted only when the kind is a fact
# ---------------------------------------------------------------------------


def test_a_staged_deck_mpco_converts_and_keeps_the_mpco(files, tmp_path) -> None:
    src = shutil.copy(files["mpco"], tmp_path / "deck.mpco")
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        res = read_deck_mpco(src, model_h5=files["model"])
    try:
        assert [s.kind for s in res.stages] == ["static"]
        _assert_nodes_equal(res, files["mpco"], files["model"])
    finally:
        res.close()
    assert (tmp_path / "deck.ladruno").exists() and Path(src).exists()
    with pytest.raises(FileExistsError, match="overwrite=True"):
        read_deck_mpco(src, model_h5=files["model"])
    read_deck_mpco(src, model_h5=files["model"], overwrite=True).close()


@pytest.mark.parametrize("case", [
    "another-model", "another-model-two-stages", "another-mesh-unstaged",
    "partition-gap", "parts-disagree",
])
def test_a_refusal_that_is_not_about_kinds_raises(files, case, tmp_path) -> None:
    """Only a stage-kind refusal falls back; the rest re-raise, unwarned.

    A foreign archive whose stages would not pair either (two stages
    against one, or no stage at all) still refuses as a foreign archive:
    the lineage check runs before any kind refusal is raised. A later part
    that does not pair, after part 0 did, is a set from two runs."""
    src: Any = shutil.copy(files["mpco"], tmp_path / "deck.mpco")
    model, match = files["model"], "is not a run of"
    if case == "another-model":
        model = _frame_model(tmp_path / "frame.h5", [("only", "static")])
    elif case == "another-model-two-stages":
        model = _frame_model(tmp_path / "frame2.h5",
                             [("g", "static"), ("s", "transient")])
    elif case == "another-mesh-unstaged":
        model = tmp_path / "truss.h5"
        truss_bridge(truss_fem()).h5(str(model))
    elif case == "partition-gap":
        for i in (0, 2):
            shutil.copy(files["mpco"], tmp_path / f"deck.part-{i}.mpco")
        src, match = tmp_path / "deck.part-0.mpco", "contiguous"
    else:
        for i in (0, 1):
            shutil.copy(files["mpco"], tmp_path / f"deck.part-{i}.mpco")
        import h5py

        with h5py.File(tmp_path / "deck.part-1.mpco", "r+") as f:
            f.copy(f["MODEL_STAGE[1]"], "MODEL_STAGE[2]")
        src, match = tmp_path / "deck.part-0.mpco", "not from one run"
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with pytest.raises(MPCOConversionRefused, match=match) as e:
            read_deck_mpco(src, model_h5=model)
    assert not isinstance(e.value, StageKindRefused)
    assert DefaultRouteWarning not in _cats(caught)


def test_a_stage_of_unknown_kind_warns_once_and_reads_the_mpco(
    files, tmp_path,
) -> None:
    """A staged archive whose stage names no known kind is a kind refusal:
    it falls back with one warning, and ``from_mpco`` then refuses the
    archive on its own terms (its stage binding names the same token)."""
    import h5py

    model = Path(shutil.copy(files["model"], tmp_path / "bogus.h5"))
    with h5py.File(model, "r+") as f:
        f["opensees/stages/stage_000/analysis"].attrs["analysis"] = "Bogus"
    src = shutil.copy(files["mpco"], tmp_path / "deck.mpco")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with pytest.raises(ValueError, match="no stage kind") as e:
            read_deck_mpco(src, model_h5=model)
    assert not isinstance(e.value, MPCOConversionRefused)  # from_mpco's own
    ours = [w for w in caught if w.category is DefaultRouteWarning]
    assert len(ours) == 1 and "Bogus" in str(ours[0].message)
    assert not (tmp_path / "deck.ladruno").exists()


def test_an_unstaged_deck_whose_kind_is_archived_converts(files, tmp_path) -> None:
    """Ruling (ii): one ``analysis`` call, the deck's one ``analyze`` call,
    and a file of one stage of ``analyze_steps`` steps convert, silently;
    the ``.mpco`` is kept."""
    model = files["deck"]
    src = _bridge_copy(SPRINGS, model, tmp_path / "deck.mpco")
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        res = read_deck_mpco(src, model_h5=model)
    try:
        assert [s.kind for s in res.stages] == ["static"]
        _assert_nodes_equal(res, src, model)
    finally:
        res.close()
    assert (tmp_path / "deck.ladruno").exists() and src.exists()


@pytest.mark.parametrize("name, reason", [
    ("flat", r"1 analysis and 0 analyze"),
    ("other", r"2 analysis and 0 analyze"),
    ("off", rf"runs {SPRINGS_STEPS + 1}"),
], ids=["no-analyze-call", "two-analysis-calls", "step-count-mismatch"])
def test_an_unstaged_deck_warns_once_and_reads_the_mpco(
    files, name, reason, tmp_path,
) -> None:
    """Ruling (ii): a missed condition reads the ``.mpco`` with
    ``from_mpco`` after one warning that names it. ``flat`` (one
    ``analysis``, no deck ``analyze``) is what a rule-(i) acceptance would
    convert."""
    import re

    model = files[name]
    src = _bridge_copy(SPRINGS, model, tmp_path / "deck.mpco")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        res = read_deck_mpco(src, model_h5=model)
    try:
        assert _cats(caught) == [DefaultRouteWarning]
        assert re.search(reason, str(caught[0].message))
        assert not (tmp_path / "deck.ladruno").exists()
        _assert_nodes_equal(res, src, model)
    finally:
        res.close()


# ---------------------------------------------------------------------------
# A real domain: the two-node truss (shared with the fork lane's file)
# ---------------------------------------------------------------------------

TRUSS_L, TRUSS_E, TRUSS_A, TRUSS_P = 2.0, 2.0e11, 1.0e-4, 1.0e4
TRUSS_STEPS = 4
#: The tip's displacement at full load, P L / (E A).
TRUSS_TIP = TRUSS_P * TRUSS_L / (TRUSS_E * TRUSS_A)


def truss_fem() -> FEMData:
    node_ids = np.array([1, 2], dtype=np.int64)
    coords = np.array([[0.0, 0.0, 0.0], [TRUSS_L, 0.0, 0.0]])
    info = make_type_info(code=1, gmsh_name="Line 2", dim=1, order=1, npe=2,
                          count=1)
    group = ElementGroup(element_type=info, ids=np.array([1], dtype=np.int64),
                         connectivity=np.array([[1, 2]], dtype=np.int64))
    pg = {
        (1, 1): {"name": "Bar", "node_ids": node_ids, "node_coords": coords,
                 "element_ids": group.ids},
        (0, 2): {"name": "Base", "node_ids": node_ids[:1],
                 "node_coords": coords[:1]},
        (0, 3): {"name": "Tip", "node_ids": node_ids[1:],
                 "node_coords": coords[1:]},
    }
    return FEMData(
        nodes=NodeComposite(node_ids=node_ids, node_coords=coords,
                            physical=PhysicalGroupSet(pg), labels=LabelSet({})),
        elements=ElementComposite(groups={1: group},
                                  physical=PhysicalGroupSet(pg),
                                  labels=LabelSet({})),
        info=MeshInfo(n_nodes=2, n_elems=1, bandwidth=2, types=[info]),
        model_name="emit_default_truss",
    )


def truss_bridge(
    fem: FEMData, *, analyses: int = 1, staged: bool = False,
    mpco: "Path | None" = None,
) -> Any:
    """The truss on a bridge: a static load ramp in TRUSS_STEPS steps.

    ``analyses`` static ``analysis`` calls are declared (unstaged), or one
    ``ops.stage`` block holds the load and the chain (``staged``). ``mpco``
    declares a bridge MPCO recorder on the tip's displacement.
    """
    from apeGmsh.opensees import apeSees

    ops = apeSees(fem, default_orientation=None)
    ops.model(ndm=2, ndf=2)
    mat = ops.uniaxialMaterial.ElasticMaterial(E=TRUSS_E)
    ops.element.Truss(pg="Bar", A=TRUSS_A, material=mat)
    ops.fix(pg="Base", dofs=(1, 1))
    ops.fix(pg="Tip", dofs=(0, 1))
    if mpco is not None:
        ops.recorder.MPCO(file=str(mpco), nodal_responses=("displacement",))

    def chain() -> "dict[str, Any]":
        return dict(
            test=ops.test.NormDispIncr(tol=1e-12, max_iter=10),
            algorithm=ops.algorithm.Linear(),
            integrator=ops.integrator.LoadControl(dlam=1.0 / TRUSS_STEPS),
            constraints=ops.constraints.Plain(), numberer=ops.numberer.RCM(),
            system=ops.system.BandGeneral(), analysis=ops.analysis.Static(),
        )

    if staged:
        with ops.stage(name="load") as s:
            with s.pattern(series=ops.timeSeries.Linear()) as pat:
                pat.load(pg="Tip", forces=(TRUSS_P, 0.0))
            s.analysis(**chain())
            s.run(n_increments=TRUSS_STEPS)
        return ops
    with ops.pattern.Plain(series=ops.timeSeries.Linear()) as pat:
        pat.load(pg="Tip", forces=(TRUSS_P, 0.0))
    for _ in range(analyses):
        chain()  # each primitive registers on the bridge as it is made
    return ops


def truss_spec(fem: FEMData) -> ResolvedRecorderSpec:
    return ResolvedRecorderSpec(
        fem_snapshot_id=fem.snapshot_id,
        records=(ResolvedRecorderRecord(
            category="nodes", name="tip",
            components=("displacement_x", "displacement_y"),
            dt=None, n_steps=None, node_ids=np.array([2]),
        ),),
    )


def tip_x(res: Results) -> np.ndarray:
    """The tip's ``displacement_x`` history, one value per recorded step."""
    slab = res.stage(res.stages[-1].id).nodes.get(component="displacement_x")
    values = np.asarray(slab.values)
    ids = list(np.asarray(slab.node_ids))
    return values[:, ids.index(2)]


class NoMPCO:
    """The live module with no fork signal and no MPCO recorder.

    On pip openseespy this is the module itself: its probe fails on its
    own. On a build that has MPCO the wrapper refuses it, so the same test
    runs the classic-recorder route on any build.
    """

    _HIDDEN = frozenset({"ladrunoBuild", "ladrunoSchema"})

    def __init__(self, ops: Any) -> None:
        self._ops = ops

    def __getattr__(self, name: str) -> Any:
        if name in NoMPCO._HIDDEN:
            raise AttributeError(name)
        return getattr(self._ops, name)

    def recorder(self, *args: Any) -> Any:
        if args and args[0] == "mpco":
            raise RuntimeError("recorder mpco: not in this build")
        return self._ops.recorder(*args)


@pytest.mark.live
def test_live_classic_recorder_route_reads_the_run_and_leaves_no_recorder(
    tmp_path,
) -> None:
    from apeGmsh.opensees.emitter.live import get_ops

    opspy = get_ops()
    fem = truss_fem()
    bridge = truss_bridge(fem)
    model = tmp_path / "model.h5"
    bridge.h5(str(model))
    bridge.run()
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            with truss_spec(fem).emit_default(tmp_path / "run", model_h5=model,
                                              ops=NoMPCO(opspy)) as run:
                for _ in range(TRUSS_STEPS):
                    assert opspy.analyze(1) == 0
        assert run.route == "recorders"
        assert [w.category for w in caught] == [DefaultRouteWarning]
        tip = opspy.nodeDisp(2, 1)
        out = sorted(run.path.iterdir())
        assert out, "the classic recorders wrote nothing"
        sizes = [f.stat().st_size for f in out]
        assert opspy.analyze(1) == 0            # a step after the block ...
        assert [f.stat().st_size for f in out] == sizes  # ... records nothing
    finally:
        opspy.wipe()
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        res = run.results(cache_root=tmp_path / "cache")
    try:
        x = tip_x(res)
        assert x.shape == (TRUSS_STEPS,)
        np.testing.assert_allclose(x[-1], tip, rtol=1e-12)
        np.testing.assert_allclose(x[-1], TRUSS_TIP, rtol=1e-12)
    finally:
        res.close()
