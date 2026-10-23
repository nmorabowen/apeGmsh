"""#2463: every class-wide stage-flag group states its fresh stage at the
first solve, so an in-process re-run never inherits the previous run's.

The stage flag of each :data:`~apeGmsh.opensees.stage._builder.
CLASS_WIDE_STAGE_FLAG` group is a C++ ``static``: it survives
``ops.wipe()``.  A second model run in the same process therefore starts
at whatever stage the first one left.  #2440 wrote ``updateMaterialStage
-stage 0`` for the ManzariDafalias group only; the ledger
(``StageZeroLedger``) now writes, for every group, the stage that group
holds in a fresh process:

=======================  ===============  ==========================
group (static)           fresh stage      measured, both builds
=======================  ===============  ==========================
``ManzariDafalias``      0 (elastic)      #2440's normalisation: the
                                          fork starts 0, upstream 1
``SAniSandMS``           1 (plastic)      ``mElastFlag = 1`` in
                                          ``SAniSandMS.cpp``
``PM4Sand``              0 (elastic)      ``me2p = 0``
``PM4Silt``              0 (elastic)      ``me2p = 0``
=======================  ===============  ==========================

A fresh run's answer is therefore unchanged for every group, and a re-run
gets the same answer.  Writing ``-stage 0`` for SAniSandMS instead would
turn every unflipped SAniSandMS model elastic on both builds.

* **Where.** As in #2440: after the elements, at the first solve verb,
  again after a stage adds elements, and never once a user flip (either
  route) names the group.  A user flip of one group leaves the others'.
* **Archive.** ``model.h5`` stores none of the lines, so ``model_hash`` is
  unchanged and the replayed deck writes them back (R1b).
* **Oracle (live).** In one process: a run that leaves the static at the
  other stage, then the default model.  Its answer equals the same model
  with the fresh stage stated by the user, bit for bit, and for the
  elastic-start groups it meets elastic K0 (``sxx = nu / (1 - nu) syy``)
  on the oedometric column.  The deck is run through ``get_ops()``, the
  module the live emitter drives, so the static the two share is one.
"""
from __future__ import annotations

import importlib.util
import warnings
from pathlib import Path
from typing import Any

import h5py
import pytest

from apeGmsh.opensees import OpenSeesModel
from apeGmsh.opensees._internal.build.errors import BridgeError
from apeGmsh.opensees._internal.build.stage_validate import StageZeroLedger
from apeGmsh.opensees._internal.lineage import read_stored_lineage
from apeGmsh.opensees.apesees import apeSees
from apeGmsh.opensees.emitter.py import PyEmitter
from apeGmsh.opensees.emitter.recording import RecordingEmitter
from apeGmsh.opensees.emitter.tcl import TclEmitter
from apeGmsh.opensees.stage import _builder as stage_builder
from apeGmsh.opensees.stage._builder import CLASS_WIDE_STAGE_FLAG

from tests.opensees.h5._opensees_model_fixtures import build_two_quad_fem
from tests.opensees.stage.test_class_wide_stage_flag import _KW

#: The stage each group's static holds in a fresh process (measured on
#: stock openseespy 3.8.0 and the fork, see the module docstring).
_FRESH = {"ManzariDafalias": 0, "LadrunoSANISAND": 0, "SAniSandMS": 1,
          "PM4Sand": 0, "PM4Silt": 0}
#: Poisson's ratio of each class at ``_KW`` (PM4Sand/PM4Silt default 0.3).
_NU = {"ManzariDafalias": 0.05, "SAniSandMS": 0.05, "PM4Sand": 0.3,
       "PM4Silt": 0.3}
#: The classes both builds know (LadrunoSANISAND is the fork's).
_LIVE = ["ManzariDafalias", "SAniSandMS", "PM4Sand", "PM4Silt"]


def test_the_table_is_every_class_wide_class() -> None:
    assert set(_FRESH) == set(CLASS_WIDE_STAGE_FLAG)


def _chain(ops: apeSees, iters: int = 100) -> dict[str, object]:
    return dict(
        constraints=ops.constraints.Plain(), numberer=ops.numberer.Plain(),
        system=ops.system.FullGeneral(),
        test=ops.test.NormDispIncr(tol=1e-10, max_iter=iters),
        algorithm=ops.algorithm.Newton(),
        integrator=ops.integrator.LoadControl(dlam=0.1),
        analysis=ops.analysis.Static())


def _column(cls: str, flip: "int | None" = None, *, steps: int = 10,
            iters: int = 100) -> apeSees:
    """Two stacked quads as an oedometer, each quad its own ``cls``
    material, 100 on top in stage ``confine``, which first flips both
    materials to ``flip`` when given.  ``steps``/``iters`` shorten the
    stale run, whose only job is its flip."""
    ops = apeSees(build_two_quad_fem(), default_orientation=None)
    ops.model(ndm=2, ndf=2)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ns = getattr(ops.nDMaterial, cls)
        sand = ns(name="sand", **_KW[cls])
        fill = ns(name="fill", **_KW[cls])
    ops.element.FourNodeQuad(pg="Rock", thickness=1.0, material=sand)
    ops.element.FourNodeQuad(pg="Fill", thickness=1.0, material=fill)
    ops.fix(nodes=[1, 2], dofs=(1, 1))
    ops.fix(nodes=[3, 4, 5, 6], dofs=(1, 0))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")   # the NormDispIncr advisory
        with ops.stage(name="confine") as s:
            if flip is not None:
                s.update_material_stage(materials=[sand, fill], stage=flip)
            with s.pattern(series=ops.timeSeries.Linear()) as p:
                for node in (5, 6):
                    p.load(node=node, forces=(0.0, -50.0))
            s.analysis(**_chain(ops, iters))
            s.run(n_increments=steps)
    return ops


def _flat(cls: str) -> apeSees:
    """The column of :func:`_column`, unstaged, for in-process analyze."""
    ops = apeSees(build_two_quad_fem(), default_orientation=None)
    ops.model(ndm=2, ndf=2)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ns = getattr(ops.nDMaterial, cls)
        sand = ns(name="sand", **_KW[cls])
        fill = ns(name="fill", **_KW[cls])
        ops.element.FourNodeQuad(pg="Rock", thickness=1.0, material=sand)
        ops.element.FourNodeQuad(pg="Fill", thickness=1.0, material=fill)
        ops.fix(nodes=[1, 2], dofs=(1, 1))
        ops.fix(nodes=[3, 4, 5, 6], dofs=(1, 0))
        with ops.pattern.Plain(series=ops.timeSeries.Linear()) as p:
            for node in (5, 6):
                p.load(node=node, forces=(0.0, -50.0))
        _chain(ops)
    return ops


def _tags(ops: apeSees) -> list[int]:
    return [ops.tag_for(ops._names[n]) for n in ("sand", "fill")]


def _deck(ops: apeSees, path: Path) -> list[str]:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if path.suffix == ".tcl":
            ops.tcl(str(path), progress=False)
        else:
            ops.py(str(path), progress=False)
    return path.read_text(encoding="utf-8").splitlines()


def _stage_rows(lines: list[str]) -> list[tuple[int, int, int]]:
    """``(line index, material tag, stage)`` of every stage line, in either
    deck dialect."""
    rows = []
    for i, ln in enumerate(lines):
        if "updateMaterialStage" not in ln:
            continue
        words = [w.strip("'\"") for w in ln.replace("(", " ").replace(
            ")", " ").replace(",", " ").split()]
        rows.append((i, int(words[words.index("-material") + 1]),
                     int(words[words.index("-stage") + 1])))
    return rows


# ---------------------------------------------------------------------------
# The bridge's decks
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("suffix", [".tcl", ".py"])
@pytest.mark.parametrize("cls", sorted(_FRESH))
def test_the_deck_states_the_fresh_stage_of_every_group(
    cls: str, suffix: str, tmp_path: Path,
) -> None:
    ops = _column(cls)
    lines = _deck(ops, tmp_path / f"deck{suffix}")
    rows = _stage_rows(lines)
    assert [(t, s) for _i, t, s in rows] == [
        (t, _FRESH[cls]) for t in _tags(ops)]
    last_element = max(i for i, ln in enumerate(lines) if "quad" in ln)
    first_analyze = min(i for i, ln in enumerate(lines)
                        if "analyze 1" in ln or "analyze(1)" in ln)
    assert all(last_element < i < first_analyze for i, _t, _s in rows)


@pytest.mark.parametrize("cls", sorted(_FRESH))
def test_a_user_flip_still_wins(cls: str, tmp_path: Path) -> None:
    """A user flip to the other stage before the solve: only its lines."""
    flip = 1 - _FRESH[cls]
    ops = _column(cls, flip=flip)
    rows = _stage_rows(_deck(ops, tmp_path / "deck.tcl"))
    assert [(t, s) for _i, t, s in rows] == [(t, flip) for t in _tags(ops)]


_EMITTERS = [TclEmitter, PyEmitter, RecordingEmitter]


def _emitted(em: Any) -> list[tuple[int, int]]:
    if isinstance(em, RecordingEmitter):
        return [(int(a[0]), int(a[1])) for n, a, _k in em.calls
                if n == "update_material_stage"]
    return [(t, s) for _i, t, s in _stage_rows(list(em.lines()))]


@pytest.mark.parametrize("cls", _EMITTERS, ids=lambda c: c.__name__)
def test_a_flip_of_one_group_leaves_the_other_groups(cls: type) -> None:
    """Each static is its own: the ledger hands over one group at a time,
    by either route (``updateMaterialStage`` or a stage-flag
    ``updateParameter``)."""
    em = cls()
    em.nDMaterial("ManzariDafalias", 7, *([1.0] * 18))
    em.nDMaterial("PM4Sand", 8, *([1.0] * 4))
    em.nDMaterial("SAniSandMS", 9, *([1.0] * 19))
    em.nDMaterial("PM4Silt", 10, *([1.0] * 5))
    em.nDMaterial("ElasticIsotropic", 11, 1.0, 0.3, 0.0)
    em.element("quad", 1, 1, 2, 3, 4, 1.0, "PlaneStrain", 7)
    em.analyze(steps=1)
    assert _emitted(em) == [(7, 0), (8, 0), (9, 1), (10, 0)]
    em.update_material_stage(8, 1)            # PM4Sand is the user's now
    em.update_parameter(1, (1,), ("materialState", 9), 0.0)  # and SAniSandMS
    em.element("quad", 2, 5, 6, 7, 8, 1.0, "PlaneStrain", 8)
    em.analyze(steps=1)
    assert _emitted(em)[4:] == [(8, 1), (7, 0), (10, 0)]


def test_a_class_wide_static_without_a_fresh_stage_refuses(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Fail closed: a new table entry must say which stage its static
    starts at, or no emitter builds."""
    monkeypatch.setitem(stage_builder.CLASS_WIDE_STAGE_FLAG,
                        "FooSand", "FooSand::mElastFlag")
    with pytest.raises(BridgeError, match="FooSand::mElastFlag"):
        StageZeroLedger()


# ---------------------------------------------------------------------------
# The archive: nothing stored, the replay writes the same lines
# ---------------------------------------------------------------------------


def _model_hash(ops: apeSees, path: Path) -> "str | None":
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ops.h5(str(path))
    with h5py.File(path, "r") as f:
        return read_stored_lineage(f["meta"])[1]


@pytest.mark.parametrize("cls", sorted(_FRESH))
def test_model_hash_does_not_see_the_lines(
    cls: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    with_lines = _model_hash(_column(cls), tmp_path / "with.h5")
    monkeypatch.setattr(StageZeroLedger, "due", lambda self: ())
    assert with_lines is not None
    assert with_lines == _model_hash(_column(cls), tmp_path / "without.h5")


@pytest.mark.parametrize("flip", [None, "user"])
@pytest.mark.parametrize("cls", sorted(_FRESH))
def test_the_replayed_deck_writes_the_same_lines(
    cls: str, flip: "str | None", tmp_path: Path,
) -> None:
    ops = _column(cls, flip=None if flip is None else 1 - _FRESH[cls])
    bridge = _deck(ops, tmp_path / "deck.tcl")
    archive = tmp_path / "model.h5"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ops.h5(str(archive))
        replay = OpenSeesModel.from_h5(str(archive)).build("tcl")
    assert isinstance(replay, str)
    assert _stage_rows(bridge)
    assert ([(t, s) for _i, t, s in _stage_rows(replay.splitlines())]
            == [(t, s) for _i, t, s in _stage_rows(bridge)])


# ---------------------------------------------------------------------------
# Live: run 2 in the same process starts at the fresh stage, on every build
# ---------------------------------------------------------------------------


def _backend() -> bool:
    return (importlib.util.find_spec("openseespy") is not None
            or importlib.util.find_spec("opensees") is not None)


def _exec_deck(ops: apeSees, tmp_path: Path, name: str) -> dict[str, Any]:
    """Run the bridge's py deck in this process, on ``get_ops()``."""
    from apeGmsh.opensees.emitter.live import get_ops

    dom = get_ops()
    path = tmp_path / f"{name}.py"
    src = "\n".join(_deck(ops, path))
    head = "import openseespy.opensees as ops"
    assert head in src
    ns: dict[str, Any] = {"__name__": "deck", "ops": dom}
    try:
        exec(compile(src.replace(head, "", 1), str(path), "exec"), ns)
        ok = True
    except SystemExit:
        ok = False
    return {"ok": ok, **_read(dom)}


def _read(dom: Any) -> dict[str, Any]:
    tag = max(dom.getEleTags())
    s = dom.eleResponse(tag, "stresses")
    return {"uy": dom.nodeDisp(5, 2), "sxx": s[0], "syy": s[1]}


def _stale_then(cls: str, tmp_path: Path) -> None:
    """Run 1: leave the group's static at the other stage (the flip line
    sets it before the solve, so one short increment is enough)."""
    _exec_deck(_column(cls, flip=1 - _FRESH[cls], steps=1, iters=2),
               tmp_path, "stale")


def _check(cls: str, got: dict[str, Any], ref: dict[str, Any]) -> None:
    for key in ("uy", "sxx", "syy"):
        assert got[key] == pytest.approx(ref[key], rel=1e-12, abs=1e-15)
    k0 = _NU[cls] / (1.0 - _NU[cls])
    elastic = got["sxx"] == pytest.approx(k0 * got["syy"], rel=1e-8)
    assert elastic is (_FRESH[cls] == 0)
    if elastic:
        assert got["syy"] == pytest.approx(-100.0, rel=1e-10)


@pytest.mark.live
@pytest.mark.skipif(not _backend(), reason="no OpenSees backend")
@pytest.mark.parametrize("cls", _LIVE)
def test_live_a_rerun_in_process_starts_at_the_fresh_stage(
    cls: str, tmp_path: Path,
) -> None:
    """Each time after a stale run 1: the deck with the fresh stage stated
    by the user (the reference), the bridge's default deck, and
    ``apeSees.analyze`` of the unstaged model (the live emitter's
    ledger)."""
    from apeGmsh.opensees.emitter.live import get_ops

    dom = get_ops()
    try:
        _stale_then(cls, tmp_path)
        ref = _exec_deck(_column(cls, flip=_FRESH[cls]), tmp_path, "ref")
        _stale_then(cls, tmp_path)
        deck = _exec_deck(_column(cls), tmp_path, "rerun")
        _stale_then(cls, tmp_path)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            rc = _flat(cls).analyze(steps=10)
        live = _read(dom)
    finally:
        dom.wipe()
    assert ref["ok"] and deck["ok"]
    _check(cls, deck, ref)
    assert rc == 0
    _check(cls, live, ref)
