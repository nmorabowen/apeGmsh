"""#2437 / #2440: every ManzariDafalias deck states ``-stage 0`` explicitly.

Upstream OpenSees starts ManzariDafalias elastoplastic (its class-wide
``mElastFlag`` starts at 1); the Ladruno fork starts it elastic (its
constructors set 0).  The same deck therefore gave two answers.  The deck
emitters now write ``updateMaterialStage -material <tag> -stage 0`` for every
ManzariDafalias-group material (``StageZeroLedger`` in
``_internal/build/stage_validate.py``).

* **Where.** The line acts only through elements already in the domain, so it
  is written at the first solve verb (``analyze`` / ``eigen``) after the
  elements, and again after a later stage adds elements.  Once a user
  ``s.update_material_stage`` names the group it is never written again, so a
  user stage 1 still wins.
* **Archive.** The line is the emitter's, not the model's.  ``model.h5``
  stores none of it, so ``model_hash`` does not move, and the replayed deck
  writes it exactly as the bridge's deck does (R1b,
  ``tests/opensees/golden/test_roundtrip_oracle.py``).
* **Oracle.** On the oedometric column below the elastic stage has a closed
  form: the lateral stress is ``nu / (1 - nu)`` times the vertical one.  The
  bridge deck meets it on stock and on the fork, and its settlement equals
  the fork's committed value to 1e-10.  The same deck with the line stripped
  does neither on stock.
"""
from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import warnings
from pathlib import Path
from typing import Any, Callable

import h5py
import pytest

from apeGmsh.opensees import OpenSeesModel
from apeGmsh.opensees._internal.build import stage_validate
from apeGmsh.opensees._internal.build.stage_validate import StageZeroLedger
from apeGmsh.opensees._internal.lineage import read_stored_lineage
from apeGmsh.opensees.apesees import apeSees
from apeGmsh.opensees.emitter.py import PyEmitter
from apeGmsh.opensees.emitter.recording import RecordingEmitter
from apeGmsh.opensees.emitter.tcl import TclEmitter

from tests.opensees.h5._opensees_model_fixtures import build_two_quad_fem

#: Toyoura sand (Dafalias & Manzari 2004); every optional field at the
#: bridge's default.
_MD = dict(
    G0=125.0, nu=0.05, e_init=0.8, Mc=1.25, c=0.712, lambda_c=0.019,
    e0=0.934, ksi=0.7, P_atm=100.0, m=0.01, h0=7.05, Ch=0.968, nb=1.1,
    A0=0.704, nd=3.5, z_max=4.0, cz=600.0, rho=1.42,
)
_MS = dict(
    G0=125.0, nu=0.05, e_init=0.8, Mc=1.25, c=0.712, lambda_c=0.019,
    e0=0.934, ksi=0.7, P_atm=100.0, m=0.01, h0=7.05, Ch=0.968, nb=1.1,
    A0=0.704, nd=3.5, zeta=0.0005, mu0=260.0, beta=1.0, rho=1.6,
)


def _chain(ops: apeSees) -> dict[str, object]:
    return dict(
        constraints=ops.constraints.Plain(), numberer=ops.numberer.Plain(),
        system=ops.system.FullGeneral(),
        test=ops.test.NormDispIncr(tol=1e-10, max_iter=100),
        algorithm=ops.algorithm.Newton(),
        integrator=ops.integrator.LoadControl(dlam=0.1),
        analysis=ops.analysis.Static())


def _md_column(
    *, flip: "int | None" = None, second: bool = True,
    activate_fill: bool = False, cls: str = "ManzariDafalias",
) -> apeSees:
    """The two stacked quads of ``build_two_quad_fem`` as an oedometer: no
    lateral strain, base fixed, 100 on top in stage ``confine``, then 20
    more in stage ``load`` (``second``), which flips the sand to ``flip``
    first when given.  A flipped column gives each quad its own material:
    a flip to 1 initialises one element per material (#2445), which the
    bridge refuses for a material on two.  ``activate_fill`` puts the upper quad in the first
    stage instead of the global pass."""
    ops = apeSees(build_two_quad_fem(), default_orientation=None)
    ops.model(ndm=2, ndf=2)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if cls == "ElasticIsotropic":
            sand = ops.nDMaterial.ElasticIsotropic(
                name="sand", E=1e5, nu=0.05, rho=0.0)
        else:
            sand = getattr(ops.nDMaterial, cls)(
                name="sand", **(_MS if cls == "SAniSandMS" else _MD))
        fill = sand
        if flip is not None:
            fill = getattr(ops.nDMaterial, cls)(name="fill", **_MD)
    ops.element.FourNodeQuad(pg="Rock", thickness=1.0, material=sand)
    ops.element.FourNodeQuad(pg="Fill", thickness=1.0, material=fill)
    ops.fix(nodes=[1, 2], dofs=(1, 1))
    ops.fix(nodes=[3, 4], dofs=(1, 0))
    if not activate_fill:
        ops.fix(nodes=[5, 6], dofs=(1, 0))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")   # the NormDispIncr advisory
        with ops.stage(name="confine") as s:
            if activate_fill:
                s.activate(pgs=["Fill"])
                s.fix(pg="FillTop", dofs=(1, 0))
            with s.pattern(series=ops.timeSeries.Linear()) as p:
                for node in (5, 6):
                    p.load(node=node, forces=(0.0, -50.0))
            s.analysis(**_chain(ops))
            s.run(n_increments=10)
        if second:
            with ops.stage(name="load") as s:
                if flip is not None:
                    s.update_material_stage(
                        materials=[sand, fill], stage=flip)
                with s.pattern(series=ops.timeSeries.Linear()) as p:
                    for node in (5, 6):
                        p.load(node=node, forces=(0.0, -10.0))
                s.analysis(**_chain(ops))
                s.run(n_increments=10)
    return ops


def _md_column_flipped() -> apeSees:
    """The R1b corpus row: a stage-activated quad, the derived line in both
    stages, and a user stage 1."""
    return _md_column(flip=1, activate_fill=True)


def _tcl(ops: apeSees, tmp_path: Path) -> list[str]:
    deck = tmp_path / "deck.tcl"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ops.tcl(str(deck), progress=False)
    return deck.read_text(encoding="utf-8").splitlines()


def _stage_lines(lines: list[str]) -> list[tuple[int, str]]:
    return [(i, ln.strip()) for i, ln in enumerate(lines)
            if ln.strip().startswith("updateMaterialStage")]


def _index(lines: list[str], pred: Callable[[str], bool]) -> list[int]:
    return [i for i, ln in enumerate(lines) if pred(ln.strip())]


# ---------------------------------------------------------------------------
# The ledger, on each deck emitter
# ---------------------------------------------------------------------------


_EMITTERS = [TclEmitter, PyEmitter, RecordingEmitter]


@pytest.mark.parametrize("cls", _EMITTERS, ids=lambda c: c.__name__)
def test_the_ledger_writes_once_per_element_batch_until_the_user_flips(
    cls: type,
) -> None:
    em = cls()
    em.nDMaterial("ManzariDafalias", 7, *([1.0] * 18))
    em.nDMaterial("LadrunoSANISAND", 8, *([1.0] * 18))
    em.nDMaterial("SAniSandMS", 9, *([1.0] * 19))
    em.nDMaterial("ElasticIsotropic", 10, 1.0, 0.3, 0.0)
    em.analyze(steps=1)                       # no element yet: nothing owed
    assert _stage_zero_tags(em) == []
    em.element("quad", 1, 1, 2, 3, 4, 1.0, "PlaneStrain", 7)
    em.analyze(steps=1)                       # the group, in declaration order
    assert _stage_zero_tags(em) == [7, 8]
    em.analyze(steps=1)                       # no new element: not again
    em.eigen(1)
    assert _stage_zero_tags(em) == [7, 8]
    em.element("quad", 2, 5, 6, 7, 8, 1.0, "PlaneStrain", 8)
    em.eigen(1)                               # a new batch: again, at eigen
    assert _stage_zero_tags(em) == [7, 8, 7, 8]
    em.update_material_stage(9, 1)            # SAniSandMS: another static
    em.element("quad", 3, 5, 6, 7, 8, 1.0, "PlaneStrain", 9)
    em.eigen_feast(0.0, 1.0)
    assert _stage_zero_tags(em) == [7, 8, 7, 8, 7, 8]
    em.update_material_stage(8, 1)            # the user owns the group now
    em.element("quad", 4, 5, 6, 7, 8, 1.0, "PlaneStrain", 7)
    em.analyze(steps=1)
    assert _stage_zero_tags(em) == [7, 8, 7, 8, 7, 8]


@pytest.mark.parametrize("cls", _EMITTERS, ids=lambda c: c.__name__)
@pytest.mark.parametrize(("args", "takes_over"), [
    (("materialState", 7), True), (("updateMaterialStage", 7), True),
    (("poissonRatio", 7), False), (("materialState", 9), False)])
def test_a_stage_flag_parameter_takes_the_group_over(
    cls: type, args: tuple[str, int], takes_over: bool,
) -> None:
    """The ``s.update_parameter`` route sets the same static."""
    em = cls()
    em.nDMaterial("ManzariDafalias", 7, *([1.0] * 18))
    em.element("quad", 1, 1, 2, 3, 4, 1.0, "PlaneStrain", 7)
    em.update_parameter(1, (1,), args, 1.0)
    em.analyze(steps=1)
    assert _stage_zero_tags(em) == ([] if takes_over else [7])


def _stage_zero_tags(em: Any) -> list[int]:
    if isinstance(em, RecordingEmitter):
        return [a[0] for n, a, _k in em.calls
                if n == "update_material_stage" and a[1] == 0]
    tags: list[int] = []
    for ln in em.lines():
        toks = ln.replace("(", " ").replace(")", " ").replace(",", " ").split()
        if "-material" in " ".join(toks) and "updateMaterialStage" in ln:
            words = [t.strip("'\"") for t in toks]
            if words[words.index("-stage") + 1] == "0":
                tags.append(int(words[words.index("-material") + 1]))
    return tags


def test_a_group_without_a_manzari_class_owes_nothing() -> None:
    led = StageZeroLedger()
    led.material("SAniSandMS", 1)
    led.material("PressureDependMultiYield", 2)
    led.owed = True
    assert led.due() == ()


def test_the_group_is_the_table_entries_sharing_the_manzari_static() -> None:
    assert stage_validate.stage_zero_classes() == frozenset(
        {"ManzariDafalias", "LadrunoSANISAND"})


# ---------------------------------------------------------------------------
# The bridge's decks
# ---------------------------------------------------------------------------


def test_tcl_writes_stage_zero_after_the_elements_and_before_analyze(
    tmp_path: Path,
) -> None:
    ops = _md_column(second=False)
    lines = _tcl(ops, tmp_path)
    tag = ops.tag_for(ops._names["sand"])
    (row,) = _stage_lines(lines)
    assert row[1] == f"updateMaterialStage -material {tag} -stage 0"
    last_element = max(_index(lines, lambda s: s.startswith("element ")))
    first_analyze = min(_index(lines, lambda s: "analyze 1" in s))
    assert last_element < row[0] < first_analyze


def test_py_writes_the_same_line(tmp_path: Path) -> None:
    ops = _md_column(second=False)
    deck = tmp_path / "deck.py"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ops.py(str(deck), progress=False)
    text = deck.read_text(encoding="utf-8")
    tag = ops.tag_for(ops._names["sand"])
    assert text.count("updateMaterialStage") == 1
    assert (f"ops.updateMaterialStage('-material', {tag}, '-stage', 0)"
            in text)


def test_a_flat_deck_writes_it_before_its_analyze(tmp_path: Path) -> None:
    """No stage: the deck's one ``analyze`` (``analyze_steps=``)."""
    ops = apeSees(build_two_quad_fem(), default_orientation=None)
    ops.model(ndm=2, ndf=2)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sand = ops.nDMaterial.ManzariDafalias(name="sand", **_MD)
        ops.element.FourNodeQuad(pg="Rock", thickness=1.0, material=sand)
        ops.fix(pg="Base", dofs=(1, 1))
        with ops.pattern.Plain(series=ops.timeSeries.Linear()) as p:
            p.load(node=3, forces=(0.0, -1.0))
        _chain(ops)   # the flat chain, registered globally
        deck = tmp_path / "flat.tcl"
        ops.tcl(str(deck), progress=False, analyze_steps=2)
        bare = tmp_path / "bare.tcl"
        ops.tcl(str(bare), progress=False)
    lines = deck.read_text(encoding="utf-8").splitlines()
    (row,) = _stage_lines(lines)
    assert row[0] < min(_index(lines, lambda s: "analyze 1" in s))
    # A deck that drives no solve writes no line: nothing reads the flag.
    assert not _stage_lines(bare.read_text(encoding="utf-8").splitlines())


def test_a_user_stage_one_comes_after_and_wins(tmp_path: Path) -> None:
    ops = _md_column(flip=1)
    lines = _tcl(ops, tmp_path)
    tags = [ops.tag_for(ops._names[n]) for n in ("sand", "fill")]
    rows = _stage_lines(lines)
    assert [r[1] for r in rows] == [
        f"updateMaterialStage -material {t} -stage {v}"
        for v in (0, 1) for t in tags]
    stage_open = _index(lines, lambda s: s.startswith("# === Stage: load"))
    assert rows[1][0] < stage_open[0] < rows[2][0]


def test_a_user_flip_before_the_first_solve_leaves_no_derived_line(
    tmp_path: Path,
) -> None:
    ops = apeSees(build_two_quad_fem(), default_orientation=None)
    ops.model(ndm=2, ndf=2)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sand = ops.nDMaterial.ManzariDafalias(name="sand", **_MD)
        ops.element.FourNodeQuad(pg="Rock", thickness=1.0, material=sand)
        ops.fix(pg="Base", dofs=(1, 1))
        with ops.stage(name="plastic") as s:
            s.update_material_stage(materials=[sand], stage=1)
            s.analysis(**_chain(ops))
            s.run(n_increments=1)
    rows = _stage_lines(_tcl(ops, tmp_path))
    assert [r[1].split()[-1] for r in rows] == ["1"]


def test_a_stage_that_adds_elements_writes_it_again(tmp_path: Path) -> None:
    """The upper quad enters in stage ``confine``; stage ``load`` adds none."""
    ops = _md_column(activate_fill=True)
    lines = _tcl(ops, tmp_path)
    rows = _stage_lines(lines)
    assert len(rows) == 1
    fill_element = max(_index(lines, lambda s: s.startswith("element ")))
    first_analyze = min(_index(lines, lambda s: "analyze 1" in s))
    assert fill_element < rows[0][0] < first_analyze


@pytest.mark.parametrize(("cls", "n"), [
    ("ManzariDafalias", 1), ("LadrunoSANISAND", 1), ("SAniSandMS", 0),
    ("ElasticIsotropic", 0)])
def test_only_the_manzari_group_gets_the_line(
    cls: str, n: int, tmp_path: Path,
) -> None:
    rows = _stage_lines(_tcl(_md_column(second=False, cls=cls), tmp_path))
    assert len(rows) == n


# ---------------------------------------------------------------------------
# The archive: nothing stored, the replay writes it back
# ---------------------------------------------------------------------------


def test_the_archive_stores_only_the_user_flip(tmp_path: Path) -> None:
    ops = _md_column(flip=1)
    path = tmp_path / "model.h5"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ops.h5(str(path))
    tags = [ops.tag_for(ops._names[n]) for n in ("sand", "fill")]
    rows: list[list[int]] = []

    def _visit(name: str, obj: object) -> None:
        # The (material, stage) rows; the declaration index beside them
        # is one column.
        if (isinstance(obj, h5py.Dataset) and obj.ndim == 2
                and name.endswith("update_material_stage")):
            rows.extend(obj[()].tolist())

    with h5py.File(path, "r") as f:
        f.visititems(_visit)
    assert rows == [[t, 1] for t in tags]


def test_model_hash_does_not_see_the_line(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The archive with every derived line suppressed has the same hash."""
    def _hash(name: str) -> "str | None":
        path = tmp_path / name
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            _md_column(flip=1).h5(str(path))
        with h5py.File(path, "r") as f:
            return read_stored_lineage(f["meta"])[1]

    with_line = _hash("with.h5")
    monkeypatch.setattr(StageZeroLedger, "due", lambda self: ())
    assert with_line is not None and with_line == _hash("without.h5")


@pytest.mark.parametrize("build", [_md_column_flipped, _md_column])
def test_the_replayed_deck_writes_the_same_lines(
    build: Callable[[], apeSees], tmp_path: Path,
) -> None:
    ops = build()
    bridge = _tcl(ops, tmp_path)
    archive = tmp_path / "model.h5"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ops.h5(str(archive))
        replay = OpenSeesModel.from_h5(str(archive)).build("tcl")
    assert isinstance(replay, str)
    assert _stage_lines(bridge)
    assert ([r[1] for r in _stage_lines(replay.splitlines())]
            == [r[1] for r in _stage_lines(bridge)])


# ---------------------------------------------------------------------------
# Live: stock and the fork give one answer
# ---------------------------------------------------------------------------

#: The fork's settlement of node 5 after stage ``confine`` and after stage
#: ``load`` with no flip (fork 79e06, Windows).  Stock 3.8.0 gives the same
#: digits once the deck states ``-stage 0``; without it, stock's
#: elastoplastic start does not converge on this leg.
_UY_CONFINE = -0.002897089609708827
_UY_LOAD = -0.0034765075316505935
_NU = _MD["nu"]

_CHILD = r"""
import json, sys
src = open(sys.argv[1], encoding="utf-8").read()
if len(sys.argv) > 2:
    src = src.replace(sys.argv[2], "pass")
ns = {"__name__": "deck"}
try:
    exec(compile(src, sys.argv[1], "exec"), ns)
    ok = True
except SystemExit:
    ok = False
ops = ns["ops"]
tag = max(ops.getEleTags())
s = ops.eleResponse(tag, "stresses")
print("RESULT " + json.dumps({
    "ok": ok, "uy": ops.nodeDisp(5, 2), "sxx": s[0], "syy": s[1],
    "fork": hasattr(ops, "ladrunoBuild")}))
"""


def _run(ops: apeSees, tmp_path: Path, strip: "str | None" = None,
         ) -> dict[str, Any]:
    deck = tmp_path / "deck.py"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ops.py(str(deck), progress=False)
    child = tmp_path / "child.py"
    child.write_text(_CHILD, encoding="utf-8")
    argv = [sys.executable, str(child), str(deck)]
    if strip is not None:
        assert strip in deck.read_text(encoding="utf-8")
        argv.append(strip)
    out = subprocess.run(argv, capture_output=True, text=True, check=False,
                         timeout=300)
    assert out.returncode == 0, out.stderr[-2000:]
    line = next(ln for ln in out.stdout.splitlines()
                if ln.startswith("RESULT "))
    result: dict[str, Any] = json.loads(line[len("RESULT "):])
    return result


def _openseespy() -> bool:
    return importlib.util.find_spec("openseespy") is not None


@pytest.mark.parametrize("second", [False, True], ids=["confine", "load"])
@pytest.mark.live
@pytest.mark.skipif(not _openseespy(), reason="no openseespy")
def test_live_the_default_deck_is_elastic_on_every_build(
    second: bool, tmp_path: Path,
) -> None:
    """Elastic K0 (closed form) and the fork's settlement, to 1e-10."""
    r = _run(_md_column(second=second), tmp_path)
    syy = -120.0 if second else -100.0
    assert r["ok"]
    assert r["syy"] == pytest.approx(syy, rel=1e-10)
    assert r["sxx"] == pytest.approx(_NU / (1.0 - _NU) * syy, rel=1e-10)
    assert r["uy"] == pytest.approx(
        _UY_LOAD if second else _UY_CONFINE, rel=1e-10)


@pytest.mark.live
@pytest.mark.skipif(not _openseespy(), reason="no openseespy")
def test_live_without_the_line_stock_is_not_elastic(tmp_path: Path) -> None:
    """The deck with the derived line stripped: the fork still starts
    elastic, stock does not (the defect this line closes)."""
    ops = _md_column(second=False)
    tag = ops.tag_for(ops._names["sand"])
    r = _run(ops, tmp_path,
             strip=f"ops.updateMaterialStage('-material', {tag}, "
                   "'-stage', 0)")
    elastic = (r["ok"]
               and r["sxx"] == pytest.approx(_NU / (1 - _NU) * -100.0,
                                             rel=1e-10)
               and r["uy"] == pytest.approx(_UY_CONFINE, rel=1e-10))
    assert elastic is bool(r["fork"])


def _flat_column() -> apeSees:
    """The column of :func:`_md_column`, unstaged, with a flat chain."""
    ops = apeSees(build_two_quad_fem(), default_orientation=None)
    ops.model(ndm=2, ndf=2)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sand = ops.nDMaterial.ManzariDafalias(name="sand", **_MD)
        ops.element.FourNodeQuad(pg="Rock", thickness=1.0, material=sand)
        ops.element.FourNodeQuad(pg="Fill", thickness=1.0, material=sand)
        ops.fix(nodes=[1, 2], dofs=(1, 1))
        ops.fix(nodes=[3, 4, 5, 6], dofs=(1, 0))
        with ops.pattern.Plain(series=ops.timeSeries.Linear()) as p:
            for node in (5, 6):
                p.load(node=node, forces=(0.0, -50.0))
        _chain(ops)
    return ops


@pytest.mark.parametrize("verb", ["run", "analyze"])
def test_run_and_analyze_both_issue_stage_zero(
    monkeypatch: pytest.MonkeyPatch, verb: str,
) -> None:
    """``ops.run()`` leaves the solve to its caller, so it issues the line
    at its end; ``ops.analyze`` issues it before its solve."""
    import apeGmsh.opensees.emitter.live as live_mod

    from tests.opensees.h5.test_h5_solve_stamp import _StockOps

    stock = _StockOps()
    monkeypatch.setattr(live_mod, "_get_ops", lambda: stock)
    ops = _flat_column()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if verb == "run":
            ops.run()
        else:
            ops.analyze(steps=1)
    tag = ops.tag_for(ops._names["sand"])
    flips = [c for c in stock.calls if c[0] == "updateMaterialStage"]
    assert flips == [("updateMaterialStage", "-material", tag, "-stage", 0)]
    if verb == "run":
        assert stock.calls[-1] == flips[0]


@pytest.mark.live
@pytest.mark.skipif(not _openseespy(), reason="no openseespy")
def test_live_run_then_the_users_analyze_is_elastic_on_every_build() -> None:
    """``ops.run()`` and then the user's own ``analyze``: K0 and the
    fork's settlement on stock too."""
    from apeGmsh.opensees.emitter.live import get_ops

    ops = _flat_column()
    dom = get_ops()
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ops.run()
        assert dom.analyze(10) == 0
        tag = max(dom.getEleTags())
        sxx, syy = dom.eleResponse(tag, "stresses")[:2]
        uy = dom.nodeDisp(5, 2)
    finally:
        dom.wipe()
    assert sxx == pytest.approx(_NU / (1.0 - _NU) * syy, rel=1e-10)
    assert syy == pytest.approx(-100.0, rel=1e-10)
    assert uy == pytest.approx(_UY_CONFINE, rel=1e-10)


@pytest.mark.live
@pytest.mark.skipif(not _openseespy(), reason="no openseespy")
def test_live_in_process_analyze_is_elastic_on_every_build() -> None:
    """The live emitter's own ledger: in-process ``apeSees.analyze`` of the
    unstaged column meets the same closed form and settlement."""
    from apeGmsh.opensees.emitter.live import get_ops

    ops = _flat_column()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        dom = get_ops()
        try:
            assert ops.analyze(steps=10) == 0
            tag = max(dom.getEleTags())
            sxx, syy = dom.eleResponse(tag, "stresses")[:2]
            uy = dom.nodeDisp(5, 2)
        finally:
            dom.wipe()
    assert syy == pytest.approx(-100.0, rel=1e-10)
    assert sxx == pytest.approx(_NU / (1.0 - _NU) * -100.0, rel=1e-10)
    assert uy == pytest.approx(_UY_CONFINE, rel=1e-10)


@pytest.mark.live
@pytest.mark.skipif(not _openseespy(), reason="no openseespy")
def test_live_a_user_stage_one_still_wins(tmp_path: Path) -> None:
    """Stage ``load`` flipped to 1 leaves the elastic K0 line."""
    r = _run(_md_column(flip=1), tmp_path)
    assert r["ok"]
    k0 = _NU / (1.0 - _NU) * -120.0
    assert abs(r["sxx"] - k0) > 1e-3 * abs(k0)
    assert abs(r["uy"] - _UY_LOAD) > 1e-2 * abs(_UY_LOAD)
