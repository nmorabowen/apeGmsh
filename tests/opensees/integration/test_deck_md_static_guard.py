"""#2516, #2518: a deck the fork's ManzariDafalias constructor reset would
change checks the running build first.

On a Ladruno fork build without commit ``43b235bbb`` (fork PR #1005) the
constructor of ``ManzariDafalias`` and ``LadrunoSANISAND`` sets their
class-wide stage flag back to elastic, and every new element copies its
material through it (#2444).  So a stage that builds an element after an
earlier stage flipped the group, and solves before flipping it again,
silently runs every material of the group elastic on such a build.  Stock
OpenSees keeps the flag (``test_live_reset_happens_exactly_where_the_table_says``
on ``live-stock``), as does ``FORK_PIN``.

No command marks the fix, but ``ladrunoSchema`` v1 (fork PR #1017) lists
it as the ``mdStageFlagPreserved`` capability.  So the deck asks: a build
without ``ladrunoBuild`` (stock) runs on, and a build with it runs only
when ``ladrunoSchema`` answers a v1 schema whose ``capabilities`` hold the
name (#2518; #2516 matched an allowlist of measured shas instead).  A v0
schema, no ``ladrunoSchema``, or any other answer stops it by name.  The
stage close does not refuse the pattern: the build is known only where
the model runs.

The Tcl guard is evaluated in ``tkinter.Tcl`` and the py guard with
``exec`` against stub engines, on every lane.  The live cases run the
bridge's py deck in a child process on the bound build.
"""
from __future__ import annotations

import os
import subprocess
import sys
import types
import warnings
from pathlib import Path
from typing import Any

import pytest

import apeGmsh
from apeGmsh.opensees import apeSees
from apeGmsh.opensees._internal.build import stage_emit
from apeGmsh.opensees._internal.build.errors import BridgeError
from apeGmsh.opensees._internal.build.stage_emit import (
    MD_STATIC_RESET_GUARD,
    MD_STATIC_RESET_MESSAGE,
    ForkBuildCapability,
    fork_guard_lines,
)
from apeGmsh.opensees._target import OpenSeesTarget
from apeGmsh.opensees.emitter.py import PyEmitter
from apeGmsh.opensees.emitter.tcl import TclEmitter
from tests.opensees.h5._opensees_model_fixtures import build_two_quad_fem

_ROOT = Path(__file__).resolve().parents[3]

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
_KW = {"ManzariDafalias": _MD, "LadrunoSANISAND": _MD, "SAniSandMS": _MS}
_MD_ARGS = tuple(_MD.values())

_MESSAGE = MD_STATIC_RESET_GUARD.message
_TCL_GUARD = fork_guard_lines(MD_STATIC_RESET_GUARD, "tcl")
_PY_GUARD = fork_guard_lines(MD_STATIC_RESET_GUARD, "py")

#: The fork build measured to reset (the maintainer's venv build).
_STALE_FORK = "79e0623670c61c768357a01dff1477a4a0ef7e06"

#: The capability the guard asks for.
_CAP = "mdStageFlagPreserved"


def schema_json(
    version: int = 1, caps: "tuple[str, ...] | None" = (_CAP,),
    build: str = "6050d78e105bd4fc3a15ddb8ee521aa861a4dafe", *,
    indent: "int | None" = None,
) -> str:
    """A ``ladrunoSchema()`` answer as the fork renders it: one JSON
    object, keys sorted, compact (``caps=None``: no ``capabilities`` key,
    the v0 shape)."""
    import json

    obj: dict[str, Any] = {
        "build": build, "commands": ["ladrunoBuild", "ladrunoSchema"],
        "engine": "dl", "options": {"x": {"-y": {"nargs": 1,
                                                "since": "PR-1"}}},
        "schema_version": version}
    if caps is not None:
        obj["capabilities"] = list(caps)
    return json.dumps(obj, sort_keys=True, indent=indent,
                      separators=None if indent else (",", ":"))


#: ``ladrunoSchema`` answers of a fork build (one with ``ladrunoBuild``),
#: and whether the guard lets the deck run on.  Stock (no
#: ``ladrunoBuild``) and a fork with no ``ladrunoSchema`` are cases of
#: their own.
SCHEMA_CASES: "dict[str, tuple[str, bool]]" = {
    "v1 lists it": (schema_json(), True),
    "v1 lists it, indented": (schema_json(indent=2), True),
    "v1 lists it among others": (
        schema_json(caps=("aaa", _CAP, "zzz")), True),
    "v1 lists it last": (schema_json(caps=("aaa", _CAP)), True),
    "v1 empty list": (schema_json(caps=()), False),
    "v1 a longer name": (schema_json(caps=(_CAP + "X",)), False),
    "v1 a prefixed name": (schema_json(caps=("X" + _CAP,)), False),
    "v0": (schema_json(version=0, caps=None), False),
    "v2 lists it": (schema_json(version=2), False),
    "v10 lists it": (schema_json(version=10), False),
    # A JSON true is not the integer 1, and a string is not the list.
    "version true": (schema_json().replace(
        '"schema_version":1', '"schema_version":true'), False),
    "capabilities a string": (schema_json().replace(
        f'["{_CAP}"]', f'"{_CAP}"'), False),
    "not JSON": ("not json", False),
    "empty": ("", False),
}


def _pin_sha() -> str:
    for line in (_ROOT / "FORK_PIN").read_text(encoding="utf-8").splitlines():
        if line.startswith("sha="):
            return line.split("=", 1)[1].strip()
    raise AssertionError("FORK_PIN has no sha= line")


def _chain(ops: apeSees) -> "dict[str, Any]":
    return dict(
        test=ops.test.NormUnbalance(tol=1e-6, max_iter=50),
        algorithm=ops.algorithm.Newton(),
        integrator=ops.integrator.LoadControl(dlam=0.1),
        constraints=ops.constraints.Plain(),
        numberer=ops.numberer.RCM(),
        system=ops.system.UmfPack(),
        analysis=ops.analysis.Static(),
    )


def _bridge(
    cls: str = "ManzariDafalias", mode: "str | None" = None, *,
    first: int = 1, reflip: bool = False, activate_first: bool = False,
    fill: str = "own",
) -> apeSees:
    """``Rock`` (material a) is global; ``Fill`` is activated by stage
    ``install``.  Stage ``plastic`` flips the group to ``first`` (the
    materials live then); with ``activate_first`` it runs after
    ``install``.  ``fill="own"``: Fill carries material b, of the same
    class.  ``fill="elastic"``: it carries an ElasticIsotropic.  A carrier
    of the group activated after a plastic flip, with no flip of the group
    in its stage, is refused at stage close (#2445, #2534), so a guarded
    deck builds an element of another material there: the deck's watch
    counts every element, whatever it carries (fail closed)."""
    target = None if mode is None else OpenSeesTarget(mode=mode)  # type: ignore[arg-type]
    ops = apeSees(build_two_quad_fem(), default_orientation=None,
                  opensees=target)
    ops.model(ndm=2, ndf=2)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mats = [getattr(ops.nDMaterial, cls)(name=n, **_KW[cls])
                for n in ("a", "b")[:2 if fill == "own" else 1]]
        carried = (mats[1] if fill == "own" else
                   ops.nDMaterial.ElasticIsotropic(name="fill", E=1e5,
                                                   nu=0.3))
    ops.element.FourNodeQuad(pg="Rock", thickness=1.0, material=mats[0])
    ops.element.FourNodeQuad(pg="Fill", thickness=1.0, material=carried)
    ops.fix(pg="Base", dofs=(1, 1))

    def plastic() -> None:
        with ops.stage(name="plastic") as s:
            # Name every material live at the flip (validator V7 and the
            # class-wide check): b only once Fill is in.
            live = mats if activate_first else mats[:1]
            s.update_material_stage(materials=live, stage=first)
            s.analysis(**_chain(ops))
            s.run(n_increments=1)

    def install() -> None:
        with ops.stage(name="install") as s:
            s.activate(pgs=["Fill"])
            if reflip:
                s.update_material_stage(materials=mats, stage=first)
            s.analysis(**_chain(ops))
            s.run(n_increments=1)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if activate_first:
            install()
            plastic()
        else:
            plastic()
            install()
    return ops


def _tcl(ops: apeSees) -> list[str]:
    em = TclEmitter()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ops.build().emit(em)
    return em.lines()


def _py(ops: apeSees) -> list[str]:
    em = PyEmitter()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ops.build().emit(em)
    return em.lines()


def _guards(lines: "list[str]") -> int:
    return lines.count(_TCL_GUARD[0]) + lines.count(_PY_GUARD[0])


# ---------------------------------------------------------------------------
# The guard asks for the capability, not for a sha
# ---------------------------------------------------------------------------


def test_the_guard_asks_for_the_capability() -> None:
    assert MD_STATIC_RESET_GUARD == ForkBuildCapability(
        capability=_CAP, message=MD_STATIC_RESET_MESSAGE)
    for lines in (_TCL_GUARD, _PY_GUARD):
        text = "\n".join(lines)
        assert "ladrunoSchema" in text and _CAP in text
        assert _pin_sha() not in text and _STALE_FORK not in text


def test_the_message_names_the_flag_and_the_fix() -> None:
    for name in ("ManzariDafalias::mElastFlag", "LadrunoSANISAND",
                 "43b235bbb", "#2444", "ladrunoSchema", _CAP, "FORK_PIN"):
        assert name in _MESSAGE, name


def test_a_malformed_guard_raises_at_emit() -> None:
    bad_sha = stage_emit.ForkBuildAllowlist(builds=("bdb69cea5",),
                                           message="m")
    bad_msg = stage_emit.ForkBuildAllowlist(builds=(_STALE_FORK,),
                                           message="m [x]")
    cases: "list[tuple[Any, str]]" = [
        (bad_sha, "40-hex"), (bad_msg, "Tcl syntax"),
        (ForkBuildCapability(capability=_CAP, message="m $x"),
         "Tcl syntax")]
    for name in ("md stage", 'md"x', "md]", "1md", ""):
        cases.append((ForkBuildCapability(capability=name, message="m"),
                      "capability name"))
    for guard, match in cases:
        for dialect in ("py", "tcl"):
            with pytest.raises(BridgeError, match=match):
                fork_guard_lines(guard, dialect)  # type: ignore[arg-type]


def test_an_unknown_guard_kind_raises() -> None:
    with pytest.raises(BridgeError, match="not a fork build guard"):
        fork_guard_lines(object(), "tcl")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# What the guard does, on stub engines
# ---------------------------------------------------------------------------


def _tcl_interp() -> Any:
    tkinter = pytest.importorskip("tkinter")
    return tkinter, tkinter.Tcl()


def tcl_guard_passes(tkinter: Any, guard: str, schema: "str | None", *,
                     build: str = "6050d78e105bd4fc3a15ddb8ee521aa861a4dafe",
                     raises: bool = False) -> bool:
    """Run the Tcl ``guard`` in a fresh interpreter whose stub fork answers
    ``ladrunoBuild`` with ``build`` and ``ladrunoSchema`` with ``schema``
    (``None``: no such command; ``raises``: the command errors).  True
    when it runs on; a stop must carry the message."""
    tcl = tkinter.Tcl()
    tcl.eval(f"proc ladrunoBuild {{}} {{ return {{{build}}} }}")
    if raises:
        tcl.eval("proc ladrunoSchema {} { error boom }")
    elif schema is not None:
        tcl.eval(f"proc ladrunoSchema {{}} {{ return {{{schema}}} }}")
    try:
        tcl.eval(guard)
    except tkinter.TclError as err:
        assert str(err) == _MESSAGE
        return False
    return True


def py_guard_passes(code: Any, schema: "str | None", *,
                    raises: bool = False) -> bool:
    """:func:`tcl_guard_passes` for the compiled py guard ``code``."""
    def boom() -> str:
        raise RuntimeError("boom")

    attrs: dict[str, Any] = {"ladrunoBuild": lambda: _pin_sha()}
    if raises:
        attrs["ladrunoSchema"] = boom
    elif schema is not None:
        attrs["ladrunoSchema"] = lambda: schema
    try:
        exec(code, {"ops": types.SimpleNamespace(**attrs)})
    except SystemExit as err:
        assert err.code == _MESSAGE
        return False
    return True


def test_the_tcl_guard_on_stub_builds() -> None:
    """The key present: the deck runs on.  Absent (v0, no
    ``ladrunoSchema``, a failing call, another version, a near name):
    it stops, always with the one message."""
    tkinter, tcl = _tcl_interp()
    guard = "\n".join(_TCL_GUARD)
    tcl.eval(guard)  # stock: no ladrunoBuild, runs on
    for name, (schema, ok) in SCHEMA_CASES.items():
        assert tcl_guard_passes(tkinter, guard, schema) is ok, name
    # The sha is not the question: the stale build listing the key runs,
    # the pin's sha with a v0 schema stops.
    assert tcl_guard_passes(tkinter, guard, schema_json(),
                            build=_STALE_FORK)
    assert not tcl_guard_passes(tkinter, guard,
                                schema_json(version=0, caps=None),
                                build=_pin_sha())
    assert not tcl_guard_passes(tkinter, guard, None)
    assert not tcl_guard_passes(tkinter, guard, None, raises=True)


def test_the_py_guard_on_stub_builds() -> None:
    code = compile("\n".join(_PY_GUARD), "<guard>", "exec")
    exec(code, {"ops": types.SimpleNamespace()})  # stock runs on
    for name, (schema, ok) in SCHEMA_CASES.items():
        assert py_guard_passes(code, schema) is ok, name
    assert not py_guard_passes(code, None)
    assert not py_guard_passes(code, None, raises=True)


# ---------------------------------------------------------------------------
# Where the guard goes, and when
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("cls", ["ManzariDafalias", "LadrunoSANISAND"])
@pytest.mark.parametrize("mode", ["fork", "stock", "auto", None])
def test_the_decks_check_before_their_first_command(
    cls: str, mode: "str | None",
) -> None:
    """Every target mode: an ``"auto"`` deck is guarded as a fork one."""
    tcl = _tcl(_bridge(cls, mode, fill="elastic"))
    at = next(i for i, ln in enumerate(tcl) if not ln.startswith("#"))
    assert tcl[at:at + len(_TCL_GUARD)] == _TCL_GUARD
    assert tcl[at + len(_TCL_GUARD)].startswith("model ")
    py = _py(_bridge(cls, mode, fill="elastic"))
    at = py.index("import openseespy.opensees as ops")
    assert py[at + 1:at + 1 + len(_PY_GUARD)] == _PY_GUARD
    assert py[at + 1 + len(_PY_GUARD)] == "ops.wipe()"
    assert _guards(tcl) == _guards(py) == 1


@pytest.mark.parametrize("kw", [
    dict(cls="SAniSandMS"),            # its constructor keeps the flag
    dict(first=0),                     # an elastic flip loses nothing
    dict(reflip=True),                 # the stage flips the group again
    dict(activate_first=True),         # nothing is built after the flip
], ids=["SAniSandMS", "elastic", "reflip", "flip-last"])
def test_no_guard_where_every_build_agrees(kw: "dict[str, Any]") -> None:
    ops = _bridge(mode="fork", **kw)
    for lines in (_tcl(ops), _py(ops)):
        assert _guards(lines) == 0
        assert not any("ladrunoBuild" in ln for ln in lines)


def _without(lines: "list[str]", guard: "list[str]") -> "list[str]":
    at = lines.index(guard[0])
    assert lines[at:at + len(guard)] == guard
    return lines[:at] + lines[at + len(guard):]


def test_the_guard_is_the_only_difference(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tcl, py = _tcl(_bridge(fill="elastic")), _py(_bridge(fill="elastic"))
    monkeypatch.setattr(stage_emit.MaterialStageCarriers,
                        "solve_fork_guards", lambda self: ())
    assert _without(tcl, _TCL_GUARD) == _tcl(_bridge(fill="elastic"))
    assert _without(py, _PY_GUARD) == _py(_bridge(fill="elastic"))


def test_the_archive_replay_carries_the_guard(tmp_path: Path) -> None:
    """Derived from the emit calls: the deck replayed from ``model.h5``
    carries it as the bridge's deck does."""
    from apeGmsh.opensees import OpenSeesModel

    ops = _bridge(mode="auto", fill="elastic")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ops.h5(str(tmp_path / "a.h5"))
        deck = tmp_path / "a.tcl"
        ops.tcl(str(deck), flat=True, progress=False)
        replay = OpenSeesModel.from_h5(str(tmp_path / "a.h5")).build(
            "tcl", flat=True)
    assert isinstance(replay, str)
    assert replay.splitlines() == deck.read_text(encoding="utf-8").splitlines()
    assert replay.splitlines().count(_TCL_GUARD[0]) == 1


# -- Emitter level: ranks, streaming, the parameter route -------------------


def _flip_then_build(em: Any, *, rank: "int | None" = None,
                     solve: bool = True) -> None:
    em.model(ndm=2, ndf=2)
    em.nDMaterial("ManzariDafalias", 1, *_MD_ARGS)
    em.element("quad", 1, 1, 2, 3, 4, 1.0, "PlaneStrain", 1)
    em.update_material_stage(1, 1)
    if rank is not None:
        em.partition_open(rank)
    em.element("quad", 2, 5, 6, 7, 8, 1.0, "PlaneStrain", 1)
    if rank is not None:
        em.partition_close()
    if solve:
        em.analyze(steps=1)


@pytest.mark.parametrize("rank", [None, 0, 1])
def test_a_rank_block_construction_guards_the_whole_deck(
    rank: "int | None",
) -> None:
    """The guard goes before every rank block, so every rank runs it, and
    the per-rank spans move down with it (#2504's rule)."""
    em = TclEmitter()
    _flip_then_build(em, rank=rank)
    lines = em.lines()
    assert lines[1:1 + len(_TCL_GUARD)] == _TCL_GUARD
    assert _guards(lines) == 1
    for span in em.partition_spans():
        assert lines[span.header] == f"if {{[getPID] == {span.rank}}} {{"
        assert any(ln.startswith("    element quad 2")
                   for ln in lines[span.body_start:span.body_end])
    py = PyEmitter()
    _flip_then_build(py, rank=rank)
    assert _guards(py.lines()) == 1
    at = py.lines().index("import openseespy.opensees as ops")
    assert py.lines()[at + 1] == _PY_GUARD[0]


def test_a_streamed_deck_checks_before_the_solve(tmp_path: Path) -> None:
    em = TclEmitter()
    path = tmp_path / "s.tcl"
    em.stream_to(str(path))
    _flip_then_build(em)
    em.stream_finish()
    lines = path.read_text(encoding="utf-8").splitlines()
    at = lines.index(_TCL_GUARD[0])
    assert lines[at:at + len(_TCL_GUARD)] == _TCL_GUARD
    assert any(ln.startswith("element quad 2") for ln in lines[:at])
    assert not any("analyze" in ln for ln in lines[:at]
                   if not ln.startswith("#"))


def _quad(em: Any, tag: int, *, rank: "int | None" = None) -> None:
    if rank is not None:
        em.partition_open(rank)
    em.element("quad", tag, 1, 2, 3, 4, 1.0, "PlaneStrain", 1)
    if rank is not None:
        em.partition_close()


def test_a_flip_on_another_rank_does_not_heal() -> None:
    """The static is per process: a re-flip on rank 0 does not set it again
    on rank 1, where an element was built after the flip (the review's
    F4 probe)."""
    for em in (TclEmitter(), PyEmitter()):
        em.model(ndm=2, ndf=2)
        em.nDMaterial("ManzariDafalias", 1, *_MD_ARGS)
        _quad(em, 1)
        em.update_material_stage(1, 1)
        _quad(em, 10, rank=1)
        em.partition_open(0)
        em.update_parameter(9, (1,), ("materialState", 1), 1.0)
        em.partition_close()
        em.analyze(steps=1)
        assert _guards(em.lines()) == 1, type(em).__name__


@pytest.mark.parametrize("flip_rank", [1, None])
def test_a_flip_on_the_same_rank_or_every_rank_heals(
    flip_rank: "int | None",
) -> None:
    for em in (TclEmitter(), PyEmitter()):
        em.model(ndm=2, ndf=2)
        em.nDMaterial("ManzariDafalias", 1, *_MD_ARGS)
        _quad(em, 1)
        em.update_material_stage(1, 1)
        _quad(em, 10, rank=1)
        if flip_rank is not None:
            em.partition_open(flip_rank)
        em.update_parameter(9, (1,), ("materialState", 1), 1.0)
        if flip_rank is not None:
            em.partition_close()
        em.analyze(steps=1)
        assert _guards(em.lines()) == 0, type(em).__name__


def test_a_rank_flip_never_heals_a_construction_on_every_rank() -> None:
    em = TclEmitter()
    em.model(ndm=2, ndf=2)
    em.nDMaterial("ManzariDafalias", 1, *_MD_ARGS)
    _quad(em, 1)
    em.update_material_stage(1, 1)
    _quad(em, 10)                      # every rank builds it
    em.partition_open(0)
    em.update_material_stage(1, 1)     # rank 0 sets it again; rank 1 not
    em.partition_close()
    em.analyze(steps=1)
    assert _guards(em.lines()) == 1


def test_no_solve_no_guard_and_a_flip_heals() -> None:
    for em in (TclEmitter(), PyEmitter()):
        _flip_then_build(em, solve=False)
        assert _guards(em.lines()) == 0
        em.update_material_stage(1, 1)  # sets the flag again: heals
        em.analyze(steps=1)
        assert _guards(em.lines()) == 0


@pytest.mark.parametrize("args", [("materialState", 1), ("materialState",)])
def test_the_parameter_route_flips_too(args: "tuple[Any, ...]") -> None:
    """A stage-flag ``updateParameter`` sets the flag as
    ``updateMaterialStage`` does; one with no material tag may set any."""
    for em in (TclEmitter(), PyEmitter()):
        em.model(ndm=2, ndf=2)
        em.nDMaterial("ManzariDafalias", 1, *_MD_ARGS)
        em.element("quad", 1, 1, 2, 3, 4, 1.0, "PlaneStrain", 1)
        em.update_parameter(9, (1,), args, 1.0)
        em.element("quad", 2, 5, 6, 7, 8, 1.0, "PlaneStrain", 1)
        em.eigen(1)
        assert _guards(em.lines()) == 1


def test_a_new_material_after_the_flip_counts() -> None:
    """The ``nDMaterial`` command runs the constructor too."""
    em = PyEmitter()
    em.model(ndm=2, ndf=2)
    em.nDMaterial("ManzariDafalias", 1, *_MD_ARGS)
    em.element("quad", 1, 1, 2, 3, 4, 1.0, "PlaneStrain", 1)
    em.update_material_stage(1, 1)
    em.nDMaterial("ManzariDafalias", 2, *_MD_ARGS)
    em.analyze(steps=1)
    assert _guards(em.lines()) == 1


# ---------------------------------------------------------------------------
# Live: the bridge's py deck on the bound build, in a child process
# ---------------------------------------------------------------------------

_PROBE = """\
import json
{import_line}
def _ask(name):
    if not hasattr(ops, name):
        return None
    try:
        return str(getattr(ops, name)())
    except Exception as exc:
        return "RAISED " + repr(exc)
print("MD_PROBE " + json.dumps([_ask("ladrunoBuild"), _ask("ladrunoSchema")]),
      flush=True)
"""


def lists_the_capability(schema: "str | None") -> bool:
    """Whether a ``ladrunoSchema()`` answer is a v1 schema listing the
    capability: the guard's question, read here without its code."""
    import json

    try:
        obj = json.loads(schema or "")
    except ValueError:
        return False
    return (isinstance(obj, dict) and obj.get("schema_version") == 1
            and obj.get("schema_version") is not True
            and isinstance(obj.get("capabilities"), list)
            and _CAP in obj["capabilities"])

_FORK_IMPORT = (
    "from apeGmsh.opensees.emitter.live import get_ops\nops = get_ops()")
_STOCK_IMPORT = "import openseespy.opensees as ops"


def _env() -> "dict[str, str]":
    src = str(Path(apeGmsh.__file__).resolve().parents[1])
    env = dict(os.environ, LADRUNO_OPENSEES_QUIET="1")
    env["PYTHONPATH"] = os.pathsep.join(
        [src, *filter(None, [os.environ.get("PYTHONPATH")])])
    return env


def _run_py_deck(tmp_path: Path, *, fork_module: bool) -> None:
    deck = tmp_path / "deck.py"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        _bridge(mode="auto", fill="elastic").py(str(deck))
    run_guarded_head(deck.read_text(encoding="utf-8"), tmp_path,
                     fork_module=fork_module)


def run_guarded_head(text: str, tmp_path: Path, *, fork_module: bool) -> None:
    """Run the guarded py deck ``text`` up to its first command and check
    the verdict against the build the child bound: the guard is the
    question, not the analysis after it.  The replay tests run it too."""
    import json

    deck = tmp_path / "deck.py"
    assert _STOCK_IMPORT in text and _PY_GUARD[0] in text
    import_line = _FORK_IMPORT if fork_module else _STOCK_IMPORT
    head, _, _rest = text.partition("ops.wipe()")
    body = head.replace(_STOCK_IMPORT, import_line)
    deck.write_text(_PROBE.format(import_line=import_line) + body
                    + 'print("GUARD_PASSED", flush=True)\n', encoding="utf-8")
    proc = subprocess.run([sys.executable, str(deck)], cwd=tmp_path,
                          env=_env(), capture_output=True, text=True,
                          check=False)
    out = proc.stdout + proc.stderr
    line = next((ln for ln in proc.stdout.splitlines()
                 if ln.startswith("MD_PROBE ")), None)
    assert line is not None, out[-2000:]
    build, schema = json.loads(line.split(" ", 1)[1])
    accepted = build is None or lists_the_capability(schema)
    # Anchors that do not trust the schema: the pin (v1, fork PR #1017)
    # runs, 79e06 (no capabilities) stops.
    if build is not None and build.strip() == _pin_sha():
        assert accepted, schema
    if build is not None and build.strip() == _STALE_FORK:
        assert not accepted, schema
    if accepted:
        assert proc.returncode == 0 and "GUARD_PASSED" in proc.stdout, out
        assert _MESSAGE not in out
    else:
        assert proc.returncode != 0, out[-2000:]
        assert _MESSAGE in proc.stderr, out[-2000:]
        assert "GUARD_PASSED" not in proc.stdout


@pytest.mark.live
def test_the_py_deck_on_the_bound_openseespy(tmp_path: Path) -> None:
    """``live-stock``: stock openseespy has no ladrunoBuild and runs on."""
    _run_py_deck(tmp_path, fork_module=False)


@pytest.mark.live
@pytest.mark.ladruno_fork
def test_the_py_deck_on_the_fork(tmp_path: Path) -> None:
    """``live-fork``: at FORK_PIN, whose ``ladrunoSchema`` lists
    ``mdStageFlagPreserved``, the deck runs; on the stale fork build (the
    local 79e062367) it stops by name."""
    _run_py_deck(tmp_path, fork_module=True)
