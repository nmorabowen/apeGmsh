"""Replay emits materials in the bridge's dependency order (#1814).

``OpenSeesModel.from_h5(...).build(...)`` used to emit the materials in
the archive's group order (``/opensees/materials/<family>/{type}_{tag}``,
which HDF5 iterates by name), so ``uniaxialMaterial Parallel 3 1 2``
came before ``Steel01 1`` and the deck aborted at that line with "no
existing material with tag 1".

The oracle is the bridge's own deck: the replay writes the material
lines in the order the bridge wrote them (its ``topological_order``,
recorded in ``/opensees/program``), so each wrapper follows the
materials it references. ``Series`` is not a bridge primitive; the
wrappers covered are ``Parallel`` and ``InitialStressMaterial``
(uniaxial) and ``PlaneStrain`` / ``PlateFiber`` (nD), nested.
"""
from __future__ import annotations

import dataclasses
import importlib.util
import re
import subprocess
import sys
import warnings
from pathlib import Path
from typing import Any, Callable, cast

import pytest

from apeGmsh.opensees import OpenSeesModel, apeSees
from apeGmsh.opensees.material.nd import ElasticIsotropic, PlaneStrain
from apeGmsh.opensees.material.uniaxial import Parallel, Steel01, Viscous

from tests.opensees.h5._opensees_model_fixtures import build_simple_frame_fem
from tests.opensees.h5.test_h5_decl_params import _broad_frame


def _bridge_and_replay(
    ops: apeSees, tmp_path: Path, target: str = "tcl",
) -> tuple[str, str]:
    deck = tmp_path / f"bridge.{'tcl' if target == 'tcl' else 'py'}"
    archive = tmp_path / "model.h5"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # auto-emit / ledger advisories
        if target == "tcl":
            ops.tcl(str(deck), progress=False, flat=True)
        else:
            ops.py(str(deck), progress=False)
        ops.h5(str(archive))
        replay = OpenSeesModel.from_h5(str(archive)).build(target)
    assert isinstance(replay, str)
    return deck.read_text(encoding="utf-8"), replay


def _norm(line: str) -> tuple[Any, ...]:
    """A material line as values: ``1`` and ``1.0`` read alike (the
    archive stores the params as floats, the bridge deck as written)."""
    out: list[Any] = []
    for a in re.findall(r"'[^']*'|[^\s(),]+", line):
        a = a.strip("'")
        try:
            out.append(float(a))
        except ValueError:
            out.append(a)
    return tuple(out)


def _material_lines(deck: str) -> list[str]:
    heads = ("uniaxialMaterial", "nDMaterial",
             "ops.uniaxialMaterial", "ops.nDMaterial")
    return [ln.strip() for ln in deck.splitlines()
            if ln.strip().startswith(heads)]


# The tag arguments each wrapper references, from its Tcl line's tail.
def _refs(token: str, tail: list[str]) -> list[int]:
    if token == "Parallel":
        out: list[int] = []
        for a in tail:
            if a.startswith("-"):
                break
            out.append(int(a))
        return out
    if token in ("InitialStressMaterial", "PlaneStrain", "PlateFiber"):
        return [int(tail[0])]
    return []


def _assert_components_first(deck: str) -> None:
    """Every wrapper line follows the lines declaring the tags it names."""
    declared: dict[str, set[int]] = {"uniaxialMaterial": set(), "nDMaterial": set()}
    for ln in _material_lines(deck):
        verb, token, tag, *tail = ln.split()
        refs = _refs(token, tail)
        # PlaneStrain / PlateFiber wrap an nD; the uniaxial ones a uniaxial.
        missing = [r for r in refs if r not in declared[verb]]
        assert not missing, f"{ln!r} names undeclared {verb} {missing}"
        declared[verb].add(int(tag))


def _frame(ops: apeSees) -> apeSees:
    """Finish ``ops`` into the one-column force-based frame."""
    transf = ops.geomTransf.Linear(vecxz=(1.0, 0.0, 0.0))
    sec = ops.section.Elastic(E=200e9, A=0.01, Iz=1e-4, Iy=1e-4, G=80e9, J=1e-4)
    integ = ops.beamIntegration.Lobatto(section=sec, n_ip=3)
    ops.element.forceBeamColumn(pg="Cols", transf=transf, integration=integ)
    ops.fix(nodes=[1], dofs=(1,) * 6)
    return ops


def _new() -> apeSees:
    ops = apeSees(cast("Any", build_simple_frame_fem()))
    ops.model(ndm=3, ndf=6)
    return ops


def _nested() -> apeSees:
    """Wrappers of wrappers: Parallel(Parallel(Steel01, Viscous), Steel01)."""
    ops = _new()
    steel = ops.uniaxialMaterial.Steel01(fy=420e6, E=200e9, b=0.01)
    visc = ops.uniaxialMaterial.Viscous(C=1.0e5)
    inner = ops.uniaxialMaterial.Parallel(materials=(steel, visc))
    ops.uniaxialMaterial.Parallel(materials=(inner, steel), factors=(1.0, 2.0))
    return _frame(ops)


def _initial_stress() -> apeSees:
    """A second wrapper type, nested: Parallel(InitialStress(Steel01), Viscous).

    Not in the live set: stock openseespy builds may lack
    ``InitialStressMaterial``."""
    ops = _new()
    steel = ops.uniaxialMaterial.Steel01(fy=420e6, E=200e9, b=0.01)
    visc = ops.uniaxialMaterial.Viscous(C=1.0e5)
    pre = ops.uniaxialMaterial.InitialStress(base_material=steel, sigma_init=1.0e6)
    ops.uniaxialMaterial.Parallel(materials=(pre, visc))
    return _frame(ops)


def _wrapper_registered_first() -> apeSees:
    """A wrapper registered before its members (P11 ``register``), so its
    tag is LOWER than theirs: tag order is not dependency order here."""
    ops = _new()
    steel = Steel01(fy=420e6, E=200e9, b=0.01)
    visc = Viscous(C=1.0e5)
    ops.register(Parallel(materials=(steel, visc), factors=None))
    ops.register(steel)
    ops.register(visc)
    # nD likewise: the plane-strain wrapper before its 3-D base.
    base = ElasticIsotropic(E=30e9, nu=0.2)
    ops.register(PlaneStrain(base=base))
    ops.register(base)
    return _frame(ops)


_LIVE: tuple[tuple[str, Callable[[], apeSees]], ...] = (
    ("broad_frame", _broad_frame),
    ("nested", _nested),
    ("wrapper_registered_first", _wrapper_registered_first),
)
_BUILDS = _LIVE + (("initial_stress", _initial_stress),)


@pytest.mark.parametrize("target", ["tcl", "py"])
@pytest.mark.parametrize("build", [b for _, b in _BUILDS], ids=[n for n, _ in _BUILDS])
def test_replay_writes_the_bridge_material_order(
    tmp_path: Path, build: Callable[[], apeSees], target: str,
) -> None:
    """Family by family: the replay writes every uniaxial before every nD
    (``_replay_into``'s category order) where the bridge may interleave
    them, which no reference between the two families can break."""
    bridge, replay = _bridge_and_replay(build(), tmp_path, target)
    for verb in ("uniaxialMaterial", "nDMaterial"):
        want = [_norm(ln) for ln in _material_lines(bridge) if verb in ln.split("(")[0]]
        got = [_norm(ln) for ln in _material_lines(replay) if verb in ln.split("(")[0]]
        assert got == want, verb
    assert len(_material_lines(bridge)) >= 3


@pytest.mark.parametrize("build", [b for _, b in _BUILDS], ids=[n for n, _ in _BUILDS])
def test_replay_declares_components_before_wrappers(
    tmp_path: Path, build: Callable[[], apeSees],
) -> None:
    bridge, replay = _bridge_and_replay(build(), tmp_path)
    _assert_components_first(bridge)  # the oracle's own premise
    _assert_components_first(replay)


def test_wrapper_tags_precede_members_in_the_archive(tmp_path: Path) -> None:
    """The case's premise: the wrapper holds the lower tag, so ordering
    the replay by tag (or by group name) would still abort."""
    archive = tmp_path / "model.h5"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        _wrapper_registered_first().h5(str(archive))
    om = OpenSeesModel.from_h5(str(archive))
    tag = {(r.type_token, ): int(r.tag) for r in om.materials()}
    assert tag[("Parallel",)] < min(tag[("Steel01",)], tag[("Viscous",)])
    assert tag[("PlaneStrain",)] < tag[("ElasticIsotropic",)]


def test_a_rewritten_archive_keeps_the_order(tmp_path: Path) -> None:
    """``to_h5`` echoes the program and its joins, so the rewrite's replay
    orders as the source's does."""
    archive, rewrite = tmp_path / "model.h5", tmp_path / "rewrite.h5"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        _wrapper_registered_first().h5(str(archive))
        src = OpenSeesModel.from_h5(str(archive))
        src.to_h5(str(rewrite))
        want = src.build("tcl")
        got = OpenSeesModel.from_h5(str(rewrite)).build("tcl")
    assert isinstance(want, str) and isinstance(got, str)
    assert _material_lines(got) == _material_lines(want)
    _assert_components_first(got)


def test_without_program_joins_replay_orders_by_tag(tmp_path: Path) -> None:
    """An archive that records no call→declaration joins (below opensees
    2.31.0) falls back to tag order, which is the bridge's registration
    order: components first unless a wrapper was ``register``-ed early."""
    archive = tmp_path / "model.h5"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        _nested().h5(str(archive))
        om = OpenSeesModel.from_h5(str(archive))
        old = dataclasses.replace(om, _declarations=None)
        replay = old.build("tcl")
    assert isinstance(replay, str)
    lines = _material_lines(replay)
    assert [int(ln.split()[2]) for ln in lines] == sorted(
        int(ln.split()[2]) for ln in lines)
    _assert_components_first(replay)


def test_material_accessor_keeps_the_archive_order(tmp_path: Path) -> None:
    """Only the deck replay reorders; ``materials()`` still reads the store."""
    archive = tmp_path / "model.h5"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        _broad_frame().h5(str(archive))
    om = OpenSeesModel.from_h5(str(archive))
    assert [r.type_token for r in om.materials(family="uniaxial")] == [
        "Parallel", "Parallel", "Steel01", "Viscous"]


# ---------------------------------------------------------------------------
# Live: the replayed deck runs (child interpreter, stock openseespy)
# ---------------------------------------------------------------------------

_CHILD = """
import sys
src = open(sys.argv[1], encoding="utf-8").read()
ns = {"__name__": "replayed_deck"}
exec(compile(src, sys.argv[1], "exec"), ns)
print("DECK-OK")
"""


def _child_has_openseespy() -> bool:
    return importlib.util.find_spec("openseespy") is not None


@pytest.mark.live
@pytest.mark.skipif(not _child_has_openseespy(), reason="openseespy not importable")
@pytest.mark.parametrize("build", [b for _, b in _LIVE], ids=[n for n, _ in _LIVE])
def test_replayed_deck_runs_in_opensees(
    tmp_path: Path, build: Callable[[], apeSees],
) -> None:
    """The replayed openseespy deck runs to the end (exit 0), as the
    bridge's own deck does. A child interpreter keeps the shared pytest
    process free of openseespy state."""
    bridge, replay = _bridge_and_replay(build(), tmp_path, "py")
    child = tmp_path / "child.py"
    child.write_text(_CHILD, encoding="utf-8")
    for name, text in (("bridge.py", bridge), ("replayed.py", replay)):
        deck = tmp_path / f"run_{name}"
        deck.write_text(text, encoding="utf-8")
        out = subprocess.run(
            [sys.executable, str(child), str(deck)],
            capture_output=True, text=True, check=False, timeout=300,
        )
        assert out.returncode == 0, (name, out.stdout[-1000:], out.stderr[-2000:])
        assert "DECK-OK" in out.stdout, name
