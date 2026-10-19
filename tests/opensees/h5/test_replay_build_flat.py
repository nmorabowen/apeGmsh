"""``OpenSeesModel.build(target, flat=True)`` (#1832).

A partitioned archive records the ADR 0027 INV-5 auto-chain: the
``numberer`` / ``system`` pair the partitioned emit writes as a runtime
conditional, stored with ``numberer_runtime_fallback`` /
``system_runtime_fallback`` attrs. ``apeSees.tcl(flat=True)`` never writes
it, so the replay's ``flat=True`` drops exactly those lines and nothing
else. ``flat=False`` (the default) is unchanged.

The oracle for "exactly those lines" is the emitter itself: the removed
block must equal what ``TclEmitter`` / ``PyEmitter`` write for
``parallel_runtime_fallback_numberer("ParallelPlain", "RCM")`` and
``parallel_runtime_fallback_system("Mumps", "UmfPack")``, in one run.
"""
from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any

import pytest

from apeGmsh.opensees import OpenSeesModel
from apeGmsh.opensees._internal import compose
from apeGmsh.opensees.apesees import apeSees
from apeGmsh.opensees.emitter.py import PyEmitter
from apeGmsh.opensees.emitter.recording import RecordingEmitter
from apeGmsh.opensees.emitter.tcl import TclEmitter
from tests.opensees.h5.test_h5_commands_program import _staged
from tests.opensees.h5.test_h5_partitions_roundtrip import _build_bridge


def _archive(ops: apeSees, tmp_path: Path) -> OpenSeesModel:
    path = tmp_path / "model.h5"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # the INV-5 auto-emit advisories
        ops.h5(str(path))
    return OpenSeesModel.from_h5(str(path))


def _auto_chain(kind: str) -> list[str]:
    """The INV-5 auto-chain lines, as the target's emitter writes them."""
    emitter = TclEmitter() if kind == "tcl" else PyEmitter()
    header = len(list(emitter.lines()))  # the deck preamble
    emitter.parallel_runtime_fallback_numberer("ParallelPlain", "RCM")
    emitter.parallel_runtime_fallback_system("Mumps", "UmfPack")
    return list(emitter.lines())[header:]


def _build(om: OpenSeesModel, kind: str, *, flat: bool) -> list[str]:
    deck = om.build(kind, flat=flat)
    assert isinstance(deck, str)
    return deck.splitlines()


@pytest.mark.parametrize("kind", ["tcl", "py"])
def test_flat_drops_exactly_the_recorded_auto_chain(
        kind: str, tmp_path: Path) -> None:
    om = _archive(_build_bridge(), tmp_path)
    assert om.analysis()["numberer_runtime_fallback"] == "RCM"
    assert om.analysis()["system_runtime_fallback"] == "UmfPack"
    default = _build(om, kind, flat=False)
    flat = _build(om, kind, flat=True)
    chain = _auto_chain(kind)
    assert len(chain) == (4 if kind == "tcl" else 10)
    at = default.index(chain[0])
    assert default[at:at + len(chain)] == chain
    assert default[:at] + default[at + len(chain):] == flat
    assert not any("ParallelPlain" in ln or "Mumps" in ln for ln in flat)


@pytest.mark.parametrize("kind", ["tcl", "py"])
def test_flat_keeps_a_declared_numberer_and_system(
        kind: str, tmp_path: Path) -> None:
    """A user-declared numberer/system carries no fallback attr: the flat
    deck keeps it and equals the default deck."""
    ops = _build_bridge()
    ops.numberer.RCM()
    ops.system.UmfPack()
    om = _archive(ops, tmp_path)
    attrs = om.analysis()
    assert attrs["numberer"] == "RCM" and attrs["system"] == "UmfPack"
    assert "numberer_runtime_fallback" not in attrs
    assert "system_runtime_fallback" not in attrs
    flat = _build(om, kind, flat=True)
    assert flat == _build(om, kind, flat=False)
    want = (("numberer RCM", "system UmfPack") if kind == "tcl"
            else ("ops.numberer('RCM')", "ops.system('UmfPack')"))
    assert all(w in flat for w in want), want


@pytest.mark.parametrize("partitioned", [False, True])
def test_flat_is_a_no_op_without_a_recorded_fallback(
        partitioned: bool, tmp_path: Path) -> None:
    """Staged archives, serial and partitioned, record no fallback: both
    decks are the default ones."""
    om = _archive(
        _staged(partitioned=partitioned, rayleigh=True, profile=False),
        tmp_path)
    assert len(om.partitions()) == (2 if partitioned else 0)
    for kind in ("tcl", "py"):
        assert _build(om, kind, flat=True) == _build(om, kind, flat=False)


@pytest.mark.parametrize("flat", [False, True])
def test_flat_reaches_every_chain(
        flat: bool, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The global chain and each stage's get the caller's ``flat``."""
    om = _archive(
        _staged(partitioned=True, rayleigh=False, profile=False), tmp_path)
    seen: list[bool] = []
    real = compose._replay_analysis_chain

    def spy(emitter: Any, attrs: "dict[str, Any]", *, flat: bool) -> None:
        seen.append(flat)
        real(emitter, attrs, flat=flat)

    monkeypatch.setattr(compose, "_replay_analysis_chain", spy)
    om.build("tcl", flat=flat)
    n_chains = sum(1 for s in om.stages() if s.chain_attrs)
    assert n_chains == 2
    assert seen == [flat] * (n_chains + bool(om.analysis()))


def _chain_calls(attrs: "dict[str, Any]", *, flat: bool) -> list[str]:
    emitter = RecordingEmitter()
    compose._replay_analysis_chain(emitter, attrs, flat=flat)
    return [name for name, _a, _k in emitter.calls]


def test_the_chain_replay_drops_only_fallback_attrs() -> None:
    auto = {
        "handler": "Transformation",
        "numberer": "ParallelPlain", "numberer_runtime_fallback": "RCM",
        "system": "Mumps", "system_runtime_fallback": "UmfPack",
        "test": "NormDispIncr", "test_args": (1e-8, 10),
        "algorithm": "Newton", "integrator": "LoadControl",
        "integrator_args": (0.1,), "analysis": "Static",
    }
    rest = ["constraints", "test", "algorithm", "integrator", "analysis"]
    assert _chain_calls(auto, flat=True) == rest
    assert _chain_calls(auto, flat=False) == [
        "constraints", "parallel_runtime_fallback_numberer",
        "parallel_runtime_fallback_system", *rest[1:]]
    declared = {"numberer": "RCM", "system": "UmfPack"}
    assert _chain_calls(declared, flat=True) == ["numberer", "system"]
    assert _chain_calls(declared, flat=False) == ["numberer", "system"]


@pytest.mark.parametrize("target", ["live", "h5"])
def test_flat_is_refused_off_the_decks(target: str, tmp_path: Path) -> None:
    """``flat`` means one thing: the deck targets. ``live`` runs the
    recorded conditional and ``h5`` must keep its attrs, so both refuse."""
    om = _archive(_build_bridge(), tmp_path)
    out = tmp_path / "again.h5"
    with pytest.raises(ValueError, match="flat applies to the 'tcl' and "
                       "'py' decks only"):
        om.build(target, out=out, flat=True)
    assert not out.exists()


def test_the_default_build_is_unchanged(tmp_path: Path) -> None:
    """``build(kind)`` with no ``flat`` is ``flat=False``: the recorded
    auto-chain stays in the deck."""
    om = _archive(_build_bridge(), tmp_path)
    for kind in ("tcl", "py"):
        assert _build(om, kind, flat=False) == om.build(kind).splitlines()
        assert _auto_chain(kind)[1] in om.build(kind).splitlines()
