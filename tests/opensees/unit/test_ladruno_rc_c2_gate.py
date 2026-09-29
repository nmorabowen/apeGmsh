"""Build gate for ``LadrunoRCConcrete`` ``-betaC`` / ``-crackedNu`` (#1184 hole).

Fork PR #873 carried the flags and was closed unmerged. No build of the
fork's ``ladruno`` branch parses them, and that parser ignores unknown
tokens, so a deck that carries them runs the pre-C2 law with no error.
Fork-free: drives the emitters against fake ``ops`` modules.
"""
from __future__ import annotations

import warnings

import pytest

from apeGmsh.opensees import _rc_c2_flags
from apeGmsh.opensees.emitter.live import LiveOpsEmitter
from apeGmsh.opensees.emitter.py import PyEmitter
from apeGmsh.opensees.emitter.tcl import TclEmitter
from apeGmsh.opensees.material.nd import (
    LADRUNO_RC_C2_MIN_BUILD,
    LadrunoRCBuildWarning,
    LadrunoRCConcrete,
    LadrunoRCFiniteStrain,
)

_RAW = dict(E=30e9, nu=0.2, Ce=(0.0, 1e-3), Cs=(0.0, 30e6),
            Te=(0.0, 1e-4), Ts=(0.0, 3e6))


class _Ops:
    def __init__(self) -> None:
        self.calls: list[tuple[object, ...]] = []

    def nDMaterial(self, *args: object) -> None:   # noqa: N802
        self.calls.append(args)


class _StampedOps(_Ops):
    def ladrunoBuild(self) -> str:                  # noqa: N802
        return "0" * 40


def _live(ops: object) -> LiveOpsEmitter:
    le = LiveOpsEmitter.__new__(LiveOpsEmitter)
    le._ops = ops  # type: ignore[assignment]
    le._in_partition = False
    return le


def test_no_ladruno_build_carries_the_flags_yet() -> None:
    # Flip this (and the live test) when fork PR #877 merges.
    assert LADRUNO_RC_C2_MIN_BUILD is None


@pytest.mark.parametrize("cls", [LadrunoRCConcrete, LadrunoRCFiniteStrain])
@pytest.mark.parametrize("kw", [{"beta_c": 189.0}, {"cracked_nu": 0.0}])
def test_live_refuses_c2_flags_before_the_call(cls: type, kw: dict) -> None:
    for ops in (_Ops(), _StampedOps()):   # no floor: even a stamped build
        with pytest.raises(RuntimeError, match="silently discards"):
            cls(**_RAW, **kw)._emit(_live(ops), tag=1)
        assert ops.calls == []


def test_live_runs_without_c2_flags() -> None:
    ops = _Ops()
    LadrunoRCConcrete(**_RAW, tens_stiff="vc",  # type: ignore[arg-type]
                      tens_stiff_c=500.0)._emit(_live(ops), tag=1)
    assert len(ops.calls) == 1


def test_floor_set_admits_a_stamped_build_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(_rc_c2_flags, "LADRUNO_RC_C2_MIN_BUILD", "abc1234")
    mat = LadrunoRCConcrete(**_RAW, cracked_nu=0.0)  # type: ignore[arg-type]
    with pytest.raises(RuntimeError, match="abc1234"):
        mat._emit(_live(_Ops()), tag=1)   # stock / pre-#718: no stamp
    ops = _StampedOps()
    mat._emit(_live(ops), tag=1)
    assert len(ops.calls) == 1


@pytest.mark.parametrize("emitter_cls", [TclEmitter, PyEmitter])
def test_deck_emitters_warn_and_still_emit(emitter_cls: type) -> None:
    em = emitter_cls()
    with pytest.warns(LadrunoRCBuildWarning, match="-betaC -crackedNu"):
        LadrunoRCConcrete(**_RAW, beta_c=189.0,  # type: ignore[arg-type]
                          cracked_nu=0.0)._emit(em, tag=1)
    line = em.lines()[-1]
    assert "-betaC" in line and "-crackedNu" in line


@pytest.mark.parametrize("emitter_cls", [TclEmitter, PyEmitter])
def test_deck_emitters_silent_without_c2_flags(emitter_cls: type) -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("error", LadrunoRCBuildWarning)
        LadrunoRCConcrete(**_RAW)._emit(  # type: ignore[arg-type]
            emitter_cls(), tag=1)
