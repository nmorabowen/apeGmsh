"""``TargetCaps`` (ADR 0114 D6, K1-5): every emitter declares its capabilities once.

Each emitter's ``caps`` must say what its class attributes and construction
defaults said before the field existed, so the bridge can read
``emitter.caps.<field>`` where it used to probe ``getattr(emitter, "<flag>",
default)`` and sniff ``type(emitter).__name__``. ``TargetCaps()`` itself
carries the former probe defaults.
"""
from __future__ import annotations

import dataclasses

import pytest

from apeGmsh.opensees.emitter.base import Emitter
from apeGmsh.opensees.emitter.caps import SolveStamp, TargetCaps
from apeGmsh.opensees.emitter.h5 import H5Emitter
from apeGmsh.opensees.emitter.live import LiveOpsEmitter
from apeGmsh.opensees.emitter.py import PyEmitter
from apeGmsh.opensees.emitter.recording import RecordingEmitter
from apeGmsh.opensees.emitter.tcl import TclEmitter

EMITTERS = (TclEmitter, PyEmitter, LiveOpsEmitter, H5Emitter, RecordingEmitter)

FIELDS = (
    "archival", "supports_partitions", "per_rank_fragments",
    "suppress_analysis_chain_auto_emit", "model_reissue_purges",
    "emit_stage_markers",
)


def test_target_caps_has_exactly_the_six_fields() -> None:
    assert tuple(f.name for f in dataclasses.fields(TargetCaps)) == FIELDS


def test_target_caps_defaults_are_the_former_getattr_defaults() -> None:
    # ``getattr(emitter, "supports_partitions", True)``; every other probe
    # defaulted to False, and nothing was archival unless it said so.
    assert TargetCaps() == TargetCaps(
        archival=False, supports_partitions=True, per_rank_fragments=False,
        suppress_analysis_chain_auto_emit=False, model_reissue_purges=False,
        emit_stage_markers=False,
    )


def test_protocol_declares_caps() -> None:
    assert Emitter.__annotations__["caps"] in ("TargetCaps", TargetCaps)


@pytest.mark.parametrize("cls", EMITTERS, ids=lambda c: c.__name__)
def test_every_emitter_declares_a_frozen_target_caps(cls: type) -> None:
    caps = cls.caps
    assert isinstance(caps, TargetCaps)
    with pytest.raises(dataclasses.FrozenInstanceError):
        caps.archival = not caps.archival  # type: ignore[misc]


@pytest.mark.parametrize("cls", EMITTERS, ids=lambda c: c.__name__)
def test_caps_match_the_class_attributes_they_replace(cls: type) -> None:
    caps = cls.caps
    assert caps.archival is (cls is H5Emitter)
    assert caps.supports_partitions is getattr(cls, "supports_partitions", True)
    assert caps.model_reissue_purges is getattr(cls, "model_reissue_purges", False)
    # Never declared at class level: the bridge set them per emit.
    assert not hasattr(cls, "per_rank_fragments")
    assert not hasattr(cls, "suppress_analysis_chain_auto_emit")
    assert caps.per_rank_fragments is False
    assert caps.suppress_analysis_chain_auto_emit is False
    assert caps.emit_stage_markers is False


@pytest.mark.parametrize("cls", (TclEmitter, PyEmitter), ids=lambda c: c.__name__)
def test_deck_emitters_start_without_stage_markers(cls: type) -> None:
    # ``_emit_stage_markers`` is the per-instance flag ``caps.emit_stage_markers``
    # types; a fresh deck emitter has it off.
    assert cls()._emit_stage_markers is cls.caps.emit_stage_markers is False


def test_caps_values_per_emitter() -> None:
    assert TclEmitter.caps == TargetCaps(model_reissue_purges=True)
    assert PyEmitter.caps == TargetCaps()
    assert RecordingEmitter.caps == TargetCaps()
    assert LiveOpsEmitter.caps == TargetCaps(supports_partitions=False)
    assert H5Emitter.caps == TargetCaps(archival=True)


# -- SolveStamp -----------------------------------------------------------


def test_solve_stamp_defaults_to_no_refusals_and_no_requirements() -> None:
    stamp = SolveStamp(will_solve=True)
    assert stamp == SolveStamp(True, (), ())


def test_solve_stamp_requires_must_be_sorted_and_unique() -> None:
    with pytest.raises(ValueError, match="sorted and unique"):
        SolveStamp(True, requires=("fork", "fork"))
    with pytest.raises(ValueError, match="sorted and unique"):
        SolveStamp(True, requires=("mp", "fork"))
    assert SolveStamp(True, requires=("fork", "mp")).requires == ("fork", "mp")


@pytest.mark.parametrize("bad", [("", ), ("fork", 3), ["fork"], "fork"])
def test_solve_stamp_tokens_must_be_non_empty_strings(bad: object) -> None:
    with pytest.raises(TypeError, match="tuple of non-empty strings"):
        SolveStamp(True, solve_refusals=bad)  # type: ignore[arg-type]
