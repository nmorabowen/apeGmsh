"""``TargetCaps`` (ADR 0114 D6, K1-5): every emitter declares its capabilities once.

Each emitter's ``caps`` says what its class attributes and construction
defaults said before the field existed, and those attributes are gone:
the bridge reads ``emitter.caps.<field>`` where it used to probe
``getattr(emitter, "<flag>", default)``, set ``emitter.<flag> = ...`` behind
``type: ignore`` and sniff ``type(emitter).__name__``. ``TargetCaps()``
itself carries the former probe defaults.
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
    "emit_stage_markers", "supports_stages",
)

#: The attributes the bridge used to probe or set on an emitter; each is
#: a ``TargetCaps`` field now and nothing else may carry it.
RETIRED_ATTRIBUTES = (
    "supports_partitions", "per_rank_fragments",
    "suppress_analysis_chain_auto_emit", "model_reissue_purges",
    "_emit_stage_markers",
)


def test_target_caps_has_exactly_these_fields() -> None:
    assert tuple(f.name for f in dataclasses.fields(TargetCaps)) == FIELDS


def test_target_caps_defaults_are_the_former_getattr_defaults() -> None:
    # ``getattr(emitter, "supports_partitions", True)``; every other probe
    # defaulted to False, nothing was archival unless it said so, and every
    # target but live accepted stage brackets.
    assert TargetCaps() == TargetCaps(
        archival=False, supports_partitions=True, per_rank_fragments=False,
        suppress_analysis_chain_auto_emit=False, model_reissue_purges=False,
        emit_stage_markers=False, supports_stages=True,
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
def test_the_retired_attributes_are_gone(cls: type) -> None:
    for name in RETIRED_ATTRIBUTES:
        assert not hasattr(cls, name), f"{cls.__name__}.{name} is a caps field now"


@pytest.mark.parametrize("cls", (TclEmitter, PyEmitter, H5Emitter, RecordingEmitter),
                         ids=lambda c: c.__name__)
def test_an_instance_carries_no_retired_attribute(cls: type) -> None:
    inst = cls()
    for name in RETIRED_ATTRIBUTES:
        assert not hasattr(inst, name), f"{cls.__name__}().{name} is a caps field now"
    assert inst.caps is cls.caps


def test_caps_values_per_emitter() -> None:
    assert TclEmitter.caps == TargetCaps(model_reissue_purges=True)
    assert PyEmitter.caps == TargetCaps()
    assert RecordingEmitter.caps == TargetCaps()
    assert LiveOpsEmitter.caps == TargetCaps(
        supports_partitions=False, supports_stages=False)
    assert H5Emitter.caps == TargetCaps(archival=True)


def test_a_per_emit_override_stays_on_the_instance() -> None:
    e = TclEmitter()
    e.caps = dataclasses.replace(e.caps, supports_partitions=False)
    assert e.caps.supports_partitions is False
    assert TclEmitter.caps.supports_partitions is True
    assert TclEmitter().caps.supports_partitions is True


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
