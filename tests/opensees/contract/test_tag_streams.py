"""Tag law (K1-3, #1361): one ``BuiltModel`` mints the same tags on every target.

``BuiltModel.emit`` mints derived tags (element fan-out, transform
overrides, regions, parameters) on a fresh allocator per emit. Until the
build-time tag plan (#1445) lands, nothing but this pin says the four
targets see the same tags: the ``(verb, tag)`` multiset that the bridge
drives into the Recording, Tcl, Py and H5 emitters of one ``BuiltModel``
must be identical. Order is not compared, since the targets may legitimately
interleave differently; the in-order stream across processes is pinned by
``tests/opensees/subprocess/test_tag_determinism.py``.
"""
from __future__ import annotations

from collections import Counter
from typing import Any

import pytest

from apeGmsh.opensees.emitter.h5 import H5Emitter
from apeGmsh.opensees.emitter.py import PyEmitter
from apeGmsh.opensees.emitter.recording import RecordingEmitter
from apeGmsh.opensees.emitter.tcl import TclEmitter

from tests.opensees.contract import _tag_streams as ts

_MODELS = ts.models()

#: Kinds whose tags emit mints rather than copies from a primitive or the
#: mesh. The model set must reach every one, or the pin is weaker than it
#: reads.
_MINTED_KINDS = (
    "element:", "geomTransf:", "region", "addToParameter",
)


#: The four targets: (label, emitter class, constructor kwargs).
_TARGETS: tuple[tuple[str, type, dict[str, Any]], ...] = (
    ("recording", RecordingEmitter, {}),
    ("tcl", TclEmitter, {}),
    ("py", PyEmitter, {}),
    ("h5", H5Emitter, {"model_name": "tag_law"}),
)


@pytest.mark.parametrize("name", sorted(_MODELS))
def test_verb_tag_multiset_matches_across_targets(name: str) -> None:
    bm = _MODELS[name]().build()
    streams = {
        label: Counter(ts.emit_stream(bm, cls, **kwargs))
        for label, cls, kwargs in _TARGETS
    }
    ref = streams["recording"]
    assert ref, f"{name}: the Recording emit drove no tagged verb"
    for label, got in streams.items():
        assert got == ref, (
            f"{name}: {label} saw (verb, tag) rows the Recording emit did "
            f"not: extra={sorted((got - ref).items())} "
            f"missing={sorted((ref - got).items())}"
        )


def test_model_set_reaches_every_minted_kind() -> None:
    kinds = {
        kind
        for recipe in _MODELS.values()
        for kind, _tag in ts.tag_stream(recipe(), RecordingEmitter)
    }
    for prefix in _MINTED_KINDS:
        assert any(k.startswith(prefix) for k in kinds), (
            f"no model emits a {prefix!r} row; the tag-stream pins do not "
            "cover that minting site"
        )


def test_tag_positions_cover_the_declaring_verbs() -> None:
    """The derived tag table reaches the verbs that declare an object."""
    pos = ts.tag_positions()
    for verb, index in {
        "node": 0, "element": 1, "uniaxialMaterial": 1, "nDMaterial": 1,
        "section": 1, "geomTransf": 1, "beamIntegration": 1,
        "timeSeries": 1, "pattern_open": 1, "region": 0, "damping": 1,
        "contact_surface": 0, "contact": 0, "contact_plane": 0,
        "embedded_rebar": 0, "embedded_node": 0, "addToParameter": 0,
        "update_parameter": 0, "flip_element_stage": 0,
    }.items():
        assert pos.get(verb) == index, (verb, pos.get(verb))


def test_projection_fails_closed_on_a_non_integer_tag() -> None:
    emitter = ts.tapped(RecordingEmitter)()
    emitter.tap = []
    with pytest.raises(TypeError, match="should be a tag"):
        emitter.region("seven")

