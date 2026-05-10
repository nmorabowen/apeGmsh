"""Unit tests for the ``RecordingEmitter`` test fixture.

These verify that every Protocol method is callable and that calls
are captured in the documented ``(name, args, kwargs)`` shape.
Concrete emitters (Tcl, py, live, h5) get their own test files in
later phases.
"""
from __future__ import annotations

from typing import Any

import pytest

from apeGmsh.opensees.emitter.recording import RecordingEmitter


def test_recording_emitter_constructs_with_no_args() -> None:
    emitter = RecordingEmitter()
    assert emitter.calls == []


# ---------------------------------------------------------------------------
# Per-method recording — one parametrize per group, each verifying the
# (name, args, kwargs) shape.
# ---------------------------------------------------------------------------

def test_model_records_kwargs() -> None:
    e = RecordingEmitter()
    e.model(ndm=3, ndf=6)
    assert e.calls == [("model", (), {"ndm": 3, "ndf": 6})]


def test_node_records_tag_and_coords() -> None:
    e = RecordingEmitter()
    e.node(7, 1.0, 2.0, 3.0)
    assert e.calls == [("node", (7, 1.0, 2.0, 3.0), {})]


def test_fix_records_tag_and_dofs() -> None:
    e = RecordingEmitter()
    e.fix(1, 1, 1, 0, 0, 0, 0)
    assert e.calls == [("fix", (1, 1, 1, 0, 0, 0, 0), {})]


def test_uniaxialMaterial_records_type_tag_params() -> None:
    e = RecordingEmitter()
    e.uniaxialMaterial("Steel02", 42, 420e6, 200e9, 0.01)
    assert e.calls == [
        ("uniaxialMaterial", ("Steel02", 42, 420e6, 200e9, 0.01), {})
    ]


def test_element_records_type_tag_args() -> None:
    e = RecordingEmitter()
    e.element("forceBeamColumn", 5, 1, 2, 3, 4)
    assert e.calls == [
        ("element", ("forceBeamColumn", 5, 1, 2, 3, 4), {})
    ]


def test_pattern_open_close_pair() -> None:
    e = RecordingEmitter()
    e.pattern_open("Plain", 1, 1)
    e.load(7, 100.0, 0.0, 0.0)
    e.pattern_close()
    assert [c[0] for c in e.calls] == [
        "pattern_open", "load", "pattern_close",
    ]
    assert e.calls[0] == ("pattern_open", ("Plain", 1, 1), {})
    assert e.calls[1] == ("load", (7, 100.0, 0.0, 0.0), {})
    assert e.calls[2] == ("pattern_close", (), {})


def test_section_open_close_pair_with_patch_and_fiber() -> None:
    e = RecordingEmitter()
    e.section_open("Fiber", 3)
    e.patch("rect", 1, 10, 10, -1.0, -1.0, 1.0, 1.0)
    e.fiber(0.0, 0.0, 0.001, 1)
    e.section_close()
    names = [c[0] for c in e.calls]
    assert names == ["section_open", "patch", "fiber", "section_close"]


def test_analyze_returns_int_and_records() -> None:
    e = RecordingEmitter()
    rc = e.analyze(steps=20, dt=0.01)
    assert isinstance(rc, int)
    assert rc == 0
    assert e.calls == [
        ("analyze", (), {"steps": 20, "dt": 0.01})
    ]


def test_analyze_dt_default_none() -> None:
    e = RecordingEmitter()
    e.analyze(steps=5)
    assert e.calls == [("analyze", (), {"steps": 5, "dt": None})]


# ---------------------------------------------------------------------------
# Coverage spot-check: every Protocol method should at least be defined
# on the emitter. We don't enumerate the full Protocol surface here —
# that's the contract layer's job — but we sanity-check a representative
# set so the fixture isn't silently missing a method.
# ---------------------------------------------------------------------------

REPRESENTATIVE_METHODS = (
    "model", "node", "fix", "mass",
    "uniaxialMaterial", "nDMaterial", "section", "geomTransf",
    "section_open", "section_close", "patch", "fiber", "layer",
    "element", "timeSeries",
    "pattern_open", "pattern_close", "load", "eleLoad", "sp",
    "recorder",
    "constraints", "numberer", "system", "test",
    "algorithm", "integrator", "analysis", "analyze",
)


@pytest.mark.parametrize("method_name", REPRESENTATIVE_METHODS)
def test_recording_emitter_has_method(method_name: str) -> None:
    e = RecordingEmitter()
    method: Any = getattr(e, method_name, None)
    assert callable(method), f"RecordingEmitter missing method: {method_name}"
