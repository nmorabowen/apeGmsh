"""``p.surface_load(..., pressure=q)``: a per-pattern LadrunoMembrane magnitude.

The fork's ``LadrunoMembrane::addLoad`` reads a ``-beamUniform wy wz wx``
load as the traction ``wy e2 + wz n + wx e1`` and distributes it through the
same transfer as ``-surfaceLoad`` (whose traction is ``pressure * n``). So
``surface_load(pressure=q)`` emits ``-beamUniform 0.0 q 0.0``, and D, SD and
L can each carry their own magnitude on one panel.
"""
from __future__ import annotations

from typing import Any

import pytest

from apeGmsh.opensees._internal.build.errors import BridgeError
from apeGmsh.opensees._internal.build.membrane_load_guard import (
    InertMembranePressureError,
)
from apeGmsh.opensees.emitter.recording import RecordingEmitter
from tests.opensees.unit.test_ladruno_membrane_inert_pressure import (
    _SLAB_TAG,
    _slab_on_beams,
)


def _ele_loads(ops: Any) -> list[tuple[int | None, tuple[object, ...]]]:
    """``(pattern tag, eleLoad args)`` in emit order."""
    rec = RecordingEmitter()
    ops.build().emit(rec)
    out: list[tuple[int | None, tuple[object, ...]]] = []
    current: int | None = None
    for name, args, _kw in rec.calls:
        if name == "pattern":
            current = int(args[1])
        elif name == "eleLoad":
            out.append((current, args))
    return out


def test_each_pattern_emits_its_own_magnitude() -> None:
    ops, slab = _slab_on_beams()
    for q in (-4.0, -1.5, -2.0):
        with ops.pattern.Plain(series=ops.timeSeries.Linear()) as p:
            p.surface_load(ele=slab, pressure=q)
    loads = _ele_loads(ops)
    assert [args for _pat, args in loads] == [
        ("-ele", _SLAB_TAG, "-type", "-beamUniform", 0.0, q, 0.0)
        for q in (-4.0, -1.5, -2.0)
    ]


def test_pg_target_resolves_to_the_membrane() -> None:
    ops, _slab = _slab_on_beams()
    with ops.pattern.Plain(series=ops.timeSeries.Linear()) as p:
        p.surface_load(pg="Slab", pressure=-3.0)
    assert [a for _p, a in _ele_loads(ops)] == [
        ("-ele", _SLAB_TAG, "-type", "-beamUniform", 0.0, -3.0, 0.0)]


def test_plain_surface_load_still_reads_the_element_pressure() -> None:
    ops, slab = _slab_on_beams(pressure=-8.0)
    with ops.pattern.Plain(series=ops.timeSeries.Linear()) as p:
        p.surface_load(ele=slab)
    assert [a for _p, a in _ele_loads(ops)] == [
        ("-ele", _SLAB_TAG, "-type", "-surfaceLoad")]


def test_pattern_pressures_leave_an_element_pressure_inert_and_refused() -> None:
    """surface_load(pressure=q) never reads the element's pressure=."""
    ops, slab = _slab_on_beams(pressure=-8.0)
    with ops.pattern.Plain(series=ops.timeSeries.Linear()) as p:
        p.surface_load(ele=slab, pressure=-3.0)
    with pytest.raises(InertMembranePressureError, match="own magnitude"):
        ops.build()


def test_label_target_is_refused() -> None:
    ops, _slab = _slab_on_beams()
    with ops.pattern.Plain(series=ops.timeSeries.Linear()) as p:
        with pytest.raises(ValueError, match="label="):
            p.surface_load(label="south_beam", pressure=-1.0)


def test_non_membrane_handle_is_refused() -> None:
    ops, _slab = _slab_on_beams()
    beams = next(p for p in ops._primitives
                 if type(p).__name__ == "elasticBeamColumn")
    with ops.pattern.Plain(series=ops.timeSeries.Linear()) as p:
        with pytest.raises(ValueError, match="LadrunoMembrane targets only"):
            p.surface_load(ele=beams, pressure=-1.0)


def test_pg_naming_no_membrane_is_refused() -> None:
    ops, _slab = _slab_on_beams()
    with ops.pattern.Plain(series=ops.timeSeries.Linear()) as p:
        p.surface_load(pg="Beams", pressure=-1.0)
    with pytest.raises(BridgeError, match="names no LadrunoMembrane"):
        ops.build().emit(RecordingEmitter())


def test_non_finite_pressure_is_refused() -> None:
    ops, slab = _slab_on_beams()
    with ops.pattern.Plain(series=ops.timeSeries.Linear()) as p:
        with pytest.raises(ValueError, match="finite"):
            p.surface_load(ele=slab, pressure=float("nan"))


def test_two_dimensional_model_is_refused() -> None:
    from apeGmsh.opensees import apeSees
    from tests.opensees.unit.test_ladruno_membrane import _2x2_slab_fem

    ops = apeSees(_2x2_slab_fem(), element_tags="fem")
    ops.model(ndm=2, ndf=3)
    slab = ops.element.LadrunoMembrane(pg="Slab", support_nodes="Corners")
    ops.fix(pg="Corners", dofs=(1, 1, 1))
    with ops.pattern.Plain(series=ops.timeSeries.Linear()) as p:
        p.surface_load(ele=slab, pressure=-1.0)
    with pytest.raises(BridgeError, match="ndm=3"):
        ops.build().emit(RecordingEmitter())


def test_archive_replay_keeps_each_patterns_magnitude(tmp_path: Any) -> None:
    from apeGmsh.opensees import OpenSeesModel

    ops, slab = _slab_on_beams()
    for q in (-4.0, -1.5):
        with ops.pattern.Plain(series=ops.timeSeries.Linear()) as p:
            p.surface_load(ele=slab, pressure=q)
    ops.h5(tmp_path / "m.h5")
    deck = OpenSeesModel.from_h5(tmp_path / "m.h5").build("tcl")
    assert isinstance(deck, str)
    for q in (-4.0, -1.5):
        assert f"eleLoad -ele {_SLAB_TAG} -type -beamUniform 0.0 {q} 0.0" in deck
