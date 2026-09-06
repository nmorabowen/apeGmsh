"""``ops.section.LadrunoShellModifier`` through the real bridge (ADR 91).

The unit tests in ``unit/primitives/test_sections_plate.py`` drive
``_emit`` directly with a stub resolver. These drive the bridge, which
is what actually has to get the *ordering* right: the decorator declares
its wrapped section via ``dependencies()``, so the inner ``section``
line must precede the wrapper's and the wrapper must reference the
inner's allocated tag.
"""
from __future__ import annotations

from typing import cast

from apeGmsh.opensees import apeSees
from apeGmsh.opensees.emitter.recording import RecordingEmitter
from apeGmsh.opensees.emitter.tcl import TclEmitter

from tests.opensees.fixtures.fem_stub import make_two_column_frame


def _bridge() -> apeSees:
    ops = apeSees(cast("object", make_two_column_frame()))
    ops.model(ndm=3, ndf=6)
    transf = ops.geomTransf.Linear(vecxz=(1.0, 0.0, 0.0))
    ops.element.elasticBeamColumn(
        pg="Cols", transf=transf,
        A=0.01, E=200e9, Iz=1e-4, Iy=1e-4, G=80e9, J=1e-4,
    )
    return ops


def _sections(rec: RecordingEmitter) -> list[tuple]:
    return [a for (n, a, _k) in rec.calls if n == "section"]


def test_inner_section_emits_before_the_wrapper_and_is_referenced() -> None:
    ops = _bridge()
    inner = ops.section.ElasticMembranePlateSection(
        E=25e6, nu=0.2, h=0.3, name="slab",
    )
    ops.section.LadrunoShellModifier(
        inner=inner, f11=0.35, f22=0.35, f12=0.35, name="wall_cracked",
    )
    rec = RecordingEmitter()
    ops.build().emit(rec)

    plate, wrapper = _sections(rec)
    assert plate[0] == "ElasticMembranePlateSection"
    assert wrapper[0] == "LadrunoShellModifier"
    inner_tag = plate[1]
    # The wrapper's first parameter is the inner section's tag.
    assert wrapper[2] == inner_tag
    assert wrapper[1] != inner_tag


def test_all_defaults_wrap_reaches_the_deck_as_a_bare_wrapper() -> None:
    ops = _bridge()
    inner = ops.section.ElasticMembranePlateSection(
        E=25e6, nu=0.2, h=0.3, name="slab",
    )
    ops.section.LadrunoShellModifier(inner=inner, name="uncracked")
    tcl = TclEmitter()
    ops.build().emit(tcl)
    line = next(
        ln for ln in tcl._lines if ln.startswith("section LadrunoShellModifier")
    )
    # Sparse emit — tag, inner tag, and nothing else.
    assert len(line.split()) == 4, line
