"""Archive round-trip of frame property modifiers (``FrameModifiers``).

The archive stores each declaration's emitted argument tail, so a modified
frame must come back with the *modified* values: ``OpenSeesModel.from_h5``
holds them in its element and section records, and its re-emitted deck
carries the same element / section / beamIntegration lines as the bridge's
own deck. Nothing about the modifiers themselves is archived beside the
values (no schema change): a replay is an unmodified model with the
effective properties.
"""
from __future__ import annotations

import warnings
from pathlib import Path

import pytest

from apeGmsh.opensees import FrameModifiers, OpenSeesModel, apeSees
from apeGmsh.opensees._internal.replay_types import ReplaySlotTypeWarning

from tests.opensees.h5._opensees_model_fixtures import build_simple_frame_fem

E, G = 2.0e11, 8.0e10
A, IZ, IY, J = 0.01, 2.0e-4, 1.0e-4, 5.0e-5
MASS = 50.0
M = FrameModifiers.etabs(A=0.81, J=0.24, I22=0.35, I33=0.76, mass=0.57)


def _author(family: str) -> apeSees:
    ops = apeSees(build_simple_frame_fem())
    ops.model(ndm=3, ndf=6)
    t = ops.geomTransf.Linear(vecxz=(0.0, 1.0, 0.0))
    if family == "elasticBeamColumn":
        ops.element.elasticBeamColumn(
            pg="Cols", transf=t, A=A, E=E, Iz=IZ, Iy=IY, G=G, J=J,
            mass=MASS, modifiers=M,
        )
    else:
        sec = ops.section.Elastic(E=E, A=A, Iz=IZ, Iy=IY, G=G, J=J)
        getattr(ops.element, family)(
            pg="Cols", transf=t, mass=MASS, modifiers=M,
            integration=ops.beamIntegration.Lobatto(section=sec, n_ip=3),
        )
    return ops


def _lines(tcl: str, *tokens: str) -> list[str]:
    return [ln.strip() for ln in tcl.splitlines()
            if ln.strip().split(" ", 1)[0] in tokens]


@pytest.mark.parametrize(
    "family", ["elasticBeamColumn", "forceBeamColumn", "dispBeamColumn"],
)
def test_replay_carries_the_modified_values(
    tmp_path: Path, family: str,
) -> None:
    ops = _author(family)
    src = tmp_path / "frame.h5"
    ops.h5(str(src))
    bridge_tcl = tmp_path / "frame.tcl"
    ops.tcl(str(bridge_tcl))

    om = OpenSeesModel.from_h5(src)
    (elem,) = [r for r in om.elements() if r.type_token == family]
    assert "-mass" in elem.args
    assert elem.args[elem.args.index("-mass") + 1] == pytest.approx(
        MASS * M.mass, rel=1e-15)
    if family == "elasticBeamColumn":
        # tail: A E G J Iy Iz transfTag -mass m
        assert tuple(elem.args[:6]) == pytest.approx(
            (A * M.A, E, G, J * M.J, IY * M.Iy, IZ * M.Iz), rel=1e-15)
    else:
        secs = {s.tag: s for s in om.sections()}
        assert len(secs) == 2                       # declared + copy
        integ = {r.tag: r for r in om.beam_integration()}
        used = integ[int(elem.args[1])]
        copy = secs[int(used.args[0])]
        assert tuple(copy.params) == pytest.approx(
            (E, A * M.A, IZ * M.Iz, IY * M.Iy, G, J * M.J), rel=1e-15)

    tokens = ("element", "section", "beamIntegration")
    with warnings.catch_warnings():
        # The slot-type dry run rebuilds the element over a field-less
        # stub of its integration; the modifiers must not read it.
        warnings.simplefilter("error", ReplaySlotTypeWarning)
        replayed = om.build("tcl")
    assert isinstance(replayed, str)
    assert _lines(replayed, *tokens) == _lines(
        bridge_tcl.read_text(encoding="utf-8"), *tokens)
