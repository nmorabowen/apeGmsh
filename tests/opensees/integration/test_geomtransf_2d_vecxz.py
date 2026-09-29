"""2-D ``geomTransf`` never carries a vecxz vector.

OpenSees' Tcl ``geomTransf`` in 2-D accepts only ``<Type> tag
<-jntOffset ...>``; a trailing ``0.0 0.0 1.0`` fails with "bad command"
while the interpreter still exits 0, and in-process openseespy ignores
the extra args — so an explicit ``vecxz=`` in a 2-D model used to write
a Tcl deck that silently lost every beam.  The bridge now drops a vecxz
along global Z (the only direction a 2-D model can mean) and refuses
any other vector.
"""
from __future__ import annotations

from pathlib import Path
from typing import cast

import pytest

from apeGmsh.opensees import apeSees
from apeGmsh.opensees._internal.build import BridgeError
from apeGmsh.opensees.emitter.recording import RecordingEmitter

from tests.opensees.fixtures.fem_stub import (
    FEMStub,
    _ElementGroupView,
    _ElementsStub,
    _NodesStub,
    make_two_column_frame,
)


def _portal_xy() -> FEMStub:
    """Two columns in the XY plane (``make_two_column_frame`` runs its
    columns along Z, which collapse to zero length in a 2-D run)."""
    nodes = _NodesStub(
        ids=[1, 2, 3, 4],
        coords=[(0.0, 0.0, 0.0), (0.0, 1.0, 0.0),
                (1.0, 0.0, 0.0), (1.0, 1.0, 0.0)],
        node_pgs={"Base": [1, 3], "Top": [2, 4]},
    )
    elements = _ElementsStub(
        elem_pgs={"Cols": _ElementGroupView(
            ids=(1, 2), connectivity=((1, 2), (3, 4)),
        )},
    )
    return FEMStub(nodes=nodes, elements=elements)


def _frame_2d(transf_type: str, vecxz: tuple[float, float, float]) -> apeSees:
    fem = _portal_xy()
    ops = apeSees(cast("object", fem))  # type: ignore[arg-type]
    ops.model(ndm=2, ndf=3)
    transf = getattr(ops.geomTransf, transf_type)(vecxz=vecxz)
    ops.element.elasticBeamColumn(
        pg="Cols", transf=transf, A=0.01, E=200e9, Iz=1e-4,
    )
    ops.fix(pg="Base", dofs=(1, 1, 1))
    return ops


@pytest.mark.parametrize("transf_type", ["Linear", "PDelta", "Corotational"])
@pytest.mark.parametrize("vecxz", [(0.0, 0.0, 1.0), (0.0, 0.0, -2.0)])
def test_2d_tcl_deck_emits_bare_geomtransf(
    tmp_path: Path, transf_type: str, vecxz: tuple[float, float, float],
) -> None:
    ops = _frame_2d(transf_type, vecxz)
    deck = tmp_path / "frame.tcl"
    ops.tcl(str(deck))

    lines = [
        ln.split() for ln in deck.read_text(encoding="utf-8").splitlines()
        if ln.startswith("geomTransf")
    ]
    assert len(lines) == 1
    tok, ttype, tag, *rest = lines[0]
    assert (tok, ttype) == ("geomTransf", transf_type)
    assert tag.isdigit()
    assert rest == [], f"2-D geomTransf must carry no vecxz: {lines[0]}"


def test_2d_recording_emitter_drops_z_vecxz() -> None:
    ops = _frame_2d("Linear", (0.0, 0.0, 1.0))
    rec = RecordingEmitter()
    ops.build().emit(rec)
    calls = [c[1] for c in rec.calls if c[0] == "geomTransf"]
    assert len(calls) == 1
    assert calls[0][0] == "Linear" and len(calls[0]) == 2


def test_2d_non_z_vecxz_fails_loud() -> None:
    ops = _frame_2d("Linear", (1.0, 0.0, 0.0))
    with pytest.raises(BridgeError, match="vecxz.*ndm=2"):
        ops.build().emit(RecordingEmitter())


def test_3d_vecxz_still_emitted() -> None:
    fem = make_two_column_frame()
    ops = apeSees(cast("object", fem))  # type: ignore[arg-type]
    ops.model(ndm=3, ndf=6)
    transf = ops.geomTransf.Linear(vecxz=(1.0, 0.0, 0.0))
    ops.element.elasticBeamColumn(
        pg="Cols", transf=transf, A=0.01, E=200e9, Iz=1e-4, Iy=1e-4,
        G=80e9, J=2e-4,
    )
    rec = RecordingEmitter()
    ops.build().emit(rec)
    calls = [c[1] for c in rec.calls if c[0] == "geomTransf"]
    assert calls == [("Linear", calls[0][1], 1.0, 0.0, 0.0)]
