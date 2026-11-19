"""A frame transform or element under an envelope its builder rejects raises.

YOO Guayaquil (2026-11): a LadrunoBrick mat with ``elasticBeamColumn``
piles was declared ``ops.model(ndm=3, ndf=3)``. The deck was emitted, and
the fork refused its first ``geomTransf`` at run time
(``OpenSeesCrdTransfCommands.cpp``: the parser gates on the BUILDER ndf,
never on a node's ``-ndf``). The beam parsers gate the same way, each on
its own. The bridge now refuses both before writing.
"""
from __future__ import annotations

from typing import Any, cast

import pytest

from apeGmsh.opensees import apeSees
from apeGmsh.opensees._element_capabilities import (
    _FRAME_BUILDER_NDF,
    element_frame_builder_ndf,
)
from apeGmsh.opensees._internal.build.errors import BridgeError
from apeGmsh.opensees._internal.build.transforms import (
    validate_frame_builder_ndf,
)
from apeGmsh.opensees.element import beam_column
from apeGmsh.opensees.emitter.recording import RecordingEmitter
from apeGmsh.opensees.transform import Corotational, Linear, PDelta
from tests.opensees.fixtures.fem_stub import make_two_node_beam


def _beam_model(ndm: int, ndf: int, transf: Any = None) -> apeSees:
    ops = apeSees(cast(Any, make_two_node_beam(ndm=ndm)))
    ops.model(ndm=ndm, ndf=ndf)
    vec = {"vecxz": (0.0, 1.0, 0.0)} if ndm == 3 else {}
    ops.element.elasticBeamColumn(
        pg="Cols", transf=transf or ops.geomTransf.Linear(**vec),
        A=0.01, E=2.0e8, Iz=2.0e-4, Iy=1.0e-4, G=8.0e7, J=5.0e-5,
    )
    return ops


def test_yoo_case_3d_ndf3_with_a_beam_raises_at_emit_naming_the_fix() -> None:
    rec = RecordingEmitter()
    with pytest.raises(BridgeError, match=r"ops\.model\(ndm=3, ndf=6\)") as e:
        _beam_model(3, 3).build().emit(rec)
    msg = str(e.value)
    assert "geomTransf Linear" in msg and "ndf=3" in msg
    assert "incompatible with frame elements" in msg
    # Nothing past the model line was written.
    assert [c[0] for c in rec.calls if c[0] in {"geomTransf", "element"}] == []


@pytest.mark.parametrize(("ndm", "ndf"), [(3, 6), (2, 3)])
def test_frame_envelope_emits_the_transform(ndm: int, ndf: int) -> None:
    rec = RecordingEmitter()
    _beam_model(ndm, ndf).build().emit(rec)
    assert [c[0] for c in rec.calls].count("geomTransf") == 1


def test_corotational_at_2d_ndf4_passes_the_transform_but_not_the_beam() -> None:
    # Corotational parses at 2-D ndf 4 (warping); elasticBeamColumn does not
    # (ElasticBeam2d.cpp:80, TclElasticBeamCommand.cpp:65): the element half
    # of the check catches what the transform half lets through.
    ops = apeSees(cast(Any, make_two_node_beam(ndm=2)))
    ops.model(ndm=2, ndf=4)
    ops.element.elasticBeamColumn(
        pg="Cols", transf=ops.geomTransf.Corotational(), A=0.01, E=2.0e8,
        Iz=2.0e-4)
    with pytest.raises(BridgeError, match=r"element elasticBeamColumn.*ndf=3\)"):
        ops.build().emit(RecordingEmitter())


@pytest.mark.parametrize(
    ("cls", "ndm", "ok", "bad"),
    [
        (Linear, 3, {6}, {3, 4, 7}),
        (PDelta, 3, {6}, {3, 7}),
        # 7 is the py parser's warping form; the Tcl parser refuses it.
        (Corotational, 3, {6}, {3, 4, 7}),
        (Linear, 2, {3}, {2, 4}),
        (PDelta, 2, {3}, {2}),
        (Corotational, 2, {3, 4}, {2, 6}),
    ],
)
def test_accepted_builder_ndf_mirrors_the_parser(
    cls: type, ndm: int, ok: set[int], bad: set[int],
) -> None:
    t = cls(vecxz=(0.0, 1.0, 0.0)) if ndm == 3 else cls()
    for ndf in ok:
        validate_frame_builder_ndf([t], [], ndm=ndm, envelope_ndf=ndf)
    for ndf in bad:
        with pytest.raises(BridgeError, match=rf"ndf={ndf}\)"):
            validate_frame_builder_ndf([t], [], ndm=ndm, envelope_ndf=ndf)


@pytest.mark.parametrize("token", sorted(_FRAME_BUILDER_NDF))
def test_every_frame_token_is_a_real_beam_class_gated_at_3_and_6(
    token: str,
) -> None:
    # A typo in the table would silently gate nothing.
    assert hasattr(beam_column, token)
    assert element_frame_builder_ndf(token, 2) == 3
    assert element_frame_builder_ndf(token, 3) == 6


def test_a_solid_is_not_frame_gated() -> None:
    assert element_frame_builder_ndf("LadrunoBrick", 3) is None
