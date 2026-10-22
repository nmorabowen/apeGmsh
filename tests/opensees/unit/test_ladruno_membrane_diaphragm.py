"""``LadrunoMembrane`` ``diaphragm_nodes=`` and ``f11=`` / ``f22=`` / ``f12=``.

The fork flags of build ``c83ee812e`` (fork WP-204): ``-diaphragmNodes``
(in-plane stiffness and mass, no gravity share) and the in-plane modifiers
``-f11 -f22 -f12``. Numeric oracles:
``tests/opensees/live/test_ladruno_membrane_diaphragm_live.py``.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from apeGmsh.opensees import OpenSeesModel, apeSees
from apeGmsh.opensees._internal.build.errors import BridgeError
from apeGmsh.opensees.element.shell import LadrunoMembrane
from apeGmsh.opensees.emitter.recording import RecordingEmitter
from apeGmsh.opensees.emitter.tcl import TclEmitter
from apeGmsh.opensees.node import Node, NodeSet
from tests.opensees.unit.test_ladruno_membrane import _2x2_slab_fem

_STIFF = (25000.5, 0.2, 0.2)


def _node_set(*tags: int) -> NodeSet:
    bridge = object()
    return NodeSet(
        nodes=tuple(Node(tag=t, coords=(0.0, 0.0, 0.0), bridge=bridge) for t in tags),  # type: ignore[arg-type]
        bridge=bridge,  # type: ignore[arg-type]
    )


def _membrane_args(rec: RecordingEmitter) -> tuple[object, ...]:
    calls = [
        c for c in rec.calls
        if c[0] == "element" and c[1][0] == "LadrunoMembrane"
    ]
    assert len(calls) == 1
    return calls[0][1]


def _flag(args: tuple[object, ...], flag: str, n: int) -> tuple[object, ...]:
    i = args.index(flag)
    return args[i + 1:i + 1 + n]


class TestConstruction:
    @pytest.mark.parametrize("name", ["f11", "f22", "f12"])
    @pytest.mark.parametrize("bad", [0.0, -0.5, float("nan"), float("inf")])
    def test_modifier_must_be_finite_and_positive(self, name: str, bad: float) -> None:
        with pytest.raises(ValueError, match=f"{name}= must be finite and > 0"):
            LadrunoMembrane(nodes=(1, 2, 3), stiff=_STIFF, **{name: bad})

    @pytest.mark.parametrize(
        "kw", [{"f11": 0.5}, {"f12": 0.3}, {"diaphragm_nodes": (5,)}],
    )
    def test_stiffness_flags_need_a_stiffness(self, kw: dict[str, object]) -> None:
        with pytest.raises(ValueError, match="act on the in-plane stiffness"):
            LadrunoMembrane(nodes=(1, 2, 3), pressure=-1.0, **kw)

    def test_diaphragm_nodes_forms_are_normalised(self) -> None:
        ele = LadrunoMembrane(
            nodes=(1, 2, 3), stiff=_STIFF,
            diaphragm_nodes=(7, "Edge", _node_set(8, 9)),
        )
        assert ele.diaphragm_nodes == (7, "Edge", 8, 9)
        assert LadrunoMembrane(
            nodes=(1, 2, 3), stiff=_STIFF, diaphragm_nodes=4,
        ).diaphragm_nodes == (4,)

    @pytest.mark.parametrize("bad", [(), (0,), (True,), (1.5,)])
    def test_bad_diaphragm_nodes_refused(self, bad: tuple[object, ...]) -> None:
        with pytest.raises((TypeError, ValueError)):
            LadrunoMembrane(nodes=(1, 2, 3), stiff=_STIFF, diaphragm_nodes=bad)  # type: ignore[arg-type]


class TestEmit:
    def test_direct_emit_places_the_new_flags(self) -> None:
        ele = LadrunoMembrane(
            nodes=(1, 2, 3, 4), support_nodes=(1,), diaphragm_nodes=(5, 6),
            stiff=_STIFF, f11=0.7, f22=0.35, f12=0.25,
        )
        rec = RecordingEmitter()
        ele._emit(rec, tag=9)
        args = rec.calls[0][1]
        assert args == (
            "LadrunoMembrane", 9,
            "-nodes", 1, 2, 3, 4,
            "-supportNodes", 1,
            "-diaphragmNodes", 5, 6,
            "-method", "tributary",
            "-stiff", 25000.5, 0.2, 0.2,
            "-f11", 0.7, "-f22", 0.35, "-f12", 0.25,
        )

    def test_defaults_emit_neither_flag(self) -> None:
        rec = RecordingEmitter()
        LadrunoMembrane(nodes=(1, 2, 3), stiff=_STIFF)._emit(rec, tag=1)
        args = rec.calls[0][1]
        assert "-diaphragmNodes" not in args
        assert not {"-f11", "-f22", "-f12"} & set(args)

    def test_direct_emit_refuses_a_name(self) -> None:
        ele = LadrunoMembrane(nodes=(1, 2, 3), stiff=_STIFF, diaphragm_nodes="Edge")
        with pytest.raises(RuntimeError, match="requires emitting through"):
            ele._emit(RecordingEmitter(), tag=1)

    def test_bridge_resolves_pg_and_label_names(self) -> None:
        ops = apeSees(_2x2_slab_fem())
        ops.model(ndm=3, ndf=6)
        ops.element.LadrunoMembrane(
            pg="Slab", support_nodes="Corners",
            diaphragm_nodes=("c1", "Corners", 2), stiff=_STIFF, f22=0.5,
        )
        ops.fix(pg="Corners", dofs=(1, 1, 1, 1, 1, 1))
        rec = RecordingEmitter()
        ops.build().emit(rec)
        args = _membrane_args(rec)
        # Label c1 is node 1; first occurrence kept.
        assert _flag(args, "-diaphragmNodes", 5) == (1, 3, 7, 9, 2)
        assert _flag(args, "-f22", 1) == (0.5,)

    def test_unknown_name_is_refused(self) -> None:
        ops = apeSees(_2x2_slab_fem())
        ops.model(ndm=3, ndf=6)
        ops.element.LadrunoMembrane(
            pg="Slab", support_nodes="Corners", diaphragm_nodes="Nope",
            stiff=_STIFF,
        )
        with pytest.raises(BridgeError):
            ops.build().emit(RecordingEmitter())

    def test_h5_replay_keeps_the_new_flags(self, tmp_path: Path) -> None:
        ops = apeSees(_2x2_slab_fem())
        ops.model(ndm=3, ndf=6)
        ops.element.LadrunoMembrane(
            pg="Slab", support_nodes="Corners", diaphragm_nodes=(2, 8),
            stiff=_STIFF, f11=0.6, f12=0.4,
        )
        ops.fix(pg="Corners", dofs=(1, 1, 1, 1, 1, 1))
        tcl = TclEmitter()
        ops.build().emit(tcl)
        h5 = tmp_path / "m.h5"
        ops.h5(h5)
        direct = [ln.strip() for ln in tcl.lines() if ln.strip().startswith("element LadrunoMembrane")]
        replay = [
            ln.strip() for ln in OpenSeesModel.from_h5(h5).build("tcl").splitlines()
            if ln.strip().startswith("element LadrunoMembrane")
        ]
        assert len(direct) == 1 and "-diaphragmNodes 2 8" in direct[0]
        assert "-f11 0.6" in direct[0] and "-f12 0.4" in direct[0]
        assert replay == direct
