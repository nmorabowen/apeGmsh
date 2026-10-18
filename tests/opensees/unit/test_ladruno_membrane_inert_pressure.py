"""#2151: ``ops.build()`` refuses a ``LadrunoMembrane`` whose ``pressure=``
no ``Plain`` pattern activates.

The oracle is the fork element itself (FORK_PIN ``100eb9ae1``,
``LadrunoMembrane.cpp``): its load vector is filled only in
``LadrunoMembrane::addLoad``, from a pattern's ``eleLoad``. A pressured
membrane with no activating pattern applied no load, and the nightly
``live-fork`` lane read every corner reaction as exactly 0.0 (#2143).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import numpy as np
import pytest

from apeGmsh.opensees import apeSees
from apeGmsh.opensees._internal.build.errors import BridgeError
from apeGmsh.opensees._internal.build.membrane_load_guard import (
    InertMembranePressureError,
)
from apeGmsh.opensees.element.shell import LadrunoMembrane
from apeGmsh.opensees.emitter.recording import RecordingEmitter
from apeGmsh.opensees.pattern.pattern import Plain
from tests.opensees.fixtures.fem_stub import FEMStub, _ElementsStub, _NodesStub
from tests.opensees.unit.test_ladruno_membrane import _2x2_slab_fem


def _deck_2143(*, pressure: float | None = -10.0) -> tuple[apeSees, LadrunoMembrane]:
    """The deck of the 10-18 ``live-fork`` red (#2143), with no pattern."""
    nodes = _NodesStub(
        ids=np.array([1, 2, 3, 4], dtype=np.int64),
        coords=np.array(
            [[0.0, 0.0, 0.0], [4.0, 0.0, 0.0], [4.0, 4.0, 0.0], [0.0, 4.0, 0.0]],
            dtype=np.float64,
        ),
        node_pgs={"Corners": (1, 2, 3, 4)},
    )
    fem = FEMStub(nodes=nodes, elements=_ElementsStub(elem_pgs={}))
    ops = apeSees(cast("object", fem))
    ops.model(ndm=3, ndf=6)
    slab = ops.element.LadrunoMembrane(
        nodes=(1, 2, 3, 4),
        support_nodes="Corners",
        method="tributary",
        pressure=pressure,
        stiff=(30e6, 0.2, 0.2),
    )
    ops.fix(pg="Corners", dofs=(1, 1, 1, 1, 1, 1))
    ops.constraints.Plain()
    ops.numberer.Plain()
    ops.system.BandGeneral()
    ops.test.NormDispIncr(tol=1e-10, max_iter=10)
    ops.algorithm.Linear()
    ops.integrator.LoadControl(dlam=1.0)
    ops.analysis.Static()
    return ops, slab


def _slab_on_beams(**membrane: object) -> tuple[apeSees, LadrunoMembrane]:
    """A 2x2 slab membrane (``pg="Slab"``) on perimeter beams."""
    ops = apeSees(_2x2_slab_fem(), element_tags="fem")
    ops.model(ndm=3, ndf=6)
    transf = ops.geomTransf.Linear(vecxz=(0.0, 0.0, 1.0))
    beams = ops.element.elasticBeamColumn(
        pg="Beams", transf=transf, A=0.1, E=30e9, Iz=1e-3, Iy=1e-3, G=12e9, J=1e-4,
    )
    slab = ops.element.LadrunoMembrane(pg="Slab", frames=beams, **membrane)  # type: ignore[arg-type]
    ops.fix(pg="Corners", dofs=(1, 1, 1, 1, 1, 1))
    return ops, slab


def _ele_load_tags(ops: apeSees) -> list[tuple[object, ...]]:
    rec = RecordingEmitter()
    ops.build().emit(rec)
    return [c[1] for c in rec.calls if c[0] == "eleLoad"]


#: Under ``element_tags="fem"`` the 2x2 fixture's FEM ids end at 204, so the
#: membrane macro element is planned as tag 205.
_SLAB_TAG = 205


class TestRefuses:
    def test_the_2143_deck_refuses_by_name(self) -> None:
        ops, _slab = _deck_2143()
        with pytest.raises(InertMembranePressureError) as exc:
            ops.build()
        msg = str(exc.value)
        assert "LadrunoMembrane(nodes=(1, 2, 3, 4))" in msg
        assert "pressure=-10.0" in msg
        assert "p.surface_load(ele=<membrane>)" in msg
        # A named error that is still a BridgeError, like every build refusal.
        assert isinstance(exc.value, BridgeError)

    def test_every_emit_route_refuses(self, tmp_path: object) -> None:
        """tcl / py / h5 all go through build(), so none writes the inert deck."""
        from pathlib import Path

        out = cast(Path, tmp_path)
        for write in (
            lambda ops: ops.tcl(out / "m.tcl"),
            lambda ops: ops.py(out / "m.py"),
            lambda ops: ops.h5(out / "m.h5"),
        ):
            ops, _slab = _deck_2143()
            with pytest.raises(InertMembranePressureError):
                write(ops)

    def test_a_pattern_on_another_element_does_not_count(self) -> None:
        ops, _slab = _slab_on_beams(pressure=-5.0)
        with ops.pattern.Plain(series=ops.timeSeries.Linear()) as p:
            p.beam_uniform(pg="Beams", wy=-1.0)
            p.beam_uniform(ele=101, wy=-1.0)
        with pytest.raises(InertMembranePressureError, match=r"pg='Slab'"):
            ops.build()

    def test_label_does_not_reach_a_membrane(self) -> None:
        """``label=`` resolves to mesh cells, and a membrane has none.

        The pressure-free twin proves the emit side: the same ``label=``
        load lands on the beam it names (101), never on the membrane.
        """
        ops, _slab = _slab_on_beams()
        with ops.pattern.Plain(series=ops.timeSeries.Linear()) as p:
            p.surface_load(label="south_beam")
        assert _ele_load_tags(ops) == [("-ele", 101, "-type", "-surfaceLoad")]

        ops, _slab = _slab_on_beams(pressure=-5.0)
        with ops.pattern.Plain(series=ops.timeSeries.Linear()) as p:
            p.surface_load(label="south_beam")
        with pytest.raises(InertMembranePressureError, match="label= does not"):
            ops.build()

    def test_an_int_ele_tag_fails_closed_and_names_the_handle(self) -> None:
        """A membrane's tag is planned at emit, and the tag law keeps planning
        off the build side, so an int ``ele=`` is never matched to one, even
        the membrane's own planned tag (205 here; the emit would load it)."""
        ops, _slab = _slab_on_beams(pressure=-5.0)
        with ops.pattern.Plain(series=ops.timeSeries.Linear()) as p:
            p.surface_load(ele=_SLAB_TAG)
        with pytest.raises(
            InertMembranePressureError, match="integer ele= tag is not matched",
        ):
            ops.build()

    def test_every_unhit_membrane_is_counted(self) -> None:
        ops, slab = _slab_on_beams(pressure=-5.0)
        ops.element.LadrunoMembrane(nodes=(1, 3, 9, 7), pressure=-2.0)
        ops.element.LadrunoMembrane(nodes=(1, 2, 5, 4), pressure=-3.0)
        with ops.pattern.Plain(series=ops.timeSeries.Linear()) as p:
            p.surface_load(ele=slab)
        with pytest.raises(
            InertMembranePressureError, match=r"pressure=-2\.0.* 1 more pressured",
        ):
            ops.build()

    def test_an_unknown_element_load_type_fails_closed(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        @dataclass(frozen=True)
        class _Unknown:
            load_type: str = "beamPoint"
            target_kind: str = "ele"
            target: tuple[int, ...] = (1,)

        ops, _slab = _deck_2143()
        with ops.pattern.Plain(series=ops.timeSeries.Linear()):
            pass
        monkeypatch.setattr(Plain, "ele_loads", property(lambda self: (_Unknown(),)))
        with pytest.raises(BridgeError, match="'beamPoint' is unknown"):
            ops.build()


class TestBuilds:
    def test_the_2143_deck_plus_surface_load_builds(self) -> None:
        ops, slab = _deck_2143()
        with ops.pattern.Plain(series=ops.timeSeries.Linear()) as p:
            p.surface_load(ele=slab)
        assert _ele_load_tags(ops) == [("-ele", 2, "-type", "-surfaceLoad")]

    @pytest.mark.parametrize("verb", ["surface_load", "self_weight", "beam_uniform"])
    def test_each_activating_verb_counts(self, verb: str) -> None:
        ops, slab = _deck_2143()
        with ops.pattern.Plain(series=ops.timeSeries.Linear()) as p:
            getattr(p, verb)(ele=slab)
        ops.build()

    def test_pg_target_counts(self) -> None:
        ops, _slab = _slab_on_beams(pressure=-5.0)
        with ops.pattern.Plain(series=ops.timeSeries.Linear()) as p:
            p.surface_load(pg="Slab")
        assert _ele_load_tags(ops) == [("-ele", _SLAB_TAG, "-type", "-surfaceLoad")]

    def test_a_stage_pattern_counts(self) -> None:
        ops, slab = _deck_2143()
        with ops.stage(name="load") as s:
            with s.pattern(series=ops.timeSeries.Linear()) as p:
                p.surface_load(ele=slab)
            s.analysis(
                test=ops.test.NormDispIncr(tol=1e-10, max_iter=10),
                algorithm=ops.algorithm.Linear(),
                integrator=ops.integrator.LoadControl(dlam=1.0),
                constraints=ops.constraints.Plain(),
                numberer=ops.numberer.Plain(),
                system=ops.system.BandGeneral(),
                analysis=ops.analysis.Static(),
            )
            s.run(n_increments=1, dt=1.0)
        assert _ele_load_tags(ops) == [("-ele", 2, "-type", "-surfaceLoad")]

    @pytest.mark.parametrize("pressure", [None, 0.0])
    def test_a_stiffness_only_membrane_builds(self, pressure: float | None) -> None:
        ops, _slab = _deck_2143(pressure=pressure)
        ops.build()

    def test_rho_with_thickness_is_mass_and_builds_without_a_pattern(self) -> None:
        """``rho`` x thickness is the membrane's mass (``LadrunoMembrane::getMass``),
        read by a modal run with no ``Plain`` pattern, so it is not refused."""
        ops, _slab = _slab_on_beams(E=30e9, nu=0.2, thickness=0.2, rho=2500.0)
        ops.build()
