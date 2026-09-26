"""Unit tests for the shell-layer helper nD materials.

``PlateRebar`` / ``PlateFromPlaneStress`` (PlateFiber, order 5 — valid
``LayeredShell*`` layers) and ``PlaneStressRebar`` (PlaneStress, order 3 —
not a layer). Covers validation, ``_emit`` through the recording, Tcl and
Python emitters, ``dependencies``, the namespace methods, and the emit order
of a layered section built from concrete + two ``PlateRebar`` layers.
"""
from __future__ import annotations

import math
from typing import cast

import pytest

from apeGmsh.opensees import apeSees
from apeGmsh.opensees._internal.tag_resolution import set_tag_resolver
from apeGmsh.opensees.emitter.py import PyEmitter
from apeGmsh.opensees.emitter.recording import RecordingEmitter
from apeGmsh.opensees.emitter.tcl import TclEmitter
from apeGmsh.opensees.material.nd import (
    ElasticIsotropic,
    PlaneStressRebar,
    PlateFromPlaneStress,
    PlateRebar,
)
from apeGmsh.opensees.material.uniaxial import ElasticMaterial
from apeGmsh.opensees.section.plate import (
    LayeredShell,
    LayeredShellFiberSection,
    ShellLayer,
)

from tests.opensees.fixtures.fem_stub import (
    FEMStub,
    _ElementGroupView,
    _ElementsStub,
    _NodesStub,
)


def _steel() -> ElasticMaterial:
    return ElasticMaterial(E=200e9)


def _emit_one(prim: object, emitter: object, dep: object, dep_tag: int, tag: int) -> None:
    set_tag_resolver(emitter, lambda p: dep_tag if p is dep else 0)  # type: ignore[arg-type]
    prim._emit(emitter, tag=tag)  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# PlateRebar
# ---------------------------------------------------------------------------

class TestPlateRebar:
    def test_dependencies_is_the_uniaxial(self) -> None:
        s = _steel()
        assert PlateRebar(material=s, angle=0.0).dependencies() == (s,)

    def test_emit_records_call(self) -> None:
        s = _steel()
        rec = RecordingEmitter()
        _emit_one(PlateRebar(material=s, angle=90.0), rec, s, 3, 7)
        assert rec.calls == [("nDMaterial", ("PlateRebar", 7, 3, 90.0), {})]

    def test_tcl_line(self) -> None:
        s = _steel()
        e = TclEmitter()
        _emit_one(PlateRebar(material=s, angle=90.0), e, s, 3, 7)
        assert "nDMaterial PlateRebar 7 3 90.0" in e.lines()

    def test_py_line(self) -> None:
        s = _steel()
        e = PyEmitter()
        _emit_one(PlateRebar(material=s, angle=90.0), e, s, 3, 7)
        assert "ops.nDMaterial('PlateRebar', 7, 3, 90.0)" in e.lines()

    def test_rejects_non_uniaxial_material(self) -> None:
        with pytest.raises(TypeError, match="material must be a UniaxialMaterial"):
            PlateRebar(material=ElasticIsotropic(E=30e9, nu=0.2), angle=0.0)  # type: ignore[arg-type]

    @pytest.mark.parametrize("angle", [math.nan, math.inf, -math.inf])
    def test_rejects_non_finite_angle(self, angle: float) -> None:
        with pytest.raises(ValueError, match="angle must be finite"):
            PlateRebar(material=_steel(), angle=angle)

    def test_negative_and_large_angles_are_accepted(self) -> None:
        PlateRebar(material=_steel(), angle=-45.0)
        PlateRebar(material=_steel(), angle=450.0)

    def test_repr_includes_type_token(self) -> None:
        assert "PlateRebar" in repr(PlateRebar(material=_steel(), angle=0.0))


# ---------------------------------------------------------------------------
# PlateFromPlaneStress
# ---------------------------------------------------------------------------

class TestPlateFromPlaneStress:
    def test_dependencies_is_the_wrapped_material(self) -> None:
        c = ElasticIsotropic(E=30e9, nu=0.2)
        assert PlateFromPlaneStress(material=c, G_out=12.5e9).dependencies() == (c,)

    def test_emit_records_call(self) -> None:
        c = ElasticIsotropic(E=30e9, nu=0.2)
        rec = RecordingEmitter()
        _emit_one(PlateFromPlaneStress(material=c, G_out=12.5e9), rec, c, 1, 2)
        assert rec.calls == [
            ("nDMaterial", ("PlateFromPlaneStress", 2, 1, 12.5e9), {}),
        ]

    def test_tcl_line(self) -> None:
        c = ElasticIsotropic(E=30e9, nu=0.2)
        e = TclEmitter()
        _emit_one(PlateFromPlaneStress(material=c, G_out=12.5e9), e, c, 1, 2)
        assert "nDMaterial PlateFromPlaneStress 2 1 12500000000.0" in e.lines()

    def test_py_line(self) -> None:
        c = ElasticIsotropic(E=30e9, nu=0.2)
        e = PyEmitter()
        _emit_one(PlateFromPlaneStress(material=c, G_out=12.5e9), e, c, 1, 2)
        assert "ops.nDMaterial('PlateFromPlaneStress', 2, 1, 12500000000.0)" in e.lines()

    @pytest.mark.parametrize("G_out", [0.0, -1.0])
    def test_rejects_non_positive_G_out(self, G_out: float) -> None:
        with pytest.raises(ValueError, match="G_out must be > 0"):
            PlateFromPlaneStress(material=ElasticIsotropic(E=30e9, nu=0.2), G_out=G_out)

    @pytest.mark.parametrize("G_out", [math.nan, math.inf])
    def test_rejects_non_finite_G_out(self, G_out: float) -> None:
        with pytest.raises(ValueError, match="G_out must be finite"):
            PlateFromPlaneStress(material=ElasticIsotropic(E=30e9, nu=0.2), G_out=G_out)

    def test_rejects_non_nd_material(self) -> None:
        with pytest.raises(TypeError, match="material must be an NDMaterial"):
            PlateFromPlaneStress(material=_steel(), G_out=1.0)  # type: ignore[arg-type]

    def test_rejects_plate_fiber_only_inner(self) -> None:
        bar = PlateRebar(material=_steel(), angle=0.0)
        with pytest.raises(TypeError, match="plane-stress view"):
            PlateFromPlaneStress(material=bar, G_out=1.0)
        wrapped = PlateFromPlaneStress(material=ElasticIsotropic(E=30e9, nu=0.2), G_out=1.0)
        with pytest.raises(TypeError, match="plane-stress view"):
            PlateFromPlaneStress(material=wrapped, G_out=1.0)


# ---------------------------------------------------------------------------
# PlaneStressRebar
# ---------------------------------------------------------------------------

class TestPlaneStressRebar:
    def test_dependencies_is_the_uniaxial(self) -> None:
        s = _steel()
        assert PlaneStressRebar(material=s, angle=45.0).dependencies() == (s,)

    def test_emit_uses_the_material_keyword(self) -> None:
        # The class is PlaneStressRebar; the only keyword every Tcl
        # interpreter registers is PlaneStressRebarMaterial.
        s = _steel()
        rec = RecordingEmitter()
        _emit_one(PlaneStressRebar(material=s, angle=45.0), rec, s, 4, 9)
        assert rec.calls == [
            ("nDMaterial", ("PlaneStressRebarMaterial", 9, 4, 45.0), {}),
        ]

    def test_tcl_line(self) -> None:
        s = _steel()
        e = TclEmitter()
        _emit_one(PlaneStressRebar(material=s, angle=45.0), e, s, 4, 9)
        assert "nDMaterial PlaneStressRebarMaterial 9 4 45.0" in e.lines()

    def test_py_line(self) -> None:
        s = _steel()
        e = PyEmitter()
        _emit_one(PlaneStressRebar(material=s, angle=45.0), e, s, 4, 9)
        assert "ops.nDMaterial('PlaneStressRebarMaterial', 9, 4, 45.0)" in e.lines()

    def test_rejects_non_uniaxial_material(self) -> None:
        with pytest.raises(TypeError, match="material must be a UniaxialMaterial"):
            PlaneStressRebar(material=ElasticIsotropic(E=30e9, nu=0.2), angle=0.0)  # type: ignore[arg-type]

    def test_rejects_non_finite_angle(self) -> None:
        with pytest.raises(ValueError, match="angle must be finite"):
            PlaneStressRebar(material=_steel(), angle=math.nan)

    def test_is_refused_as_a_shell_layer(self) -> None:
        # LayeredShellFiberSection would call exit(-1) on its null
        # getCopy("PlateFiber") answer.
        with pytest.raises(TypeError, match="Use PlateRebar"):
            ShellLayer(material=PlaneStressRebar(material=_steel(), angle=0.0), thickness=0.001)


# ---------------------------------------------------------------------------
# Namespace
# ---------------------------------------------------------------------------

def _stub_bridge() -> apeSees:
    from unittest.mock import MagicMock
    return apeSees(cast("object", MagicMock(name="FEMData")))  # type: ignore[arg-type]


class TestNamespace:
    def test_PlateRebar_registers_after_its_uniaxial(self) -> None:
        ops = _stub_bridge()
        steel = ops.uniaxialMaterial.ElasticMaterial(E=200e9)
        bar = ops.nDMaterial.PlateRebar(material=steel, angle=0.0)
        assert isinstance(bar, PlateRebar)
        assert bar.material is steel
        assert ops.tag_for(bar) == 1

    def test_PlateRebar_accepts_uniaxial_by_name(self) -> None:
        ops = _stub_bridge()
        steel = ops.uniaxialMaterial.ElasticMaterial(E=200e9, name="steel")
        bar = ops.nDMaterial.PlateRebar(material="steel", angle=90.0)
        assert bar.material is steel

    def test_PlateFromPlaneStress_wraps_nd(self) -> None:
        ops = _stub_bridge()
        conc = ops.nDMaterial.ElasticIsotropic(E=30e9, nu=0.2, name="conc")
        layer = ops.nDMaterial.PlateFromPlaneStress(material="conc", G_out=12.5e9)
        assert isinstance(layer, PlateFromPlaneStress)
        assert layer.material is conc
        assert ops.tag_for(conc) == 1
        assert ops.tag_for(layer) == 2

    def test_PlaneStressRebar_registers(self) -> None:
        ops = _stub_bridge()
        steel = ops.uniaxialMaterial.ElasticMaterial(E=200e9)
        m = ops.nDMaterial.PlaneStressRebar(material=steel, angle=45.0)
        assert isinstance(m, PlaneStressRebar)
        assert m.angle == 45.0


# ---------------------------------------------------------------------------
# Layered section: concrete + two PlateRebar layers, emitted end to end
# ---------------------------------------------------------------------------

def _one_quad() -> FEMStub:
    nodes = _NodesStub(
        ids=[1, 2, 3, 4],
        coords=[(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (1.0, 1.0, 0.0), (0.0, 1.0, 0.0)],
        node_pgs={"All": [1, 2, 3, 4]},
    )
    elements = _ElementsStub(
        elem_pgs={"Wall": _ElementGroupView(ids=(1,), connectivity=((1, 2, 3, 4),))},
    )
    return FEMStub(nodes=nodes, elements=elements)


@pytest.mark.parametrize(
    ("section_ns", "token"),
    [("LayeredShellFiberSection", "LayeredShellFiberSection"),
     ("LayeredShell", "LayeredShell")],
)
def test_layered_section_with_two_plate_rebar_layers_emits_in_order(
    section_ns: str, token: str,
) -> None:
    ops = apeSees(cast("object", _one_quad()))  # type: ignore[arg-type]
    ops.model(ndm=3, ndf=6)
    # The deck order comes from the dependency walk, not from registration
    # order; every tag a line references must be emitted before it.
    conc = ops.nDMaterial.ElasticIsotropic(E=30e9, nu=0.2)
    conc_layer = ops.nDMaterial.PlateFromPlaneStress(material=conc, G_out=12.5e9)
    steel = ops.uniaxialMaterial.ElasticMaterial(E=200e9)
    bar_x = ops.nDMaterial.PlateRebar(material=steel, angle=0.0)
    bar_y = ops.nDMaterial.PlateRebar(material=steel, angle=90.0)
    sec = getattr(ops.section, section_ns)(layers=(
        ShellLayer(material=conc_layer, thickness=0.1),
        ShellLayer(material=bar_x, thickness=0.002),
        ShellLayer(material=bar_y, thickness=0.002),
        ShellLayer(material=conc_layer, thickness=0.1),
    ))
    assert isinstance(sec, (LayeredShell, LayeredShellFiberSection))
    ops.element.ASDShellQ4(pg="Wall", section=sec)

    for emitter in (TclEmitter(), PyEmitter()):
        ops.build().emit(emitter)
        lines = emitter.lines()
        tcl = isinstance(emitter, TclEmitter)
        t_conc, t_layer = ops.tag_for(conc), ops.tag_for(conc_layer)
        t_steel = ops.tag_for(steel)
        t_x, t_y, t_sec = ops.tag_for(bar_x), ops.tag_for(bar_y), ops.tag_for(sec)
        if tcl:
            want = [
                f"uniaxialMaterial Elastic {t_steel} 200000000000.0",
                f"nDMaterial ElasticIsotropic {t_conc} 30000000000.0 0.2 0.0",
                f"nDMaterial PlateFromPlaneStress {t_layer} {t_conc} 12500000000.0",
                f"nDMaterial PlateRebar {t_x} {t_steel} 0.0",
                f"nDMaterial PlateRebar {t_y} {t_steel} 90.0",
                f"section {token} {t_sec} 4 {t_layer} 0.1 {t_x} 0.002 "
                f"{t_y} 0.002 {t_layer} 0.1",
            ]
        else:
            want = [
                f"ops.uniaxialMaterial('Elastic', {t_steel}, 200000000000.0)",
                f"ops.nDMaterial('ElasticIsotropic', {t_conc}, 30000000000.0, 0.2, 0.0)",
                f"ops.nDMaterial('PlateFromPlaneStress', {t_layer}, {t_conc}, 12500000000.0)",
                f"ops.nDMaterial('PlateRebar', {t_x}, {t_steel}, 0.0)",
                f"ops.nDMaterial('PlateRebar', {t_y}, {t_steel}, 90.0)",
                f"ops.section('{token}', {t_sec}, 4, {t_layer}, 0.1, {t_x}, 0.002, "
                f"{t_y}, 0.002, {t_layer}, 0.1)",
            ]
        for line in want:
            assert line in lines, (line, lines)
        at = {line: lines.index(line) for line in want}
        # Each wrapper after what it references; the section after all.
        assert at[want[0]] < at[want[3]] and at[want[0]] < at[want[4]]
        assert at[want[1]] < at[want[2]]
        assert max(at[w] for w in want[:5]) < at[want[5]]
