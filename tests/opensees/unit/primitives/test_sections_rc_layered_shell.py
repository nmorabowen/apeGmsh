"""Unit tests for the RC layered-shell builder.

``RCLayeredShell`` composes a plain ``LayeredShell`` from a concrete law
and ``RebarMesh`` bar meshes: each mesh is its own thin ``PlateRebar``
layer (thickness ``A_s / s``) centred on the bar centroid, and the concrete
fills the gaps, reduced by the steel it gives way to. Covers the layer
arithmetic (sum, bar depth, concrete reduction, apportioning), ``PlateRebar``
dedup, validation, the coarse-layering warning, and the namespace method's
registration and emit order.
"""
from __future__ import annotations

import math
import random
import warnings
from typing import cast

import pytest

from apeGmsh.opensees import apeSees
from apeGmsh.opensees.emitter.py import PyEmitter
from apeGmsh.opensees.emitter.tcl import TclEmitter
from apeGmsh.opensees.material.nd import ElasticIsotropic, PlateRebar
from apeGmsh.opensees.material.uniaxial import ElasticMaterial, Steel01
from apeGmsh.opensees.section.plate import (
    MIN_RC_CONCRETE_LAYERS,
    CoarseShellLayeringWarning,
    LayeredShell,
    RCLayeredShell,
    RebarMesh,
    ShellLayer,
)

from tests.opensees.fixtures.fem_stub import (
    FEMStub,
    _ElementGroupView,
    _ElementsStub,
    _NodesStub,
)

H = 0.20
D12 = 113.1e-6          # one 12 mm bar, m^2
S = 0.15                # spacing, m
T = D12 / S             # smeared thickness, m


def _steel() -> Steel01:
    return Steel01(fy=420e6, E=200e9, b=0.01)


def _conc() -> ElasticIsotropic:
    return ElasticIsotropic(E=30e9, nu=0.2)


def _two_curtains(steel: Steel01) -> list[RebarMesh]:
    """Two-way, two-curtain wall: x outermost, y one bar inside it."""
    return [
        RebarMesh.from_bars(material=steel, angle=0.0, bar_area=D12,
                            spacing=S, cover=0.031, face="bottom"),
        RebarMesh.from_bars(material=steel, angle=90.0, bar_area=D12,
                            spacing=S, cover=0.043, face="bottom"),
        RebarMesh.from_bars(material=steel, angle=0.0, bar_area=D12,
                            spacing=S, cover=0.031, face="top"),
        RebarMesh.from_bars(material=steel, angle=90.0, bar_area=D12,
                            spacing=S, cover=0.043, face="top"),
    ]


def _midpoints(sec: LayeredShell, h: float) -> list[float]:
    """Layer mid-planes as LayeredShellFiberSection places them."""
    out, z = [], -h / 2
    for layer in sec.layers:
        out.append(z + layer.thickness / 2)
        z += layer.thickness
    return out


def _bar_layers(sec: LayeredShell) -> list[tuple[int, ShellLayer]]:
    return [(i, lay) for i, lay in enumerate(sec.layers)
            if isinstance(lay.material, PlateRebar)]


# ---------------------------------------------------------------------------
# RebarMesh
# ---------------------------------------------------------------------------

class TestRebarMesh:
    def test_from_bars_is_area_over_spacing(self) -> None:
        m = RebarMesh.from_bars(material=_steel(), angle=0.0, bar_area=D12,
                                spacing=S, z=0.0)
        assert m.area_per_width == pytest.approx(D12 / S, rel=1e-15)

    def test_centroid_from_z_and_from_cover(self) -> None:
        s = _steel()
        assert RebarMesh(material=s, angle=0.0, area_per_width=T,
                         z=0.01).centroid(H) == 0.01
        assert RebarMesh(material=s, angle=0.0, area_per_width=T,
                         cover=0.03, face="bottom").centroid(H) == pytest.approx(-0.07)
        assert RebarMesh(material=s, angle=0.0, area_per_width=T,
                         cover=0.03, face="top").centroid(H) == pytest.approx(0.07)

    def test_rejects_nd_material(self) -> None:
        with pytest.raises(TypeError, match="UniaxialMaterial"):
            RebarMesh(material=_conc(), angle=0.0, area_per_width=T, z=0.0)  # type: ignore[arg-type]

    @pytest.mark.parametrize("area", [0.0, -1e-4, math.inf, math.nan])
    def test_rejects_bad_area(self, area: float) -> None:
        with pytest.raises(ValueError, match="area_per_width must be finite and > 0"):
            RebarMesh(material=_steel(), angle=0.0, area_per_width=area, z=0.0)

    def test_rejects_non_finite_angle(self) -> None:
        with pytest.raises(ValueError, match="angle must be finite"):
            RebarMesh(material=_steel(), angle=math.nan, area_per_width=T, z=0.0)

    def test_z_and_cover_are_exclusive(self) -> None:
        with pytest.raises(ValueError, match="not both"):
            RebarMesh(material=_steel(), angle=0.0, area_per_width=T,
                      z=0.0, cover=0.03, face="top")

    def test_needs_a_position(self) -> None:
        with pytest.raises(ValueError, match="as z, or as cover and face"):
            RebarMesh(material=_steel(), angle=0.0, area_per_width=T)
        with pytest.raises(ValueError, match="as z, or as cover and face"):
            RebarMesh(material=_steel(), angle=0.0, area_per_width=T, cover=0.03)

    def test_rejects_bad_face_and_cover(self) -> None:
        with pytest.raises(ValueError, match="face must be"):
            RebarMesh(material=_steel(), angle=0.0, area_per_width=T,
                      cover=0.03, face="middle")  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="cover must be finite and >= 0"):
            RebarMesh(material=_steel(), angle=0.0, area_per_width=T,
                      cover=-0.01, face="top")

    @pytest.mark.parametrize("kw", [{"bar_area": 0.0}, {"spacing": -0.1},
                                    {"spacing": math.inf}])
    def test_from_bars_rejects_bad_inputs(self, kw: dict[str, float]) -> None:
        args = {"bar_area": D12, "spacing": S, **kw}
        with pytest.raises(ValueError, match="must be finite and > 0"):
            RebarMesh.from_bars(material=_steel(), angle=0.0, z=0.0, **args)


# ---------------------------------------------------------------------------
# RCLayeredShell — layer arithmetic
# ---------------------------------------------------------------------------

class TestLayerArithmetic:
    def test_thicknesses_sum_to_h(self) -> None:
        sec = RCLayeredShell(h=H, concrete=_conc(), meshes=_two_curtains(_steel()))
        assert math.fsum(lay.thickness for lay in sec.layers) == pytest.approx(H, rel=1e-14)
        assert sum(lay.thickness for lay in sec.layers) == pytest.approx(H, rel=1e-14)

    def test_each_bar_sits_at_its_centroid(self) -> None:
        meshes = _two_curtains(_steel())
        sec = RCLayeredShell(h=H, concrete=_conc(), meshes=meshes)
        mids = _midpoints(sec, H)
        got = [mids[i] for i, _ in _bar_layers(sec)]
        want = sorted(m.centroid(H) for m in meshes)
        assert got == pytest.approx(want, abs=1e-12)
        assert want == pytest.approx([-0.069, -0.057, 0.057, 0.069], abs=1e-12)

    def test_bar_layer_thickness_is_area_per_width(self) -> None:
        sec = RCLayeredShell(h=H, concrete=_conc(), meshes=_two_curtains(_steel()))
        assert [lay.thickness for _, lay in _bar_layers(sec)] == [T] * 4

    def test_concrete_is_reduced_by_the_steel(self) -> None:
        conc = _conc()
        sec = RCLayeredShell(h=H, concrete=conc, meshes=_two_curtains(_steel()))
        concrete = [lay.thickness for lay in sec.layers if lay.material is conc]
        assert math.fsum(concrete) == pytest.approx(H - 4 * T, rel=1e-13)
        assert all(t > 0 for t in concrete)

    def test_n_concrete_is_the_concrete_layer_count(self) -> None:
        conc = _conc()
        for n in (6, 10, 12):
            sec = RCLayeredShell(h=H, concrete=conc,
                                 meshes=_two_curtains(_steel()), n_concrete=n)
            assert sum(lay.material is conc for lay in sec.layers) == n
            assert len(sec.layers) == n + 4

    def test_every_concrete_region_gets_a_layer(self) -> None:
        # Five regions (below, three between, above); n below that is raised.
        conc = _conc()
        with pytest.warns(CoarseShellLayeringWarning):
            sec = RCLayeredShell(h=H, concrete=conc,
                                 meshes=_two_curtains(_steel()), n_concrete=2)
        kinds = ["c" if lay.material is conc else "s" for lay in sec.layers]
        assert kinds == ["c", "s", "c", "s", "c", "s", "c", "s", "c"]

    def test_no_meshes_is_equal_concrete_layers(self) -> None:
        conc = _conc()
        sec = RCLayeredShell(h=H, concrete=conc, n_concrete=8)
        assert len(sec.layers) == 8
        assert all(lay.material is conc for lay in sec.layers)
        assert [lay.thickness for lay in sec.layers] == pytest.approx([H / 8] * 8, rel=1e-13)

    def test_mesh_order_does_not_matter(self) -> None:
        steel, conc = _steel(), _conc()
        meshes = _two_curtains(steel)
        ref = RCLayeredShell(h=H, concrete=conc, meshes=meshes)
        shuffled = list(meshes)
        random.Random(7).shuffle(shuffled)
        got = RCLayeredShell(h=H, concrete=conc, meshes=shuffled)
        assert [lay.thickness for lay in got.layers] == [lay.thickness for lay in ref.layers]
        assert [getattr(lay.material, "angle", None) for lay in got.layers] == \
            [getattr(lay.material, "angle", None) for lay in ref.layers]

    def test_touching_bar_layers_have_no_concrete_between(self) -> None:
        steel, conc = _steel(), _conc()
        z1 = -0.06
        meshes = [
            RebarMesh(material=steel, angle=0.0, area_per_width=T, z=z1),
            RebarMesh(material=steel, angle=90.0, area_per_width=T, z=z1 + T),
        ]
        sec = RCLayeredShell(h=H, concrete=conc, meshes=meshes)
        (i, _), (j, _) = _bar_layers(sec)
        assert j == i + 1
        assert math.fsum(lay.thickness for lay in sec.layers) == pytest.approx(H, rel=1e-14)

    def test_returns_a_plain_layered_shell(self) -> None:
        sec = RCLayeredShell(h=H, concrete=_conc(), meshes=_two_curtains(_steel()))
        assert type(sec) is LayeredShell


# ---------------------------------------------------------------------------
# PlateRebar dedup
# ---------------------------------------------------------------------------

class TestPlateRebarDedup:
    def test_same_material_and_angle_share_one_plate_rebar(self) -> None:
        steel, conc = _steel(), _conc()
        sec = RCLayeredShell(h=H, concrete=conc, meshes=_two_curtains(steel))
        bars = [lay.material for _, lay in _bar_layers(sec)]
        by_angle: dict[float, set[int]] = {}
        for b in bars:
            assert isinstance(b, PlateRebar)
            assert b.material is steel
            by_angle.setdefault(b.angle, set()).add(id(b))
        assert {a: len(ids) for a, ids in by_angle.items()} == {0.0: 1, 90.0: 1}
        # concrete + one PlateRebar per direction
        assert len(sec.dependencies()) == 3

    def test_different_steel_instances_do_not_share(self) -> None:
        a, b = _steel(), _steel()
        sec = RCLayeredShell(h=H, concrete=_conc(), meshes=[
            RebarMesh(material=a, angle=0.0, area_per_width=T, z=-0.06),
            RebarMesh(material=b, angle=0.0, area_per_width=T, z=0.06),
        ])
        bars = [lay.material for _, lay in _bar_layers(sec)]
        assert bars[0] is not bars[1]


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

class TestValidation:
    @pytest.mark.parametrize("z", [-0.0998, 0.0998])
    def test_bar_outside_the_section(self, z: float) -> None:
        with pytest.raises(ValueError, match="outside the section"):
            RCLayeredShell(h=H, concrete=_conc(), meshes=[
                RebarMesh(material=_steel(), angle=0.0, area_per_width=T, z=z)])

    def test_cover_smaller_than_half_the_layer(self) -> None:
        with pytest.raises(ValueError, match="outside the section"):
            RCLayeredShell(h=H, concrete=_conc(), meshes=[
                RebarMesh(material=_steel(), angle=0.0, area_per_width=T,
                          cover=0.0, face="top")])

    def test_overlapping_bar_layers(self) -> None:
        steel = _steel()
        with pytest.raises(ValueError, match=r"meshes\[0\].*meshes\[1\].*overlap"):
            RCLayeredShell(h=H, concrete=_conc(), meshes=[
                RebarMesh(material=steel, angle=0.0, area_per_width=T, z=0.05),
                RebarMesh(material=steel, angle=90.0, area_per_width=T, z=0.05),
            ])

    def test_no_concrete_left(self) -> None:
        with pytest.raises(ValueError, match="no concrete is left"):
            RCLayeredShell(h=H, concrete=_conc(), meshes=[
                RebarMesh(material=_steel(), angle=0.0, area_per_width=H, z=0.0)])

    @pytest.mark.parametrize("h", [0.0, -0.2, math.nan, math.inf])
    def test_bad_h(self, h: float) -> None:
        with pytest.raises(ValueError, match="h must be finite and > 0"):
            RCLayeredShell(h=h, concrete=_conc())

    @pytest.mark.parametrize("n", [0, -3, True, 8.0])
    def test_bad_n_concrete(self, n: object) -> None:
        with pytest.raises(ValueError, match="n_concrete must be an int >= 1"):
            RCLayeredShell(h=H, concrete=_conc(), n_concrete=n)  # type: ignore[arg-type]

    def test_uniaxial_concrete_points_to_plate_rebar(self) -> None:
        with pytest.raises(TypeError, match="PlateRebar"):
            RCLayeredShell(h=H, concrete=ElasticMaterial(E=30e9))  # type: ignore[arg-type]

    def test_mesh_must_be_a_rebar_mesh(self) -> None:
        with pytest.raises(TypeError, match=r"meshes\[0\] must be a RebarMesh"):
            RCLayeredShell(h=H, concrete=_conc(), meshes=[(_steel(), 0.0)])  # type: ignore[list-item]


class TestCoarseLayeringWarning:
    def test_warns_below_the_minimum(self) -> None:
        with pytest.warns(CoarseShellLayeringWarning, match="5 concrete layers"):
            RCLayeredShell(h=H, concrete=_conc(),
                           n_concrete=MIN_RC_CONCRETE_LAYERS - 1)

    def test_silent_at_the_minimum(self) -> None:
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            RCLayeredShell(h=H, concrete=_conc(), meshes=_two_curtains(_steel()),
                           n_concrete=MIN_RC_CONCRETE_LAYERS)


# ---------------------------------------------------------------------------
# ops.section.RCLayeredShell — registration and emit order
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


class TestNamespace:
    def _model(self) -> tuple[apeSees, Steel01, ElasticIsotropic, LayeredShell]:
        ops = apeSees(cast("object", _one_quad()))  # type: ignore[arg-type]
        ops.model(ndm=3, ndf=6)
        conc = ops.nDMaterial.ElasticIsotropic(E=30e9, nu=0.2, name="conc")
        steel = ops.uniaxialMaterial.Steel01(fy=420e6, E=200e9, b=0.01)
        sec = ops.section.RCLayeredShell(
            h=H, concrete="conc", meshes=_two_curtains(steel), name="wall",
        )
        ops.element.ASDShellQ4(pg="Wall", section=sec, local_cs=(1.0, 0.0, 0.0))
        return ops, steel, conc, sec

    def test_registers_the_new_plate_rebars_and_the_section(self) -> None:
        ops, _steel_, conc, sec = self._model()
        assert isinstance(sec, LayeredShell)
        assert ops.tag_for(sec) is not None
        bars = {id(lay.material): lay.material for _, lay in _bar_layers(sec)}
        assert len(bars) == 2
        for bar in bars.values():
            assert ops.tag_for(bar) is not None
        assert all(lay.material is conc for lay in sec.layers
                   if not isinstance(lay.material, PlateRebar))

    @pytest.mark.parametrize("emitter_cls", [TclEmitter, PyEmitter])
    def test_emit_order_and_dedup(self, emitter_cls: type) -> None:
        ops, steel, conc, sec = self._model()
        emitter = emitter_cls()
        ops.build().emit(emitter)
        lines = emitter.lines()
        tcl = emitter_cls is TclEmitter

        def find(pred: str) -> list[int]:
            return [i for i, ln in enumerate(lines) if pred in ln]

        steel_tag, conc_tag, sec_tag = (ops.tag_for(p) for p in (steel, conc, sec))
        if tcl:
            steel_at = find(f"uniaxialMaterial Steel01 {steel_tag} ")
            conc_at = find(f"nDMaterial ElasticIsotropic {conc_tag} ")
            bar_at = find("nDMaterial PlateRebar ")
            sec_at = find(f"section LayeredShell {sec_tag} 14 ")
        else:
            steel_at = find(f"ops.uniaxialMaterial('Steel01', {steel_tag},")
            conc_at = find(f"ops.nDMaterial('ElasticIsotropic', {conc_tag},")
            bar_at = find("ops.nDMaterial('PlateRebar',")
            sec_at = find(f"ops.section('LayeredShell', {sec_tag}, 14,")
        assert len(steel_at) == len(conc_at) == len(sec_at) == 1
        # One PlateRebar line per (steel, angle), not per mesh.
        assert len(bar_at) == 2
        assert steel_at[0] < min(bar_at)
        assert max(bar_at + conc_at) < sec_at[0]
        for i in bar_at:
            # nDMaterial PlateRebar <tag> <uniaxialTag> <angle>
            args = (lines[i].split()[2:] if tcl
                    else [a.strip(" )") for a in lines[i].split(",")[1:]])
            assert args[1] == str(steel_tag)
            assert float(args[2]) in (0.0, 90.0)

    def test_section_line_lists_the_bars_at_their_depths(self) -> None:
        ops, _steel_, _conc_, sec = self._model()
        emitter = TclEmitter()
        ops.build().emit(emitter)
        (line,) = [ln for ln in emitter.lines() if ln.startswith("section LayeredShell")]
        tokens = line.split()[4:]
        pairs = list(zip(tokens[0::2], tokens[1::2]))
        assert len(pairs) == 14
        bar_tags = {str(ops.tag_for(lay.material)) for _, lay in _bar_layers(sec)}
        assert [i for i, (tag, _t) in enumerate(pairs) if tag in bar_tags] == \
            [i for i, _ in _bar_layers(sec)]
