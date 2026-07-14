"""Unit tests for the ``LadrunoUP`` typed element (ADR 0074).

Covers:

* Construction validation mirroring the fork parser's fatality matrix
  (OPS_LadrunoUP.cpp): required kwargs, perm XOR permH/gammaW pairing,
  storage-domain police (0 < n <= alpha <= 1), thick-is-2D-only, the
  ``-stab`` grammar, dynSeepage / geom legality.
* ``_emit`` kwarg -> token mapping, including pass-through elision
  (``None`` kwargs are never emitted; std/linear defaults elided),
  the automatic ``-pOrder linear`` on tri6/tet10 fan-outs, shape
  dispatch by node count, and the perm-dimension cross-check.
* Capability-registry surface: per-slot ndf floors, strictness, the
  ndm-dependent builder-ndf gate, scalar floors and ndf_ok.
* Bridge-namespace integration (``ops.element.LadrunoUP``).
"""
from __future__ import annotations

from typing import cast
from unittest.mock import MagicMock

import pytest

from apeGmsh.opensees import apeSees
from apeGmsh.opensees._element_capabilities import (
    _ELEM_REGISTRY,
    element_builder_ndf,
    element_class_ndf_ok,
    element_ndf_slot_floors,
    element_ndf_strict,
    element_required_floor,
)
from apeGmsh.opensees._internal.tag_resolution import (
    set_element_nodes,
    set_tag_resolver,
)
from apeGmsh.opensees._internal.types import Primitive
from apeGmsh.opensees.element.solid import LadrunoUP
from apeGmsh.opensees.emitter.recording import RecordingEmitter
from apeGmsh.opensees.material.nd import ElasticIsotropic


def _make_material() -> ElasticIsotropic:
    return ElasticIsotropic(E=30e6, nu=0.3, rho=2.0)


def _resolver_for(material: object, tag: int) -> object:
    def _resolve(prim: Primitive) -> int:
        if id(prim) == id(material):
            return tag
        raise KeyError(f"unexpected primitive {prim!r}")
    return _resolve


def _emit_with(
    elem: Primitive,
    *,
    tag: int,
    nodes: tuple[int, ...],
    mat_tag: int,
    material: object,
) -> RecordingEmitter:
    e = RecordingEmitter()
    set_tag_resolver(e, _resolver_for(material, mat_tag))
    set_element_nodes(e, nodes)
    elem._emit(e, tag=tag)  # type: ignore[attr-defined]
    return e


def _up2d(m: ElasticIsotropic, **kw: object) -> LadrunoUP:
    """A known-good minimal 2D LadrunoUP with overridable kwargs."""
    base: dict[str, object] = dict(
        pg="Soil", material=m, Kf=2.2e6, poro=0.4, rhoF=1.0,
        perm=(1e-8, 1e-8),
    )
    base.update(kw)
    return LadrunoUP(**base)  # type: ignore[arg-type]


def _up3d(m: ElasticIsotropic, **kw: object) -> LadrunoUP:
    base: dict[str, object] = dict(
        pg="Soil", material=m, Kf=2.2e6, poro=0.4, rhoF=1.0,
        perm=(1e-8, 1e-8, 1e-8),
    )
    base.update(kw)
    return LadrunoUP(**base)  # type: ignore[arg-type]


# ===========================================================================
# Construction validation
# ===========================================================================

class TestLadrunoUPConstruction:
    def test_minimal_2d(self) -> None:
        m = _make_material()
        e = _up2d(m)
        assert e.pg == "Soil" and e.material is m and e.perm_dim == 2
        assert e.dependencies() == (m,)
        assert "LadrunoUP" in repr(e)

    def test_minimal_3d(self) -> None:
        m = _make_material()
        assert _up3d(m).perm_dim == 3

    def test_is_frozen(self) -> None:
        from dataclasses import FrozenInstanceError

        e = _up2d(_make_material())
        with pytest.raises(FrozenInstanceError):
            e.pg = "Other"  # type: ignore[misc]

    @pytest.mark.parametrize("kf", [0.0, -1.0])
    def test_rejects_non_positive_kf(self, kf: float) -> None:
        with pytest.raises(ValueError, match="Kf must be > 0"):
            _up2d(_make_material(), Kf=kf)

    @pytest.mark.parametrize("n", [0.0, -0.1, 1.1])
    def test_rejects_bad_poro(self, n: float) -> None:
        with pytest.raises(ValueError, match="0 < n <= 1"):
            _up2d(_make_material(), poro=n)

    def test_rejects_negative_rhof(self) -> None:
        with pytest.raises(ValueError, match="rhoF must be >= 0"):
            _up2d(_make_material(), rhoF=-1.0)

    def test_permeability_is_required(self) -> None:
        with pytest.raises(ValueError, match="permeability is REQUIRED"):
            LadrunoUP(
                pg="S", material=_make_material(),
                Kf=2.2e6, poro=0.4, rhoF=1.0,
            )

    def test_perm_and_permh_are_exclusive(self) -> None:
        with pytest.raises(ValueError, match="not both"):
            _up2d(
                _make_material(),
                permH=(1e-5, 1e-5), gammaW=9.81,
            )

    def test_permh_requires_gammaw(self) -> None:
        with pytest.raises(ValueError, match="come together"):
            LadrunoUP(
                pg="S", material=_make_material(),
                Kf=2.2e6, poro=0.4, rhoF=1.0, permH=(1e-5, 1e-5),
            )

    def test_gammaw_requires_permh(self) -> None:
        with pytest.raises(ValueError, match="come together"):
            _up2d(_make_material(), gammaW=9.81)

    def test_rejects_non_positive_gammaw(self) -> None:
        with pytest.raises(ValueError, match="gammaW must be > 0"):
            LadrunoUP(
                pg="S", material=_make_material(),
                Kf=2.2e6, poro=0.4, rhoF=1.0,
                permH=(1e-5, 1e-5), gammaW=0.0,
            )

    @pytest.mark.parametrize("kvec", [(1e-8,), (1e-8,) * 4])
    def test_rejects_bad_perm_arity(self, kvec: tuple) -> None:
        with pytest.raises(ValueError, match="2 .2D. or 3 .3D."):
            _up2d(_make_material(), perm=kvec)

    def test_rejects_negative_perm_component(self) -> None:
        with pytest.raises(ValueError, match=">= 0"):
            _up2d(_make_material(), perm=(1e-8, -1e-8))

    @pytest.mark.parametrize("a", [0.0, -0.5, 1.5])
    def test_rejects_bad_alpha(self, a: float) -> None:
        with pytest.raises(ValueError, match="0 < alpha <= 1"):
            _up2d(_make_material(), alpha=a)

    def test_storage_police_poro_gt_alpha(self) -> None:
        with pytest.raises(ValueError, match="exceeds Biot"):
            _up2d(_make_material(), poro=0.6, alpha=0.5)

    def test_rejects_non_positive_thick(self) -> None:
        with pytest.raises(ValueError, match="thick must be > 0"):
            _up2d(_make_material(), thick=0.0)

    def test_thick_is_2d_only(self) -> None:
        with pytest.raises(ValueError, match="2D only"):
            _up3d(_make_material(), thick=1.0)

    def test_body_arity_must_match_perm(self) -> None:
        with pytest.raises(ValueError, match="body needs 2 components"):
            _up2d(_make_material(), body=(0.0, 0.0, -9.81))

    def test_fluidbody_arity_must_match_perm(self) -> None:
        with pytest.raises(ValueError, match="fluidBody needs 3 components"):
            _up3d(_make_material(), fluidBody=(0.0, -9.81))

    def test_rejects_non_up_formulation(self) -> None:
        with pytest.raises(ValueError, match="not u-p axes"):
            _up2d(_make_material(), formulation="ssp")

    @pytest.mark.parametrize(
        "stab", ["auto", "off", 6.8e-5, 0.0, ("auto", 0.25), ("auto", 0.1)],
    )
    def test_accepts_legal_stab(self, stab: object) -> None:
        assert _up2d(_make_material(), stab=stab).stab == stab

    @pytest.mark.parametrize(
        "stab, frag",
        [
            (("auto", 0.0), "alpha0 > 0"),
            (("auto", -0.1), "alpha0 > 0"),
            (("off", 0.1), r"\('auto', alpha0\)"),
            (-1e-5, "anti-stabilizes"),
            ("on", "manual numeric alpha"),
            (True, "manual numeric alpha"),
        ],
    )
    def test_rejects_bad_stab(self, stab: object, frag: str) -> None:
        with pytest.raises(ValueError, match=frag):
            _up2d(_make_material(), stab=stab)

    @pytest.mark.parametrize("ds", ["on", "off", True, False])
    def test_accepts_legal_dynseepage(self, ds: object) -> None:
        assert _up2d(_make_material(), dynSeepage=ds).dynSeepage == ds

    def test_rejects_bad_dynseepage(self) -> None:
        with pytest.raises(ValueError, match="dynSeepage"):
            _up2d(_make_material(), dynSeepage="maybe")

    def test_geom_accepts_only_linear(self) -> None:
        assert _up2d(_make_material(), geom="linear").geom == "linear"
        with pytest.raises(ValueError, match="only 'linear'"):
            _up2d(_make_material(), geom="corot")


# ===========================================================================
# Emission — kwarg -> token mapping
# ===========================================================================

class TestLadrunoUPEmission:
    def test_emit_q4_minimal_pass_through(self) -> None:
        """None kwargs emit NOTHING — the parser's defaults are the single
        source of truth (no -alpha/-Ks/-stab/-dynSeepage/-formulation/-geom
        and no -pOrder on an equal-order shape)."""
        m = _make_material()
        elem = _up2d(m)
        rec = _emit_with(
            elem, tag=7, nodes=(1, 2, 3, 4), mat_tag=9, material=m,
        )
        assert rec.calls == [
            (
                "element",
                (
                    "LadrunoUP", 7, 1, 2, 3, 4, 9,
                    "-Kf", 2.2e6, "-poro", 0.4, "-rhoF", 1.0,
                    "-perm", 1e-8, 1e-8,
                ),
                {},
            )
        ]

    def test_emit_q4_full_options(self) -> None:
        m = _make_material()
        elem = _up2d(
            m,
            thick=0.5, alpha=0.9, Ks=1e9,
            body=(0.0, -9.81), fluidBody=(0.0, -9.81),
            formulation="bbar", lumped=True, stab=("auto", 0.3),
            dynSeepage="on", geom="linear",
        )
        rec = _emit_with(
            elem, tag=3, nodes=(11, 12, 13, 14), mat_tag=2, material=m,
        )
        assert rec.calls == [
            (
                "element",
                (
                    "LadrunoUP", 3, 11, 12, 13, 14, 2,
                    "-thick", 0.5,
                    "-Kf", 2.2e6, "-poro", 0.4, "-rhoF", 1.0,
                    "-perm", 1e-8, 1e-8,
                    "-alpha", 0.9, "-Ks", 1e9,
                    "-body", 0.0, -9.81,
                    "-fluidBody", 0.0, -9.81,
                    "-formulation", "bbar",
                    "-lumped",
                    "-stab", "auto", 0.3,
                    "-dynSeepage", "on",
                ),
                {},
            )
        ]

    def test_emit_permh_gammaw_sugar(self) -> None:
        m = _make_material()
        elem = LadrunoUP(
            pg="Soil", material=m, Kf=2.2e6, poro=0.4, rhoF=1.0,
            permH=(1e-5, 2e-5), gammaW=9.81,
        )
        rec = _emit_with(
            elem, tag=1, nodes=(1, 2, 3), mat_tag=4, material=m,
        )
        args = rec.calls[0][1]
        i = args.index("-permH")
        assert args[i:i + 5] == ("-permH", 1e-5, 2e-5, "-gammaW", 9.81)
        assert "-perm" not in args

    @pytest.mark.parametrize(
        "stab, tail",
        [
            ("auto", ("-stab", "auto")),
            ("off", ("-stab", "off")),
            (6.8e-5, ("-stab", 6.8e-5)),
            (("auto", 0.5), ("-stab", "auto", 0.5)),
        ],
    )
    def test_emit_stab_variants(self, stab: object, tail: tuple) -> None:
        m = _make_material()
        elem = _up2d(m, stab=stab)
        rec = _emit_with(
            elem, tag=1, nodes=(1, 2, 3, 4), mat_tag=4, material=m,
        )
        args = rec.calls[0][1]
        i = args.index("-stab")
        assert args[i:i + len(tail)] == tail

    @pytest.mark.parametrize(
        "ds, token", [(True, "on"), ("on", "on"), (False, "off"), ("off", "off")],
    )
    def test_emit_dynseepage_normalized(self, ds: object, token: str) -> None:
        m = _make_material()
        elem = _up2d(m, dynSeepage=ds)
        rec = _emit_with(
            elem, tag=1, nodes=(1, 2, 3, 4), mat_tag=4, material=m,
        )
        args = rec.calls[0][1]
        i = args.index("-dynSeepage")
        assert args[i + 1] == token

    def test_emit_tri6_appends_porder_linear(self) -> None:
        m = _make_material()
        elem = _up2d(m)
        rec = _emit_with(
            elem, tag=5, nodes=(1, 2, 3, 4, 5, 6), mat_tag=8, material=m,
        )
        args = rec.calls[0][1]
        assert args[-2:] == ("-pOrder", "linear")

    def test_emit_tet10_appends_porder_linear(self) -> None:
        m = _make_material()
        elem = _up3d(m)
        rec = _emit_with(
            elem, tag=5, nodes=tuple(range(1, 11)), mat_tag=8, material=m,
        )
        args = rec.calls[0][1]
        assert args[-2:] == ("-pOrder", "linear")

    def test_emit_h8_no_porder(self) -> None:
        m = _make_material()
        elem = _up3d(m)
        rec = _emit_with(
            elem, tag=5, nodes=tuple(range(1, 9)), mat_tag=8, material=m,
        )
        assert "-pOrder" not in rec.calls[0][1]

    def test_emit_th_with_stab_raises(self) -> None:
        """Defense-in-depth mirror of the build gate: -stab is parser-fatal
        on the Taylor–Hood shapes."""
        m = _make_material()
        elem = _up2d(m, stab="auto")
        e = RecordingEmitter()
        set_tag_resolver(e, _resolver_for(m, 1))
        set_element_nodes(e, (1, 2, 3, 4, 5, 6))
        with pytest.raises(ValueError, match="parser-fatal"):
            elem._emit(e, tag=1)

    def test_emit_perm_dim_mismatch_raises(self) -> None:
        m = _make_material()
        elem = _up2d(m)  # 2-component perm
        e = RecordingEmitter()
        set_tag_resolver(e, _resolver_for(m, 1))
        set_element_nodes(e, tuple(range(1, 9)))  # H8 = 3D cell
        with pytest.raises(ValueError, match="perm/permH"):
            elem._emit(e, tag=1)

    @pytest.mark.parametrize("bad_count", [0, 2, 5, 7, 9, 11])
    def test_emit_wrong_node_count_raises(self, bad_count: int) -> None:
        m = _make_material()
        elem = _up2d(m)
        e = RecordingEmitter()
        set_tag_resolver(e, _resolver_for(m, 1))
        set_element_nodes(e, tuple(range(1, bad_count + 1)))
        with pytest.raises(ValueError, match="no shape provider"):
            elem._emit(e, tag=1)


# ===========================================================================
# Capability registry (ADR 0074 D2 surface)
# ===========================================================================

class TestLadrunoUPCapabilities:
    def test_registry_entry(self) -> None:
        spec = _ELEM_REGISTRY["LadrunoUP"]
        assert spec.ndm_ok == frozenset({2, 3})
        assert spec.gmsh_etypes == frozenset({2, 3, 9, 5, 11})
        assert spec.mat_family == "nd"

    def test_scalar_floors(self) -> None:
        assert element_required_floor("LadrunoUP", 2) == 3
        assert element_required_floor("LadrunoUP", 3) == 4

    def test_slot_floors_th_shapes(self) -> None:
        assert element_ndf_slot_floors("LadrunoUP", 6) == (3, 3, 3, 2, 2, 2)
        assert element_ndf_slot_floors("LadrunoUP", 10) == (
            4, 4, 4, 4, 3, 3, 3, 3, 3, 3,
        )

    def test_slot_floors_equal_order_shapes_are_scalar(self) -> None:
        for count in (3, 4, 8):
            assert element_ndf_slot_floors("LadrunoUP", count) is None

    def test_slot_floors_absent_for_scalar_classes(self) -> None:
        assert element_ndf_slot_floors("stdBrick", 8) is None
        assert element_ndf_slot_floors("BezierTet10", 10) is None

    def test_strictness(self) -> None:
        assert element_ndf_strict("LadrunoUP") is True
        assert element_ndf_strict("stdBrick") is False
        assert element_ndf_strict("NoSuchElement") is False

    def test_class_ndf_ok_union(self) -> None:
        assert element_class_ndf_ok("LadrunoUP") == frozenset({2, 3, 4})

    def test_builder_ndf_is_ndm_dependent(self) -> None:
        assert element_builder_ndf("LadrunoUP", 2) == 3
        assert element_builder_ndf("LadrunoUP", 3) == 4
        with pytest.raises(ValueError, match="pass ndm"):
            element_builder_ndf("LadrunoUP")
        # scalar-gated classes are unaffected by the extension
        assert element_builder_ndf("quad") == 2
        assert element_builder_ndf("quad", 2) == 2
        assert element_builder_ndf("stdBrick", 3) is None


# ===========================================================================
# Bridge-namespace integration
# ===========================================================================

class TestLadrunoUPNamespace:
    def test_ops_element_ladruno_up_registers(self) -> None:
        ops = apeSees(cast("object", MagicMock(name="FEMData")))  # type: ignore[arg-type]
        m = ops.nDMaterial.ElasticIsotropic(E=30e6, nu=0.3, rho=2.0)
        e = ops.element.LadrunoUP(
            pg="Soil", material=m, Kf=2.2e6, poro=0.4, rhoF=1.0,
            perm=(1e-8, 1e-8),
        )
        assert isinstance(e, LadrunoUP)
        assert ops.tag_for(e) == 1
