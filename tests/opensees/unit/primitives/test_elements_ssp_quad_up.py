"""Unit tests for the stock ``SSPquadUP`` typed element (UWelements).

Covers:

* Construction validation (positive thickness / fluid bulk / void ratio,
  non-negative fluid density / perm / alpha, two-component perm and body).
* ``_emit`` positional order against ``OPS_SSPquadUP``
  (``SSPquadUP.cpp:65``): ``tag n1..n4 matTag t fBulk fDen k1 k2 e alpha
  <b1 b2>``, the body pair emitted whole or not at all (the parser
  counts 13 / 15 / 19 arguments and drops any other tail).
* Capability-registry surface: ndm 2, per-node ndf 3, no builder gate.
* The u-p rotation-role membership (DOF 3 is p, never ``rotation_z``).
* Bridge-namespace integration (``ops.element.SSPquadUP``).
"""
from __future__ import annotations

import pytest

from apeGmsh.opensees import apeSees
from apeGmsh.opensees._element_capabilities import (
    _ELEM_REGISTRY,
    element_builder_ndf,
    element_class_ndf_ok,
    element_required_floor,
)
from apeGmsh.opensees._internal.tag_resolution import (
    set_element_nodes,
    set_tag_resolver,
)
from apeGmsh.opensees._internal.types import Primitive
from apeGmsh.opensees._readback import _UP_FAMILIES, rotation_roles_from_tokens
from apeGmsh.opensees.element import SSPquadUP
from apeGmsh.opensees.emitter.recording import RecordingEmitter
from apeGmsh.opensees.material.nd import ElasticIsotropic


def _mat() -> ElasticIsotropic:
    return ElasticIsotropic(E=1e5, nu=0.3, rho=2.0)


def _el(m: ElasticIsotropic, **kw: object) -> SSPquadUP:
    base: dict[str, object] = dict(
        pg="Soil", material=m, thickness=1.0, fluid_bulk=2.2e6,
        fluid_density=1.0, perm=(1e-4, 2e-4), void_ratio=0.7, alpha=1.5e-6,
    )
    base.update(kw)
    return SSPquadUP(**base)  # type: ignore[arg-type]


def _emit(elem: SSPquadUP, m: object, *, mat_tag: int = 9) -> RecordingEmitter:
    e = RecordingEmitter()

    def _resolve(prim: Primitive) -> int:
        assert prim is m
        return mat_tag

    set_tag_resolver(e, _resolve)
    set_element_nodes(e, (1, 2, 3, 4))
    elem._emit(e, tag=7)
    return e


class TestConstruction:
    def test_minimal_ok(self) -> None:
        m = _mat()
        el = _el(m)
        assert el.dependencies() == (m,)
        assert "SSPquadUP" in repr(el)

    @pytest.mark.parametrize("kw", [
        {"thickness": 0.0},
        {"fluid_bulk": 0.0},
        {"fluid_density": -1.0},
        {"perm": (1e-4,)},
        {"perm": (1e-4, -1.0)},
        {"void_ratio": 0.0},
        {"alpha": -1e-6},
        {"body": (0.0, -9.81, 0.0)},
    ])
    def test_refusals(self, kw: dict) -> None:
        with pytest.raises(ValueError, match="SSPquadUP"):
            _el(_mat(), **kw)


class TestEmit:
    def test_positional_order_without_body(self) -> None:
        m = _mat()
        rec = _emit(_el(m), m)
        assert rec.calls == [(
            "element",
            ("SSPquadUP", 7, 1, 2, 3, 4, 9,
             1.0, 2.2e6, 1.0, 1e-4, 2e-4, 0.7, 1.5e-6),
            {},
        )]

    def test_body_pair_appended(self) -> None:
        m = _mat()
        rec = _emit(_el(m, body=(0.0, -9.81)), m)
        args = rec.calls[0][1]
        # 13 parser arguments + the body pair = 15 (+ the type token).
        assert len(args) == 1 + 15
        assert args[-2:] == (0.0, -9.81)

    def test_wrong_node_count_refused(self) -> None:
        m = _mat()
        e = RecordingEmitter()
        set_tag_resolver(e, lambda p: 9)
        set_element_nodes(e, (1, 2, 3))
        with pytest.raises(ValueError, match="4 node tags"):
            _el(m)._emit(e, tag=1)


class TestCapabilities:
    def test_registry_entry(self) -> None:
        spec = _ELEM_REGISTRY["SSPquadUP"]
        assert spec.ndm_ok == frozenset({2})
        assert spec.gmsh_etypes == frozenset({3})
        assert spec.has_gauss

    def test_ndf_three_per_node(self) -> None:
        assert element_class_ndf_ok("SSPquadUP") == frozenset({3})
        assert element_required_floor("SSPquadUP", 2) == 3

    def test_no_builder_gate(self) -> None:
        # OPS_SSPquadUP reads no OPS_GetNDF: no model-builder bracket.
        assert element_builder_ndf("SSPquadUP", 2) is None

    def test_dof3_is_pressure_not_rotation(self) -> None:
        assert SSPquadUP in _UP_FAMILIES
        roles = rotation_roles_from_tokens([("SSPquadUP", (1, 2, 3, 4))])
        assert roles.refusal(1) is not None
        assert roles.model_blocker == "SSPquadUP"


class TestNamespace:
    def test_ops_element_registers(self) -> None:
        from unittest.mock import MagicMock

        ops = apeSees(MagicMock())
        mat = ops.nDMaterial.ElasticIsotropic(E=1e5, nu=0.3, rho=2.0)
        el = ops.element.SSPquadUP(
            pg="Soil", material=mat, thickness=1.0, fluid_bulk=2.2e6,
            fluid_density=1.0, perm=(1e-4, 1e-4), void_ratio=0.7,
            alpha=1.5e-6, body=(0.0, -9.81),
        )
        assert isinstance(el, SSPquadUP)
        assert el.material is mat
        assert el.body == (0.0, -9.81)
