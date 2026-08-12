"""ADR 0093 S3 — record skeleton for ``g.constraints.interface()``.

Covers :class:`NormalLaw` / :class:`TangentialLaw` (D1 declarative
per-area laws, validated per kind at construction) and
:class:`InterfaceRecord` (fields + ``tag_rewrite_spec``). No verb, no
resolver, no emit — those are S4/S5; this only locks the record
skeleton the later slices build on.
"""
from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from apeGmsh._kernel.records._constraints import (
    InterfaceRecord,
    NodePairRecord,
    NormalLaw,
    TangentialLaw,
)
from apeGmsh._kernel.records._kinds import ConstraintKind


# ---------------------------------------------------------------------------
# NormalLaw
# ---------------------------------------------------------------------------


def test_normal_law_ent_valid() -> None:
    law = NormalLaw(kind="ent", k_per_area=1.0e6)
    assert law.kind == "ent"
    assert law.tau_b_n is None
    assert law.gap is None


def test_normal_law_elastic_valid() -> None:
    law = NormalLaw(kind="elastic", k_per_area=1.0e6)
    assert law.tau_b_n is None
    assert law.gap is None


def test_normal_law_epp_gap_valid() -> None:
    law = NormalLaw(kind="epp_gap", k_per_area=1.0e6, tau_b_n=2.0e5, gap=-1.0e-3)
    assert law.tau_b_n == 2.0e5
    assert law.gap == -1.0e-3


def test_normal_law_unknown_kind_fails_loud() -> None:
    with pytest.raises(ValueError, match="kind"):
        NormalLaw(kind="bogus", k_per_area=1.0e6)


@pytest.mark.parametrize("k_per_area", [0.0, -1.0])
def test_normal_law_nonpositive_stiffness_fails_loud(k_per_area: float) -> None:
    with pytest.raises(ValueError, match="k_per_area"):
        NormalLaw(kind="ent", k_per_area=k_per_area)


def test_normal_law_epp_gap_requires_tau_b_n() -> None:
    with pytest.raises(ValueError, match="tau_b_n"):
        NormalLaw(kind="epp_gap", k_per_area=1.0e6, gap=-1.0e-3)


def test_normal_law_epp_gap_requires_gap() -> None:
    with pytest.raises(ValueError, match="gap"):
        NormalLaw(kind="epp_gap", k_per_area=1.0e6, tau_b_n=2.0e5)


def test_normal_law_epp_gap_rejects_nonpositive_tau_b_n() -> None:
    with pytest.raises(ValueError, match="tau_b_n"):
        NormalLaw(kind="epp_gap", k_per_area=1.0e6, tau_b_n=0.0, gap=-1.0e-3)


def test_normal_law_epp_gap_rejects_positive_gap() -> None:
    with pytest.raises(ValueError, match="gap"):
        NormalLaw(kind="epp_gap", k_per_area=1.0e6, tau_b_n=2.0e5, gap=1.0e-3)


@pytest.mark.parametrize("kind", ["ent", "elastic"])
def test_normal_law_rejects_epp_gap_params_on_other_kinds(kind: str) -> None:
    with pytest.raises(ValueError):
        NormalLaw(kind=kind, k_per_area=1.0e6, tau_b_n=2.0e5)
    with pytest.raises(ValueError):
        NormalLaw(kind=kind, k_per_area=1.0e6, gap=-1.0e-3)


def test_normal_law_is_frozen() -> None:
    law = NormalLaw(kind="ent", k_per_area=1.0e6)
    with pytest.raises(dataclasses.FrozenInstanceError):
        law.k_per_area = 2.0e6  # type: ignore[misc]


# ---------------------------------------------------------------------------
# TangentialLaw
# ---------------------------------------------------------------------------


def test_tangential_law_elastic_valid() -> None:
    law = TangentialLaw(kind="elastic", k_per_area=1.0e6)
    assert law.tau_b is None


def test_tangential_law_epp_valid() -> None:
    law = TangentialLaw(kind="epp", k_per_area=1.0e6, tau_b=5.0e4)
    assert law.tau_b == 5.0e4


def test_tangential_law_unknown_kind_fails_loud() -> None:
    with pytest.raises(ValueError, match="kind"):
        TangentialLaw(kind="bogus", k_per_area=1.0e6)


@pytest.mark.parametrize("k_per_area", [0.0, -1.0])
def test_tangential_law_nonpositive_stiffness_fails_loud(
    k_per_area: float,
) -> None:
    with pytest.raises(ValueError, match="k_per_area"):
        TangentialLaw(kind="elastic", k_per_area=k_per_area)


def test_tangential_law_epp_requires_tau_b() -> None:
    with pytest.raises(ValueError, match="tau_b"):
        TangentialLaw(kind="epp", k_per_area=1.0e6)


def test_tangential_law_epp_rejects_nonpositive_tau_b() -> None:
    with pytest.raises(ValueError, match="tau_b"):
        TangentialLaw(kind="epp", k_per_area=1.0e6, tau_b=0.0)


def test_tangential_law_elastic_rejects_tau_b() -> None:
    with pytest.raises(ValueError, match="tau_b"):
        TangentialLaw(kind="elastic", k_per_area=1.0e6, tau_b=5.0e4)


def test_tangential_law_is_frozen() -> None:
    law = TangentialLaw(kind="elastic", k_per_area=1.0e6)
    with pytest.raises(dataclasses.FrozenInstanceError):
        law.tau_b = 1.0  # type: ignore[misc]


# ---------------------------------------------------------------------------
# InterfaceRecord
# ---------------------------------------------------------------------------


def test_interface_record_construction_equal_ndf() -> None:
    """Equal-ndf pair: no phantom, no nested equalDOF (D4)."""
    rec = InterfaceRecord(
        kind=ConstraintKind.INTERFACE,
        name="RockLinerInterface",
        master_node=1,
        slave_node=2,
        backing_element=100,
        orient=(0.0, 0.0, 1.0, 1.0, 0.0, 0.0),
        a_trib=0.5,
        normal_law=NormalLaw(kind="ent", k_per_area=1.0e6),
        tangential_law=TangentialLaw(kind="epp", k_per_area=1.0e6, tau_b=5.0e4),
    )
    assert rec.kind == "interface"
    assert rec.master_node == 1
    assert rec.slave_node == 2
    assert rec.backing_element == 100
    assert rec.phantom_node is None
    assert rec.phantom_coords is None
    assert rec.phantom_ndf is None
    assert rec.equal_dof_records == []


def test_interface_record_construction_mixed_ndf_phantom() -> None:
    """Mixed-ndf pair: phantom + nested equalDOF (D4)."""
    equal_dof = NodePairRecord(
        kind=ConstraintKind.EQUAL_DOF,
        master_node=2,
        slave_node=900,
        dofs=[1, 2],
    )
    rec = InterfaceRecord(
        kind=ConstraintKind.INTERFACE,
        master_node=1,
        slave_node=2,
        backing_element=100,
        orient=(0.0, 0.0, 1.0, 1.0, 0.0, 0.0),
        a_trib=0.5,
        normal_law=NormalLaw(kind="ent", k_per_area=1.0e6),
        tangential_law=TangentialLaw(kind="epp", k_per_area=1.0e6, tau_b=5.0e4),
        phantom_node=900,
        phantom_coords=np.array([1.0, 2.0, 3.0]),
        phantom_ndf=2,
        equal_dof_records=[equal_dof],
    )
    assert rec.phantom_node == 900
    assert rec.phantom_ndf == 2
    assert len(rec.equal_dof_records) == 1
    assert rec.equal_dof_records[0].slave_node == 900


def test_interface_record_tag_rewrite_spec_shape() -> None:
    """``tag_rewrite_spec`` is a dict with the standard cover-set keys."""
    spec = InterfaceRecord.tag_rewrite_spec
    assert isinstance(spec, dict)
    for key in ("tag_fields_scalar", "tag_fields_array", "name_fields"):
        assert key in spec


def test_interface_record_tag_rewrite_covers_node_fields() -> None:
    """master_node / slave_node / phantom_node are node-tag references
    and must offset with the module (ADR 0038)."""
    scalar = InterfaceRecord.tag_rewrite_spec["tag_fields_scalar"]
    assert "master_node" in scalar
    assert "slave_node" in scalar
    assert "phantom_node" in scalar


def test_interface_record_tag_rewrite_covers_backing_element() -> None:
    """``backing_element`` rides the rewrite too (ADR 0093 S6).

    It is an *element* tag while the other three are node tags, and
    ``_rewrite_record`` (mesh/_compose.py) applies ONE ``offset`` to
    every field named in the spec — which is exact here, because a
    compose module reserves a single tag window spanning nodes AND
    elements (``_scan_min_max_tags``). ``tests/test_compose_rewrite.py``
    pins that the two really move together on a real bundle; this test
    just locks the field into the cover-set so it cannot silently drop
    back out (an un-rewritten backing element points at the HOST's
    element of the same tag — a wrong partition owner in S8, and
    nothing else would notice).
    """
    scalar = InterfaceRecord.tag_rewrite_spec["tag_fields_scalar"]
    assert "backing_element" in scalar


def test_interface_record_tag_rewrite_nested_equal_dof() -> None:
    spec = InterfaceRecord.tag_rewrite_spec
    assert spec.get("nested_records") == ("equal_dof_records",)


def test_interface_record_name_field_in_spec() -> None:
    assert InterfaceRecord.tag_rewrite_spec["name_fields"] == ("name",)


# ---------------------------------------------------------------------------
# orient validation (probe NIT #4) — this slice's #1 downstream risk is
# orientation (INV-1 sign convention); a wrong-length tuple constructing
# fine is a trap that would only surface as a wrong sign deep in emit.
# ---------------------------------------------------------------------------


def test_interface_record_accepts_none_orient() -> None:
    InterfaceRecord(kind=ConstraintKind.INTERFACE, orient=None)


def test_interface_record_accepts_six_tuple_orient() -> None:
    rec = InterfaceRecord(
        kind=ConstraintKind.INTERFACE,
        orient=(0.0, 0.0, 1.0, 1.0, 0.0, 0.0),
    )
    assert len(rec.orient) == 6


@pytest.mark.parametrize("bad_orient", [
    (0.0, 0.0, 1.0),                     # a bare 3-vector normal
    (0.0, 0.0, 1.0, 1.0, 0.0),           # missing one local-y component
    (0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0), # one too many
    (),
])
def test_interface_record_rejects_wrong_length_orient(bad_orient) -> None:
    with pytest.raises(ValueError, match="orient"):
        InterfaceRecord(kind=ConstraintKind.INTERFACE, orient=bad_orient)


# ---------------------------------------------------------------------------
# _split_constraints misroute guard (probe WEAKENS #3): InterfaceRecord
# is a side-list family (ADR 0093 "Alternatives rejected") and must never
# be routed onto fem.nodes.constraints / fem.elements.constraints — the
# _DISPATCH MP-constraint lane has no dtype/decoder for it, so a misrouted
# record writes a /constraints/interface group that round-trips back as 0
# records with no warning.
# ---------------------------------------------------------------------------


def test_split_constraints_refuses_interface_record() -> None:
    from apeGmsh.mesh._fem_factory import _split_constraints

    rec = InterfaceRecord(
        kind=ConstraintKind.INTERFACE,
        master_node=1,
        slave_node=2,
        backing_element=10,
        normal_law=NormalLaw(kind="ent", k_per_area=1.0e6),
    )
    with pytest.raises(TypeError, match="InterfaceRecord"):
        _split_constraints([rec])
