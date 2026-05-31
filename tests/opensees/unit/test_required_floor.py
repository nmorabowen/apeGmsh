"""ADR 0048 P1 — `_ElemSpec.required_floor` + `element_required_floor`.

The registry-enrichment unit table: every registered element must yield a
defined per-node ndf floor at each ndm it supports, and that floor must lie
in the element's `ndf_ok` tolerance set (the invariant the inference's
`∩ ndf_ok` gate relies on).
"""
import pytest

from apeGmsh.opensees._element_capabilities import (
    _ELEM_REGISTRY,
    _EXTRA_CLASS_NDF_OK,
    _EXTRA_CLASS_REQUIRED_FLOOR,
    _ElemSpec,
    element_class_ndf_ok,
    element_required_floor,
)


def test_required_floor_in_ndf_ok_for_every_registered_element():
    """The floor at any supported ndm is always a tolerated node-ndf."""
    for token, spec in _ELEM_REGISTRY.items():
        for ndm in spec.ndm_ok:
            floor = spec.required_floor(ndm)
            assert floor in spec.ndf_ok, (
                f"{token}: floor {floor} at ndm={ndm} not in "
                f"ndf_ok={sorted(spec.ndf_ok)}"
            )


@pytest.mark.parametrize(
    "token, ndm, expected",
    [
        # solids — single-valued ndf_ok, floor derived
        ("stdBrick", 3, 3),
        ("FourNodeTetrahedron", 3, 3),
        ("quad", 2, 2),
        ("tri31", 2, 2),
        # shells — fixed 6
        ("ShellMITC4", 3, 6),
        ("ShellMITC3", 3, 6),
        # flexible multi-ndm — explicit ndf_required map
        ("truss", 2, 2),
        ("truss", 3, 3),
        ("elasticBeamColumn", 2, 3),
        ("elasticBeamColumn", 3, 6),  # NOT the set-min 3 — the whole point
        ("ElasticTimoshenkoBeam", 3, 6),
    ],
)
def test_required_floor_values(token, ndm, expected):
    assert _ELEM_REGISTRY[token].required_floor(ndm) == expected


@pytest.mark.parametrize(
    "class_name, ndm, expected",
    [
        ("FourNodeQuad", 2, 2),   # alias -> "quad"
        ("Tri31", 2, 2),          # alias -> "tri31"
        ("Truss", 3, 3),          # alias -> "truss"
        ("CorotTruss", 2, 2),     # alias -> "corotTruss"
        ("stdBrick", 3, 3),       # name == token
        ("ShellMITC4", 3, 6),
        ("ASDShellT3", 3, 6),     # _EXTRA_CLASS_NDF_OK single-valued fallback
        ("NotARealElement", 3, None),  # unclassifiable -> None (fail-loud upstream)
    ],
)
def test_element_required_floor_resolves_aliases_and_extras(class_name, ndm, expected):
    assert element_required_floor(class_name, ndm) == expected


def test_multi_valued_ndf_ok_without_map_raises():
    """A multi-ndm element that forgot its ndf_required map must fail loud,
    never silently pick a set-min (which would give a 3D beam ndf=3)."""
    bad = _ElemSpec(
        mat_family="none", needs_transf=True,
        ndm_ok=frozenset({2, 3}), ndf_ok=frozenset({3, 6}),
        gmsh_etypes=frozenset({1}), node_reorder={1: (0, 1)},
        # no ndf_required
    )
    with pytest.raises(ValueError, match="ndf_required"):
        bad.required_floor(3)


def test_required_floor_unknown_ndm_in_map_raises():
    spec = _ELEM_REGISTRY["elasticBeamColumn"]
    with pytest.raises(ValueError, match="ndm=1"):
        spec.required_floor(1)


# ---- ADR 0048 P-C: emittable elements outside _ELEM_REGISTRY --------------

@pytest.mark.parametrize(
    "class_name, ndm, ndf_ok, floor",
    [
        # nonlinear beams: same DOFs as the elastic forms
        ("forceBeamColumn", 2, {3, 6}, 3),
        ("forceBeamColumn", 3, {3, 6}, 6),
        ("dispBeamColumn", 3, {3, 6}, 6),
        # truss-like
        ("InertiaTruss", 2, {2, 3, 6}, 2),
        ("InertiaTruss", 3, {2, 3, 6}, 3),
        # adaptive springs: tolerate any ndf, impose only the floor=1 minimum
        ("ZeroLength", 3, {1, 2, 3, 4, 5, 6}, 1),
        ("ZeroLength", 2, {1, 2, 3, 4, 5, 6}, 1),
        ("ZeroLengthSection", 3, {1, 2, 3, 4, 5, 6}, 1),
    ],
)
def test_extra_emittable_elements_now_covered(class_name, ndm, ndf_ok, floor):
    """These ship a _emit but have no _ELEM_REGISTRY entry; inference still
    needs them classifiable. PR-C registers them via the _EXTRA maps."""
    assert element_class_ndf_ok(class_name) == frozenset(ndf_ok)
    assert element_required_floor(class_name, ndm) == floor


def test_extra_floor_maps_are_in_ndf_ok():
    """Same invariant as the registry: every extra floor is a tolerated ndf
    (so the ∩ ndf_ok gate never rejects an element's own minimum)."""
    for cls, floor_map in _EXTRA_CLASS_REQUIRED_FLOOR.items():
        ndf_ok = _EXTRA_CLASS_NDF_OK[cls]
        for ndm, floor in floor_map.items():
            assert floor in ndf_ok, f"{cls}: floor {floor} (ndm={ndm}) not in {sorted(ndf_ok)}"


def test_zerolength_adaptive_does_not_inflate_a_real_node():
    """A spring sharing a node with a 3-DOF brick must resolve to 3, not be
    dragged down to the spring's floor=1 — adaptive contributes the minimum."""
    from apeGmsh.opensees._internal.build import _infer_ndf_from_incidence
    out = _infer_ndf_from_incidence({1: ["stdBrick", "ZeroLength"]}, 3)
    assert out == {1: 3}
