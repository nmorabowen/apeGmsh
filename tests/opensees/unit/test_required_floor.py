"""ADR 0048 — ``required_floor`` registry seam (PR-1 rebuild).

Pure-function unit tests for the per-element DOF *floor* that feeds element-
class ndf inference. No gmsh / openseespy needed.
"""
import pytest

from apeGmsh.opensees._element_capabilities import (
    _ELEM_REGISTRY,
    element_class_ndm_ok,
    element_required_floor,
)


@pytest.mark.parametrize(
    "cls, ndm, expected",
    [
        # ── single-ndm registry elements: floor == the sole ndf_ok member ──
        ("stdBrick", 3, 3),
        ("FourNodeTetrahedron", 3, 3),
        ("quad", 2, 2),
        ("tri31", 2, 2),
        ("ShellMITC4", 3, 6),
        ("ShellMITC3", 3, 6),
        # ── multi-ndm registry elements: floor from the ndf_required map ──
        ("elasticBeamColumn", 2, 3),   # ux, uy, rz
        ("elasticBeamColumn", 3, 6),   # 3 disp + 3 rot
        ("ElasticTimoshenkoBeam", 3, 6),
        ("truss", 2, 2),               # truss adapts; floor = ndm
        ("truss", 3, 3),
        ("corotTruss", 3, 3),
        # ── Python class-name aliases resolve to their token ──
        ("FourNodeQuad", 2, 2),
        ("Truss", 3, 3),
        ("CorotTruss", 2, 2),
        # ── non-registry extras (_EXTRA_CLASS_REQUIRED_FLOOR) ──
        ("forceBeamColumn", 2, 3),
        ("forceBeamColumn", 3, 6),
        ("dispBeamColumn", 3, 6),
        ("InertiaTruss", 3, 3),
        # ── single-valued extra: floor from the sole ndf_ok member ──
        ("ASDShellT3", 3, 6),
        # ── adaptive plain zeroLength: floor 1, never inflates the per-node max ──
        ("ZeroLength", 2, 1),
        ("ZeroLength", 3, 1),
        # ── zeroLengthSection is NOT adaptive: demands the full section-DOF
        #    floor (3 in 2D, 6 in 3D) or OpenSees silently drops it ──
        ("ZeroLengthSection", 2, 3),
        ("ZeroLengthSection", 3, 6),
    ],
)
def test_element_required_floor(cls, ndm, expected):
    assert element_required_floor(cls, ndm) == expected


def test_unclassifiable_returns_none():
    assert element_required_floor("NoSuchElement", 3) is None


def test_floor_is_always_a_member_of_ndf_ok():
    """The inferred floor MUST be a value the element actually accepts, for
    every registered class at every ndm it supports — otherwise inference would
    propose an ndf the element's own ``ndf_ok`` rejects."""
    for token, spec in _ELEM_REGISTRY.items():
        for ndm in spec.ndm_ok:
            floor = spec.required_floor(ndm)
            assert floor in spec.ndf_ok, (token, ndm, floor, sorted(spec.ndf_ok))


def test_required_floor_raises_on_unknown_ndm_for_mapped_element():
    spec = _ELEM_REGISTRY["truss"]  # ndf_required={2: 2, 3: 3}
    with pytest.raises(ValueError, match="no ndf_required entry"):
        spec.required_floor(5)


def test_element_class_ndm_ok():
    assert element_class_ndm_ok("stdBrick") == frozenset({3})
    assert element_class_ndm_ok("quad") == frozenset({2})
    assert element_class_ndm_ok("truss") == frozenset({2, 3})
    assert element_class_ndm_ok("FourNodeQuad") == frozenset({2})  # alias
    assert element_class_ndm_ok("NoSuchElement") is None
