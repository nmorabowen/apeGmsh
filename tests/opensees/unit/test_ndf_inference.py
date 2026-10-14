"""ADR 0048 — element-class ndf inference engine (authoritative).

Covers the pure inference core, the adaptive-element carve-out, the ``ndm``
guard, and the PG-walk over a lightweight FEM stub. No gmsh / openseespy
needed. (The pre-clean-break shadow-mode parity gate was removed once
inference became the only path.)
"""
import pytest

from apeGmsh.opensees._internal.build.errors import BridgeError
from apeGmsh.opensees._internal.build.ndf import (
    _infer_ndf_from_incidence,
    assert_ndm_compatible,
    infer_node_ndf,
    validate_adaptive_element_endpoints,
)


# ── element-spec stubs: type(spec).__name__ must be the OpenSees class ───────
def _spec(class_name: str, pg: str):
    """A throwaway Element-spec stand-in whose type name is *class_name*."""
    return type(class_name, (), {"pg": pg})()


# ── lightweight FEM stub for the PG-walk ─────────────────────────────────────
class _StubSel:
    def __init__(self, groups):
        self._groups = groups

    def groups(self):
        return self._groups


class _StubElements:
    def __init__(self, pg_groups):
        self._pg_groups = pg_groups

    def select(self, pg):
        if pg not in self._pg_groups:
            raise KeyError(pg)
        return _StubSel(self._pg_groups[pg])


class _StubFem:
    """Exposes just the surface ``infer_node_ndf`` touches:
    ``elements.select(pg).groups()``."""

    def __init__(self, pg_groups):
        self.elements = _StubElements(pg_groups)


# ─────────────────────────── pure core ──────────────────────────────────────

@pytest.mark.parametrize(
    "classes, ndm, expected",
    [
        (["stdBrick"], 3, 3),
        (["ShellMITC4"], 3, 6),
        (["elasticBeamColumn"], 3, 6),
        (["elasticBeamColumn"], 2, 3),
        (["truss"], 3, 3),
        (["truss"], 2, 2),
        (["quad"], 2, 2),
        # adaptive spring never inflates the structural side's count
        (["stdBrick", "ZeroLength"], 3, 3),
        (["ShellMITC4", "ZeroLength"], 3, 6),
        # truss legitimately shares a 3D beam node at 6 (truss adapts)
        (["elasticBeamColumn", "truss"], 3, 6),
    ],
)
def test_infer_core(classes, ndm, expected):
    assert _infer_ndf_from_incidence({1: classes}, ndm)[1] == expected


@pytest.mark.parametrize(
    "classes",
    [
        ["ZeroLength"],
        ["TwoNodeLink"],
        ["CoupledZeroLength"],
        ["ZeroLength", "TwoNodeLink"],
    ],
)
def test_infer_adaptive_only_node_is_omitted(classes):
    """A node touched ONLY by adaptive elements (the zeroLength family,
    ndf_ok={1..6}) carries no inference opinion — it is omitted from the
    map so the ``ops.model`` envelope supplies its ndf at emit (ADR 0048)."""
    out = _infer_ndf_from_incidence({5: classes}, 3)
    assert 5 not in out


@pytest.mark.parametrize(
    "classes, ndm",
    [
        (["quad", "elasticBeamColumn"], 2),   # {2} ∩ {3,6} = ∅ at floor 3
        (["ShellMITC4", "stdBrick"], 3),      # {6} ∩ {3} = ∅
        (["elasticBeamColumn", "stdBrick"], 3),  # floor 6 ∉ brick {3} (strict)
    ],
)
def test_infer_incompatible_shared_node_fails_loud(classes, ndm):
    with pytest.raises(BridgeError):
        _infer_ndf_from_incidence({7: classes}, ndm)


def test_infer_unclassifiable_fails_loud():
    with pytest.raises(BridgeError, match="not in the capability registry"):
        _infer_ndf_from_incidence({1: ["TotallyUnknownElement"]}, 3)


# ─────────────────────────── ndm guard ──────────────────────────────────────

def test_assert_ndm_compatible_ok():
    assert_ndm_compatible(["stdBrick", "truss"], 3)  # truss adapts to 3


def test_assert_ndm_mix_2d_3d_fails():
    with pytest.raises(BridgeError, match="mix 2D and 3D"):
        assert_ndm_compatible(["quad", "stdBrick"], 2)


def test_assert_ndm_excluded_value_fails():
    with pytest.raises(BridgeError, match="incompatible"):
        assert_ndm_compatible(["stdBrick"], 2)  # brick is ndm=3 only


def test_assert_ndm_skips_unclassifiable():
    assert_ndm_compatible(["NoSuchElement", "stdBrick"], 3)  # no raise


# ─────────────────────── the PG-walk (infer_node_ndf) ────────────────────────

def test_infer_node_ndf_walk():
    # solid PG: one brick on nodes 10-13; beam PG: one 3D beam on disjoint
    # nodes 14,15 (no shared node → both families coexist cleanly).
    fem = _StubFem({
        "Solid": [[(1, (10, 11, 12, 13))]],
        "Frame": [[(2, (14, 15))]],
    })
    elements = [_spec("stdBrick", "Solid"), _spec("elasticBeamColumn", "Frame")]
    out = infer_node_ndf(fem, elements, ndm=3)
    assert out[10] == 3 and out[11] == 3 and out[12] == 3 and out[13] == 3
    assert out[14] == 6 and out[15] == 6


def test_infer_node_ndf_walk_shared_incompatible_fails():
    fem = _StubFem({
        "Solid": [[(1, (10, 11, 12, 13))]],
        "Frame": [[(2, (13, 14))]],  # node 13 shared with the brick
    })
    elements = [_spec("stdBrick", "Solid"), _spec("elasticBeamColumn", "Frame")]
    with pytest.raises(BridgeError):
        infer_node_ndf(fem, elements, ndm=3)


# ──────────────── adaptive-element endpoint guard (ADR 0048 review) ───────────

def test_adaptive_endpoint_mismatch_fails_loud():
    """A zeroLength whose structural end infers a value != envelope while
    its ground end (element-less) falls to the envelope would emit
    mismatched ndf and OpenSees silently drops the spring — fail loud."""
    fem = _StubFem({"Spring": [[(9, (20, 21))]]})
    elements = [_spec("ZeroLength", "Spring")]
    # node 20 is structural (inferred 2 here); node 21 is element-less
    # (absent → envelope). Envelope 6 → 20 emits 2, 21 emits 6 → mismatch,
    # and an end below ndf 3 is outside the fork's 3D relaxation.
    with pytest.raises(BridgeError, match="differing effective ndf"):
        validate_adaptive_element_endpoints(
            fem, elements, ndm=3, inferred={20: 2}, envelope_ndf=6,
        )


def test_adaptive_endpoint_mismatch_in_2d_fails_loud():
    """The 2D lane is untouched by fork #808 / ADR 96 — the engine still
    only warns and leaves the element inert there, so every mismatch is
    refused before emission."""
    fem = _StubFem({"Spring": [[(9, (20, 21))]]})
    elements = [_spec("ZeroLength", "Spring")]
    with pytest.raises(BridgeError, match="differing effective ndf"):
        validate_adaptive_element_endpoints(
            fem, elements, ndm=2, inferred={20: 2}, envelope_ndf=3,
        )


@pytest.mark.parametrize("structural, envelope", [
    (3, 4),      # u-p soil ground against a 3-dof continuum end
    (3, 6),      # shell / beam ground
    (6, 4),
])
def test_adaptive_endpoint_mixed_ndf_ok_in_3d(structural, envelope):
    """Fork #808 / ADR 96: a 3D zeroLength-family element takes ANY pair
    whose ends both carry ndf >= 3 — it acts on DOFs 1-3 and every DOF
    past the third rides as an untouched passenger. Refusing these would
    block the 3D interfaces of TIMs A10."""
    fem = _StubFem({"Spring": [[(9, (20, 21))]]})
    elements = [_spec("ZeroLength", "Spring")]
    validate_adaptive_element_endpoints(
        fem, elements, ndm=3, inferred={20: structural},
        envelope_ndf=envelope,
    )


def test_adaptive_endpoint_match_ok():
    """Both ends resolve to the same effective ndf → no raise (the common
    uniform-envelope spring-to-ground case)."""
    fem = _StubFem({"Spring": [[(9, (20, 21))]]})
    elements = [_spec("ZeroLength", "Spring")]
    # node 20 structural 3, node 21 absent → envelope 3 → both 3 → OK.
    validate_adaptive_element_endpoints(
        fem, elements, ndm=3, inferred={20: 3}, envelope_ndf=3,
    )
    # both ends structural and equal → OK.
    validate_adaptive_element_endpoints(
        fem, elements, ndm=3, inferred={20: 6, 21: 6}, envelope_ndf=3,
    )


def test_adaptive_guard_ignores_non_adaptive_elements():
    """Non-adaptive elements (e.g. ZeroLengthSection {3,6}) are not subject
    to the endpoint-equality guard here (their ndf is inferred, and the
    shared-node validity gate handles their compatibility)."""
    fem = _StubFem({"Sec": [[(9, (20, 21))]]})
    elements = [_spec("ZeroLengthSection", "Sec")]
    validate_adaptive_element_endpoints(
        fem, elements, ndm=3, inferred={20: 6}, envelope_ndf=3,
    )


# ───────────── columnar fan-out path (#1922, ADR 0100 lever 1) ──────────────
# infer_node_ndf reads node ids from the PGElementFanout conn arrays, never
# from per-element tuples. A homogeneous PG yields an int64[N, k] conn; a PG
# whose groups differ in node count yields an object array of per-row arrays
# (the mixed-arity route). Both must give the reference core's map.

def _reference_map(pg_groups, specs, ndm):
    """The map _infer_ndf_from_incidence gives for the same incidence."""
    node_to_classes: dict = {}
    for spec in specs:
        for group in pg_groups[spec.pg]:
            for _eid, conn in group:
                for n in conn:
                    node_to_classes.setdefault(n, []).append(
                        type(spec).__name__)
    return _infer_ndf_from_incidence(node_to_classes, ndm)


def test_columnar_mixed_arity_class_matches_reference_core():
    pg_groups = {
        # one class, two node counts in one PG -> object-dtype conn
        "Frame": [[(1, (1, 2)), (2, (2, 3))], [(3, (3, 4, 5))],
                  [(4, (5, 6))]],
        "Solid": [[(5, (10, 11, 12, 13, 14, 15, 16, 17))]],
        "Truss": [[(6, (6, 7))]],          # 6 adapts to the beam; 7 is 3
        "Spring": [[(7, (17, 30))]],       # adaptive: 30 stays absent
    }
    specs = [
        _spec("elasticBeamColumn", "Frame"), _spec("stdBrick", "Solid"),
        _spec("truss", "Truss"), _spec("ZeroLength", "Spring"),
    ]
    out = infer_node_ndf(_StubFem(pg_groups), specs, ndm=3)
    assert out == _reference_map(pg_groups, specs, 3)
    assert out == {
        1: 6, 2: 6, 3: 6, 4: 6, 5: 6, 6: 6, 7: 3,
        10: 3, 11: 3, 12: 3, 13: 3, 14: 3, 15: 3, 16: 3, 17: 3,
    }


def _up2d(pg: str = "Soil"):
    from apeGmsh.opensees.element.solid import LadrunoUP
    from apeGmsh.opensees.material.nd import ElasticIsotropic

    return LadrunoUP(
        pg=pg, material=ElasticIsotropic(E=30e6, nu=0.3, rho=2.0),
        Kf=2.2e6, poro=0.4, rhoF=1.0, perm=(1e-8, 1e-8),
    )


def test_columnar_strict_per_slot_class_pins_map():
    """ADR 0074 through the 2-D conn path: tri6 vertex slots take
    ndm+1 = 3, mid-edge slots ndm = 2, mapped column by column."""
    fem = _StubFem({"Soil": [[(1, (1, 2, 4, 5, 7, 9)),
                              (2, (2, 3, 4, 6, 8, 7))]]})
    out = infer_node_ndf(fem, [_up2d()], ndm=2)
    assert out == {1: 3, 2: 3, 3: 3, 4: 3, 5: 2, 6: 2, 7: 2, 8: 2, 9: 2}


def test_columnar_strict_mixed_arity_class_pins_map():
    """One strict class with T3 (scalar floor 3) and tri6 (per-slot) rows
    in one PG: the object-dtype conn must still split by node count."""
    fem = _StubFem({"Soil": [
        [(1, (2, 3, 4))],                        # T3 first: scalar 3
        [(2, (1, 2, 4, 5, 7, 9))],               # tri6: per slot
        [(3, (3, 10, 11))],                      # T3 again
    ]})
    out = infer_node_ndf(fem, [_up2d()], ndm=2)
    assert out == {
        1: 3, 2: 3, 3: 3, 4: 3, 10: 3, 11: 3, 5: 2, 7: 2, 9: 2,
    }


def test_columnar_strict_midedge_shared_by_t3_vertex_fails():
    """A T3 vertex (exactly 3) on a tri6 mid-edge node (exactly 2) in the
    same mixed-arity class is refused with the strict-slot message."""
    fem = _StubFem({"Soil": [
        [(1, (1, 2, 4, 5, 7, 9))],
        [(2, (5, 20, 21))],                      # T3 grabs mid-edge 5
    ]})
    with pytest.raises(BridgeError, match=r"mesh node 5: .* only accepts ndf"):
        infer_node_ndf(fem, [_up2d()], ndm=2)


def test_columnar_unregistered_class_quotes_first_node():
    """The refusal quotes the class's first fan-out row's first node, from
    the first non-empty spec, as the per-tuple walk did."""
    fem = _StubFem({
        "Empty": [],
        "A": [[(1, (42, 43))], [(2, (7, 8, 9))]],
        "B": [[(3, (5, 6))]],
    })
    specs = [_spec("NoSuchElement", "Empty"), _spec("NoSuchElement", "A"),
             _spec("NoSuchElement", "B")]
    with pytest.raises(BridgeError, match=r"^node 42: element class "
                       r"'NoSuchElement' is not in the capability registry"):
        infer_node_ndf(fem, specs, ndm=3)


def test_columnar_empty_class_is_skipped():
    """A class whose specs fan out to no rows is never probed, so even an
    unregistered one does not raise."""
    fem = _StubFem({"Empty": [], "Solid": [[(1, (1, 2, 3, 4, 5, 6, 7, 8))]]})
    specs = [_spec("NoSuchElement", "Empty"), _spec("stdBrick", "Solid")]
    assert infer_node_ndf(fem, specs, ndm=3) == {n: 3 for n in range(1, 9)}
