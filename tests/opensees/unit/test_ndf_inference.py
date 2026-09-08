"""ADR 0048 — element-class ndf inference engine (authoritative).

Covers the pure inference core, the adaptive-element carve-out, the ``ndm``
guard, and the PG-walk over a lightweight FEM stub. No gmsh / openseespy
needed. (The pre-clean-break shadow-mode parity gate was removed once
inference became the only path.)
"""
import pytest

from apeGmsh.opensees._internal.build import (
    BridgeError,
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
