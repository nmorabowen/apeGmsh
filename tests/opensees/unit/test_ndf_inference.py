"""ADR 0048 — element-class ndf inference engine + shadow-mode parity (PR-1).

Covers the pure inference core, the ``ndm`` guard, the PG-walk over a
lightweight FEM stub (the path the deleted P2a version never unit-tested), and
the non-breaking parity check. No gmsh / openseespy needed.
"""
import warnings

import pytest

from apeGmsh.opensees._internal.build import (
    BridgeError,
    NdfInferenceParityWarning,
    _infer_ndf_from_incidence,
    assert_ndm_compatible,
    infer_node_ndf,
    warn_on_ndf_inference_parity,
)


@pytest.fixture(autouse=True)
def _enable_ndf_parity(monkeypatch):
    """``warn_on_ndf_inference_parity`` is opt-in (a migration diagnostic);
    enable it for the tests in this module. Harmless for the pure-core tests
    that never call it."""
    monkeypatch.setenv("APEGMSH_NDF_PARITY", "1")


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


class _StubNodes:
    def __init__(self, ndf_map=None):
        self._ndf = ndf_map or {}

    def ndf_for(self, tag):
        try:
            return self._ndf[int(tag)]
        except KeyError:
            raise LookupError(tag) from None


class _StubFem:
    """Exposes just the surface ``infer_node_ndf`` / parity touch:
    ``elements.select(pg).groups()`` and ``nodes.ndf_for(tag)``."""

    def __init__(self, pg_groups, ndf_map=None):
        self.elements = _StubElements(pg_groups)
        self.nodes = _StubNodes(ndf_map)


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


# ─────────────────────── shadow-mode parity ─────────────────────────────────

def test_parity_clean_homogeneous_no_warning():
    fem = _StubFem({"Solid": [[(1, (10, 11, 12, 13))]]})
    elements = [_spec("stdBrick", "Solid")]
    with warnings.catch_warnings():
        warnings.simplefilter("error", NdfInferenceParityWarning)
        # envelope ndf 3 == inferred 3 for every brick node → no warning
        warn_on_ndf_inference_parity(fem, elements, ndm=3, envelope_ndf=3)


def test_parity_mismatch_warns():
    fem = _StubFem({"Solid": [[(1, (10, 11, 12, 13))]]})
    elements = [_spec("stdBrick", "Solid")]
    with pytest.warns(NdfInferenceParityWarning, match="inferred 3 vs emitted 6"):
        # envelope ndf 6 but bricks infer 3 → mismatch on every node
        warn_on_ndf_inference_parity(fem, elements, ndm=3, envelope_ndf=6)


def test_parity_respects_broker_override_no_warning():
    # g.node_ndf set node 13 to 6, others uncovered (fall back to envelope 3).
    # Brick infers 3 everywhere → node 13 mismatches (6 vs 3) → warns.
    fem = _StubFem({"Solid": [[(1, (10, 11, 12, 13))]]}, ndf_map={13: 6})
    elements = [_spec("stdBrick", "Solid")]
    with pytest.warns(NdfInferenceParityWarning, match="node 13"):
        warn_on_ndf_inference_parity(fem, elements, ndm=3, envelope_ndf=3)


def test_parity_would_fail_warns_not_raises():
    fem = _StubFem({"Bad": [[(1, (10, 11))]]})
    elements = [_spec("TotallyUnknownElement", "Bad")]
    with pytest.warns(NdfInferenceParityWarning, match="would fail loud"):
        warn_on_ndf_inference_parity(fem, elements, ndm=3, envelope_ndf=3)
