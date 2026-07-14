"""ADR 0074 — LadrunoUP build-time machinery.

Covers, without gmsh / openseespy:

* per-SLOT ndf inference (D2): Taylor–Hood tri6/tet10 fan-outs put
  ``ndm+1`` on vertex slots and ``ndm`` on mid-edge slots; equal-order
  shapes stay scalar at ``ndm+1``; conforming TH patches resolve
  consistently; foreign elements on a mid-edge (or over-floored vertex)
  node fail loud through the strict per-slot ``ndf_ok`` gate.
* the ADR-0074 legality pass (``validate_ladruno_up_specs``): shape
  providers per ndm, perm-dimension coherence, ``stab``-on-TH fatality,
  and the D3 straight-side pre-check with the fork's 1e-6·edge tolerance.
* the D4 solver gate (``validate_ladruno_up_solver``): missing system,
  symmetric-storage / diagonal systems, the general-solver allow-list,
  and per-stage checking.
"""
from __future__ import annotations

import pytest

from apeGmsh.opensees._internal.build import (
    BridgeError,
    infer_node_ndf,
    validate_ladruno_up_solver,
    validate_ladruno_up_specs,
)
from apeGmsh.opensees.element.solid import LadrunoUP
from apeGmsh.opensees.material.nd import ElasticIsotropic


# ── lightweight FEM stub (element PG walk + nodal coords) ────────────────────

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
    """Exposes the ``index`` / ``coords`` surface ``_node_coord`` touches."""

    def __init__(self, coords_by_tag):
        self._tags = sorted(coords_by_tag)
        self._pos = {t: i for i, t in enumerate(self._tags)}
        self.coords = [coords_by_tag[t] for t in self._tags]

    def index(self, tag):
        return self._pos[int(tag)]


class _StubFem:
    def __init__(self, pg_groups, coords_by_tag=None):
        self.elements = _StubElements(pg_groups)
        if coords_by_tag is not None:
            self.nodes = _StubNodes(coords_by_tag)


def _make_material() -> ElasticIsotropic:
    return ElasticIsotropic(E=30e6, nu=0.3, rho=2.0)


def _up(pg: str = "Soil", dim: int = 2, **kw: object) -> LadrunoUP:
    base: dict[str, object] = dict(
        pg=pg, material=_make_material(), Kf=2.2e6, poro=0.4, rhoF=1.0,
        perm=(1e-8,) * dim,
    )
    base.update(kw)
    return LadrunoUP(**base)  # type: ignore[arg-type]


def _spec(class_name: str, pg: str):
    """A throwaway Element-spec stand-in whose type name is *class_name*."""
    return type(class_name, (), {"pg": pg})()


# ── unit-square BT6 patch coordinates (fork guide §6.2 idiom) ─────────────────
# Two straight-sided tri6 cells over the unit square: vertices 1-4, mid-edge
# nodes 5-9 at exact edge midpoints.  Element A = (1,2,4) edges (1,2),(2,4),
# (4,1); element B = (2,3,4) edges (2,3),(3,4),(4,2).  The diagonal 2-4 is
# SHARED: its mid-edge node 7 appears in both cells' mid-edge slots
# (mid-edge ↔ mid-edge only, the conforming-mesh invariant).
_BT6_COORDS = {
    1: (0.0, 0.0, 0.0),
    2: (1.0, 0.0, 0.0),
    3: (1.0, 1.0, 0.0),
    4: (0.0, 1.0, 0.0),
    5: (0.5, 0.0, 0.0),   # mid (1,2)
    6: (1.0, 0.5, 0.0),   # mid (2,3)
    7: (0.5, 0.5, 0.0),   # mid (2,4) — the shared diagonal
    8: (0.5, 1.0, 0.0),   # mid (3,4)
    9: (0.0, 0.5, 0.0),   # mid (4,1)
}
_BT6_CONN_A = (1, 2, 4, 5, 7, 9)   # v0,v1,v2, m(v0,v1), m(v1,v2), m(v2,v0)
_BT6_CONN_B = (2, 3, 4, 6, 8, 7)


def _bt6_fem(coords=None):
    return _StubFem(
        {"Soil": [[(1, _BT6_CONN_A), (2, _BT6_CONN_B)]]},
        coords_by_tag=coords or _BT6_COORDS,
    )


# ============================================================================
# D2 — per-slot ndf inference
# ============================================================================

class TestPerSlotInference:
    def test_tri6_vertices_3_midedges_2(self) -> None:
        fem = _StubFem({"Soil": [[(1, _BT6_CONN_A)]]})
        out = infer_node_ndf(fem, [_up()], ndm=2)
        assert out[1] == 3 and out[2] == 3 and out[4] == 3
        assert out[5] == 2 and out[7] == 2 and out[9] == 2

    def test_conforming_bt6_patch_shared_midedge(self) -> None:
        """The shared-diagonal mid-edge node (7) sits in a mid-edge slot of
        BOTH cells — same floor from both, resolves to ndm=2 cleanly."""
        out = infer_node_ndf(_bt6_fem(), [_up()], ndm=2)
        assert {t: out[t] for t in (1, 2, 3, 4)} == {1: 3, 2: 3, 3: 3, 4: 3}
        assert {t: out[t] for t in (5, 6, 7, 8, 9)} == {
            5: 2, 6: 2, 7: 2, 8: 2, 9: 2,
        }

    def test_tet10_vertices_4_midedges_3(self) -> None:
        conn = tuple(range(1, 11))
        fem = _StubFem({"Soil": [[(1, conn)]]})
        out = infer_node_ndf(fem, [_up(dim=3)], ndm=3)
        assert all(out[t] == 4 for t in (1, 2, 3, 4))
        assert all(out[t] == 3 for t in range(5, 11))

    @pytest.mark.parametrize(
        "conn, ndm, dim, expected",
        [
            ((1, 2, 3), 2, 2, 3),                    # T3
            ((1, 2, 3, 4), 2, 2, 3),                 # Q4
            (tuple(range(1, 9)), 3, 3, 4),           # H8
        ],
    )
    def test_equal_order_shapes_scalar_floor(
        self, conn: tuple, ndm: int, dim: int, expected: int,
    ) -> None:
        fem = _StubFem({"Soil": [[(1, conn)]]})
        out = infer_node_ndf(fem, [_up(dim=dim)], ndm=ndm)
        assert all(out[t] == expected for t in conn)

    def test_midedge_node_shared_with_beam_fails_loud(self) -> None:
        """ADR 0074: a mid-edge node touched by a foreign element resolves
        through the strict slot gate — floor max(2, 3)=3 is rejected by the
        mid-edge slot's {2}."""
        fem = _StubFem({
            "Soil": [[(1, _BT6_CONN_A)]],
            "Frame": [[(2, (5, 99))]],   # beam grabs mid-edge node 5
        })
        elements = [_up(), _spec("elasticBeamColumn", "Frame")]
        with pytest.raises(BridgeError, match="only accepts ndf"):
            infer_node_ndf(fem, elements, ndm=2)

    def test_vertex_node_shared_with_shell_fails_loud(self) -> None:
        """A shell (floor 6) on an H8 u-p vertex (exactly 4) must raise."""
        fem = _StubFem({
            "Soil": [[(1, tuple(range(1, 9)))]],
            "Skin": [[(2, (1, 2, 3, 4))]],
        })
        elements = [_up(dim=3), _spec("ShellMITC4", "Skin")]
        with pytest.raises(BridgeError, match="only accepts ndf"):
            infer_node_ndf(fem, elements, ndm=3)

    def test_equal_order_node_shared_with_2d_solid_fails_loud(self) -> None:
        """Dry quad (ndf 2) sharing a saturated Q4 node (exactly 3) is the
        ADR-0069 duplicated-interface case — inference must refuse the
        shared node (quad's {2} rejects floor 3)."""
        fem = _StubFem({
            "Wet": [[(1, (1, 2, 3, 4))]],
            "Dry": [[(2, (3, 4, 5, 6))]],
        })
        elements = [_up(pg="Wet"), _spec("quad", "Dry")]
        with pytest.raises(BridgeError, match="SEPARATE coincident nodes"):
            infer_node_ndf(fem, elements, ndm=2)


# ============================================================================
# Legality pass + D3 straight-side
# ============================================================================

class TestValidateSpecs:
    def test_straight_bt6_patch_passes(self) -> None:
        validate_ladruno_up_specs(_bt6_fem(), [_up()], ndm=2)

    def test_curved_midedge_fails_with_context(self) -> None:
        coords = dict(_BT6_COORDS)
        coords[5] = (0.5, 0.01, 0.0)   # bow the (1,2) mid-edge node
        with pytest.raises(BridgeError, match="straight-side") as ei:
            validate_ladruno_up_specs(_bt6_fem(coords), [_up()], ndm=2)
        msg = str(ei.value)
        assert "node 5" in msg and "element 1" in msg
        assert "high-order optimization" in msg

    def test_tolerance_is_relative_to_edge_length(self) -> None:
        """An offset just inside 1e-6·edge passes; just outside fails."""
        coords = dict(_BT6_COORDS)
        coords[5] = (0.5, 0.9e-6, 0.0)   # edge (1,2) has length 1.0
        validate_ladruno_up_specs(_bt6_fem(coords), [_up()], ndm=2)
        coords[5] = (0.5, 1.1e-6, 0.0)
        with pytest.raises(BridgeError, match="straight-side"):
            validate_ladruno_up_specs(_bt6_fem(coords), [_up()], ndm=2)

    def test_tet10_straight_passes_and_curved_fails(self) -> None:
        verts = {
            1: (0.0, 0.0, 0.0), 2: (1.0, 0.0, 0.0),
            3: (0.0, 1.0, 0.0), 4: (0.0, 0.0, 1.0),
        }
        edges = [(1, 2), (2, 3), (1, 3), (1, 4), (3, 4), (2, 4)]
        coords = dict(verts)
        for i, (a, b) in enumerate(edges):
            xa, xb = verts[a], verts[b]
            coords[5 + i] = tuple((xa[k] + xb[k]) / 2.0 for k in range(3))
        conn = tuple(range(1, 11))
        fem = _StubFem({"Soil": [[(1, conn)]]}, coords_by_tag=coords)
        validate_ladruno_up_specs(fem, [_up(dim=3)], ndm=3)
        coords[5] = (0.5, 0.05, 0.0)   # bow mid-edge (1,2)
        fem = _StubFem({"Soil": [[(1, conn)]]}, coords_by_tag=coords)
        with pytest.raises(BridgeError, match="straight-side"):
            validate_ladruno_up_specs(fem, [_up(dim=3)], ndm=3)

    def test_tet4_cell_has_no_provider(self) -> None:
        fem = _StubFem({"Soil": [[(1, (1, 2, 3, 4))]]})
        with pytest.raises(BridgeError, match="no shape provider"):
            validate_ladruno_up_specs(fem, [_up(dim=3)], ndm=3)

    def test_perm_dim_must_match_ndm(self) -> None:
        fem = _StubFem({"Soil": [[(1, (1, 2, 3, 4))]]})
        with pytest.raises(BridgeError, match="exactly ndm values"):
            validate_ladruno_up_specs(fem, [_up(dim=3)], ndm=2)

    def test_stab_on_th_pg_fails_loud(self) -> None:
        with pytest.raises(BridgeError, match="inf-sup stable"):
            validate_ladruno_up_specs(_bt6_fem(), [_up(stab="auto")], ndm=2)

    def test_stab_on_equal_order_pg_passes(self) -> None:
        fem = _StubFem({"Soil": [[(1, (1, 2, 3, 4))]]})
        validate_ladruno_up_specs(fem, [_up(stab="auto")], ndm=2)

    def test_non_up_specs_are_ignored(self) -> None:
        fem = _StubFem({"Other": [[(1, (1, 2))]]})
        validate_ladruno_up_specs(
            fem, [_spec("elasticBeamColumn", "Other")], ndm=2,
        )


# ============================================================================
# D4 — solver gate
# ============================================================================

class TestSolverGate:
    def test_no_system_declared_raises(self) -> None:
        with pytest.raises(BridgeError, match="ProfileSPD"):
            validate_ladruno_up_solver([_up()], [])

    @pytest.mark.parametrize(
        "sys_name",
        ["ProfileSPD", "SProfileSPD", "ParallelProfileSPD", "BandSPD",
         "SparseSYM", "Diagonal", "MPIDiagonal"],
    )
    def test_symmetric_or_diagonal_system_raises(self, sys_name: str) -> None:
        import apeGmsh.opensees.analysis.system as sysmod

        system = getattr(sysmod, sys_name)()
        with pytest.raises(BridgeError, match="UNSYMMETRIC"):
            validate_ladruno_up_solver([_up()], [("global", system)])

    @pytest.mark.parametrize(
        "sys_name",
        ["UmfPack", "SparseGeneral", "FullGeneral", "BandGeneral", "Mumps"],
    )
    def test_general_system_passes(self, sys_name: str) -> None:
        import apeGmsh.opensees.analysis.system as sysmod

        system = getattr(sysmod, sys_name)()
        validate_ladruno_up_solver([_up()], [("global", system)])

    def test_one_bad_stage_raises_naming_the_stage(self) -> None:
        from apeGmsh.opensees.analysis.system import ProfileSPD, UmfPack

        with pytest.raises(BridgeError, match="stage 'shake'"):
            validate_ladruno_up_solver(
                [_up()],
                [
                    ("stage 'gravity'", UmfPack()),
                    ("stage 'shake'", ProfileSPD()),
                ],
            )

    def test_all_stages_legal_passes(self) -> None:
        from apeGmsh.opensees.analysis.system import UmfPack

        validate_ladruno_up_solver(
            [_up()],
            [("stage 'gravity'", UmfPack()), ("stage 'shake'", UmfPack())],
        )

    def test_no_up_elements_no_gate(self) -> None:
        """Non-u-p decks keep the OpenSees default freedom — no raise even
        with nothing declared."""
        validate_ladruno_up_solver([_spec("stdBrick", "Body")], [])
        validate_ladruno_up_solver([], [])
