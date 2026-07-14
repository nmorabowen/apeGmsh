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

import numpy as np
import pytest

from apeGmsh.opensees._internal.build import (
    BridgeError,
    infer_node_ndf,
    validate_ladruno_up_pressure_dof,
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
    """Exposes the ``ids`` / ``index`` / ``coords`` surface the coord
    lookups (``_node_coord`` + the vectorized straight-side gather) touch."""

    def __init__(self, coords_by_tag):
        self._tags = sorted(coords_by_tag)
        self._pos = {t: i for i, t in enumerate(self._tags)}
        self.ids = np.asarray(self._tags, dtype=np.int64)
        self.coords = [coords_by_tag[t] for t in self._tags]

    def index(self, tag):
        return self._pos[int(tag)]


class _StubFem:
    def __init__(self, pg_groups, coords_by_tag=None):
        self.elements = _StubElements(pg_groups)
        if coords_by_tag is not None:
            self.nodes = _StubNodes(coords_by_tag)


# ── typed-group stub: carries an EXPLICIT Gmsh etype code, so the
#    count-aliasing shapes (quad8 code 16 = 8 nodes but NOT an H8; line3
#    code 8 = 3 nodes but NOT a T3) can be exercised without a real mesh.
class _TypeInfo:
    def __init__(self, code, name):
        self.code = code
        self.name = name


class _TypedGroup:
    def __init__(self, code, name, rows):
        self.element_type = _TypeInfo(code, name)
        self.ids = np.asarray([e for e, _c in rows], dtype=np.int64)
        self.connectivity = np.asarray([c for _e, c in rows], dtype=np.int64)


class _TypedSel:
    def __init__(self, groups):
        self._groups = groups

    def groups(self):
        return self._groups


class _TypedElements:
    def __init__(self, pg_groups):
        self._pg_groups = pg_groups

    def select(self, pg):
        if pg not in self._pg_groups:
            raise KeyError(pg)
        return _TypedSel(self._pg_groups[pg])


class _TypedFem:
    """FEM stub whose groups carry a real ``element_type.code`` (the etype
    path) plus nodal coords for any straight-side checks."""

    def __init__(self, pg_typed_groups, coords_by_tag=None):
        self.elements = _TypedElements(pg_typed_groups)
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
# D4 — solver gate (rescoped: validates what the deck EMITS-and-SOLVES)
# ============================================================================

def _sys(name: str):
    import apeGmsh.opensees.analysis.system as sysmod

    return getattr(sysmod, name)()


def _flat(elements, systems, *, enforce=True, partitioned=False):
    validate_ladruno_up_solver(
        elements, enforce=enforce, staged=False, partitioned=partitioned,
        flat_systems=systems, stage_systems=[],
    )


def _staged(elements, stage_systems, *, enforce=True):
    validate_ladruno_up_solver(
        elements, enforce=enforce, staged=True, partitioned=False,
        flat_systems=[], stage_systems=stage_systems,
    )


class TestSolverGate:
    # -- flat --------------------------------------------------------------
    def test_no_system_declared_raises(self) -> None:
        with pytest.raises(BridgeError, match="no linear system declared"):
            _flat([_up()], [])

    @pytest.mark.parametrize(
        "sys_name",
        ["ProfileSPD", "SProfileSPD", "ParallelProfileSPD", "BandSPD",
         "SparseSYM", "Diagonal", "MPIDiagonal"],
    )
    def test_symmetric_or_diagonal_system_raises(self, sys_name: str) -> None:
        with pytest.raises(BridgeError, match="UNSYMMETRIC"):
            _flat([_up()], [_sys(sys_name)])

    @pytest.mark.parametrize(
        "sys_name",
        ["UmfPack", "SparseGeneral", "FullGeneral", "BandGeneral", "Mumps"],
    )
    def test_general_system_passes(self, sys_name: str) -> None:
        _flat([_up()], [_sys(sys_name)])

    def test_last_declared_system_is_the_effective_one(self) -> None:
        """OpenSees uses the most-recently-declared system at analyze; a
        superseded ProfileSPD before a legal UmfPack must NOT falsely
        reject."""
        _flat([_up()], [_sys("ProfileSPD"), _sys("UmfPack")])
        with pytest.raises(BridgeError, match="UNSYMMETRIC"):
            _flat([_up()], [_sys("UmfPack"), _sys("ProfileSPD")])

    def test_flat_no_system_partitioned_is_allowed(self) -> None:
        """A partitioned flat deck with no system rides the ADR-0027 INV-5
        auto-emitted general Mumps/UmfPack fallback — no raise."""
        _flat([_up()], [], partitioned=True)

    # -- enforce gate (archival / eigen / model-only) ---------------------
    def test_missing_system_skipped_when_not_enforced(self) -> None:
        """Archival / eigen-only / model-only-skeleton emits never solve, so a
        MISSING system is not refused."""
        _flat([_up()], [], enforce=False)

    def test_declared_symmetric_system_raises_even_when_not_enforced(
        self,
    ) -> None:
        """A DECLARED symmetric system is unambiguously wrong with u-p — it is
        refused whether or not this emit runs analyze (a model-only Tcl export
        still catches ops.system.ProfileSPD())."""
        with pytest.raises(BridgeError, match="UNSYMMETRIC"):
            _flat([_up()], [_sys("ProfileSPD")], enforce=False)

    # -- staged ------------------------------------------------------------
    def test_one_bad_stage_raises_naming_the_stage(self) -> None:
        with pytest.raises(BridgeError, match="stage 'shake'"):
            _staged(
                [_up()],
                [("'gravity'", _sys("UmfPack")),
                 ("'shake'", _sys("ProfileSPD"))],
            )

    def test_stage_with_no_system_raises(self) -> None:
        """The load-bearing fix: a stage that analyzes with system=None runs
        on the wipeAnalysis ProfileSPD default — must raise even though
        another stage declared a legal system."""
        with pytest.raises(BridgeError, match="stage 'shake'.*ProfileSPD"):
            _staged(
                [_up()],
                [("'gravity'", _sys("UmfPack")), ("'shake'", None)],
            )

    def test_all_stages_legal_passes(self) -> None:
        _staged(
            [_up()],
            [("'gravity'", _sys("UmfPack")), ("'shake'", _sys("UmfPack"))],
        )

    def test_staged_ignores_global_systems(self) -> None:
        """A stray global system is never emitted in staged mode, so it must
        not be validated (staged decks pass their own per-stage systems as
        stage_systems; flat_systems is empty)."""
        _staged([_up()], [("'g'", _sys("UmfPack"))])  # global ProfileSPD absent

    # -- shared ------------------------------------------------------------
    def test_no_up_elements_no_gate(self) -> None:
        _flat([_spec("stdBrick", "Body")], [])
        _flat([], [])


# ============================================================================
# F5 — etype-based shape legality (count-aliasing cells)
# ============================================================================

class TestEtypeLegality:
    def test_quad8_surface_in_3d_is_rejected(self) -> None:
        """An 8-node serendipity quad8 (code 16) in a 3D model has 8 nodes
        like an H8 but is NOT a volume provider — node count alone would
        wave it through; the etype must reject it."""
        fem = _TypedFem({
            "Soil": [_TypedGroup(16, "quad8", [(1, tuple(range(1, 9)))])],
        })
        with pytest.raises(BridgeError, match="no shape provider"):
            validate_ladruno_up_specs(fem, [_up(dim=3)], ndm=3)

    def test_line3_curve_in_2d_is_rejected(self) -> None:
        fem = _TypedFem({
            "Soil": [_TypedGroup(8, "line3", [(1, (1, 2, 3))])],
        })
        with pytest.raises(BridgeError, match="no shape provider"):
            validate_ladruno_up_specs(fem, [_up(dim=2)], ndm=2)

    def test_true_hex8_in_3d_passes(self) -> None:
        fem = _TypedFem({
            "Soil": [_TypedGroup(5, "hexa8", [(1, tuple(range(1, 9)))])],
        })
        validate_ladruno_up_specs(fem, [_up(dim=3)], ndm=3)

    def test_true_quad4_in_2d_passes(self) -> None:
        fem = _TypedFem({
            "Soil": [_TypedGroup(3, "quad4", [(1, (1, 2, 3, 4))])],
        })
        validate_ladruno_up_specs(fem, [_up(dim=2)], ndm=2)


# ============================================================================
# F2 — rotation-vs-pressure DOF aliasing
# ============================================================================

class TestPressureDofAliasing:
    def _soil_and_beam_sharing_node(self):
        # Equal-order quad4 LadrunoUP soil (nodes 1-4) + a 2D beam on nodes
        # (4, 5): node 4 is a shared saturated pressure-carrier.
        return _StubFem({
            "Soil": [[(1, (1, 2, 3, 4))]],
            "Frame": [[(2, (4, 5))]],
        })

    def test_beam_sharing_equal_order_carrier_raises(self) -> None:
        fem = self._soil_and_beam_sharing_node()
        elements = [_up(pg="Soil"), _spec("elasticBeamColumn", "Frame")]
        with pytest.raises(BridgeError, match="pressure-carrier node"):
            validate_ladruno_up_pressure_dof(fem, elements, ndm=2)

    def test_error_names_node_and_beam_class(self) -> None:
        fem = self._soil_and_beam_sharing_node()
        elements = [_up(pg="Soil"), _spec("elasticBeamColumn", "Frame")]
        with pytest.raises(BridgeError) as ei:
            validate_ladruno_up_pressure_dof(fem, elements, ndm=2)
        msg = str(ei.value)
        assert "node 4" in msg and "elasticBeamColumn" in msg
        assert "equal_dof" in msg

    def test_beam_not_sharing_a_carrier_passes(self) -> None:
        fem = _StubFem({
            "Soil": [[(1, (1, 2, 3, 4))]],
            "Frame": [[(2, (5, 6))]],   # disjoint nodes
        })
        elements = [_up(pg="Soil"), _spec("elasticBeamColumn", "Frame")]
        validate_ladruno_up_pressure_dof(fem, elements, ndm=2)

    def test_truss_sharing_carrier_is_allowed(self) -> None:
        """A truss uses only translation DOFs (floor ndm, not ndm+1), so it
        does not touch the pressure slot — sharing is fine."""
        fem = _StubFem({
            "Soil": [[(1, (1, 2, 3, 4))]],
            "Tie": [[(2, (4, 5))]],
        })
        elements = [_up(pg="Soil"), _spec("truss", "Tie")]
        validate_ladruno_up_pressure_dof(fem, elements, ndm=2)

    def test_th_midedge_shared_with_beam_is_not_this_guards_job(self) -> None:
        """A beam on a TH MID-EDGE node (ndf=ndm) resolves to a strict-set
        mismatch caught by infer_node_ndf; the carrier guard only covers
        pressure-carrier (vertex / equal-order) nodes and must not fire on a
        mid-edge-only share."""
        fem = _StubFem({
            "Soil": [[(1, (1, 2, 3, 4, 5, 6))]],   # tri6: verts 1-3, mids 4-6
            "Frame": [[(2, (5, 9))]],              # beam on mid-edge node 5
        })
        elements = [_up(pg="Soil"), _spec("elasticBeamColumn", "Frame")]
        # No raise from THIS guard (node 5 is a mid-edge, not a carrier).
        validate_ladruno_up_pressure_dof(fem, elements, ndm=2)
