"""Chain-phase fail-loud matrix — silent-failures slice 2.

In a ``from_h5`` / compose session (``g._fem_from_h5``) there is no gmsh
and never another extraction, so any def the chain-phase router cannot
apply used to be a **permanent silent no-op**: the def landed on its
list, the counter bumped, and the next ``get_fem_data()`` handed back
the untouched cached broker.  The model then solved without the
constraint / load / mass — converged, plausible, wrong.

These tests lock the new contract:

* misspelled targets (``bc`` / point load / point mass) → ``KeyError``
  propagates (was: swallowed);
* def kinds the router does not cover (``displacements.surface``,
  ``loads.gravity``, distributed masses) → ``ChainPhaseError``
  (was: stored-never-applied);
* gmsh-resolving verbs (``contact`` / ``contact_plane`` / ``embed`` /
  ``reinforce`` / ``decouple_node``) → ``ChainPhaseError`` at the verb
  (was: def stored, resolver needs live gmsh, never ran);
* a tie whose slave nodes all fail the projection tolerance →
  ``ValueError`` from the shared resolver; a partial projection warns;
* LIVE sessions (no ``_fem_from_h5``) keep the lenient fall-through —
  re-extraction resolves the def and fails loud there.

Runs entirely off the FEMData broker — no live gmsh, no openseespy.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from apeGmsh._core import apeGmsh
from apeGmsh._kernel.defs.constraints import TieDef
from apeGmsh._kernel.defs.loads import GravityLoadDef
from apeGmsh._kernel.payloads import ElementGroup
from apeGmsh._kernel.record_sets import ComposeSet
from apeGmsh._kernel.resolvers._chain_phase_router import (
    route_def_to_fem,
    try_chain_phase_route,
)
from apeGmsh.core._compose_errors import ChainPhaseError
from apeGmsh.mesh._element_types import make_type_info
from apeGmsh.mesh._group_set import LabelSet, PhysicalGroupSet
from apeGmsh.mesh.FEMData import (
    ElementComposite,
    FEMData,
    MeshInfo,
    NodeComposite,
)


def _quad_face_fem(
    *, slave_z: float = 0.0, slave_z2: float | None = None,
) -> FEMData:
    """One master quad at z=0 + two slave nodes at ``slave_z``.

    ``slave_z=0`` puts the slaves ON the quad (ties resolve);
    ``slave_z=5`` puts them far above it (every projection fails a
    small tolerance).  ``slave_z2`` overrides the second slave's z for
    the partial-projection case.
    """
    z2 = slave_z if slave_z2 is None else slave_z2
    coords = np.array(
        [
            [0.0, 0.0, 0.0], [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0], [0.0, 1.0, 0.0],       # quad corners
            [0.25, 0.25, slave_z], [0.75, 0.75, z2],   # slaves
        ],
        dtype=np.float64,
    )
    node_ids = np.array([1, 2, 3, 4, 5, 6], dtype=np.int64)
    quad_info = make_type_info(
        code=3, gmsh_name="Quadrangle 4", dim=2, order=1, npe=4, count=1,
    )
    quad_group = ElementGroup(
        element_type=quad_info,
        ids=np.array([500], dtype=np.int64),
        connectivity=np.array([[1, 2, 3, 4]], dtype=np.int64),
    )
    nodes = NodeComposite(
        node_ids=node_ids, node_coords=coords,
        physical=PhysicalGroupSet({
            (0, 1): {"name": "m_face",
                     "node_ids": np.array([1, 2, 3, 4], dtype=np.int64),
                     "node_coords": coords[:4]},
            (0, 2): {"name": "s_nodes",
                     "node_ids": np.array([5, 6], dtype=np.int64),
                     "node_coords": coords[4:6]},
        }),
        labels=LabelSet({}),
    )
    elements = ElementComposite(
        groups={3: quad_group},
        physical=PhysicalGroupSet({}), labels=LabelSet({}),
    )
    info = MeshInfo(n_nodes=6, n_elems=1, bandwidth=1, types=[quad_info])
    return FEMData(nodes=nodes, elements=elements, info=info,
                   composed_from=ComposeSet(()))


def _from_h5(fem: FEMData, tmp_path: Path) -> apeGmsh:
    path = tmp_path / "m.h5"
    fem.to_h5(str(path))
    return apeGmsh.from_h5(path)


# ---------------------------------------------------------------------
# Misspelled targets → KeyError propagates in from_h5 sessions
# ---------------------------------------------------------------------


class TestMisspelledTargetsRaise:
    def test_bc_typo_raises(self, tmp_path: Path) -> None:
        g = _from_h5(_quad_face_fem(), tmp_path)
        with pytest.raises(KeyError, match="m_face_TYPO"):
            g.constraints.bc("m_face_TYPO", dofs=[1, 1, 1])
        assert len(list(g._fem.nodes.sp)) == 0

    def test_point_load_typo_raises(self, tmp_path: Path) -> None:
        g = _from_h5(_quad_face_fem(), tmp_path)
        with pytest.raises(KeyError, match="s_nodes_TYPO"):
            with g.loads.case("push"):
                g.loads.point.force("s_nodes_TYPO", force=(0, 0, -1.0))
        assert len(list(g._fem.nodes.loads)) == 0

    def test_point_mass_typo_raises(self, tmp_path: Path) -> None:
        g = _from_h5(_quad_face_fem(), tmp_path)
        with pytest.raises(KeyError, match="s_nodes_TYPO"):
            g.masses.point("s_nodes_TYPO", mass=10.0)
        assert len(list(g._fem.nodes.masses)) == 0


# ---------------------------------------------------------------------
# Unrouted def kinds → ChainPhaseError in from_h5 sessions
# ---------------------------------------------------------------------


class TestUnroutedKindsRaise:
    def test_displacement_surface_raises(self, tmp_path: Path) -> None:
        """FaceSPDef has no router branch — the displacement would be
        stored and silently never applied (the defect-3 companion
        trap: the case exists, the records never do)."""
        g = _from_h5(_quad_face_fem(), tmp_path)
        with pytest.raises(ChainPhaseError, match="FaceSPDef"):
            with g.displacements.case("push_gap"):
                g.displacements.surface("m_face", disp_xyz=(0, 0, -1.0))
        assert len(list(g._fem.nodes.sp)) == 0

    def test_gravity_load_raises(self, tmp_path: Path) -> None:
        g = _from_h5(_quad_face_fem(), tmp_path)
        with pytest.raises(ChainPhaseError, match="GravityLoadDef"):
            with g.loads.case("dead"):
                g.loads.gravity("m_face", g=(0, 0, -9.81), density=2400.0)
        assert len(list(g._fem.nodes.loads)) == 0

    def test_live_session_stays_lenient(self) -> None:
        """A session WITHOUT _fem_from_h5 (live post-extraction cache)
        keeps the bump-counter fall-through: the router returns False,
        nothing raises — re-extraction applies the def later."""
        fem = _quad_face_fem()

        class Stub:
            _fem = fem

        defn = GravityLoadDef(target="m_face", target_source="auto",
                              g=(0.0, 0.0, -9.81), density=2400.0)
        assert try_chain_phase_route(Stub(), defn) is False


# ---------------------------------------------------------------------
# gmsh-resolving verbs → ChainPhaseError at the verb
# ---------------------------------------------------------------------


class TestGmshResolvingVerbsRaise:
    def test_contact_raises(self, tmp_path: Path) -> None:
        g = _from_h5(_quad_face_fem(), tmp_path)
        with pytest.raises(ChainPhaseError, match="contact"):
            g.constraints.contact("m_face", "s_nodes",
                                  formulation="mortar", tie=True)
        assert len(g.constraints.contact_defs) == 0

    def test_contact_plane_raises(self, tmp_path: Path) -> None:
        g = _from_h5(_quad_face_fem(), tmp_path)
        with pytest.raises(ChainPhaseError, match="contact_plane"):
            g.constraints.contact_plane(
                "s_nodes", normal=(0, 0, 1), point=(0, 0, 0), kn=1e6)
        assert len(g.constraints.contact_plane_defs) == 0

    def test_embed_raises(self, tmp_path: Path) -> None:
        g = _from_h5(_quad_face_fem(), tmp_path)
        with pytest.raises(ChainPhaseError, match="g.embed"):
            g.embed(host="m_face", nodes="s_nodes")

    def test_reinforce_raises(self, tmp_path: Path) -> None:
        g = _from_h5(_quad_face_fem(), tmp_path)
        with pytest.raises(ChainPhaseError, match="g.reinforce"):
            g.reinforce("m_face", "s_nodes")

    def test_decouple_node_raises(self, tmp_path: Path) -> None:
        g = _from_h5(_quad_face_fem(), tmp_path)
        with pytest.raises(ChainPhaseError, match="decouple_node"):
            g.decouple_node(coords=(0.5, 0.5, 2.0))


# ---------------------------------------------------------------------
# Zero-record / partial ties
# ---------------------------------------------------------------------


class TestTieProjectionGuards:
    def test_zero_projection_raises(self, tmp_path: Path) -> None:
        """Valid labels, but every slave node fails the tolerance —
        the tie would bond nothing.  Was: silent, zero records."""
        g = _from_h5(_quad_face_fem(slave_z=5.0), tmp_path)
        with pytest.raises(ValueError, match="resolved 0 records"):
            g.constraints.tie("m_face", "s_nodes", dofs=[1, 2, 3],
                              tolerance=0.1)
        assert len(list(g._fem.elements.constraints)) == 0

    def test_partial_projection_warns(self) -> None:
        """One slave on the face, one far above it: the tie resolves
        but warns with counts (build/chain shared resolver guard)."""
        fem = _quad_face_fem(slave_z=0.0, slave_z2=5.0)
        defn = TieDef(master_label="m_face", slave_label="s_nodes",
                      dofs=[1, 2, 3], tolerance=0.1)
        with pytest.warns(UserWarning, match="only 1 of 2"):
            new_fem = route_def_to_fem(fem, defn)
        assert len(list(new_fem.elements.constraints)) == 1

    def test_full_projection_stays_silent(self, tmp_path: Path) -> None:
        """Both slaves on the quad: resolves 2 records, no warning."""
        import warnings

        g = _from_h5(_quad_face_fem(), tmp_path)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            g.constraints.tie("m_face", "s_nodes", dofs=[1, 2, 3],
                              tolerance=0.1)
        assert len(list(g._fem.elements.constraints)) == 2
