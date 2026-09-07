"""A2 — the build-time pressure-datum gate over real Gmsh meshes.

A saturated (u-p) region whose pressure DOFs are ALL free is impervious: its
STATIC tangent is singular in ``p``, and the fork MEASURED (2026-07-11) that
every serial general solver factorises it through round-off and returns
``rc = 0`` with an arbitrary, solver-dependent pressure level — a silent wrong
answer, pinned by the strict xfail at
``OpenSees/tests/test_ladruno_up_element_analytic.py:533-548``.

Mesh-only lanes (gmsh required, NO fork build — nothing is ever solved):

* a sealed LadrunoUP column under ``ops.analysis.Static()`` is refused at
  emit, naming a node of the offending region;
* the same column with ONE drained-surface node's pressure DOF fixed passes;
* a Taylor-Hood tet10 box is ONE region (dropping the pressure-less mid-edge
  nodes must not shatter the graph), and its drained surface grounds through
  the VERTEX nodes — the idiom the error message names, because a whole-pg
  mask over-runs the mid-edge nodes' ndf and G3 refuses it;
* two element-disconnected u-p columns each need their OWN datum — fixing one
  column's surface leaves the other refused; fixing both passes;
* a pure-mechanical (no u-p element) model is untouched by the gate;
* the gate is scoped to Static: a sealed column under
  ``ops.analysis.Transient()`` — undrained loading, physically correct, the
  storage term regularises the p rows — is NOT refused.

The union-find component walk and the datum taxonomy (fix / sp / s.support on
slot ``ndm+1``) are unit-tested in
``tests/opensees/unit/test_ladruno_up_build_gates.py``.
"""
from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("gmsh")

from apeGmsh import apeGmsh  # noqa: E402
from apeGmsh.opensees import apeSees  # noqa: E402
from apeGmsh.opensees._internal.build import BridgeError  # noqa: E402


# ---------------------------------------------------------------------------
# Mesh helpers (the test_ladruno_up_emission.py fixture pattern)
# ---------------------------------------------------------------------------

def _quad_column_fem(g: apeGmsh, *, h: float = 2.0, size: float = 1.0):
    """A structured 1 x h quad column: pg 'Soil' (surface), 'Top' (y=h)."""
    g.model.geometry.add_rectangle(0.0, 0.0, 0.0, 1.0, h, label="soil")
    g.physical.add_surface("soil", name="Soil")
    g.model.select(dim=1).on_plane(
        (0, 0, 0), (0, 1, 0), tol=1e-6,
    ).to_physical("Base")
    g.model.select(dim=1).on_plane(
        (0, h, 0), (0, 1, 0), tol=1e-6,
    ).to_physical("Top")
    g.mesh.structured.set_recombine("soil", dim=2)
    g.mesh.sizing.set_global_size(size)
    g.mesh.generation.generate(2)
    g.mesh.structured.recombine()
    g.mesh.partitioning.renumber(base=1)
    return g.mesh.queries.get_fem_data(dim=2)


def _two_column_fem(g: apeGmsh, *, size: float = 1.0):
    """TWO element-disconnected quad columns, 4 units apart.

    Different heights (2 and 3) so each top edge gets its own plane
    selection: pgs 'SoilA' / 'TopA' (y = 2) and 'SoilB' / 'TopB' (y = 3).
    """
    g.model.geometry.add_rectangle(0.0, 0.0, 0.0, 1.0, 2.0, label="a")
    g.model.geometry.add_rectangle(5.0, 0.0, 0.0, 1.0, 3.0, label="b")
    g.physical.add_surface("a", name="SoilA")
    g.physical.add_surface("b", name="SoilB")
    g.model.select(dim=1).on_plane(
        (0, 2.0, 0), (0, 1, 0), tol=1e-6,
    ).to_physical("TopA")
    g.model.select(dim=1).on_plane(
        (0, 3.0, 0), (0, 1, 0), tol=1e-6,
    ).to_physical("TopB")
    g.mesh.structured.set_recombine("a", dim=2)
    g.mesh.structured.set_recombine("b", dim=2)
    g.mesh.sizing.set_global_size(size)
    g.mesh.generation.generate(2)
    g.mesh.structured.recombine()
    g.mesh.partitioning.renumber(base=1)
    return g.mesh.queries.get_fem_data(dim=2)


def _up_bridge(fem, *, pgs=("Soil",), ndm: int = 2, ndf: int = 3) -> apeSees:
    """A solve-bearing LadrunoUP bridge: legal general solver + Static."""
    ops = apeSees(fem)
    ops.model(ndm=ndm, ndf=ndf)
    mat = ops.nDMaterial.ElasticIsotropic(E=1e4, nu=0.3, rho=2.0)
    for pg in pgs:
        ops.element.LadrunoUP(
            pg=pg, material=mat,
            Kf=2.2e6, poro=0.4, rhoF=1.0, perm=(1e-4,) * ndm,
        )
    ops.system.UmfPack()      # legal general solver — D4 must not fire first
    ops.analysis.Static()
    return ops


# ---------------------------------------------------------------------------
# 1-2. one column: sealed is refused, one drained node is enough
# ---------------------------------------------------------------------------

class TestSingleColumnDatum:
    def test_sealed_column_refused_at_emit(self, tmp_path: Path) -> None:
        with apeGmsh(model_name="datum_sealed") as g:
            fem = _quad_column_fem(g)
            ops = _up_bridge(fem)
            ops.fix(pg="Base", dofs=(1, 1))    # u only — p free everywhere
            with pytest.raises(BridgeError) as ei:
                ops.tcl(str(tmp_path / "deck.tcl"))

        msg = str(ei.value)
        assert "NO fixed pressure DOF" in msg
        assert "1 of 1 u-p region(s)" in msg
        assert "slot 3" in msg
        # Names a real node of the offending region.
        tag = int(msg.split("LadrunoUP node ", 1)[1].split(":", 1)[0])
        assert tag in {int(n) for n in fem.nodes.ids}

    def test_one_drained_surface_node_is_enough(self, tmp_path: Path) -> None:
        with apeGmsh(model_name="datum_drained") as g:
            fem = _quad_column_fem(g)
            ops = _up_bridge(fem)
            ops.fix(pg="Base", dofs=(1, 1))
            top = sorted(int(n) for n in fem.nodes.select(pg="Top").ids)
            ops.fix(nodes=[top[0]], dofs=(0, 0, 1))    # ONE node's p pinned
            deck = (tmp_path / "deck.tcl")
            ops.tcl(str(deck))                          # must not raise

        text = deck.read_text(encoding="utf-8")
        assert f"fix {top[0]} 0 0 1" in text

    def test_drained_top_pg_passes(self, tmp_path: Path) -> None:
        """The ergonomic idiom — a whole drained surface pg."""
        with apeGmsh(model_name="datum_top_pg") as g:
            fem = _quad_column_fem(g)
            ops = _up_bridge(fem)
            ops.fix(pg="Base", dofs=(1, 1))
            ops.fix(pg="Top", dofs=(0, 0, 1))
            ops.tcl(str(tmp_path / "deck.tcl"))         # must not raise


# ---------------------------------------------------------------------------
# Taylor-Hood: mid-edge nodes neither ground nor shatter a region
# ---------------------------------------------------------------------------

class TestTaylorHoodRegion:
    def _tet10_box_fem(self, g: apeGmsh):
        g.model.geometry.add_box(0.0, 0.0, 0.0, 1.0, 1.0, 1.0, label="body")
        g.physical.add_volume("body", name="Soil")
        g.model.select(dim=2).on_plane(
            (0, 0, 1.0), (0, 0, 1), tol=1e-6,
        ).to_physical("Top")
        g.mesh.sizing.set_global_size(0.5)
        g.mesh.generation.generate(dim=3)
        g.mesh.generation.set_order(2)
        g.mesh.partitioning.renumber(base=1)
        return g.mesh.queries.get_fem_data(dim=3)

    @staticmethod
    def _vertices(fem) -> set[int]:
        """Tet10 vertex slots are 0-3; slots 4-9 are mid-edge (no p)."""
        out: set[int] = set()
        for grp in fem.elements.select(pg="Soil").groups():
            for conn in grp.connectivity:
                out.update(int(t) for t in conn[:4])
        return out

    def test_sealed_tet10_is_one_region_not_many(
        self, tmp_path: Path,
    ) -> None:
        """Dropping the mid-edge nodes must not shatter the pressure graph:
        a sealed tet10 box is ONE sealed region, not one per element."""
        with apeGmsh(model_name="datum_t10_sealed") as g:
            fem = self._tet10_box_fem(g)
            ops = _up_bridge(fem, ndm=3, ndf=4)
            with pytest.raises(BridgeError) as ei:
                ops.tcl(str(tmp_path / "deck.tcl"))

        msg = str(ei.value)
        assert "1 of 1 u-p region(s)" in msg
        assert "slot 4" in msg
        assert "Taylor-Hood" in msg      # the vertex-nodes hint

    def test_drained_vertex_nodes_ground_a_tet10_region(
        self, tmp_path: Path,
    ) -> None:
        """The idiom the error message names: the VERTEX nodes of the drained
        surface (a whole-pg mask would over-run the mid-edge nodes' ndf)."""
        with apeGmsh(model_name="datum_t10_drained") as g:
            fem = self._tet10_box_fem(g)
            top = {int(n) for n in fem.nodes.select(pg="Top").ids}
            top_vertices = sorted(top & self._vertices(fem))
            assert top_vertices and len(top_vertices) < len(top)
            ops = _up_bridge(fem, ndm=3, ndf=4)
            ops.fix(nodes=top_vertices, dofs=(0, 0, 0, 1))
            ops.tcl(str(tmp_path / "deck.tcl"))         # must not raise


# ---------------------------------------------------------------------------
# 3. two disconnected u-p regions each need their own datum
# ---------------------------------------------------------------------------

class TestTwoRegionsNeedTwoData:
    def test_neither_grounded_is_refused(self, tmp_path: Path) -> None:
        with apeGmsh(model_name="datum_two_none") as g:
            fem = _two_column_fem(g)
            ops = _up_bridge(fem, pgs=("SoilA", "SoilB"))
            with pytest.raises(BridgeError, match=r"2 of 2 u-p region\(s\)"):
                ops.tcl(str(tmp_path / "deck.tcl"))

    def test_grounding_one_column_leaves_the_other_refused(
        self, tmp_path: Path,
    ) -> None:
        with apeGmsh(model_name="datum_two_half") as g:
            fem = _two_column_fem(g)
            ops = _up_bridge(fem, pgs=("SoilA", "SoilB"))
            ops.fix(pg="TopA", dofs=(0, 0, 1))
            b_nodes = {int(n) for n in fem.nodes.select(pg="SoilB").ids}
            with pytest.raises(BridgeError) as ei:
                ops.tcl(str(tmp_path / "deck.tcl"))

        msg = str(ei.value)
        assert "1 of 2 u-p region(s)" in msg
        named = int(msg.split("LadrunoUP node ", 1)[1].split(":", 1)[0])
        assert named in b_nodes, (
            f"expected the UNgrounded column B to be named; got {named}"
        )

    def test_grounding_both_columns_passes(self, tmp_path: Path) -> None:
        with apeGmsh(model_name="datum_two_both") as g:
            fem = _two_column_fem(g)
            ops = _up_bridge(fem, pgs=("SoilA", "SoilB"))
            ops.fix(pg="TopA", dofs=(0, 0, 1))
            ops.fix(pg="TopB", dofs=(0, 0, 1))
            ops.tcl(str(tmp_path / "deck.tcl"))         # must not raise


# ---------------------------------------------------------------------------
# 4. blast radius — models the gate must leave alone
# ---------------------------------------------------------------------------

class TestGateScope:
    def test_pure_mechanical_model_untouched(self, tmp_path: Path) -> None:
        """No u-p element anywhere → the gate never looks at the mesh."""
        with apeGmsh(model_name="datum_mech") as g:
            fem = _quad_column_fem(g)
            ops = apeSees(fem)
            ops.model(ndm=2, ndf=2)
            ops.element.FourNodeQuad(
                pg="Soil",
                material=ops.nDMaterial.ElasticIsotropic(E=1e4, nu=0.3),
                thickness=1.0,
            )
            ops.fix(pg="Base", dofs=(1, 1))
            ops.system.UmfPack()
            ops.analysis.Static()
            ops.tcl(str(tmp_path / "deck.tcl"))         # must not raise

    def test_transient_sealed_column_allowed(self, tmp_path: Path) -> None:
        """Undrained dynamic loading is physically correct — the storage term
        puts a nonzero diagonal on the p rows, so the sealed tangent is
        regular.  The gate is Static-scoped and must not false-refuse it."""
        with apeGmsh(model_name="datum_transient") as g:
            fem = _quad_column_fem(g)
            ops = apeSees(fem)
            ops.model(ndm=2, ndf=3)
            ops.element.LadrunoUP(
                pg="Soil",
                material=ops.nDMaterial.ElasticIsotropic(E=1e4, nu=0.3, rho=2.0),
                Kf=2.2e6, poro=0.4, rhoF=1.0, perm=(1e-4, 1e-4),
            )
            ops.fix(pg="Base", dofs=(1, 1))
            ops.system.UmfPack()
            ops.analysis.Transient()
            ops.tcl(str(tmp_path / "deck.tcl"))         # must not raise

    def test_model_only_export_allowed(self, tmp_path: Path) -> None:
        """No analysis chain at all — a skeleton export never solves, the
        same F6 scope the D4 solver gate uses."""
        with apeGmsh(model_name="datum_skeleton") as g:
            fem = _quad_column_fem(g)
            ops = apeSees(fem)
            ops.model(ndm=2, ndf=3)
            ops.element.LadrunoUP(
                pg="Soil",
                material=ops.nDMaterial.ElasticIsotropic(E=1e4, nu=0.3, rho=2.0),
                Kf=2.2e6, poro=0.4, rhoF=1.0, perm=(1e-4, 1e-4),
            )
            ops.tcl(str(tmp_path / "deck.tcl"))         # must not raise
