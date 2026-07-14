"""ADR 0074 — LadrunoUP end-to-end emission over real Gmsh meshes.

Mesh-only lanes (gmsh required, NO fork build — the element is never run):

* Q4 column deck: ``element LadrunoUP`` lines with the full flag grammar,
  no ``-pOrder`` on the equal-order shape, ``-ndf`` elision under the
  recommended ``ndf = ndm+1`` envelope.
* D4 solver gate at ``ops.tcl(...)``: missing system and ProfileSPD both
  refuse; UmfPack passes.
* tri6 Taylor–Hood deck: automatic ``-pOrder linear``; per-slot ndf
  emission under BOTH envelope choices (``ndf=3`` → ``-ndf 2`` tokens on
  mid-edge nodes only; ``ndf=2`` → ``-ndf 3`` on vertices only) plus the
  equal-order builder bracket under the non-matching envelope.
* tet10 3D lane: ``-ndf 3`` mid-edge tokens under the ``ndf=4`` envelope.
* D3: a disk meshed to tri6 has curved boundary mid-edge nodes → build
  refuses with the straight-side BridgeError (mesh context + hint).
* G3 ergonomics: a fix mask longer than a TH mid-edge node's resolved
  ndf fails loud; the leading-u mask passes.

Fork-gated lanes (``live`` marker + runtime skip when the in-process
openseespy lacks the fork / the LadrunoUP element):

* mini Terzaghi Q4 column — emitted deck runs (rc=0) and the pore
  pressure at the sealed base decays monotonically after the load step
  (the loose sanity gate; the tight series gates live in the fork repo).
"""
from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("gmsh")

from apeGmsh import apeGmsh  # noqa: E402
from apeGmsh.opensees import apeSees  # noqa: E402
from apeGmsh.opensees._internal.build import BridgeError  # noqa: E402


# ---------------------------------------------------------------------------
# Mesh helpers
# ---------------------------------------------------------------------------

def _quad_column_fem(g: apeGmsh, *, h: float = 4.0, size: float = 1.0):
    """A structured 1 x h quad column in the xy plane: pg 'Soil' (surface),
    'Base' (y=0 line), 'Top' (y=h line)."""
    g.model.geometry.add_rectangle(0.0, 0.0, 0.0, 1.0, h, label="soil")
    g.physical.add_surface("soil", name="Soil")
    g.model.select(dim=1).on_plane((0, 0, 0), (0, 1, 0), tol=1e-6).to_physical("Base")
    g.model.select(dim=1).on_plane((0, h, 0), (0, 1, 0), tol=1e-6).to_physical("Top")
    g.mesh.structured.set_recombine("soil", dim=2)
    g.mesh.sizing.set_global_size(size)
    g.mesh.generation.generate(2)
    g.mesh.structured.recombine()
    g.mesh.partitioning.renumber(base=1)
    return g.mesh.queries.get_fem_data(dim=2)


def _tri6_column_fem(g: apeGmsh, *, h: float = 2.0, size: float = 1.0):
    """An unstructured tri6 column (order-2 triangles, straight sides)."""
    g.model.geometry.add_rectangle(0.0, 0.0, 0.0, 1.0, h, label="soil")
    g.physical.add_surface("soil", name="Soil")
    g.model.select(dim=1).on_plane((0, 0, 0), (0, 1, 0), tol=1e-6).to_physical("Base")
    g.mesh.sizing.set_global_size(size)
    g.mesh.generation.generate(2)
    g.mesh.generation.set_order(2)   # quadratic -> tri6 (MSH type 9)
    g.mesh.partitioning.renumber(base=1)
    return g.mesh.queries.get_fem_data(dim=2)


def _tet10_box_fem(g: apeGmsh):
    g.model.geometry.add_box(0.0, 0.0, 0.0, 1.0, 1.0, 1.0, label="body")
    g.physical.add_volume("body", name="Soil")
    g.mesh.sizing.set_global_size(1.0)
    g.mesh.generation.generate(dim=3)
    g.mesh.generation.set_order(2)   # quadratic -> tet10 (MSH type 11)
    g.mesh.partitioning.renumber(base=1)
    return g.mesh.queries.get_fem_data(dim=3)


def _up_kwargs(dim: int = 2) -> dict:
    return dict(
        Kf=2.2e6, poro=0.4, rhoF=1.0, perm=(1e-4,) * dim,
    )


def _bridge(fem, *, ndm: int, ndf: int, dim: int = 2, **up_kw) -> apeSees:
    ops = apeSees(fem)
    ops.model(ndm=ndm, ndf=ndf)
    mat = ops.nDMaterial.ElasticIsotropic(E=1e4, nu=0.3, rho=2.0)
    kw = _up_kwargs(dim)
    kw.update(up_kw)
    ops.element.LadrunoUP(pg="Soil", material=mat, **kw)
    return ops


def _deck(ops: apeSees, tmp_path: Path, name: str = "deck.tcl") -> str:
    out = tmp_path / name
    ops.tcl(str(out))
    return out.read_text(encoding="utf-8")


def _node_lines(deck: str) -> list[str]:
    return [
        ln.strip() for ln in deck.splitlines()
        if ln.strip().startswith("node ")
    ]


def _element_lines(deck: str) -> list[str]:
    return [
        ln.strip() for ln in deck.splitlines()
        if ln.strip().startswith("element LadrunoUP")
    ]


# ---------------------------------------------------------------------------
# Q4 equal-order lane
# ---------------------------------------------------------------------------

class TestQ4Deck:
    def test_deck_grammar_and_ndf_elision(self, tmp_path: Path) -> None:
        with apeGmsh(model_name="up_q4") as g:
            fem = _quad_column_fem(g)
            ops = _bridge(fem, ndm=2, ndf=3)
            ops.system.UmfPack()
            deck = _deck(ops, tmp_path)

        ele = _element_lines(deck)
        assert ele, deck
        for ln in ele:
            assert "-Kf 2200000.0 -poro 0.4 -rhoF 1.0" in ln
            assert "-perm 0.0001 0.0001" in ln
            assert "-pOrder" not in ln     # equal-order shape
            assert "-stab" not in ln       # pass-through: parser owns default
        # Envelope ndf = ndm+1 -> every node matches -> no -ndf tokens.
        assert all("-ndf" not in ln for ln in _node_lines(deck))

    def test_missing_system_refused_when_solve_bearing(
        self, tmp_path: Path,
    ) -> None:
        """A deck that WILL solve (an analysis chain is registered) but
        declares no system → refused (the ProfileSPD-default footgun)."""
        with apeGmsh(model_name="up_q4_nosys") as g:
            fem = _quad_column_fem(g)
            ops = _bridge(fem, ndm=2, ndf=3)
            ops.analysis.Static()   # solve-bearing, but no system declared
            with pytest.raises(BridgeError, match="no linear system"):
                ops.tcl(str(tmp_path / "deck.tcl"))

    def test_missing_system_allowed_for_model_only_export(
        self, tmp_path: Path,
    ) -> None:
        """A model-only skeleton export (no analysis chain) never solves, so a
        missing system is NOT refused — the user completes the deck later
        (ADR 0074 F6: archival / eigen / skeleton emits are not footguns)."""
        with apeGmsh(model_name="up_q4_skeleton") as g:
            fem = _quad_column_fem(g)
            ops = _bridge(fem, ndm=2, ndf=3)
            ops.tcl(str(tmp_path / "deck.tcl"))   # must not raise

    def test_symmetric_system_refused(self, tmp_path: Path) -> None:
        """A DECLARED symmetric system is refused even for a model-only
        export (declaring the wrong solver is unambiguously a mistake)."""
        with apeGmsh(model_name="up_q4_spd") as g:
            fem = _quad_column_fem(g)
            ops = _bridge(fem, ndm=2, ndf=3)
            ops.system.ProfileSPD()
            with pytest.raises(BridgeError, match="UNSYMMETRIC"):
                ops.tcl(str(tmp_path / "deck.tcl"))

    def test_builder_bracket_under_foreign_envelope(
        self, tmp_path: Path,
    ) -> None:
        """Envelope ndf=2: nodes carry -ndf 3, and the equal-order builder
        gate (parser demands ndf = ndm+1) gets the model bracket."""
        with apeGmsh(model_name="up_q4_env2") as g:
            fem = _quad_column_fem(g)
            ops = _bridge(fem, ndm=2, ndf=2)
            ops.system.UmfPack()
            deck = _deck(ops, tmp_path)

        node_lines = _node_lines(deck)
        assert node_lines and all("-ndf 3" in ln for ln in node_lines)
        lines = [ln.strip() for ln in deck.splitlines()]
        first_ele = next(
            i for i, ln in enumerate(lines)
            if ln.startswith("element LadrunoUP")
        )
        bracket = [
            i for i, ln in enumerate(lines)
            if ln == "model BasicBuilder -ndm 2 -ndf 3"
        ]
        restore = [
            i for i, ln in enumerate(lines[first_ele:], start=first_ele)
            if ln == "model BasicBuilder -ndm 2 -ndf 2"
        ]
        assert any(i < first_ele for i in bracket), (
            "expected a `model -ndf 3` bracket before the LadrunoUP block"
        )
        assert restore, "expected the envelope restore after the block"


# ---------------------------------------------------------------------------
# Taylor–Hood lanes (tri6 / tet10)
# ---------------------------------------------------------------------------

class TestTaylorHoodDeck:
    def test_tri6_porder_and_midedge_ndf_under_vertex_envelope(
        self, tmp_path: Path,
    ) -> None:
        with apeGmsh(model_name="up_t6_env3") as g:
            fem = _tri6_column_fem(g)
            n_vertex, n_mid = _th_counts(fem, npe=6, n_corner=3)
            ops = _bridge(fem, ndm=2, ndf=3)
            ops.system.UmfPack()
            deck = _deck(ops, tmp_path)

        for ln in _element_lines(deck):
            assert ln.rstrip().endswith("-pOrder linear"), ln
        tokens = [ln for ln in _node_lines(deck) if "-ndf 2" in ln]
        assert len(tokens) == n_mid, (
            f"expected {n_mid} mid-edge -ndf 2 overrides, got {len(tokens)}"
        )
        assert not any("-ndf 3" in ln for ln in _node_lines(deck))

    def test_tri6_vertex_ndf_under_midedge_envelope(
        self, tmp_path: Path,
    ) -> None:
        with apeGmsh(model_name="up_t6_env2") as g:
            fem = _tri6_column_fem(g)
            n_vertex, n_mid = _th_counts(fem, npe=6, n_corner=3)
            ops = _bridge(fem, ndm=2, ndf=2)
            ops.system.UmfPack()
            deck = _deck(ops, tmp_path)

        tokens = [ln for ln in _node_lines(deck) if "-ndf 3" in ln]
        assert len(tokens) == n_vertex
        assert not any("-ndf 2" in ln for ln in _node_lines(deck))

    def test_tet10_midedge_ndf(self, tmp_path: Path) -> None:
        with apeGmsh(model_name="up_t10") as g:
            fem = _tet10_box_fem(g)
            n_vertex, n_mid = _th_counts(fem, npe=10, n_corner=4)
            ops = _bridge(fem, ndm=3, ndf=4, dim=3)
            ops.system.UmfPack()
            deck = _deck(ops, tmp_path)

        for ln in _element_lines(deck):
            assert ln.rstrip().endswith("-pOrder linear"), ln
        tokens = [ln for ln in _node_lines(deck) if "-ndf 3" in ln]
        assert len(tokens) == n_mid

    def test_fix_mask_longer_than_midedge_ndf_fails_loud(
        self, tmp_path: Path,
    ) -> None:
        """The 'Base' pg holds vertices (ndf 3) AND mid-edge nodes (ndf 2):
        a 3-slot drained mask overruns the mid-edge nodes and G3 refuses;
        the leading-u 2-slot mask passes (shorter masks fix leading DOFs)."""
        with apeGmsh(model_name="up_t6_fix") as g:
            fem = _tri6_column_fem(g)
            ops = _bridge(fem, ndm=2, ndf=3)
            ops.system.UmfPack()
            ops.fix(pg="Base", dofs=(1, 1, 1))
            with pytest.raises(BridgeError, match="ndf is only 2"):
                ops.tcl(str(tmp_path / "deck.tcl"))

        with apeGmsh(model_name="up_t6_fix2") as g:
            fem = _tri6_column_fem(g)
            ops = _bridge(fem, ndm=2, ndf=3)
            ops.system.UmfPack()
            ops.fix(pg="Base", dofs=(1, 1))
            _deck(ops, tmp_path)   # no raise

    def test_stab_on_th_pg_refused_at_build(self, tmp_path: Path) -> None:
        with apeGmsh(model_name="up_t6_stab") as g:
            fem = _tri6_column_fem(g)
            ops = _bridge(fem, ndm=2, ndf=3, stab="auto")
            ops.system.UmfPack()
            with pytest.raises(BridgeError, match="inf-sup stable"):
                ops.tcl(str(tmp_path / "deck.tcl"))


def _th_counts(fem, *, npe: int, n_corner: int) -> tuple[int, int]:
    """(n_vertex_nodes, n_midedge_nodes) of a single-type TH mesh."""
    conn = fem.elements.connectivity
    assert conn.shape[1] == npe, f"expected {npe}-node cells, got {conn.shape}"
    vertex = {int(n) for row in conn for n in row[:n_corner]}
    mid = {int(n) for row in conn for n in row[n_corner:]}
    assert not (vertex & mid), "conforming mesh: mid-edge <-> mid-edge only"
    return len(vertex), len(mid)


# ---------------------------------------------------------------------------
# D3 — curved TH mesh refused at build
# ---------------------------------------------------------------------------

class TestStraightSideGate:
    def test_curved_tri6_disk_refused(self, tmp_path: Path) -> None:
        """Order-2 meshing of a disk puts boundary mid-edge nodes ON the
        circle (off the chord midpoint) — the fork would deactivate those
        elements at setDomain; apeGmsh refuses at build with mesh context."""
        with apeGmsh(model_name="up_t6_disk") as g:
            G = g.model.geometry
            circle = G.add_circle(0.0, 0.0, 0.0, 1.0)
            loop = G.add_curve_loop([circle])
            disk = G.add_plane_surface([loop])
            g.model.sync()
            g.physical.add(2, [disk], name="Soil")
            g.mesh.sizing.set_global_size(0.6)
            g.mesh.generation.generate(2)
            g.mesh.generation.set_order(2)
            g.mesh.partitioning.renumber(base=1)
            fem = g.mesh.queries.get_fem_data(dim=2)

            ops = _bridge(fem, ndm=2, ndf=3)
            ops.system.UmfPack()
            with pytest.raises(BridgeError, match="straight-side") as ei:
                ops.tcl(str(tmp_path / "deck.tcl"))
            assert "high-order optimization" in str(ei.value)


# ---------------------------------------------------------------------------
# Fork-gated live lane — mini Terzaghi Q4 column
# ---------------------------------------------------------------------------

def _skip_unless_fork_up() -> None:
    try:
        from apeGmsh.opensees._target import probe_live_capabilities

        caps = probe_live_capabilities()
    except Exception as e:  # pragma: no cover - no openseespy
        pytest.skip(f"openseespy unavailable: {e}")
    if not caps.has_ladruno_up:
        pytest.skip("in-process openseespy is not a LadrunoUP-bearing fork")


@pytest.mark.live
def test_terzaghi_q4_column_runs_and_p_decays() -> None:
    """Loose Terzaghi sanity gate (the tight series gates live in the fork
    repo): a drained-top Q4 column under a sudden top load runs rc=0 and
    the base pore pressure decays monotonically as it consolidates."""
    pytest.importorskip("openseespy.opensees")
    _skip_unless_fork_up()
    from apeGmsh.opensees.emitter.live import LiveOpsEmitter

    with apeGmsh(model_name="up_terzaghi") as g:
        fem = _quad_column_fem(g, h=4.0, size=0.5)
        ops = apeSees(fem)
        ops.model(ndm=2, ndf=3)
        mat = ops.nDMaterial.ElasticIsotropic(E=1e4, nu=0.0, rho=2.0)
        ops.element.LadrunoUP(
            pg="Soil", material=mat,
            Kf=2.2e6, poro=0.4, rhoF=1.0, perm=(1e-4, 1e-4),
            dynSeepage="off",
        )
        # u fixed at the base; sides roll (ux only); p free everywhere but
        # the drained top (slot 3 fixed = p = 0).
        ops.fix(pg="Base", dofs=(1, 1))
        ops.fix(pg="Top", dofs=(0, 0, 1))

        ts = ops.timeSeries.Constant()
        top_nodes = [
            int(n) for n in fem.nodes.select(pg="Top").ids
        ]
        with ops.pattern.Plain(series=ts) as p:
            for n in top_nodes:
                # Raw DOF-space vector; the p slot (DOF 3) gets no load.
                p.load(node=n, forces=(0.0, -1.0 / len(top_nodes)))

        ops.constraints.Transformation()
        ops.numberer.RCM()
        ops.system.UmfPack()
        ops.test.NormDispIncr(tol=1e-8, max_iter=20)
        ops.algorithm.Newton()
        # ZS84 production Newmark set: gamma >= 0.6 damps the parasitic
        # p-row root (gamma = 0.5 measurably diverges).
        ops.integrator.Newmark(gamma=0.6, beta=0.3025)
        ops.analysis.Transient()

        emitter = LiveOpsEmitter(wipe=True)
        bm = ops.build()
        try:
            bm.emit(emitter)
        except RuntimeError as e:  # pragma: no cover - stock build
            if "Ladruno fork" in str(e):
                pytest.skip(str(e))
            raise

        live = emitter.ops
        # Base centre node: y = 0, x = 0.5.
        base_node = min(
            (int(n) for n in fem.nodes.select(pg="Base").ids),
            key=lambda t: abs(live.nodeCoord(t)[0] - 0.5),
        )
        dt = 0.05
        p_hist: list[float] = []
        for _ in range(20):
            assert emitter.analyze(steps=1, dt=dt) == 0
            p_hist.append(float(live.nodeDisp(base_node, 3)))

    p_peak = max(p_hist[:3])
    assert p_peak > 0.0, f"no undrained pressure response: {p_hist[:5]}"
    late = p_hist[3:]
    assert all(b <= a + 1e-12 for a, b in zip(late, late[1:])), (
        f"base pore pressure is not decaying monotonically: {p_hist}"
    )
    assert late[-1] < p_peak, "no consolidation decay over the run"


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(pytest.main([__file__, "-v"]))
