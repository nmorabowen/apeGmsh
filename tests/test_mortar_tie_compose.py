"""ADR 0086 acceptance — mortar mesh-ties on COMPOSED assemblies.

The capability this locks: an order-mismatched interface (hex20 faces
tied onto hex8 faces) expressed on an assembly built with
``from_h5`` + ``compose`` — impossible before ADR 0086 because mortar
lived only in the (build-phase, gmsh-needing, tri3/quad4-only) contact
subsystem, and ``set_order`` being session-global forces mixed-order
models through compose.

Also locks ADR 0086 D4: ``g.constraints.contact`` on a chain-phase
session raises ``ChainPhaseError`` instead of the historic silent drop.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from apeGmsh import apeGmsh

from tests.test_meshable_part_route import _build_block, E, NU, SIDE, H


# --------------------------------------------------------------------------
# Def-level validation (no gmsh)
# --------------------------------------------------------------------------

def test_tiedef_method_validation():
    from apeGmsh._kernel.defs.constraints import TieDef

    with pytest.raises(ValueError, match="method must be"):
        TieDef(master_label="m", slave_label="s", method="mortor")
    with pytest.raises(ValueError, match="requires enforce='equation'"):
        TieDef(master_label="m", slave_label="s", method="mortar")
    with pytest.raises(ValueError, match="only valid with method='mortar'"):
        TieDef(master_label="m", slave_label="s",
               outward=(0.0, 0.0, 1.0))
    d = TieDef(master_label="m", slave_label="s",
               method="mortar", enforce="equation",
               outward=(0.0, 0.0, 1.0))
    assert d.method == "mortar"


# --------------------------------------------------------------------------
# Composed assembly: hex20 slave faces on hex8 master faces (chain phase)
# --------------------------------------------------------------------------

def _build_two_block_parts(tmp_path: Path):
    cover = tmp_path / "part_cover.h5"          # hex8, order 1
    ribs = tmp_path / "part_ribs.h5"            # hex20, order 2
    _build_block(
        cover, name="cover", z0=0.0, mesh="hex", order=1, size=5.0,
        vol_pg="CoverVol", bot_pg="Base", top_pg="CoverTop",
    )
    # size 10/3 (3x3 facets) against the cover's 2x2: NON-NESTED — slave
    # facets straddle master face boundaries. Deliberate: on a nested
    # grid (2.5 vs 5) the dual projection provably collapses to nodal
    # evaluation (bilinear ⊂ quad8 span ⇒ mortar ≡ collocation), so a
    # nested rig cannot discriminate the mortar path from collocation.
    _build_block(
        ribs, name="ribs", z0=H, mesh="hex", order=2, size=10.0 / 3.0,
        vol_pg="RibsVol", bot_pg="RibsBot", top_pg="RibsTop",
    )
    return cover, ribs


def test_mortar_tie_resolves_on_composed_assembly(tmp_path: Path):
    from apeGmsh.assembly import Assembly

    cover, ribs = _build_two_block_parts(tmp_path)
    g = (
        Assembly("stack")
        .add("cover", str(cover))
        .add("rb", str(ribs))
        .couple("cover", "rb", kind="tie", ports=("CoverTop", "RibsBot"),
                dofs=[1, 2, 3], method="mortar", enforce="equation")
        .materialize()
    )
    fem = g.mesh.queries.get_fem_data(dim=None)

    # Acceptance #1: never trust the call — a non-zero record count.
    recs = [r for r in fem.elements.constraints
            if getattr(r, "master_nodes", None)]
    assert len(recs) > 0
    # One row per slave node of the quad8 interface (5x5 grid of hex20
    # corner nodes + midsides on a 4x4-facet face = 41 unique nodes...
    # assert the exact count structurally instead: every record is an
    # equation-route mortar row with partition-of-unity weights.
    for r in recs:
        assert r.enforce == "equation"
        assert float(np.sum(r.weights)) == pytest.approx(1.0, abs=1e-6)
    # Interface node count: the RibsBot 3x3 quad8 faces carry a 7x7
    # grid minus the 3x3 face centres = 40 unique slave nodes.
    assert len(recs) == 40
    # Mortar discriminator: collocation couples each slave node to ONE
    # master face (<= 4 masters); mortar rows integrate over the slave
    # node's facet support and couple across face boundaries. Guards
    # against a silent fall-through to the collocation path.
    assert max(len(r.master_nodes) for r in recs) > 4

    # The rows must survive the neutral-zone round-trip (assemble →
    # save → reload is the production workflow).
    from apeGmsh.mesh.FEMData import FEMData
    out = tmp_path / "assembled.h5"
    g.save(str(out))
    fem2 = FEMData.from_h5(str(out))
    recs2 = sorted(
        (r for r in fem2.elements.constraints
         if getattr(r, "master_nodes", None)),
        key=lambda r: r.slave_node,
    )
    assert len(recs2) == len(recs)
    by_slave = {r.slave_node: r for r in recs}
    for r2 in recs2:
        r1 = by_slave[r2.slave_node]
        assert list(r2.master_nodes) == list(r1.master_nodes)
        assert np.allclose(r2.weights, r1.weights, atol=1e-12)
        assert r2.enforce == "equation"


def test_contact_on_chain_phase_session_raises(tmp_path: Path):
    """ADR 0086 D4 — the silent drop becomes a loud ChainPhaseError."""
    from apeGmsh.core._compose_errors import ChainPhaseError

    cover, ribs = _build_two_block_parts(tmp_path)
    g = apeGmsh.from_h5(str(cover))
    g.compose(str(ribs), label="rb")
    with pytest.raises(ChainPhaseError, match="mortar"):
        g.constraints.contact("CoverTop", "rb.RibsBot",
                              formulation="mortar", tie=True,
                              outward=(0.0, 0.0, 1.0))


# --------------------------------------------------------------------------
# LIVE — closed form + order-swap symmetry (fork build: LadrunoBrick20)
# --------------------------------------------------------------------------

def _solve_stack_K(tmp_path, *, swap_master_slave: bool) -> float:
    """Compose the hex8+hex20 stack, mortar-tie, solve, return K."""
    import openseespy.opensees as openseespy
    from apeGmsh.assembly import Assembly
    from apeGmsh.mesh.FEMData import FEMData
    from apeGmsh.opensees import apeSees
    from apeGmsh.opensees.emitter.live import LiveOpsEmitter

    sub = tmp_path / ("swap" if swap_master_slave else "fwd")
    sub.mkdir()
    cover, ribs = _build_two_block_parts(sub)

    asm = (
        Assembly("two_blocks")
        .add("cover", str(cover))
        .add("rb", str(ribs))
    )
    if swap_master_slave:
        # hex20 quad8 faces as MASTER, hex8 quad4 faces as SLAVE
        asm.couple("rb", "cover", kind="tie", ports=("RibsBot", "CoverTop"),
                   dofs=[1, 2, 3], method="mortar", enforce="equation")
    else:
        # hex8 quad4 faces as MASTER, hex20 quad8 faces as SLAVE —
        # the configuration collocation over-constrains and mortar fixes
        asm.couple("cover", "rb", kind="tie", ports=("CoverTop", "RibsBot"),
                   dofs=[1, 2, 3], method="mortar", enforce="equation")
    g = asm.materialize()
    fem = g.mesh.queries.get_fem_data(dim=None)

    delta = 0.01
    ops = apeSees(fem)
    ops.model(ndm=3, ndf=3)
    steel = ops.nDMaterial.ElasticIsotropic(E=E, nu=NU)
    ops.element.stdBrick(pg="CoverVol", material=steel)
    ops.element.LadrunoBrick20(pg="rb.RibsVol", material=steel)
    ops.fix(pg="Base", dofs=(1, 1, 1))
    ts = ops.timeSeries.Linear()
    with ops.pattern.Plain(series=ts) as pat:
        pat.sp(pg="rb.RibsTop", dof=3, value=-delta)
    ops.constraints.Lagrange()
    ops.numberer.Plain()
    ops.system.FullGeneral()
    ops.test.NormDispIncr(tol=1e-10, max_iter=10)
    ops.algorithm.Linear()
    ops.integrator.LoadControl(dlam=1.0)
    ops.analysis.Static()

    emitter = LiveOpsEmitter(wipe=True)
    ops.build().emit(emitter)
    ret = emitter.analyze(steps=1)
    assert ret == 0, f"analyze returned {ret}"

    openseespy.reactions()
    base_ids = [int(t) for t in fem.nodes.select(pg="Base").ids]
    r_z = sum(openseespy.nodeReaction(t, 3) for t in base_ids)
    openseespy.wipe()
    return abs(r_z) / delta


def test_mortar_tied_stack_closed_form_and_order_swap(tmp_path: Path):
    """Acceptance #2 + #3: K = EA/L/2 to 0.1 %, and master/slave swap
    gives the same K — the symmetry collocation does not have, which is
    the property the mortar tie is buying."""
    openseespy = pytest.importorskip("openseespy.opensees")
    if not hasattr(openseespy, "ladrunoProjectionTieForce"):
        pytest.skip("needs the Ladruno fork build (LadrunoBrick20)")

    k_exact = E * SIDE * SIDE / (2 * H)

    k_fwd = _solve_stack_K(tmp_path, swap_master_slave=False)
    assert k_fwd == pytest.approx(k_exact, rel=1e-3), (
        f"mortar stack K = {k_fwd:.6g} vs closed form {k_exact:.6g} "
        f"({(k_fwd / k_exact - 1) * 100:+.3f} %)"
    )

    k_swap = _solve_stack_K(tmp_path, swap_master_slave=True)
    assert k_swap == pytest.approx(k_exact, rel=1e-3)
    assert k_swap == pytest.approx(k_fwd, rel=1e-3), (
        f"order-swap asymmetry: {k_fwd:.6g} vs {k_swap:.6g}"
    )
