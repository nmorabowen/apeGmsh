"""ADR 0085 acceptance — independently-meshed parts via the compose seam.

`Part` stays geometry-only (ADR 0085): when parts need different element
types AND different orders — impossible in one gmsh session because
``set_order`` is global — the supported route is one full session per part
→ ``g.save()`` → ``Assembly`` (`from_h5` host + `compose` + couples).

Locked here:

* three parts meshed independently as **hex8 / hex20 / tet10** (orders 1
  and 2 side by side), each saved to its own ``model.h5``;
* ``Assembly`` with the ADR 0085 ``kind="tie"`` couple assembles them and
  fails loud if an interface ties nothing;
* every part's physical groups survive composition **by name** (the
  PG-tag-collision fix ``7b63d67a`` is load-bearing here — before it,
  later modules silently destroyed earlier modules' PGs);
* the assembled FEMData reports mixed element types;
* LIVE (openseespy-gated): a two-block hex8+tet10 stack with ``nu = 0``
  tied by ``enforce="equation"`` reproduces the series closed form
  ``K = EA/L_total`` — the tie is exact, so the only tolerance is solver
  precision.

Hazards this test deliberately exercises (both from the Cerro Lindo
rung-4 production model): parts are extracted with
``get_fem_data(dim=None)`` (the tie resolver needs dim-2 element groups),
and PG survival is asserted explicitly (a tie against a missing PG is a
silent no-op on the raw compose path).
"""
from __future__ import annotations

from pathlib import Path

import pytest

from apeGmsh import apeGmsh

# Units: N, mm, MPa.
E = 200_000.0     # MPa
NU = 0.0          # keeps the axial stack exactly one-dimensional
SIDE = 10.0       # block plan dimensions (A = SIDE**2)
H = 10.0          # each block's height
TOL = 0.01        # face-selection half-thickness


def _faces_at_z(g, z: float) -> list[int]:
    """Surface tags whose plane is z = value (within the block bbox)."""
    lo = (-SIDE, -SIDE, z - TOL)
    hi = (2 * SIDE, 2 * SIDE, z + TOL)
    return g.model.select(None, dim=2).in_box(lo, hi).result().tags()


def _build_block(
    path: Path,
    *,
    name: str,
    z0: float,
    mesh: str,
    order: int,
    size: float,
    vol_pg: str,
    bot_pg: str | None,
    top_pg: str | None,
):
    """One block = one full session with its OWN mesh recipe and order."""
    with apeGmsh(model_name=name, save_to=str(path), overwrite=True) as g:
        g.model.geometry.add_box(0.0, 0.0, z0, SIDE, SIDE, H, label="v")
        g.physical.add_volume("v", name=vol_pg)
        if bot_pg is not None:
            g.physical.add_surface(_faces_at_z(g, z0), name=bot_pg)
        if top_pg is not None:
            g.physical.add_surface(_faces_at_z(g, z0 + H), name=top_pg)
        if mesh == "hex":
            g.mesh.recipe.structured(size=size, fallback="strict")
        else:
            g.mesh.recipe.unstructured(max_size=size)
        if order == 2:
            # serendipity: hex8 -> hex20, tet4 -> tet10
            g.mesh.generation.set_order(2, bubble=False)
        # dim=None, NOT dim=3: the tie resolver needs dim-2 element groups.
        fem = g.mesh.queries.get_fem_data(dim=None)
    return fem


def _solid_types(fem) -> dict[str, int]:
    """{type_name: order} for the dim-3 element types present."""
    return {t.name: t.order for t in fem.elements.types if t.dim == 3}


# --------------------------------------------------------------------------
# Structural — 3 parts, 3 element types, 2 orders, composed + tied
# --------------------------------------------------------------------------

def test_three_parts_mixed_type_and_order_compose_and_tie(tmp_path: Path):
    from apeGmsh.assembly import Assembly

    cover = tmp_path / "part_cover.h5"
    ribs = tmp_path / "part_ribs.h5"
    plate = tmp_path / "part_plate.h5"

    # Three INDEPENDENT meshes; two different orders. The single-session
    # route cannot do this — set_order is global per gmsh session.
    f_cover = _build_block(
        cover, name="cover", z0=0.0, mesh="hex", order=1, size=5.0,
        vol_pg="CoverVol", bot_pg="Base", top_pg="CoverTop",
    )
    f_ribs = _build_block(
        ribs, name="ribs", z0=H, mesh="hex", order=2, size=2.5,
        vol_pg="RibsVol", bot_pg="RibsBot", top_pg="RibsTop",
    )
    f_plate = _build_block(
        plate, name="plate", z0=2 * H, mesh="tet", order=2, size=6.0,
        vol_pg="PlateVol", bot_pg="PlateBot", top_pg="PlateTop",
    )

    # Each part really is what the recipe claims.
    assert _solid_types(f_cover) == {"hex8": 1}
    assert _solid_types(f_ribs) == {"hex20": 2}
    assert _solid_types(f_plate) == {"tet10": 2}

    # Assemble: host + two composed modules + two tie couples. The
    # interfaces are genuinely non-matching (2x2 quad4 vs 4x4 quad8 at
    # z=10; quad8 vs unstructured tri6 at z=20). materialize() raises
    # AssemblyError if a couple ties nothing.
    g = (
        Assembly("stack")
        .add("cover", str(cover))                 # host: bare PG names
        .add("rb", str(ribs))
        .add("pl", str(plate))
        .couple("cover", "rb", kind="tie", ports=("CoverTop", "RibsBot"),
                dofs=[1, 2, 3], enforce="equation")
        .couple("rb", "pl", kind="tie", ports=("RibsTop", "PlateBot"),
                dofs=[1, 2, 3], enforce="equation")
        .materialize()
    )
    fem = g.mesh.queries.get_fem_data(dim=None)

    # (i) EVERY part's PGs survive, by explicit name — host bare, composed
    # namespaced. Never trust the tie call alone: on the raw compose path
    # a tie against a lost PG is a silent no-op.
    expected = {
        "CoverVol", "Base", "CoverTop",
        "rb.RibsVol", "rb.RibsBot", "rb.RibsTop",
        "pl.PlateVol", "pl.PlateBot", "pl.PlateTop",
    }
    got = set(fem.inspect.physical_table()["name"].tolist())
    assert expected <= got, f"compose lost PGs: {sorted(expected - got)}"

    # (ii) the assembly reports mixed element types AND mixed orders.
    solids = _solid_types(fem)
    assert set(solids) == {"hex8", "hex20", "tet10"}
    assert set(solids.values()) == {1, 2}

    # (iii) both ties produced records (belt to materialize's braces).
    n_ties = len(list(fem.elements.constraints))
    assert n_ties > 0


def test_assembly_rejects_unknown_kind_still(tmp_path: Path):
    """Adding "tie" must not have loosened the couple-kind gate."""
    from apeGmsh.assembly import Assembly, AssemblyError

    asm = Assembly("x")
    with pytest.raises(AssemblyError, match="unsupported kind"):
        asm.couple("a", "b", kind="mortar", ports=("p", "q"))


# --------------------------------------------------------------------------
# LIVE — the tied interface transmits correctly: series closed form
# --------------------------------------------------------------------------

# ``enforce="equation"`` is fork-only at run time: stock openseespy takes
# the equationConstraint command but does not enforce it equivalently (this
# assertion read 71 % soft there before the live route was gated).
@pytest.mark.ladruno_fork
def test_tied_stack_matches_series_closed_form(tmp_path: Path):
    """Two stacked blocks, nu=0, equation tie: K = EA/L_total exactly.

    hex8 (order 1) under tet10 (order 2) — mixed order ACROSS the tie.
    With nu=0 the exact solution is uniform axial strain, representable
    by both meshes, and ``enforce="equation"`` is exact — so the
    assembled stiffness must match ``EA/L_total`` to solver precision
    (asserted at 0.1 %; measured error is orders tighter).
    """
    openseespy = pytest.importorskip("openseespy.opensees")
    from apeGmsh.assembly import Assembly
    from apeGmsh.opensees import apeSees
    from apeGmsh.opensees.emitter.live import LiveOpsEmitter

    cover = tmp_path / "part_cover.h5"
    plate = tmp_path / "part_plate.h5"
    _build_block(
        cover, name="cover", z0=0.0, mesh="hex", order=1, size=5.0,
        vol_pg="CoverVol", bot_pg="Base", top_pg="CoverTop",
    )
    _build_block(
        plate, name="plate", z0=H, mesh="tet", order=2, size=6.0,
        vol_pg="PlateVol", bot_pg="PlateBot", top_pg="PlateTop",
    )

    g = (
        Assembly("two_blocks")
        .add("cover", str(cover))
        .add("pl", str(plate))
        .couple("cover", "pl", kind="tie", ports=("CoverTop", "PlateBot"),
                dofs=[1, 2, 3], enforce="equation")
        .materialize()
    )
    fem = g.mesh.queries.get_fem_data(dim=None)

    delta = 0.01                                  # prescribed shortening, mm
    area = SIDE * SIDE
    k_exact = E * area / (2 * H)                  # series: EA/L / 2 per block

    ops = apeSees(fem)
    ops.model(ndm=3, ndf=3)
    steel = ops.nDMaterial.ElasticIsotropic(E=E, nu=NU)
    ops.element.stdBrick(pg="CoverVol", material=steel)
    ops.element.TenNodeTetrahedron(pg="pl.PlateVol", material=steel)
    ops.fix(pg="Base", dofs=(1, 1, 1))
    ts = ops.timeSeries.Linear()
    with ops.pattern.Plain(series=ts) as pat:
        pat.sp(pg="pl.PlateTop", dof=3, value=-delta)
    # equation ties need the Lagrange handler + an unsymmetric solver.
    ops.constraints.Lagrange()
    ops.numberer.Plain()
    ops.system.FullGeneral()
    ops.test.NormDispIncr(tol=1e-10, max_iter=10)
    ops.algorithm.Linear()
    ops.integrator.LoadControl(dlam=1.0)
    ops.analysis.Static()

    emitter = LiveOpsEmitter(wipe=True)
    ops.build().emit(emitter)
    assert emitter.analyze(steps=1) == 0

    openseespy.reactions()
    base_ids = [int(t) for t in fem.nodes.select(pg="Base").ids]
    r_z = sum(openseespy.nodeReaction(t, 3) for t in base_ids)
    k_measured = abs(r_z) / delta

    assert k_measured == pytest.approx(k_exact, rel=1e-3), (
        f"tied stack K = {k_measured:.6g} vs closed form {k_exact:.6g} "
        f"({(k_measured / k_exact - 1) * 100:+.3f} %)"
    )
    openseespy.wipe()
