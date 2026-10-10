"""ADR 0085 acceptance — independently-meshed parts via the compose seam.

`Part` stays geometry-only (ADR 0085): when parts need different element
types AND different orders — impossible in one gmsh session because
``set_order`` is global — the supported route is one full session per part
→ ``ops.h5()`` → ``Assembly`` (ADR 0117: ``instance`` per part + ``tie``
+ ``bridge``).

Locked here:

* three parts meshed independently as **hex8 / hex20 / tet10** (orders 1
  and 2 side by side), each saved to its own ``model.h5``;
* ``Assembly.tie(..., enforce="equation")`` assembles them and
  ``bridge()`` fails loud if an interface ties nothing;
* every part's physical groups survive composition **by name**, each
  under its instance label (the PG-tag-collision fix ``7b63d67a`` is
  load-bearing here — before it, later modules silently destroyed
  earlier modules' PGs);
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
    """One block = one full session with its OWN mesh recipe and order,
    written to ``path`` as an instanceable source (``/opensees`` model
    ndm=3, ndf=3; the elements are declared on the assembly's bridge)."""
    from apeGmsh.opensees import apeSees

    with apeGmsh(model_name=name) as g:
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
    ops = apeSees(fem)
    ops.model(ndm=3, ndf=3)
    ops.h5(str(path))
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

    # Assemble: three instances + two ties. The interfaces are genuinely
    # non-matching (2x2 quad4 vs 4x4 quad8 at z=10; quad8 vs unstructured
    # tri6 at z=20). bridge() raises AssemblyError if a tie ties nothing.
    ops = (
        Assembly("stack")
        .instance("cover", cover)
        .instance("rb", ribs)
        .instance("pl", plate)
        .tie("cover.CoverTop", "rb.RibsBot", dofs=[1, 2, 3],
             enforce="equation")
        .tie("rb.RibsTop", "pl.PlateBot", dofs=[1, 2, 3],
             enforce="equation")
        .bridge(ndm=3, ndf=3)
    )
    fem = ops.fem

    # (i) EVERY part's PGs survive, by explicit name, each namespaced
    # under its instance. Never trust the tie call alone: a tie against a
    # lost PG would resolve nothing.
    expected = {
        "cover.CoverVol", "cover.Base", "cover.CoverTop",
        "rb.RibsVol", "rb.RibsBot", "rb.RibsTop",
        "pl.PlateVol", "pl.PlateBot", "pl.PlateTop",
    }
    got = set(fem.inspect.physical_table()["name"].tolist())
    assert expected <= got, f"compose lost PGs: {sorted(expected - got)}"

    # (ii) the assembly reports mixed element types AND mixed orders.
    solids = _solid_types(fem)
    assert set(solids) == {"hex8", "hex20", "tet10"}
    assert set(solids.values()) == {1, 2}

    # (iii) both ties produced records (belt to bridge's braces).
    n_ties = len(list(fem.elements.constraints))
    assert n_ties > 0


# --------------------------------------------------------------------------
# LIVE — the tied interface transmits correctly: series closed form
# --------------------------------------------------------------------------
#
# Two blocks of equal height under nu=0: the exact solution is uniform axial
# strain, which every mesh here represents, so K = EA/L_total is an oracle
# wherever the tie passes the patch test — a nested interface, or
# ``method="mortar"``. Node-to-surface collocation on an unstructured
# interface does not quite (it reproduces uniform displacement, not uniform
# traction transfer) and reads a few tenths of a percent soft, identically
# on every engine.
#
# The engine question and the element question are separated on purpose.
# Stock ``TenNodeTetrahedron`` is 6x too soft (upstream ``shp3d`` applies the
# tetrahedral 1/6 twice; fork PR #520 fixed it), and one 6x-soft block in
# series reads (1/(1+6)) / (1/(1+1)) = 2/7 of the closed form: the "71 %
# soft" once blamed on the stock tie. With a plate that is right on every
# build, stock and fork agree to nine digits.
#
# Each tied model runs in a FRESH interpreter: stock ``wipe()`` keeps
# ``equationConstraint`` rows (upstream ``Domain::clearAll()`` omits them),
# so the live emitter refuses a second model in a stock process that ran
# one — and a test process that ran one could not run any later live test.

DELTA = 0.01                          # prescribed shortening, mm
K_EXACT = E * SIDE * SIDE / (2 * H)   # series: EA/L / 2 per block

#: plate -> (mesh, order, size, OpenSees element, plate element count).
#: The cover is always hex8 at size 5 (2x2x2), so every interface is
#: non-matching: tri6 / tri3 faces on quad4, 4x4 or 3x3 quad4 on 2x2.
_PLATES = {
    "tet10": ("tet", 2, 6.0, "TenNodeTetrahedron", None),
    "tet4": ("tet", 1, 6.0, "FourNodeTetrahedron", None),
    "hex8-nested": ("hex", 1, 2.5, "stdBrick", 64),
    "hex8": ("hex", 1, 3.4, "stdBrick", 27),
}


def _live_ops():
    """The resolved OpenSees backend module, or skip when there is none."""
    from apeGmsh.opensees.emitter.live import _get_ops
    try:
        return _get_ops()
    except ImportError as e:
        pytest.skip(f"no OpenSees backend: {e}")


def _require_equation_constraint() -> None:
    # openseespy 3.7.1.2 (the last wheel for Python < 3.12 on Linux)
    # predates upstream's equationConstraint (2025-05-10, OpenSees 3.8.0).
    if not hasattr(_live_ops(), "equationConstraint"):
        pytest.skip("this OpenSees build predates equationConstraint "
                    "(upstream 2025-05-10, openseespy >= 3.8.0)")


def _tet10_volume_fixed():
    from apeGmsh.opensees.emitter.live import _tet10_volume_fixed
    return _tet10_volume_fixed(_live_ops())


def _tied_stack(workdir: Path, plate: str, method: str) -> dict:
    """Solve the stack IN THIS PROCESS; K plus the worst tie-row residual."""
    import numpy as np

    from apeGmsh.assembly import Assembly
    from apeGmsh.opensees.emitter.live import LiveOpsEmitter

    mesh, order, size, element, n_plate = _PLATES[plate]
    cover_h5 = workdir / "part_cover.h5"
    plate_h5 = workdir / "part_plate.h5"
    _build_block(
        cover_h5, name="cover", z0=0.0, mesh="hex", order=1, size=5.0,
        vol_pg="CoverVol", bot_pg="Base", top_pg="CoverTop",
    )
    _build_block(
        plate_h5, name="plate", z0=H, mesh=mesh, order=order, size=size,
        vol_pg="PlateVol", bot_pg="PlateBot", top_pg="PlateTop",
    )
    ops = (
        Assembly("two_blocks")
        .instance("cover", cover_h5)
        .instance("pl", plate_h5)
        .tie("cover.CoverTop", "pl.PlateBot", dofs=[1, 2, 3],
             enforce="equation", method=method)
        .bridge(ndm=3, ndf=3)
    )
    fem = ops.fem
    if n_plate is not None:
        got = len(list(fem.elements.select(pg="pl.PlateVol").ids))
        assert got == n_plate, f"{plate} plate meshed as {got} elements"

    steel = ops.nDMaterial.ElasticIsotropic(E=E, nu=NU)
    ops.element.stdBrick(pg="cover.CoverVol", material=steel)
    getattr(ops.element, element)(pg="pl.PlateVol", material=steel)
    ops.fix(pg="cover.Base", dofs=(1, 1, 1))
    ts = ops.timeSeries.Linear()
    with ops.pattern.Plain(series=ts) as pat:
        pat.sp(pg="pl.PlateTop", dof=3, value=-DELTA)
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

    live = emitter.ops
    live.reactions()
    base_ids = [int(t) for t in fem.nodes.select(pg="cover.Base").ids]
    r_z = sum(live.nodeReaction(t, 3) for t in base_ids)
    residual = 0.0
    for rec in fem.elements.constraints:
        if getattr(rec, "enforce", None) != "equation":
            continue
        weights = np.asarray(rec.weights, dtype=float)
        masters = [int(m) for m in rec.master_nodes]
        for d in rec.dofs:
            u_s = live.nodeDisp(int(rec.slave_node), int(d))
            u_m = sum(w * live.nodeDisp(m, int(d))
                      for w, m in zip(weights, masters))
            residual = max(residual, abs(u_s - u_m))
    return {"k": abs(r_z) / DELTA, "residual": residual}


def _stack_main() -> None:
    """``python -c`` entry: argv = workdir, then plate:method pairs solved
    back to back in this one process. ``--no-guards`` lifts the stock tet10
    refusal and the stock second-model refusal, to measure the engine
    behind them."""
    import json
    import sys

    args = sys.argv[1:]
    if "--no-guards" in args:
        args.remove("--no-guards")
        from apeGmsh.opensees.emitter import live

        live._tet10_volume_fixed = lambda ops: True  # type: ignore[assignment]
        real_init = live.LiveOpsEmitter.__init__

        def init(self, *, wipe=True):
            live._STOCK_EQ_ROWS_LIVE = False
            real_init(self, wipe=wipe)

        live.LiveOpsEmitter.__init__ = init  # type: ignore[method-assign]
    workdir = Path(args[0])
    out = []
    for i, case in enumerate(args[1:]):
        plate, method = case.split(":")
        sub = workdir / f"m{i}"
        sub.mkdir(parents=True)
        out.append(_tied_stack(sub, plate, method))
    print("RESULT " + json.dumps(out))


def _in_fresh_process(tmp_path: Path, *cases: str, no_guards: bool = False):
    """Run ``_stack_main`` in a new interpreter; one result dict per case."""
    import json
    import os
    import subprocess
    import sys

    root = Path(__file__).resolve().parents[1]
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        [str(root), str(root / "src"), env.get("PYTHONPATH", "")])
    env.setdefault("LADRUNO_OPENSEES_QUIET", "1")
    argv = [sys.executable, "-c",
            "from tests.test_meshable_part_route import _stack_main; "
            "_stack_main()", str(tmp_path), *cases]
    if no_guards:
        argv.append("--no-guards")
    proc = subprocess.run(argv, env=env, capture_output=True, text=True,
                          encoding="utf-8", errors="replace", timeout=600)
    lines = [ln for ln in proc.stdout.splitlines() if ln.startswith("RESULT ")]
    assert proc.returncode == 0 and lines, (
        f"tied-stack subprocess failed (rc={proc.returncode}):\n"
        f"{proc.stdout[-2000:]}\n{proc.stderr[-2000:]}")
    return json.loads(lines[-1][len("RESULT "):])


def _vs_closed_form(k: float) -> str:
    return (f"tied stack K = {k:.9g} vs closed form {K_EXACT:.9g} "
            f"({(k / K_EXACT - 1) * 100:+.4f} %)")


@pytest.mark.live
@pytest.mark.parametrize("plate, method, rel", [
    ("hex8-nested", "collocation", 1e-6),   # nested: collocation is exact
    ("tet4", "mortar", 1e-6),               # mortar passes the patch test
    ("hex8", "mortar", 1e-6),
    # collocation's own patch-test error, the same on every engine
    # (-0.132 % on this mesh, stock and fork alike)
    ("tet4", "collocation", 5e-3),
], ids=["hex8-nested-collocation", "tet4-mortar", "hex8-mortar",
        "tet4-collocation"])
def test_tied_stack_without_tet10_matches_series_closed_form(
    tmp_path: Path, plate: str, method: str, rel: float,
):
    """Any build with ``equationConstraint``, stock included: the tie is exact.

    The plate is an element that is right on every build, so this is the
    engine's tie alone against the closed form, and every emitted row holds
    to round-off in the solved state. The ``live`` marker puts it in CI's
    stock lane.
    """
    _require_equation_constraint()
    (res,) = _in_fresh_process(tmp_path, f"{plate}:{method}")
    assert res["residual"] < 1e-12, res
    assert res["k"] == pytest.approx(K_EXACT, rel=rel), _vs_closed_form(res["k"])


@pytest.mark.ladruno_fork
def test_tied_stack_matches_series_closed_form(tmp_path: Path):
    """Two stacked blocks, nu=0, equation tie: K = EA/L_total.

    hex8 (order 1) under tet10 (order 2) — mixed order ACROSS the tie
    (asserted at 0.1 %; the residual is collocation's, about -0.01 %).
    Fork-only because stock ``TenNodeTetrahedron`` is 6x too soft — not
    because of the tie, which is exact on stock (the test above) — and only
    on a fork build that can be shown to carry that fix.
    """
    if _tet10_volume_fixed() is not True:
        pytest.skip("fork build predates the ladrunoBuild stamp: cannot "
                    "confirm the TenNodeTetrahedron fix (fork PR #520)")
    (res,) = _in_fresh_process(tmp_path, "tet10:collocation")
    assert res["k"] == pytest.approx(K_EXACT, rel=1e-3), _vs_closed_form(res["k"])


@pytest.mark.live
def test_tied_stack_tet10_plate_reads_two_sevenths_on_stock(tmp_path: Path):
    """Stock: the old "71 % soft" is 2/7 — the tet10 defect in series.

    Measures the engine behind the stock tet10 refusal. The tie is exact
    (above) and the plate alone is 1/6 stiff, so the stack reads
    (1/(1+6)) / (1/(1+1)) = 2/7, to collocation's ~1e-4.
    """
    _require_equation_constraint()
    if _tet10_volume_fixed() is not False:
        pytest.skip("stock-only: this build's TenNodeTetrahedron is not "
                    "the known-defective upstream element")
    (res,) = _in_fresh_process(tmp_path, "tet10:collocation", no_guards=True)
    assert res["k"] / K_EXACT == pytest.approx(2 / 7, rel=1e-3), (
        _vs_closed_form(res["k"]))


@pytest.mark.live
def test_stock_wipe_keeps_equation_rows(tmp_path: Path):
    """Stock: a second tied model in one process is WRONG — why it is refused.

    Upstream ``Domain::clearAll()`` does not clear EQ constraints (the fork
    does, since fork PR #312), so model A's rows survive ``wipe()`` into
    model B. With the guard lifted, B converges to twice the closed form.
    With the guard on, B is refused. If upstream ever clears them, the
    first assertion on B fails: lift the guard.
    """
    _require_equation_constraint()
    if hasattr(_live_ops(), "criticalTimeStep"):
        pytest.skip("stock-only: the fork clears EQ rows on wipe()")
    a, b = _in_fresh_process(tmp_path / "unguarded", "tet4:mortar",
                             "hex8-nested:collocation", no_guards=True)
    assert a["k"] == pytest.approx(K_EXACT, rel=1e-6), _vs_closed_form(a["k"])
    assert b["k"] / K_EXACT == pytest.approx(2.0, rel=1e-6), (
        _vs_closed_form(b["k"]))

    with pytest.raises(AssertionError, match="stock wipe\\(\\) cannot clear"):
        _in_fresh_process(tmp_path / "guarded", "tet4:mortar",
                          "hex8-nested:collocation")
