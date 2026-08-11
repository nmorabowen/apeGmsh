"""Bernstein-aware consistent surface loads on BezierTet10 — ADR 0091.

The TIMs T2 incident: ``reduction="consistent"`` integrated tractions
against the LAGRANGE tri6 face functions (corners ~0, midsides q·A/3).
BezierTet10 DOFs are Bernstein CONTROL values, so those loads represent a
strongly oscillatory traction — elastically the resultant is exact, but
locally it spikes, which drove near-surface DruckerPrager Gauss points
into apex/tension on a strip-footing deck. ``basis="bernstein"``
integrates against the Bernstein face functions instead: uniform q maps
to EQUAL control-point loads q·A_face/6.

The live check is a uniform-compression patch test: with the Bernstein
load vector the discrete solution IS the exact linear field (uniform
σ_zz, uniform top settlement); with the Lagrange vector on the same
Bézier mesh the top displacement field oscillates. Gated on the backend
resolver via the ``ladruno_fork`` marker.

The same runs double as the end-to-end gate on the ADR 0091 bridge
guard: the Lagrange arm must raise ``WarnLoadBasisMismatch`` at
``build()`` on a real Bézier deck, and the Bernstein arm must be silent.
"""
from __future__ import annotations

import warnings as _warnings

import numpy as np
import pytest

from apeGmsh import apeGmsh
from apeGmsh.opensees import apeSees
from apeGmsh.opensees._internal.build import WarnLoadBasisMismatch
from apeGmsh.opensees.emitter.live import LiveOpsEmitter

pytestmark = pytest.mark.ladruno_fork

E, NU = 2.0e8, 0.25
L = 2.0            # column height
Q = 1.0e4          # uniform pressure on the top face (force / area)


def _nodes_on_plane(fem, axis: int, value: float) -> list[int]:
    ids = np.asarray(fem.nodes.ids)
    xyz = np.asarray(fem.nodes.coords)
    return [
        int(n) for n, p in zip(ids, xyz) if abs(float(p[axis]) - value) < 1e-9
    ]


def _solve_column(basis: str) -> tuple[dict[int, float], float]:
    """Mesh + solve the 1x1xL BezierTet10 column under top traction.

    Returns (top-node uz map, total imported vertical force).
    """
    eps = 1e-6
    with apeGmsh(model_name=f"bezier_{basis}", verbose=False) as g:
        box = g.model.geometry.add_box(0, 0, 0, 1, 1, L)
        g.model.sync()
        g.physical.add(3, [box], name="Body")
        top_faces = [
            t for _, t in g.model.queries.entities_in_bounding_box(
                -eps, -eps, L - eps, 1 + eps, 1 + eps, L + eps, dim=2,
            )
        ]
        assert top_faces, "no top face found"
        g.physical.add(2, top_faces, name="Top")
        with g.loads.case("press"):
            g.loads.surface.traction(
                "Top", vector=(0.0, 0.0, -Q),
                reduction="consistent", basis=basis,
            )
        g.mesh.sizing.set_global_size(0.6)
        g.mesh.generation.generate(dim=3)
        g.mesh.generation.set_order(2, bubble=False)
        fem = g.mesh.queries.get_fem_data(dim=3)

    group = list(fem.elements)[0]
    assert group.element_type.npe == 10, "set_order(2) did not give tet10"

    xyz_of = {
        int(n): p for n, p in
        zip(np.asarray(fem.nodes.ids), np.asarray(fem.nodes.coords))
    }
    base = _nodes_on_plane(fem, 2, 0.0)
    top = _nodes_on_plane(fem, 2, L)
    assert base and top

    ops = apeSees(fem)
    ops.model(ndm=3, ndf=3)
    mat = ops.nDMaterial.ElasticIsotropic(E=E, nu=NU)
    ops.element.BezierTet10(pg="Body", material=mat)

    # Rollers at the base (z only) + two pins so the uniform-stress
    # closed form u_z(L) = -Q·L/E applies (Poisson expansion is free).
    ops.fix(nodes=base, dofs=(0, 0, 1))
    corner = min(base, key=lambda t: (xyz_of[t][0], xyz_of[t][1]))
    ops.fix(nodes=[corner], dofs=(1, 1, 0))
    ops.fix(nodes=[max(base, key=lambda t: xyz_of[t][0])], dofs=(0, 1, 0))

    ts = ops.timeSeries.Linear()
    with ops.pattern.Plain(series=ts) as p:
        p.from_model("press")

    ops.constraints.Plain()
    ops.numberer.RCM()
    ops.system.UmfPack()
    ops.test.NormDispIncr(tol=1e-10, max_iter=10)
    ops.algorithm.Linear()
    ops.integrator.LoadControl(dlam=1.0)
    ops.analysis.Static()

    emitter = LiveOpsEmitter(wipe=True)
    # ADR 0091 guard, end-to-end on a real Bézier deck: the Lagrange
    # arm must be flagged at build(); the Bernstein arm must be silent.
    if basis == "lagrange":
        with pytest.warns(WarnLoadBasisMismatch, match="bernstein"):
            ops.build().emit(emitter)
    else:
        with _warnings.catch_warnings():
            _warnings.simplefilter("error", WarnLoadBasisMismatch)
            ops.build().emit(emitter)
    assert emitter.analyze(steps=1) == 0, f"basis={basis} did not solve"

    uz = {int(n): float(emitter.ops.nodeDisp(int(n), 3)) for n in top}
    total_fz = sum(
        (rec.force_xyz or (0.0, 0.0, 0.0))[2] for rec in fem.nodes.loads
    )
    return uz, total_fz


def test_bernstein_traction_is_the_exact_patch_test() -> None:
    uz, total_fz = _solve_column("bernstein")
    # Resultant: partition of unity keeps -Q·A regardless of basis.
    assert total_fz == pytest.approx(-Q * 1.0, rel=1e-9)
    # Equal control-point loads reproduce the exact uniform-stress
    # solution: every top control point settles by Q·L/E.
    exact = -Q * L / E
    vals = np.array(list(uz.values()))
    np.testing.assert_allclose(vals, exact, rtol=1e-6)


def test_lagrange_on_bezier_oscillates_bernstein_does_not() -> None:
    """The T2 mechanism, made visible on a clean elastic column."""
    uz_b, _ = _solve_column("bernstein")
    uz_l, total_fz_l = _solve_column("lagrange")
    # The Lagrange vector still carries the exact resultant …
    assert total_fz_l == pytest.approx(-Q * 1.0, rel=1e-9)
    # … but applied to CONTROL values it is an oscillatory traction:
    # the top surface no longer settles uniformly.
    spread_b = float(np.ptp(list(uz_b.values())))
    spread_l = float(np.ptp(list(uz_l.values())))
    mean_u = abs(float(np.mean(list(uz_b.values()))))
    assert spread_b < 1e-5 * mean_u
    assert spread_l > 100.0 * max(spread_b, 1e-30), (
        "expected the Lagrange-consistent load on Bézier control values "
        "to produce a visibly oscillatory top-surface settlement"
    )
