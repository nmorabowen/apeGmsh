"""Fork-only second-order solids end-to-end — LadrunoBrick20 / LadrunoLST.

Mesh in apeGmsh, elevate with ``set_order(2)``, emit through the bridge and
solve on the live fork domain.

Why the analysis (and not just "the element loaded"): a wrongly ordered H20
still *loads* — ``Domain::addElement`` calls ``setDomain``, the element finds
a non-positive Jacobian, prints a warning and marks itself **DEAD**, and it
still shows up in ``getEleTags``. What a DEAD element does is zero its
tangent, so the only assertion that actually pins the Gmsh → serendipity
permutation is that the system solves. ``analyze() == 0`` here is the live
counterpart of the table check in
``tests/opensees/unit/primitives/test_elements_solid.py``.

Gated on the backend resolver via the ``ladruno_fork`` marker.
"""
from __future__ import annotations

import numpy as np
import pytest

from apeGmsh import apeGmsh
from apeGmsh.opensees import apeSees
from apeGmsh.opensees.emitter.live import LiveOpsEmitter

pytestmark = pytest.mark.ladruno_fork

ELE_TAG_LadrunoBrick20 = 33018      # classTags.h (ADR 72)
ELE_TAG_LadrunoLST = 33016          # classTags.h (ADR 70 P3)


def _live_class_tags(ops_mod) -> list[int]:
    tags = ops_mod.getEleTags() or []
    if isinstance(tags, int):
        tags = [tags]
    get_class = getattr(ops_mod, "getEleClassTags", None)
    if get_class is None:
        return []
    out: list[int] = []
    for t in tags:
        ct = get_class(t)
        out.extend(ct if isinstance(ct, (list, tuple)) else [ct])
    return out


def _nodes_on_plane(fem, axis: int, value: float) -> list[int]:
    ids = np.asarray(fem.nodes.ids)
    xyz = np.asarray(fem.nodes.coords)
    return [
        int(n) for n, p in zip(ids, xyz) if abs(float(p[axis]) - value) < 1e-9
    ]


# =====================================================================
# LadrunoBrick20 — hex20, the one element with a non-identity reorder
# =====================================================================

@pytest.mark.parametrize("formulation", ["std", "uri"])
def test_ladruno_brick20_solves_on_fork(formulation: str) -> None:
    with apeGmsh(model_name="h20_live", verbose=False) as g:
        box = g.model.geometry.add_box(0, 0, 0, 1, 1, 2)
        g.model.sync()
        g.mesh.structured.set_transfinite_box(box, n=3)
        g.mesh.generation.generate(3)
        g.mesh.generation.set_order(2, bubble=False)
        g.physical.add(3, [box], name="Body")
        fem = g.mesh.queries.get_fem_data(dim=3)

    group = list(fem.elements)[0]
    assert group.element_type.npe == 20, "set_order(2) did not give hex20"

    base = _nodes_on_plane(fem, 2, 0.0)
    top = _nodes_on_plane(fem, 2, 2.0)
    assert base and top

    ops = apeSees(fem)
    ops.model(ndm=3, ndf=3)
    mat = ops.nDMaterial.ElasticIsotropic(E=2.0e8, nu=0.25)
    ops.element.LadrunoBrick20(
        pg="Body", material=mat, formulation=formulation,
    )
    ops.fix(nodes=base, dofs=(1, 1, 1))

    ts = ops.timeSeries.Linear()
    with ops.pattern.Plain(series=ts) as p:
        for nid in top:
            p.load(node=nid, forces=(0.0, 0.0, -1.0e3))

    ops.constraints.Plain()
    ops.numberer.RCM()
    ops.system.BandGeneral()
    ops.test.NormDispIncr(tol=1e-9, max_iter=10)
    ops.algorithm.Linear()
    ops.integrator.LoadControl(dlam=1.0)
    ops.analysis.Static()

    emitter = LiveOpsEmitter(wipe=True)
    ops.build().emit(emitter)

    assert ELE_TAG_LadrunoBrick20 in _live_class_tags(emitter.ops)
    # The permutation check: a scrambled H20 is DEAD, its tangent is zero,
    # and the system is singular — analyze() would not return 0.
    assert emitter.analyze(steps=1) == 0

    uz = [emitter.ops.nodeDisp(int(n), 3) for n in top]
    assert all(np.isfinite(uz))
    assert max(uz) < 0.0, "compressed column did not shorten"


def test_ladruno_brick20_reproduces_the_self_weight_solution() -> None:
    """Magnitude, not just sign — a self-weight patch test.

    Load through ``body_force`` rather than nodal loads on purpose: each
    element integrates its own consistent body-force vector, so this
    compares two elements on the *same* load. (Splitting a face load
    equally over nodes would not: the consistent share of a uniform
    traction differs between a bilinear and a quadratic face, so equal
    nodal loads mean different tractions for H8 and H20.)

    A prism of length ``L`` under body force ``b`` (force per unit
    volume), supported on rollers at the base, shortens by
    ``b·L² / (2E)``. Both elements should land on it; a wrongly permuted
    H20 would not be close (or would not solve at all).
    """
    E, nu, L, b = 2.0e8, 0.25, 2.0, 1.0e4
    results = {}

    for order, element in ((1, "LadrunoBrick"), (2, "LadrunoBrick20")):
        with apeGmsh(model_name=f"cmp{order}", verbose=False) as g:
            box = g.model.geometry.add_box(0, 0, 0, 1, 1, L)
            g.model.sync()
            g.mesh.structured.set_transfinite_box(box, n=3)
            g.mesh.generation.generate(3)
            if order > 1:
                g.mesh.generation.set_order(2, bubble=False)
            g.physical.add(3, [box], name="Body")
            fem = g.mesh.queries.get_fem_data(dim=3)

        xyz_of = {
            int(n): p for n, p in
            zip(np.asarray(fem.nodes.ids), np.asarray(fem.nodes.coords))
        }
        base = _nodes_on_plane(fem, 2, 0.0)
        top = _nodes_on_plane(fem, 2, L)

        ops = apeSees(fem)
        ops.model(ndm=3, ndf=3)
        mat = ops.nDMaterial.ElasticIsotropic(E=E, nu=nu)
        getattr(ops.element, element)(
            pg="Body", material=mat, body_force=(0.0, 0.0, -b),
        )
        # Rollers: the base is held only in z, so Poisson contraction is
        # free and the closed form applies. Two extra pins remove the
        # in-plane rigid translations and the spin about z.
        ops.fix(nodes=base, dofs=(0, 0, 1))
        corner = min(base, key=lambda t: (xyz_of[t][0], xyz_of[t][1]))
        ops.fix(nodes=[corner], dofs=(1, 1, 0))
        ops.fix(nodes=[max(base, key=lambda t: xyz_of[t][0])], dofs=(0, 1, 0))

        ops.constraints.Plain()
        ops.numberer.RCM()
        ops.system.BandGeneral()
        ops.test.NormDispIncr(tol=1e-10, max_iter=10)
        ops.algorithm.Linear()
        ops.integrator.LoadControl(dlam=1.0)
        ops.analysis.Static()

        emitter = LiveOpsEmitter(wipe=True)
        ops.build().emit(emitter)
        assert emitter.analyze(steps=1) == 0, f"{element} did not solve"
        results[element] = float(np.mean(
            [emitter.ops.nodeDisp(int(n), 3) for n in top]
        ))

    exact = -b * L**2 / (2.0 * E)
    h8, h20 = results["LadrunoBrick"], results["LadrunoBrick20"]
    assert h20 == pytest.approx(exact, rel=0.02)
    assert h8 == pytest.approx(exact, rel=0.02)
    assert h20 == pytest.approx(h8, rel=0.01)


# =====================================================================
# LadrunoLST — tri6, identity reorder (same basis as SixNodeTri)
# =====================================================================

# ``finite`` is PlaneStrain only — the finite plane-stress view omits the
# thickness stretch in the volume weight (ADR 70), and both the fork and
# LadrunoLST.__post_init__ refuse the combination.
@pytest.mark.parametrize("geom, plane_type", [
    ("linear", "PlaneStress"),
    ("linear", "PlaneStrain"),
    ("finite", "PlaneStrain"),
])
def test_ladruno_lst_solves_on_fork(geom: str, plane_type: str) -> None:
    with apeGmsh(model_name="lst_live", verbose=False) as g:
        rect = g.model.geometry.add_rectangle(0, 0, 0, 4, 1)
        g.model.sync()
        g.mesh.sizing.set_global_size(0.5)
        g.mesh.generation.generate(2)
        g.mesh.generation.set_order(2, bubble=False)
        g.physical.add(2, [rect], name="Plate")
        fem = g.mesh.queries.get_fem_data(dim=2)

    group = list(fem.elements)[0]
    assert group.element_type.npe == 6, "set_order(2) did not give tri6"

    left = _nodes_on_plane(fem, 0, 0.0)
    right = _nodes_on_plane(fem, 0, 4.0)
    assert left and right

    ops = apeSees(fem)
    ops.model(ndm=2, ndf=2)
    mat = ops.nDMaterial.ElasticIsotropic(E=2.0e8, nu=0.25)
    if geom == "finite":
        # The fork demands a FiniteStrainND2DMaterial here (its LogStrain2D,
        # ND_TAG 33016) — the 3D LogStrain lift apeGmsh models is rejected.
        # apeGmsh does not expose a 2D finite-strain wrapper yet, so the
        # finite lane is emit-only for now: LadrunoLST writes "-geom finite"
        # correctly (pinned by the unit tests), it just cannot be *run*
        # from the bridge until the material lands. This skip clears itself
        # automatically when it does.
        log2d = getattr(ops.nDMaterial, "LogStrain2D", None)
        if log2d is None:
            pytest.skip(
                "LadrunoLST(geom='finite') needs a FiniteStrainND2DMaterial "
                "(fork LogStrain2D); apeGmsh models no 2D finite-strain "
                "material yet, so this lane is emit-only."
            )
        mat = log2d(inner=mat)
    ops.element.LadrunoLST(
        pg="Plate", material=mat, thickness=0.1,
        plane_type=plane_type, geom=geom,
    )
    ops.fix(nodes=left, dofs=(1, 1))

    ts = ops.timeSeries.Linear()
    with ops.pattern.Plain(series=ts) as p:
        for nid in right:
            p.load(node=nid, forces=(0.0, -1.0e2))

    ops.constraints.Plain()
    ops.numberer.RCM()
    ops.system.BandGeneral()
    ops.test.NormDispIncr(tol=1e-8, max_iter=20)
    ops.algorithm.Newton() if geom == "finite" else ops.algorithm.Linear()
    ops.integrator.LoadControl(dlam=1.0)
    ops.analysis.Static()

    emitter = LiveOpsEmitter(wipe=True)
    ops.build().emit(emitter)

    assert ELE_TAG_LadrunoLST in _live_class_tags(emitter.ops)
    assert emitter.analyze(steps=1) == 0

    uy = [emitter.ops.nodeDisp(int(n), 2) for n in right]
    assert all(np.isfinite(uy))
    assert max(uy) < 0.0, "tip-loaded cantilever did not deflect downward"
