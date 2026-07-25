"""Fork-only 2-D finite strain end-to-end — LogStrain2D + the plane family.

``LogStrain2D`` (ND_TAG 33016) is the fork's only ``FiniteStrainND2DMaterial``,
so it is the single gate on the whole ``Ladruno*(geom="finite")`` plane lane:
``LadrunoQuad``, ``LadrunoCST`` and ``LadrunoLST`` all drive the material by
``setTrialF`` and reject anything else — including the 3-D ``LogStrain`` lift.

These tests solve a tip-loaded cantilever with each element in both kinematic
regimes and check the fork accepts the pairing. ``LadrunoLST`` also appears in
``test_second_order_solids_live.py`` (it is the second-order member of the
family); here it stands in as the T6 arm of the finite lane.

Gated on the backend resolver via the ``ladruno_fork`` marker.
"""
from __future__ import annotations

import numpy as np
import pytest

from apeGmsh import apeGmsh
from apeGmsh.opensees import apeSees
from apeGmsh.opensees.emitter.live import LiveOpsEmitter

pytestmark = pytest.mark.ladruno_fork

ND_TAG_LogStrain2D = 33016      # classTags.h (ADR 70)


def _nodes_on_plane(fem, axis: int, value: float) -> list[int]:
    ids = np.asarray(fem.nodes.ids)
    xyz = np.asarray(fem.nodes.coords)
    return [
        int(n) for n, p in zip(ids, xyz) if abs(float(p[axis]) - value) < 1e-9
    ]


def _plate(*, order: int, recombine: bool):
    """A 4 x 1 plate: quads when recombined, triangles otherwise."""
    with apeGmsh(model_name="plane_fs", verbose=False) as g:
        rect = g.model.geometry.add_rectangle(0, 0, 0, 4, 1)
        g.model.sync()
        if recombine:
            g.mesh.structured.set_recombine(rect, dim=2)
        g.mesh.sizing.set_global_size(0.5)
        g.mesh.generation.generate(2)
        if recombine:
            g.mesh.structured.recombine()
        if order > 1:
            g.mesh.generation.set_order(2, bubble=False)
        g.physical.add(2, [rect], name="Plate")
        return g.mesh.queries.get_fem_data(dim=2)


# (label, element, mesh order, recombine, expected npe)
_CASES = [
    ("quad4", "LadrunoQuad", 1, True, 4),
    ("tri3", "LadrunoCST", 1, False, 3),
    ("tri6", "LadrunoLST", 2, False, 6),
]


@pytest.mark.parametrize(
    "label, element, order, recombine, npe",
    _CASES, ids=[c[0] for c in _CASES],
)
@pytest.mark.parametrize("geom", ["linear", "finite"])
def test_plane_element_solves_in_both_kinematic_regimes(
    label, element, order, recombine, npe, geom,
) -> None:
    fem = _plate(order=order, recombine=recombine)
    group = list(fem.elements)[0]
    assert group.element_type.npe == npe, (
        f"{label}: expected npe={npe}, got {group.element_type.npe}"
    )

    left = _nodes_on_plane(fem, 0, 0.0)
    right = _nodes_on_plane(fem, 0, 4.0)
    assert left and right

    ops = apeSees(fem)
    ops.model(ndm=2, ndf=2)
    mat = ops.nDMaterial.ElasticIsotropic(E=2.0e8, nu=0.25)
    if geom == "finite":
        # The one material the finite plane kernel accepts.
        mat = ops.nDMaterial.LogStrain2D(inner=mat)

    getattr(ops.element, element)(
        pg="Plate", material=mat, thickness=0.1, geom=geom,
    )
    ops.fix(nodes=left, dofs=(1, 1))

    ts = ops.timeSeries.Linear()
    with ops.pattern.Plain(series=ts) as p:
        for nid in right:
            p.load(node=nid, forces=(0.0, -1.0e2))

    ops.constraints.Plain()
    ops.numberer.RCM()
    ops.system.BandGeneral()
    ops.test.NormDispIncr(tol=1e-8, max_iter=30)
    if geom == "finite":
        ops.algorithm.Newton()
    else:
        ops.algorithm.Linear()
    ops.integrator.LoadControl(dlam=1.0)
    ops.analysis.Static()

    emitter = LiveOpsEmitter(wipe=True)
    ops.build().emit(emitter)
    assert emitter.analyze(steps=1) == 0, f"{label}/{geom} did not solve"

    uy = [emitter.ops.nodeDisp(int(n), 2) for n in right]
    assert all(np.isfinite(uy))
    assert max(uy) < 0.0, "tip-loaded cantilever did not deflect downward"


def test_logstrain2d_loads_on_the_fork() -> None:
    """The wrapper reaches the domain with its own class tag, and the
    inner is emitted before it (dependency ordering)."""
    fem = _plate(order=1, recombine=True)
    left = _nodes_on_plane(fem, 0, 0.0)

    ops = apeSees(fem)
    ops.model(ndm=2, ndf=2)
    inner = ops.nDMaterial.ElasticIsotropic(E=2.0e8, nu=0.25)
    wrapper = ops.nDMaterial.LogStrain2D(inner=inner)
    ops.element.LadrunoQuad(
        pg="Plate", material=wrapper, thickness=0.1, geom="finite",
    )
    ops.fix(nodes=left, dofs=(1, 1))

    emitter = LiveOpsEmitter(wipe=True)
    ops.build().emit(emitter)

    live = emitter.ops
    get_class = getattr(live, "getNDMaterialClassTags", None)
    if get_class is None:
        pytest.skip("this build cannot enumerate nDMaterial class tags")
    tags = get_class()
    tags = [tags] if isinstance(tags, int) else list(tags)
    assert ND_TAG_LogStrain2D in tags


def test_the_three_dimensional_lift_is_refused_by_a_plane_element() -> None:
    """``LogStrain`` (3-D, ND_TAG 33010) is finite-strain but NOT a
    ``FiniteStrainND2DMaterial`` — apeGmsh happily emits it, and the fork
    is the one that says no. Pins that the two lifts are not
    interchangeable, which is the whole reason LogStrain2D exists."""
    fem = _plate(order=1, recombine=True)
    left = _nodes_on_plane(fem, 0, 0.0)

    ops = apeSees(fem)
    ops.model(ndm=2, ndf=2)
    inner = ops.nDMaterial.ElasticIsotropic(E=2.0e8, nu=0.25)
    ops.element.LadrunoQuad(
        pg="Plate",
        material=ops.nDMaterial.LogStrain(inner=inner),   # 3-D lift
        thickness=0.1, geom="finite",
    )
    ops.fix(nodes=left, dofs=(1, 1))

    emitter = LiveOpsEmitter(wipe=True)
    with pytest.raises(Exception):
        ops.build().emit(emitter)
