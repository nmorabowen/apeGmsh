"""Fork-only — Gauss stress/strain survives the fork's UNNAMED columns.

The fork's plain ``stress`` / ``strain`` element responses on its plane
elements emit no ``output.tag("ResponseType", …)``, so the ``.ladruno``
recorder writes ``C1,C2,…,Cn`` and the reader — which named columns from
``COMP_NAMES`` alone — dropped every one of them. ``LadrunoCST`` /
``LadrunoLST`` / ``LadrunoQuad`` all answered
``elements.gauss.available_components() == []``: all continuum stress and
strain on every Ladruno plane element was invisible on this path.

The values are cross-checked against ``ops.eleResponse(eid, "stress")`` on
the live domain — the engine's own flat vector, an authority independent
of the reader under test.

Gated by the ``ladruno_fork`` marker (root conftest auto-skips off-fork).
"""
from __future__ import annotations

import numpy as np
import pytest

from apeGmsh import apeGmsh
from apeGmsh.opensees import apeSees
from apeGmsh.opensees.emitter.live import LiveOpsEmitter
from apeGmsh.results import Results

pytestmark = pytest.mark.ladruno_fork

# The six neutral names a 2-D continuum Gauss point must expose. Spelled
# out, not derived from RESPONSE_CATALOG — the catalog is what the fix
# reads from, so deriving them here would assert nothing.
PLANE_GAUSS = (
    "stress_xx", "stress_yy", "stress_xy",
    "strain_xx", "strain_yy", "strain_xy",
)


def _nodes_on_x(fem, value: float) -> list[int]:
    ids = np.asarray(fem.nodes.ids)
    xyz = np.asarray(fem.nodes.coords)
    return [
        int(n) for n, p in zip(ids, xyz) if abs(float(p[0]) - value) < 1e-9
    ]


def _run(kind: str, path: str, responses=("stress", "strain"), material=None):
    """Solve a cantilevered plate of ``kind`` and record it to ``path``."""
    with apeGmsh(model_name=f"gc_{kind}", verbose=False) as g:
        rect = g.model.geometry.add_rectangle(0, 0, 0, 4, 1)
        g.model.sync()
        g.mesh.sizing.set_global_size(1.0)
        if kind == "LadrunoQuad":
            g.mesh.structured.set_recombine(rect)
        g.mesh.generation.generate(2)
        if kind == "LadrunoLST":
            g.mesh.generation.set_order(2, bubble=False)
        g.physical.add(2, [rect], name="Plate")
        fem = g.mesh.queries.get_fem_data(dim=2)

    ops = apeSees(fem)
    ops.model(ndm=2, ndf=2)
    mat = (
        ops.nDMaterial.ElasticIsotropic(E=2.0e8, nu=0.25) if material is None
        else material(ops)
    )
    getattr(ops.element, kind)(
        pg="Plate", material=mat, thickness=0.1, plane_type="PlaneStrain",
    )
    ops.fix(nodes=_nodes_on_x(fem, 0.0), dofs=(1, 1))

    ts = ops.timeSeries.Linear()
    with ops.pattern.Plain(series=ts) as p:
        for nid in _nodes_on_x(fem, 4.0):
            p.load(node=nid, forces=(0.0, -1.0e2))

    ops.recorder.Ladruno(file=path, elem_responses=responses)

    ops.constraints.Plain()
    ops.numberer.RCM()
    ops.system.BandGeneral()
    ops.test.NormDispIncr(tol=1e-9, max_iter=20)
    ops.algorithm.Linear()
    ops.integrator.LoadControl(dlam=1.0)
    ops.analysis.Static()

    emitter = LiveOpsEmitter(wipe=True)
    ops.build().emit(emitter)
    assert emitter.analyze(steps=1) == 0
    emitter.ops.remove("recorders")   # flush the .ladruno
    return emitter


@pytest.mark.parametrize(
    ("kind", "n_gp"),
    [("LadrunoCST", 1), ("LadrunoLST", 3), ("LadrunoQuad", 4)],
)
def test_plain_stress_strain_reaches_the_gauss_level(
    tmp_path, kind: str, n_gp: int,
) -> None:
    path = str(tmp_path / f"{kind}.ladruno")
    emitter = _run(kind, path)

    eids = emitter.ops.getEleTags()
    if isinstance(eids, int):
        eids = [eids]
    eids = [int(e) for e in eids]
    assert emitter.ops.eleType(eids[0]) == kind

    r = Results.from_ladruno(path)
    available = r.elements.gauss.available_components()
    missing = [c for c in PLANE_GAUSS if c not in available]
    assert not missing, f"{kind}: {missing} absent from {sorted(available)}"

    for component in PLANE_GAUSS:
        slab = r.elements.gauss.get(component=component)
        assert slab.values.shape[1] == len(eids) * n_gp, (
            f"{kind}/{component}: expected one column per (element, GP)"
        )
        assert np.all(np.isfinite(slab.values)), f"{kind}/{component}"

    # Values, against the engine rather than against the reader. Compared
    # as a per-element multiset: the slab is GP-major across the whole
    # element set, and these buckets carry no GP_PARAM to order them by.
    slab = r.elements.gauss.get(component="stress_xx")
    for eid in eids:
        flat = np.asarray(
            emitter.ops.eleResponse(eid, "stress"), dtype=np.float64,
        )
        assert flat.size == 3 * n_gp
        live = np.sort(flat.reshape(n_gp, 3)[:, 0])
        read = np.sort(slab.values[-1][slab.element_index == eid])
        np.testing.assert_allclose(read, live, rtol=1e-10, atol=1e-12)

    # Not every column is the same column: a uniformly-labelled slab would
    # pass everything above. The cantilever's σ_xx varies along the span.
    assert np.ptp(slab.values[-1]) > 0.0


def _ladruno_j2(ops):
    """Plane-strain J2 — a material that DOES expose the out-of-plane σ_zz."""
    E, nu, = 2.0e8, 0.25
    return ops.nDMaterial.LadrunoJ2(
        K=E / (3.0 * (1.0 - 2.0 * nu)), G=E / (2.0 * (1.0 + nu)),
        sig0=2.5e5,
    )


def test_both_stress_tokens_in_one_recorder_do_not_double_count(
    tmp_path,
) -> None:
    """`stress` + `stressesPlaneStrain` cover the SAME (element, GP) slots.

    The plain token is anonymous (named from RESPONSE_CATALOG, 3
    components) and `stressesPlaneStrain` tags its own names and carries
    σ_zz too (4 components). Concatenating both gave `stress_xx` twice the
    columns of `stress_zz`, and any derived scalar needing both blew up
    with a broadcast error. This is the configuration the out-of-plane
    σ_zz work targets, not an edge case.
    """
    n_gp = 3                                # LadrunoLST
    path = str(tmp_path / "both_tokens.ladruno")
    emitter = _run(
        "LadrunoLST", path,
        responses=("stress", "stressesPlaneStrain"), material=_ladruno_j2,
    )
    eids = [int(e) for e in emitter.ops.getEleTags()]

    r = Results.from_ladruno(path)
    wanted = ("stress_xx", "stress_yy", "stress_zz", "stress_xy",
              "von_mises_stress")
    slabs = {c: r.elements.gauss.get(component=c) for c in wanted}

    for component, slab in slabs.items():
        # Single-count: one column per (element, GP), not two.
        assert slab.values.shape[1] == len(eids) * n_gp, component
        assert np.all(np.isfinite(slab.values)), component
        np.testing.assert_array_equal(
            slab.element_index, slabs["stress_xx"].element_index,
        )

    # σ_zz survives with the engine's own values — the named bucket won,
    # and it is the only token carrying the out-of-plane component.
    zz = slabs["stress_zz"]
    for eid in eids:
        flat = np.asarray(
            emitter.ops.eleResponse(eid, "stressPlaneStrain"),
            dtype=np.float64,
        )
        assert flat.size == 4 * n_gp
        np.testing.assert_allclose(
            np.sort(zz.values[-1][zz.element_index == eid]),
            np.sort(flat.reshape(n_gp, 4)[:, 3]),
            rtol=1e-10, atol=1e-12,
        )
    assert np.ptp(zz.values[-1]) > 0.0

    # The four components line up slot-for-slot: von Mises from the
    # recorded tensor must equal the derived scalar.
    xx, yy, zzv, xy = (
        slabs[c].values[-1]
        for c in ("stress_xx", "stress_yy", "stress_zz", "stress_xy")
    )
    hand = np.sqrt(
        0.5 * ((xx - yy) ** 2 + (yy - zzv) ** 2 + (zzv - xx) ** 2)
        + 3.0 * xy ** 2
    )
    np.testing.assert_allclose(
        slabs["von_mises_stress"].values[-1], hand, rtol=1e-10, atol=1e-9,
    )
