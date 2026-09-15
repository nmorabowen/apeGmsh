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


# ---------------------------------------------------------------------------
# ADR 0105 D6 — the ADR-94 build still labels the same Gauss columns for an
# ASDPlasticMaterial3D on LadrunoBrick.
# ---------------------------------------------------------------------------

SOLID_GAUSS = (
    "stress_xx", "stress_yy", "stress_zz", "stress_xy", "stress_yz", "stress_xz",
    "strain_xx", "strain_yy", "strain_zz", "strain_xy", "strain_yz", "strain_xz",
)

# ADR 0105 Amendment 1 — the five material-level buckets
# ``ASDPlasticMaterial3D::setResponse`` answers, in apeGmsh's names.
MATERIAL_LEVEL_GAUSS = (
    "material_mean_stress", "material_j2_stress",
    "material_volumetric_strain", "material_j2_strain",
    "back_stress_xx", "back_stress_yy", "back_stress_zz",
    "back_stress_xy", "back_stress_yz", "back_stress_xz",
)


def _single_hex_fem():
    with apeGmsh(model_name="gc_hex", verbose=False) as g:
        box = g.model.geometry.add_box(0, 0, 0, 1, 1, 1)
        g.model.sync()
        g.mesh.structured.set_transfinite_box(box, n=2)
        g.mesh.generation.generate(3)
        g.physical.add(3, [box], name="rock")
        return g.mesh.queries.get_fem_data(dim=3)


def _plane(fem, axis: int, value: float) -> list[int]:
    ids = np.asarray(fem.nodes.ids)
    xyz = np.asarray(fem.nodes.coords)
    return [
        int(n) for n, p in zip(ids, xyz) if abs(float(p[axis]) - value) < 1e-9
    ]


def test_asdplastic_on_ladruno_brick_keeps_its_gauss_columns(tmp_path) -> None:
    """Fork ADR-94 rewrote the material, not its response tokens: the
    ``stress`` / ``strain`` element responses of a ``MohrCoulombSoil`` deck
    on ``LadrunoBrick`` still land on the six neutral names each, and the
    material-level ``material.pstrain`` / ``material.eqpstrain`` requests
    (the recorder forwards ``material.<token>`` to every Gauss point) still
    land on ``plastic_strain_*`` / ``equivalent_plastic_strain`` — one column
    per (element, GP), engine-checked.  Pinned here because the ADR-94 build
    is the first one apeGmsh's ASDP decks are contracted against.

    ADR 0105 Amendment 1 adds the five remaining material-level buckets:
    ``PStress`` / ``J2Stress`` / ``VolStrain`` / ``J2Strain`` /
    ``BackStress`` → ``material_mean_stress`` / ``material_j2_stress`` /
    ``material_volumetric_strain`` / ``material_j2_strain`` /
    ``back_stress_*``.  MEASURED on fork build ``3622d6214`` with this
    deck, and asserted below to 1e-6 relative:

    * ``material_mean_stress`` == the tensor-derived ``mean_stress``,
      **same sign, factor 1** — both are ``trace(σ)/3``, tension-positive
      (``VoigtVector::meanStress()``).  The fork header's "mean
      (hydrostatic) stress" comment does not negate it; several yield
      functions negate it themselves at the call site.
    * ``material_j2_stress`` == the tensor-derived ``j2_stress``, **factor
      1** — both are J2 = ½·s:s, NOT √J2.
    * ``material_volumetric_strain`` == ``volumetric_strain`` (both
      ``trace(ε)``, not ``/3``) and ``material_j2_strain`` ==
      ``j2_strain`` (the material stores tensorial shear, so its J2
      matches the reader's halve-the-engineering-shear convention).

    The names stay provenance-distinct anyway: the equality above is a
    MEASUREMENT of one material on one build, not a contract, and only
    an audit per material would make it one.
    """
    fem = _single_hex_fem()
    ops = apeSees(fem)
    ops.model(ndm=3, ndf=3)
    mat = ops.nDMaterial.MohrCoulombSoil(
        c=100.0, phi=30.0, psi=30.0, E=1.0e6, nu=0.25,
    )
    ops.element.LadrunoBrick(pg="rock", material=mat)
    ops.fix(nodes=_plane(fem, 0, 0.0), dofs=(1, 0, 0))
    ops.fix(nodes=_plane(fem, 1, 0.0), dofs=(0, 1, 0))
    ops.fix(nodes=_plane(fem, 2, 0.0), dofs=(0, 0, 1))
    ts = ops.timeSeries.Linear()
    with ops.pattern.Plain(series=ts) as p:
        for n in _plane(fem, 0, 1.0):
            p.sp(node=n, dof=1, value=0.002)
        for n in _plane(fem, 1, 1.0):
            p.sp(node=n, dof=2, value=0.003)
        for n in _plane(fem, 2, 1.0):
            p.sp(node=n, dof=3, value=-0.01)
    path = str(tmp_path / "asdp_hex.ladruno")
    ops.recorder.Ladruno(
        file=path,
        elem_responses=("stress", "strain", "material.pstrain",
                        "material.eqpstrain", "material.PStress",
                        "material.J2Stress", "material.VolStrain",
                        "material.J2Strain", "material.BackStress"),
    )
    ops.constraints.Transformation()
    ops.numberer.Plain()
    ops.system.UmfPack()
    ops.test.NormDispIncr(tol=1e-10, max_iter=50)
    ops.algorithm.Newton()
    ops.integrator.LoadControl(dlam=0.05)
    ops.analysis.Static()

    emitter = LiveOpsEmitter(wipe=True)
    ops.build().emit(emitter)
    for _ in range(20):
        assert emitter.analyze(steps=1) == 0
    o = emitter.ops
    tags = o.getEleTags()
    eid = int(tags if isinstance(tags, int) else tags[0])
    o.eleResponse(eid, "forces")
    n_gp = 8
    live_stress = np.asarray(o.eleResponse(eid, "stresses"), dtype=np.float64)
    live_pstrain = np.array([
        np.asarray(o.eleResponse(eid, "material", k, "pstrain"), dtype=np.float64)
        for k in range(1, n_gp + 1)
    ])
    live_pstress = np.array([
        float(np.asarray(
            o.eleResponse(eid, "material", k, "PStress"), dtype=np.float64,
        )[0])
        for k in range(1, n_gp + 1)
    ])
    o.remove("recorders")

    r = Results.from_ladruno(path)
    available = r.elements.gauss.available_components()
    wanted = SOLID_GAUSS + (
        "plastic_strain_xx", "plastic_strain_yy", "plastic_strain_zz",
        "plastic_strain_xy", "plastic_strain_yz", "plastic_strain_xz",
        "equivalent_plastic_strain",
    ) + MATERIAL_LEVEL_GAUSS
    missing = [c for c in wanted if c not in available]
    assert not missing, f"{missing} absent from {sorted(available)}"
    for component in wanted:
        slab = r.elements.gauss.get(component=component)
        assert slab.values.shape[1] == n_gp, component
        assert np.all(np.isfinite(slab.values)), component
    # Values against the engine: stress_zz is slot 2 of each 6-block of the
    # element's own vector; plastic_strain_zz is slot 2 of each GP
    # material's own `pstrain`.
    assert live_stress.size == 6 * n_gp
    slab = r.elements.gauss.get(component="stress_zz")
    np.testing.assert_allclose(
        np.sort(slab.values[-1]), np.sort(live_stress.reshape(n_gp, 6)[:, 2]),
        rtol=1e-6, atol=1e-9,
    )
    assert live_pstrain.shape == (n_gp, 6)
    slab = r.elements.gauss.get(component="plastic_strain_zz")
    np.testing.assert_allclose(
        np.sort(slab.values[-1]), np.sort(live_pstrain[:, 2]),
        rtol=1e-6, atol=1e-12,
    )
    # The leg yielded: the plastic strain is not a column of zeros.
    assert np.max(np.abs(slab.values[-1])) > 1e-4

    # --- ADR 0105 Amendment 1 -------------------------------------------
    # The material's own mean stress, against the engine, per Gauss point.
    mean = r.elements.gauss.get(component="material_mean_stress")
    np.testing.assert_allclose(
        np.sort(mean.values[-1]), np.sort(live_pstress), rtol=1e-6, atol=1e-9,
    )
    # NullHardening: the back stress is initialised to zero and never
    # moves, so a nonzero column here means the block was mis-sliced.
    for component in (
        "back_stress_xx", "back_stress_yy", "back_stress_zz",
        "back_stress_xy", "back_stress_yz", "back_stress_xz",
    ):
        slab = r.elements.gauss.get(component=component)
        assert np.all(slab.values == 0.0), component
    # Measured relations to the tensor-derived scalars (see the docstring):
    # identical, same sign, factor 1 — the material and the reader use the
    # same definition for all four.
    for recorded, derived in (
        ("material_mean_stress", "mean_stress"),
        ("material_j2_stress", "j2_stress"),
        ("material_volumetric_strain", "volumetric_strain"),
        ("material_j2_strain", "j2_strain"),
    ):
        np.testing.assert_allclose(
            r.elements.gauss.get(component=recorded).values[-1],
            r.elements.gauss.get(component=derived).values[-1],
            rtol=1e-6, atol=1e-12, err_msg=f"{recorded} vs {derived}",
        )
    # J2, not √J2: the two differ by ~4e3 on this deck, so the equality
    # above is a real check on the definition, not on a near-zero column.
    j2 = r.elements.gauss.get(component="material_j2_stress").values[-1]
    assert np.min(np.abs(j2)) > 1.0
