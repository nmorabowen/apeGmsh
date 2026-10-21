"""Fork lane: RESPONSE_CATALOG against a real solve, read back from MPCO and ``.ladruno``.

Gap G6 of the DomainCapture parity table (#1767, #1945, slice #1953). The
stock-lane half reads the same one-element models from ``.out``:
``tests/test_results_catalog_out_real.py``. That file owns the case table,
the model builders and both oracles: the live ``eleResponse`` vector, and the
``D B(xi) u`` or section anchor. This file only changes the route:

* ``MPCO-spec``: the spec's own emit, ``ResolvedRecorderSpec.emit_mpco``,
  with a nodal record too.
* ``MPCO`` and ``Ladruno``: ``recorder mpco|ladruno f -N displacement -E
  <token>``, the grammar the bridge's ``ops.recorder.mpco`` and
  ``ops.recorder.ladruno`` write.

Every file is read through the public factory at ``rtol=1e-10``. The values
and the GP natural coordinates are separate tests, so a route that gets one
right and the other wrong stays pinned on both. Every class x route cell
passes:

================= ============ ============ ===============================
Family            MPCO-spec    MPCO         Ladruno
================= ============ ============ ===============================
FourNodeQuad,     pass         pass         pass
Tri31, SSPquad,
solids
SixNodeTri,       pass         pass         pass (#1998)
EightNodeQuad     (rule 0)     (rule 0)     (rule 0)
shells (8 cases)  pass (#1991) pass         pass (#1990)
trusses           G7           G7           G7
================= ============ ============ ===============================

The fork writes SixNodeTri and EightNodeQuad under the bracket key
``[0:0:0]``. MPCO resolves rule 0 to the class's one catalogued rule
(#1970). On ``.ladruno`` the named-column bucket carries no GP_PARAM, so the
reader places its GPs from the catalog's one fitting layout (#1998); the
``D B(xi) u`` anchor checks those coordinates. The shells write their
resultants as ``p11,p22,p12,m11,m22,m12,q1,q2`` per GP, which the reader
names in the catalog's 8-component shell order (#1990), and their strains as
``eps11,eps22,gamma12,theta11,theta22,theta33,gamma13,gamma23``, which it
names as the catalog's generalised strains (#2169). Trusses are not
repeated: ``test_element_families_ladruno_live.py`` (G7, #1949) already pins
them on both routes.

The material-state tests at the end take the ``MPCO-spec`` route alone: a
stdBrick of ASDConcrete3D pulled past cracking (``damage_*``,
``equivalent_plastic_strain_*`` -> ``material.damage`` /
``material.equivalentPlasticStrain``) and one of LadrunoJ2 pushed past yield
(``plastic_strain_*`` and its derived scalars -> ``material.plasticStrain``),
each against ``eleResponse(1, "material", gp, key)`` at every GP.

Gated by the ``ladruno_fork`` marker, so the nightly ``live-fork`` lane runs it
against the pinned build.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest

from tests.test_results_catalog_out_real import (
    _CUBE,
    CASES,
    SHELL,
    Built,
    Case,
    MinimalFem,
    _nodes,
    assert_catalog_entry,
    assert_coords_match,
    assert_values_match,
    capture_truth,
    get_live_ops,
    record_components,
    solve,
)

pytestmark = pytest.mark.ladruno_fork

ROUTES = ("MPCO-spec", "MPCO", "Ladruno")

#: Trusses are G7's (see the module docstring).
FORK_CASES = [k for k, c in CASES.items() if c.anchor != "truss"]


def _params() -> list[Any]:
    return [pytest.param(key, route, id=f"{key}-{route}")
            for key in FORK_CASES for route in ROUTES]


def _run(case: Case, route: str, tmp_path: Path, stub_model_h5: Path):
    from apeGmsh.results import Results
    from apeGmsh.results._bind import ModelFemMismatchWarning
    from apeGmsh.results.spec._resolved import (
        ResolvedRecorderRecord,
        ResolvedRecorderSpec,
    )

    ops = get_live_ops()
    built = case.build(ops)
    assert ops.eleType(built.eid) == case.cpp_class
    solve(ops)
    if route == "MPCO-spec":
        fem = MinimalFem(built)
        path = tmp_path / "run.mpco"
        spec = ResolvedRecorderSpec(fem_snapshot_id=fem.snapshot_id, records=(
            ResolvedRecorderRecord(
                category="gauss", name="rec",
                components=record_components(case), dt=None, n_steps=None,
                element_ids=np.array([built.eid]),
                element_class_name=case.cpp_class,
            ),
            ResolvedRecorderRecord(
                category="nodes", name="u", components=("displacement_x",),
                dt=None, n_steps=None, node_ids=built.node_ids,
            ),
        ))
        with spec.emit_mpco(str(path), ops=ops):
            assert ops.analyze(1) == 0
            truth = capture_truth(ops, case, built)
    else:
        path = tmp_path / ("run.mpco" if route == "MPCO" else "run.ladruno")
        ops.recorder(route.lower(), str(path), "-N", "displacement",
                     "-E", case.token)
        assert ops.analyze(1) == 0
        truth = capture_truth(ops, case, built)
        ops.remove("recorders")
    # The stub model.h5 is another model's archive: each factory says so
    # and binds the FEMData it synthesizes from the file instead.
    if route == "Ladruno":
        with pytest.warns(ModelFemMismatchWarning):
            r = Results.from_ladruno(str(path), model_h5=stub_model_h5)
    else:
        with pytest.warns(ModelFemMismatchWarning):
            r = Results.from_mpco(str(path), model_h5=stub_model_h5)
    assert [s.n_steps for s in r.stages] == [1]
    return built, truth, r


@pytest.mark.parametrize("key,route", _params())
def test_catalog_values_read_back_from_hdf5(
    key: str, route: str, tmp_path: Path, stub_model_h5: Path,
) -> None:
    case = CASES[key]
    assert_catalog_entry(case)
    built, truth, r = _run(case, route, tmp_path, stub_model_h5)
    assert_values_match(case, built, truth,
                        lambda c: r.elements.gauss.get(component=c), where=route)


@pytest.mark.parametrize("key,route", _params())
def test_catalog_gp_coords_read_back_from_hdf5(
    key: str, route: str, tmp_path: Path, stub_model_h5: Path,
) -> None:
    case = CASES[key]
    built, truth, r = _run(case, route, tmp_path, stub_model_h5)
    assert_coords_match(case, built, truth,
                        lambda c: r.elements.gauss.get(component=c), where=route)


#: The cases #1990 (shells) and #1998 (rule-0 named buckets) fixed.
PARITY_CASES = [k for k in FORK_CASES
                if CASES[k].comps == SHELL or CASES[k].cpp_class in
                ("SixNodeTri", "EightNodeQuad")]


#: The shells' generalised section strains in source order (#2169): every
#: shell tags ``eps11,eps22,gamma12,theta11,theta22,theta33,gamma13,gamma23``
#: from ``getSectionDeformation()`` (fork 100eb9ae1). Spelled out, not read
#: from RESPONSE_CATALOG.
SHELL_STRAINS = ("membrane_strain_xx", "membrane_strain_yy",
                 "membrane_strain_xy", "curvature_xx", "curvature_yy",
                 "curvature_xy", "transverse_shear_strain_xz",
                 "transverse_shear_strain_yz")

#: ``(case, recorder token, components)``: each case's own response, and
#: ``-E strains`` for every shell.
PARITY_READS = [
    pytest.param(k, CASES[k].token, CASES[k].comps, id=k)
    for k in PARITY_CASES
] + [
    pytest.param(k, "strains", SHELL_STRAINS, id=f"{k}-strains")
    for k in PARITY_CASES if CASES[k].comps == SHELL
]


@pytest.mark.parametrize("key,token,comps", PARITY_READS)
def test_ladruno_matches_mpco_on_the_same_solve(
    key: str, token: str, comps: "tuple[str, ...]", tmp_path: Path,
    stub_model_h5: Path,
) -> None:
    """``.ladruno`` and MPCO, recording the same step, read back the same.

    Values at ``rtol=1e-10``, GP for GP, and the same natural coordinates.
    A shell's ``-E strains`` reads as the generalised strains on both, and
    never as the continuum ``strain_xx``.
    """
    from apeGmsh.results import Results
    from apeGmsh.results._bind import ModelFemMismatchWarning

    case = CASES[key]
    ops = get_live_ops()
    built = case.build(ops)
    assert ops.eleType(built.eid) == case.cpp_class
    solve(ops)
    mpco, lad = tmp_path / "run.mpco", tmp_path / "run.ladruno"
    ops.recorder("mpco", str(mpco), "-N", "displacement", "-E", token)
    ops.recorder("ladruno", str(lad), "-N", "displacement", "-E", token)
    assert ops.analyze(1) == 0
    ops.remove("recorders")
    with pytest.warns(ModelFemMismatchWarning):
        r_lad = Results.from_ladruno(str(lad), model_h5=stub_model_h5)
    with pytest.warns(ModelFemMismatchWarning):
        r_mpco = Results.from_mpco(str(mpco), model_h5=stub_model_h5)
    if comps == SHELL_STRAINS:
        listed = set(r_lad.elements.gauss.available_components())
        assert set(SHELL_STRAINS) <= listed, (key, sorted(listed))
        assert not {"strain_xx", "strain_yy"} & listed, (key, sorted(listed))
    slabs = [(r_lad.elements.gauss.get(component=c),
              r_mpco.elements.gauss.get(component=c)) for c in comps]
    scale = max(np.abs(np.asarray(m.values)).max() for _, m in slabs)
    assert scale > 0.0, f"{key}: MPCO reads all zeros"
    for comp, (a, b) in zip(comps, slabs):
        assert np.asarray(a.values).shape == (1, case.n_gp), (key, comp)
        np.testing.assert_array_equal(a.element_index, b.element_index)
        np.testing.assert_allclose(
            a.values, b.values, rtol=1e-10, atol=1e-10 * scale,
            err_msg=f"{key} {comp}: .ladruno vs MPCO",
        )
        np.testing.assert_allclose(
            a.natural_coords, b.natural_coords, rtol=1e-10, atol=1e-14,
            err_msg=f"{key} {comp}: GP coords, .ladruno vs MPCO",
        )


# ---------------------------------------------------------------------------
# Material state through the spec's MPCO emit
# ---------------------------------------------------------------------------


def _material_brick_run(
    material: str, components: tuple[str, ...], keys: tuple[str, ...],
    tmp_path: Path, stub_model_h5: Path,
) -> tuple[dict[str, np.ndarray], Any]:
    """One stdBrick, four steps, recorded by ``spec.emit_mpco``.

    Returns ``eleResponse(1, "material", gp, key)`` for each of ``keys`` at
    the eight GPs after the last step, and the MPCO file read back.
    """
    from apeGmsh.results import Results
    from apeGmsh.results._bind import ModelFemMismatchWarning
    from apeGmsh.results.spec._resolved import (
        ResolvedRecorderRecord,
        ResolvedRecorderSpec,
    )

    ops = get_live_ops()
    ops.wipe()
    ops.model("basic", "-ndm", 3, "-ndf", 3)
    if material == "ASDConcrete3D":
        ops.nDMaterial("ASDConcrete3D", 1, 30e6, 0.2, "-fc", 30e3, "-ft", 3e3)
    else:
        e, nu = 200e9, 0.3
        ops.nDMaterial("LadrunoJ2", 1, e / (3 * (1 - 2 * nu)), e / (2 * (1 + nu)),
                       "-iso", "voce", 250e6, 0.0, 0.0, 2e9)
    ids = _nodes(ops, _CUBE)
    for nid in (1, 2, 3, 4):
        ops.fix(nid, 1, 1, 1)
    ops.element("stdBrick", 1, *(int(n) for n in ids), 1)
    ops.timeSeries("Linear", 1)
    ops.pattern("Plain", 1, 1)
    for nid in (5, 6, 7, 8):
        if material == "ASDConcrete3D":
            ops.sp(nid, 3, 6.0e-4)          # 6x the cracking strain ft/E
        else:
            ops.load(nid, 1.0e7, 0.0, -1.0e8)   # past yield, with shear
    ops.system("FullGeneral")
    ops.numberer("Plain")
    ops.constraints("Transformation")
    ops.test("NormDispIncr", 1e-10, 50)
    ops.algorithm("Newton")
    ops.integrator("LoadControl", 0.25)
    ops.analysis("Static")

    built = Built(ids, _CUBE, ndf=3)
    fem = MinimalFem(built)
    path = tmp_path / "run.mpco"
    spec = ResolvedRecorderSpec(fem_snapshot_id=fem.snapshot_id, records=(
        ResolvedRecorderRecord(
            category="gauss", name="mat", components=components,
            dt=None, n_steps=None, element_ids=np.array([1]),
            element_class_name="Brick",
        ),
        ResolvedRecorderRecord(
            category="nodes", name="u", components=("displacement_x",),
            dt=None, n_steps=None, node_ids=built.node_ids,
        ),
    ))
    with spec.emit_mpco(str(path), ops=ops):
        for _ in range(4):
            assert ops.analyze(1) == 0
        truth = {
            key: np.array([
                np.asarray(ops.eleResponse(1, "material", str(gp), key), float)
                for gp in range(1, 9)
            ])
            for key in keys
        }
    with pytest.warns(ModelFemMismatchWarning):
        r = Results.from_mpco(str(path), model_h5=stub_model_h5)
    assert [s.n_steps for s in r.stages] == [4]
    return truth, r


def _last_step(r: Any, component: str) -> np.ndarray:
    slab = r.elements.gauss.get(component=component)
    values = np.asarray(slab.values)
    assert values.shape == (4, 8), (component, values.shape)
    return values[-1]


def test_mpco_spec_concrete_damage_and_peeq_read_back(
    tmp_path: Path, stub_model_h5: Path,
) -> None:
    """``damage_*`` and ``equivalent_plastic_strain_*`` (ASDConcrete3D).

    Before the fix the first two raised and the PEEQ pair emitted
    ``-E strains``, the total-strain bucket, and read back empty.
    """
    truth, r = _material_brick_run(
        "ASDConcrete3D",
        ("damage_tension", "damage_compression",
         "equivalent_plastic_strain_tension",
         "equivalent_plastic_strain_compression"),
        ("damage", "equivalentPlasticStrain"), tmp_path, stub_model_h5,
    )
    damage, peeq = truth["damage"], truth["equivalentPlasticStrain"]
    assert damage.shape == peeq.shape == (8, 2)
    # Cracked in tension, untouched in compression: a column swap shows.
    assert damage[:, 0].min() > 0.1 and np.allclose(damage[:, 1], 0.0)
    assert peeq[:, 0].min() > 1e-4
    for k, suffix in enumerate(("tension", "compression")):
        np.testing.assert_allclose(_last_step(r, f"damage_{suffix}"),
                                   damage[:, k], rtol=1e-10, atol=1e-14)
        np.testing.assert_allclose(
            _last_step(r, f"equivalent_plastic_strain_{suffix}"),
            peeq[:, k], rtol=1e-10, atol=1e-14)


_PLASTIC_TENSOR = ("plastic_strain_xx", "plastic_strain_yy", "plastic_strain_zz",
                   "plastic_strain_xy", "plastic_strain_yz", "plastic_strain_xz")


def test_mpco_spec_plastic_strain_tensor_and_derived_read_back(
    tmp_path: Path, stub_model_h5: Path,
) -> None:
    """``plastic_strain_*`` and a derived scalar (LadrunoJ2).

    Before the fix both raised at the emit. The tensor is checked
    component by component (LadrunoJ2's Voigt order xx, yy, zz, xy, yz, xz;
    engineering shear) and ``equivalent_plastic_strain_current`` against
    its closed form sqrt(2/3 e:e).
    """
    truth, r = _material_brick_run(
        "LadrunoJ2",
        _PLASTIC_TENSOR + ("equivalent_plastic_strain_current",),
        ("plasticStrain",), tmp_path, stub_model_h5,
    )
    ep = truth["plasticStrain"]
    assert ep.shape == (8, 6) and np.abs(ep).max() > 1e-3   # yielded
    for k, comp in enumerate(_PLASTIC_TENSOR):
        np.testing.assert_allclose(_last_step(r, comp), ep[:, k],
                                   rtol=1e-10, atol=1e-14)
    normal, shear = ep[:, :3], 0.5 * ep[:, 3:]
    peeq_current = np.sqrt(2.0 / 3.0 * (
        (normal ** 2).sum(axis=1) + 2.0 * (shear ** 2).sum(axis=1)))
    np.testing.assert_allclose(
        _last_step(r, "equivalent_plastic_strain_current"), peeq_current,
        rtol=1e-10)
