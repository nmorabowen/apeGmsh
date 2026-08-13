"""Tests — ADR 0080 B7: ``moment_curvature`` (the M–κ harness).

The gate is the elastic slope: for elastic fibers, ``EI0`` must equal the
**exact fiber sum** ``Σ E·A·r²`` about the bending axis, on both axes —
the identity an axis swap or a sign flip in the lowering cannot survive.
Then the inelastic keystone: ``ElasticPP`` fibers pushed deep plateau at
the closed-form ``Mp`` in both signs.

Every case here is closed-form. A rect patch's fiber grid is the midpoint
rule, so:

* ``Σ A·|y| = b·h²/4`` exactly, for any ``ny`` → ``Mp = fy·b·h²/4``;
* ``Σ A·y² = (b·h³/12)·(1 − 1/ny²)`` exactly.

openseespy-dependent cases skip when no backend is installed.
"""
from __future__ import annotations

import pytest

from apeGmsh.sections import (
    MomentCurvatureError,
    SectionDocument,
    moment_curvature,
)
from apeGmsh.sections._mc import backend_available


E = 200e3
FY = 345.0
B, H = 200.0, 400.0

def requires_backend(fn):
    """Gate a test that drives an OpenSees backend **in process**.

    Two mechanisms on purpose. ``live`` is the house marker the curated
    CI suite deselects, so native OpenSees never enters the shared
    process even if a backend is installed there; the ``skipif`` is the
    honest per-machine check for everywhere else.
    """
    fn = pytest.mark.live(fn)
    return pytest.mark.skipif(
        not backend_available(), reason="no OpenSees backend installed",
    )(fn)


def _points_doc() -> SectionDocument:
    """Four elastic point fibers at (±100, ±50) — Σ A r² is exact by
    construction, with a different value on each axis."""
    doc = SectionDocument.new(name="pts", kind="fiber")
    doc.set_material("st", uniaxial=("ElasticMaterial", {"E": E}))
    for y in (-100.0, 100.0):
        for z in (-50.0, 50.0):
            doc.add_point(material="st", y=y, z=z, area=100.0)
    doc.set_GJ(1.0e12)
    return doc


def _patch_doc(*, ny: int = 40, elastic: bool = False) -> SectionDocument:
    doc = SectionDocument.new(name="pp", kind="fiber")
    spec = (
        ("ElasticMaterial", {"E": E}) if elastic
        else ("ElasticPP", {"E": E, "epsyP": FY / E})
    )
    doc.set_material("st", uniaxial=spec)
    doc.add_patch_rect(
        material="st", ny=ny, nz=2,
        yI=-H / 2, zI=-B / 2, yJ=H / 2, zJ=B / 2,
    )
    doc.set_GJ(1.0e12)
    return doc


# ─────────────────────────────────────────────────────────────────────
# gate — elastic slope == exact fiber sum, both axes
# ─────────────────────────────────────────────────────────────────────

@requires_backend
def test_elastic_slope_both_axes_exact():
    """``EI0`` equals ``Σ E·A·r²`` about each axis. The two values differ
    by 4× here, so an axis swap cannot pass both."""
    doc = _points_doc()
    for axis, arm in (("z", 100.0), ("y", 50.0)):
        curve = moment_curvature(
            doc, axis=axis, kappa_max=1e-6, n_steps=4,
        )
        assert curve.complete
        assert curve.EI0 == pytest.approx(
            E * 4 * 100.0 * arm ** 2, rel=1e-9
        )


@requires_backend
def test_elastic_slope_matches_patch_midpoint_sum():
    """A rect patch integrates by the midpoint rule, so its discrete
    ``I`` is ``(b·h³/12)·(1 − 1/ny²)`` — not the continuum value. The
    harness must reproduce the discretization it was handed, exactly."""
    ny = 40
    curve = moment_curvature(
        _patch_doc(ny=ny, elastic=True), kappa_max=1e-7, n_steps=2,
    )
    assert curve.EI0 == pytest.approx(
        E * B * H ** 3 / 12.0 * (1.0 - 1.0 / ny ** 2), rel=1e-9
    )


@requires_backend
def test_negative_kappa_mirrors_the_elastic_branch():
    """Signed input: a negative κ max walks the third quadrant and the
    slope is unchanged (an unsigned harness would return |M|)."""
    doc = _points_doc()
    up = moment_curvature(doc, kappa_max=1e-6, n_steps=4)
    down = moment_curvature(doc, kappa_max=-1e-6, n_steps=4)
    assert down.curvature[-1] == pytest.approx(-up.curvature[-1])
    assert down.moment[-1] == pytest.approx(-up.moment[-1])
    assert down.EI0 == pytest.approx(up.EI0, rel=1e-9)


# ─────────────────────────────────────────────────────────────────────
# keystone — ElasticPP plateau vs closed-form Mp, both signs
# ─────────────────────────────────────────────────────────────────────

@requires_backend
def test_elasticpp_plateau_vs_mp_both_signs():
    """Pushed to 20·κ_y the section is plastic but for a thin elastic
    core, so ``|M| → fy·b·h²/4`` from below (the residual core deficit is
    ~(1/3)(κ_y/κ)² ≈ 0.08 %)."""
    doc = _patch_doc()
    Mp = FY * B * H ** 2 / 4.0
    kappa_y = 2.0 * FY / (E * H)
    for sign in (+1.0, -1.0):
        curve = moment_curvature(
            doc, kappa_max=sign * 20.0 * kappa_y, n_steps=40,
        )
        assert curve.complete
        assert curve.M_max == pytest.approx(sign * Mp, rel=1e-2)
        # sign-carrying, not magnitude
        assert curve.M_max * sign > 0.0


@requires_backend
def test_axial_compression_shifts_the_plateau():
    """A constant axial pre-load is held through the push (``loadConst``)
    and reduces the plastic moment — the axial force eats plastic
    capacity. Compression is negative, per OpenSees."""
    doc = _patch_doc()
    kappa_y = 2.0 * FY / (E * H)
    Np = FY * B * H          # squash load
    free = moment_curvature(doc, kappa_max=20.0 * kappa_y, n_steps=40)
    loaded = moment_curvature(
        doc, kappa_max=20.0 * kappa_y, n_steps=40, axial=-0.3 * Np,
    )
    assert loaded.axial == pytest.approx(-0.3 * Np)
    assert abs(loaded.M_max) < abs(free.M_max)
    # closed form for a rectangle at n = N/Np: Mp(n) = Mp·(1 − n²)
    assert loaded.M_max == pytest.approx(
        free.M_max * (1.0 - 0.3 ** 2), rel=2e-2
    )


@requires_backend
def test_partial_curve_when_the_section_goes_singular():
    """Every fiber at the same |y| plastifies at once, leaving no
    stiffness for the free axial DOF. The curve stops and says so — a
    partial result, not an exception."""
    doc = SectionDocument.new(name="degenerate", kind="fiber")
    doc.set_material("st", uniaxial=("ElasticPP", {"E": E, "epsyP": FY / E}))
    for z in (-50.0, 50.0):
        for y in (-100.0, 100.0):
            doc.add_point(material="st", y=y, z=z, area=100.0)
    kappa_y = FY / E / 100.0
    curve = moment_curvature(doc, kappa_max=5.0 * kappa_y, n_steps=10)
    assert curve.complete is False
    assert len(curve.curvature) >= 2          # the elastic branch survives
    assert curve.EI0 == pytest.approx(E * 4 * 100.0 * 100.0 ** 2, rel=1e-9)


# ─────────────────────────────────────────────────────────────────────
# handoff parity — the M–κ section IS the deck's section
# ─────────────────────────────────────────────────────────────────────

@requires_backend
def test_mc_lowers_the_same_items_as_the_bridge_handoff():
    """The harness and ``to_section`` share
    ``typed_fiber_items``, so a template expands to the same fibers on
    both paths — checked through the recipe both consume."""
    from apeGmsh.sections._document import typed_fiber_items

    doc = SectionDocument.new(name="col", kind="fiber")
    doc.set_material("c", uniaxial=("ElasticMaterial", {"E": 25e3}))
    doc.set_material("s", uniaxial=("ElasticMaterial", {"E": E}))
    doc.add_template(
        "rc_rect_column", materials={"concrete": "c", "bars": "s"},
        b=400.0, h=400.0, cover=50.0, bars_x=3, bars_y=3, bar_area=510.0,
    )
    recipe = doc.build()
    mats = {"c": object(), "s": object()}
    patches, layers, points = typed_fiber_items(recipe, mats)
    assert len(patches) == len(recipe.patches)
    assert len(layers) == len(recipe.layers)
    assert len(points) == len(recipe.points)

    curve = moment_curvature(doc, kappa_max=1e-6, n_steps=2)
    # elastic template: EI0 is the exact fiber sum over the SAME items
    expected = 0.0
    E_of = {"c": 25e3, "s": E}
    for p in recipe.patches:
        ny, nz = p["ny"], p["nz"]
        cell = abs((p["yJ"] - p["yI"]) * (p["zJ"] - p["zI"])) / (ny * nz)
        for i in range(ny):
            y = min(p["yI"], p["yJ"]) + (i + 0.5) * abs(
                p["yJ"] - p["yI"]
            ) / ny
            expected += E_of[p["material"]] * cell * nz * y ** 2
    for la in recipe.layers:
        n = la["n_bars"]
        for i in range(n):
            t = 0.0 if n == 1 else i / (n - 1)
            y = la["yI"] + t * (la["yJ"] - la["yI"])
            expected += E_of[la["material"]] * la["area"] * y ** 2
    for pt in recipe.points:
        expected += E_of[pt["material"]] * pt["area"] * pt["y"] ** 2
    assert curve.EI0 == pytest.approx(expected, rel=1e-9)


# ─────────────────────────────────────────────────────────────────────
# guards + fail-soft
# ─────────────────────────────────────────────────────────────────────

def test_continuum_document_is_refused_with_guidance():
    doc = SectionDocument.new(name="cont")
    with pytest.raises(MomentCurvatureError, match="fiber-lane operation"):
        moment_curvature(doc, kappa_max=1e-6)


def test_bad_arguments_fail_loud():
    doc = _points_doc()
    with pytest.raises(MomentCurvatureError, match="axis must be"):
        moment_curvature(doc, axis="x", kappa_max=1e-6)  # type: ignore[arg-type]
    with pytest.raises(MomentCurvatureError, match="non-zero"):
        moment_curvature(doc, kappa_max=0.0)
    with pytest.raises(MomentCurvatureError, match="n_steps"):
        moment_curvature(doc, kappa_max=1e-6, n_steps=0)


def test_material_without_uniaxial_spec_fails_loud():
    doc = SectionDocument.new(name="nospec", kind="fiber")
    doc.set_material("m", E=1.0, nu=0.3)      # continuum role only
    doc.add_point(material="m", y=1.0, z=0.0, area=1.0)
    with pytest.raises(MomentCurvatureError, match="no uniaxial spec"):
        moment_curvature(doc, kappa_max=1e-6)


def test_unknown_uniaxial_type_fails_loud():
    doc = SectionDocument.new(name="bogus", kind="fiber")
    doc.set_material("m", uniaxial=("NotAMaterial", {}))
    doc.add_point(material="m", y=1.0, z=0.0, area=1.0)
    with pytest.raises(MomentCurvatureError, match="no typed uniaxial"):
        moment_curvature(doc, kappa_max=1e-6)


def test_backend_probe_rejects_the_tests_opensees_package(monkeypatch):
    """``tests/opensees/`` registers as a top-level ``opensees`` when
    pytest imports in importlib mode — the same impostor
    ``_resolve_ops`` guards against. The probe must not claim a backend
    it cannot import: a real one is an extension module, never a
    package.
    """
    import importlib.util

    from apeGmsh.sections import _mc

    real = importlib.util.find_spec

    def _fake(name, *a, **k):
        if name == "openseespy":
            return None
        if name == "opensees":
            return importlib.util.spec_from_file_location(
                "opensees", __file__,
                submodule_search_locations=[],   # marks it a package
            )
        return real(name, *a, **k)

    monkeypatch.setattr(importlib.util, "find_spec", _fake)
    assert _mc.backend_available() is False


def test_backend_absent_raises_guided_import_error(monkeypatch):
    """openseespy missing → ``ImportError`` naming the two ways to get
    one. Nothing else in ``apeGmsh.sections`` needs a backend."""
    import apeGmsh.opensees.emitter.live as live

    def _boom() -> None:
        raise ImportError("no backend")

    monkeypatch.setattr(live, "_get_ops", _boom)
    with pytest.raises(ImportError, match="pip install openseespy"):
        moment_curvature(_points_doc(), kappa_max=1e-6)


def test_EI0_needs_a_converged_step():
    from apeGmsh.sections import MomentCurvature

    empty = MomentCurvature(
        axis="z", curvature=(0.0,), moment=(0.0,), axial=0.0,
        complete=False,
    )
    assert empty.M_max == 0.0
    with pytest.raises(MomentCurvatureError, match="converged step"):
        _ = empty.EI0
