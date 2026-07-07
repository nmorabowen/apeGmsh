"""Compute-on-read wiring for derived scalars through the Results composite.

The pure math is locked in ``test_derived_scalars.py``; here we verify the
``results.elements.gauss`` path — that derived names surface in
``available_components()`` and that ``get(component=...)`` assembles the
stored tensor columns and returns a correct synthesized slab.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from apeGmsh.results import Results
from apeGmsh.results.writers import NativeWriter

from tests.conftest import _open_model_from_h5


_STRESS_SUFFIXES = ("xx", "yy", "zz", "xy", "yz", "xz")


def _write(path: Path, *, comps: dict[str, np.ndarray], time: np.ndarray,
           elem_idx: np.ndarray, nat: np.ndarray) -> None:
    with NativeWriter(path) as w:
        w.open(source_type="domain_capture")
        sid = w.begin_stage(name="s", kind="transient", time=time)
        w.write_gauss_group(
            sid, "partition_0", "group_0",
            class_tag=4, int_rule=1,
            element_index=elem_idx, natural_coords=nat,
            components=comps,
        )
        w.end_stage()


def _make_3d_stress(tmp_path: Path) -> Path:
    """Two elements, one GP, three steps — full 6-component stress tensor."""
    path = tmp_path / "stress3d.h5"
    time = np.array([0.0, 1.0, 2.0])
    elem_idx = np.array([10, 20], dtype=np.int64)
    nat = np.array([[0.0, 0.0, 0.0]], dtype=np.float64)   # 1 GP
    rng = np.random.default_rng(42)
    comps = {
        f"stress_{s}": rng.standard_normal((3, 2, 1))
        for s in _STRESS_SUFFIXES
    }
    _write(path, comps=comps, time=time, elem_idx=elem_idx, nat=nat)
    return path


def test_derived_surface_in_available_components(tmp_path: Path) -> None:
    path = _make_3d_stress(tmp_path)
    with Results.from_native(path, model=_open_model_from_h5(path)) as r:
        avail = r.elements.gauss.available_components()
        # raw stays; derived stress scalars are advertised
        assert "stress_xx" in avail
        assert "von_mises_stress" in avail
        assert "principal_stress_1" in avail
        assert "tresca_stress" in avail
        assert "mean_stress" in avail
        # no strain stored → no strain-derived advertised
        assert "von_mises_strain" not in avail


def test_von_mises_matches_base_columns(tmp_path: Path) -> None:
    path = _make_3d_stress(tmp_path)
    with Results.from_native(path, model=_open_model_from_h5(path)) as r:
        g = r.elements.gauss
        base = {s: g.get(component=f"stress_{s}").values for s in _STRESS_SUFFIXES}
        sxx, syy, szz = base["xx"], base["yy"], base["zz"]
        sxy, syz, sxz = base["xy"], base["yz"], base["xz"]
        expect = np.sqrt(
            0.5 * ((sxx - syy) ** 2 + (syy - szz) ** 2 + (szz - sxx) ** 2)
            + 3.0 * (sxy**2 + syz**2 + sxz**2)
        )
        vm = g.get(component="von_mises_stress")
        assert vm.component == "von_mises_stress"
        assert vm.values.shape == sxx.shape          # (T, sum_GP) preserved
        np.testing.assert_array_equal(vm.element_index, [10, 20])
        np.testing.assert_allclose(vm.values, expect, rtol=1e-12)


def test_principal_ordering_through_composite(tmp_path: Path) -> None:
    path = _make_3d_stress(tmp_path)
    with Results.from_native(path, model=_open_model_from_h5(path)) as r:
        g = r.elements.gauss
        p1 = g.get(component="principal_stress_1").values
        p3 = g.get(component="principal_stress_3").values
        tresca = g.get(component="tresca_stress").values
        assert np.all(p1 >= p3 - 1e-12)
        np.testing.assert_allclose(tresca, p1 - p3, rtol=1e-12)


def test_missing_tensor_raises(tmp_path: Path) -> None:
    """Only stress stored → asking for a strain-derived scalar fails loud."""
    path = _make_3d_stress(tmp_path)
    with Results.from_native(path, model=_open_model_from_h5(path)) as r:
        with pytest.raises(ValueError, match="von_mises_strain"):
            r.elements.gauss.get(component="von_mises_strain")


def test_2d_plane_stress_default(tmp_path: Path) -> None:
    """In-plane-only storage → derived computed with out-of-plane = 0."""
    path = tmp_path / "stress2d.h5"
    time = np.array([0.0, 1.0])
    elem_idx = np.array([1], dtype=np.int64)
    nat = np.array([[0.0, 0.0]], dtype=np.float64)
    S = 4.0
    comps = {
        "stress_xx": np.full((2, 1, 1), S),
        "stress_yy": np.zeros((2, 1, 1)),
        "stress_xy": np.zeros((2, 1, 1)),
    }
    _write(path, comps=comps, time=time, elem_idx=elem_idx, nat=nat)
    with Results.from_native(path, model=_open_model_from_h5(path)) as r:
        vm = r.elements.gauss.get(component="von_mises_stress")
        np.testing.assert_allclose(vm.values, S)     # uniaxial → vM = S
        p1 = r.elements.gauss.get(component="principal_stress_1").values
        np.testing.assert_allclose(p1, S)


def test_shell_von_mises_through_composite(tmp_path: Path) -> None:
    """Shell resultants stored → von_mises_shell computed with thickness=."""
    path = tmp_path / "shell.h5"
    time = np.array([0.0, 1.0])
    elem_idx = np.array([7], dtype=np.int64)
    nat = np.array([[0.0, 0.0]], dtype=np.float64)
    N, t = 100.0, 0.25   # pure uniaxial membrane → σ = N/t
    comps = {
        "membrane_force_xx": np.full((2, 1, 1), N),
        "membrane_force_yy": np.zeros((2, 1, 1)),
        "membrane_force_xy": np.zeros((2, 1, 1)),
        "bending_moment_xx": np.zeros((2, 1, 1)),
        "bending_moment_yy": np.zeros((2, 1, 1)),
        "bending_moment_xy": np.zeros((2, 1, 1)),
    }
    _write(path, comps=comps, time=time, elem_idx=elem_idx, nat=nat)
    with Results.from_native(path, model=_open_model_from_h5(path)) as r:
        g = r.elements.gauss
        assert "von_mises_shell" in g.available_components()
        vm = g.get(component="von_mises_shell", thickness=t)
        assert vm.component == "von_mises_shell"
        np.testing.assert_allclose(vm.values, N / t)
        # thickness omitted → loud
        with pytest.raises(ValueError, match="thickness"):
            g.get(component="von_mises_shell")


def test_recorded_stress_zz_is_used_verbatim(tmp_path: Path) -> None:
    """A native file that stores stress_zz → the real σ_zz is used, not
    reconstructed (this is how a fork that records the material's σ33 flows
    through: self-describing native reader + auto-recovery skips the fill)."""
    path = tmp_path / "stress_with_zz.h5"
    time = np.array([0.0])
    elem_idx = np.array([1], dtype=np.int64)
    nat = np.array([[0.0, 0.0, 0.0]], dtype=np.float64)
    # Recorded σ_zz = 7 — deliberately NOT ν·(σxx+σyy) for any ν.
    comps = {
        "stress_xx": np.full((1, 1, 1), 10.0),
        "stress_yy": np.zeros((1, 1, 1)),
        "stress_xy": np.zeros((1, 1, 1)),
        "stress_zz": np.full((1, 1, 1), 7.0),
    }
    _write(path, comps=comps, time=time, elem_idx=elem_idx, nat=nat)
    with Results.from_native(path, model=_open_model_from_h5(path)) as r:
        g = r.elements.gauss
        # stress_zz surfaces as a component (self-describing reader).
        assert "stress_zz" in g.available_components()
        # principals of diag(10, 0, 7) → 10, 7, 0; the recorded 7 is a principal.
        assert float(g.get(component="principal_stress_2").values[0, 0]) == pytest.approx(7.0)
        # von Mises uses the real σ_zz, even under plane="auto" (default).
        vm = float(g.get(component="von_mises_stress").values[0, 0])
        expect = np.sqrt(0.5 * ((10 - 0) ** 2 + (0 - 7) ** 2 + (7 - 10) ** 2))
        assert vm == pytest.approx(expect)


def test_auto_plane_detect_recovers_per_element(tmp_path: Path, monkeypatch) -> None:
    """Default plane='auto' recovers σ_zz per element from the model map."""
    path = tmp_path / "stress2d.h5"
    time = np.array([0.0])
    elem_idx = np.array([1], dtype=np.int64)
    nat = np.array([[0.0, 0.0]], dtype=np.float64)
    comps = {
        "stress_xx": np.full((1, 1, 1), 10.0),
        "stress_yy": np.zeros((1, 1, 1)),
        "stress_xy": np.zeros((1, 1, 1)),
    }
    _write(path, comps=comps, time=time, elem_idx=elem_idx, nat=nat)
    with Results.from_native(path, model=_open_model_from_h5(path)) as r:
        from apeGmsh.results import _plane_recovery
        # Pretend the model says element 1 is plane-strain with ν=0.3.
        monkeypatch.setattr(
            _plane_recovery, "plane_recovery_map",
            lambda model: {1: ("PlaneStrain", 0.3)},
        )
        g = r.elements.gauss
        # auto (default) → σ_zz = 0.3·(10+0) = 3 → middle principal = 3.
        p2 = g.get(component="principal_stress_2").values
        np.testing.assert_allclose(p2, 3.0)
        # explicit plane=None disables recovery → σ_zz = 0 → middle = 0.
        p2_off = g.get(component="principal_stress_2", plane=None).values
        np.testing.assert_allclose(p2_off, 0.0, atol=1e-9)


def test_auto_plane_noop_without_model_records(tmp_path: Path) -> None:
    """A synthetic file with no element records → auto falls back to σ_zz=0."""
    path = tmp_path / "stress2d_plain.h5"
    time = np.array([0.0])
    comps = {
        "stress_xx": np.full((1, 1, 1), 10.0),
        "stress_yy": np.zeros((1, 1, 1)),
        "stress_xy": np.zeros((1, 1, 1)),
    }
    _write(path, comps=comps, time=time,
           elem_idx=np.array([1], dtype=np.int64),
           nat=np.array([[0.0, 0.0]], dtype=np.float64))
    with Results.from_native(path, model=_open_model_from_h5(path)) as r:
        # default auto, but the model resolves nothing → plane stress (σ_zz=0)
        p2 = r.elements.gauss.get(component="principal_stress_2").values
        np.testing.assert_allclose(p2, 0.0, atol=1e-9)
        vm = r.elements.gauss.get(component="von_mises_stress").values
        np.testing.assert_allclose(vm, 10.0)   # uniaxial


def test_plastic_strain_tensor_invariants(tmp_path: Path) -> None:
    """A native file storing plastic_strain_* → derived plastic invariants."""
    path = tmp_path / "pstrain3d.h5"
    time = np.array([0.0])
    elem_idx = np.array([1], dtype=np.int64)
    nat = np.array([[0.0, 0.0, 0.0]], dtype=np.float64)
    e = 0.03
    # Deviatoric (volume-preserving) plastic strain diag(e, -e/2, -e/2).
    comps = {
        "plastic_strain_xx": np.full((1, 1, 1), e),
        "plastic_strain_yy": np.full((1, 1, 1), -e / 2),
        "plastic_strain_zz": np.full((1, 1, 1), -e / 2),
        "plastic_strain_xy": np.zeros((1, 1, 1)),
        "plastic_strain_yz": np.zeros((1, 1, 1)),
        "plastic_strain_xz": np.zeros((1, 1, 1)),
    }
    _write(path, comps=comps, time=time, elem_idx=elem_idx, nat=nat)
    with Results.from_native(path, model=_open_model_from_h5(path)) as r:
        g = r.elements.gauss
        avail = g.available_components()
        assert "plastic_strain_xx" in avail
        assert "equivalent_plastic_strain_current" in avail
        assert "principal_plastic_strain_1" in avail
        np.testing.assert_allclose(
            g.get(component="equivalent_plastic_strain_current").values, e,
        )
        np.testing.assert_allclose(
            g.get(component="volumetric_plastic_strain").values, 0.0, atol=1e-12,
        )
        np.testing.assert_allclose(
            g.get(component="principal_plastic_strain_1").values, e,
        )


def test_plastic_strain_2d_no_elastic_recovery(tmp_path: Path, monkeypatch) -> None:
    """Plastic strain must NOT get the elastic ν out-of-plane recovery."""
    path = tmp_path / "pstrain2d.h5"
    comps = {
        "plastic_strain_xx": np.full((1, 1, 1), 0.02),
        "plastic_strain_yy": np.zeros((1, 1, 1)),
        "plastic_strain_xy": np.zeros((1, 1, 1)),
    }
    _write(path, comps=comps, time=np.array([0.0]),
           elem_idx=np.array([1], dtype=np.int64),
           nat=np.array([[0.0, 0.0]], dtype=np.float64))
    with Results.from_native(path, model=_open_model_from_h5(path)) as r:
        from apeGmsh.results import _plane_recovery
        # Even if the model would resolve a plane type + ν, plastic strain
        # must ignore it — the guard skips the elastic recovery entirely.
        monkeypatch.setattr(
            _plane_recovery, "plane_recovery_map",
            lambda model: {1: ("PlaneStrain", 0.3)},
        )
        g = r.elements.gauss
        # εᵖ_zz stays 0 → principals of diag(0.02, 0, 0) → 0.02, 0, 0.
        np.testing.assert_allclose(
            g.get(component="principal_plastic_strain_2").values, 0.0, atol=1e-12,
        )


def test_2d_plane_strain_kwarg_threads_through(tmp_path: Path) -> None:
    """plane='strain' + nu on get() recovers σ_zz = ν(σ_xx+σ_yy)."""
    path = tmp_path / "stress2d.h5"
    time = np.array([0.0, 1.0])
    elem_idx = np.array([1], dtype=np.int64)
    nat = np.array([[0.0, 0.0]], dtype=np.float64)
    S, nu = 10.0, 0.3
    comps = {
        "stress_xx": np.full((2, 1, 1), S),
        "stress_yy": np.zeros((2, 1, 1)),
        "stress_xy": np.zeros((2, 1, 1)),
    }
    _write(path, comps=comps, time=time, elem_idx=elem_idx, nat=nat)
    with Results.from_native(path, model=_open_model_from_h5(path)) as r:
        g = r.elements.gauss
        # σ_zz = 0.3·S becomes the middle principal (S ≥ 0.3S ≥ 0).
        p2 = g.get(component="principal_stress_2", plane="strain", nu=nu).values
        np.testing.assert_allclose(p2, nu * S)
        # plane stress default (no kwarg) leaves it 0.
        p2_ps = g.get(component="principal_stress_2").values
        np.testing.assert_allclose(p2_ps, 0.0, atol=1e-9)
