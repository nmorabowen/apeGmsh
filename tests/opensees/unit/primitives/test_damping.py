"""Unit tests for ``ops.damping.rayleigh`` (ADR 0053, D1 — global form).

RecordingEmitter + direct Tcl/Py emitter calls only — no openseespy, no
gmsh, no subprocess.  Run with::

    pytest tests/opensees/unit/primitives/test_damping.py -v
"""
from __future__ import annotations

from typing import cast
from unittest.mock import MagicMock

import pytest

from apeGmsh.opensees import apeSees
from apeGmsh.opensees.analysis.rayleigh import rayleigh_from_ratio
from apeGmsh.opensees.apesees import BuiltModel
from apeGmsh.opensees._internal.build import ModalDampingRecord, RayleighRecord
from apeGmsh.opensees.damping.damping import SecStif, Uniform
from apeGmsh.opensees.emitter.py import PyEmitter
from apeGmsh.opensees.emitter.recording import RecordingEmitter
from apeGmsh.opensees.emitter.tcl import TclEmitter


def _make_ops() -> apeSees:
    """apeSees with a stub FEMData — the damping namespace ignores it."""
    return apeSees(cast("object", MagicMock(name="FEMData")))  # type: ignore[arg-type]


def _built(*rayleigh_records: RayleighRecord) -> BuiltModel:
    """A minimal BuiltModel carrying only the rayleigh records under test."""
    return BuiltModel(
        primitives=(),
        tag_for={},
        ndm=3,
        ndf=3,
        fem=cast("object", MagicMock(name="FEMData")),  # type: ignore[arg-type]
        fix_records=(),
        mass_records=(),
        region_records=(),
        rayleigh_records=rayleigh_records,
    )


# --- recording the declaration ---------------------------------------------

class TestRayleighRecording:
    def test_raw_form_records_four_coefficients(self) -> None:
        ops = _make_ops()
        ops.damping.rayleigh(alpha_m=0.1, beta_k=0.01)
        (rec,) = ops._rayleigh_records
        assert (rec.alpha_m, rec.beta_k, rec.beta_k_init, rec.beta_k_comm) == (
            0.1, 0.01, 0.0, 0.0,
        )

    def test_raw_form_initial_and_committed_slots(self) -> None:
        ops = _make_ops()
        ops.damping.rayleigh(beta_k_init=0.02, beta_k_comm=0.03)
        (rec,) = ops._rayleigh_records
        assert (rec.beta_k_init, rec.beta_k_comm) == (0.02, 0.03)

    def test_ratio_form_matches_helper_and_defaults_to_initial(self) -> None:
        ops = _make_ops()
        ops.damping.rayleigh(ratio=0.05, f_i=1.0, f_j=10.0)
        (rec,) = ops._rayleigh_records
        exp = rayleigh_from_ratio(ratio=0.05, f_i=1.0, f_j=10.0)
        assert (rec.alpha_m, rec.beta_k, rec.beta_k_init, rec.beta_k_comm) == exp
        # default 'initial' → β in betaK0, not betaK
        assert rec.beta_k == 0.0
        assert rec.beta_k_init != 0.0

    def test_ratio_form_respects_stiffness_switch(self) -> None:
        ops = _make_ops()
        ops.damping.rayleigh(ratio=0.05, f_i=1.0, f_j=10.0, stiffness="current")
        (rec,) = ops._rayleigh_records
        assert rec.beta_k != 0.0
        assert rec.beta_k_init == 0.0


# --- fail-loud validation --------------------------------------------------

class TestRayleighValidation:
    def test_both_forms_raises(self) -> None:
        ops = _make_ops()
        with pytest.raises(ValueError, match="not both"):
            ops.damping.rayleigh(alpha_m=0.1, ratio=0.05, f_i=1.0, f_j=10.0)

    def test_neither_form_raises(self) -> None:
        ops = _make_ops()
        with pytest.raises(ValueError, match="supply either"):
            ops.damping.rayleigh()

    def test_incomplete_ratio_form_raises(self) -> None:
        ops = _make_ops()
        with pytest.raises(ValueError, match="needs all of"):
            ops.damping.rayleigh(ratio=0.05, f_i=1.0)  # missing f_j


# --- scope (on=) normalization (D2) ----------------------------------------

class TestRayleighScope:
    def test_default_is_global(self) -> None:
        ops = _make_ops()
        ops.damping.rayleigh(alpha_m=0.1)
        (rec,) = ops._rayleigh_records
        assert rec.on == ()

    def test_single_name_becomes_one_tuple(self) -> None:
        ops = _make_ops()
        ops.damping.rayleigh(alpha_m=0.1, on="Soil")
        (rec,) = ops._rayleigh_records
        assert rec.on == ("Soil",)

    def test_list_of_names_preserved(self) -> None:
        ops = _make_ops()
        ops.damping.rayleigh(alpha_m=0.1, on=["Soil", "Rock"])
        (rec,) = ops._rayleigh_records
        assert rec.on == ("Soil", "Rock")

    def test_empty_name_raises(self) -> None:
        ops = _make_ops()
        with pytest.raises(ValueError, match="non-empty"):
            ops.damping.rayleigh(alpha_m=0.1, on="")

    def test_non_string_name_raises(self) -> None:
        ops = _make_ops()
        with pytest.raises(ValueError, match="non-empty"):
            ops.damping.rayleigh(alpha_m=0.1, on=[123])  # type: ignore[list-item]


# --- damping objects (D3): primitives -------------------------------------

class TestDampingObjects:
    def test_uniform_emit(self) -> None:
        e = RecordingEmitter()
        Uniform(zeta=0.03, freq1=0.5, freq2=10.0)._emit(e, tag=5)
        assert e.calls == [("damping", ("Uniform", 5, 0.03, 0.5, 10.0), {})]

    def test_uniform_emit_with_time_window(self) -> None:
        e = RecordingEmitter()
        Uniform(
            zeta=0.03, freq1=0.5, freq2=10.0,
            activate_time=1.0, deactivate_time=20.0,
        )._emit(e, tag=5)
        assert e.calls == [(
            "damping",
            ("Uniform", 5, 0.03, 0.5, 10.0,
             "-activateTime", 1.0, "-deactivateTime", 20.0),
            {},
        )]

    def test_sec_stif_emit(self) -> None:
        e = RecordingEmitter()
        SecStif(beta=0.002)._emit(e, tag=7)
        assert e.calls == [("damping", ("SecStif", 7, 0.002), {})]

    def test_uniform_leaf_dependencies(self) -> None:
        assert Uniform(zeta=0.03, freq1=0.5, freq2=10.0).dependencies() == ()

    def test_uniform_rejects_inverted_band(self) -> None:
        with pytest.raises(ValueError, match="freq2 must be > freq1"):
            Uniform(zeta=0.03, freq1=10.0, freq2=0.5)

    def test_uniform_rejects_negative_zeta(self) -> None:
        with pytest.raises(ValueError, match="zeta must be"):
            Uniform(zeta=-0.01, freq1=0.5, freq2=10.0)


class TestDampingObjectNamespace:
    def test_uniform_registers_and_records_attach(self) -> None:
        ops = _make_ops()
        damp = ops.damping.uniform(
            ratio=0.03, freq_lower=0.5, freq_upper=10.0, on="Soil",
        )
        assert isinstance(damp, Uniform)
        assert damp in ops._primitives                # registered (tagged)
        (rec,) = ops._damping_attach_records
        assert rec.prim is damp and rec.on == ("Soil",)

    def test_on_list_records_all_targets(self) -> None:
        ops = _make_ops()
        ops.damping.uniform(
            ratio=0.03, freq_lower=0.5, freq_upper=10.0,
            on=["Soil", "Rock"],
        )
        (rec,) = ops._damping_attach_records
        assert rec.on == ("Soil", "Rock")

    def test_on_is_required(self) -> None:
        ops = _make_ops()
        with pytest.raises(ValueError, match="on= is required"):
            ops.damping.uniform(
                ratio=0.03, freq_lower=0.5, freq_upper=10.0, on=[],
            )

    def test_sec_stif_registers(self) -> None:
        ops = _make_ops()
        damp = ops.damping.sec_stif(beta=0.002, on="Soil")
        assert isinstance(damp, SecStif)
        assert damp in ops._primitives


# --- modal damping (D4) ----------------------------------------------------

class TestModalDamping:
    def test_scalar_records_uniform_factor(self) -> None:
        ops = _make_ops()
        ops.damping.modal(0.05, modes=10)
        (rec,) = ops._modal_damping_records
        assert rec.factors == (0.05,)
        assert rec.modes == 10

    def test_sequence_records_per_mode(self) -> None:
        ops = _make_ops()
        ops.damping.modal([0.02, 0.03, 0.05], modes=3)
        (rec,) = ops._modal_damping_records
        assert rec.factors == (0.02, 0.03, 0.05)

    def test_sequence_length_must_equal_modes(self) -> None:
        ops = _make_ops()
        with pytest.raises(ValueError, match="exactly modes=3"):
            ops.damping.modal([0.02, 0.03], modes=3)

    def test_modes_must_be_positive(self) -> None:
        ops = _make_ops()
        with pytest.raises(ValueError, match="modes must be >= 1"):
            ops.damping.modal(0.05, modes=0)

    def test_no_modal_q_method(self) -> None:
        # modalDampingQ is a verified upstream anti-damping bug — the bridge
        # must not expose it (ADR 0053).
        ops = _make_ops()
        assert not hasattr(ops.damping, "modal_q")

    def test_emit_bundles_eigen_then_modal_damping(self) -> None:
        e = RecordingEmitter()
        bm = BuiltModel(
            primitives=(), tag_for={}, ndm=3, ndf=3,
            fem=cast("object", MagicMock(name="FEMData")),  # type: ignore[arg-type]
            fix_records=(), mass_records=(), region_records=(),
            modal_damping_records=(
                ModalDampingRecord(
                    factors=(0.05,), modes=10, solver="-genBandArpack",
                ),
            ),
        )
        bm._emit_modal_damping(e)
        assert e.calls == [
            ("eigen", (), {"num_modes": 10, "solver": "-genBandArpack"}),
            ("modal_damping", (0.05,), {}),
        ]


# --- emit ------------------------------------------------------------------

class TestRayleighEmit:
    def test_build_carries_records_into_built_model(self) -> None:
        ops = _make_ops()
        ops.damping.rayleigh(alpha_m=0.1, beta_k=0.01)
        # build() copies the declaration list onto the BuiltModel snapshot.
        assert tuple(ops._rayleigh_records) == (
            RayleighRecord(0.1, 0.01, 0.0, 0.0),
        )

    def test_emit_calls_emitter_rayleigh(self) -> None:
        e = RecordingEmitter()
        _built(RayleighRecord(0.1, 0.01, 0.0, 0.0))._emit_rayleigh(e)
        assert e.calls == [("rayleigh", (0.1, 0.01, 0.0, 0.0), {})]

    def test_emit_preserves_declaration_order(self) -> None:
        e = RecordingEmitter()
        _built(
            RayleighRecord(0.1, 0.0, 0.0, 0.0),
            RayleighRecord(0.2, 0.0, 0.0, 0.0),
        )._emit_rayleigh(e)
        assert [c[1][0] for c in e.calls] == [0.1, 0.2]

    def test_no_records_emits_nothing(self) -> None:
        e = RecordingEmitter()
        _built()._emit_rayleigh(e)
        assert e.calls == []

    def test_tcl_line_format(self) -> None:
        e = TclEmitter()
        e.rayleigh(0.1, 0.01, 0.0, 0.0)
        lines = [ln for ln in e.lines() if "rayleigh" in ln]
        assert lines == ["rayleigh 0.1 0.01 0.0 0.0"]

    def test_py_line_format(self) -> None:
        e = PyEmitter()
        e.rayleigh(0.1, 0.01, 0.0, 0.0)
        lines = [ln for ln in e.lines() if "rayleigh" in ln]
        assert len(lines) == 1
        assert "rayleigh(" in lines[0]
        assert "0.1" in lines[0] and "0.01" in lines[0]
