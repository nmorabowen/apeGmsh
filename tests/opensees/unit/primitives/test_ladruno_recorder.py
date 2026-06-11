"""Unit tests for the ``Ladruno`` recorder primitive (recorder-plan L1).

``recorder ladruno`` is the fork-only canonical recorder. This covers the
whole-model **value channels** (``-N``/``-E``/``-T``), mirroring
:class:`apeGmsh.opensees.recorder.MPCO`, plus the whole-model **energy
balance** (``energy=True`` → ``-G energy``, always the LAST option — the
fork's ``-G`` parser cannot rewind past a following flag). The ``-R``
region filter and per-region energy are deferred (see the class docstring).

Coverage:
  * construction (defaults, explicit values)
  * validation (at least one response; dT/nsteps mutex)
  * ``_emit`` records the right call into a ``RecordingEmitter``
  * literal deck line on the Tcl + Py emitters
  * ``dependencies()`` returns ``()`` (recorders are leaves)
  * ``ops.recorder.Ladruno(...)`` constructs + registers on the bridge

Tests use ``RecordingEmitter`` / ``TclEmitter`` / ``PyEmitter`` only — no
openseespy, no gmsh, no subprocess. (The fork-build *run* gate is verified
separately once the venv carries a build that rejects unknown recorders.)
"""
from __future__ import annotations

from typing import cast
from unittest.mock import MagicMock

import pytest

from apeGmsh.opensees import apeSees
from apeGmsh.opensees.emitter.py import PyEmitter
from apeGmsh.opensees.emitter.recording import RecordingEmitter
from apeGmsh.opensees.emitter.tcl import TclEmitter
from apeGmsh.opensees.recorder import Ladruno


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------

class TestLadrunoConstruction:
    def test_minimal_nodal(self) -> None:
        r = Ladruno(file="run.ladruno", nodal_responses=("displacement",))
        assert r.file == "run.ladruno"
        assert r.nodal_responses == ("displacement",)
        assert r.elem_responses == ()
        assert r.dT is None
        assert r.nsteps is None

    def test_explicit_all_value_channels(self) -> None:
        r = Ladruno(
            file="run.ladruno",
            nodal_responses=("displacement", "reactionForce"),
            elem_responses=("stresses",),
            dT=0.02,
        )
        assert r.nodal_responses == ("displacement", "reactionForce")
        assert r.elem_responses == ("stresses",)
        assert r.dT == 0.02
        assert r.nsteps is None

    def test_repr_includes_class_name(self) -> None:
        r = Ladruno(file="run.ladruno", elem_responses=("stresses",))
        assert "Ladruno" in repr(r)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

class TestLadrunoValidation:
    def test_no_responses_raises(self) -> None:
        with pytest.raises(
            ValueError, match="at least one of nodal_responses"
        ):
            Ladruno(file="run.ladruno")

    def test_both_dt_and_nsteps_raises(self) -> None:
        with pytest.raises(ValueError, match="only one of dT or nsteps"):
            Ladruno(
                file="run.ladruno",
                nodal_responses=("displacement",),
                dT=0.01,
                nsteps=10,
            )

    def test_dependencies_empty(self) -> None:
        r = Ladruno(file="run.ladruno", nodal_responses=("displacement",))
        assert r.dependencies() == ()


# ---------------------------------------------------------------------------
# _emit — RecordingEmitter
# ---------------------------------------------------------------------------

class TestLadrunoEmit:
    def test_nodal_only(self) -> None:
        r = Ladruno(file="run.ladruno", nodal_responses=("displacement",))
        e = RecordingEmitter()
        r._emit(e, tag=1)
        assert e.calls == [
            ("recorder", ("ladruno", "run.ladruno", "-N", "displacement"), {})
        ]

    def test_elem_only(self) -> None:
        r = Ladruno(file="run.ladruno", elem_responses=("stresses",))
        e = RecordingEmitter()
        r._emit(e, tag=1)
        assert e.calls == [
            ("recorder", ("ladruno", "run.ladruno", "-E", "stresses"), {})
        ]

    def test_nodal_and_elem(self) -> None:
        r = Ladruno(
            file="run.ladruno",
            nodal_responses=("displacement", "reactionForce"),
            elem_responses=("stresses",),
        )
        e = RecordingEmitter()
        r._emit(e, tag=1)
        assert e.calls == [
            (
                "recorder",
                (
                    "ladruno", "run.ladruno",
                    "-N", "displacement", "reactionForce",
                    "-E", "stresses",
                ),
                {},
            )
        ]

    def test_dt_cadence(self) -> None:
        r = Ladruno(
            file="run.ladruno", nodal_responses=("displacement",), dT=0.05
        )
        e = RecordingEmitter()
        r._emit(e, tag=1)
        assert e.calls == [
            (
                "recorder",
                ("ladruno", "run.ladruno", "-N", "displacement", "-T", "dt", 0.05),
                {},
            )
        ]

    def test_nsteps_cadence(self) -> None:
        r = Ladruno(
            file="run.ladruno", nodal_responses=("displacement",), nsteps=10
        )
        e = RecordingEmitter()
        r._emit(e, tag=1)
        assert e.calls == [
            (
                "recorder",
                ("ladruno", "run.ladruno", "-N", "displacement", "-T", "nsteps", 10),
                {},
            )
        ]

    def test_no_cadence_omits_T_flag(self) -> None:
        r = Ladruno(file="run.ladruno", nodal_responses=("displacement",))
        e = RecordingEmitter()
        r._emit(e, tag=1)
        assert "-T" not in e.calls[0][1]


# ---------------------------------------------------------------------------
# Energy channel (-G energy)
# ---------------------------------------------------------------------------

class TestLadrunoEnergy:
    def test_default_off(self) -> None:
        r = Ladruno(file="run.ladruno", nodal_responses=("displacement",))
        e = RecordingEmitter()
        r._emit(e, tag=1)
        assert "-G" not in e.calls[0][1]

    def test_energy_only_is_valid(self) -> None:
        r = Ladruno(file="run.ladruno", energy=True)
        e = RecordingEmitter()
        r._emit(e, tag=1)
        assert e.calls == [
            ("recorder", ("ladruno", "run.ladruno", "-G", "energy"), {})
        ]

    def test_energy_emits_last_after_dt_cadence(self) -> None:
        """-G energy MUST trail every other option: the fork's -G parser
        eagerly consumes trailing region-tag ints and cannot rewind past
        a following flag ("-G energy -T dt 0.05" is a parse error)."""
        r = Ladruno(
            file="run.ladruno", nodal_responses=("displacement",),
            dT=0.05, energy=True,
        )
        e = RecordingEmitter()
        r._emit(e, tag=1)
        assert e.calls[0][1] == (
            "ladruno", "run.ladruno", "-N", "displacement",
            "-T", "dt", 0.05, "-G", "energy",
        )

    def test_energy_emits_last_after_nsteps_cadence(self) -> None:
        r = Ladruno(
            file="run.ladruno", elem_responses=("stresses",),
            nsteps=10, energy=True,
        )
        e = RecordingEmitter()
        r._emit(e, tag=1)
        assert e.calls[0][1][-2:] == ("-G", "energy")
        assert e.calls[0][1][-5:-2] == ("-T", "nsteps", 10)


# ---------------------------------------------------------------------------
# Literal deck line — Tcl + Py emitters
# ---------------------------------------------------------------------------

class TestLadrunoDeckEmission:
    def test_tcl_line_energy_last(self) -> None:
        r = Ladruno(
            file="run.ladruno", nodal_responses=("velocity",),
            nsteps=10, energy=True,
        )
        e = TclEmitter()
        r._emit(e, tag=1)
        assert (
            "recorder ladruno run.ladruno -N velocity -T nsteps 10 -G energy"
            in e.lines()
        )

    def test_tcl_line(self) -> None:
        r = Ladruno(
            file="run.ladruno",
            nodal_responses=("displacement",),
            elem_responses=("stresses",),
            dT=0.02,
        )
        e = TclEmitter()
        r._emit(e, tag=1)
        assert (
            "recorder ladruno run.ladruno -N displacement -E stresses -T dt 0.02"
            in e.lines()
        )

    def test_py_line(self) -> None:
        r = Ladruno(
            file="run.ladruno",
            nodal_responses=("displacement",),
            elem_responses=("stresses",),
            nsteps=5,
        )
        e = PyEmitter()
        r._emit(e, tag=1)
        assert (
            "ops.recorder('ladruno', 'run.ladruno', '-N', 'displacement', "
            "'-E', 'stresses', '-T', 'nsteps', 5)"
            in e.lines()
        )


# ---------------------------------------------------------------------------
# Namespace — ops.recorder.Ladruno(...)
# ---------------------------------------------------------------------------

class TestLadrunoNamespace:
    def test_constructs_and_registers(self) -> None:
        ops = apeSees(cast("object", MagicMock(name="FEMData")))
        r = ops.recorder.Ladruno(
            file="run.ladruno",
            nodal_responses=("displacement",),
            elem_responses=("stresses",),
        )
        assert isinstance(r, Ladruno)
        assert r.file == "run.ladruno"
        # The recorder is registered as a leaf primitive on the bridge.
        assert r in ops._primitives

    def test_namespace_validation_propagates(self) -> None:
        ops = apeSees(cast("object", MagicMock(name="FEMData")))
        with pytest.raises(
            ValueError, match="at least one of nodal_responses"
        ):
            ops.recorder.Ladruno(file="run.ladruno")

    def test_namespace_energy_passthrough(self) -> None:
        ops = apeSees(cast("object", MagicMock(name="FEMData")))
        r = ops.recorder.Ladruno(
            file="run.ladruno", nodal_responses=("velocity",), energy=True,
        )
        assert r.energy is True
