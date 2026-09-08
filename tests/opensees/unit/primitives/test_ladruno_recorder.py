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
# Region filter (-R) — inherited from FilterableRecorder (ADR 0064)
# ---------------------------------------------------------------------------

class TestLadrunoFilterValidation:
    def test_nodes_and_nodes_pg_mutex_raises(self) -> None:
        with pytest.raises(
            ValueError, match="supply only one of nodes= or nodes_pg="
        ):
            Ladruno(
                file="run.ladruno",
                nodal_responses=("displacement",),
                nodes=(1, 2),
                nodes_pg="Targets",
            )

    def test_elements_and_elements_pg_mutex_raises(self) -> None:
        with pytest.raises(
            ValueError, match="supply only one of elements= or elements_pg="
        ):
            Ladruno(
                file="run.ladruno",
                elem_responses=("stresses",),
                elements=(1,),
                elements_pg="Cols",
            )

    def test_node_only_filter_with_elem_responses_raises(self) -> None:
        # Message must carry the recorder kind so the user knows which
        # recorder rejected the combo.
        with pytest.raises(ValueError, match="Ladruno: node-only filter"):
            Ladruno(
                file="run.ladruno",
                nodal_responses=("displacement",),
                elem_responses=("stresses",),
                nodes_pg="Targets",
            )

    def test_element_only_filter_with_nodal_responses_raises(self) -> None:
        with pytest.raises(ValueError, match="Ladruno: element-only filter"):
            Ladruno(
                file="run.ladruno",
                nodal_responses=("displacement",),
                elem_responses=("stresses",),
                elements_pg="Cols",
            )

    def test_nodes_pg_alone_is_legal(self) -> None:
        r = Ladruno(
            file="run.ladruno",
            nodal_responses=("displacement",),
            nodes_pg="Targets",
        )
        assert r.nodes_pg == "Targets"
        assert r.has_filter() is True

    def test_whole_model_has_no_filter(self) -> None:
        r = Ladruno(file="run.ladruno", nodal_responses=("displacement",))
        assert r.has_filter() is False
        assert r._region_tag is None

    def test_filter_with_energy_is_legal(self) -> None:
        # ADR 0064 §4 (shipped): a region filter + energy=True records
        # per-region energy over the same region (run-verified on fork).
        r = Ladruno(
            file="run.ladruno",
            nodal_responses=("displacement",),
            nodes_pg="Targets",
            energy=True,
        )
        assert r.has_filter() is True
        assert r.energy is True

    def test_energy_without_filter_is_legal(self) -> None:
        r = Ladruno(file="run.ladruno", energy=True)
        assert r.energy is True
        assert r.has_filter() is False

    def test_energy_pg_alone_is_legal(self) -> None:
        # energy_pg implies energy recording — no other channel needed.
        r = Ladruno(file="run.ladruno", energy_pg="Core")
        assert r.energy_pg == "Core"
        assert r.has_filter() is False  # energy_pg is NOT a value filter

    def test_energy_pg_with_independent_value_filter_is_legal(self) -> None:
        # Decoupled: value channels over "Shell", energy over "Core".
        r = Ladruno(
            file="run.ladruno",
            elem_responses=("stresses",),
            elements_pg="Shell",
            energy_pg="Core",
        )
        assert r.elements_pg == "Shell"
        assert r.energy_pg == "Core"
        assert r.has_filter() is True


class TestLadrunoFilterEmit:
    def test_emit_with_region_tag_appends_R_flag(self) -> None:
        # _region_tag is set by the build pipeline after region emission;
        # _emit forwards it onto the ladruno command tail.
        r = Ladruno(
            file="run.ladruno",
            nodal_responses=("displacement",),
            nodes=(5, 6),
            _region_tag=42,
        )
        e = RecordingEmitter()
        r._emit(e, tag=1)
        args = e.calls[0][1]
        assert "-R" in args
        r_idx = args.index("-R")
        assert args[r_idx + 1] == 42

    def test_region_tag_emits_after_cadence(self) -> None:
        r = Ladruno(
            file="run.ladruno",
            nodal_responses=("displacement",),
            nodes=(5,),
            nsteps=10,
            _region_tag=7,
        )
        e = RecordingEmitter()
        r._emit(e, tag=1)
        assert e.calls[0][1] == (
            "ladruno", "run.ladruno", "-N", "displacement",
            "-T", "nsteps", 10, "-R", 7,
        )

    def test_no_region_tag_omits_R_flag(self) -> None:
        r = Ladruno(file="run.ladruno", nodal_responses=("displacement",))
        e = RecordingEmitter()
        r._emit(e, tag=1)
        assert "-R" not in e.calls[0][1]

    def test_filter_plus_energy_emits_per_region_energy(self) -> None:
        # ADR 0064 §4: filter + energy -> ``-R $tag ... -G energy $tag``;
        # the region tag trails ``-G energy`` (and -R precedes it).
        r = Ladruno(
            file="run.ladruno",
            elem_responses=("stresses",),
            elements=(5,),
            energy=True,
            _region_tag=9,
        )
        e = RecordingEmitter()
        r._emit(e, tag=1)
        assert e.calls[0][1] == (
            "ladruno", "run.ladruno", "-E", "stresses",
            "-R", 9, "-G", "energy", 9,
        )

    def test_whole_model_energy_omits_region_tag(self) -> None:
        # energy without a filter -> ``-G energy`` (no trailing tag).
        r = Ladruno(file="run.ladruno", energy=True)
        e = RecordingEmitter()
        r._emit(e, tag=1)
        assert e.calls[0][1] == ("ladruno", "run.ladruno", "-G", "energy")

    def test_decoupled_energy_region_emits_energy_tag(self) -> None:
        # energy_pg materialized into _energy_region_tags -> the energy
        # tag trails ``-G energy`` (decoupled from any -R value filter).
        r = Ladruno(
            file="run.ladruno",
            nodal_responses=("displacement",),
            _energy_region_tags=(7,),
        )
        e = RecordingEmitter()
        r._emit(e, tag=1)
        assert e.calls[0][1] == (
            "ladruno", "run.ladruno", "-N", "displacement", "-G", "energy", 7,
        )

    def test_decoupled_energy_takes_precedence_over_filter_tag(self) -> None:
        # When BOTH a value filter (_region_tag) and a decoupled energy
        # region (_energy_region_tags) are present, -R uses the filter tag
        # and -G energy uses the energy tag (not the filter tag).
        r = Ladruno(
            file="run.ladruno",
            elem_responses=("stresses",),
            elements=(5,),
            energy=True,
            _region_tag=3,
            _energy_region_tags=(8,),
        )
        e = RecordingEmitter()
        r._emit(e, tag=1)
        assert e.calls[0][1] == (
            "ladruno", "run.ladruno", "-E", "stresses",
            "-R", 3, "-G", "energy", 8,
        )

    def test_energy_pg_path_raises_not_implemented(self) -> None:
        # Defense-in-depth — an unmaterialised energy_pg= must raise.
        r = Ladruno(file="run.ladruno", energy_pg="Core")
        e = RecordingEmitter()
        with pytest.raises(NotImplementedError, match="build pipeline"):
            r._emit(e, tag=1)

    def test_pg_path_raises_not_implemented(self) -> None:
        # Defense-in-depth — calling _emit directly with a pg= selector
        # must raise; the build pipeline is the only legal driver.
        r = Ladruno(
            file="run.ladruno",
            nodal_responses=("displacement",),
            nodes_pg="Targets",
        )
        e = RecordingEmitter()
        with pytest.raises(NotImplementedError, match="build pipeline"):
            r._emit(e, tag=1)


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

    def test_namespace_filter_passthrough(self) -> None:
        ops = apeSees(cast("object", MagicMock(name="FEMData")))
        r = ops.recorder.Ladruno(
            file="run.ladruno",
            nodal_responses=("displacement",),
            nodes_pg="Targets",
        )
        assert r.nodes_pg == "Targets"
        assert r.has_filter() is True


# ---------------------------------------------------------------------------
# ADR 0105 — material-level tokens need the ``material.`` prefix
# ---------------------------------------------------------------------------


class TestMaterialLevelTokens:
    @pytest.mark.parametrize(
        "token",
        ["pstrain", "pstrains", "eqpstrain", "PStress", "J2Stress",
         "VolStrain", "J2Strain"],
    )
    def test_bare_material_token_is_refused_naming_the_prefix(
        self, token: str,
    ) -> None:
        with pytest.raises(ValueError, match=f"Use 'material.{token}'"):
            Ladruno(file="x.ladruno", elem_responses=("stress", token))

    def test_prefixed_and_element_level_tokens_pass(self) -> None:
        r = Ladruno(
            file="x.ladruno",
            elem_responses=("stress", "strain", "material.pstrain",
                            "material.eqpstrain", "material.stress"),
        )
        assert "material.pstrain" in r.elem_responses

    def test_mpco_shares_the_guard(self) -> None:
        from apeGmsh.opensees.recorder import MPCO

        with pytest.raises(ValueError, match="MPCO: elem_responses token"):
            MPCO(file="x.mpco", elem_responses=("pstrain",))
