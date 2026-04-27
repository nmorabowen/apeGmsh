"""Phase 5 — Tcl / Python recorder command emission.

Pure unit tests against synthetic resolved specs — no session
required. Snapshot-style assertions verify the emitted command
strings.
"""
from __future__ import annotations

import warnings

import numpy as np
import pytest

from apeGmsh.solvers._recorder_specs import (
    ResolvedRecorderRecord,
    ResolvedRecorderSpec,
)


def _make_spec(*records: ResolvedRecorderRecord) -> ResolvedRecorderSpec:
    return ResolvedRecorderSpec(
        fem_snapshot_id="testhash",
        records=tuple(records),
    )


# =====================================================================
# Nodal emission — single ops_type
# =====================================================================

def test_node_displacement_only_3d() -> None:
    spec = _make_spec(ResolvedRecorderRecord(
        category="nodes", name="top_disp",
        components=("displacement_x", "displacement_y", "displacement_z"),
        dt=None, n_steps=None,
        node_ids=np.array([1, 2, 3]),
    ))
    [line] = spec.to_tcl_commands()
    assert line.startswith("recorder Node ")
    assert "-file top_disp_disp.out" in line
    assert "-time" in line
    assert "-node 1 2 3" in line
    assert "-dof 1 2 3" in line
    assert line.rstrip().endswith("disp  ;# top_disp disp")


def test_node_displacement_with_dt() -> None:
    spec = _make_spec(ResolvedRecorderRecord(
        category="nodes", name="r",
        components=("displacement_x",),
        dt=0.01, n_steps=None,
        node_ids=np.array([1]),
    ))
    [line] = spec.to_tcl_commands()
    assert "-dT 0.01" in line


def test_node_displacement_xml_format() -> None:
    spec = _make_spec(ResolvedRecorderRecord(
        category="nodes", name="r",
        components=("displacement_x",),
        dt=None, n_steps=None,
        node_ids=np.array([1]),
    ))
    [line] = spec.to_tcl_commands(file_format="xml")
    assert "-xml r_disp.xml" in line
    assert "-file" not in line


def test_node_output_dir_prefix() -> None:
    spec = _make_spec(ResolvedRecorderRecord(
        category="nodes", name="r",
        components=("displacement_x",),
        dt=None, n_steps=None,
        node_ids=np.array([1]),
    ))
    [line] = spec.to_tcl_commands(output_dir="out/")
    assert "-file out/r_disp.out" in line


def test_output_dir_no_trailing_slash() -> None:
    spec = _make_spec(ResolvedRecorderRecord(
        category="nodes", name="r",
        components=("displacement_x",),
        dt=None, n_steps=None,
        node_ids=np.array([1]),
    ))
    [line] = spec.to_tcl_commands(output_dir="results")
    # Emitter inserts the separator
    assert "-file results/r_disp.out" in line


# =====================================================================
# Nodal emission — multiple ops_types per record
# =====================================================================

def test_node_disp_plus_velocity_emits_two_commands() -> None:
    spec = _make_spec(ResolvedRecorderRecord(
        category="nodes", name="kinematics",
        components=("displacement_x", "velocity_x"),
        dt=None, n_steps=None,
        node_ids=np.array([1, 2]),
    ))
    lines = spec.to_tcl_commands()
    assert len(lines) == 2
    cmds = sorted(lines)
    assert any("kinematics_disp.out" in l and "disp" in l for l in cmds)
    assert any("kinematics_vel.out" in l and "vel" in l for l in cmds)


def test_node_translational_plus_rotational_one_command() -> None:
    """Translation and rotation are both ``disp`` recorder type — one cmd."""
    spec = _make_spec(ResolvedRecorderRecord(
        category="nodes", name="full",
        components=(
            "displacement_x", "displacement_y", "displacement_z",
            "rotation_x", "rotation_y", "rotation_z",
        ),
        dt=None, n_steps=None,
        node_ids=np.array([1]),
    ))
    [line] = spec.to_tcl_commands()
    assert "-dof 1 2 3 4 5 6" in line
    assert line.rstrip().endswith("disp  ;# full disp")


def test_node_dofs_sorted() -> None:
    """Out-of-order canonical components emit sorted DOFs."""
    spec = _make_spec(ResolvedRecorderRecord(
        category="nodes", name="r",
        components=("displacement_z", "displacement_x"),
        dt=None, n_steps=None,
        node_ids=np.array([1]),
    ))
    [line] = spec.to_tcl_commands()
    assert "-dof 1 3" in line


# =====================================================================
# Reaction / unbalance / pressure
# =====================================================================

def test_reaction_force_emits_reaction_recorder() -> None:
    spec = _make_spec(ResolvedRecorderRecord(
        category="nodes", name="r",
        components=("reaction_force_x", "reaction_force_y", "reaction_force_z"),
        dt=None, n_steps=None,
        node_ids=np.array([1]),
    ))
    [line] = spec.to_tcl_commands()
    assert line.rstrip().endswith("reaction  ;# r reaction")


def test_force_emits_unbalance_recorder() -> None:
    spec = _make_spec(ResolvedRecorderRecord(
        category="nodes", name="r",
        components=("force_x",),
        dt=None, n_steps=None,
        node_ids=np.array([1]),
    ))
    [line] = spec.to_tcl_commands()
    assert "unbalance" in line


def test_pore_pressure_emits_pressure_recorder() -> None:
    spec = _make_spec(ResolvedRecorderRecord(
        category="nodes", name="r",
        components=("pore_pressure",),
        dt=None, n_steps=None,
        node_ids=np.array([1]),
    ))
    [line] = spec.to_tcl_commands()
    assert "pressure" in line
    # Default pressure DOF in 3D u-p is 4
    assert "-dof 4" in line


# =====================================================================
# Element-level emission
# =====================================================================

def test_gauss_emits_element_stress_recorder() -> None:
    spec = _make_spec(ResolvedRecorderRecord(
        category="gauss", name="body",
        components=("stress_xx",),
        dt=0.01, n_steps=None,
        element_ids=np.array([10, 20]),
    ))
    [line] = spec.to_tcl_commands()
    assert line.startswith("recorder Element ")
    assert "-ele 10 20" in line
    assert "-dT 0.01" in line
    # OpenSees solid elements register the keyword "stresses" (plural)
    # in setResponse — see _recorder_emit._ELEMENT_CATEGORY_RESPONSE.
    assert line.rstrip().endswith("stresses  ;# body gauss")


def test_elements_emits_globalForce() -> None:
    spec = _make_spec(ResolvedRecorderRecord(
        category="elements", name="r",
        components=("nodal_resisting_force_x",),
        dt=None, n_steps=None,
        element_ids=np.array([1]),
    ))
    [line] = spec.to_tcl_commands()
    assert "globalForce" in line


def test_line_stations_emits_section_force() -> None:
    spec = _make_spec(ResolvedRecorderRecord(
        category="line_stations", name="r",
        components=("axial_force",),
        dt=None, n_steps=None,
        element_ids=np.array([1]),
    ))
    [line] = spec.to_tcl_commands()
    assert "section force" in line


# =====================================================================
# Deferred categories — emit a TODO comment
# =====================================================================

@pytest.mark.parametrize("category", ["fibers", "layers", "modal"])
def test_deferred_category_emits_todo(category: str) -> None:
    rec = ResolvedRecorderRecord(
        category=category, name="r",
        components=("fiber_stress",) if category in ("fibers",) else (
            ("stress_xx",) if category == "layers" else ()
        ),
        dt=None, n_steps=None,
        element_ids=(np.array([1]) if category != "modal" else None),
        n_modes=5 if category == "modal" else None,
    )
    spec = _make_spec(rec)
    [line] = spec.to_tcl_commands()
    assert "TODO Phase 5+" in line
    assert category in line


# =====================================================================
# Cadence warnings
# =====================================================================

def test_n_steps_warning_in_tcl_emit() -> None:
    spec = _make_spec(ResolvedRecorderRecord(
        category="nodes", name="r",
        components=("displacement_x",),
        dt=None, n_steps=5,
        node_ids=np.array([1]),
    ))
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        spec.to_tcl_commands()
    assert any("n_steps" in str(w.message) for w in caught)


# =====================================================================
# Python emission
# =====================================================================

def test_python_node_displacement() -> None:
    spec = _make_spec(ResolvedRecorderRecord(
        category="nodes", name="r",
        components=("displacement_x",),
        dt=0.01, n_steps=None,
        node_ids=np.array([1, 2]),
    ))
    [line] = spec.to_python_commands()
    assert line.startswith("ops.recorder(")
    assert "'Node'" in line
    assert "'-file'" in line and "'r_disp.out'" in line
    assert "'-time'" in line
    assert "'-dT', 0.01" in line
    assert "'-node', 1, 2" in line
    assert "'-dof', 1" in line
    assert "'disp'" in line


def test_python_element() -> None:
    spec = _make_spec(ResolvedRecorderRecord(
        category="gauss", name="r",
        components=("stress_xx",),
        dt=None, n_steps=None,
        element_ids=np.array([100]),
    ))
    [line] = spec.to_python_commands()
    assert "'Element'" in line
    assert "'-ele', 100" in line
    assert "'stresses'" in line


# =====================================================================
# Empty record / empty IDs short-circuit
# =====================================================================

def test_empty_node_ids_emits_nothing() -> None:
    spec = _make_spec(ResolvedRecorderRecord(
        category="nodes", name="r",
        components=("displacement_x",),
        dt=None, n_steps=None,
        node_ids=np.array([], dtype=np.int64),
    ))
    assert spec.to_tcl_commands() == []


def test_empty_element_ids_emits_nothing() -> None:
    spec = _make_spec(ResolvedRecorderRecord(
        category="gauss", name="r",
        components=("stress_xx",),
        dt=None, n_steps=None,
        element_ids=np.array([], dtype=np.int64),
    ))
    assert spec.to_tcl_commands() == []
