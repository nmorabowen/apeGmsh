"""Phase 8 — MPCO bridge emission (single ``recorder mpco`` line)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from apeGmsh.solvers._recorder_specs import (
    ResolvedRecorderRecord,
    ResolvedRecorderSpec,
)


def _make_spec(*records, snapshot_id="hash"):
    return ResolvedRecorderSpec(
        fem_snapshot_id=snapshot_id,
        records=tuple(records),
    )


# =====================================================================
# Token aggregation: nodes
# =====================================================================

def test_displacement_components_collapse_to_one_token() -> None:
    spec = _make_spec(ResolvedRecorderRecord(
        category="nodes", name="r",
        components=("displacement_x", "displacement_y", "displacement_z"),
        dt=None, n_steps=None,
        node_ids=np.array([1, 2]),
    ))
    line = spec.to_mpco_tcl_command()
    # All three components share one MPCO token.
    assert "-N displacement" in line
    # Only one occurrence (not three).
    assert line.count("displacement") == 1


def test_node_kinematics_full_token_set() -> None:
    spec = _make_spec(ResolvedRecorderRecord(
        category="nodes", name="r",
        components=(
            "displacement_x", "rotation_x",
            "velocity_x", "angular_velocity_x",
            "acceleration_x", "angular_acceleration_x",
        ),
        dt=None, n_steps=None,
        node_ids=np.array([1]),
    ))
    line = spec.to_mpco_tcl_command()
    for tok in [
        "displacement", "rotation",
        "velocity", "angularVelocity",
        "acceleration", "angularAcceleration",
    ]:
        assert tok in line, f"missing {tok!r} in: {line}"


def test_reaction_tokens() -> None:
    spec = _make_spec(ResolvedRecorderRecord(
        category="nodes", name="r",
        components=("reaction_force_x", "reaction_moment_z"),
        dt=None, n_steps=None,
        node_ids=np.array([1]),
    ))
    line = spec.to_mpco_tcl_command()
    assert "reactionForce" in line
    assert "reactionMoment" in line


def test_pressure_token() -> None:
    spec = _make_spec(ResolvedRecorderRecord(
        category="nodes", name="r",
        components=("pore_pressure",),
        dt=None, n_steps=None,
        node_ids=np.array([1]),
    ))
    line = spec.to_mpco_tcl_command()
    assert "-N pressure" in line


# =====================================================================
# Token aggregation: elements
# =====================================================================

def test_stress_strain_separate_tokens() -> None:
    spec = _make_spec(ResolvedRecorderRecord(
        category="gauss", name="g",
        components=("stress_xx", "strain_xx"),
        dt=None, n_steps=None,
        element_ids=np.array([1]),
    ))
    line = spec.to_mpco_tcl_command()
    assert "stress" in line
    assert "strain" in line


def test_line_stations_section_force_token() -> None:
    spec = _make_spec(ResolvedRecorderRecord(
        category="line_stations", name="b",
        components=("axial_force", "bending_moment_y", "torsion"),
        dt=None, n_steps=None,
        element_ids=np.array([1]),
    ))
    line = spec.to_mpco_tcl_command()
    assert "section.force" in line


def test_fiber_section_token() -> None:
    spec = _make_spec(ResolvedRecorderRecord(
        category="fibers", name="f",
        components=("fiber_stress",),
        dt=None, n_steps=None,
        element_ids=np.array([1]),
    ))
    line = spec.to_mpco_tcl_command()
    assert "section.fiber.stress" in line


def test_global_force_token() -> None:
    spec = _make_spec(ResolvedRecorderRecord(
        category="elements", name="e",
        components=("nodal_resisting_force_x",),
        dt=None, n_steps=None,
        element_ids=np.array([1]),
    ))
    line = spec.to_mpco_tcl_command()
    assert "globalForce" in line


def test_local_force_token() -> None:
    spec = _make_spec(ResolvedRecorderRecord(
        category="elements", name="e",
        components=("nodal_resisting_force_local_x",),
        dt=None, n_steps=None,
        element_ids=np.array([1]),
    ))
    line = spec.to_mpco_tcl_command()
    assert "localForce" in line


# =====================================================================
# Modal
# =====================================================================

def test_modal_emits_modes_tokens() -> None:
    spec = _make_spec(ResolvedRecorderRecord(
        category="modal", name="m",
        components=(),
        dt=None, n_steps=None,
        n_modes=5,
    ))
    line = spec.to_mpco_tcl_command()
    assert "modesOfVibration" in line
    assert "modesOfVibrationRotational" in line


# =====================================================================
# Cadence aggregation
# =====================================================================

def test_smallest_dt_wins() -> None:
    spec = _make_spec(
        ResolvedRecorderRecord(
            category="nodes", name="a",
            components=("displacement_x",),
            dt=0.1, n_steps=None,
            node_ids=np.array([1]),
        ),
        ResolvedRecorderRecord(
            category="nodes", name="b",
            components=("velocity_x",),
            dt=0.01, n_steps=None,
            node_ids=np.array([1]),
        ),
    )
    line = spec.to_mpco_tcl_command()
    assert "-T dt 0.01" in line


def test_n_steps_used_when_no_dt() -> None:
    spec = _make_spec(ResolvedRecorderRecord(
        category="nodes", name="r",
        components=("displacement_x",),
        dt=None, n_steps=5,
        node_ids=np.array([1]),
    ))
    line = spec.to_mpco_tcl_command()
    assert "-T nsteps 5" in line


def test_no_cadence_omits_T_flag() -> None:
    spec = _make_spec(ResolvedRecorderRecord(
        category="nodes", name="r",
        components=("displacement_x",),
        dt=None, n_steps=None,
        node_ids=np.array([1]),
    ))
    line = spec.to_mpco_tcl_command()
    assert "-T" not in line


# =====================================================================
# File path / output_dir
# =====================================================================

def test_default_filename() -> None:
    spec = _make_spec(ResolvedRecorderRecord(
        category="nodes", name="r",
        components=("displacement_x",),
        dt=None, n_steps=None,
        node_ids=np.array([1]),
    ))
    line = spec.to_mpco_tcl_command()
    assert "run.mpco" in line


def test_custom_filename() -> None:
    spec = _make_spec(ResolvedRecorderRecord(
        category="nodes", name="r",
        components=("displacement_x",),
        dt=None, n_steps=None,
        node_ids=np.array([1]),
    ))
    line = spec.to_mpco_tcl_command(filename="my_results.mpco")
    assert "my_results.mpco" in line


def test_output_dir_prefix() -> None:
    spec = _make_spec(ResolvedRecorderRecord(
        category="nodes", name="r",
        components=("displacement_x",),
        dt=None, n_steps=None,
        node_ids=np.array([1]),
    ))
    line = spec.to_mpco_tcl_command(output_dir="out")
    assert "out/run.mpco" in line


# =====================================================================
# Python emission
# =====================================================================

def test_python_emission_format() -> None:
    spec = _make_spec(ResolvedRecorderRecord(
        category="nodes", name="r",
        components=("displacement_x",),
        dt=0.01, n_steps=None,
        node_ids=np.array([1]),
    ))
    line = spec.to_mpco_python_command()
    assert line.startswith("ops.recorder(")
    assert "'mpco'" in line
    assert "'run.mpco'" in line
    assert "'-N'" in line
    assert "'displacement'" in line
    assert "'-T'" in line and "'dt'" in line


# =====================================================================
# Integration through export.tcl/py
# =====================================================================

def test_export_tcl_with_mpco_format(g, tmp_path: Path) -> None:
    g.model.geometry.add_box(0, 0, 0, 1, 1, 1, label="box")
    g.physical.add_volume("box", name="Body")
    g.physical.add_surface(g.model.queries.boundary([(3, 1)]), name="Skin")
    g.mesh.sizing.set_global_size(2.0)
    g.mesh.generation.generate(dim=3)

    g.opensees.set_model(ndm=3, ndf=3)
    g.opensees.materials.add_nd_material(
        "C", "ElasticIsotropic", E=2e10, nu=0.2, rho=2400,
    )
    g.opensees.elements.assign("Body", "FourNodeTetrahedron", material="C")
    g.opensees.elements.fix("Skin", dofs=[1, 1, 1])
    fem = g.mesh.queries.get_fem_data(dim=3)
    g.opensees.build()

    g.opensees.recorders.nodes(
        pg="Body", components=["displacement"], dt=0.01, name="all_disp",
    )
    g.opensees.recorders.gauss(
        pg="Body", components=["stress_xx"], name="body_stress",
    )
    spec = g.opensees.recorders.resolve(fem, ndm=3, ndf=3)

    tcl_path = tmp_path / "model.tcl"
    g.opensees.export.tcl(
        tcl_path, recorders=spec, recorders_file_format="mpco",
    )
    text = tcl_path.read_text()

    # Single mpco line, no per-record .out commands.
    assert "recorder mpco" in text
    assert "model.mpco" in text
    assert text.count("recorder Node") == 0
    assert text.count("recorder Element") == 0
    # Tokens appear
    assert "displacement" in text
    assert "stress" in text
    # Cadence carried over
    assert "-T dt 0.01" in text


def test_export_py_with_mpco_format(g, tmp_path: Path) -> None:
    g.model.geometry.add_box(0, 0, 0, 1, 1, 1, label="box")
    g.physical.add_volume("box", name="Body")
    g.mesh.sizing.set_global_size(2.0)
    g.mesh.generation.generate(dim=3)
    g.opensees.set_model(ndm=3, ndf=3)
    g.opensees.materials.add_nd_material(
        "C", "ElasticIsotropic", E=2e10, nu=0.2,
    )
    g.opensees.elements.assign("Body", "FourNodeTetrahedron", material="C")
    fem = g.mesh.queries.get_fem_data(dim=3)
    g.opensees.build()

    g.opensees.recorders.nodes(
        pg="Body", components=["displacement", "velocity"],
    )
    spec = g.opensees.recorders.resolve(fem, ndm=3, ndf=3)

    py_path = tmp_path / "model.py"
    g.opensees.export.py(py_path, recorders=spec, recorders_file_format="mpco")
    text = py_path.read_text()
    assert "ops.recorder('mpco'" in text
    assert "model.mpco" in text
    assert "'displacement'" in text
    assert "'velocity'" in text
