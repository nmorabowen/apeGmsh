"""Phase 5 — recorders flow through ``g.opensees.export.tcl/py``.

Integration tests that build a small model, declare recorders,
export, and check the resulting script + manifest sidecar.
"""
from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pytest


def _build_min_model(g):
    """Smallest viable apeGmsh + OpenSees model — single tet box."""
    g.model.geometry.add_box(0, 0, 0, 1, 1, 1, label="box")
    g.physical.add_volume("box", name="Body")
    g.physical.add_surface(g.model.queries.boundary([(3, 1)]), name="Skin")
    g.mesh.sizing.set_global_size(2.0)
    g.mesh.generation.generate(dim=3)

    g.opensees.set_model(ndm=3, ndf=3)
    g.opensees.materials.add_nd_material(
        "Concrete", "ElasticIsotropic", E=30e9, nu=0.2, rho=2400,
    )
    g.opensees.elements.assign(
        "Body", "FourNodeTetrahedron", material="Concrete",
    )
    g.opensees.elements.fix("Skin", dofs=[1, 1, 1])
    fem = g.mesh.queries.get_fem_data(dim=3)
    g.opensees.build()
    return fem


def test_tcl_export_with_recorders_writes_commands_and_manifest(g, tmp_path: Path) -> None:
    fem = _build_min_model(g)

    g.opensees.recorders.nodes(
        pg="Body", components=["displacement"], dt=0.01, name="all_disp",
    )
    g.opensees.recorders.gauss(
        pg="Body", components=["stress_xx"], name="body_stress",
    )
    spec = g.opensees.recorders.resolve(fem, ndm=3, ndf=3)

    tcl_path = tmp_path / "model.tcl"
    g.opensees.export.tcl(tcl_path, recorders=spec, recorders_output_dir="out/")

    text = tcl_path.read_text()
    assert "recorder Node" in text
    assert "all_disp_disp.out" in text
    assert "recorder Element" in text
    assert "body_stress_gauss.out" in text
    assert "-dT 0.01" in text
    # Manifest sidecar lands next to the script.
    manifest = tcl_path.with_suffix(".tcl.manifest.h5")
    assert manifest.exists()
    with h5py.File(manifest, "r") as f:
        assert "fem_snapshot_id" in f.attrs
        assert "records" in f


def test_py_export_with_recorders(g, tmp_path: Path) -> None:
    fem = _build_min_model(g)

    g.opensees.recorders.nodes(
        pg="Body", components=["displacement", "velocity"], name="kin",
    )
    spec = g.opensees.recorders.resolve(fem, ndm=3, ndf=3)

    py_path = tmp_path / "model.py"
    g.opensees.export.py(py_path, recorders=spec)

    text = py_path.read_text()
    assert "ops.recorder('Node'" in text
    assert "'kin_disp.out'" in text
    assert "'kin_vel.out'" in text
    # Manifest landed
    manifest = py_path.with_suffix(".py.manifest.h5")
    assert manifest.exists()


def test_tcl_export_explicit_manifest_path(g, tmp_path: Path) -> None:
    fem = _build_min_model(g)
    g.opensees.recorders.nodes(pg="Body", components=["displacement"])
    spec = g.opensees.recorders.resolve(fem, ndm=3, ndf=3)

    tcl_path = tmp_path / "model.tcl"
    custom_manifest = tmp_path / "elsewhere" / "manifest.h5"
    g.opensees.export.tcl(
        tcl_path, recorders=spec, manifest_path=custom_manifest,
    )
    assert custom_manifest.exists()
    # Default location not used.
    assert not tcl_path.with_suffix(".tcl.manifest.h5").exists()


def test_export_without_recorders_unchanged(g, tmp_path: Path) -> None:
    """Pre-Phase-5 callers (no recorders=) still work unchanged."""
    _build_min_model(g)

    tcl_path = tmp_path / "no_recorders.tcl"
    g.opensees.export.tcl(tcl_path)
    text = tcl_path.read_text()
    assert "recorder " not in text
    # No manifest sidecar created when no recorders.
    assert not tcl_path.with_suffix(".tcl.manifest.h5").exists()


def test_manifest_loadable_via_resolved_spec(g, tmp_path: Path) -> None:
    """Manifest sidecar round-trips back to a ResolvedRecorderSpec."""
    fem = _build_min_model(g)
    g.opensees.recorders.nodes(
        pg="Body", components=["displacement"], dt=0.01, name="r1",
    )
    g.opensees.recorders.gauss(
        pg="Body", components=["stress_xx"], name="r2",
    )
    spec = g.opensees.recorders.resolve(fem, ndm=3, ndf=3)

    tcl_path = tmp_path / "model.tcl"
    g.opensees.export.tcl(tcl_path, recorders=spec)

    from apeGmsh.solvers._recorder_specs import ResolvedRecorderSpec
    loaded = ResolvedRecorderSpec.from_manifest_h5(
        tcl_path.with_suffix(".tcl.manifest.h5"),
    )
    assert loaded.fem_snapshot_id == spec.fem_snapshot_id
    assert len(loaded.records) == 2
    names = sorted(r.name for r in loaded.records)
    assert names == ["r1", "r2"]
