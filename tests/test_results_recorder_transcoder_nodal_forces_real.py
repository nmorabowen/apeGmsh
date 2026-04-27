"""Phase 11b Step 3c.3 — full Tcl-subprocess cycle for the nodal-forces transcoder.

Builds a 3D ElasticBeam3d cantilever, emits the matching ``recorder
Element ... globalForce`` and ``... localForce`` Tcl lines via
``ResolvedRecorderSpec.to_tcl_commands()``, runs OpenSees through the
El Ladruno launcher, then runs ``RecorderTranscoder`` over the
emitted ``.out`` files. Reads the resulting native HDF5 through
``Results.from_native`` and asserts the same physics that the MPCO
real-file test (Step 3a) and the DomainCapture real-openseespy test
(Step 3b) saw — full three-way agreement on identical model.

Skipped if the launcher is not available.
"""
from __future__ import annotations

import os
import subprocess
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


_DEFAULT_TCL_LAUNCHERS = [
    r"C:\Program Files\El Ladruno OpenSees\opensees_ladruno.bat",
]


def _resolve_tcl_launcher() -> Path | None:
    env = os.environ.get("APEGMSH_OPENSEES_TCL")
    if env:
        p = Path(env)
        if p.exists():
            return p
    for cand in _DEFAULT_TCL_LAUNCHERS:
        p = Path(cand)
        if p.exists():
            return p
    return None


class _MinimalFem:
    def __init__(self, node_ids: np.ndarray, coords: np.ndarray) -> None:
        self.nodes = SimpleNamespace(ids=node_ids, coords=coords)
        self.elements = []

    @property
    def snapshot_id(self) -> str:
        from apeGmsh.mesh._femdata_hash import compute_snapshot_id
        return compute_snapshot_id(self)

    def to_native_h5(self, group) -> None:
        group.attrs["snapshot_id"] = self.snapshot_id
        group.attrs["ndm"] = 3
        group.attrs["ndf"] = 6
        group.attrs["model_name"] = ""
        group.attrs["units"] = ""
        n = group.create_group("nodes")
        n.create_dataset("ids", data=self.nodes.ids)
        n.create_dataset("coords", data=self.nodes.coords)
        group.create_group("elements")


def test_elastic_beam_3d_global_and_local_three_way_agreement(
    tmp_path: Path,
) -> None:
    """Three-way agreement on an ElasticBeam3d cantilever.

    Same model as Step 3a's MPCO test and Step 3b's DomainCapture
    test. Combined transverse + axial tip load → axial force = applied
    Fx at both nodes; Mz at fixed end = -Fy*L; Mz at free end ≈ 0.
    """
    launcher = _resolve_tcl_launcher()
    if launcher is None:
        pytest.skip("No OpenSees Tcl launcher found.")

    L = 5.0
    Fx, Fy = 1.0e3, 5.0e2

    coords = np.array([[0.0, 0.0, 0.0], [L, 0.0, 0.0]], dtype=np.float64)
    node_ids = np.array([1, 2], dtype=np.int64)
    fem = _MinimalFem(node_ids=node_ids, coords=coords)

    from apeGmsh.solvers._recorder_specs import (
        ResolvedRecorderRecord,
        ResolvedRecorderSpec,
    )
    spec = ResolvedRecorderSpec(
        fem_snapshot_id=fem.snapshot_id,
        records=(
            ResolvedRecorderRecord(
                category="elements", name="beam_global",
                components=(
                    "nodal_resisting_force_x", "nodal_resisting_force_y",
                    "nodal_resisting_force_z", "nodal_resisting_moment_x",
                    "nodal_resisting_moment_y", "nodal_resisting_moment_z",
                ),
                dt=None, n_steps=None,
                element_ids=np.array([1], dtype=np.int64),
                element_class_name="ElasticBeam3d",
            ),
            ResolvedRecorderRecord(
                category="elements", name="beam_local",
                components=(
                    "nodal_resisting_force_local_x",
                    "nodal_resisting_moment_local_z",
                ),
                dt=None, n_steps=None,
                element_ids=np.array([1], dtype=np.int64),
                element_class_name="ElasticBeam3d",
            ),
        ),
    )

    recorder_lines = spec.to_tcl_commands()
    # Two records → two Tcl recorder lines.
    assert len(recorder_lines) == 2
    global_line = next(ln for ln in recorder_lines if "globalForce" in ln)
    local_line = next(ln for ln in recorder_lines if "localForce" in ln)
    assert "beam_global_elements.out" in global_line
    assert "beam_local_elements.out" in local_line

    tcl_lines = [
        "wipe",
        "model BasicBuilder -ndm 3 -ndf 6",
        "node 1 0.0 0.0 0.0",
        f"node 2 {L} 0.0 0.0",
        "fix 1 1 1 1 1 1 1",
        "geomTransf Linear 1 0.0 1.0 0.0",
        "element elasticBeamColumn 1 1 2 0.01 200e9 80e9 1.0e-4 1.0e-4 1.5e-4 1",
        global_line,
        local_line,
        "timeSeries Linear 1",
        "pattern Plain 1 1 {",
        f"    load 2 {Fx} {Fy} 0.0 0.0 0.0 0.0",
        "}",
        "system ProfileSPD",
        "numberer RCM",
        "constraints Plain",
        "algorithm Linear",
        "integrator LoadControl 1.0",
        "analysis Static",
        "analyze 1",
        "wipe",
        'puts "DONE"',
    ]
    script = tmp_path / "model.tcl"
    script.write_text("\n".join(tcl_lines) + "\n", encoding="utf-8")

    result = subprocess.run(
        [str(launcher), str(script)],
        capture_output=True, text=True, timeout=60,
        cwd=str(tmp_path),
    )
    combined = (result.stdout or "") + "\n" + (result.stderr or "")
    assert result.returncode == 0, combined
    assert "DONE" in combined, combined
    assert (tmp_path / "beam_global_elements.out").exists()
    assert (tmp_path / "beam_local_elements.out").exists()

    from apeGmsh.results import Results
    from apeGmsh.results.transcoders._recorder import RecorderTranscoder

    target = tmp_path / "transcoded.h5"
    transcoder = RecorderTranscoder(
        spec, output_dir=tmp_path, target_path=target, fem=fem,
        stage_name="static", stage_kind="static",
    )
    transcoder.run()
    assert transcoder.unsupported == [], (
        f"unexpected unsupported: {transcoder.unsupported}"
    )

    with Results.from_native(target) as r:
        s = r.stage(r.stages[0].id)
        comps = set(s.elements.available_components())
        for n in ("nodal_resisting_force_x", "nodal_resisting_moment_z",
                  "nodal_resisting_force_local_x",
                  "nodal_resisting_moment_local_z"):
            assert n in comps, f"missing {n} in {sorted(comps)}"

        # Global Fx — magnitude equals applied axial at both nodes.
        slab_fx = s.elements.get(component="nodal_resisting_force_x")
        assert slab_fx.values.shape == (1, 1, 2)
        np.testing.assert_allclose(
            np.abs(slab_fx.values[0, 0]), [Fx, Fx], rtol=1e-9, atol=1e-6,
        )

        # Mz at fixed end = -Fy*L; at free end ≈ 0.
        slab_mz = s.elements.get(component="nodal_resisting_moment_z")
        np.testing.assert_allclose(
            abs(slab_mz.values[0, 0, 0]), Fy * L, rtol=1e-9, atol=1e-6,
        )
        assert abs(slab_mz.values[0, 0, 1]) < 1e-6

        # Local axial = Fx (beam along global x).
        slab_n = s.elements.get(component="nodal_resisting_force_local_x")
        np.testing.assert_allclose(
            np.abs(slab_n.values[0, 0]), [Fx, Fx], rtol=1e-9, atol=1e-6,
        )
