"""Phase 11b Step 2c.3 — full Tcl-subprocess cycle for the line-stations transcoder.

Builds a 3D ForceBeamColumn3d cantilever, emits the matching paired
``recorder Element ... section force`` + ``... integrationPoints``
Tcl lines via ``ResolvedRecorderSpec.to_tcl_commands()``, runs
OpenSees through the El Ladruno launcher, then runs
``RecorderTranscoder`` over the emitted ``.out`` + ``_gpx.out``
files. Reads the resulting native HDF5 through
``Results.from_native`` and asserts the same physics that the MPCO
real-file (Step 2a) and DomainCapture real-openseespy (Step 2b)
tests produced — full three-way agreement on identical model.

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


class _BeamFem:
    """Minimal FEMData with a duck-typed element-end-coords method.

    The line-stations transcoder needs per-element length to
    normalise IP positions. We expose ``element_end_coords(eid)``;
    the transcoder's ``_resolve_element_lengths`` checks for that
    method first before falling back to the full FEMData.elements
    API.
    """

    def __init__(
        self,
        node_ids: np.ndarray,
        coords: np.ndarray,
        connectivity: dict[int, tuple[int, int]],
    ) -> None:
        ids = np.asarray(node_ids, dtype=np.int64)
        self.nodes = SimpleNamespace(
            ids=ids, coords=np.asarray(coords, dtype=np.float64),
        )
        self.elements = []
        self._conn = connectivity
        self._coords_by_id = {
            int(nid): np.asarray(c, dtype=np.float64)
            for nid, c in zip(ids, coords)
        }

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
        n.create_dataset(
            "ids", data=np.asarray(self.nodes.ids, dtype=np.int64),
        )
        n.create_dataset(
            "coords", data=np.asarray(self.nodes.coords, dtype=np.float64),
        )
        group.create_group("elements")

    def element_end_coords(self, eid: int) -> tuple[np.ndarray, np.ndarray]:
        n_a, n_b = self._conn[int(eid)]
        return self._coords_by_id[n_a], self._coords_by_id[n_b]


def test_force_beam_3d_lobatto5_aggregated_three_way_agreement(
    tmp_path: Path,
) -> None:
    """Three-way physics agreement on a 5-IP Lobatto ForceBeamColumn3d cantilever.

    Same model as Step 2a's MPCO test and Step 2b's DomainCapture
    test. Pure axial tip load → axial force = applied load at every
    IP, all transverse components zero. Verifies the .out transcoder
    produces the same numbers as the other two paths.
    """
    launcher = _resolve_tcl_launcher()
    if launcher is None:
        pytest.skip(
            "No OpenSees Tcl launcher found. Set APEGMSH_OPENSEES_TCL "
            "or install the El Ladruno OpenSees build to enable this test."
        )

    L = 5.0
    P_axial = 1.0e3
    expected_gp_x = np.array(
        [-1.0, -0.65465367, 0.0, 0.65465367, 1.0],
        dtype=np.float64,
    )

    coords = np.array(
        [[0.0, 0.0, 0.0], [L, 0.0, 0.0]], dtype=np.float64,
    )
    node_ids = np.array([1, 2], dtype=np.int64)
    fem = _BeamFem(
        node_ids=node_ids,
        coords=coords,
        connectivity={1: (1, 2)},
    )

    from apeGmsh.solvers._recorder_specs import (
        ResolvedRecorderRecord,
        ResolvedRecorderSpec,
    )
    spec = ResolvedRecorderSpec(
        fem_snapshot_id=fem.snapshot_id,
        records=(
            ResolvedRecorderRecord(
                category="line_stations", name="beam",
                components=(
                    "axial_force", "bending_moment_z",
                    "bending_moment_y", "torsion",
                    "shear_y", "shear_z",
                ),
                dt=None, n_steps=None,
                element_ids=np.array([1], dtype=np.int64),
                element_class_name="ForceBeamColumn3d",
            ),
        ),
    )

    # ── Emit the paired Tcl commands and write a complete model script ──
    recorder_lines = spec.to_tcl_commands()
    # Step 2c.1 emits two recorders: section.force + integrationPoints.
    assert len(recorder_lines) == 2
    section_line = next(
        ln for ln in recorder_lines if "section force" in ln
    )
    gpx_line = next(
        ln for ln in recorder_lines if "integrationPoints" in ln
    )
    # Confirm file paths match the convention.
    assert "beam_line_stations.out" in section_line
    assert "beam_line_stations_gpx.out" in gpx_line

    tcl_lines = [
        "wipe",
        "model BasicBuilder -ndm 3 -ndf 6",
        f"node 1 0.0 0.0 0.0",
        f"node 2 {L} 0.0 0.0",
        "fix 1 1 1 1 1 1 1",
        "geomTransf Linear 1 0.0 1.0 0.0",
        # Aggregated 6-component section: [P, Mz, My, T, Vy, Vz].
        "section Elastic 1 200e9 0.01 1.0e-4 1.5e-4 80e9 1.0e-4",
        "uniaxialMaterial Elastic 100 1.0e9",
        "uniaxialMaterial Elastic 101 1.0e9",
        "section Aggregator 11 100 Vy 101 Vz -section 1",
        "beamIntegration Lobatto 1 11 5",
        "element forceBeamColumn 1 1 2 1 1",
        section_line,
        gpx_line,
        "timeSeries Linear 1",
        "pattern Plain 1 1 {",
        f"    load 2 {P_axial} 0.0 0.0 0.0 0.0 0.0",
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
    assert result.returncode == 0, (
        f"OpenSees.exe failed (rc={result.returncode})\n{combined}"
    )
    assert "DONE" in combined, f"Tcl script didn't reach DONE.\n{combined}"
    assert (tmp_path / "beam_line_stations.out").exists()
    assert (tmp_path / "beam_line_stations_gpx.out").exists()

    # ── Run the transcoder over the emitted files ────────────────
    from apeGmsh.results import Results
    from apeGmsh.results.transcoders._recorder import RecorderTranscoder

    target = tmp_path / "transcoded.h5"
    transcoder = RecorderTranscoder(
        spec, output_dir=tmp_path, target_path=target, fem=fem,
        stage_name="static", stage_kind="static",
    )
    transcoder.run()
    assert transcoder.unsupported == [], (
        f"unexpected unsupported records: {transcoder.unsupported}"
    )

    # ── Read back and assert the same physics MPCO/DomainCapture saw ─
    with Results.from_native(target) as r:
        s = r.stage(r.stages[0].id)
        comps = set(s.elements.line_stations.available_components())
        for name in (
            "axial_force", "bending_moment_z", "bending_moment_y",
            "torsion", "shear_y", "shear_z",
        ):
            assert name in comps, f"missing {name} in {sorted(comps)}"

        slab = s.elements.line_stations.get(component="axial_force")
        assert slab.values.shape == (1, 5), (
            f"expected (1, 5) axial-force slab; got {slab.values.shape}"
        )
        assert slab.element_index.tolist() == [1, 1, 1, 1, 1]
        np.testing.assert_allclose(
            slab.station_natural_coord, expected_gp_x, atol=1e-5,
        )
        # Axial force = applied tip load at every IP — matches MPCO
        # (test_results_mpco_line_real.py) and DomainCapture
        # (test_results_domain_capture_line_stations_real.py).
        np.testing.assert_allclose(
            slab.values[0], np.full(5, P_axial), rtol=1e-9, atol=1e-6,
        )
        # All transverse components zero.
        for name in ("shear_y", "shear_z", "torsion",
                     "bending_moment_y", "bending_moment_z"):
            other = s.elements.line_stations.get(component=name)
            np.testing.assert_allclose(
                other.values[0], np.zeros(5), atol=1e-6,
                err_msg=f"{name} should be zero under pure axial load",
            )
