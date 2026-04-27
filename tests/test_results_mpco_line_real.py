"""Phase 11b Step 2a — full cycle: real ForceBeamColumn3d .mpco read.

Builds a 3-D cantilever ForceBeamColumn3d under axial tip load, runs
OpenSees through the El Ladruno Tcl launcher to produce a real
``.mpco`` file, and reads section forces back through
``MPCOReader.read_line_stations``. Verifies physical correctness
(constant axial force = applied tip load along the beam) and that
the resolved layout's component names match the assigned
SectionAggregator's response codes.

Skipped if the launcher is not available (mirrors
``test_results_mpco_element_real.py::test_tet_stress_full_cycle``).
"""
from __future__ import annotations

import os
import subprocess
from pathlib import Path

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


def test_force_beam_3d_aggregated_section(tmp_path: Path) -> None:
    """Lobatto-5 ForceBeamColumn3d with aggregated [P,Mz,My,T,Vy,Vz] section."""
    launcher = _resolve_tcl_launcher()
    if launcher is None:
        pytest.skip(
            "No OpenSees Tcl launcher found. Set APEGMSH_OPENSEES_TCL "
            "or install the El Ladruno OpenSees build to enable this test."
        )

    # 1.0e3 N axial tip load. With 5-IP Lobatto, the section force's
    # axial component must equal the load at every IP (free-free
    # cantilever, no distributed load). The Lobatto rule places IPs at
    # the natural-coord values below in [-1, +1].
    expected_gp_x = np.array(
        [-1.0, -0.65465367, 0.0, 0.65465367, 1.0],
        dtype=np.float64,
    )
    P_axial = 1.0e3

    tcl_lines = [
        "wipe",
        "model BasicBuilder -ndm 3 -ndf 6",
        "node 1 0.0 0.0 0.0",
        "node 2 5.0 0.0 0.0",
        "fix 1 1 1 1 1 1 1",
        "geomTransf Linear 1 0.0 1.0 0.0",
        # Aggregated 6-component section: [P, Mz, My, T, Vy, Vz].
        "section Elastic 1 200e9 0.01 1.0e-4 1.5e-4 80e9 1.0e-4",
        "uniaxialMaterial Elastic 100 1.0e9",
        "uniaxialMaterial Elastic 101 1.0e9",
        "section Aggregator 11 100 Vy 101 Vz -section 1",
        "beamIntegration Lobatto 1 11 5",
        "element forceBeamColumn 1 1 2 1 1",
        # MPCOReader's time vector reads from ON_NODES; combine -N
        # (displacement) and -E (section.force) on the same recorder.
        "recorder mpco run.mpco -N displacement -E section.force -T dt 1.0",
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
    mpco_path = tmp_path / "run.mpco"
    assert mpco_path.exists(), f"MPCO file not produced.\n{combined}"

    from apeGmsh.results import Results
    with Results.from_mpco(mpco_path) as r:
        assert len(r.stages) >= 1
        s = r.stage(r.stages[0].id)

        # ── Component coverage ─────────────────────────────────────
        comps = set(s.elements.line_stations.available_components())
        for name in (
            "axial_force",
            "bending_moment_z",
            "bending_moment_y",
            "torsion",
            "shear_y",
            "shear_z",
        ):
            assert name in comps, f"missing {name} in {sorted(comps)}"

        # ── Axial-force slab shape + station coordinates ──────────
        slab = s.elements.line_stations.get(component="axial_force")
        assert slab.values.shape == (1, 5), (
            f"expected (1, 5) axial-force slab; got {slab.values.shape}"
        )
        assert slab.element_index.tolist() == [1, 1, 1, 1, 1]
        np.testing.assert_allclose(
            slab.station_natural_coord, expected_gp_x, atol=1e-6,
        )

        # ── Physical sanity: axial force = applied tip load at every IP ──
        np.testing.assert_allclose(
            slab.values[0], np.full(5, P_axial), rtol=1e-9, atol=1e-6,
        )

        # ── Other components are zero (no transverse loads / moments) ─
        for name in ("shear_y", "shear_z", "torsion",
                     "bending_moment_y", "bending_moment_z"):
            other = s.elements.line_stations.get(component=name)
            np.testing.assert_allclose(
                other.values[0], np.zeros(5), atol=1e-6,
                err_msg=f"{name} should be zero under pure axial load",
            )


def test_force_beam_3d_bare_section(tmp_path: Path) -> None:
    """Bare FiberSection3d (4-comp [P, Mz, My, T]) — verify shear_y NOT exposed."""
    launcher = _resolve_tcl_launcher()
    if launcher is None:
        pytest.skip("No OpenSees Tcl launcher found.")

    tcl_lines = [
        "wipe",
        "model BasicBuilder -ndm 3 -ndf 6",
        "node 1 0.0 0.0 0.0",
        "node 2 4.0 0.0 0.0",
        "fix 1 1 1 1 1 1 1",
        "geomTransf Linear 1 0.0 1.0 0.0",
        # Plain elastic section returns [P, Mz, My, T] only — no shear.
        "section Elastic 1 200e9 0.01 1.0e-4 1.5e-4 80e9 1.0e-4",
        "beamIntegration Legendre 1 1 3",
        "element forceBeamColumn 1 1 2 1 1",
        # MPCOReader's time vector reads from ON_NODES; combine -N
        # (displacement) and -E (section.force) on the same recorder.
        "recorder mpco run.mpco -N displacement -E section.force -T dt 1.0",
        "timeSeries Linear 1",
        "pattern Plain 1 1 {",
        "    load 2 5.0e2 0.0 0.0 0.0 0.0 0.0",
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
        capture_output=True, text=True, timeout=60, cwd=str(tmp_path),
    )
    combined = (result.stdout or "") + "\n" + (result.stderr or "")
    assert result.returncode == 0, combined
    assert "DONE" in combined, combined
    mpco_path = tmp_path / "run.mpco"
    assert mpco_path.exists()

    from apeGmsh.results import Results
    with Results.from_mpco(mpco_path) as r:
        s = r.stage(r.stages[0].id)
        comps = set(s.elements.line_stations.available_components())

        # Bare section exposes axial / Mz / My / T but NOT shears.
        for present in ("axial_force", "bending_moment_z",
                        "bending_moment_y", "torsion"):
            assert present in comps, f"{present} missing in {sorted(comps)}"
        for absent in ("shear_y", "shear_z"):
            slab = s.elements.line_stations.get(component=absent)
            assert slab.values.shape == (1, 0), (
                f"{absent} should be empty for a bare 4-comp section; "
                f"got values shape {slab.values.shape}"
            )

        # Axial force = 500 N at every IP.
        slab = s.elements.line_stations.get(component="axial_force")
        assert slab.values.shape == (1, 3)
        np.testing.assert_allclose(
            slab.values[0], np.full(3, 5.0e2), rtol=1e-9, atol=1e-6,
        )
