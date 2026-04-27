"""Phase 11a Step B — full cycle: record stress via real OpenSees, read back.

Builds a 4-node tetrahedron model under apex compression, emits a
``recorder mpco -E stress`` command via ``ResolvedRecorderSpec``,
runs OpenSees through the El Ladruno Tcl launcher, and reads the
resulting ``.mpco`` file back through ``MPCOReader.read_gauss``.

Skipped if the launcher is not available (mirrors
``test_results_mpco_emit_real.py::test_tcl_subprocess_full_cycle``).
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


def test_tet_stress_full_cycle(tmp_path: Path) -> None:
    """Record continuum stress on FourNodeTetrahedron and read it back."""
    launcher = _resolve_tcl_launcher()
    if launcher is None:
        pytest.skip(
            "No OpenSees Tcl launcher found. Set APEGMSH_OPENSEES_TCL "
            "or install the El Ladruno OpenSees build to enable this test."
        )

    coords = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.5, 0.5, 1.0],
    ], dtype=np.float64)
    node_ids = np.array([1, 2, 3, 4], dtype=np.int64)
    element_ids = np.array([1], dtype=np.int64)

    class _Fem:
        def __init__(self):
            self.nodes = SimpleNamespace(ids=node_ids, coords=coords)
            self.elements = []

        @property
        def snapshot_id(self):
            from apeGmsh.mesh._femdata_hash import compute_snapshot_id
            return compute_snapshot_id(self)

    fem = _Fem()

    from apeGmsh.solvers._recorder_specs import (
        ResolvedRecorderRecord,
        ResolvedRecorderSpec,
    )
    spec = ResolvedRecorderSpec(
        fem_snapshot_id=fem.snapshot_id,
        records=(
            ResolvedRecorderRecord(
                category="nodes", name="all_kin",
                components=("displacement_x", "displacement_y", "displacement_z"),
                dt=None, n_steps=None,
                node_ids=node_ids,
            ),
            ResolvedRecorderRecord(
                category="gauss", name="solid_stress",
                components=(
                    "stress_xx", "stress_yy", "stress_zz",
                    "stress_xy", "stress_yz", "stress_xz",
                ),
                dt=None, n_steps=None,
                element_ids=element_ids,
            ),
        ),
    )

    mpco_path = tmp_path / "run.mpco"
    mpco_recorder_line = spec.to_mpco_tcl_command(filename="run.mpco")
    # Sanity-check the spec composed both -N and -E tokens.
    assert "-N" in mpco_recorder_line
    assert "-E" in mpco_recorder_line
    assert "stress" in mpco_recorder_line

    tcl_lines = [
        "wipe",
        "model BasicBuilder -ndm 3 -ndf 3",
        "node 1 0.0 0.0 0.0",
        "node 2 1.0 0.0 0.0",
        "node 3 0.0 1.0 0.0",
        "node 4 0.5 0.5 1.0",
        "fix 1 1 1 1",
        "fix 2 1 1 1",
        "fix 3 1 1 1",
        "nDMaterial ElasticIsotropic 1 200e9 0.3",
        "element FourNodeTetrahedron 1 1 2 3 4 1",
        mpco_recorder_line,
        "timeSeries Linear 1",
        "pattern Plain 1 1 {",
        "  load 4 0.0 0.0 -1.0e6",
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
    assert mpco_path.exists(), f"MPCO file not produced.\n{combined}"

    # ── Read element stress through the new code path ───────────────
    from apeGmsh.results import Results
    with Results.from_mpco(mpco_path) as r:
        assert len(r.stages) >= 1
        s = r.stage(r.stages[0].id)

        # All six stress components surface through available_components.
        comps = set(s.elements.gauss.available_components())
        for name in [
            "stress_xx", "stress_yy", "stress_zz",
            "stress_xy", "stress_yz", "stress_xz",
        ]:
            assert name in comps, f"missing {name} in {sorted(comps)}"

        # FourNodeTetrahedron has 1 GP per element — slab is (T, 1).
        sxx = s.elements.gauss.get(component="stress_xx")
        assert sxx.values.shape == (1, 1), sxx.values.shape
        assert sxx.element_index.tolist() == [1]
        np.testing.assert_allclose(
            sxx.natural_coords, [[0.25, 0.25, 0.25]], atol=1e-12,
        )

        # ── Physical sanity: a 1-MN compressive load at the apex ────
        # produces non-zero stress with negative vertical (z-z) stress.
        szz = s.elements.gauss.get(component="stress_zz").values[0, 0]
        assert szz < 0.0, f"expected compressive σ_zz, got {szz}"
        assert abs(szz) > 1e3, f"σ_zz too small to be plausible: {szz}"

    # ── Sanity: filtering by element ID still works on real files ───
    with Results.from_mpco(mpco_path) as r:
        s = r.stage(r.stages[0].id)
        # Filter to an ID that does not exist → empty slab.
        empty = s.elements.gauss.get(
            component="stress_xx", ids=np.array([999]),
        )
        assert empty.values.shape == (1, 0)
