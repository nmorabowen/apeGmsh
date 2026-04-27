"""Phase 11a Step D — full Tcl-subprocess cycle for the gauss transcoder.

Builds a 4-node tet model under apex compression, emits the matching
``recorder Element ... stresses`` Tcl line via ``ResolvedRecorderSpec``,
runs OpenSees through the El Ladruno launcher, then runs
``RecorderTranscoder`` over the emitted ``.out`` file. Reads the
resulting native HDF5 through ``Results.from_native`` and asserts the
same numbers the MPCO real-file and DomainCapture real-openseespy
tests produced — full three-way agreement on identical physics.
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
        group.attrs["ndf"] = 3
        group.attrs["model_name"] = ""
        group.attrs["units"] = ""
        n = group.create_group("nodes")
        n.create_dataset("ids", data=self.nodes.ids)
        n.create_dataset("coords", data=self.nodes.coords)
        group.create_group("elements")


def test_tet_stress_full_cycle_through_transcoder(tmp_path: Path) -> None:
    """Tcl emit → OpenSees.exe → transcode .out → read back via composite."""
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
    fem = _MinimalFem(node_ids=node_ids, coords=coords)

    from apeGmsh.solvers._recorder_specs import (
        ResolvedRecorderRecord,
        ResolvedRecorderSpec,
    )
    spec = ResolvedRecorderSpec(
        fem_snapshot_id=fem.snapshot_id,
        records=(
            ResolvedRecorderRecord(
                category="gauss", name="solid_stress",
                components=(
                    "stress_xx", "stress_yy", "stress_zz",
                    "stress_xy", "stress_yz", "stress_xz",
                ),
                dt=None, n_steps=None,
                element_ids=np.array([1]),
                # FourNodeTet shares flat_size=6 with SSPbrick;
                # the .out transcoder needs the hint to disambiguate.
                element_class_name="FourNodeTetrahedron",
            ),
        ),
    )

    out_dir = tmp_path / "out"
    out_dir.mkdir()
    # The spec emits the `recorder Element ...` line; we paste it into
    # the Tcl model alongside the rest of the analysis.
    [recorder_line] = spec.to_tcl_commands(output_dir=str(out_dir).replace("\\", "/"))
    assert recorder_line.startswith("recorder Element ")
    assert recorder_line.rstrip().endswith("stresses  ;# solid_stress gauss")

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
        recorder_line,
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
        "remove recorders",   # flush the .out file before exit
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

    out_file = out_dir / "solid_stress_gauss.out"
    assert out_file.exists(), (
        f"Recorder output file not produced at {out_file}\n{combined}"
    )

    # ── Transcode ────────────────────────────────────────────────────
    from apeGmsh.results.transcoders._recorder import RecorderTranscoder
    target = tmp_path / "transcoded.h5"
    RecorderTranscoder(
        spec, out_dir, target, fem,
        stage_name="static_load", stage_kind="static",
    ).run()

    # ── Read back ────────────────────────────────────────────────────
    from apeGmsh.results import Results
    with Results.from_native(target) as r:
        assert len(r.stages) == 1
        s = r.stage(r.stages[0].id)

        sxx = s.elements.gauss.get(component="stress_xx")
        assert sxx.values.shape == (1, 1)
        assert sxx.element_index.tolist() == [1]
        np.testing.assert_allclose(
            sxx.natural_coords, [[0.25, 0.25, 0.25]], atol=1e-12,
        )

        # Match the MPCO real-file and DomainCapture real-ops numbers
        # for this exact model (E=200 GPa, ν=0.3, 1 MN apex compression).
        sxx_v = float(sxx.values[0, 0])
        syy_v = float(s.elements.gauss.get(component="stress_yy").values[0, 0])
        szz_v = float(s.elements.gauss.get(component="stress_zz").values[0, 0])
        sxy_v = float(s.elements.gauss.get(component="stress_xy").values[0, 0])

        np.testing.assert_allclose(sxx_v, -2.571428571428e6, rtol=1e-6)
        np.testing.assert_allclose(syy_v, -2.571428571428e6, rtol=1e-6)
        np.testing.assert_allclose(szz_v, -6.0e6, rtol=1e-6)
        assert abs(sxy_v) < 1e-6
