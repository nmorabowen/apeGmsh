"""Phase 8 — full cycle: emit MPCO recorder, run OpenSees, read via from_mpco.

Two parallel paths for validating end-to-end MPCO emission:

1. **openseespy-in-process** — emit the Python ``ops.recorder('mpco',
   ...)`` call and run via openseespy. Skipped on builds whose
   openseespy doesn't ship MPCO support.
2. **OpenSees.exe via subprocess** — emit the Tcl ``recorder mpco
   ...`` line, write a complete Tcl script, run a separate
   ``OpenSees.exe`` process. Used when the El Ladruno build (or any
   other MPCO-capable Tcl binary) is on disk.

Set ``APEGMSH_OPENSEES_TCL`` to point at an OpenSees launcher
(typically the ``opensees_*.bat`` script) to enable the second path.
The default checks the El Ladruno install location.
"""
from __future__ import annotations

import os
import subprocess
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

openseespy = pytest.importorskip(
    "openseespy.opensees", reason="openseespy required",
)
ops = openseespy

_DEFAULT_TCL_LAUNCHERS = [
    r"C:\Program Files\El Ladruno OpenSees\opensees_ladruno.bat",
]


def _resolve_tcl_launcher() -> Path | None:
    """Find an OpenSees Tcl launcher with MPCO support."""
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


def _has_mpco_recorder_openseespy() -> bool:
    """Return True if the active openseespy build supports MPCO."""
    ops.wipe()
    ops.model("basic", "-ndm", 1, "-ndf", 1)
    ops.node(1, 0.0)
    try:
        ops.recorder("mpco", "_probe.mpco", "-N", "displacement")
        ops.wipe()
        try:
            Path("_probe.mpco").unlink(missing_ok=True)
        except Exception:
            pass
        return True
    except Exception:
        ops.wipe()
        return False


def test_openseespy_full_cycle(tmp_path: Path) -> None:
    """Emit ``ops.recorder('mpco', ...)``, run via openseespy in-process."""
    if not _has_mpco_recorder_openseespy():
        pytest.skip(
            "Active openseespy build does not include the MPCO recorder. "
            "Phase 8 emission is still validated by the snapshot tests "
            "in test_results_mpco_emit.py and the Tcl-subprocess test."
        )
    # ── Set up a tiny model directly via openseespy ──────────────────
    ops.wipe()
    ops.model("basic", "-ndm", 3, "-ndf", 3)

    coords = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.5, 0.5, 1.0],
    ], dtype=np.float64)
    node_ids = np.array([1, 2, 3, 4], dtype=np.int64)
    for nid, xyz in zip(node_ids, coords):
        ops.node(int(nid), float(xyz[0]), float(xyz[1]), float(xyz[2]))
    for nid in (1, 2, 3):
        ops.fix(nid, 1, 1, 1)
    ops.nDMaterial("ElasticIsotropic", 1, 200e9, 0.3)
    ops.element("FourNodeTetrahedron", 1, 1, 2, 3, 4, 1)

    # ── Build a synthetic FEMData matching that model ────────────────
    class _Fem:
        def __init__(self):
            self.nodes = SimpleNamespace(ids=node_ids, coords=coords)
            self.elements = []

        @property
        def snapshot_id(self):
            from apeGmsh.mesh._femdata_hash import compute_snapshot_id
            return compute_snapshot_id(self)

        def to_native_h5(self, group):
            group.attrs["snapshot_id"] = self.snapshot_id
            group.attrs["ndm"] = 3
            group.attrs["ndf"] = 3
            group.attrs["model_name"] = ""
            group.attrs["units"] = ""
            n = group.create_group("nodes")
            n.create_dataset("ids", data=self.nodes.ids)
            n.create_dataset("coords", data=self.nodes.coords)
            group.create_group("elements")

    fem = _Fem()

    # ── Build a hand-rolled recorder spec & emit the MPCO command ────
    from apeGmsh.solvers._recorder_specs import (
        ResolvedRecorderRecord,
        ResolvedRecorderSpec,
    )
    spec = ResolvedRecorderSpec(
        fem_snapshot_id=fem.snapshot_id,
        records=(ResolvedRecorderRecord(
            category="nodes", name="all_kin",
            components=("displacement_x", "displacement_y", "displacement_z"),
            dt=None, n_steps=None,
            node_ids=node_ids,
        ),),
    )

    mpco_path = tmp_path / "run.mpco"
    cmd = spec.to_mpco_python_command(filename=str(mpco_path))
    assert cmd.startswith("ops.recorder('mpco'")

    # Issue the recorder command before analysis.
    exec(cmd, {"ops": ops})

    # ── Run a single static step ─────────────────────────────────────
    ops.timeSeries("Linear", 1)
    ops.pattern("Plain", 1, 1)
    ops.load(4, 0.0, 0.0, -1e6)
    ops.system("ProfileSPD")
    ops.numberer("RCM")
    ops.constraints("Plain")
    ops.algorithm("Linear")
    ops.integrator("LoadControl", 1.0)
    ops.analysis("Static")
    ok = ops.analyze(1)
    assert ok == 0, f"analyze() returned {ok}"

    # MPCO recorder writes on close — wipe to flush.
    ops.wipe()
    assert mpco_path.exists(), f"MPCO recorder did not produce {mpco_path}"

    # ── Read the MPCO file via Phase 3's MPCOReader ──────────────────
    from apeGmsh.results import Results
    with Results.from_mpco(mpco_path) as r:
        # MPCO synthesises its own partial fem; we don't bind to ours.
        assert len(r.stages) >= 1
        # Pick the first stage and verify displacement_z
        s = r.stage(r.stages[0].id)
        slab = s.nodes.get(component="displacement_z")
        # Apex (node 4) should have moved downward.
        # Find node 4's column in the slab.
        idx = list(slab.node_ids).index(4)
        assert slab.values[0, idx] < 0


def test_tcl_subprocess_full_cycle(tmp_path: Path) -> None:
    """Emit Tcl ``recorder mpco``, run via OpenSees.exe subprocess.

    Validates that the emission language is consumable by a real
    standalone OpenSees Tcl interpreter (the El Ladruno build, by
    default — set ``APEGMSH_OPENSEES_TCL`` to override).
    """
    launcher = _resolve_tcl_launcher()
    if launcher is None:
        pytest.skip(
            "No OpenSees Tcl launcher found. Set APEGMSH_OPENSEES_TCL "
            "or install the El Ladruno OpenSees build to enable this test."
        )

    # ── Build a recorder spec against synthetic FEMData ──────────────
    coords = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.5, 0.5, 1.0],
    ], dtype=np.float64)
    node_ids = np.array([1, 2, 3, 4], dtype=np.int64)

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
        records=(ResolvedRecorderRecord(
            category="nodes", name="all_kin",
            components=("displacement_x", "displacement_y", "displacement_z"),
            dt=None, n_steps=None,
            node_ids=node_ids,
        ),),
    )

    # ── Generate the Tcl recorder command from our spec ──────────────
    mpco_path = tmp_path / "run.mpco"
    # Use a relative filename — OpenSees runs with the script's dir as CWD.
    mpco_recorder_line = spec.to_mpco_tcl_command(filename="run.mpco")
    assert mpco_recorder_line.startswith("recorder mpco run.mpco")

    # ── Hand-write the rest of the Tcl model + analysis ──────────────
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
        # The recorder line our spec produced
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
        # MPCO recorder flushes on wipe
        "wipe",
        'puts "DONE"',
    ]
    script = tmp_path / "model.tcl"
    script.write_text("\n".join(tcl_lines) + "\n", encoding="utf-8")

    # ── Run OpenSees.exe via the launcher ────────────────────────────
    result = subprocess.run(
        [str(launcher), str(script)],
        capture_output=True, text=True, timeout=60,
        cwd=str(tmp_path),
    )
    # OpenSees.exe routes its output through stderr on Windows; the
    # launcher inherits that. Check the combined streams.
    combined = (result.stdout or "") + "\n" + (result.stderr or "")
    assert result.returncode == 0, (
        f"OpenSees.exe failed (rc={result.returncode})\n{combined}"
    )
    assert "DONE" in combined, (
        f"Tcl script didn't reach DONE marker.\n{combined}"
    )
    assert mpco_path.exists(), (
        f"MPCO file not produced at {mpco_path}\n{combined}"
    )

    # ── Read back via Phase 3's MPCOReader ───────────────────────────
    from apeGmsh.results import Results
    with Results.from_mpco(mpco_path) as r:
        assert len(r.stages) >= 1
        s = r.stage(r.stages[0].id)
        slab = s.nodes.get(component="displacement_z")
        idx = list(slab.node_ids).index(4)
        # 1 MN downward at the apex of a 200 GPa, ν=0.3 tet → small but
        # non-zero downward displacement.
        assert slab.values[0, idx] < 0, (
            f"Apex did not move downward; got {slab.values[0, idx]}"
        )
