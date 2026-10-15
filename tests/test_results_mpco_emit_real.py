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

Both paths check values, not signs (#1955, gap G16): every recorded node
and component of ``displacement`` equals ``nodeDisp`` at the same step, and
``reaction_force`` equals ``nodeReaction`` after ``reactions``, at
``rtol=1e-10`` over two load steps. No CI lane runs this file (pip
``openseespy`` has no MPCO recorder); the fork lane runs the same oracle
in-process in ``tests/opensees/integration_ladruno/
test_nodal_gauss_oracle_live.py``.
"""
from __future__ import annotations

import os
import subprocess
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from tests.conftest import _stub_model_h5_path

openseespy = pytest.importorskip(
    "openseespy.opensees", reason="openseespy required",
)
ops = openseespy

_DEFAULT_TCL_LAUNCHERS = [
    r"C:\Program Files\El Ladruno OpenSees\opensees_ladruno.bat",
]

RTOL = 1e-10
N_STEPS = 2
NODE_IDS = (1, 2, 3, 4)
DISP = ("displacement_x", "displacement_y", "displacement_z")
REAC = ("reaction_force_x", "reaction_force_y", "reaction_force_z")
#: Base tilted (on a flat base the apex moves the strain's zz/xz/yz terms
#: only) and a load with every component non-zero.
COORDS = np.array([
    [0.0, 0.0, 0.0],
    [1.0, 0.0, 0.2],
    [0.0, 1.0, 0.4],
    [0.3, 0.4, 1.0],
], dtype=np.float64)
APEX_LOAD = (2.0e5, -1.0e5, -1.0e6)


def _assert_nodes_match(r, live: list) -> None:
    """Every node and component of every step equals the live domain.

    ``live[k]`` is ``{"disp": {nid: (ux, uy, uz)}, "reac": {...}}`` captured
    after step ``k``.
    """
    (stage,) = r.stages
    s = r.stage(stage.id)
    for channel, comps in (("disp", DISP), ("reac", REAC)):
        for k, comp in enumerate(comps):
            slab = s.nodes.get(component=comp)
            ids = [int(n) for n in slab.node_ids]
            assert sorted(ids) == list(NODE_IDS), comp
            got = np.asarray(slab.values)
            want = np.array([[step[channel][n][k] for n in ids]
                             for step in live])
            assert got.shape == want.shape == (N_STEPS, len(NODE_IDS))
            np.testing.assert_allclose(
                got, want, rtol=RTOL, atol=RTOL * np.abs(want).max(),
                err_msg=comp,
            )


def _read_tcl_oracle(path: Path) -> list:
    """Parse ``<channel> <step> <id> <values...>`` lines from a Tcl deck.

    ``nodeDisp`` / ``nodeReaction`` print ``%35.20f``: the smallest value
    here (~4e-6) keeps 15 significant digits, well inside ``RTOL``.
    """
    live: list = [{"disp": {}, "reac": {}, "stress": {}}
                  for _ in range(N_STEPS)]
    for line in path.read_text(encoding="utf-8").splitlines():
        channel, step, tag, *vals = line.split()
        live[int(step)][channel][int(tag)] = np.array(vals, dtype=float)
    return live


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

    coords = COORDS
    node_ids = np.array(NODE_IDS, dtype=np.int64)
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
            from apeGmsh.mesh._femdata_h5_io import write_neutral_zone_into_group
            write_neutral_zone_into_group(self, group, ndf=3)

    fem = _Fem()

    # ── Build a hand-rolled recorder spec & emit the MPCO command ────
    from apeGmsh.results.spec._resolved import (
        ResolvedRecorderRecord,
        ResolvedRecorderSpec,
    )
    spec = ResolvedRecorderSpec(
        fem_snapshot_id=fem.snapshot_id,
        records=(ResolvedRecorderRecord(
            category="nodes", name="all_kin",
            components=DISP + REAC,
            dt=None, n_steps=None,
            node_ids=node_ids,
        ),),
    )

    mpco_path = tmp_path / "run.mpco"
    cmd = spec.to_mpco_python_command(filename=str(mpco_path))
    assert cmd.startswith("ops.recorder('mpco'")
    assert "'reactionForce'" in cmd

    # Issue the recorder command before analysis.
    exec(cmd, {"ops": ops})

    # ── Run two static steps, capturing the live domain after each ───
    ops.timeSeries("Linear", 1)
    ops.pattern("Plain", 1, 1)
    ops.load(4, *APEX_LOAD)
    ops.system("ProfileSPD")
    ops.numberer("RCM")
    ops.constraints("Plain")
    ops.algorithm("Linear")
    ops.integrator("LoadControl", 1.0 / N_STEPS)
    ops.analysis("Static")
    live = []
    for _ in range(N_STEPS):
        ok = ops.analyze(1)
        assert ok == 0, f"analyze() returned {ok}"
        ops.reactions()
        live.append({
            "disp": {n: ops.nodeDisp(n) for n in NODE_IDS},
            "reac": {n: ops.nodeReaction(n) for n in NODE_IDS},
        })

    # MPCO recorder writes on close — wipe to flush.
    ops.wipe()
    assert mpco_path.exists(), f"MPCO recorder did not produce {mpco_path}"

    # ── Read the MPCO file via Phase 3's MPCOReader ──────────────────
    from apeGmsh.results import Results
    with Results.from_mpco(mpco_path, model_h5=_stub_model_h5_path()) as r:
        # MPCO synthesises its own partial fem; we don't bind to ours.
        _assert_nodes_match(r, live)


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
    coords = COORDS
    node_ids = np.array(NODE_IDS, dtype=np.int64)

    class _Fem:
        def __init__(self):
            self.nodes = SimpleNamespace(ids=node_ids, coords=coords)
            self.elements = []

        @property
        def snapshot_id(self):
            from apeGmsh.mesh._femdata_hash import compute_snapshot_id
            return compute_snapshot_id(self)

    fem = _Fem()

    from apeGmsh.results.spec._resolved import (
        ResolvedRecorderRecord,
        ResolvedRecorderSpec,
    )
    spec = ResolvedRecorderSpec(
        fem_snapshot_id=fem.snapshot_id,
        records=(ResolvedRecorderRecord(
            category="nodes", name="all_kin",
            components=DISP + REAC,
            dt=None, n_steps=None,
            node_ids=node_ids,
        ),),
    )

    # ── Generate the Tcl recorder command from our spec ──────────────
    mpco_path = tmp_path / "run.mpco"
    # Use a relative filename — OpenSees runs with the script's dir as CWD.
    mpco_recorder_line = spec.to_mpco_tcl_command(filename="run.mpco")
    assert mpco_recorder_line.startswith("recorder mpco run.mpco")
    assert "reactionForce" in mpco_recorder_line

    # ── Hand-write the rest of the Tcl model + analysis ──────────────
    # The deck writes the live domain's answer after every step to
    # ``oracle.txt``: the oracle the recorded file is checked against.
    tcl_lines = [
        "wipe",
        "model BasicBuilder -ndm 3 -ndf 3",
        *(f"node {n} {x!r} {y!r} {z!r}"
          for n, (x, y, z) in zip(NODE_IDS, COORDS.tolist())),
        "fix 1 1 1 1",
        "fix 2 1 1 1",
        "fix 3 1 1 1",
        "nDMaterial ElasticIsotropic 1 200e9 0.3",
        "element FourNodeTetrahedron 1 1 2 3 4 1",
        # The recorder line our spec produced
        mpco_recorder_line,
        "timeSeries Linear 1",
        "pattern Plain 1 1 {",
        "  load 4 {} {} {}".format(*map(repr, APEX_LOAD)),
        "}",
        "system ProfileSPD",
        "numberer RCM",
        "constraints Plain",
        "algorithm Linear",
        f"integrator LoadControl {1.0 / N_STEPS!r}",
        "analysis Static",
        "set fh [open oracle.txt w]",
        f"for {{set k 0}} {{$k < {N_STEPS}}} {{incr k}} {{",
        "  if {[analyze 1] != 0} { error analyze }",
        "  reactions",
        "  foreach n {" + " ".join(map(str, NODE_IDS)) + "} {",
        '    puts $fh "disp $k $n [nodeDisp $n]"',
        '    puts $fh "reac $k $n [nodeReaction $n]"',
        "  }",
        "}",
        "close $fh",
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
    live = _read_tcl_oracle(tmp_path / "oracle.txt")
    with Results.from_mpco(mpco_path, model_h5=_stub_model_h5_path()) as r:
        _assert_nodes_match(r, live)
