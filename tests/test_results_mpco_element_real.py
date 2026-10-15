"""Phase 11a Step B — full cycle: record stress via real OpenSees, read back.

Builds a 4-node tetrahedron model under apex compression, emits a
``recorder mpco -E stress`` command via ``ResolvedRecorderSpec``,
runs OpenSees through the El Ladruno Tcl launcher, and reads the
resulting ``.mpco`` file back through ``MPCOReader.read_gauss``.

Skipped if the launcher is not available (mirrors
``test_results_mpco_emit_real.py::test_tcl_subprocess_full_cycle``).

It checks values, not signs (#1955, gap G16): every component of the GP
equals ``eleResponse 1 stresses`` and every node equals ``nodeDisp``,
printed by the same deck after each of two load steps, at ``rtol=1e-10``.
No CI lane has a Tcl launcher; the in-process twin, which the fork lane
runs (with an 8-GP ``stdBrick`` too), is
``tests/opensees/integration_ladruno/test_nodal_gauss_oracle_live.py``.
"""
from __future__ import annotations

import os
import subprocess
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from tests.conftest import _stub_model_h5_path


_DEFAULT_TCL_LAUNCHERS = [
    r"C:\Program Files\El Ladruno OpenSees\opensees_ladruno.bat",
]

RTOL = 1e-10
N_STEPS = 2
NODE_IDS = (1, 2, 3, 4)
DISP = ("displacement_x", "displacement_y", "displacement_z")
#: ``eleResponse(e, "stresses")`` order at each GP. Spelled out, not read
#: from RESPONSE_CATALOG, which the reader under test uses.
STRESS = ("stress_xx", "stress_yy", "stress_zz",
          "stress_xy", "stress_yz", "stress_xz")
#: Base tilted: on a flat base the apex shape function's gradient is
#: (0, 0, 1) and the tet has no xy stress at all.
COORDS = np.array([
    [0.0, 0.0, 0.0],
    [1.0, 0.0, 0.2],
    [0.0, 1.0, 0.4],
    [0.3, 0.4, 1.0],
], dtype=np.float64)
APEX_LOAD = (2.0e5, -1.0e5, -1.0e6)


def _read_tcl_oracle(path: Path) -> list:
    """Parse ``<channel> <step> <id> <values...>`` lines from the deck.

    The Tcl getters print ``%35.20f``; the smallest value here (~4e-6)
    keeps 15 significant digits, well inside ``RTOL``.
    """
    live: list = [{"disp": {}, "stress": {}} for _ in range(N_STEPS)]
    for line in path.read_text(encoding="utf-8").splitlines():
        channel, step, tag, *vals = line.split()
        live[int(step)][channel][int(tag)] = np.array(vals, dtype=float)
    return live


def _close(got: np.ndarray, want: np.ndarray, msg: str) -> None:
    np.testing.assert_allclose(
        got, want, rtol=RTOL, atol=RTOL * np.abs(want).max(), err_msg=msg,
    )


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

    coords = COORDS
    node_ids = np.array(NODE_IDS, dtype=np.int64)
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

    from apeGmsh.results.spec._resolved import (
        ResolvedRecorderRecord,
        ResolvedRecorderSpec,
    )
    spec = ResolvedRecorderSpec(
        fem_snapshot_id=fem.snapshot_id,
        records=(
            ResolvedRecorderRecord(
                category="nodes", name="all_kin",
                components=DISP,
                dt=None, n_steps=None,
                node_ids=node_ids,
            ),
            ResolvedRecorderRecord(
                category="gauss", name="solid_stress",
                components=STRESS,
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
        "  foreach n {" + " ".join(map(str, NODE_IDS)) + "} {",
        '    puts $fh "disp $k $n [nodeDisp $n]"',
        "  }",
        '  puts $fh "stress $k 1 [eleResponse 1 stresses]"',
        "}",
        "close $fh",
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
    with Results.from_mpco(mpco_path, model_h5=_stub_model_h5_path()) as r:
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
        assert sxx.values.shape == (N_STEPS, 1), sxx.values.shape
        assert sxx.element_index.tolist() == [1]
        np.testing.assert_allclose(
            sxx.natural_coords, [[0.25, 0.25, 0.25]], atol=1e-12,
        )

        # ── Value oracle: the deck's own eleResponse / nodeDisp ─────
        live = _read_tcl_oracle(tmp_path / "oracle.txt")
        want_stress = np.array([step["stress"][1] for step in live])
        assert want_stress.shape == (N_STEPS, len(STRESS))
        assert np.all(want_stress != 0.0)     # no component passes as 0 == 0
        for k, comp in enumerate(STRESS):
            got = np.asarray(s.elements.gauss.get(component=comp).values)
            _close(got, want_stress[:, k:k + 1], comp)
        for k, comp in enumerate(DISP):
            slab = s.nodes.get(component=comp)
            ids = [int(n) for n in slab.node_ids]
            assert sorted(ids) == list(NODE_IDS), comp
            want = np.array([[step["disp"][n][k] for n in ids]
                             for step in live])
            _close(np.asarray(slab.values), want, comp)

    # ── Sanity: filtering by element ID still works on real files ───
    with Results.from_mpco(mpco_path, model_h5=_stub_model_h5_path()) as r:
        s = r.stage(r.stages[0].id)
        # Filter to an ID that does not exist → empty slab.
        empty = s.elements.gauss.get(
            component="stress_xx", ids=np.array([999]),
        )
        assert empty.values.shape == (N_STEPS, 0)
