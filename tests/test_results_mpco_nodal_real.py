"""Phase 11b Step 3a — full cycle: real ElasticBeam3d .mpco read.

Builds a 3D ElasticBeam3d cantilever under a transverse tip load,
runs OpenSees through the El Ladruno Tcl launcher to produce a real
``.mpco`` file with both ``globalForce`` and ``localForce`` buckets,
then reads them back through ``MPCOReader.read_elements``. Verifies
physical correctness (axial / shear / moment equilibrium at end nodes)
and that frame-distinguished canonical names route to the right buckets.

Skipped if the launcher is not available.
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


def test_elastic_beam_3d_global_and_local_force(tmp_path: Path) -> None:
    """Cantilever ElasticBeam3d with combined transverse + axial tip load.

    With the beam along the global x axis and a load
    ``P = (Fx=1000, Fy=500, Fz=0, Mx=My=Mz=0)`` at node 2 (free end):

    - The cantilever's reaction at node 1 must balance the load.
      In global frame: F_n1 = (-Fx, -Fy, 0, 0, 0, -Fy*L), and
      F_n2 = (+Fx, +Fy, 0, 0, 0, 0).
    - In element-local frame (local x = global x for this geometry):
      N=Fx (axial), Vy=Fy (transverse shear), Vz=0, T=My=0,
      Mz at node 1 = -Fy*L (cantilever bending moment).
    """
    launcher = _resolve_tcl_launcher()
    if launcher is None:
        pytest.skip("No OpenSees Tcl launcher found.")

    L = 5.0
    Fx, Fy = 1.0e3, 5.0e2

    tcl_lines = [
        "wipe",
        "model BasicBuilder -ndm 3 -ndf 6",
        "node 1 0.0 0.0 0.0",
        f"node 2 {L} 0.0 0.0",
        "fix 1 1 1 1 1 1 1",
        "geomTransf Linear 1 0.0 1.0 0.0",
        # ElasticBeam3d: A E G Iz Iy J transfTag.
        "element elasticBeamColumn 1 1 2 0.01 200e9 80e9 1.0e-4 1.0e-4 1.5e-4 1",
        "recorder mpco run.mpco -N displacement -E globalForce localForce -T dt 1.0",
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
    mpco_path = tmp_path / "run.mpco"
    assert mpco_path.exists()

    from apeGmsh.results import Results
    with Results.from_mpco(mpco_path) as r:
        s = r.stage(r.stages[0].id)
        comps = set(s.elements.available_components())

        # Both global and local frames are exposed.
        for n in (
            "nodal_resisting_force_x", "nodal_resisting_force_y",
            "nodal_resisting_force_z", "nodal_resisting_moment_x",
            "nodal_resisting_moment_y", "nodal_resisting_moment_z",
            "nodal_resisting_force_local_x",
            "nodal_resisting_force_local_y",
            "nodal_resisting_force_local_z",
            "nodal_resisting_moment_local_x",
            "nodal_resisting_moment_local_y",
            "nodal_resisting_moment_local_z",
        ):
            assert n in comps, f"missing {n} in {sorted(comps)}"

        # ── Global frame: Fx at end nodes ────────────────────────
        slab_fx = s.elements.get(component="nodal_resisting_force_x")
        # 1 element × 2 nodes × 1 step.
        assert slab_fx.values.shape == (1, 1, 2)
        # Internal forces: at node 1 (fixed), the element pulls inward by
        # -Fx; at node 2 (loaded), it pushes outward by +Fx (the
        # signs depend on OpenSees's sign convention but magnitudes
        # equal Fx).
        np.testing.assert_allclose(
            np.abs(slab_fx.values[0, 0]), [Fx, Fx], rtol=1e-9, atol=1e-6,
        )

        # ── Global frame: Fy at end nodes ────────────────────────
        slab_fy = s.elements.get(component="nodal_resisting_force_y")
        np.testing.assert_allclose(
            np.abs(slab_fy.values[0, 0]), [Fy, Fy], rtol=1e-9, atol=1e-6,
        )

        # ── Local frame: axial = Fx (since beam is along global x) ──
        slab_n = s.elements.get(component="nodal_resisting_force_local_x")
        np.testing.assert_allclose(
            np.abs(slab_n.values[0, 0]), [Fx, Fx], rtol=1e-9, atol=1e-6,
        )

        # ── Mz at node 1 should equal -Fy*L (cantilever moment); at node 2 = 0
        slab_mz = s.elements.get(component="nodal_resisting_moment_z")
        node1_mz = slab_mz.values[0, 0, 0]
        node2_mz = slab_mz.values[0, 0, 1]
        np.testing.assert_allclose(
            abs(node1_mz), Fy * L, rtol=1e-9, atol=1e-6,
        )
        assert abs(node2_mz) < 1e-6, (
            f"Mz at free end must be ~0; got {node2_mz}"
        )

        # ── Element ID filter survives the read ────────────────
        slab_filtered = s.elements.get(
            component="nodal_resisting_force_x", ids=np.array([1]),
        )
        assert slab_filtered.element_ids.tolist() == [1]
        slab_empty = s.elements.get(
            component="nodal_resisting_force_x", ids=np.array([999]),
        )
        assert slab_empty.element_ids.size == 0
