"""The single-node damping check of the San Ramón T2S springs, on the engine
(``Tier2_springs.md`` section 6; ADR 0119).

One unit mass on a ``spring_bed`` spring per direction: k = 100, 400 and
900 with c = 2, 4 and 0, so ``zeta = c / (2 sqrt(k m)) = 0.1, 0.1, 0``. A
global ``rayleigh 0 0 1.0 0`` is active: with ``beta_K = 1`` a spring that
took the Rayleigh term would be overdamped (``beta_K * omega / 2 = 5`` in
X), so a measured 0.1 says the dashpot alone damps it. The oracle is the
logarithmic decrement of the free vibration from an initial velocity,
read off the recorder file; nothing in it comes from apeGmsh.

The patched decks gave 0.1000 / 0.0999 / 0 with the local Ladruno binary.
"""
from __future__ import annotations

import math
import os
import subprocess
from pathlib import Path

import numpy as np
import pytest

from apeGmsh import apeGmsh
from apeGmsh.opensees import apeSees

K = (100.0, 400.0, 900.0)
C = (2.0, 4.0, 0.0)
M = 1.0


def _dist_bin() -> "Path | None":
    d = os.environ.get("APEGMSH_OPENSEES_BIN")
    if not d:
        return None
    p = Path(d)
    return p if (p / "OpenSees.exe").is_file() else None


pytestmark = [
    pytest.mark.subprocess,
    pytest.mark.skipif(
        _dist_bin() is None,
        reason="APEGMSH_OPENSEES_BIN unset or does not hold OpenSees.exe",
    ),
]


def _zeta(x: np.ndarray) -> float:
    """Damping ratio from the logarithmic decrement of the positive peaks."""
    i = np.where((x[1:-1] > x[:-2]) & (x[1:-1] >= x[2:]) & (x[1:-1] > 0))[0] + 1
    peaks = x[i]
    assert len(peaks) >= 3, peaks
    n = len(peaks) - 1
    delta = math.log(peaks[0] / peaks[-1]) / n
    return delta / math.sqrt(4.0 * math.pi ** 2 + delta ** 2)


def test_dashpot_damps_and_rayleigh_does_not_reach_the_spring(tmp_path) -> None:
    with apeGmsh(model_name="spring_bed_zeta", verbose=False) as g:
        g.model.geometry.add_rectangle(0.0, 0.0, 0.0, 1.0, 1.0, label="p")
        g.physical.add_surface("p", name="P")
        gnd = g.decouple_node_set("P", label="ground")
        g.mesh.recipe.structured(size=1.0, fallback="strict")
        fem = g.mesh.queries.get_fem_data(dim=2)
    ops = apeSees(fem)
    ops.model(ndm=3, ndf=3)
    ops.spring_bed(gnd, k=K, c=C)                 # do_rayleigh=False
    ops.mass(pg="P", values=(M, M, M))
    ops.tcl(str(tmp_path / "model.tcl"))

    probe = gnd.source_ids[0]
    (tmp_path / "run.tcl").write_text(
        "source model.tcl\n"
        + "".join(f"setNodeVel {probe} {d} 1.0 -commit\n" for d in (1, 2, 3))
        + "rayleigh 0.0 0.0 1.0 0.0\n"
        f"recorder Node -file disp.out -time -node {probe} -dof 1 2 3 disp\n"
        "constraints Plain\nnumberer Plain\nsystem BandGeneral\n"
        "test NormDispIncr 1e-12 10\nalgorithm Linear\n"
        "integrator Newmark 0.5 0.25\nanalysis Transient\n"
        "set ok [analyze 4000 0.001]\nputs \"RESULT $ok\"\n",
        encoding="utf-8",
    )
    dist = _dist_bin()
    assert dist is not None
    r = subprocess.run(
        [str(dist / "OpenSees.exe"), "run.tcl"], cwd=tmp_path,
        capture_output=True, text=True, timeout=300,
    )
    out = r.stdout + r.stderr
    assert "RESULT 0" in out, out[-2000:]
    data = np.loadtxt(tmp_path / "disp.out")
    for col, (k, c) in enumerate(zip(K, C), start=1):
        expected = c / (2.0 * math.sqrt(k * M))
        assert _zeta(data[:, col]) == pytest.approx(expected, abs=2e-4), col
