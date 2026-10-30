"""X #2613: ``recorder.Ladruno(precision="f32")`` writes a float32 ``.ladruno``
on the fork; the default writes float64 and emits no flag.

The two-node truss of ``tests/test_results_live_default_format.py`` (tip
displacement ``P L / (E A)`` in closed form) runs twice on the fork, once per
precision, each with a bridge ``recorder.Ladruno`` on the nodal displacement.

Oracles, read with h5py from the file the fork wrote:

* ``INFO@STORED_PRECISION`` is the token asked for (the fork stamps the
  *effective* precision there, ``LadrunoRecorder.cpp``).
* The per-step ``RESULTS/ON_NODES/DISPLACEMENT/DATA`` dataset is float32
  under f32 and float64 under f64; the model, ``TIME`` and ``ID`` stay f64 /
  int (``-precision`` touches the result DATA only).
* The f64 tip equals the closed form to 1e-12; the f32 data is exactly the
  float32 cast of the f64 data (the lossy cast is the only difference), so
  the two runs agree to float32 resolution.
* The deck line carries ``-precision f32`` iff f32.
* ``Results.from_ladruno`` opens the f32 file and reads the same tip history
  (the reader does not depend on the stored element type).

Gated by ``ladruno_fork`` (the root conftest auto-skips off-fork; the
nightly ``live-fork`` lane runs it on the pinned build).
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pytest

from apeGmsh.results import Results

from tests.test_results_live_default_format import (
    TRUSS_STEPS,
    TRUSS_TIP,
    tip_x,
    truss_bridge,
    truss_fem,
)

pytestmark = pytest.mark.ladruno_fork

_DATA = "MODEL_STAGE[1]/RESULTS/ON_NODES/DISPLACEMENT/DATA"
_TIME = "MODEL_STAGE[1]/RESULTS/ON_NODES/DISPLACEMENT/TIME"
_COORDS = "MODEL_STAGE[1]/MODEL/NODES/COORDINATES"


def _run(tmp_path: Path, precision: str) -> "tuple[Path, str, float]":
    """Run the truss with a Ladruno recorder at ``precision``; return the
    ``.ladruno`` path, the deck's recorder line and the live tip."""
    from apeGmsh.opensees.emitter.live import get_ops

    ops = get_ops()
    bridge = truss_bridge(truss_fem())
    path = tmp_path / f"run_{precision}.ladruno"
    bridge.recorder.Ladruno(file=str(path), nodal_responses=("displacement",),
                            precision=precision)
    deck = tmp_path / f"deck_{precision}.tcl"
    bridge.tcl(str(deck))
    (line,) = [ln for ln in deck.read_text(encoding="utf-8").splitlines()
               if ln.startswith("recorder ")]
    bridge.run()
    try:
        assert ops.analyze(TRUSS_STEPS) == 0
        tip = float(ops.nodeDisp(2, 1))
    finally:
        ops.wipe()
    assert path.exists(), path
    return path, line, tip


def _read(path: Path) -> "tuple[str, np.ndarray, Any, Any]":
    with h5py.File(path, "r") as f:
        stored = f["INFO"].attrs["STORED_PRECISION"]
        stored = stored[0] if isinstance(stored, np.ndarray) else stored
        if isinstance(stored, bytes):
            stored = stored.decode()
        data = f[_DATA][()]
        return str(stored), data, f[_TIME].dtype, f[_COORDS].dtype


def test_f32_writes_a_float32_ladruno_and_f64_stays_the_default(
    tmp_path: Path,
) -> None:
    p64, line64, tip64 = _run(tmp_path, "f64")
    p32, line32, tip32 = _run(tmp_path, "f32")
    assert " -precision f32 " in line32, line32
    assert "precision" not in line64, line64
    # Same domain, same solve: the live answer is the closed form both times.
    assert tip64 == pytest.approx(TRUSS_TIP, rel=1e-12)
    assert tip32 == pytest.approx(TRUSS_TIP, rel=1e-12)

    stored64, data64, time64, coords64 = _read(p64)
    stored32, data32, time32, coords32 = _read(p32)
    assert (stored64, stored32) == ("f64", "f32")
    assert data64.dtype == np.float64 and data32.dtype == np.float32
    assert data64.shape == data32.shape == (TRUSS_STEPS, 2, 2)
    # -precision touches the per-step result DATA only.
    assert time64 == time32 == np.float64
    assert coords64 == coords32 == np.float64

    # Tip Ux (node 2, component 0) at the last step: f64 exact; f32 is the
    # float32 cast of the f64 data and nothing else.
    assert data64[-1, 1, 0] == pytest.approx(TRUSS_TIP, rel=1e-12)
    np.testing.assert_array_equal(data32, data64.astype(np.float32))
    np.testing.assert_allclose(data32.astype(np.float64), data64, rtol=1e-6,
                               atol=0.0)
    assert float(data32[-1, 1, 0]) == pytest.approx(TRUSS_TIP, rel=1e-6)
    assert float(data32[-1, 1, 0]) != data64[-1, 1, 0]

    # The results reader opens the f32 file like any other .ladruno.
    res = Results.from_ladruno(str(p32))
    try:
        x = tip_x(res)
    finally:
        res.close()
    np.testing.assert_array_equal(x.astype(np.float32), data32[:, 1, 0])
    assert x[-1] == pytest.approx(TRUSS_TIP, rel=1e-6)
