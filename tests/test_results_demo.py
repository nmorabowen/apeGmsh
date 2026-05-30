"""make_demo_results / Results.demo — zero-setup sample results.

Headless: builds the demo (mesh + apeSees model emit + synthetic
pushover, no OpenSees solve, no render) and checks the shape of the
result — one pushover stage, the right step count, and a deflection
that ramps from zero to the requested tip drift. The viewer rendering
itself is covered by the WebViewer / viewer tests.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

# apeSees model emit needs the opensees bridge; mesh needs gmsh (base dep).
pytest.importorskip("openseespy.opensees", reason="apeSees bridge import")


def test_make_demo_results_shape():
    from apeGmsh.results import make_demo_results

    r = make_demo_results(length=10.0, n_elements=6, n_steps=5, tip_drift=2.0)
    slab = r.nodes.get(component="displacement_x")
    vals = np.asarray(slab.values, dtype=np.float64)   # (n_steps, n_nodes)
    assert vals.shape[0] == 5
    # Step 0 is the undeformed reference.
    np.testing.assert_allclose(vals[0], 0.0, atol=1e-12)
    # Last step is non-trivially deflected.
    assert np.abs(vals[-1]).max() > 0.0
    # Monotone ramp at the most-deflected node.
    tip_col = int(np.argmax(np.abs(vals[-1])))
    series = vals[:, tip_col]
    assert np.all(np.diff(series) >= -1e-12)
    # Tip drift lands near the requested amplitude (analytic shape == 1.0
    # at the free end, ramped to tip_drift).
    np.testing.assert_allclose(np.abs(series[-1]), 2.0, rtol=0.05)
    r.close()


def test_results_demo_classmethod():
    from apeGmsh.results import Results

    r = Results.demo(n_steps=3, n_elements=4)
    slab = r.nodes.get(component="displacement_x")
    assert np.asarray(slab.values).shape[0] == 3
    r.close()


def test_demo_rejects_bad_args():
    from apeGmsh.results import make_demo_results

    with pytest.raises(ValueError):
        make_demo_results(n_steps=0)
    with pytest.raises(ValueError):
        make_demo_results(n_elements=0)


def test_demo_records_model_path_in_viewer_argv():
    """The non-blocking subprocess viewer must forward --model-h5 to the
    sibling demo_model.h5, since the composed demo_results.h5 has no
    independently-readable /model zone."""
    from apeGmsh.results import Results

    r = Results.demo(n_steps=3, n_elements=4)
    try:
        assert r._model_path is not None
        assert r._model_path.name == "demo_model.h5"
        argv = r._build_viewer_argv(title=None)
        assert "--model-h5" in argv
        assert str(r._model_path) in argv
    finally:
        r.close()


def test_demo_subprocess_viewer_opens():
    """Regression: `Results.demo().viewer(blocking=False)` crashed the child
    with MalformedH5Error (/model zone empty) because --model-h5 was never
    forwarded. Run the real child argv (APEGMSH_SKIP_VIEWER so it opens the
    results then skips the GUI) and assert it exits 0."""
    import os
    import subprocess
    from apeGmsh.results import Results

    r = Results.demo(n_steps=3, n_elements=4)
    argv = r._build_viewer_argv(title=None)
    r.close()  # release the file handle before the child opens it (Windows)

    env = dict(os.environ, APEGMSH_SKIP_VIEWER="1")
    src = str(Path(__file__).resolve().parents[1] / "src")
    env["PYTHONPATH"] = src + os.pathsep + env.get("PYTHONPATH", "")
    proc = subprocess.run(argv, capture_output=True, text=True, env=env)
    assert proc.returncode == 0, proc.stderr[-2000:]
