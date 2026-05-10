"""Subprocess tests for the Py path: write deck, run python, verify recorder.

Gated by ``@pytest.mark.subprocess``. The deck runs through the
opensees venv's python (``$OPENSEES_VENV``) so openseespy is
available in the subprocess.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import cast

import pytest

from apeGmsh.opensees import apeSees

from tests.opensees.fixtures.fem_stub import make_two_node_beam


def _python_for_subprocess() -> str | None:
    """Resolve the openseespy-bearing python for subprocess use."""
    venv = os.environ.get("OPENSEES_VENV")
    if venv:
        if os.name == "nt":
            candidate = os.path.join(venv, "Scripts", "python.exe")
        else:
            candidate = os.path.join(venv, "bin", "python")
        if os.path.exists(candidate):
            return candidate
    # Fallback: the python running this test (we already imported
    # openseespy at the live-test level, so this interpreter has it).
    return sys.executable


def _has_openseespy_in(python_bin: str) -> bool:
    """True if ``python_bin`` can import openseespy."""
    import subprocess
    proc = subprocess.run(
        [python_bin, "-c", "import openseespy.opensees"],
        capture_output=True,
        check=False,
    )
    return proc.returncode == 0


_PY = _python_for_subprocess()
_HAVE_OPSPY = _PY is not None and _has_openseespy_in(_PY)

pytestmark = [
    pytest.mark.subprocess,
    pytest.mark.skipif(
        not _HAVE_OPSPY,
        reason="openseespy not available in subprocess python",
    ),
]


def test_py_subprocess_returns_zero_and_writes_recorder(tmp_path: Path) -> None:
    """ops.py(path, run=True) writes the deck, subprocesses python on it,
    and the recorder's output file appears."""
    deck_path = tmp_path / "model.py"
    disp_out = tmp_path / "disp.out"

    fem = make_two_node_beam()
    ops = apeSees(cast("object", fem))  # type: ignore[arg-type]
    ops.model(ndm=3, ndf=6)
    transf = ops.geomTransf.Linear(vecxz=(1.0, 0.0, 0.0))
    ops.element.elasticBeamColumn(
        pg="Cols",
        transf=transf,
        A=0.01, E=200e9, Iz=1e-4, Iy=1e-4, G=80e9, J=1e-4,
    )
    ops.fix(pg="Base", dofs=(1, 1, 1, 1, 1, 1))
    ts = ops.timeSeries.Linear()
    with ops.pattern.Plain(series=ts) as p:
        p.load(node=2, forces=(1000.0, 0.0, 0.0, 0.0, 0.0, 0.0))
    ops.recorder.Node(
        file=str(disp_out),
        response="disp",
        nodes=(2,),
        dofs=(1, 2, 3),
    )
    ops.constraints.Plain()
    ops.numberer.Plain()
    ops.system.BandGeneral()
    ops.test.NormDispIncr(tol=1e-9, max_iter=10)
    ops.algorithm.Linear()
    ops.integrator.LoadControl(dlam=1.0)
    ops.analysis.Static()

    # Write the py deck without run=, append analyze, and subprocess.
    ops.py(str(deck_path), run=False)
    with open(deck_path, "a", encoding="utf-8") as f:
        f.write("ops.analyze(1)\n")

    import subprocess
    assert _PY is not None
    proc = subprocess.run(
        [_PY, str(deck_path)],
        capture_output=True,
        text=True,
        check=False,
        cwd=tmp_path,
    )
    assert proc.returncode == 0, (
        f"py subprocess returned {proc.returncode}.\n"
        f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
    )
    assert disp_out.exists()


def test_apesees_py_writes_valid_python(tmp_path: Path) -> None:
    """ops.py(path) without run=True writes a deck that is at least
    valid Python source (compiles)."""
    deck_path = tmp_path / "model.py"

    fem = make_two_node_beam()
    ops = apeSees(cast("object", fem))  # type: ignore[arg-type]
    ops.model(ndm=3, ndf=6)
    transf = ops.geomTransf.Linear(vecxz=(1.0, 0.0, 0.0))
    ops.element.elasticBeamColumn(
        pg="Cols",
        transf=transf,
        A=0.01, E=200e9, Iz=1e-4, Iy=1e-4, G=80e9, J=1e-4,
    )
    ops.fix(pg="Base", dofs=(1, 1, 1, 1, 1, 1))
    ops.py(str(deck_path), run=False)

    src = deck_path.read_text()
    compile(src, str(deck_path), "exec")
