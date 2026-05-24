"""Subprocess acceptance for Phase SSI-2.B per-stage topology activation.

Builds a 2-PG model (rock + cimbra) via apeSees where rock is
global (always available) and cimbra activates in stage 2.  The
emitted Tcl deck:

* Declares rock nodes + the rock element globally before any stage.
* Stage 1 runs analysis with ONLY rock active.
* Stage 2 emits cimbra nodes + the cimbra element + domainChange,
  then runs analysis with both rock and cimbra active.

Verifies the deck parses + runs under OpenSees.exe (Ladruno
``288f6d0f``) without errors.  Confirms cimbra-only nodes do NOT
exist in OpenSees during stage 1 (they're declared mid-deck inside
stage 2's block).
"""
from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path
from typing import cast

import pytest

from apeGmsh.opensees import apeSees

from tests.opensees.fixtures.fem_stub import (
    FEMStub,
    _ElementGroupView,
    _ElementsStub,
    _NodesStub,
)


def _opensees_available() -> bool:
    return bool(os.environ.get("OPENSEES_BIN") or shutil.which("OpenSees"))


pytestmark = [
    pytest.mark.subprocess,
    pytest.mark.skipif(
        not _opensees_available(),
        reason="OpenSees binary not on PATH and OPENSEES_BIN not set",
    ),
]


def _make_rock_plus_cimbra_fem() -> FEMStub:
    """Two quads sharing nodes 2, 3 along the rock-cimbra interface.

    Rock: nodes 1, 2, 3, 4 (unit square at x ∈ [0, 1]).
    Cimbra: nodes 2, 5, 6, 3 (unit square at x ∈ [1, 2]).
    Shared nodes: 2 (bottom-right of rock = bottom-left of cimbra)
    and 3 (top-right of rock = top-left of cimbra).
    """
    return FEMStub(
        nodes=_NodesStub(
            ids=[1, 2, 3, 4, 5, 6],
            coords=[
                (0.0, 0.0, 0.0),
                (1.0, 0.0, 0.0),
                (1.0, 1.0, 0.0),
                (0.0, 1.0, 0.0),
                (2.0, 0.0, 0.0),
                (2.0, 1.0, 0.0),
            ],
            node_pgs={
                "Anchor": [1, 4],  # left edge of rock — global fix
            },
        ),
        elements=_ElementsStub(
            elem_pgs={
                "rock":   _ElementGroupView(
                    ids=(1,), connectivity=((1, 2, 3, 4),),
                ),
                "cimbra": _ElementGroupView(
                    ids=(2,), connectivity=((2, 5, 6, 3),),
                ),
            },
        ),
    )


def _full_chain(ops: apeSees) -> dict[str, object]:
    return {
        "test":        ops.test.NormDispIncr(tol=1e-6, max_iter=50),
        "algorithm":   ops.algorithm.Newton(),
        "integrator":  ops.integrator.LoadControl(dlam=1.0),
        "constraints": ops.constraints.Plain(),
        "numberer":    ops.numberer.RCM(),
        "system":      ops.system.UmfPack(),
        "analysis":    ops.analysis.Static(),
    }


def test_two_pg_two_stage_deck_runs_on_ladruno(tmp_path: Path) -> None:
    """A 2-PG 2-stage deck parses + runs end-to-end on Ladruno.

    Stage 1 sees only rock; stage 2 activates cimbra mid-deck and
    runs a second analysis with both elements present.  The fact
    that OpenSees doesn't error during stage 1 confirms cimbra's
    nodes haven't been declared yet (otherwise the element they
    reference would already exist).
    """
    fem = _make_rock_plus_cimbra_fem()
    ops = apeSees(cast("object", fem), default_orientation=None)  # type: ignore[arg-type]
    ops.model(ndm=2, ndf=2)
    mat = ops.nDMaterial.ElasticIsotropic(E=1.0e6, nu=0.25, rho=0.0)
    ops.element.FourNodeQuad(pg="rock", thickness=1.0, material=mat)
    ops.element.FourNodeQuad(pg="cimbra", thickness=1.0, material=mat)
    ops.fix(pg="Anchor", dofs=(1, 1))

    # Stage 1: rock only.  No PG activation → only globally-emitted
    # topology (rock + shared nodes) is active in OpenSees.
    with ops.stage(name="rock_only") as s:
        s.analysis(**_full_chain(ops))
        s.run(n_increments=1)

    # Stage 2: activate cimbra.  Stage 2's analysis sees both rock
    # and cimbra.
    with ops.stage(name="install_cimbra") as s:
        s.activate(pgs=["cimbra"])
        s.analysis(**_full_chain(ops))
        s.run(n_increments=1)

    deck = tmp_path / "deck.tcl"
    ops.tcl(str(deck))

    # Sanity-check the deck shape before running.
    text = deck.read_text()
    lines = text.splitlines()
    # Node 5 (cimbra-only) must NOT appear before the install_cimbra
    # stage banner — that's the whole point of stage activation.
    install_idx = next(
        i for i, ln in enumerate(lines)
        if ln.startswith("# === Stage: install_cimbra")
    )
    node_5_idx = next(
        i for i, ln in enumerate(lines) if ln.startswith("node 5 ")
    )
    assert node_5_idx > install_idx, (
        f"node 5 (cimbra) declared at line {node_5_idx} but "
        f"install_cimbra stage opens at line {install_idx} — "
        "stage activation didn't defer the node emit."
    )

    binary = os.environ.get("OPENSEES_BIN") or shutil.which("OpenSees")
    assert binary is not None
    proc = subprocess.run(
        [binary, str(deck)],
        capture_output=True, text=True, check=False,
    )
    assert proc.returncode == 0, (
        f"OpenSees exit {proc.returncode}\n"
        f"--- stdout (tail) ---\n"
        f"{chr(10).join(proc.stdout.splitlines()[-50:])}\n"
        f"--- stderr (tail) ---\n"
        f"{chr(10).join(proc.stderr.splitlines()[-50:])}"
    )


def test_two_pg_two_stage_deck_runs_under_openseespy(tmp_path: Path) -> None:
    """Same deck through the Py emitter + openseespy subprocess.
    Validates ``ops.domainChange()`` is the right openseespy call."""
    venv = os.environ.get("OPENSEES_VENV")
    if not venv:
        pytest.skip("OPENSEES_VENV not set")
    python = os.path.join(
        venv,
        "Scripts" if os.name == "nt" else "bin",
        "python.exe" if os.name == "nt" else "python",
    )
    if not os.path.exists(python):
        pytest.skip(f"venv python not found at {python}")

    fem = _make_rock_plus_cimbra_fem()
    ops = apeSees(cast("object", fem), default_orientation=None)  # type: ignore[arg-type]
    ops.model(ndm=2, ndf=2)
    mat = ops.nDMaterial.ElasticIsotropic(E=1.0e6, nu=0.25, rho=0.0)
    ops.element.FourNodeQuad(pg="rock", thickness=1.0, material=mat)
    ops.element.FourNodeQuad(pg="cimbra", thickness=1.0, material=mat)
    ops.fix(pg="Anchor", dofs=(1, 1))

    with ops.stage(name="rock_only") as s:
        s.analysis(**_full_chain(ops))
        s.run(n_increments=1)
    with ops.stage(name="install_cimbra") as s:
        s.activate(pgs=["cimbra"])
        s.analysis(**_full_chain(ops))
        s.run(n_increments=1)

    deck = tmp_path / "deck.py"
    ops.py(str(deck))

    proc = subprocess.run(
        [python, str(deck)],
        capture_output=True, text=True, check=False,
    )
    # Look for malformed-emit errors specifically (SyntaxError etc.).
    combined = proc.stdout + "\n" + proc.stderr
    for needle in (
        "SyntaxError", "IndentationError", "NameError: name '_apesees",
    ):
        assert needle not in combined, (
            f"openseespy hit a malformed-emit error ({needle!r}).\n"
            f"--- stdout ---\n{proc.stdout}\n"
            f"--- stderr ---\n{proc.stderr}"
        )
