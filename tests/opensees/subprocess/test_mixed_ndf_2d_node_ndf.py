"""A mixed-ndf 2-D deck must land the per-node ndf it declares.

apeGmsh stored every node as an ``(x, y, z)`` triple and wrote all three
onto every ``node`` line.  In a ``-ndm 2`` deck OpenSees' Tcl ``node``
command consumes TWO coordinates and then scans what follows for
optional keywords; the trailing ``0.0`` desynchronises that scan and the
``-ndf K`` behind it is **silently swallowed**.  The node still lands
with ``getCrds().Size() == 2``, so nothing about the geometry looks
wrong — only the per-node ndf is quietly replaced by the model envelope,
and the one deck line you would read to diagnose that is exactly the
line that did nothing.

The assertions below therefore read the **resolved** ndf —
``llength [nodeDisp $tag]``, the DOF count OpenSees actually gave the
node — and never the emitted ``-ndf`` token.

This has to run through the real Tcl binary: measured 2026-08-18 the
openseespy ``ops.node`` path parses the same over-long argument list
correctly, so a ``LiveOpsEmitter`` gate would pass on the broken build
and prove nothing.  ``test_the_swallow_is_real`` is the falsifier that
keeps the main assertion from going vacuous if that ever changes.
"""
from __future__ import annotations

import os
import re
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


def _opensees_binary() -> str | None:
    return os.environ.get("OPENSEES_BIN") or shutil.which("OpenSees")


pytestmark = [
    pytest.mark.subprocess,
    pytest.mark.skipif(
        _opensees_binary() is None,
        reason="OpenSees binary not on PATH and OPENSEES_BIN not set",
    ),
]

#: Truss nodes (ndf 2, below the envelope) and beam nodes (ndf 3, AT the
#: envelope, so their ``-ndf`` token is elided by design).
TRUSS_NODES = (1, 2)
BEAM_NODES = (3, 4)


def _mixed_ndf_2d_fem() -> FEMStub:
    """Two disjoint line elements in the z=0 plane: a truss and a beam.

    In ``ndm=2`` the truss infers ndf 2 and the beam infers ndf 3
    (``_element_capabilities`` ``ndf_required``), so with an envelope of
    3 the truss nodes are the ones carrying a real ``-ndf 2`` override —
    the population the swallow silently promoted to 3.
    """
    return FEMStub(
        nodes=_NodesStub(
            ids=[1, 2, 3, 4],
            coords=[
                (0.0, 0.0, 0.0), (1.0, 0.0, 0.0),
                (0.0, 1.0, 0.0), (0.0, 2.0, 0.0),
            ],
            node_pgs={"Anchors": [1, 3], "Free": [2, 4]},
        ),
        elements=_ElementsStub(
            elem_pgs={
                "Bars": _ElementGroupView(ids=(1,), connectivity=((1, 2),)),
                "Beams": _ElementGroupView(ids=(2,), connectivity=((3, 4),)),
            },
        ),
    )


def _mixed_ndf_2d_ops(fem: FEMStub) -> apeSees:
    ops = apeSees(cast("object", fem), default_orientation=None)
    ops.model(ndm=2, ndf=3)
    mat = ops.uniaxialMaterial.ElasticMaterial(E=200e9)
    ops.element.Truss(pg="Bars", A=0.01, material=mat)
    ops.element.elasticBeamColumn(
        pg="Beams", transf=ops.geomTransf.Linear(),
        A=0.01, E=200e9, Iz=1e-4,
    )
    ops.fix(pg="Anchors", dofs=(1, 1))
    return ops


def _run_tcl(deck: Path, cwd: Path) -> str:
    binary = _opensees_binary()
    assert binary is not None
    proc = subprocess.run(
        [binary, str(deck)],
        capture_output=True, text=True, check=False, cwd=cwd,
    )
    assert proc.returncode == 0, (
        f"OpenSees returned {proc.returncode}\n"
        f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
    )
    # This build routes Tcl ``puts`` to stderr; read both so the probe
    # keeps working either way.
    return f"{proc.stdout}\n{proc.stderr}"


def _resolved_ndf(stdout: str) -> dict[int, int]:
    """Parse the ``NDF <tag> <llength [nodeDisp tag]>`` probe lines."""
    return {
        int(tag): int(n)
        for tag, n in re.findall(r"^NDF (\d+) (\d+)\s*$", stdout, re.M)
    }


def test_mixed_ndf_2d_deck_lands_the_declared_per_node_ndf(
    tmp_path: Path,
) -> None:
    deck = tmp_path / "model.tcl"
    _mixed_ndf_2d_ops(_mixed_ndf_2d_fem()).tcl(str(deck), run=False)

    with deck.open("a", encoding="utf-8") as f:
        for tag in (*TRUSS_NODES, *BEAM_NODES):
            f.write(f'puts "NDF {tag} [llength [nodeDisp {tag}]]"\n')

    ndf = _resolved_ndf(_run_tcl(deck, tmp_path))
    assert ndf == {1: 2, 2: 2, 3: 3, 4: 3}, (
        "a mixed-ndf 2-D deck must give each node the ndf it declared; "
        f"got {ndf}. Truss nodes at 3 mean the -ndf 2 token was "
        "swallowed (see module docstring)."
    )


def test_the_deck_writes_ndm_coordinates_per_node_line(
    tmp_path: Path,
) -> None:
    """The mechanism behind the gate above, read off the deck text.

    Deliberately NOT the ndf proof — the ``-ndf`` token can be present
    and inert, which is the whole defect.  This pins the coordinate
    count, which is what makes the token bind.
    """
    deck = tmp_path / "model.tcl"
    _mixed_ndf_2d_ops(_mixed_ndf_2d_fem()).tcl(str(deck), run=False)
    node_lines = [
        ln for ln in deck.read_text(encoding="utf-8").splitlines()
        if ln.startswith("node ")
    ]
    assert node_lines == [
        "node 1 0.0 0.0 -ndf 2",
        "node 2 1.0 0.0 -ndf 2",
        "node 3 0.0 1.0",
        "node 4 0.0 2.0",
    ], node_lines


def test_the_swallow_is_real(tmp_path: Path) -> None:
    """Falsifier for the gate: on this OpenSees build the historical
    three-coordinate line really does lose its ``-ndf``.

    Without this row the main assertion would keep passing on a build
    where the parser had been fixed upstream, and would stop testing
    apeGmsh at all.
    """
    deck = tmp_path / "probe.tcl"
    deck.write_text(
        "model BasicBuilder -ndm 2 -ndf 3\n"
        "node 1 0.0 0.0 0.0 -ndf 2\n"   # the form apeGmsh used to emit
        "node 2 1.0 0.0 -ndf 2\n"       # the ndm-coordinate form
        'puts "NDF 1 [llength [nodeDisp 1]]"\n'
        'puts "NDF 2 [llength [nodeDisp 2]]"\n',
        encoding="utf-8",
    )
    ndf = _resolved_ndf(_run_tcl(deck, tmp_path))
    assert ndf == {1: 3, 2: 2}, (
        "expected the trailing 0.0 to swallow '-ndf 2' (node 1 -> 3) and "
        f"the 2-coordinate form to honour it (node 2 -> 2); got {ndf}"
    )
