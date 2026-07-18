"""ADR 0077 Tier 1 — ``apeSees.modal_deck`` distributed-FEAST emit.

Deck-text tests pinning the REPLICATED deck shape established by the P2
live run (2026-07-16, fork classic-Tcl ``-feast`` build, ``mpiexec -n 2``):
the fork's L3 FEAST requires every rank to assemble the FULL model (a
partitioned ``if {[getPID]==K}`` deck fails ``FeastEigenSOE::setSize —
vertex not in graph``), and the deck's ``system`` line plays no part in
the FEAST solve (distribution lives inside the RCI kernel's dmumps).

Pins: flat emit (no partition blocks, even for a partition-authored fem),
the ``getPID`` shim, the deterministic ``Transformation``/``RCM``/
``UmfPack`` preamble, the SINGLE captured FEAST solve (INV-5 — never a
second ``[eigen ...]``), the rank-0 eigenvalue write-out, no
``modalProperties`` (MPI-blind, INV-2), and the fail-loud guards.
"""
from __future__ import annotations

import re
from typing import cast

import pytest

from apeGmsh.opensees import apeSees
from tests.opensees.fixtures.fem_stub import (
    FEMStub,
    make_two_column_frame,
    make_two_column_frame_partitioned,
)


def _build_frame(ops: apeSees) -> None:
    transf = ops.geomTransf.Linear(vecxz=(1.0, 0.0, 0.0))
    ops.element.elasticBeamColumn(
        pg="Cols", transf=transf,
        A=0.01, E=200e9, Iz=1e-4, Iy=1e-4, G=80e9, J=1e-4,
    )
    ops.fix(pg="Base", dofs=(1, 1, 1, 1, 1, 1))
    ops.mass(pg="Top", values=(100.0, 100.0, 1e-6, 1e-6, 1e-6, 1e-6))


def _emit(fem: FEMStub, tmp_path, **kw) -> str:
    ops = apeSees(cast("object", fem))
    ops.model(ndm=3, ndf=6)
    _build_frame(ops)
    deck = tmp_path / "modal.tcl"
    ops.modal_deck(str(deck), band=(0.0, 200.0), **kw)
    return deck.read_text()


def test_modal_deck_emits_captured_feast_and_writeout(tmp_path) -> None:
    text = _emit(make_two_column_frame(), tmp_path, certify=True)

    # Single captured distributed FEAST solve (INV-5: exactly one `eigen`).
    assert "set _lam [eigen -feast 0.0 200.0 -rci -certify]" in text
    assert len(re.findall(r"\beigen -feast\b", text)) == 1

    # getPID shim + rank-0 eigenvalue write-out.
    assert 'if {[info commands getPID] == ""}' in text
    assert "if {[getPID] == 0} {" in text
    assert "open eigenvalues.out w" in text
    assert "puts $_fp $_lam" in text

    # Deterministic eigen preamble; the system line is NOT in the FEAST
    # solve path, so serial UmfPack is correct even distributed (P2).
    assert "constraints Transformation" in text
    assert "numberer RCM" in text
    assert "system UmfPack" in text

    # modalProperties is MPI-blind — never emitted (INV-2).
    assert "modalProperties" not in text


def test_modal_deck_partitioned_fem_emits_flat_replicated(tmp_path) -> None:
    """A partition-authored fem emits FLAT — no partition blocks; every
    rank builds the full model (L3 FEAST requirement, P2 live finding)."""
    text = _emit(make_two_column_frame_partitioned(), tmp_path)

    # Scan the model-build region only — the P3 harvest legitimately
    # opens a multi-line rank-0 block AFTER the eigen preamble.
    model_region = text[: text.index("constraints Transformation")]
    assert not re.search(r"if \{\[getPID\] == \d+\} \{\n", model_region), (
        "partition blocks must not appear in a replicated modal deck"
    )
    # All four nodes present unguarded (both partitions' topology).
    for tag in (1, 2, 3, 4):
        assert re.search(rf"^node {tag} ", text, re.M), f"node {tag} missing"
    assert "set _lam [eigen -feast 0.0 200.0 -rci]" in text


def test_modal_deck_emits_rank0_mode_shape_harvest(tmp_path) -> None:
    """P3: rank-0-guarded mode-shape harvest AFTER the captured solve —
    sidecar write, one dynamically-created eigenvector recorder per FOUND
    mode (``llength $_lam`` — the band count is dynamic; recording an
    unfound mode corrupts the row cursor), a single ``record`` trigger
    (recorders never fire on their own — no analyze step in this deck),
    and ``remove recorders`` to close the files."""
    text = _emit(make_two_column_frame(), tmp_path)

    # Pinned column order: sorted mesh node tags, uniform envelope-ndf
    # dof list (missing DOFs pad 0.0 — cursor-safe in NodeRecorder).
    assert "set _shape_nodes {1 2 3 4}" in text
    assert 'puts $_fp {{"nodes": [1,2,3,4], "ndf": 6, "ndm": 3}}' in text

    # Dynamic per-found-mode recorder creation + trigger + close.
    assert "for {set _k 1} {$_k <= [llength $_lam]} {incr _k} {" in text
    assert (
        "eval recorder Node -file mode_shape_${_k}.out "
        '-node $_shape_nodes -dof 1 2 3 4 5 6 [list "eigen $_k"]'
    ) in text
    record_line = re.search(r"^\s*record$", text, re.M)
    assert record_line is not None, "bare `record` trigger missing"
    assert "remove recorders" in text

    # Order: solve -> recorders -> record (eigenvectors are read at
    # record time; the found count only exists after the solve).
    i_solve = text.index("set _lam [eigen -feast")
    i_recorder = text.index("eval recorder Node")
    assert i_solve < i_recorder < record_line.start()

    # The harvest block is rank-0-guarded (a second guard beyond the
    # eigenvalue write-out's).
    assert text.count("if {[getPID] == 0}") >= 2


def test_modal_deck_without_certify_omits_flag(tmp_path) -> None:
    text = _emit(make_two_column_frame(), tmp_path)
    assert "set _lam [eigen -feast 0.0 200.0 -rci]" in text
    assert "-certify" not in text


def test_modal_deck_pymp_target_not_implemented(tmp_path) -> None:
    with pytest.raises(NotImplementedError, match="pymp"):
        _emit(make_two_column_frame(), tmp_path, target="pymp")


def test_modal_deck_rejects_bad_band(tmp_path) -> None:
    ops = apeSees(cast("object", make_two_column_frame()))
    ops.model(ndm=3, ndf=6)
    _build_frame(ops)
    with pytest.raises(ValueError, match="band"):
        ops.modal_deck(str(tmp_path / "d.tcl"), band=(5.0, 1.0))
