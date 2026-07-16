"""ADR 0077 Tier 1 — ``apeSees.modal_deck`` distributed-FEAST emit.

Deck-text tests only (the live distributed run needs the fork classic-Tcl
``-feast`` parity build; ADR 0077 unlock 2b). Pin the emitted deck:
the partitioned preamble (``numberer ParallelPlain`` / ``system Mumps``),
the SINGLE captured FEAST solve (``set _lam [eigen -feast ... -rci ...]``,
INV-5 — never a second ``[eigen ...]``), the rank-0 eigenvalue write-out,
no ``modalProperties`` (MPI-blind, INV-2), and the fail-loud guards.
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

# Silence the ADR 0027 partitioned auto-emit warnings (numberer / system /
# MP) — contracted elsewhere; here they would only mask the deck assertions.
_FILTERS = (
    "ignore:len.fem.partitions.:UserWarning",
    "ignore:MP constraints are present:UserWarning",
)
pytestmark = [pytest.mark.filterwarnings(f) for f in _FILTERS]


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
    ops.modal_deck(str(deck), band=(0.0, 5.0), **kw)
    return deck.read_text()


def test_modal_deck_emits_captured_feast_and_writeout(tmp_path) -> None:
    text = _emit(make_two_column_frame_partitioned(), tmp_path, certify=True)

    # Single captured distributed FEAST solve (INV-5: exactly one `eigen`).
    assert "set _lam [eigen -feast 0.0 5.0 -rci -certify]" in text
    assert len(re.findall(r"\beigen -feast\b", text)) == 1

    # Rank-0 eigenvalue write-out.
    assert "if {[getPID] == 0} {" in text
    assert "open eigenvalues.out w" in text
    assert "puts $_fp $_lam" in text

    # Forced preamble comes from the partitioned emit (INV-4); system Mumps
    # is load-bearing.
    assert "system Mumps" in text
    assert "numberer ParallelPlain" in text

    # modalProperties is MPI-blind — never emitted in the parallel deck (INV-2).
    assert "modalProperties" not in text


def test_modal_deck_without_certify_omits_flag(tmp_path) -> None:
    text = _emit(make_two_column_frame_partitioned(), tmp_path)
    assert "set _lam [eigen -feast 0.0 5.0 -rci]" in text
    assert "-certify" not in text


def test_modal_deck_requires_partitioned_model(tmp_path) -> None:
    with pytest.raises(ValueError, match="partitioned model"):
        _emit(make_two_column_frame(), tmp_path)


def test_modal_deck_pymp_target_not_implemented(tmp_path) -> None:
    with pytest.raises(NotImplementedError, match="pymp"):
        _emit(make_two_column_frame_partitioned(), tmp_path, target="pymp")


def test_modal_deck_rejects_bad_band(tmp_path) -> None:
    ops = apeSees(cast("object", make_two_column_frame_partitioned()))
    ops.model(ndm=3, ndf=6)
    _build_frame(ops)
    with pytest.raises(ValueError, match="band"):
        ops.modal_deck(str(tmp_path / "d.tcl"), band=(5.0, 1.0))


def test_modal_deck_per_rank_writes_fragments(tmp_path) -> None:
    text = _emit(
        make_two_column_frame_partitioned(), tmp_path, per_rank=True,
    )
    # Driver still carries the global modal solve; per-rank topology is
    # sourced from fragments.
    assert "set _lam [eigen -feast 0.0 5.0 -rci]" in text
    assert (tmp_path / "ranks").is_dir()
    assert "source" in text
