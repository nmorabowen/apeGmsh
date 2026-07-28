"""ADR 0077 Tier 1B — ``apeSees.modal_deck(solver="arpack")`` emit.

Deck-text tests for the **partitioned** ARPACK backend fork PR #668
(``5a522b03b``) unlocked. This deck inverts the FEAST one
(``test_modal_deck_parallel_feast.py``) on the two facts that matter,
and every assertion below exists because getting one of them wrong is
silent:

* **partitioned, not flat** (INV-9) — the FEAST deck's
  ``supports_partitions = False`` seam must NOT be reused here.
* **``system Mumps`` is load-bearing** (INV-8) — the opposite of the
  FEAST deck, where INV-4 records that the system line plays no part in
  the solve. The fork wires the ``ArpackSOE`` collectives only for
  ``MumpsParallelSOE``; anything else and they stay dormant with no
  error.
* **per-rank shape harvest** (INV-9 harvest half) — a rank-0 recorder
  would capture rank 0's slice only and return a partial field, silently.
* **additive nodal mass on ONE owning rank** (INV-12) — a boundary node
  massed on two ranks is summed twice by the ``M*v`` merge; with
  ``shift = 0`` K stays right while M goes wrong, so the run yields a
  plausible spectrum biased low rather than an error.

Every one of these is mutation-tested in
``test_modal_deck_parallel_arpack_mutations.py`` — the discipline #668
itself needed, where the original smoke *passed* a gate-deleted binary.
"""
from __future__ import annotations

import re
import warnings
from typing import cast

import pytest

from apeGmsh.opensees import apeSees
from tests.opensees.fixtures.fem_stub import (
    FEMStub,
    make_axial_chain_partitioned,
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
    kw.setdefault("solver", "arpack")
    kw.setdefault("num_modes", 4)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ops.modal_deck(str(deck), **kw)
    return deck.read_text()


def _emit_chain(tmp_path, *, n_mass: int = 8, n_parts: int = 2, **kw) -> str:
    """The shared-boundary fixture — ranks that actually touch."""
    ops = apeSees(cast("object", make_axial_chain_partitioned(n_mass, n_parts)))
    ops.model(ndm=2, ndf=2)
    mat = ops.uniaxialMaterial.ElasticMaterial(E=100.0)
    ops.element.Truss(pg="Chain", A=1.0, material=mat)
    ops.fix(pg="Base", dofs=(1, 1))
    ops.fix(pg="Masses", dofs=(0, 1))
    ops.mass(pg="Masses", values=(1.0, 0.0))
    deck = tmp_path / "chain.tcl"
    kw.setdefault("num_modes", 4)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ops.modal_deck(str(deck), solver="arpack", **kw)
    return deck.read_text()


# -- the deck shape ---------------------------------------------------------


def test_arpack_deck_is_partitioned_not_flat(tmp_path) -> None:
    """INV-9. The whole point of this backend is that each rank holds
    only its slice — so the model MUST come out in ``getPID`` blocks.
    (Contrast the FEAST deck, which is deliberately flat: its L3 kernel
    needs the full model on every rank.)"""
    text = _emit(make_two_column_frame_partitioned(), tmp_path)

    assert "if {[getPID] == 0} {\n" in text
    assert "if {[getPID] == 1} {\n" in text
    # Rank-owned topology, not the full model on every rank.
    model_region = text[: text.index("constraints Transformation")]
    rank0 = model_region[
        model_region.index("if {[getPID] == 0} {"):
        model_region.index("if {[getPID] == 1} {")
    ]
    assert re.search(r"^\s+node 1 ", rank0, re.M)
    assert re.search(r"^\s+node 2 ", rank0, re.M)
    assert not re.search(r"^\s+node 3 ", rank0, re.M), (
        "rank 0's block must not carry rank 1's nodes — that is the "
        "replicated (FEAST) deck, not this one"
    )


def test_arpack_deck_preamble_is_forced_and_mumps_backed(tmp_path) -> None:
    """INV-8 / INV-10. ``Mumps`` is what engages the #668 wiring; the
    runtime-conditional form (ADR 0027 INV-5) keeps the same deck
    runnable single-process as its own serial oracle."""
    text = _emit(make_two_column_frame_partitioned(), tmp_path)

    assert "constraints Transformation" in text
    assert "if {[catch {numberer ParallelPlain} _err]} { numberer RCM }" in text
    assert "if {[catch {system Mumps} _err]} { system UmfPack }" in text
    # Exactly one of each — the modal deck opts out of the ADR 0027
    # auto-emit rather than stack a second identical pair above its own.
    assert len(re.findall(r"\bnumberer ParallelPlain\b", text)) == 1
    assert len(re.findall(r"\bsystem Mumps\b", text)) == 1

    # ...and the preamble precedes the solve.
    assert text.index("system Mumps") < text.index("set _lam [eigen")


def test_arpack_deck_captures_solve_exactly_once(tmp_path) -> None:
    """INV-11. A second ``[eigen ...]`` in the rank-0 write-out is a
    redundant distributed solve AND a rank-0-only collective ⇒ deadlock."""
    text = _emit(make_two_column_frame_partitioned(), tmp_path)

    assert "set _lam [eigen 4]" in text
    assert len(re.findall(r"\[eigen\b", text)) == 1
    assert "if {[getPID] == 0} { set _fp [open eigenvalues.out w]; " in text
    assert "puts $_fp $_lam" in text
    assert "-feast" not in text


def test_arpack_deck_omits_modal_properties(tmp_path) -> None:
    """INV-2 / INV-A5 — MPI-blind upstream; #668 did not change it."""
    text = _emit(make_two_column_frame_partitioned(), tmp_path)
    assert "modalProperties" not in text


def test_arpack_deck_emits_per_rank_shape_harvest(tmp_path) -> None:
    """The FEAST rank-0 harvest cannot be copied: on a partitioned deck
    it captures rank 0's slice and returns a partial field with NO error.
    Each rank must write its own sidecar + its own per-mode recorders."""
    text = _emit_chain(tmp_path)

    for rank, tags, json_tags in (
        (0, "{1 2 3 4 5}", "[1,2,3,4,5]"),
        (1, "{5 6 7 8 9}", "[5,6,7,8,9]"),
    ):
        assert f"set _shape_nodes {tags}" in text
        assert f"open mode_shapes_rank{rank}.json w" in text
        assert (
            f'puts $_fp {{{{"nodes": {json_tags}, "ndf": 2, "ndm": 2}}}}'
        ) in text
        assert (
            f"eval recorder Node -file mode_shape_${{_k}}_rank{rank}.out "
            "-node $_shape_nodes -dof 1 2 " '[list "eigen $_k"]'
        ) in text

    # No replicated-deck artifacts leaked in.
    assert "open mode_shapes.json w" not in text
    assert "mode_shape_${_k}.out" not in text

    # Dynamic per-found-mode creation + trigger + close, per rank.
    assert text.count(
        "for {set _k 1} {$_k <= [llength $_lam]} {incr _k} {"
    ) == 2
    assert len(re.findall(r"^\s*record$", text, re.M)) == 2
    assert text.count("remove recorders") == 2

    # Order: solve -> recorders -> record (eigenvectors are read at
    # record time, and nothing else fires a recorder in this deck).
    i_solve = text.index("set _lam [eigen")
    i_recorder = text.index("eval recorder Node")
    i_record = re.search(r"^\s*record$", text, re.M)
    assert i_record is not None
    assert i_solve < i_recorder < i_record.start()


def test_arpack_deck_records_shared_boundary_node_on_both_ranks(
    tmp_path,
) -> None:
    """INV-12, harvest half. Node 5 is on the partition boundary and is
    recorded by BOTH ranks *on purpose* — comparing the two copies at
    harvest is the cheapest proof the run was really distributed."""
    text = _emit_chain(tmp_path)

    assert "set _shape_nodes {1 2 3 4 5}" in text   # rank 0 ends at 5
    assert "set _shape_nodes {5 6 7 8 9}" in text   # rank 1 starts at 5


def test_arpack_deck_masses_shared_node_on_one_rank_only(tmp_path) -> None:
    """INV-12, emit half — the silent one. The ``M*v`` merge SUMS per-rank
    contributions, so a boundary node massed on both ranks counts twice.
    Because the Tcl ``eigen`` path always has ``shift = 0``, K stays
    exactly right while M goes wrong: no error, just a spectrum biased
    low. ``_emit_partitioned`` routes mass through ``primary_owner_map``;
    this pins that the modal deck inherits it."""
    text = _emit_chain(tmp_path)

    mass_lines = re.findall(r"^\s*mass (\d+) ", text, re.M)
    assert sorted(mass_lines) == sorted(set(mass_lines)), (
        f"a node is massed on more than one rank: {mass_lines}"
    )
    # Node 5 (the shared one) is massed exactly once...
    assert mass_lines.count("5") == 1
    # ...while both ranks still DECLARE it (idempotent lines replicate).
    assert len(re.findall(r"^\s*node 5 ", text, re.M)) == 2


def test_arpack_deck_carries_exactly_one_getpid_shim(tmp_path) -> None:
    """The partitioned deck carries the ``info commands getPID`` shim so
    it PARSES under plain ``OpenSees`` (ADR 0027 INV-5) — emitted once,
    by ``partition_open``, not a second time by ``eigen_parallel``.

    Parsing is all it buys: single-process, the shim returns 0 so only
    rank 0's block runs and OpenSees solves rank 0's *submodel*. This
    deck is NOT its own serial oracle (that is the replicated FEAST
    deck); a live 8-mass/2-rank run at np=1 returns an EMPTY spectrum
    with no error at all."""
    text = _emit(make_two_column_frame_partitioned(), tmp_path)
    assert 'if {[info commands getPID] == ""} ' in text
    assert text.count('if {[info commands getPID] == ""}') == 1


# -- guardrails -------------------------------------------------------------


def test_arpack_rejects_unpartitioned_model(tmp_path) -> None:
    """Silently emitting a one-rank deck would hide the mistake; point
    the user at Tier 0 instead."""
    with pytest.raises(ValueError, match="PARTITIONED"):
        _emit(make_two_column_frame(), tmp_path)


def test_arpack_rejects_band_and_certify(tmp_path) -> None:
    with pytest.raises(ValueError, match="FEAST-only"):
        _emit(make_two_column_frame_partitioned(), tmp_path, band=(0.0, 200.0))
    with pytest.raises(ValueError, match="FEAST-only"):
        _emit(make_two_column_frame_partitioned(), tmp_path, certify=True)


def test_arpack_requires_num_modes(tmp_path) -> None:
    with pytest.raises(ValueError, match="num_modes"):
        _emit(make_two_column_frame_partitioned(), tmp_path, num_modes=None)
    with pytest.raises(ValueError, match="num_modes"):
        _emit(make_two_column_frame_partitioned(), tmp_path, num_modes=0)


def test_arpack_rejects_pymp_target(tmp_path) -> None:
    """#668 is classic-Tcl only — the modern interpreter still builds its
    ArpackSOE bare (same latent F1 defect)."""
    with pytest.raises(NotImplementedError, match="ArpackSOE bare"):
        _emit(make_two_column_frame_partitioned(), tmp_path, target="pymp")


def test_arpack_rejects_non_mumps_system(tmp_path) -> None:
    """INV-8, fail-loud half. ``ParallelProfileSPD`` is the nasty case:
    genuinely distributed, own collectives live, so the deck RUNS — while
    the ArpackSOE stays at processID -1."""
    for system_name in ("UmfPack", "Pardiso", "ParallelProfileSPD"):
        ops = apeSees(cast("object", make_two_column_frame_partitioned()))
        ops.model(ndm=3, ndf=6)
        _build_frame(ops)
        getattr(ops.system, system_name)()
        with pytest.raises(ValueError, match="requires 'system Mumps'"):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                ops.modal_deck(
                    str(tmp_path / "d.tcl"), solver="arpack", num_modes=4,
                )


def test_arpack_accepts_explicit_mumps_system(tmp_path) -> None:
    ops = apeSees(cast("object", make_two_column_frame_partitioned()))
    ops.model(ndm=3, ndf=6)
    _build_frame(ops)
    ops.system.Mumps()
    deck = tmp_path / "d.tcl"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ops.modal_deck(str(deck), solver="arpack", num_modes=4)
    assert "system Mumps" in deck.read_text()


def test_unknown_solver_raises(tmp_path) -> None:
    with pytest.raises(ValueError, match="solver must be"):
        _emit(make_two_column_frame_partitioned(), tmp_path, solver="lanczos")


def test_feast_rejects_num_modes(tmp_path) -> None:
    """The mirror guard — the contour IS the band for FEAST."""
    with pytest.raises(ValueError, match="ARPACK-only"):
        _emit(
            make_two_column_frame(), tmp_path,
            solver="feast", band=(0.0, 200.0), num_modes=4,
        )


def test_feast_requires_band(tmp_path) -> None:
    with pytest.raises(ValueError, match="needs band="):
        _emit(make_two_column_frame(), tmp_path, solver="feast", num_modes=None)


def test_modal_deck_rejects_equation_ties_on_both_solvers(tmp_path) -> None:
    """ADR 0077 P6 adversarial finding A1.

    An ``enforce="equation"`` tie emits ``equationConstraint`` rows that
    only ``Lagrange`` / ``LadrunoProjection`` can enforce — while a modal
    deck FORCES ``constraints Transformation`` (INV-4 / INV-10, because
    Lagrange injects zero-mass DOFs that fabricate spurious modes). Only
    one handler can be active, so the two are mutually exclusive.

    Before the guard, the forced line simply won — it is emitted last —
    and BOTH backends emitted a deck carrying six ``equationConstraint``
    rows under a bare ``constraints Transformation``: a complete run
    producing the spectrum of a *different structure*, silently. The
    ordinary ``ops.tcl`` path is unaffected and still auto-upgrades to
    ``Lagrange``; only the modal decks refuse.
    """
    import numpy as np

    from apeGmsh._kernel.records._constraints import (
        ConstraintKind,
        InterpolationRecord,
    )

    def _fem_with_eq_tie():
        fem = make_two_column_frame_partitioned()
        fem.add_surface_constraints([
            InterpolationRecord(
                kind=ConstraintKind.TIE, slave_node=4, master_nodes=[1, 2],
                weights=np.array([0.5, 0.5]), dofs=[1, 2, 3],
                enforce="equation", name="cross_rank_eq_tie",
            ),
        ])
        return fem

    with pytest.raises(Exception, match="enforce='equation'"):
        _emit(_fem_with_eq_tie(), tmp_path)

    with pytest.raises(Exception, match="enforce='equation'"):
        _emit(
            _fem_with_eq_tie(), tmp_path,
            solver="feast", band=(0.0, 200.0), num_modes=None,
        )

    # Control: the ordinary deck path still emits it correctly.
    ops = apeSees(cast("object", _fem_with_eq_tie()))
    ops.model(ndm=3, ndf=6)
    _build_frame(ops)
    plain = tmp_path / "plain.tcl"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ops.tcl(str(plain))
    text = plain.read_text()
    assert text.count("equationConstraint ") == 6
    assert "constraints Lagrange" in text


def test_arpack_rejects_staged_model(tmp_path) -> None:
    ops = apeSees(cast("object", make_two_column_frame_partitioned()))
    ops.model(ndm=3, ndf=6)
    _build_frame(ops)
    with ops.stage(name="s1") as s:
        s.analysis(
            test=ops.test.NormDispIncr(tol=1e-4, max_iter=50),
            algorithm=ops.algorithm.Newton(),
            integrator=ops.integrator.LoadControl(dlam=0.1),
            constraints=ops.constraints.Transformation(),
            numberer=ops.numberer.ParallelPlain(),
            system=ops.system.Mumps(),
            analysis=ops.analysis.Static(),
        )
        s.run(n_increments=1)
    with pytest.raises(NotImplementedError, match="staged"):
        ops.modal_deck(str(tmp_path / "d.tcl"), solver="arpack", num_modes=4)
