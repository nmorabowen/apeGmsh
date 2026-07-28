"""ADR 0077 Tier 1B — mutation tests for the ARPACK modal-deck suite.

**Why this file exists.** In fork PR #668 the ARPACK-MP wiring shipped
with a smoke test that *passed* against a deliberately gate-deleted
binary, at 1.3e-15. The test was green, the build was broken, and
nothing noticed. A green test that cannot distinguish is not a test.

So every load-bearing assertion in
``test_modal_deck_parallel_arpack.py`` is paired here with the defect it
claims to catch: each case applies a real mutation to the **emitter**
(not to the deck text — mutating the text would only prove the regex is
sensitive, not that the test catches a code defect) and asserts the
corresponding test now FAILS. If a mutation stops failing, either the
emitter grew a second path to correctness or the test went blind — both
worth knowing.

The mutations mirror the guide's acceptance criteria §6.4 plus the two
traps that are silent in production:

* ``system UmfPack`` instead of ``Mumps``  → wiring never engages (INV-8)
* flat emit instead of partitioned         → wrong deck entirely (INV-9)
* rank-0-only shape harvest (the FEAST copy-paste) → partial field, no
  error (INV-9 harvest half)
* mass on every owner instead of the primary one → M double-counted on
  shared nodes; with ``shift = 0`` K stays right, so the spectrum is
  merely biased low (INV-12)
* boundary-agreement check removed         → the one assertion that can
  detect a private per-rank Lanczos (INV-12)
"""
from __future__ import annotations

import numpy as np
import pytest

from apeGmsh.opensees.analysis.modal import ParallelModalResult
from apeGmsh.opensees.apesees import BuiltModel
from apeGmsh.opensees.emitter.tcl import TclEmitter

from . import test_modal_deck_parallel_arpack as suite


#: Every way a pinned test can report "this deck is wrong" — a bare
#: ``assert``, a guard raising, or an inner ``pytest.raises`` finding
#: nothing to catch (``pytest.fail.Exception`` derives from
#: ``BaseException``, so it does NOT come along with ``Exception``).
_FAILURE = (
    AssertionError, ValueError, KeyError, IndexError, pytest.fail.Exception,
)


def _assert_fails(test_fn, tmp_path) -> None:
    """The mutated emitter must make ``test_fn`` fail."""
    with pytest.raises(_FAILURE):
        test_fn(tmp_path)


# -- INV-8: the load-bearing `system Mumps` --------------------------------


def test_mutation_umfpack_instead_of_mumps_is_caught(
    tmp_path, monkeypatch,
) -> None:
    """The single most natural regression on this deck — "the FEAST deck
    uses UmfPack, so this one can too". It cannot: the fork wires the
    ArpackSOE collectives ONLY for MumpsParallelSOE, and with any other
    system they stay dormant with no error at all."""
    monkeypatch.setattr(
        TclEmitter, "parallel_runtime_fallback_system",
        lambda self, primary, fallback: (
            TclEmitter.system(self, "UmfPack")
        ),
    )
    _assert_fails(suite.test_arpack_deck_preamble_is_forced_and_mumps_backed,
                  tmp_path)


# -- INV-9: partitioned, not flat ------------------------------------------


def test_mutation_flat_emit_instead_of_partitioned_is_caught(
    tmp_path, monkeypatch,
) -> None:
    """Reusing the FEAST path's ``supports_partitions = False`` seam
    silently turns this into a replicated deck — every rank builds the
    whole model, so nothing is distributed except the linear solve, and
    the memory case that justifies the backend evaporates."""
    # raising=False: the attribute is an instance-level seam, set only by
    # the FEAST path — the class does not carry it.
    monkeypatch.setattr(
        TclEmitter, "supports_partitions", False, raising=False,
    )
    _assert_fails(suite.test_arpack_deck_is_partitioned_not_flat, tmp_path)


# -- INV-9 harvest half: per-rank sidecars, not a rank-0 recorder ----------


def test_mutation_rank0_only_harvest_is_caught(
    tmp_path, monkeypatch,
) -> None:
    """The copy-paste-from-FEAST failure. A rank-0 recorder on a
    partitioned deck captures rank 0's slice and returns a partial mode
    field — with no error anywhere, which is exactly why the deck-text
    test has to be the thing that catches it."""
    original = TclEmitter.eigen_parallel

    def rank0_only(self, num_modes, *, shape_nodes_by_rank=None, **kw):
        if shape_nodes_by_rank:
            shape_nodes_by_rank = {0: shape_nodes_by_rank[0]}
        return original(
            self, num_modes, shape_nodes_by_rank=shape_nodes_by_rank, **kw
        )

    monkeypatch.setattr(TclEmitter, "eigen_parallel", rank0_only)
    _assert_fails(suite.test_arpack_deck_emits_per_rank_shape_harvest,
                  tmp_path)
    _assert_fails(
        suite.test_arpack_deck_records_shared_boundary_node_on_both_ranks,
        tmp_path,
    )


# -- INV-12 emit half: additive mass on ONE rank ---------------------------


def test_mutation_mass_on_every_owner_is_caught(
    tmp_path, monkeypatch,
) -> None:
    """Reintroduces the pre-``primary_owner_map`` defect: mass emitted on
    every owning rank. The ``M*v`` merge sums per-rank contributions, so
    a shared node's mass counts twice — and because the Tcl ``eigen``
    path always has ``shift = 0``, K stays exactly right while M goes
    wrong. Nothing errors; the spectrum just comes back biased low.
    There is no guard for this anywhere in the C++ stack, which is why
    apeGmsh has to own it."""
    from apeGmsh.opensees._internal.build import build_node_partition_owners

    def every_owner(self, primary_owner):
        owners = build_node_partition_owners(self.fem)
        out: dict = {}
        for rec in self.mass_records:
            per_rank: dict = {}
            for node_tag in self._resolve_node_target(rec.pg, rec.nodes):
                nid = int(node_tag)
                for rank in owners.get(nid) or ():
                    per_rank.setdefault(int(rank), []).append(nid)
            for rank, nodes_list in per_rank.items():
                out.setdefault(rank, []).append((rec, nodes_list))
        return out

    monkeypatch.setattr(
        BuiltModel, "_bucket_mass_targets_by_rank", every_owner,
    )
    _assert_fails(suite.test_arpack_deck_masses_shared_node_on_one_rank_only,
                  tmp_path)


# -- INV-12 harvest half: the boundary-agreement assertion ------------------


def test_mutation_boundary_check_removed_is_caught(
    tmp_path, monkeypatch,
) -> None:
    """Take-the-first-copy instead of comparing. This is the mutation
    that matters most, because the check is the ONLY thing in the whole
    stack able to detect that each rank ran a private Lanczos — the
    failure a pre-#668 binary produces, which otherwise yields a
    plausible field and no error."""
    import json
    import re as _re

    def no_check(base, rank_sidecars, *, n_modes):
        """A merge that just takes whatever it reads last — the shape the
        guide warns against ("do not silently take the first")."""
        rows_by_tag: dict = {}
        ndf = ndm = None
        for meta_path in rank_sidecars:
            rank = int(
                _re.search(r"rank(\d+)\.json$", meta_path.name).group(1)
            )
            meta = json.loads(meta_path.read_text())
            nodes = [int(t) for t in meta["nodes"]]
            ndf = int(meta["ndf"])
            ndm = int(meta.get("ndm", 3))
            for k in range(1, n_modes + 1):
                f = meta_path.parent / f"mode_shape_{k}_rank{rank}.out"
                row = np.asarray(
                    [float(t) for t in f.read_text().split()]
                ).reshape(len(nodes), ndf)
                for i, tag in enumerate(nodes):
                    rows_by_tag.setdefault(tag, {})[k] = row[i]
        tags = np.asarray(sorted(rows_by_tag), dtype=np.int64)
        shapes = np.zeros((n_modes, tags.shape[0], ndf or 0))
        for j, tag in enumerate(tags):
            for k in range(1, n_modes + 1):
                shapes[k - 1, j] = rows_by_tag[int(tag)][k]
        return tags, shapes, int(ndm or 3)

    monkeypatch.setattr(
        ParallelModalResult, "_merge_partitioned_shapes",
        staticmethod(no_check),
    )

    from tests.opensees.unit import test_parallel_modal_result as reader_suite

    _assert_fails(
        reader_suite.test_from_job_partitioned_boundary_disagreement_raises,
        tmp_path,
    )


def test_mutation_boundary_tolerance_widened_is_caught(
    tmp_path, monkeypatch,
) -> None:
    """A check with a tolerance loose enough to accept anything is the
    same as no check — and reads greener. Widening it to 10% must make
    the discrimination test fail."""
    import apeGmsh.opensees.analysis.modal as modal_mod

    monkeypatch.setattr(modal_mod, "_SHARED_NODE_RTOL", 1.0e-1)

    from tests.opensees.unit import test_parallel_modal_result as reader_suite

    _assert_fails(
        reader_suite.test_from_job_partitioned_boundary_tolerance_is_not_wide,
        tmp_path,
    )


# -- control: the suite passes UNMUTATED -----------------------------------


def test_control_unmutated_suite_passes(tmp_path) -> None:
    """The other half of a mutation test: prove the assertions pass when
    nothing is broken, so a failure above is attributable to the
    mutation rather than to a permanently-red test."""
    suite.test_arpack_deck_preamble_is_forced_and_mumps_backed(tmp_path)
    suite.test_arpack_deck_is_partitioned_not_flat(tmp_path)
    suite.test_arpack_deck_emits_per_rank_shape_harvest(tmp_path)
    suite.test_arpack_deck_records_shared_boundary_node_on_both_ranks(tmp_path)
    suite.test_arpack_deck_masses_shared_node_on_one_rank_only(tmp_path)
