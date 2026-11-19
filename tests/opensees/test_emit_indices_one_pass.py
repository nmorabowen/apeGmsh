"""``_emit_indices`` joins rows to emit indices in one pass (#2544).

``emit_index_of`` scans every program run for each row. Called once per
row, that is quadratic when each constraint is its own call, and it made
four ``tests/test_phase_3f_1.py`` cases (50,001 ``equalDOF`` rows through
``Assembly.bridge``) cost 140 s each on a slow CI runner. The batch join
must give exactly what the per-row join gives, and stay linear.

#2566 batches the loops that remained on that path: ``emit_index_map``
behind ``H5Model.commands`` and ``_ReplayOrder.row``, and the compose
merge's per-record ``with_constraint`` / ``with_load`` (the tests at the
end of this file).
"""
from __future__ import annotations

import random
import time

import pytest

from apeGmsh.opensees.emitter.h5_reader import ProgramRun, emit_index_of
from apeGmsh.opensees.opensees_model import _emit_indices


def _run(first, count, method, row, stage=-1):
    return ProgramRun(first=first, count=count, method=method,
                      store=method if row >= 0 else "", row=row,
                      stage=stage, decl=-1)


def _program(rng: random.Random, n_rows: int) -> list[ProgramRun]:
    """Runs of ``equalDOF`` covering ``n_rows`` rows in chunks, mixed with
    other verbs, staged runs, ledger runs and partition replicas."""
    runs: list[ProgramRun] = []
    first = 1
    row = 0
    while row < n_rows:
        count = rng.randint(1, 4)
        runs.append(_run(first, count, "equalDOF", row))
        first += count
        if rng.random() < 0.3:  # a partition replica repeats the rows
            runs.append(_run(first, count, "equalDOF", row))
            first += count
        if rng.random() < 0.3:
            runs.append(_run(first, 2, "rigidLink", row))
            first += 2
        if rng.random() < 0.2:
            runs.append(_run(first, count, "equalDOF", row, stage=1))
            first += count
        if rng.random() < 0.2:
            runs.append(_run(first, 1, "equalDOF", -1))  # a ledger call
            first += 1
        row += count
    rng.shuffle(runs)  # first-run-wins must follow program order, not row order
    return runs


@pytest.mark.parametrize("seed", range(20))
def test_batch_join_equals_the_per_row_join(seed: int) -> None:
    rng = random.Random(seed)
    n = rng.randint(1, 60)
    program = _program(rng, n)
    for method in ("equalDOF", "rigidLink"):
        covered = [i for i in range(n)
                   if any(r.method == method and r.stage == -1
                          and 0 <= r.row <= i < r.row + r.count
                          for r in program)]
        if len(covered) != n:
            with pytest.raises(LookupError):
                _emit_indices(program, method, n)
            continue
        assert _emit_indices(program, method, n) == [
            emit_index_of(program, method, i) for i in range(n)]


def test_a_row_no_run_covers_raises_the_same_lookup_error() -> None:
    program = [_run(1, 2, "equalDOF", 0)]
    with pytest.raises(LookupError, match="row 2 in stage -1"):
        _emit_indices(program, "equalDOF", 3)


def test_one_call_per_row_is_linear() -> None:
    """60,000 single-row runs: the per-row scan needs ~1.8e9 run visits
    (minutes); one pass needs 60,000 (well under a second)."""
    n = 60_000
    program = [_run(i + 1, 1, "equalDOF", i) for i in range(n)]
    t0 = time.perf_counter()
    got = _emit_indices(program, "equalDOF", n)
    elapsed = time.perf_counter() - t0
    assert got == list(range(1, n + 1))
    assert elapsed < 10.0, f"{elapsed:.1f} s: the join is not linear"


# The callers must use the one-pass join, not the per-row scan: with the
# per-row ``emit_index_of`` made to raise, both still answer on a program
# that covers every row (a reverted call site raises here).


def _forbid_per_row(monkeypatch: pytest.MonkeyPatch) -> None:
    from apeGmsh.opensees.emitter import h5_reader

    def _per_row(*_a, **_k):
        raise AssertionError("per-row emit_index_of scan in a caller")

    monkeypatch.setattr(h5_reader, "emit_index_of", _per_row)


def test_mp_rows_in_bridge_order_uses_the_one_pass_join(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from apeGmsh.opensees.opensees_model import _mp_rows_in_bridge_order

    program = [_run(1, 2, "equalDOF", 0), _run(3, 1, "rigidLink", 0)]
    _forbid_per_row(monkeypatch)
    got = _mp_rows_in_bridge_order(
        [["e0", "e1"], ["r0"], [], []], program)
    assert got == ((1, "e0"), (2, "e1"), (3, "r0"))


def test_equation_rows_uses_the_one_pass_join(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from types import SimpleNamespace

    from apeGmsh.opensees.opensees_model import _equation_rows

    model = SimpleNamespace(
        equation_constraints=lambda: ("q0", "q1"),
        opensees_zone_version=lambda: SimpleNamespace(
            major=2, minor=35, patch=0),
    )
    program = [_run(5, 2, "equationConstraint", 0)]
    _forbid_per_row(monkeypatch)
    assert _equation_rows(model, program, None) == ((5, "q0"), (6, "q1"))


# ---------------------------------------------------------------------------
# #2566: emit_index_map behind H5Model.commands and _ReplayOrder.row
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("seed", range(20))
@pytest.mark.parametrize("stage", [-1, 1])
def test_emit_index_map_equals_the_per_row_join(seed: int, stage: int) -> None:
    from apeGmsh.opensees.emitter.h5_reader import emit_index_map

    rng = random.Random(seed)
    n = rng.randint(1, 60)
    program = _program(rng, n)
    for method in ("equalDOF", "rigidLink"):
        got = emit_index_map(program, method, stage=stage)
        for i in range(n + 2):
            try:
                want = emit_index_of(program, method, i, stage=stage)
            except LookupError:
                assert i not in got
            else:
                assert got[i] == want


def _commands_model(program: list[ProgramRun], methods: list[str]):
    """A stand-in for an open archive: ``H5Model.commands`` reads only
    ``self._f["opensees"]["commands"]`` and ``self.program()``. Each row
    is a global ``method`` call with one int arg."""
    from types import SimpleNamespace

    import numpy as np

    n = len(methods)
    group = {
        "method": np.array(methods, dtype=object),
        "token": np.array(["t"] * n, dtype=object),
        "stage": np.full(n, -1, dtype=np.int64),
        "arg_offsets": np.arange(n + 1, dtype=np.int64),
        "args": np.arange(n, dtype=np.float64),
        "args_str": np.array([""] * n, dtype=object),
        "arg_kinds": np.zeros(n, dtype=np.int64),
        "arg_names": np.array([""] * n, dtype=object),
    }
    return SimpleNamespace(
        _f={"opensees": {"commands": group}}, program=lambda: program)


def test_commands_uses_one_map_per_method(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from apeGmsh.opensees.emitter.h5_reader import H5Model

    # region row 0 is covered twice: the first run (emit 1) wins.
    program = [_run(1, 2, "region", 0), _run(3, 1, "parameter", 0),
               _run(4, 1, "region", 2), _run(5, 1, "region", 0)]
    model = _commands_model(
        program, ["region", "parameter", "region", "region"])
    want = [emit_index_of(program, m, r) for m, r in
            (("region", 0), ("parameter", 0), ("region", 1), ("region", 2))]
    _forbid_per_row(monkeypatch)
    got = [c.emit_index for c in H5Model.commands(model)]
    assert got == want == [1, 3, 2, 4]


def test_commands_row_no_run_covers_raises_the_same_lookup_error() -> None:
    from apeGmsh.opensees.emitter.h5_reader import (
        H5Model, ProgramAbsentError,
    )

    model = _commands_model([_run(1, 1, "region", 0)], ["region", "region"])
    with pytest.raises(LookupError,
                       match="'region' call writing row 1 in stage -1"):
        H5Model.commands(model)
    with pytest.raises(ProgramAbsentError):
        H5Model.commands(_commands_model([], ["region"]))


def _replay_order(program: list[ProgramRun]):
    from types import SimpleNamespace

    from apeGmsh.opensees._internal.compose import _ReplayOrder

    return _ReplayOrder(program, SimpleNamespace(
        program_decl="complete", decls=(), tags=()))


def test_replay_order_row_uses_the_one_pass_map(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Rows 0-1 are covered twice (a partition replica): the first wins.
    program = [_run(1, 2, "recorder", 0), _run(3, 2, "recorder", 0),
               _run(5, 1, "recorder", 2)]
    order = _replay_order(program)
    want = [emit_index_of(program, "recorder", i) for i in range(3)]
    _forbid_per_row(monkeypatch)
    assert [order.row("recorder", i) for i in range(3)] == want == [1, 2, 5]


def test_replay_order_row_no_run_covers_raises_the_same_lookup_error() -> None:
    order = _replay_order([_run(1, 1, "recorder", 0)])
    with pytest.raises(LookupError,
                       match="'recorder' call writing row 3 in stage -1"):
        order.row("recorder", 3)


# ---------------------------------------------------------------------------
# #2566: the compose merge routes the bundle's records in one pass
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def host_and_bundle(tmp_path_factory):
    """A host FEMData and the real rewritten bundle of a saved module."""
    from apeGmsh.mesh import _compose
    from tests.test_phase_3f_1 import _make_module_fem, _write_source

    path = _write_source(_make_module_fem(extra_node_constraints=3),
                         tmp_path_factory.mktemp("merge") / "m.h5")
    host = _make_module_fem()
    captured = []
    real = _compose._merge_bundle_into_fem

    def _spy(fem, bundle, **kw):
        captured.append(bundle)
        return real(fem, bundle, **kw)

    _compose._merge_bundle_into_fem = _spy
    try:
        _compose._compose_module(host, path, label="m",
                                 translate=(0.0, 0.0, 0.0), rotate=None,
                                 partition_rank=None)
    finally:
        _compose._merge_bundle_into_fem = real
    (bundle,) = captured
    return host, bundle


def _every_record_kind(bundle) -> dict:
    """Record streams holding every kind ``with_constraint`` /
    ``with_load`` route, with an SP in each stream that accepts one and
    a node-pair in the element stream, on the bundle's own nodes."""
    from apeGmsh._kernel.records._constraints import (
        InterpolationRecord, NodeGroupRecord, NodePairRecord,
        NodeToSurfaceRecord, SurfaceCouplingRecord,
    )
    from apeGmsh._kernel.records._loads import (
        ElementLoadRecord, NodalLoadRecord, SPRecord,
    )

    a, b, c = (int(t) for t in bundle.node_ids[:3])
    (group,) = bundle.element_groups.values()
    e = int(group.ids[0])
    return dict(
        node_constraints=(
            *bundle.node_constraints,
            NodeGroupRecord(kind="rigid_body", master_node=a,
                            slave_nodes=[b, c], dofs=[1, 2, 3]),
            SPRecord(node_id=a, dof=1, name="sp_in_constraints"),
            NodeToSurfaceRecord(kind="node_to_surface", master_node=a,
                                slave_nodes=[b, c]),
            InterpolationRecord(kind="tie", slave_node=a,
                                master_nodes=[b, c], dofs=[1]),
        ),
        elem_constraints=(
            SurfaceCouplingRecord(kind="mortar", master_nodes=[a],
                                  slave_nodes=[b], dofs=[1]),
            NodePairRecord(kind="equal_dof", master_node=b, slave_node=c,
                           dofs=[2], name="pair_in_elem_stream"),
        ),
        nodal_loads=(NodalLoadRecord(node_id=a, force_xyz=(1.0, 0.0, 0.0)),
                     SPRecord(node_id=b, dof=2, name="sp_in_loads")),
        element_loads=(ElementLoadRecord(element_id=e,
                                         load_type="beamUniform"),),
        sp_records=(SPRecord(node_id=c, dof=3, name="sp_in_sp"),),
    )


def _per_record_reference(host, bundle):
    """The merge as it was before #2566: a record-less merge, then one
    ``with_constraint`` / ``with_load`` per record, in stream order."""
    import dataclasses

    from apeGmsh.mesh._compose import _merge_bundle_into_fem

    fem = _merge_bundle_into_fem(host, dataclasses.replace(
        bundle, node_constraints=(), elem_constraints=(), nodal_loads=(),
        element_loads=(), sp_records=()))
    for rec in (*bundle.node_constraints, *bundle.elem_constraints):
        fem = fem.with_constraint(rec)
    for rec in (*bundle.nodal_loads, *bundle.element_loads,
                *bundle.sp_records):
        fem = fem.with_load(rec)
    return fem


def _record_sets(fem) -> tuple:
    return (fem.nodes.constraints, fem.nodes.loads, fem.nodes.sp,
            fem.elements.constraints, fem.elements.loads)


def _assert_same_records(got, want) -> None:
    for g, w in zip(_record_sets(got), _record_sets(want), strict=True):
        assert type(g) is type(w)
        assert len(w) > 0  # every bucket is exercised
        assert [id(r) for r in g] == [id(r) for r in w]


def test_batched_merge_equals_the_per_record_merge(host_and_bundle) -> None:
    import dataclasses

    from apeGmsh.mesh._compose import _merge_bundle_into_fem

    host, bundle = host_and_bundle
    bundle = dataclasses.replace(bundle, **_every_record_kind(bundle))
    got = _merge_bundle_into_fem(host, bundle)
    want = _per_record_reference(host, bundle)
    _assert_same_records(got, want)
    assert got.snapshot_id == want.snapshot_id
    assert got.elements._apegmsh_nodes_ref is got.nodes


@pytest.mark.parametrize("stream, verb", [
    ("node_constraints", "with_constraint"),
    ("elem_constraints", "with_constraint"),
    ("nodal_loads", "with_load"),
    ("element_loads", "with_load"),
    ("sp_records", "with_load"),
])
def test_unknown_record_type_raises_the_per_record_error(
    host_and_bundle, stream: str, verb: str,
) -> None:
    """The refusal is the one ``FEMData.<verb>`` raises for that record,
    and the host keeps its record sets."""
    import dataclasses

    from apeGmsh.mesh._compose import _merge_bundle_into_fem

    host, bundle = host_and_bundle

    class Stray:
        pass

    with pytest.raises(TypeError) as per_record:
        getattr(host, verb)(Stray())
    before = [list(s) for s in _record_sets(host)]
    bad = dataclasses.replace(
        bundle, **{stream: (*getattr(bundle, stream), Stray())})
    with pytest.raises(TypeError) as batched:
        _merge_bundle_into_fem(host, bad)
    assert str(batched.value) == str(per_record.value)
    assert [list(s) for s in _record_sets(host)] == before


def test_merge_does_not_append_record_by_record(
    host_and_bundle, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """With the per-record transforms made to raise, the merge still
    answers, with the same records (a reverted loop raises here)."""
    import dataclasses

    from apeGmsh.mesh._compose import _merge_bundle_into_fem
    from apeGmsh.mesh.FEMData import FEMData

    host, bundle = host_and_bundle
    bundle = dataclasses.replace(bundle, **_every_record_kind(bundle))
    want = _per_record_reference(host, bundle)

    def _per_record(*_a, **_k):
        raise AssertionError("per-record with_* append in the merge")

    monkeypatch.setattr(FEMData, "with_constraint", _per_record)
    monkeypatch.setattr(FEMData, "with_load", _per_record)
    _assert_same_records(_merge_bundle_into_fem(host, bundle), want)
