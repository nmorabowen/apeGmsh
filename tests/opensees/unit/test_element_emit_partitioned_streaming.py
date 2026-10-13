"""S3-b (#1894): the partitioned element emit streams its rows.

``emit_element_spec_partitioned`` used to build an ``owned`` list of every
owned ``(eid, node_tags)`` row before writing the first element, which kept
all of the spec's node-tag tuples alive at once (ADR 0100, #1766 S3-a
attribution).  These tests pin the streaming form:

* a row-iterator spy sees the first element written after ONE row was
  consumed, not after the whole plan (fails against a ``list(...)`` revert);
* the emitted rows, order and tags match the owner filter, on the rank
  bucket path (whole plan owned) and the node-pair path (mixed owners);
* the ADR 0044 ASDConcrete sweep still warns once, with the same count and
  text, over this rank's owned rows only, and before any element is written;
* a raise mid-stream leaves the earlier rows written and propagates.
"""
from __future__ import annotations

import warnings
from types import SimpleNamespace
from typing import Any, Iterator

import numpy as np
import pytest

from apeGmsh.opensees._internal.build.element_emit import (
    emit_element_spec_partitioned,
)
from apeGmsh.opensees._internal.build.ownership import SortedIntToInt
from apeGmsh.opensees._internal.build.pg_expand import ElementPlanRows
from apeGmsh.opensees._internal.tag_resolution import current_element_nodes
from apeGmsh.opensees.material import _asdconcrete_laws as laws
from apeGmsh.opensees.material.nd import (
    ASDConcrete3D,
    ASDRegularizationWarning,
)


class _SpyRows(ElementPlanRows):
    """An element plan that counts the rows its current iteration yielded."""

    __slots__ = ("yielded", "iterations")

    def __init__(self, *a: Any, **kw: Any) -> None:
        super().__init__(*a, **kw)
        self.yielded = 0
        self.iterations = 0

    def __iter__(self) -> "Iterator[tuple[int, tuple[int, ...], int]]":
        self.iterations += 1
        self.yielded = 0
        for row in super().__iter__():
            self.yielded += 1
            yield row


class _Spec:
    """Duck-typed Element: records each ``_emit`` with the spy's progress."""

    def __init__(self, plan: "_SpyRows | None" = None, material: Any = None,
                 fail_at: "int | None" = None) -> None:
        self.pg = "block"
        self.material = material
        self.plan = plan
        self.fail_at = fail_at
        self.emitted: list[tuple[tuple[int, ...], int]] = []
        self.yielded_at_emit: list[int] = []

    def _emit(self, emitter: object, ele_tag: int) -> None:
        if self.fail_at is not None and len(self.emitted) == self.fail_at:
            raise RuntimeError("boom")
        self.emitted.append((current_element_nodes(emitter), int(ele_tag)))
        if self.plan is not None:
            self.yielded_at_emit.append(self.plan.yielded)


def _emit(spec: _Spec, plan: Any, owner: Any, rank: int, fem: Any = None) -> None:
    emit_element_spec_partitioned(
        spec=spec,  # type: ignore[arg-type]
        emitter=SimpleNamespace(),  # type: ignore[arg-type]
        fem=fem,
        pre_allocated=plan,
        base_resolver=lambda p: 0,
        transf_tag_for_element=None,
        partition_rank=rank,
        element_owner=owner,
    )


def _plan(n: int, cls: type = ElementPlanRows) -> Any:
    eids = np.arange(1, n + 1, dtype=np.int64)
    conn = np.arange(1, 3 * n + 1, dtype=np.int64).reshape(n, 3)
    return cls(eids, conn, tag_start=100)


def _owner(ranks: "list[int]") -> SortedIntToInt:
    return SortedIntToInt(
        np.arange(1, len(ranks) + 1, dtype=np.int64),
        np.asarray(ranks, dtype=np.int64),
    )


# ---------------------------------------------------------------------------
# Streaming: rows are consumed one at a time, not materialised up front
# ---------------------------------------------------------------------------


def test_bucket_path_streams_rows_to_the_writer() -> None:
    plan = _plan(6, _SpyRows)
    spec = _Spec(plan)
    _emit(spec, plan, _owner([2] * 6), rank=2)

    assert len(spec.emitted) == 6
    # Row k is written right after row k is yielded.  A ``list(...)`` of
    # the plan before the loop reads 6 at every emit.
    assert spec.yielded_at_emit == [1, 2, 3, 4, 5, 6]
    # Non-ASDConcrete spec: the sweep never walks the rows.
    assert plan.iterations == 1


def test_node_pair_path_streams_rows_through_the_owner_filter() -> None:
    plan = _plan(6, _SpyRows)
    spec = _Spec(plan)
    _emit(spec, plan, _owner([0, 1, 1, 0, 1, -1]), rank=1)

    assert spec.yielded_at_emit == [2, 3, 5]
    assert plan.iterations == 1


# ---------------------------------------------------------------------------
# Parity: the owner filter, row order and tags are unchanged
# ---------------------------------------------------------------------------


def _reference(plan: Any, owner: "dict[int, int]", rank: int) -> list:
    return [
        (tuple(nodes), tag)
        for eid, nodes, tag in plan
        if owner.get(eid, -1) == rank
    ]


@pytest.mark.parametrize("rank", [0, 1, 2, 7])
@pytest.mark.parametrize("owner_kind", ["columnar", "dict"])
@pytest.mark.parametrize("plan_kind", ["columnar", "tuples", "bucket"])
def test_emitted_rows_match_the_owner_filter(
    rank: int, owner_kind: str, plan_kind: str,
) -> None:
    ranks = [0, 1, 2, 1, 0, 2, 1, 1]
    owner_dict = {i + 1: r for i, r in enumerate(ranks)}
    owner: Any = _owner(ranks) if owner_kind == "columnar" else owner_dict
    plan: Any = _plan(len(ranks))
    if plan_kind == "tuples":
        plan = list(plan)
    elif plan_kind == "bucket":
        idx = np.flatnonzero(np.asarray(ranks) == rank)
        plan = plan.select_rows(idx)
    expected = _reference(plan, owner_dict, rank)

    spec = _Spec()
    _emit(spec, plan, owner, rank=rank)
    assert spec.emitted == expected


def _node_pair_plan() -> ElementPlanRows:
    """A 1-row node-pair plan: the whole plan goes to every rank."""
    from apeGmsh.opensees._internal.tag_resolution import MISSING_FEM_ELEMENT_ID

    return ElementPlanRows(
        np.asarray([MISSING_FEM_ELEMENT_ID], dtype=np.int64),
        np.asarray([[11, 12]], dtype=np.int64),
        tag_start=300,
    )


@pytest.mark.parametrize("case", ["mixed_plan", "node_pair"])
@pytest.mark.parametrize("rank", [0, 1, 5])
def test_bracket_opens_iff_this_rank_owns_a_row(
    monkeypatch, case: str, rank: int,
) -> None:
    """The bracket gate is "this rank owns >= 1 row", not "the plan is
    non-empty": a node-pair plan reaches every rank whole."""
    from apeGmsh.opensees._internal.build import element_emit
    from apeGmsh.opensees._internal.tag_resolution import MISSING_FEM_ELEMENT_ID

    opened: list[object] = []
    closed: list[object] = []
    monkeypatch.setattr(
        element_emit, "open_builder_ndf_bracket",
        lambda *a, **kw: opened.append(a) or True,
    )
    monkeypatch.setattr(
        element_emit, "close_builder_ndf_bracket",
        lambda *a, **kw: closed.append(a),
    )
    if case == "node_pair":
        plan: Any = _node_pair_plan()
        owner: Any = {MISSING_FEM_ELEMENT_ID: 0}   # the pair's one rank
        owns = rank == 0
    else:
        plan = _plan(3)
        owner = _owner([0, 1, 0])
        owns = rank in (0, 1)
    spec = _Spec()
    emit_element_spec_partitioned(
        spec=spec,  # type: ignore[arg-type]
        emitter=SimpleNamespace(),  # type: ignore[arg-type]
        fem=None,  # type: ignore[arg-type]
        pre_allocated=plan,
        base_resolver=lambda p: 0,
        transf_tag_for_element=None,
        partition_rank=rank,
        element_owner=owner,
        ndm=3,
        envelope_ndf=3,
    )
    assert len(opened) == len(closed) == (1 if owns else 0)
    assert bool(spec.emitted) is owns


# ---------------------------------------------------------------------------
# ADR 0044 sweep: aggregation unchanged, owned rows only, before any write
# ---------------------------------------------------------------------------


_E, _FT, _GF = 30_000.0, 3.0, 0.1
_LMAX = laws.l_max(_E, _GF, _FT)


def _line_fem(spacing: float, n_nodes: int) -> SimpleNamespace:
    """Nodes 1..n on the x-axis, ``spacing`` apart (node id -> index id-1)."""
    coords = np.zeros((n_nodes, 3))
    coords[:, 0] = spacing * np.arange(n_nodes)
    return SimpleNamespace(
        nodes=SimpleNamespace(index=lambda nid: int(nid) - 1, coords=coords),
    )


def _two_node_plan(n: int) -> ElementPlanRows:
    eids = np.arange(1, n + 1, dtype=np.int64)
    conn = np.stack([2 * eids - 1, 2 * eids], axis=1)  # disjoint node pairs
    return ElementPlanRows(eids, conn, tag_start=100)


def _conc() -> ASDConcrete3D:
    return ASDConcrete3D.from_fc(E=_E, v=0.2, fc=30.0, ft=_FT, Gf=_GF,
                                 lch_ref=50.0)


@pytest.mark.parametrize("path", ["node_pair_mask", "bucket"])
def test_asdconcrete_sweep_warns_once_over_owned_rows(path: str) -> None:
    from apeGmsh.opensees._internal.build.validators.materials import (
        sweep_asdconcrete_element_size,
    )

    ranks = [0, 1, 1, 0, 1]
    plan: Any = _two_node_plan(len(ranks))
    if path == "bucket":
        plan = plan.select_rows(np.flatnonzero(np.asarray(ranks) == 1))
    spec = _Spec(material=_conc())
    # Spacing grows along x, so the owned rows' lengths differ and the
    # warning's ``worst`` value depends on which rows were swept.
    coords = np.zeros((2 * len(ranks), 3))
    coords[1::2, 0] = _LMAX * (1.2 + 0.1 * np.arange(len(ranks)))
    coords[0::2, 1] = 10.0 * np.arange(len(ranks))
    coords[1::2, 1] = coords[0::2, 1]
    fem = SimpleNamespace(nodes=SimpleNamespace(
        index=lambda nid: int(nid) - 1, coords=coords))

    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        _emit(spec, plan, _owner(ranks), rank=1, fem=fem)
    hits = [w for w in rec if issubclass(w.category, ASDRegularizationWarning)]
    assert len(hits) == 1
    msg = str(hits[0].message)

    # Oracle: the sweep handed the materialised owned list, as main did.
    owned = [
        (e, tuple(int(t) for t in c))
        for e, c, _ in _two_node_plan(len(ranks)) if ranks[e - 1] == 1
    ]
    with warnings.catch_warnings(record=True) as ref:
        warnings.simplefilter("always")
        sweep_asdconcrete_element_size(spec, owned, fem)  # type: ignore[arg-type]
    assert len(ref) == 1
    assert msg == str(ref[0].message)
    # Rank 1 owns rows 2, 3, 5 (eids); all are over, the worst is eid 5.
    assert msg.startswith("ASDConcrete: 3/3 elements in PG 'block' exceed")
    assert f"worst lch={_LMAX * 1.6:g}" in msg
    assert len(spec.emitted) == 3


def test_asdconcrete_sweep_runs_before_the_first_element_is_written() -> None:
    plan = _two_node_plan(2)
    fem = _line_fem(spacing=_LMAX * 1.5, n_nodes=4)
    spec = _Spec(material=_conc())
    with warnings.catch_warnings():
        warnings.simplefilter("error", ASDRegularizationWarning)
        with pytest.raises(ASDRegularizationWarning):
            _emit(spec, plan, _owner([0, 0]), rank=0, fem=fem)
    assert spec.emitted == []


def test_asdconcrete_sweep_silent_within_ceiling() -> None:
    plan = _two_node_plan(3)
    fem = _line_fem(spacing=_LMAX * 0.5, n_nodes=6)
    spec = _Spec(material=_conc())
    with warnings.catch_warnings():
        warnings.simplefilter("error", ASDRegularizationWarning)
        _emit(spec, plan, _owner([0, 0, 0]), rank=0, fem=fem)
    assert len(spec.emitted) == 3


# ---------------------------------------------------------------------------
# Partial writes: a raise mid-stream keeps the earlier rows and propagates
# ---------------------------------------------------------------------------


def test_raise_mid_stream_keeps_earlier_rows_and_propagates() -> None:
    plan = _plan(5)
    spec = _Spec(fail_at=2)
    with pytest.raises(RuntimeError, match="boom"):
        _emit(spec, plan, _owner([0] * 5), rank=0)
    assert [tag for _, tag in spec.emitted] == [100, 101]
