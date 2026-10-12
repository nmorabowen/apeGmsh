"""``emit_owned_nodes_with_inferred_ndf`` against the per-node loop it
replaced in ``_emit_partitioned`` (#1670): same calls, same order, same
argument values and types."""
from __future__ import annotations

from typing import Any

import numpy as np

from apeGmsh.opensees._internal.build.ndf import (
    _emit_node_with_inferred_ndf,
    emit_owned_nodes_with_inferred_ndf,
)
from apeGmsh.opensees._internal.build.ownership import node_index_lookup


class _Rec:
    def __init__(self) -> None:
        self.calls: list[tuple[Any, ...]] = []

    def node(self, tag: int, *coords: float, ndf: "int | None" = None) -> None:
        self.calls.append(
            (tag, tag.__class__, coords, tuple(c.__class__ for c in coords), ndf),
        )


def _reference(rec, inferred, node_ids, lookup, coords, env, exclude):
    for nid in sorted(int(n) for n in node_ids):
        if any(nid in s for s in exclude):
            continue
        idx = lookup.get(nid)
        if idx is None:
            continue
        xyz = coords[idx]
        _emit_node_with_inferred_ndf(
            rec, inferred, int(nid),
            (float(xyz[0]), float(xyz[1]), float(xyz[2])), env,
        )


def test_matches_reference_loop() -> None:
    rng = np.random.default_rng(3)
    ids = rng.permutation(np.arange(1, 200)).astype(object)
    coords = rng.normal(size=(ids.shape[0], 3))
    coords[5] = (0.0, -0.0, 1e-300)
    lookup = node_index_lookup(ids)
    inferred = {int(n): 6 for n in ids[::7]} | {int(ids[1]): 3}
    # unsorted, duplicated, and missing (500, 999) owned ids
    owned = [int(x) for x in rng.choice(np.arange(1, 200), 80)] + [500, 999, 4, 4]
    exclude = [{10, 11, 12}, {int(ids[3]): 0}]
    for ex in ([], exclude):
        ref, got = _Rec(), _Rec()
        _reference(ref, inferred, owned, lookup, coords, 3, ex)
        emit_owned_nodes_with_inferred_ndf(
            got, inferred, np.asarray(owned, dtype=object), lookup, coords, 3,
            exclude=ex,
        )
        assert got.calls == ref.calls
        assert got.calls  # non-vacuous


def test_empty_rank() -> None:
    lookup = node_index_lookup(np.array([1, 2], dtype=object))
    got = _Rec()
    emit_owned_nodes_with_inferred_ndf(
        got, {}, (), lookup, np.zeros((2, 3)), 3,
    )
    assert got.calls == []
