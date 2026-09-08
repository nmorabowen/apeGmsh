"""``ops.element.<X>(pg=...)`` on a zero-element selection fails loud.

Adversarial review of A10: a physical group with no elements of the
primitive's dimension in the FEM snapshot (e.g. its cells were excluded by
``get_fem_data(dim=...)``) used to silently allocate zero element tags and
emit nothing — the section/material lines still went out, but the element
declaration itself vanished with no error. ``allocate_element_tags`` (the
single choke point every emit path — live/tcl/py/h5, staged/unstaged,
partitioned/unpartitioned — funnels ``Element`` specs through) now raises
:class:`BridgeError` naming the primitive, the ``pg=``, and what the
snapshot's physical-group registry knows about it.
"""
from __future__ import annotations

import numpy as np
import pytest

from apeGmsh.opensees._internal import build as build_mod
from apeGmsh.opensees._internal.build import BridgeError, allocate_element_tags
from apeGmsh.opensees._internal.tag_allocator import TagAllocator

from ..fixtures.fem_stub import _ElementGroupView, _ElementsStub, _NodesStub, FEMStub


class _ShellMITC4:
    def __init__(self, pg: str) -> None:
        self.pg = pg


def _fem_with_pgs(pgs: dict) -> FEMStub:
    nodes = _NodesStub(ids=[1, 2, 3, 4], coords=[
        (0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (1.0, 1.0, 0.0), (0.0, 1.0, 0.0),
    ], node_pgs={})
    elements = _ElementsStub(elem_pgs=pgs)
    return FEMStub(nodes=nodes, elements=elements)


def _empty_group() -> _ElementGroupView:
    # A real empty ElementGroup still carries a 2-D connectivity shape
    # (npe fixed by element type even with zero rows); mirror that here
    # rather than an untyped empty tuple.
    return _ElementGroupView(
        ids=np.empty((0,), dtype=np.int64),
        connectivity=np.empty((0, 4), dtype=np.int64),
    )


def test_empty_pg_selection_raises_naming_primitive_and_pg() -> None:
    fem = _fem_with_pgs({"skin": _empty_group()})
    spec = _ShellMITC4(pg="skin")

    with pytest.raises(BridgeError, match=r"ShellMITC4\(pg='skin'\) selected 0 elements"):
        allocate_element_tags([spec], fem, TagAllocator())


def test_empty_pg_selection_message_mentions_missing_registry() -> None:
    """The fem_stub carries no ``.physical`` registry (only real FEMData
    does) — the diagnostic degrades to a plain statement rather than
    crashing on the missing attribute."""
    fem = _fem_with_pgs({"skin": _empty_group()})
    spec = _ShellMITC4(pg="skin")

    with pytest.raises(BridgeError, match="no registered cells on this snapshot"):
        allocate_element_tags([spec], fem, TagAllocator())


def test_nonempty_pg_selection_unaffected() -> None:
    """The normal path — a pg= selection with real elements — is unchanged:
    no raise, and the plan carries the expected fan-out."""
    fem = _fem_with_pgs({
        "Cols": _ElementGroupView(ids=(1, 2), connectivity=((1, 2), (3, 4))),
    })
    spec = _ShellMITC4(pg="Cols")

    plan = allocate_element_tags([spec], fem, TagAllocator())
    assert len(plan) == 1
    _returned_spec, rows = plan[0]
    assert len(rows) == 2
    assert list(rows.eids) == [1, 2]


def test_node_pair_spec_never_gated() -> None:
    """A node-pair spec (``pg=None``) always fans out to exactly one
    synthetic element and must never trip the empty-pg gate."""

    class _NodePairSpec:
        pg = None
        nodes = (1, 2)

    fem = _fem_with_pgs({})
    monkeypatched = build_mod.expand_spec_to_elements(fem, _NodePairSpec())
    assert len(monkeypatched) == 1  # sanity: never empty for node-pair form

    plan = allocate_element_tags([_NodePairSpec()], fem, TagAllocator())
    assert len(plan) == 1
    _spec, rows = plan[0]
    assert len(rows) == 1
