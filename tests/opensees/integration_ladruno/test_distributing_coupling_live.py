"""Fork-only element end-to-end — RBE3 LadrunoDistributingCoupling runs.

Drives the *same emit code path* the ``g.constraints.distributing_coupling``
generator uses (``_emit_one_interpolation`` -> ``element
LadrunoDistributingCoupling``) into the live *fork* domain and asserts the
fork-only RBE3 element actually loads (appears in ``getEleTags``). This is
the reference-point-to-surface half of the "online" proof: the
ndf-mismatch moment-transfer element built by apeGmsh's emit layer and run
on the fork build.

The resolver (geometry -> InterpolationRecord) is covered by
tests/test_constraint_resolver.py + tests/opensees/unit/
test_distributing_coupling_emit.py; this test covers the missing
"does it run on the fork through apeGmsh's emitter" leg. Gated on the
backend resolver via the ``ladruno_fork`` marker.
"""
from __future__ import annotations

import pytest

from apeGmsh._kernel.records._constraints import InterpolationRecord
from apeGmsh.opensees._internal.build import _emit_one_interpolation
from apeGmsh.opensees._internal.tag_allocator import TagAllocator
from apeGmsh.opensees.emitter.live import LiveOpsEmitter

pytestmark = pytest.mark.ladruno_fork


def test_rbe3_element_loads_on_fork() -> None:
    e = LiveOpsEmitter(wipe=True)
    e.model(ndm=3, ndf=3)

    # Independent set: a unit square FACE of 3-DOF (translation-only) nodes.
    face = {1: (0.0, 0.0, 0.0), 2: (1.0, 0.0, 0.0),
            3: (1.0, 1.0, 0.0), 4: (0.0, 1.0, 0.0)}
    for tag, (x, y, z) in face.items():
        e.node(tag, x, y, z)
    # Reference point above the face — 6 DOF (3 trans + 3 rot); RBE3 spreads
    # its load/motion over the 3-DOF face as a self-equilibrated force couple.
    e.node(10, 0.5, 0.5, 1.0, ndf=6)

    # Same record the resolver produces; same emit branch the generator hits.
    rec = InterpolationRecord(
        kind="distributing", slave_node=10, master_nodes=[1, 2, 3, 4],
        weights=None,
    )
    _emit_one_interpolation(e, rec, TagAllocator())

    # The fork-only-element gate in LiveOpsEmitter.element confirms the build
    # actually accepted LadrunoDistributingCoupling (a stock build would have
    # raised/dropped it).
    assert "LadrunoDistributingCoupling" in e._fork_verified_types
    tags = e.ops.getEleTags() or []
    if isinstance(tags, int):
        tags = [tags]
    assert len(tags) >= 1

    get_class = getattr(e.ops, "getEleClassTags", None)
    if get_class is not None:
        class_tags: list[int] = []
        for t in tags:
            ct = get_class(t)
            class_tags.extend(ct if isinstance(ct, (list, tuple)) else [ct])
        assert 33011 in class_tags  # LadrunoDistributingCoupling
