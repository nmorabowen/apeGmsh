"""AS5-b (G1): ``Assembly.instance(anchor=)``, v1's ``compose(anchor=)``.

v1 resolves ``anchor`` on the host composed so far: a physical group
first, then a label, and the module's ``translate`` becomes the mean of
that name's node coordinates. v2 has no host, so the anchor is a port
``"{instance}.{pg|label}"`` of an instance declared before. Oracles, each
independent of the code under test:

* the centroid of the block's top face (a 3x3 grid on ``z = H`` over
  ``[0, SIDE]^2``) is ``(SIDE/2, SIDE/2, H)``, and of the whole block
  ``(SIDE/2, SIDE/2, H/2)``; after a quarter turn about ``z`` and a shift
  ``t`` the anchor instance puts them at ``R c + t``;
* the bridged nodes of the anchored instance are its source's plus that
  translate;
* the archive stores the resolved translate, so ``from_h5`` re-lists the
  declared instance (the anchor is sugar for the translate, as in v1,
  whose ``/composed_from`` stores no anchor either).

Migrated from v1 (#1585, G1): ``test_compose_end_to_end.py::
test_compose_with_anchor_resolution``, ``::test_compose_with_anchor_conflict_raises``,
``::test_compose_with_unknown_anchor_raises``; ``test_compose_facade.py::
test_compose_anchor_with_nonzero_translate_raises``,
``::test_compose_anchor_with_zero_translate_passes_validation``;
``test_label_pg_collisions.py::test_compose_anchor_error_lists_names``.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from tests.assembly.test_two_instances_one_tie import (
    H,
    SIDE,
    block_fem,
    declare_block,
    write_instance,
)

#: A quarter turn about z, then a shift: the anchor instance's placement.
QUARTER = ((0.0, 0.0, 1.0), math.pi / 2)
SHIFT = (100.0, 0.0, 0.0)
#: ``R c + t`` for the top-face centroid ``c = (SIDE/2, SIDE/2, H)``.
TOP_PLACED = (SHIFT[0] - SIDE / 2, SIDE / 2, H)


@pytest.fixture(scope="module")
def block(tmp_path_factory) -> Path:
    d = tmp_path_factory.mktemp("anchor")
    return write_instance(d / "block.h5", block_fem(d), declare_block)


def _host(block):
    from apeGmsh.assembly import Assembly

    return Assembly("anchored").instance("host", block)


def _coords(fem, **sel) -> np.ndarray:
    return np.asarray(fem.nodes.select(**sel).coords, dtype=float)


# v1 ``test_compose_with_anchor_resolution``
def test_anchor_resolves_to_the_centroid_of_a_physical_group(block):
    asm = _host(block).instance("m", block, anchor="host.top")
    assert asm.instances[1].translate == (SIDE / 2, SIDE / 2, H)
    fem = asm.bridge(ndm=3, ndf=3).fem
    np.testing.assert_allclose(
        np.sort(_coords(fem, pg="m.Vol"), axis=0),
        np.sort(_coords(fem, pg="host.Vol") + (SIDE / 2, SIDE / 2, H), axis=0),
        rtol=0, atol=1e-12)


def test_a_name_that_is_no_group_falls_back_to_a_label(block):
    # ``v`` is the box's geometry label, not a physical group.
    asm = _host(block).instance("m", block, anchor="host.v")
    assert asm.instances[1].translate == (SIDE / 2, SIDE / 2, H / 2)


def test_the_anchor_is_where_its_instance_puts_it(block):
    from apeGmsh.assembly import Assembly

    asm = (Assembly("turned")
           .instance("host", block, rotate=QUARTER, translate=SHIFT)
           .instance("m", block, anchor="host.top"))
    np.testing.assert_allclose(asm.instances[1].translate, TOP_PLACED,
                               rtol=0, atol=1e-12)
    # The anchored instance may rotate too: rotation, then the translate.
    asm.instance("n", block, rotate=QUARTER, anchor="host.top")
    assert asm.instances[2].translate == asm.instances[1].translate
    assert asm.instances[2].rotate == QUARTER


# v1 ``test_compose_with_anchor_conflict_raises`` and
# ``test_compose_facade.py::test_compose_anchor_with_nonzero_translate_raises``
@pytest.mark.parametrize("translate", [(1.0, 0.0, 0.0), (0.0, 0.0, -1e-9)])
def test_an_anchor_with_a_nonzero_translate_raises_and_records_nothing(
        block, translate):
    from apeGmsh.assembly import AssemblyError

    asm = _host(block)
    with pytest.raises(AssemblyError, match="mutually exclusive"):
        asm.instance("m", block, anchor="host.top", translate=translate)
    assert [i.label for i in asm.instances] == ["host"]
    asm.instance("m", block, translate=translate)   # the label is still free


# v1 ``test_compose_facade.py::test_compose_anchor_with_zero_translate_passes_validation``
def test_an_anchor_with_an_explicit_zero_translate_resolves(block):
    asm = _host(block).instance(
        "m", block, anchor="host.top", translate=(0.0, 0.0, 0.0))
    assert asm.instances[1].translate == (SIDE / 2, SIDE / 2, H)


# v1 ``test_compose_with_unknown_anchor_raises`` and
# ``test_label_pg_collisions.py::test_compose_anchor_error_lists_names``
def test_an_unknown_anchor_raises_listing_the_instance_names(block):
    from apeGmsh.assembly import AssemblyError
    from apeGmsh.mesh import FEMData

    asm = _host(block)
    with pytest.raises(AssemblyError, match="resolves no node") as info:
        asm.instance("m", block, anchor="host.does_not_exist")
    msg = str(info.value)
    # Oracle: the source's own names, under the instance prefix.
    src = FEMData.from_h5(str(block))
    for name in src.nodes.physical.names() + src.nodes.labels.names():
        assert repr(f"host.{name}") in msg, (name, msg)
    assert [i.label for i in asm.instances] == ["host"]


@pytest.mark.parametrize("anchor, match", [
    ("top", "names no assembly object"),          # a bare name: no host in v2
    ("m.top", "unknown instance 'm'"),            # the instance being declared
    ("later.top", "unknown instance 'later'"),    # one not declared yet
    ("host.", "empty name"),
    ("", "non-empty string"),
])
def test_an_anchor_names_a_port_of_an_earlier_instance(block, anchor, match):
    from apeGmsh.assembly import AssemblyError

    asm = _host(block)
    with pytest.raises(AssemblyError, match=match):
        asm.instance("m", block, anchor=anchor)
    assert [i.label for i in asm.instances] == ["host"]


def test_the_resolved_translate_round_trips_through_the_archive(block, tmp_path):
    from apeGmsh.assembly import Assembly

    asm = (Assembly("turned")
           .instance("host", block, rotate=QUARTER, translate=SHIFT)
           .instance("m", block, anchor="host.top"))
    asm.bridge(ndm=3, ndf=3)
    out = tmp_path / "anchored.h5"
    asm.h5(out)
    back = Assembly.from_h5(out)
    assert back.instances == asm.instances
    np.testing.assert_allclose(back.instances[1].translate, TOP_PLACED,
                               rtol=0, atol=1e-12)
