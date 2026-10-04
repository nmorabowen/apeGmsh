"""#1335 (1): a registering terminal on an empty selection raises.

``.in_box`` uses BRep containment, so a box a hair too small matches
nothing; ``.to_physical(name)`` then used to create no group and say
nothing, and the missing name surfaced far downstream.  All four
registering terminals (``EntitySelection`` / ``Selection`` x
``to_physical`` / ``to_label``) now raise ``ValueError`` at the call
site, naming the group, and create nothing.

Oracle: the issue's own reproduction.  The tolerance-widened box selects
exactly the bottom curve ``[(1, 1)]`` (the control); the shrunk box
selects nothing, and registering it must raise.
"""
from __future__ import annotations

import gmsh
import pytest

W, D, TOL = 10.0, 5.0, 1e-6


def _soil(g):
    g.model.geometry.add_rectangle(-W / 2, -D, 0, W, D, label="soil")
    g.physical.add_surface("soil", name="Soil")


def _good_box(g):
    return g.model.select(None, dim=1).in_box(
        (-W / 2 - TOL, -D - TOL, -1), (W / 2 + TOL, -D + TOL, 1))


def _bad_box(g):
    # Shrunk in x by 2*TOL: the bottom curve no longer lies inside.
    return g.model.select(None, dim=1).in_box(
        (-W / 2 + TOL, -D - TOL, -1), (W / 2 - TOL, -D + TOL, 1))


def _pg_names() -> set[str]:
    return {gmsh.model.getPhysicalName(d, t)
            for d, t in gmsh.model.getPhysicalGroups()}


def test_control_box_selects_the_bottom_curve(g):
    _soil(g)
    assert list(_good_box(g)) == [(1, 1)]
    _good_box(g).to_physical("Bottom")
    assert "Bottom" in _pg_names()


@pytest.mark.parametrize("terminal", ["to_physical", "to_label"])
@pytest.mark.parametrize("payload", ["chain", "result"])
def test_empty_selection_terminal_raises_naming_the_group(
        g, terminal, payload):
    _soil(g)
    sel = _bad_box(g)
    assert list(sel) == []
    if payload == "result":
        sel = sel.result()
    owner = "EntitySelection" if payload == "chain" else "Selection"
    before_pgs = _pg_names()
    before_labels = set(g.labels.get_all())

    with pytest.raises(ValueError) as exc:
        getattr(sel, terminal)("Bottom")

    msg = str(exc.value)
    assert f"{owner}.{terminal}('Bottom')" in msg
    assert "selection is empty" in msg
    assert "BRep containment" in msg
    # Nothing was registered.
    assert _pg_names() == before_pgs
    assert set(g.labels.get_all()) == before_labels
