"""Regression: BrowserTab renders staged groups now that
``SelectionState.staged_groups`` holds ``SelectionTarget`` (ADR 0045
keystone), not bare DimTags.

Before the keystone fix, ``BrowserTab.refresh()`` unpacked each member
as ``for dim, tag in members`` and called ``is_hidden((dim, tag))`` — both
assume a ``(dim, tag)`` tuple. A staged-but-unflushed group (the
``_on_new_group`` flow stages picks, then refreshes the browser before
flushing to Gmsh) put ``SelectionTarget`` objects on that path, which
unpacks/hashes wrong. This pins the conversion at the merge point.
"""
from __future__ import annotations

import os

import gmsh
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("qtpy.QtWidgets")

from apeGmsh.viewers.core.selection import SelectionState
from apeGmsh.viewers.ui._browser_tab import BrowserTab


@pytest.fixture(scope="module")
def qapp():
    from qtpy import QtWidgets
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    yield app


@pytest.fixture
def gmsh_session():
    gmsh.initialize()
    gmsh.model.add("browser_tab_staged")
    yield
    gmsh.finalize()


def test_browser_tab_renders_staged_target_group(qapp, gmsh_session):
    sel = SelectionState()
    sel.pick((3, 1))
    sel.pick((3, 2))
    # Mirror _on_new_group: stage current picks (SelectionTargets) and
    # register the group, WITHOUT flushing to Gmsh first.
    sel._staged_groups["Foo"] = list(sel._picks)
    sel._group_order.append("Foo")

    # __init__ calls refresh() — this used to raise on the unpack.
    tab = BrowserTab(sel)

    item = tab._group_items["Foo"]
    assert item.text(0) == "Foo"
    assert item.text(1) == "2"                       # member count
    # Two entity children, labelled from the DimTags (vol 1, vol 2).
    labels = {item.child(i).text(0) for i in range(item.childCount())}
    assert labels == {"vol 1", "vol 2"}
