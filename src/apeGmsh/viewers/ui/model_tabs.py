"""
Model Tabs — UI components for the BRep model viewer.

This module is a thin re-export shim. The class implementations live
in dedicated modules so each focused concern has its own file:

* :class:`BrowserTab`          — :mod:`apeGmsh.viewers.ui._browser_tab`
* :class:`FilterTab` / :class:`ViewTab`
                               — :mod:`apeGmsh.viewers.ui._filter_view_tabs`
* :class:`SelectionTreePanel`  — :mod:`apeGmsh.viewers.ui._selection_tree`
* :class:`PartsTreePanel`      — :mod:`apeGmsh.viewers.ui._parts_tree`

All names previously imported from ``apeGmsh.viewers.ui.model_tabs``
continue to be available here unchanged.

Usage::

    from apeGmsh.viewers.ui.model_tabs import (
        BrowserTab, FilterTab, ViewTab,
        SelectionTreePanel, PartsTreePanel,
    )
"""
from __future__ import annotations

from ._browser_tab import BrowserTab
from ._filter_view_tabs import FilterTab, ViewTab
from ._parts_tree import PartsTreePanel
from ._selection_tree import SelectionTreePanel


__all__ = [
    "BrowserTab",
    "FilterTab",
    "ViewTab",
    "SelectionTreePanel",
    "PartsTreePanel",
]
