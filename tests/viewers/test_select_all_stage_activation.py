"""#1743 — outline select-all selects only what the pane draws.

X2-f (#1729) narrowed every realized mesh pane to the elements ACTIVE in
its stage (``_realize._stage_active_scope``). Select-all resolved its
category without that filter, so on a staged model it selected elements
the pane was not drawing. The oracle here is the realize path itself:
the cell set ``_realize_mesh`` hands its layers, captured at the
``_stage_active_scope`` seam, at each stage.

Fixture: ``test_x2f_animation_stage_activation.staged`` — Rock (z 0-1)
is global, Fill (z 1-2) activates in ``construction``, Rock is removed
in ``excavate``.
"""
from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import pytest

pytest.importorskip("pyvista")

from tests.viewers.test_x2f_animation_stage_activation import (  # noqa: F401
    _EXPECT,
    _pg_eids,
    staged,
)

_BOTH = ("Rock", "Fill")


def _realized_set(session: Any, view: Any, monkeypatch: Any) -> Any:
    """The ScopedSet realize draws for ``view`` at the session's instant."""
    from apeGmsh.viewers.session import _realize as realize_mod

    from tests.viewers.conftest import RecordingBackend

    seen: list[Any] = []
    real = realize_mod._stage_active_scope

    def spy(*a: Any, **k: Any) -> Any:
        out = real(*a, **k)
        seen.append(out)
        return out

    with monkeypatch.context() as m:
        m.setattr(realize_mod, "_stage_active_scope", spy)
        realize_mod.realize_pane(session, view, RecordingBackend())
    assert len(seen) == 1
    return seen[0]


def _nodes_of(results: Any, eids: "set[int]") -> "set[int]":
    from apeGmsh.viewers.data import ViewerData
    from apeGmsh.viewers.session._scope import _nodes_of_elements

    arr = np.asarray(sorted(eids), dtype=np.int64)
    return {int(i) for i in _nodes_of_elements(
        arr, ViewerData.from_fem(results.fem),
    )}


@pytest.mark.parametrize("stage", sorted(_EXPECT))
def test_select_all_equals_what_realize_draws(staged, monkeypatch, stage):
    """At each stage, selecting every node of the whole model selects
    exactly the nodes of the cells realize draws at that stage."""
    from apeGmsh.results.session import Instant
    from apeGmsh.viewers.session import select_all

    results, _n = staged
    ids = {s.name: s.id for s in results.stages}
    session = results.session()
    view = session.panes[0]
    session.time = Instant(ids[stage], 0)

    drawn = _realized_set(session, view, monkeypatch)
    assert drawn.element_ids is not None  # the staged filter ran

    n = select_all(
        session, "nodes", axis="physical_groups", names=_BOTH, view=view,
    )
    selected = {int(i) for i in session.selection.nodes}
    assert n == len(selected)
    assert selected == {int(i) for i in drawn.node_ids}, stage

    # And against the fixture's own truth, independent of realize.
    pgs = _pg_eids(results)
    _z, rock, fill = _EXPECT[stage]
    active = (pgs["Rock"] if rock else set()) | (pgs["Fill"] if fill else set())
    assert selected == _nodes_of(results, active), stage


@pytest.mark.parametrize(
    "category, stage, want",
    [
        ("Rock", "construction", "Rock"),
        ("Fill", "construction", "Fill"),
        ("Fill", "excavate", "Fill"),
        ("Rock", "g", "Rock"),
    ],
)
def test_category_intersects_active(staged, category, stage, want):
    """A category select-all is category AND active."""
    from apeGmsh.results.session import Instant
    from apeGmsh.viewers.session import select_all

    results, _n = staged
    ids = {s.name: s.id for s in results.stages}
    session = results.session()
    view = session.panes[0]
    session.time = Instant(ids[stage], 0)
    select_all(
        session, "nodes", axis="physical_groups", names=(category,),
        view=view,
    )
    selected = {int(i) for i in session.selection.nodes}
    assert selected == _nodes_of(results, _pg_eids(results)[want])


@pytest.mark.parametrize("category, stage", [("Fill", "g"), ("Rock", "excavate")])
def test_inactive_category_selects_nothing_loudly(staged, category, stage):
    """Fail closed: a category with no element active in the stage has
    nothing drawn to select — select_all raises, counts reports (0, 0)
    (a disabled outline row), and the store is untouched."""
    from apeGmsh.results.session import Instant
    from apeGmsh.viewers.session import select_all
    from apeGmsh.viewers.session._select_all import counts

    results, _n = staged
    ids = {s.name: s.id for s in results.stages}
    session = results.session()
    view = session.panes[0]
    session.time = Instant(ids[stage], 0)
    assert counts(
        session, axis="physical_groups", names=(category,), view=view,
    ) == (0, 0)
    with pytest.raises(ValueError, match="activates none"):
        select_all(
            session, "nodes", axis="physical_groups", names=(category,),
            view=view,
        )
    assert session.selection.kind is None


@pytest.mark.parametrize("why", ["no_program", "unmatched_stage"])
def test_no_activation_data_selects_everything_silently(
    staged, monkeypatch, why,
):
    """Fail-soft where realize is: no staged program, or an unmatched
    stage, selects the whole category — and select-all does not repeat
    realize's unmatched-stage warning."""
    from apeGmsh.results.session import Instant
    from apeGmsh.viewers.data import _stage_activation as SA
    from apeGmsh.viewers.session import select_all

    results, _n = staged
    if why == "no_program":
        monkeypatch.setattr(SA, "build_from_model", lambda *a, **k: None)
    else:
        monkeypatch.setattr(SA, "pair_capture_to_program", lambda *a, **k: {})
    ids = {s.name: s.id for s in results.stages}
    session = results.session()
    view = session.panes[0]
    session.time = Instant(ids["g"], 0)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        select_all(
            session, "nodes", axis="physical_groups", names=_BOTH, view=view,
        )
    assert not [w for w in caught if "Stage activation" in str(w.message)]
    selected = {int(i) for i in session.selection.nodes}
    pgs = _pg_eids(results)
    assert selected == _nodes_of(results, pgs["Rock"] | pgs["Fill"])
