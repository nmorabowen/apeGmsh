"""Selection ergonomics: ``select()`` dim-mismatch guard (#4) and the
chain-level ``.tags()`` terminal (#5).

Before this, ``g.model.select("vol", dim=2)`` on a volume label silently
returned the *volume* (``dim=`` is ignored for labels/PGs/parts), so a
following ``.to_physical("faces")`` registered a volume group — a silent
wrong-dimension footgun.  And ``select(...).tags()`` raised
``AttributeError`` because ``.tags()`` lived only on the ``.result()``
payload.
"""
from __future__ import annotations

import pytest

from apeGmsh import apeGmsh


# ---------------------------------------------------------------------
# #4 — dim-mismatch guard
# ---------------------------------------------------------------------


def test_label_dim_mismatch_raises() -> None:
    with apeGmsh(model_name="dim_guard") as g:
        g.model.geometry.add_box(0, 0, 0, 1, 1, 1, label="box")
        g.model.sync()
        with pytest.raises(ValueError, match=r"dim=2"):
            g.model.select("box", dim=2)
        # the message points at the remedy
        with pytest.raises(ValueError, match=r"\.boundary\(\)"):
            g.model.select("box", dim=1)


def test_label_dim_match_ok() -> None:
    with apeGmsh(model_name="dim_match") as g:
        g.model.geometry.add_box(0, 0, 0, 1, 1, 1, label="box")
        g.model.sync()
        sel = g.model.select("box", dim=3)
        assert sorted({d for d, _ in sel}) == [3]


def test_dim_omitted_still_returns_volume() -> None:
    """No explicit dim= => no guard; a volume label resolves to dim 3."""
    with apeGmsh(model_name="dim_omit") as g:
        g.model.geometry.add_box(0, 0, 0, 1, 1, 1, label="box")
        g.model.sync()
        assert sorted({d for d, _ in g.model.select("box")}) == [3]


def test_select_none_dim_returns_that_dim() -> None:
    """target=None with dim= selects every entity at that dim — the
    entities *are* at that dim, so the guard does not fire."""
    with apeGmsh(model_name="none_dim") as g:
        g.model.geometry.add_box(0, 0, 0, 1, 1, 1, label="box")
        g.model.sync()
        faces = list(g.model.select(None, dim=2))
        assert faces and {d for d, _ in faces} == {2}


def test_boundary_walk_is_the_documented_remedy() -> None:
    """The remedy the error suggests actually yields the faces."""
    with apeGmsh(model_name="boundary_walk") as g:
        g.model.geometry.add_box(0, 0, 0, 1, 1, 1, label="box")
        g.model.sync()
        faces = list(g.model.select("box", dim=3).boundary())
        assert faces and {d for d, _ in faces} == {2}
        assert len(faces) == 6  # a box has six boundary faces


# ---------------------------------------------------------------------
# #5 — chain-level .tags()
# ---------------------------------------------------------------------


def test_entityselection_tags_matches_result() -> None:
    with apeGmsh(model_name="tags") as g:
        g.model.geometry.add_box(0, 0, 0, 1, 1, 1, label="box")
        g.model.sync()
        sel = g.model.select("box", dim=3)
        tags = sel.tags()
        assert isinstance(tags, list) and all(isinstance(t, int) for t in tags)
        assert tags == sel.result().tags()


def test_entityselection_tags_after_boundary() -> None:
    with apeGmsh(model_name="tags_boundary") as g:
        g.model.geometry.add_box(0, 0, 0, 1, 1, 1, label="box")
        g.model.sync()
        face_tags = g.model.select("box", dim=3).boundary().tags()
        assert len(face_tags) == 6
