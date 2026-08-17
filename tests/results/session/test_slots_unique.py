"""§4 slot laws — closed catalog, unique per category, fill replaces.

Plain pytest, no Qt/VTK/GL: the laws live on the IR objects.
"""
from __future__ import annotations

import pytest

from apeGmsh.results.session import (
    COLOUR_MAPPED,
    SLOT_CATALOG,
    Contour,
    Gauss,
    Line,
    Loads,
    Reactions,
    ResultsSession,
    Sand,
    Vector,
)


@pytest.fixture
def view():
    return ResultsSession().add_view()


# ----------------------------------------------------------------------
# The catalog is closed and exactly the §4 table
# ----------------------------------------------------------------------

def test_catalog_is_the_seven_categories_in_table_order() -> None:
    assert list(SLOT_CATALOG) == [
        "contour", "vector", "gauss", "line", "sand", "loads", "reactions",
    ]


def test_colour_mapped_is_the_yes_column() -> None:
    assert COLOUR_MAPPED == {"contour", "vector", "gauss", "sand"}


# ----------------------------------------------------------------------
# Fill replaces (the §4 uniqueness law)
# ----------------------------------------------------------------------

def test_filling_an_occupied_slot_replaces_the_occupant(view) -> None:
    view.contour = Contour("stress_von_mises")
    view.contour = Contour("displacement_x")
    assert view.contour == Contour("displacement_x")
    assert list(view.slots) == ["contour"]  # one occupant, not two


def test_replace_holds_for_every_category(view) -> None:
    first_second = {
        "contour": (Contour("a"), Contour("b")),
        "vector": (Vector("a"), Vector("b")),
        "gauss": (Gauss("a"), Gauss("b")),
        "line": (Line("N"), Line("M")),
        "sand": (Sand("a"), Sand("b")),
        "loads": (Loads("p1"), Loads("p2")),
        "reactions": (Reactions(), Reactions()),
    }
    assert set(first_second) == set(SLOT_CATALOG)
    for category, (first, second) in first_second.items():
        setattr(view, category, first)
        setattr(view, category, second)
        assert getattr(view, category) == second
        assert list(view.slots).count(category) == 1


# ----------------------------------------------------------------------
# Different categories stack
# ----------------------------------------------------------------------

def test_categories_stack(view) -> None:
    view.contour = Contour("S")
    view.line = Line("M")
    view.reactions = Reactions()
    assert list(view.slots) == ["contour", "line", "reactions"]


def test_slots_reports_catalog_order_not_fill_order(view) -> None:
    view.reactions = Reactions()
    view.contour = Contour("S")
    view.gauss = Gauss("S")
    assert list(view.slots) == ["contour", "gauss", "reactions"]


# ----------------------------------------------------------------------
# Clearing
# ----------------------------------------------------------------------

def test_none_clears_the_slot(view) -> None:
    view.line = Line("N")
    view.line = None
    assert view.line is None
    assert view.slots == {}


def test_clearing_an_empty_slot_is_a_silent_noop() -> None:
    session = ResultsSession()
    view = session.add_view()
    ticks = []
    session.subscribe(lambda: ticks.append(1))
    view.contour = None
    assert not ticks  # no state change, no tick


# ----------------------------------------------------------------------
# The catalog is typed — a foreign record refuses loudly
# ----------------------------------------------------------------------

def test_wrong_record_type_is_a_type_error(view) -> None:
    with pytest.raises(TypeError, match="'contour' slot"):
        view.contour = Vector("displacement")
    with pytest.raises(TypeError, match="'line' slot"):
        view.line = "M"
    with pytest.raises(TypeError, match="'reactions' slot"):
        view.reactions = Loads()


# ----------------------------------------------------------------------
# Record validation is loud
# ----------------------------------------------------------------------

def test_empty_tokens_refused() -> None:
    with pytest.raises(ValueError, match="Contour.quantity"):
        Contour("")
    with pytest.raises(ValueError, match="Line.component"):
        Line("")
    with pytest.raises(ValueError, match="Vector.quantity"):
        Vector("")


def test_contour_averaging_vocabulary() -> None:
    assert Contour("S").averaging == "averaged"
    assert Contour("S", averaging="unaveraged").averaging == "unaveraged"
    with pytest.raises(ValueError, match="averaged"):
        Contour("S", averaging="discrete")  # old-style token refused


def test_loads_pattern_optional() -> None:
    assert Loads().pattern is None
    assert Loads("push").pattern == "push"
    with pytest.raises(ValueError, match="Loads.pattern"):
        Loads("")
