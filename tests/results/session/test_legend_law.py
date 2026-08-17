"""§5 legend laws — caused by slots, owned by the view.

The ADR's oracle still, in IR form: deform on, every slot empty →
zero legends. Plus the one-scale merge (contour(S) + gauss(S)), the
non-colour-mapped exclusions, and hide-is-chrome.
"""
from __future__ import annotations

import pytest

from apeGmsh.results.session import (
    Contour,
    Deform,
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
# INV-LEGEND-1 — cause: colour-mapped occupied slots only, never deform
# ----------------------------------------------------------------------

def test_deform_on_slots_empty_means_zero_legends(view) -> None:
    view.deform = Deform(field="displacement", scale=100.0)
    assert view.legends() == ()  # the ADR's oracle still


def test_non_colour_mapped_slots_emit_nothing(view) -> None:
    view.line = Line("M")
    view.loads = Loads()
    view.reactions = Reactions()
    assert len(view.slots) == 3
    assert view.legends() == ()


def test_each_colour_mapped_slot_causes_a_legend(view) -> None:
    view.contour = Contour("stress_von_mises")
    view.vector = Vector("displacement")
    view.sand = Sand("damage")
    fields = {legend.field for legend in view.legends()}
    assert fields == {"stress_von_mises", "displacement", "damage"}


# ----------------------------------------------------------------------
# INV-LEGEND-2 — presence follows occupancy
# ----------------------------------------------------------------------

def test_slot_on_creates_slot_off_destroys(view) -> None:
    view.contour = Contour("S")
    assert [legend.field for legend in view.legends()] == ["S"]
    view.contour = None
    assert view.legends() == ()


# ----------------------------------------------------------------------
# One scale for one field — contour(S) + gauss(S)
# ----------------------------------------------------------------------

def test_same_field_on_two_slots_is_one_scale(view) -> None:
    view.contour = Contour("S")
    view.gauss = Gauss("S")
    legends = view.legends()
    assert len(legends) == 1
    assert legends[0].field == "S"
    assert legends[0].categories == ("contour", "gauss")


def test_averaging_does_not_break_the_match(view) -> None:
    # Averaging is contour display state, not field identity (§4/§5).
    view.contour = Contour("S", averaging="unaveraged")
    view.gauss = Gauss("S")
    assert len(view.legends()) == 1


def test_different_fields_are_different_scales(view) -> None:
    view.contour = Contour("S")
    view.gauss = Gauss("T")
    assert [(lg.field, lg.categories) for lg in view.legends()] == [
        ("S", ("contour",)),
        ("T", ("gauss",)),
    ]


# ----------------------------------------------------------------------
# INV-LEGEND-3 — hide is view chrome; it clears nothing
# ----------------------------------------------------------------------

def test_hiding_a_scale_does_not_clear_the_slot(view) -> None:
    view.contour = Contour("S")
    view.set_legend_hidden("S")
    (legend,) = view.legends()
    assert legend.hidden is True
    assert view.contour == Contour("S")  # picture untouched
    view.set_legend_hidden("S", False)
    (legend,) = view.legends()
    assert legend.hidden is False


def test_hiding_a_nonexistent_legend_refuses(view) -> None:
    with pytest.raises(ValueError, match="No legend for field 'S'"):
        view.set_legend_hidden("S")


def test_chrome_dies_with_the_legend(view) -> None:
    # INV-LEGEND-2: slot off destroys the legend; a re-created legend
    # is a new legend — default visible.
    view.contour = Contour("S")
    view.set_legend_hidden("S")
    view.contour = None
    view.contour = Contour("S")
    (legend,) = view.legends()
    assert legend.hidden is False


def test_chrome_survives_while_another_slot_keeps_the_field(view) -> None:
    view.contour = Contour("S")
    view.gauss = Gauss("S")
    view.set_legend_hidden("S")
    view.contour = None  # gauss still causes the S legend
    (legend,) = view.legends()
    assert legend.field == "S" and legend.hidden is True


# ----------------------------------------------------------------------
# INV-LEGEND-4 — identity: the scale names the slot quantity
# ----------------------------------------------------------------------

def test_warped_mesh_contour_stress_legend_says_stress(view) -> None:
    view.deform = Deform(field="displacement", scale=50.0)
    view.contour = Contour("stress_von_mises")
    (legend,) = view.legends()
    assert legend.field == "stress_von_mises"  # never the warp field


# ----------------------------------------------------------------------
# INV-LEGEND-5 — per view
# ----------------------------------------------------------------------

def test_each_view_carries_the_legends_of_its_own_slots() -> None:
    session = ResultsSession()
    a = session.add_view()
    b = session.add_view()
    a.contour = Contour("S")
    assert [legend.field for legend in a.legends()] == ["S"]
    assert b.legends() == ()
