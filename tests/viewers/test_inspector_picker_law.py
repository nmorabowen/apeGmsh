"""ADR 0098 A6 G2 — every applicable slot is reachable from the UI.

The law: **if the data supports a slot, the inspector must be able to
author it.** Obvious once written down, and nowhere in ADR 0098 before
Amendment 6, which is why nothing enforced it.

What it costs to omit: the Line row filtered ``_LINE_COMPONENTS``
against the nodal and Gauss sets, and a line component appears in
neither — it lives under ``line_stations``. The test could never pass,
so the row offered nothing and its Add button was dead **on every model
since it shipped**. Every automated check went through the Python
surface (``view.line = Line(...)``, which works and is tested), so
2333 tests and an acceptance gate all stayed green over a control no
human could use.

This module tests the OTHER surface: the vocabulary the pickers offer.
It is deliberately generic over the §4 catalog, so a slot added later
inherits the law instead of needing someone to remember it.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from apeGmsh.results import Results
from apeGmsh.results.session import SLOT_CATALOG
from apeGmsh.results.writers import NativeWriter

from tests.conftest import _open_model_from_h5

STAGE = "s0"

#: Which recorded family each slot's vocabulary comes from. This map is
#: the point of the test: getting it wrong is exactly the Line bug.
SLOT_FAMILY = {
    "contour": ("nodes", "gauss"),
    "vector": ("nodes", "gauss"),
    "gauss": ("gauss",),
    "line": ("line_stations",),
    "sand": ("nodes",),
    "loads": (),        # sourced from the model's load patterns
    "reactions": (),    # no quantity to choose — it is add-or-not
}


@pytest.fixture(scope="module")
def qapp():
    from qtpy import QtWidgets

    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


@pytest.fixture
def multi_family(g, tmp_path: Path):
    """One stage recording nodes, Gauss AND line stations.

    The bench does this with a 3-minute solve; this does it with a box
    and hand-written arrays, because what is under test is the
    inspector's vocabulary, not the physics.
    """
    g.model.geometry.add_box(0, 0, 0, 1, 1, 1, label="box")
    g.physical.add_volume("box", name="Box")
    g.mesh.sizing.set_global_size(2.0)
    g.mesh.generation.generate(dim=3)
    fem = g.mesh.queries.get_fem_data(dim=3)

    node_ids = np.asarray(fem.nodes.ids, dtype=np.int64)
    elem_ids = np.concatenate(
        [np.asarray(gr.ids, dtype=np.int64) for gr in fem.elements]
    )
    coords = np.asarray(fem.nodes.coords, dtype=np.float64)

    disp = np.zeros((2, node_ids.size))
    disp[1] = coords[:, 2] * 10.0
    stress = np.zeros((2, elem_ids.size, 2))
    stress[1, :, 0] = elem_ids * 1.0
    stress[1, :, 1] = elem_ids * 2.0
    line_ids = elem_ids[:4]
    moment = np.zeros((2, line_ids.size, 2))
    moment[1, :, 0] = line_ids * 1.0

    path = tmp_path / "multi_family.h5"
    with NativeWriter(path) as w:
        w.open(fem=fem)
        sid = w.begin_stage(
            name=STAGE, kind="static", stage_id=STAGE,
            time=np.array([0.0, 1.0]),
        )
        w.write_nodes(
            sid, "partition_0", node_ids=node_ids,
            components={
                "displacement_x": disp, "displacement_y": disp,
                "displacement_z": disp,
                "reaction_force_x": disp, "reaction_force_y": disp,
                "reaction_force_z": disp,
            },
        )
        w.write_gauss_group(
            sid, "partition_0", "group_0",
            class_tag=4, int_rule=1,
            element_index=elem_ids,
            natural_coords=np.array([[-0.5, 0.0, 0.0], [0.5, 0.0, 0.0]]),
            components={
                "stress_xx": stress, "stress_yy": stress,
                "stress_zz": stress, "stress_xy": stress,
            },
        )
        w.write_line_stations_group(
            sid, "partition_0", "group_1",
            class_tag=1, int_rule=0,
            element_index=line_ids,
            station_natural_coord=np.array([0.0, 1.0]),
            components={
                "bending_moment_z": moment, "axial_force": moment,
            },
        )
        w.end_stage()
    return Results.from_native(path, model=_open_model_from_h5(path))


def _rows(qapp, results):
    """The inspector's slot rows for a default view, by category."""
    from apeGmsh.viewers.session._inspector import MeshInspectorPage

    session = results.session()
    page = MeshInspectorPage(session, session.panes[0])
    return page, {row.category: row for row in page._build_slot_rows()}


# ----------------------------------------------------------------------
# The law
# ----------------------------------------------------------------------

def test_the_family_map_covers_the_whole_catalog():
    """A slot added to §4 without a decision about where its vocabulary
    comes from fails here, not in a user's hands."""
    assert set(SLOT_FAMILY) == set(SLOT_CATALOG)


@pytest.mark.parametrize(
    "category", sorted(c for c, f in SLOT_FAMILY.items() if f)
)
def test_a_slot_whose_family_is_recorded_offers_a_quantity(
    qapp, multi_family, category,
):
    """THE law. The fixture records nodes, Gauss and line stations, so
    every quantity-bearing slot must have something to offer. An empty
    picker here means a control a human cannot use, however well the
    slot works from Python."""
    _page, rows = _rows(qapp, multi_family)
    row = rows[category]
    assert _add_enabled(row), (
        f"the {category!r} row offers no quantity, though the stage "
        f"records {SLOT_FAMILY[category]}. Its Add button is dead: the "
        f"slot is unreachable from the UI."
    )


@pytest.mark.parametrize(
    "category", sorted(c for c, f in SLOT_FAMILY.items() if f)
)
def test_every_offered_quantity_belongs_to_the_slots_family(
    qapp, multi_family, category,
):
    """The converse, and the sharper half. The Line row's tokens came
    from a hardcoded tuple filtered against the WRONG families; a row
    offering something its family does not record is the same defect
    facing the other way."""
    from apeGmsh.viewers.session._realize import (
        recorded_components, recorded_line_components,
    )

    session = multi_family.session()
    view = session.panes[0]
    nodal, gauss = recorded_components(session, view)
    by_family = {
        "nodes": nodal,
        "gauss": gauss,
        "line_stations": recorded_line_components(session, view),
    }
    allowed: set = set()
    for family in SLOT_FAMILY[category]:
        allowed |= by_family[family]

    _page, rows = _rows(qapp, multi_family)
    offered = set(_tokens(rows[category]))
    assert offered <= allowed, (
        f"the {category!r} row offers {sorted(offered - allowed)}, "
        f"which the {SLOT_FAMILY[category]} family does not record."
    )


def test_reactions_needs_no_quantity_and_is_addable(qapp, multi_family):
    """The one slot with nothing to pick: it is add-or-not, so the law
    reads as 'can_add', not 'offers a token'."""
    _page, rows = _rows(qapp, multi_family)
    assert _add_enabled(rows["reactions"])


def _add_enabled(row) -> bool:
    """Whether the row's Add button is live.

    Asserted instead of the ``can_add`` ctor argument on purpose: the
    button IS the affordance, and a row that computes ``can_add=True``
    but leaves the button disabled is the same unreachable slot.
    """
    return bool(row._button.isEnabled())


def _tokens(row) -> "list[str]":
    """The quantities a row's picker offers.

    ``_SlotRow`` already carries ``can_add=bool(tokens)``, which is the
    signal the Add button uses — so that is what the law asserts. This
    reads the combo itself only where the test needs the token VALUES.
    """
    from qtpy import QtWidgets

    editor = getattr(row, "_editor", None)
    if editor is None:
        return []
    for combo in editor.findChildren(QtWidgets.QComboBox):
        return [combo.itemText(i) for i in range(combo.count())]
    if isinstance(editor, QtWidgets.QComboBox):
        return [editor.itemText(i) for i in range(editor.count())]
    return []


# ----------------------------------------------------------------------
# A6.6 / G4 — the vocabulary is what THIS PANE's cells record
# ----------------------------------------------------------------------

@pytest.fixture
def split_families(g, tmp_path: Path):
    """Two groups, and only ONE of them records Gauss stress.

    The bench's shape in miniature: solids carry Gauss stress, shells
    carry none, and a pane scoped to the shells must not be offered
    `stress_zz`.
    """
    g.model.geometry.add_box(0, 0, 0, 1, 1, 1, label="recorded")
    g.model.geometry.add_box(1, 0, 0, 1, 1, 1, label="bare")
    g.physical.add_volume("recorded", name="Recorded")
    g.physical.add_volume("bare", name="Bare")
    g.mesh.sizing.set_global_size(2.0)
    g.mesh.generation.generate(dim=3)
    fem = g.mesh.queries.get_fem_data(dim=3)

    node_ids = np.asarray(fem.nodes.ids, dtype=np.int64)
    disp = np.zeros((2, node_ids.size))
    recorded_ids = np.asarray(
        fem.elements.select(pg="Recorded").ids, dtype=np.int64,
    )
    stress = np.zeros((2, recorded_ids.size, 2))
    stress[1, :, 0] = recorded_ids * 1.0

    path = tmp_path / "split_families.h5"
    with NativeWriter(path) as w:
        w.open(fem=fem)
        sid = w.begin_stage(
            name=STAGE, kind="static", stage_id=STAGE,
            time=np.array([0.0, 1.0]),
        )
        w.write_nodes(
            sid, "partition_0", node_ids=node_ids,
            components={"displacement_z": disp},
        )
        w.write_gauss_group(
            sid, "partition_0", "group_0",
            class_tag=4, int_rule=1,
            element_index=recorded_ids,
            natural_coords=np.array([[-0.5, 0.0, 0.0], [0.5, 0.0, 0.0]]),
            components={"stress_zz": stress},
        )
        w.end_stage()
    return Results.from_native(path, model=_open_model_from_h5(path)), (
        recorded_ids
    )


def _gauss_vocab(results, groups):
    from apeGmsh.results.session import Scope
    from apeGmsh.viewers.session._realize import recorded_components

    session = results.session()
    view = session.panes[0]
    view.scope = None if groups is None else Scope("physical_groups", groups)
    _nodal, gauss = recorded_components(session, view)
    return gauss


def test_the_whole_stage_is_offered_when_unscoped(split_families):
    results, _ids = split_families
    assert "stress_zz" in _gauss_vocab(results, None)


def test_a_scope_that_records_the_component_still_offers_it(split_families):
    results, _ids = split_families
    assert "stress_zz" in _gauss_vocab(results, ("Recorded",))


def test_a_scope_that_records_nothing_is_not_offered_it(split_families):
    """THE G4 regression. Before this, the picker answered with the
    whole stage's set regardless of scope, so the inspector offered a
    component the pane could not paint and the NoDataError arrived only
    after the user picked it."""
    assert "stress_zz" not in _gauss_vocab(split_families[0], ("Bare",))


def test_a_scope_spanning_both_offers_it(split_families):
    """A partial intersection must still offer — the pane paints the
    half that has data, which is what the contour slot does."""
    results, _ids = split_families
    assert "stress_zz" in _gauss_vocab(results, ("Recorded", "Bare"))


def test_the_row_is_disabled_when_the_scope_can_paint_nothing(
    qapp, split_families,
):
    """§4's law, one level deeper: inapplicable is disabled, with the
    reason already on the button's tooltip."""
    from apeGmsh.results.session import Scope
    from apeGmsh.viewers.session._inspector import MeshInspectorPage

    results, _ids = split_families
    session = results.session()
    view = session.panes[0]
    view.scope = Scope("physical_groups", ("Bare",))
    page = MeshInspectorPage(session, view)
    rows = {row.category: row for row in page._build_slot_rows()}
    assert not _add_enabled(rows["gauss"])
    assert rows["gauss"]._button.toolTip()
