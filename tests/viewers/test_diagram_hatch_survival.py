"""ADR 0098 Amendment 2 — the six slotless kinds survive behind `show_web`.

The amendment does not merely say those kinds are "not deleted". It rests
their survival on a specific mechanism: the kind registry auto-populates on
any import under ``apeGmsh.viewers.diagrams``
(``diagrams/_kinds.py:21-27``), and the ``show_web`` hatch keeps a live
handle — ``WebViewer.director`` → ``registry.add``. That is the promise a
future reader will rely on, and until this file existed **nothing tested
it**: there is no ``test_web_viewer.py``, and the S6b sweep retired the
director/registry suites that used to cover the mechanism incidentally.

So this is the load-bearing subset those suites were standing in for, kept
deliberately rather than by accident. It is small on purpose — S6b
de-publishes the old ontology and does not owe it broad coverage — but it
pins the three things Amendment 2 actually claims:

1. all six kinds are still REGISTERED after a bare package import;
2. one of them still CONSTRUCTS and ATTACHES through the real registry,
   which is what "internal survival" means operationally;
3. none of them acquired a §4 slot by accident — the catalog stays closed
   at seven, which is the other half of the amendment.

If the web-client ADR retires the hatch, Amendment 2 §A2.5 says all six need
a fresh disposition — and this file is what will go red to say so.

Default lane: no Qt, no GL, no trame. A ``RecordingBackend`` and the plain
registry, so it runs everywhere CI runs.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from apeGmsh.results import Results
from apeGmsh.results.writers import NativeWriter

from tests.conftest import _open_model_from_h5

#: Amendment 2 §A2.1, verbatim. Not derived from the registry — a list
#: built by asking the registry what it holds would pass against a
#: registry that had lost every one of them.
SLOTLESS_KINDS = (
    "fiber_section",
    "layer_stack",
    "isochrone_map",
    "isochrone_profile",
    "isochrone_strobe",
    "spring_force",
)


@pytest.fixture
def spring_results(g, tmp_path: Path):
    """A minimal bound Results — enough to construct a diagram against.

    Deliberately has no spring elements: this file pins that the kind
    still REGISTERS and CONSTRUCTS through the hatch, not that it draws.
    What it draws is ``test_spring_force.py``'s job, and that file
    survives the sweep for exactly this reason.
    """
    g.model.geometry.add_box(0, 0, 0, 1, 1, 1, label="cube")
    g.physical.add_volume("cube", name="Body")
    g.mesh.sizing.set_global_size(2.0)
    g.mesh.generation.generate(dim=3)
    fem = g.mesh.queries.get_fem_data(dim=3)
    node_ids = np.asarray(fem.nodes.ids, dtype=np.int64)

    path = tmp_path / "hatch.h5"
    with NativeWriter(path) as w:
        w.open(fem=fem)
        sid = w.begin_stage(
            name="grav", kind="static", stage_id="grav",
            time=np.array([0.0, 1.0]),
        )
        w.write_nodes(
            sid, "partition_0", node_ids=node_ids,
            components={"displacement_z": np.zeros((2, node_ids.size))},
        )
        w.end_stage()
    return Results.from_native(path, model=_open_model_from_h5(path))


# =====================================================================
# 1. The registry still holds all six
# =====================================================================

def test_a_bare_package_import_registers_all_six_slotless_kinds():
    """The auto-population mechanism Amendment 2 §A2.2 cites.

    Imports the PACKAGE, not the six modules — importing
    ``_fiber_section`` directly would register the kind as a side effect
    and prove nothing about the mechanism the amendment relies on.
    """
    import apeGmsh.viewers.diagrams  # noqa: F401
    from apeGmsh.viewers.diagrams._kinds import kind_ids

    registered = set(kind_ids())
    missing = [k for k in SLOTLESS_KINDS if k not in registered]

    assert not missing, (
        f"ADR 0098 Amendment 2 keeps these kinds alive behind the "
        f"show_web hatch, but the registry has lost: {missing}. If they "
        f"were retired deliberately, that is a NEW disposition (A2.5) "
        f"and this file should have gone with them."
    )


def test_each_slotless_kind_still_resolves_to_a_class_and_style():
    """A kind id in the registry with no class behind it is a name, not
    a survivor."""
    from apeGmsh.viewers.diagrams._kinds import kind_def

    for kind in SLOTLESS_KINDS:
        entry = kind_def(kind)
        assert entry is not None, kind
        assert entry.diagram_class is not None, kind
        assert entry.style_class is not None, kind


# =====================================================================
# 2. One of them still constructs and attaches THROUGH the registry
# =====================================================================

def test_a_slotless_kind_attaches_through_the_real_registry(
    spring_results, backend,
):
    """The operational half of "internal survival".

    Goes through ``DiagramRegistry.add`` — the exact call
    ``WebViewer.director.registry.add`` makes — rather than attaching the
    diagram directly, because the registry is the part of the hatch this
    amendment names.
    """
    from apeGmsh.viewers.diagrams import (
        DiagramSpec,
        SlabSelector,
        SpringForceDiagram,
        SpringForceStyle,
    )
    from apeGmsh.viewers.diagrams._registry import DiagramRegistry

    spec = DiagramSpec(
        kind="spring_force",
        selector=SlabSelector(component="spring_force_0"),
        style=SpringForceStyle(scale=1.0),
    )
    diagram = SpringForceDiagram(spec, spring_results)

    registry = DiagramRegistry()
    added = registry.add(diagram)

    assert added is diagram
    assert diagram in registry.diagrams()


# =====================================================================
# 3. The catalog stayed closed — the other half of the amendment
# =====================================================================

def test_no_slotless_kind_acquired_a_slot():
    """§4 stays closed at seven. A kind that quietly grew a slot would
    make the amendment's "catalog stays closed" clause false, and it is
    the sort of thing a widening PR does by accident."""
    from apeGmsh.results.session._slots import SLOT_CATALOG

    assert len(SLOT_CATALOG) == 7, SLOT_CATALOG
    overlap = sorted(set(SLOT_CATALOG) & set(SLOTLESS_KINDS))

    assert not overlap, (
        f"{overlap} appears in BOTH the §4 slot catalog and Amendment "
        f"2's slotless list. One of the two is now wrong — a slot for "
        f"any of the six is a §4 widening at the same review bar as a "
        f"new slot (A2.5)."
    )


def test_the_snapshot_refuses_a_slotless_kind_as_a_slot_category():
    """The consequence A2.3 states: a hatch picture cannot enter a
    session. Uses a NEAR-MISS — a well-formed snapshot with the right
    marker and one wrong slot category — because a payload with no
    ``kind`` at all is refused by the marker check and would prove
    nothing about the closed catalog.
    """
    from apeGmsh.results.session import (
        Contour,
        ResultsSession,
        SnapshotError,
        restore_snapshot,
        snapshot,
    )

    session = ResultsSession()
    session.add_view().contour = Contour("displacement_z")
    payload = snapshot(session)
    assert payload["kind"] == "apegmsh.results.session"

    slots = payload["panes"][0]["slots"]
    slots["fiber_section"] = slots.pop("contour")

    with pytest.raises(SnapshotError, match="fiber_section"):
        restore_snapshot(payload)
