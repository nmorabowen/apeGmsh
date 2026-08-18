"""ADR 0098 S5a — the session snapshot, at the IR layer (no Qt, no GL).

Oracles straight from the ADR and the two decisions this slice settled
(plan decisions 14/15):

* every field of the frozen v1 schema survives a round trip — asserted
  BOTH ways, because the two directions catch different mutations: a
  ``snapshot(original) == snapshot(restored)`` comparison is blind to a
  field the WRITER never emits (both sides agree on nothing), so the
  live IR of the restored session is also compared attribute by
  attribute against the session that was saved;
* an unknown slot category refuses LOUDLY — the §4 catalog is closed
  (amended ADR 0094 INV-10) and this refusal is that enforcement point;
* the §8 selection SET is state (decision 14) and restores as ONE
  ``SET`` gesture, not a forged op history;
* an UNLINKED snapshot carries every pane's own instant (decision 15) —
  the discriminating case, since a linked session ignores pane time and
  would pass either way;
* both id counters are re-seeded, not one (S0's runway note);
* a data mismatch (a dead stage id) DEGRADES with a notice, while a
  schema violation refuses — the two families stay distinguishable.
"""
from __future__ import annotations

import json
import time
from types import SimpleNamespace

import pytest

from apeGmsh.results.session import (
    SNAPSHOT_KIND,
    SNAPSHOT_VERSION,
    Contour,
    Deform,
    Gauss,
    Instant,
    Line,
    Loads,
    MeshStyle,
    MeshView,
    PlotSeries,
    PlotSource,
    Reactions,
    ResultsSession,
    Sand,
    Scope,
    SnapshotError,
    Vector,
    default_snapshot_path,
    restore_snapshot,
    save_snapshot,
    snapshot,
)
from apeGmsh.viewers.core.selection_log import OpKind

STAGE = "grav"


# =====================================================================
# Fixtures — one session that occupies every field of the schema
# =====================================================================

def _loaded_session() -> ResultsSession:
    """A session with every slot filled, every knob off its default.

    Deliberately maximal: a round-trip test is only as strong as the
    number of fields it puts in the file.
    """
    session = ResultsSession()

    view = session.add_view("Web")
    view.scope = Scope("materials", ("concrete", "steel"))
    view.deform = Deform("velocity", scale=12.5)
    view.time = Instant(STAGE, 2)
    view.style = MeshStyle(mesh=False, outlines=True, nodes=True, gauss=True)
    view.overlay = True
    view.pick_target = "gauss"
    view.contour = Contour("stress_xx", "unaveraged")
    view.vector = Vector("principal_stress")
    view.gauss = Gauss("stress_xx")
    view.line = Line("M")
    view.sand = Sand("strain_yy")
    view.loads = Loads("pattern_1")
    view.reactions = Reactions()
    # Two fields of legends here: stress_xx (contour + gauss, one
    # scale) and the vector's and sand's own. Hide one — chrome that is
    # state, unlike the legend itself.
    view.set_legend_hidden("stress_xx", True)
    view.add_clip((1.0, 0.0, 0.0), offset=0.25, name="Cut A")
    view.add_clip(
        (0.0, 1.0, 0.0), offset=-3.0, name="Cut B",
        active=False, flipped=True, gizmo_visible=False,
    )

    second = session.add_view()  # name=None is a distinct state
    second.deform = Deform("displacement", mode=3)  # a mode pose
    second.time = Instant(STAGE, 1)

    plot = session.add_plot(
        "history",
        series=(
            PlotSeries(source=PlotSource.node(7), quantity="displacement_z"),
            PlotSeries(
                source=PlotSource.gauss(4, 2), quantity="stress_xx",
            ),
            PlotSeries(source=PlotSource.label("Web"), quantity="strain_yy"),
            PlotSeries(
                source=PlotSource.physical_group("Body"),
                quantity="displacement_x",
            ),
        ),
        name="Roof drift",
    )
    plot.cursor = Instant(STAGE, 3)

    session.time_linked = False
    session.time = Instant(STAGE, 0)
    session.selection.set_gauss([(4, 0), (4, 1), (9, 2)])
    return session


def _live_state(session: ResultsSession) -> dict:
    """The session's state read off the LIVE IR, not off its snapshot.

    The point of reading it independently: if the writer drops a field,
    ``snapshot(a) == snapshot(b)`` still holds, and only a comparison
    that never goes through the writer can see it.
    """
    panes = []
    for pane in session.panes:
        if isinstance(pane, MeshView):
            panes.append((
                "mesh", pane.id, pane.name, pane.scope, pane.deform,
                pane.time, pane.style, pane.overlay, pane.pick_target,
                pane.slots, pane.legends(), pane.clips,
            ))
        else:
            panes.append((
                "plot", pane.id, pane.name, pane.kind, pane.series,
                pane.cursor,
            ))
    return {
        "time": session.time,
        "time_linked": session.time_linked,
        "selection_kind": session.selection.kind,
        "nodes": session.selection.nodes,
        "gauss": session.selection.gauss,
        "panes": panes,
    }


def _without_timestamp(payload: dict) -> dict:
    return {k: v for k, v in payload.items() if k != "saved_at"}


class _StubResults:
    """The two things ``_validate_instants`` asks a broker.

    A double for the collaborator, not for the thing under test — the
    real-``Results`` path is proven end to end in
    ``tests/viewers/test_session_snapshot_realize.py``.
    """

    def __init__(self, stages: dict[str, int]) -> None:
        self._stages = dict(stages)

    @property
    def stages(self) -> list:
        return [SimpleNamespace(id=sid) for sid in self._stages]

    def stage(self, name_or_id: str):
        if name_or_id not in self._stages:
            raise KeyError(
                f"No stage matches {name_or_id!r}. Available: "
                f"{sorted(self._stages)}"
            )
        return SimpleNamespace(n_steps=self._stages[name_or_id])


# =====================================================================
# Round trip
# =====================================================================

def test_round_trip_preserves_the_live_ir():
    """Restored state == saved state, read off the IR both times."""
    original = _loaded_session()
    before = _live_state(original)

    restored = restore_snapshot(snapshot(original)).session

    assert restored is not original
    assert _live_state(restored) == before


def test_round_trip_is_stable_through_the_writer():
    """And re-writing the restored session reproduces the same file.

    The complement of the test above: that one catches restore losing a
    field, this one catches restore mangling one the writer emits.
    """
    original = _loaded_session()
    payload = snapshot(original)
    again = snapshot(restore_snapshot(payload).session)

    assert _without_timestamp(again) == _without_timestamp(payload)


def test_snapshot_is_json_and_carries_the_marker_not_a_schema_version():
    payload = snapshot(_loaded_session())
    text = json.dumps(payload)  # refuses a non-JSON-safe value

    assert payload["kind"] == SNAPSHOT_KIND
    assert payload["version"] == SNAPSHOT_VERSION
    # The old viewer session is discriminated by KEYS, never by
    # comparing 1 against 13 — so this file must not carry that key.
    assert "schema_version" not in payload
    assert "viewer-session" not in text


def test_derived_legends_are_not_stored_but_hidden_chrome_is():
    """§5: ``legends = f(occupied colour-mapped slots)``.

    A stored legend list would be a second source of truth for
    something the slots already say; the per-field *hidden* flag is the
    only part a human set.
    """
    payload = snapshot(_loaded_session())
    view_payload = payload["panes"][0]

    assert "legends" not in view_payload
    assert view_payload["legend_hidden"] == ["stress_xx"]

    restored = restore_snapshot(payload).session.panes[0]
    hidden = {lg.field: lg.hidden for lg in restored.legends()}
    assert hidden["stress_xx"] is True
    # ... and the other scales came back visible, so the flag is a
    # per-field flag and not a global one.
    assert set(hidden) == {"stress_xx", "principal_stress", "strain_yy"}
    assert hidden["principal_stress"] is False
    assert hidden["strain_yy"] is False


def test_a_hidden_flag_for_a_legend_that_does_not_exist_degrades():
    payload = snapshot(_loaded_session())
    payload["panes"][0]["legend_hidden"] = ["stress_xx", "not_a_field"]

    restored = restore_snapshot(payload)

    assert any("not_a_field" in n for n in restored.notices)
    fields = {lg.field for lg in restored.session.panes[0].legends()}
    assert "not_a_field" not in fields


# =====================================================================
# §8 selection — decision 14
# =====================================================================

def test_selection_set_is_snapshot_state():
    session = ResultsSession()
    session.add_view()
    session.selection.set_nodes([11, 4, 7])

    restored = restore_snapshot(snapshot(session)).session

    assert restored.selection.kind == "nodes"
    assert restored.selection.nodes == (11, 4, 7)


def test_gauss_selection_round_trips_as_pairs():
    session = ResultsSession()
    session.add_view()
    session.selection.set_gauss([(4, 0), (9, 2)])

    restored = restore_snapshot(snapshot(session)).session

    assert restored.selection.kind == "gauss"
    assert restored.selection.gauss == ((4, 0), (9, 2))


def test_an_empty_selection_round_trips_as_empty():
    session = ResultsSession()
    session.add_view()

    restored = restore_snapshot(snapshot(session)).session

    assert restored.selection.kind is None
    assert len(restored.selection) == 0


def test_restore_writes_the_set_as_one_set_gesture():
    """The SET is state; the op LOG is not (decision 14).

    A restored session's log honestly says "one write, from a file"
    instead of forging the gestures that produced it.
    """
    session = ResultsSession()
    session.add_view()
    session.selection.set_nodes([1, 2])
    session.selection.add_nodes([3])
    session.selection.toggle_node(2)  # three gestures, one final set

    restored = restore_snapshot(snapshot(session)).session

    ops = restored.selection.state._log.active_ops
    assert [op.kind for op in ops] == [OpKind.SET]
    assert restored.selection.nodes == session.selection.nodes


def test_a_foreign_selection_kind_refuses():
    payload = snapshot(ResultsSession())
    payload["selection"] = {"kind": "elements", "nodes": [1]}

    with pytest.raises(SnapshotError, match="nodes XOR Gauss"):
        restore_snapshot(payload)


# =====================================================================
# §7 time — decision 15, the unlinked case
# =====================================================================

def _unlinked_session() -> ResultsSession:
    session = ResultsSession()
    mesh = session.add_view()
    plot = session.add_plot("history")
    session.time_linked = False
    session.time = Instant(STAGE, 0)
    mesh.time = Instant(STAGE, 1)
    plot.cursor = Instant(STAGE, 2)
    return session


def test_an_unlinked_snapshot_carries_every_pane_instant():
    """Restoring must not silently relink every pane to one instant.

    Three DIFFERENT instants, so the assertion can only pass if each
    pane's own instant made the trip: drop ``time_linked`` and every
    pane reads the session's step 0; drop the pane fields and every
    pane reads ``None``.
    """
    restored = restore_snapshot(snapshot(_unlinked_session())).session
    mesh, plot = restored.panes

    assert restored.time_linked is False
    assert restored.effective_instant(mesh) == Instant(STAGE, 1)
    assert restored.effective_instant(plot) == Instant(STAGE, 2)
    assert restored.time == Instant(STAGE, 0)


def test_a_linked_snapshot_still_reads_one_instant_everywhere():
    """The complement — proves the test above is not asserting a
    coincidence of defaults: with the link ON, the same pane fields
    survive the trip and are correctly IGNORED (§7/§9)."""
    session = _unlinked_session()
    session.time_linked = True

    restored = restore_snapshot(snapshot(session)).session
    mesh, plot = restored.panes

    assert restored.time_linked is True
    assert restored.effective_instant(mesh) == Instant(STAGE, 0)
    assert restored.effective_instant(plot) == Instant(STAGE, 0)
    # Stored, not lost: unlink and each pane is back where it was.
    assert mesh.time == Instant(STAGE, 1)
    assert plot.cursor == Instant(STAGE, 2)


def test_a_mode_posed_pane_has_no_instant_after_restore():
    session = ResultsSession()
    view = session.add_view()
    view.deform = Deform("displacement", mode=2)
    session.time = Instant(STAGE, 1)

    restored = restore_snapshot(snapshot(session)).session

    assert restored.panes[0].deform == Deform("displacement", mode=2)
    assert restored.effective_instant(restored.panes[0]) is None


# =====================================================================
# The two id counters — S0's runway note
# =====================================================================

def test_restoring_reseeds_the_pane_counter():
    session = ResultsSession()
    session.add_view()          # mesh-1
    session.add_plot()          # plot-2
    session.add_view()          # mesh-3
    payload = snapshot(session)

    restored = restore_snapshot(payload).session

    assert [p.id for p in restored.panes] == ["mesh-1", "plot-2", "mesh-3"]
    assert restored.add_view().id == "mesh-4"
    assert restored.add_plot().id == "plot-5"
    assert len({p.id for p in restored.panes}) == len(restored.panes)


def test_restoring_reseeds_each_views_own_clip_counter():
    """The SECOND counter. A pane counter re-seeded alone still hands
    out a duplicate ``clip-1`` on the first plane added after a
    restore."""
    session = ResultsSession()
    first = session.add_view()
    first.add_clip((1.0, 0.0, 0.0))   # clip-1
    first.add_clip((0.0, 1.0, 0.0))   # clip-2
    second = session.add_view()
    second.add_clip((0.0, 0.0, 1.0))  # clip-1 of ITS own sequence

    restored = restore_snapshot(snapshot(session)).session
    r_first, r_second = restored.panes

    assert [c.plane_id for c in r_first.clips] == ["clip-1", "clip-2"]
    assert r_first.add_clip((1.0, 1.0, 0.0)).plane_id == "clip-3"
    # Per view, not global: the second view's own counter is its own.
    assert [c.plane_id for c in r_second.clips] == ["clip-1"]
    assert r_second.add_clip((1.0, 0.0, 1.0)).plane_id == "clip-2"


def test_restored_clips_keep_their_identity_and_fields():
    session = ResultsSession()
    view = session.add_view()
    view.add_clip((0.0, 2.0, 0.0), offset=-3.0, name="Cut B",
                  active=False, flipped=True, gizmo_visible=False)

    clip, = restore_snapshot(snapshot(session)).session.panes[0].clips

    assert clip.plane_id == "clip-1"
    assert clip.name == "Cut B"
    assert clip.normal == (0.0, 1.0, 0.0)  # unit-normalised at add time
    assert (clip.offset, clip.active, clip.flipped, clip.gizmo_visible) == (
        -3.0, False, True, False,
    )


def test_ids_that_do_not_fit_the_pattern_cannot_break_the_counter():
    payload = snapshot(ResultsSession())
    payload["panes"] = [
        {"pane": "mesh", "id": "imported-view"},
        {"pane": "mesh", "id": "mesh-2"},
    ]

    restored = restore_snapshot(payload).session

    assert restored.add_view().id == "mesh-3"


# =====================================================================
# Schema violations refuse LOUDLY — the closed §4 catalog
# =====================================================================

def test_an_unknown_slot_category_refuses_loudly():
    """The amended-0094-INV-10 enforcement point.

    A category this build does not have means an ontology this build
    does not share. Dropping it silently would draw a picture nobody
    authorised and call it the human's session.
    """
    payload = snapshot(ResultsSession())
    payload["panes"] = [{
        "pane": "mesh",
        "id": "mesh-1",
        "slots": {"fiber_section": {"quantity": "stress"}},
    }]

    with pytest.raises(SnapshotError) as excinfo:
        restore_snapshot(payload)

    message = str(excinfo.value)
    assert "fiber_section" in message
    assert "CLOSED" in message
    assert "0094" in message and "INV-10" in message


@pytest.mark.parametrize("category,payload", [
    ("contour", {"quantity": "s", "averaging": "averaged"}),
    ("vector", {"quantity": "s"}),
    ("gauss", {"quantity": "s"}),
    ("line", {"component": "M"}),
    ("sand", {"quantity": "s"}),
    ("loads", {"pattern": "p1"}),
    ("reactions", {}),
])
def test_every_catalog_category_restores(category, payload):
    """The complement of the refusal: all seven really do come back, so
    the test above is a closed door and not a wall."""
    data = snapshot(ResultsSession())
    data["panes"] = [
        {"pane": "mesh", "id": "mesh-1", "slots": {category: payload}},
    ]

    view = restore_snapshot(data).session.panes[0]

    assert list(view.slots) == [category]


def test_a_slot_payload_that_does_not_fit_its_record_refuses():
    data = snapshot(ResultsSession())
    data["panes"] = [{
        "pane": "mesh", "id": "mesh-1",
        "slots": {"contour": {"quantity": "s", "smoothing": "lots"}},
    }]

    with pytest.raises(SnapshotError, match="does not fit Contour"):
        restore_snapshot(data)


@pytest.mark.parametrize("mutate,match", [
    (lambda d: d.update(kind="something.else"), "Not an apeGmsh session"),
    (lambda d: d.update(version=99), "schema v99"),
    (lambda d: d.update(panes=[{"pane": "diagram", "id": "d-1"}]),
     "Unknown pane kind"),
    (lambda d: d.update(panes=[{"pane": "mesh", "id": ""}]),
     "non-empty string 'id'"),
    (lambda d: d.update(panes=[{
        "pane": "mesh", "id": "mesh-1",
        "scope": {"axis": "colours", "names": None}}]),
     "Scope.axis"),
    (lambda d: d.update(panes=[{
        "pane": "mesh", "id": "mesh-1",
        "deform": {"field": "temperature"}}]),
     "Deform.field"),
    (lambda d: d.update(panes=[{
        "pane": "mesh", "id": "mesh-1", "pick_target": "elements"}]),
     "pick_target"),
    (lambda d: d.update(panes=[{
        "pane": "plot", "id": "plot-1", "kind": "violin"}]),
     "PlotView.kind"),
    (lambda d: d.update(time={"stage": STAGE, "step": -1}), "step must be"),
    (lambda d: d.update(time={"stage": STAGE}), "missing 'step'"),
])
def test_schema_violations_refuse(mutate, match):
    """Restore builds the REAL frozen records, so S0's laws are the
    schema's laws — there is no second, weaker copy of them here."""
    payload = snapshot(ResultsSession())
    mutate(payload)

    with pytest.raises((SnapshotError, ValueError, TypeError), match=match):
        restore_snapshot(payload)


def test_a_non_dict_is_not_a_snapshot():
    with pytest.raises(SnapshotError, match="must be a JSON object"):
        restore_snapshot([1, 2, 3])


# =====================================================================
# Data mismatches DEGRADE — decision 15's other half
# =====================================================================

def test_a_dead_stage_id_degrades_with_a_notice():
    """A stage rename must not cost the human every pane in the file."""
    session = ResultsSession()
    view = session.add_view()
    view.contour = Contour("stress_xx")
    session.time = Instant("stage_that_was_renamed", 1)

    restored = restore_snapshot(
        snapshot(session), results=_StubResults({STAGE: 4}),
    )

    assert restored.session.time is None
    assert len(restored.notices) == 1
    notice, = restored.notices
    assert "stage_that_was_renamed" in notice
    assert STAGE in notice  # names what IS available
    # The picture survived the stage rename.
    assert restored.session.panes[0].contour == Contour("stress_xx")


def test_a_step_past_the_end_of_a_shortened_run_degrades():
    session = ResultsSession()
    session.add_view()
    session.time = Instant(STAGE, 9)

    restored = restore_snapshot(
        snapshot(session), results=_StubResults({STAGE: 4}),
    )

    assert restored.session.time is None
    assert "past stage" in restored.notices[0]
    assert "4 recorded step" in restored.notices[0]


def test_per_pane_instants_are_validated_too():
    session = _unlinked_session()
    session.panes[0].time = Instant("gone", 0)

    restored = restore_snapshot(
        snapshot(session), results=_StubResults({STAGE: 4}),
    )
    mesh, plot = restored.session.panes

    assert mesh.time is None
    assert plot.cursor == Instant(STAGE, 2)   # this one was fine
    assert len(restored.notices) == 1
    assert "mesh-1" in restored.notices[0]


def test_a_live_instant_survives_validation_untouched():
    session = ResultsSession()
    session.add_view()
    session.time = Instant(STAGE, 3)

    restored = restore_snapshot(
        snapshot(session), results=_StubResults({STAGE: 4}),
    )

    assert restored.session.time == Instant(STAGE, 3)
    assert restored.notices == ()


def test_an_ir_only_restore_keeps_instants_verbatim():
    """``results=None``: nothing to validate against, so nothing is
    dropped — the IR-only restore an inspector or a test wants."""
    session = ResultsSession()
    session.add_view()
    session.time = Instant("any_stage_at_all", 41)

    restored = restore_snapshot(snapshot(session))

    assert restored.session.results is None
    assert restored.session.time == Instant("any_stage_at_all", 41)
    assert restored.notices == ()


# =====================================================================
# Scale — the one O(n) term in the schema
# =====================================================================

def test_a_large_selection_round_trips_without_going_quadratic():
    """A catastrophe detector, not a performance assertion.

    Panes are O(10) and slots O(7), so the ONLY term that grows with the
    model is the §8 selection set. 16k targets cost 26.7 s before S4-2
    made the store set-backed, and ~40 ms after it; the bound below is
    two orders looser than the measured cost and two orders tighter than
    the quadratic it replaced, so a reintroduced O(n^2) fails here while
    CPU contention on a shared box does not.

    Measured on this branch: 1k / 16k / 100k selected nodes round-trip in
    1.5 / 36 / 163 ms (1.3 MB of JSON at 100k) — linear.
    """
    n = 16_000
    session = ResultsSession()
    session.add_view()
    session.selection.set_nodes(range(1, n + 1))

    started = time.perf_counter()
    restored = restore_snapshot(snapshot(session))
    elapsed = time.perf_counter() - started

    assert restored.session.selection.nodes == tuple(range(1, n + 1))
    assert elapsed < 5.0, f"{n} targets took {elapsed:.1f} s"


# =====================================================================
# Disk surface
# =====================================================================

def test_default_snapshot_path_is_not_the_old_viewers_path():
    path = default_snapshot_path("/tmp/run/out.h5")

    assert path.name == "out.h5.session.json"
    # The old window still owns (and overwrites on close) this one.
    assert "viewer-session" not in path.name


def test_save_snapshot_writes_json_that_reloads(tmp_path):
    session = _loaded_session()
    target = tmp_path / "sub" / "out.h5.session.json"

    written = save_snapshot(session, target)

    assert written == target
    assert target.read_text(encoding="utf-8").endswith("\n")
    payload = json.loads(target.read_text(encoding="utf-8"))
    assert _live_state(restore_snapshot(payload).session) == _live_state(
        session,
    )


def test_save_snapshot_needs_a_path_when_results_came_from_memory():
    session = ResultsSession()

    with pytest.raises(ValueError, match="pass an explicit path"):
        save_snapshot(session)


def test_session_methods_delegate_to_the_module(tmp_path):
    """``session.snapshot()`` / ``session.save_snapshot()`` are the
    surface a human and S5b/S5c use."""
    session = _loaded_session()
    target = tmp_path / "out.session.json"

    assert _without_timestamp(session.snapshot()) == _without_timestamp(
        snapshot(session),
    )
    assert session.save_snapshot(target) == target
    assert json.loads(target.read_text(encoding="utf-8"))["kind"] == (
        SNAPSHOT_KIND
    )


def test_a_restored_pane_is_a_real_member_of_its_session():
    """Not a detached copy: a mutation on a restored pane must tick the
    session, or the S2 reconciler never repaints what a restored window
    edits."""
    restored = restore_snapshot(snapshot(_loaded_session())).session
    ticks: list[int] = []
    restored.subscribe(lambda: ticks.append(1))

    restored.panes[0].overlay = False
    restored.panes[-1].name = "renamed"

    assert len(ticks) == 2
