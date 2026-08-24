"""ADR 0098 S2 — SessionReconciler: session diff → backend ops →
ONE coalesced render.

Primary oracle (no GL, runs in the default CI lane): RecordingBackend
layer/bar assertions after a driven flush —

* a slot fill repaints (contour layer + bar appear, substrate gone);
* a gesture of N session writes coalesces to ONE realize + ONE render;
* teardown is exact by ledger: no layer or bar accumulates across
  flushes, and a realize that fails mid-emission is swept;
* a flush failure reports ``pump.session_reconcile`` — the autouse
  ``pump_failures`` fixture turns a swallowed reconciler error into a
  test failure;
* after a successful flush ``controller_for(backend)`` answers the
  realization's REAL legend controller, not the null realize adopts
  (the S1 runway made checkable).

The repaint oracle is MUTATION-TESTED (mandatory for this slice): the
same assertion body that passes against the live reconciler must fail
against one whose flush is broken — a green smoke proves nothing
until a broken reconciler fails it.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from apeGmsh.results import Results
from apeGmsh.results.session import Contour, Deform
from apeGmsh.results.writers import NativeWriter
from apeGmsh.viewers.core._legend import controller_for
from apeGmsh.viewers.session import SessionReconciler
from apeGmsh.viewers.session import _reconciler as reconciler_mod
from apeGmsh.viewers.session._realize import MAX_ACTIVE_PLANES

from tests.conftest import _open_model_from_h5
from tests.viewers.conftest import RecordingBackend

STAGE = "grav"


# =====================================================================
# Fixtures
# =====================================================================


class CountingBackend(RecordingBackend):
    """RecordingBackend + a render counter — the one-coalesced-render
    oracle."""

    def __init__(self) -> None:
        super().__init__()
        self.render_count = 0

    def render(self) -> None:
        self.render_count += 1


@pytest.fixture
def session_results(g, tmp_path: Path):
    """Native results: one static stage, nodal displacement + 1-GP
    gauss stress (same shape as the S1 fixtures, mode stage omitted —
    the reconciler rides realize's instant law unchanged)."""
    g.model.geometry.add_box(0, 0, 0, 1, 1, 1, label="cube")
    g.physical.add_volume("cube", name="Body")
    g.mesh.sizing.set_global_size(2.0)
    g.mesh.generation.generate(dim=3)
    fem = g.mesh.queries.get_fem_data(dim=3)

    node_ids = np.asarray(fem.nodes.ids, dtype=np.int64)
    chunks = [np.asarray(gr.ids, dtype=np.int64) for gr in fem.elements]
    elem_ids = np.concatenate(chunks)
    n_steps = 3

    disp = np.zeros((n_steps, node_ids.size), dtype=np.float64)
    for t in range(n_steps):
        disp[t] = node_ids + t * 1000.0
    sxx = np.zeros((n_steps, elem_ids.size, 1), dtype=np.float64)
    for t in range(n_steps):
        sxx[t, :, 0] = elem_ids * 10.0 + t

    path = tmp_path / "session_s2.h5"
    with NativeWriter(path) as w:
        w.open(fem=fem)
        sid = w.begin_stage(
            name=STAGE, kind="static", stage_id=STAGE,
            time=np.arange(n_steps, dtype=np.float64),
        )
        w.write_nodes(
            sid, "partition_0",
            node_ids=node_ids,
            components={"displacement_z": disp},
        )
        w.write_gauss_group(
            sid, "partition_0", "group_0",
            class_tag=4, int_rule=1,
            element_index=elem_ids, natural_coords=np.zeros((1, 3)),
            components={"stress_xx": sxx},
        )
        w.end_stage()
    return Results.from_native(path, model=_open_model_from_h5(path))


@pytest.fixture
def rig(session_results):
    """(session, view, backend, reconciler, drain) with a hand-driven
    defer queue — the test owns exactly when a flush runs, so the
    coalescing shape is observable."""
    session = session_results.session()
    view = session.panes[0]
    backend = CountingBackend()
    queue: list = []
    rec = SessionReconciler(
        session, backend, defer_fn=queue.append,
    )

    def drain() -> None:
        while queue:
            queue.pop(0)()

    yield session, view, backend, rec, drain
    rec.dispose()


def _stable_keys(rec) -> set:
    """The last realization's STABLE layer keys — the S2 diff surface
    (raw backend layer_ids are an emission detail, S1 contract)."""
    realized = rec.realized
    return {layer.key for layer in realized.layers} if realized else set()


def _assert_backend_matches_realization(rec, backend) -> None:
    """Teardown exactness: the backend holds exactly the layers of the
    CURRENT realization — nothing stale, nothing orphaned."""
    realized = rec.realized
    wanted = (
        {layer.layer_id for layer in realized.layers}
        if realized else set()
    )
    assert set(backend.layers) == wanted


# =====================================================================
# The repaint oracle — shared body, mutation-tested below
# =====================================================================


def _boot_keys(view) -> set:
    """The §3 boot picture: grey mesh with mesh + outlines on."""
    return {f"{view.id}:substrate", f"{view.id}:outlines"}


def _assert_contour_projected(view, rec, backend) -> None:
    """After filling the contour slot and flushing: the contour has
    REPLACED the substrate (INV-MESH-2 — no grey under it), the
    outlines style survives as its own layer (INV-MESH-4), the backend
    holds exactly that, and one scalar bar exists (§5)."""
    assert _stable_keys(rec) == {
        f"{view.id}:contour", f"{view.id}:outlines",
    }
    _assert_backend_matches_realization(rec, backend)
    assert len(backend.scalar_bars) == 1


def test_slot_fill_repaints(rig):
    session, view, backend, rec, drain = rig
    drain()  # boot flush — the valid empty picture
    assert _stable_keys(rec) == _boot_keys(view)
    _assert_backend_matches_realization(rec, backend)
    assert backend.scalar_bars == {}

    view.contour = Contour("stress_xx")
    drain()
    _assert_contour_projected(view, rec, backend)


def test_repaint_oracle_bites_a_broken_reconciler(rig, monkeypatch):
    """MUTATION TEST (the plan makes it mandatory): the exact oracle
    body above must FAIL against a reconciler whose flush swallows the
    work — a green smoke proves nothing until a broken reconciler
    fails it."""
    session, view, backend, rec, drain = rig
    drain()
    monkeypatch.setattr(
        SessionReconciler, "_flush", lambda self: None,
    )
    view.contour = Contour("stress_xx")
    drain()
    with pytest.raises(AssertionError):
        _assert_contour_projected(view, rec, backend)


# =====================================================================
# Coalescing — one gesture, one realize, one render
# =====================================================================


def test_gesture_coalesces_to_one_flush(rig, monkeypatch):
    session, view, backend, rec, drain = rig
    drain()
    realizes: list = []
    real = reconciler_mod.realize_pane
    monkeypatch.setattr(
        reconciler_mod, "realize_pane",
        lambda *a, **k: realizes.append(1) or real(*a, **k),
    )
    renders_before = backend.render_count

    # One gesture, three session writes.
    view.deform = Deform("displacement")
    view.contour = Contour("stress_xx")
    view.contour = Contour("displacement_z")
    drain()

    assert len(realizes) == 1
    assert backend.render_count == renders_before + 1
    # And the picture is the LAST write's.
    _assert_contour_projected(view, rec, backend)


def test_tick_during_pending_flush_joins_it(rig):
    session, view, backend, rec, drain = rig
    drain()
    view.contour = Contour("stress_xx")
    view.gauss = view.gauss  # no-op write: equality-guarded, no tick
    view.deform = Deform("displacement")
    # Two ticks, ONE scheduled flush.
    drain()
    _assert_contour_projected(view, rec, backend)


# =====================================================================
# Teardown — exact by ledger, no accumulation, partial sweeps
# =====================================================================


def test_no_accumulation_across_flushes(rig):
    session, view, backend, rec, drain = rig
    drain()
    for quantity in ("stress_xx", "displacement_z", "stress_xx"):
        view.contour = Contour(quantity)
        drain()
    # Exactly ONE contour layer (plus nothing stale), one bar.
    _assert_contour_projected(view, rec, backend)


def test_clear_restores_substrate_and_drops_bar(rig):
    session, view, backend, rec, drain = rig
    view.contour = Contour("stress_xx")
    drain()
    view.contour = None
    drain()
    assert _stable_keys(rec) == _boot_keys(view)
    _assert_backend_matches_realization(rec, backend)
    assert backend.scalar_bars == {}


@pytest.mark.allow_pump_failures
def test_failed_realize_reports_and_sweeps_partial_emission(
    rig, monkeypatch, pump_failures,
):
    """A realize that raises MID-EMISSION: the failure is reported as
    ``pump.session_reconcile`` (never silent — ADR 0084 D4) and the
    partially emitted layer is swept by the ledger on the next flush
    instead of leaking an actor the reconciler has no handle for."""
    session, view, backend, rec, drain = rig
    drain()

    from types import SimpleNamespace

    def exploding_realize(sess, pane, backend_arg):
        # The RecordingBackend stores whatever carries a layer_id —
        # a stand-in record is enough to model a half-emitted layer.
        backend_arg.add_layer(SimpleNamespace(
            layer_id="partial:orphan", color=None,
        ))
        raise RuntimeError("boom mid-emission")

    monkeypatch.setattr(
        reconciler_mod, "realize_pane", exploding_realize,
    )
    view.contour = Contour("stress_xx")
    drain()

    assert [n for n, _e in pump_failures.failures] == [
        "pump.session_reconcile",
    ]
    assert "partial:orphan" in backend.layers  # emitted before the boom
    assert rec.realized is None

    # Next flush (realize restored) sweeps the orphan exactly.
    monkeypatch.undo()
    view.contour = None
    drain()
    assert "partial:orphan" not in backend.layers
    assert _stable_keys(rec) == _boot_keys(view)
    _assert_backend_matches_realization(rec, backend)


# =====================================================================
# Legend ownership (S1 runway → S2 contract)
# =====================================================================


def test_real_legend_controller_adopted_after_flush(rig):
    session, view, backend, rec, drain = rig
    view.contour = Contour("stress_xx")
    drain()
    realized = rec.realized
    assert realized is not None
    assert realized.legend_controller is not None
    assert controller_for(rec.backend) is realized.legend_controller


# =====================================================================
# Pane targeting
# =====================================================================


def test_default_projection_is_first_mesh_view(rig):
    session, view, backend, rec, drain = rig
    drain()
    assert rec.realized is not None
    assert rec.realized.pane_id == view.id


def test_set_pane_retargets_and_a_stale_bound_id_paints_nothing(rig):
    """ADR 0098 Amendment 1 caution 10.

    A reconciler BOUND to a pane id whose view is gone paints nothing.
    S2 fell back to the first mesh view — correct for one shared
    viewport ("the outline re-aims on its next selection") and wrong
    for a host: a pane closing on this very tick would repaint view 1
    into its dying backend, and a mis-wired pane would silently mirror
    pane 1 instead of failing pane isolation.
    """
    session, view, backend, rec, drain = rig
    second = session.add_view("Second")
    second.contour = Contour("stress_xx")
    rec.set_pane(second.id)
    drain()
    assert rec.realized.pane_id == second.id
    assert _stable_keys(rec) == {
        f"{second.id}:contour", f"{second.id}:outlines",
    }
    _assert_backend_matches_realization(rec, backend)

    session.remove_pane(second.id)
    drain()
    assert rec.realized is None
    assert backend.layers == {}
    assert backend.scalar_bars == {}


def test_empty_session_paints_nothing(session_results):
    from apeGmsh.results.session import ResultsSession

    session = ResultsSession(results=session_results)  # zero panes
    backend = CountingBackend()
    queue: list = []
    rec = SessionReconciler(session, backend, defer_fn=queue.append)
    while queue:
        queue.pop(0)()
    assert backend.layers == {}
    assert rec.realized is None
    rec.dispose()


# =====================================================================
# Lifecycle discipline
# =====================================================================


def test_flush_does_not_tick_the_session(rig):
    session, view, backend, rec, drain = rig
    ticks: list = []
    session.subscribe(lambda: ticks.append(1))
    view.contour = Contour("stress_xx")
    n_before_flush = len(ticks)
    drain()
    assert len(ticks) == n_before_flush  # a ticking flush would loop


def test_dispose_detaches_from_the_session(rig):
    session, view, backend, rec, drain = rig
    drain()
    rec.dispose()
    before = set(backend.layers)
    view.contour = Contour("stress_xx")
    drain()
    assert set(backend.layers) == before


# =====================================================================
# ADR 0098 A4 — the cursor fast path, and the parity that licenses it
# =====================================================================


def _snapshot(rec, backend) -> dict:
    """Every layer of the CURRENT realization, as comparable arrays,
    keyed by its STABLE key.

    Keyed by ``layer.key``, never the backend's ``layer_id``: a full
    realize builds a fresh diagram which mints a fresh id, and the S1
    contract already says the id is an emission detail while the key is
    the identity. Comparing ids would fail on every parity check for a
    reason that has nothing to do with the picture.

    Points AND field values: the fast path moves both (a pose moves
    points, ``update_to_step`` moves scalars), so a snapshot of one
    would license half the claim.
    """
    out: dict = {}
    realized = rec.realized
    for entry in (realized.layers if realized else ()):
        layer = backend.layers.get(entry.layer_id)
        if layer is None:
            continue
        pts = getattr(layer, "points", None)
        fields = {
            f.name: np.asarray(f.values).copy()
            for f in (getattr(layer, "fields", None) or ())
        }
        out[entry.key] = (
            None if pts is None else np.asarray(pts.coords).copy(),
            fields,
        )
    return out


def _assert_same(fast: dict, slow: dict, step: int) -> None:
    assert set(fast) == set(slow), (
        f"step {step}: the fast path emitted a different LAYER SET "
        f"({sorted(set(fast) ^ set(slow))})"
    )
    for layer_id, (fpts, ffields) in fast.items():
        spts, sfields = slow[layer_id]
        if fpts is None or spts is None:
            assert (fpts is None) == (spts is None), layer_id
        else:
            np.testing.assert_allclose(
                fpts, spts, atol=1e-9,
                err_msg=f"step {step}: {layer_id} points diverged",
            )
        assert set(ffields) == set(sfields), (
            f"step {step}: {layer_id} field set diverged"
        )
        for name, fvals in ffields.items():
            np.testing.assert_allclose(
                fvals, sfields[name], atol=1e-9,
                err_msg=f"step {step}: {layer_id}.{name} diverged",
            )


def _parity_body(rig) -> None:
    """Scrub with the fast path, then force a full realize at the SAME
    instant, and require the two pictures to be identical.

    This is A4.3.1, and it is the whole licence for the fast path: a
    re-step that drew anything other than what a realize draws is a
    stale picture, which is the failure class the one-shot contract
    existed to prevent.
    """
    from apeGmsh.results.session import Instant

    session, view, backend, rec, drain = rig
    view.contour = Contour("displacement_z")
    view.deform = Deform("displacement", 3.0)
    drain()

    for step in (1, 2, 0, 2):
        session.time = Instant(STAGE, step)
        drain()
        fast = _snapshot(rec, backend)

        rec.schedule_forced()          # same instant, the SLOW path
        drain()
        _assert_same(fast, _snapshot(rec, backend), step)


def test_a4_restep_paints_what_a_full_realize_paints(rig):
    """Criterion A4.3.1 — parity."""
    _parity_body(rig)


def test_a4_parity_oracle_bites_a_broken_restep(rig, monkeypatch):
    """The oracle above is worth exactly as much as its ability to fail.

    Stub the re-step to a no-op — the picture then keeps the previous
    step's points and scalars while the session says otherwise, which
    is precisely the stale picture A4 must not ship. The parity body
    must catch it.
    """
    monkeypatch.setattr(
        reconciler_mod, "restep_pane",
        lambda session, view, realized, backend: realized,
    )
    with pytest.raises(AssertionError):
        _parity_body(rig)


def test_a4_a_step_costs_no_realize_and_exactly_one_render(rig, monkeypatch):
    """A4.3.2 / A4.3.3 — the cost claim.

    A cursor-only tick calls ``realize_pane`` ZERO times and renders
    exactly once. The render count is the ADR 0084 discipline the
    session inherited; A4 must not turn one gesture into two frames.
    """
    from apeGmsh.results.session import Instant

    session, view, backend, rec, drain = rig
    view.contour = Contour("displacement_z")
    view.deform = Deform("displacement", 3.0)
    session.time = Instant(STAGE, 0)
    drain()

    calls: list = []
    real = reconciler_mod.realize_pane
    monkeypatch.setattr(
        reconciler_mod, "realize_pane",
        lambda s, p, b: (calls.append(p.id), real(s, p, b))[1],
    )

    before = backend.render_count
    session.time = Instant(STAGE, 1)
    drain()

    assert calls == [], "a cursor-only tick re-realized the pane"
    assert backend.render_count == before + 1, (
        "one gesture must still be one render"
    )


def test_a4_a_theme_change_still_costs_a_full_realize(rig, monkeypatch):
    """A4.3.2's other half, and criterion 12's.

    The palette is not session state, so ``schedule_forced`` is the
    only signal a repaint is owed — the fast path must not swallow it.
    """
    session, view, backend, rec, drain = rig
    view.contour = Contour("displacement_z")
    drain()

    calls: list = []
    real = reconciler_mod.realize_pane
    monkeypatch.setattr(
        reconciler_mod, "realize_pane",
        lambda s, p, b: (calls.append(p.id), real(s, p, b))[1],
    )
    rec.schedule_forced()
    drain()
    assert calls == [view.id]


def test_a4_a_structural_change_still_takes_the_slow_path(rig, monkeypatch):
    """Only the CURSOR half of the signature licenses a re-step.

    A slot fill at a fixed instant must re-realize — the layer SET
    changes, and nothing in the fast path can add or remove a layer.
    """
    session, view, backend, rec, drain = rig
    drain()

    calls: list = []
    real = reconciler_mod.realize_pane
    monkeypatch.setattr(
        reconciler_mod, "realize_pane",
        lambda s, p, b: (calls.append(p.id), real(s, p, b))[1],
    )
    view.contour = Contour("displacement_z")
    drain()
    assert calls == [view.id]


def test_a4_scalar_bars_do_not_churn_across_a_scrub(rig):
    """A4.3.4 — the bars are neither removed nor re-added.

    The contour's scale comes from the store's whole-history limits and
    is fixed at attach, so tearing the bar down per step was never
    buying correctness — and a bar that flickers on every frame of a
    drag is what the user would see.
    """
    from apeGmsh.results.session import Instant

    session, view, backend, rec, drain = rig
    view.contour = Contour("displacement_z")
    session.time = Instant(STAGE, 0)
    drain()
    before = dict(backend.scalar_bars)
    assert before, "the contour registered no bar to begin with"

    for step in (1, 2):
        session.time = Instant(STAGE, step)
        drain()
        assert dict(backend.scalar_bars) == before, (
            f"the scalar bars churned at step {step}"
        )


def test_a4_pick_targets_follow_the_pose(rig):
    """A4.3.5 — ``realized.targets`` is what ``PanePick`` reads.

    A re-step that moved the picture but not the targets would leave a
    click addressing where a node USED to be.
    """
    from apeGmsh.results.session import Instant, MeshStyle

    session, view, backend, rec, drain = rig
    view.style = MeshStyle(mesh=True, outlines=True, nodes=True, gauss=False)
    view.deform = Deform("displacement", 3.0)
    session.time = Instant(STAGE, 0)
    drain()

    session.time = Instant(STAGE, 2)
    drain()

    targets = rec.realized.targets.nodes
    assert targets is not None and targets.rows is not None
    scene_pts = np.asarray(rec.realized.restep.scene.grid.points)
    np.testing.assert_allclose(
        targets.coords, scene_pts[targets.rows], atol=1e-9,
        err_msg="pick targets did not follow the re-stepped pose",
    )


# =====================================================================
# ADR 0098 Amendment 7 (R1) — a clip edit stops rebuilding the pane
# =====================================================================
#
# The A4 block above is the template, deliberately: this is the same
# claim about a different term. ``pane.clips`` used to live in the
# STRUCTURE half of the signature, so one mouse-move of a section-plane
# gizmo drag tore down and rebuilt every layer, diagram and scalar bar —
# measured at 240.8 ms mean per frame on the bench against the
# scrubber's 33 ms budget, i.e. about 4 fps, while a scrub on the SAME
# pane cost 17.7 ms. A4's fast path could never help: it fires only when
# the CURSOR term moved alone, so ``can_restep`` was never consulted.


def _clip_snapshot(rec, backend) -> tuple:
    """The cut AND the picture — the two halves a re-cut must get right.

    Layers as well as clip specs, because the failure being guarded
    against is not "the spec list is wrong" but "the spec list is right
    and the actors did not get it". Asserting only the specs would
    license the claim without testing it.
    """
    return (tuple(backend.clip_planes), _snapshot(rec, backend))


def _clip_parity_body(rig) -> None:
    """Drag a plane with the fast path, then force a full realize with
    the SAME record, and require the two to be identical.

    A7's analogue of A4.3.1, and the whole licence for ``reclip_pane``:
    ``_apply_clips`` is the ONE writer, so a divergence here means the
    fast path invented a second meaning for ``offset`` or ``flipped``.
    """
    session, view, backend, rec, drain = rig
    view.contour = Contour("displacement_z")
    clip = view.add_clip((1.0, 0.0, 0.0), offset=0.0)
    drain()

    for i, offset in enumerate((0.25, 0.5, -0.25, 0.5)):
        view.set_clip(clip.plane_id, offset=offset)
        drain()
        fast = _clip_snapshot(rec, backend)

        rec.schedule_forced()          # same record, the SLOW path
        drain()
        slow = _clip_snapshot(rec, backend)

        assert fast[0] == slow[0], (
            f"drag {i}: the fast path produced a different CUT "
            f"({fast[0]} vs {slow[0]})"
        )
        _assert_same(fast[1], slow[1], i)


def test_a7_reclip_cuts_what_a_full_realize_cuts(rig):
    """A7 criterion 6 — parity between the fast path and a realize."""
    _clip_parity_body(rig)


def test_a7_parity_oracle_bites_a_broken_reclip(rig, monkeypatch):
    """The oracle above is worth exactly as much as its ability to fail.

    Stub the re-cut to a no-op: the backend then keeps the PREVIOUS
    plane's half-space while the record says otherwise — a model cut in
    the wrong place, which is the stale-picture class this fast path
    must not ship.
    """
    monkeypatch.setattr(
        reconciler_mod, "reclip_pane", lambda view, backend: None,
    )
    with pytest.raises(AssertionError):
        _clip_parity_body(rig)


def test_a7_a_clip_edit_costs_no_realize_and_exactly_one_render(
    rig, monkeypatch,
):
    """A7 criterion 5 — the cost claim, over a whole synthetic drag.

    Twenty frames is not decoration: a gizmo drag is per-FRAME, which is
    exactly why 240.8 ms each was unusable. Zero realizes and one render
    per frame is what makes the gesture direct manipulation rather than
    a slideshow.
    """
    session, view, backend, rec, drain = rig
    view.contour = Contour("displacement_z")
    clip = view.add_clip((1.0, 0.0, 0.0), offset=0.0)
    drain()

    calls: list = []
    real = reconciler_mod.realize_pane
    monkeypatch.setattr(
        reconciler_mod, "realize_pane",
        lambda s, p, b: (calls.append(p.id), real(s, p, b))[1],
    )

    before = backend.render_count
    for i in range(20):
        view.set_clip(clip.plane_id, offset=0.02 * (i + 1))
        drain()

    assert calls == [], (
        f"a drag re-realized the pane {len(calls)} time(s) — this is the "
        f"240.8 ms/frame defect A7 exists to fix"
    )
    assert backend.render_count == before + 20, (
        "each drag frame must be exactly one render"
    )


def test_a7_every_clip_field_takes_the_fast_path(rig, monkeypatch):
    """Not just ``offset``. ``active`` changes the spec set, the eye
    changes gizmo visibility, ``flipped`` swaps the surviving half and
    ``name`` changes nothing — none of them changes which ACTORS exist,
    so none of them needs a realize."""
    session, view, backend, rec, drain = rig
    view.contour = Contour("displacement_z")
    clip = view.add_clip((1.0, 0.0, 0.0), offset=0.1)
    drain()

    calls: list = []
    real = reconciler_mod.realize_pane
    monkeypatch.setattr(
        reconciler_mod, "realize_pane",
        lambda s, p, b: (calls.append(p.id), real(s, p, b))[1],
    )

    for changes in (
        {"flipped": True},
        {"active": False},
        {"active": True},
        {"gizmo_visible": False},
        {"name": "Cutting plane"},
        {"normal": (0.0, 1.0, 0.0)},
    ):
        view.set_clip(clip.plane_id, **changes)
        drain()

    assert calls == [], f"these clip edits re-realized: {calls}"


def test_a7_adding_and_removing_a_clip_needs_no_realize(rig, monkeypatch):
    """``add_clip`` / ``remove_clip`` move the clips term, not the
    structure term — the actor set is unchanged either way."""
    session, view, backend, rec, drain = rig
    view.contour = Contour("displacement_z")
    drain()

    calls: list = []
    real = reconciler_mod.realize_pane
    monkeypatch.setattr(
        reconciler_mod, "realize_pane",
        lambda s, p, b: (calls.append(p.id), real(s, p, b))[1],
    )

    clip = view.add_clip((0.0, 0.0, 1.0), offset=0.3)
    drain()
    assert len(backend.clip_planes) == 1, "the new plane did not reach the cut"

    view.remove_clip(clip.plane_id)
    drain()
    assert backend.clip_planes == (), "removing the plane did not clear the cut"
    assert calls == [], f"add/remove re-realized: {calls}"


def test_a7_a_structural_change_alongside_a_clip_still_realizes(
    rig, monkeypatch,
):
    """The guard the fast path must not swallow.

    A slot edit in the same tick as a clip edit changes which actors
    exist, so it takes the full path — and the cut must still be right
    afterwards, because a realize applies the clips itself.
    """
    session, view, backend, rec, drain = rig
    clip = view.add_clip((1.0, 0.0, 0.0), offset=0.0)
    drain()

    calls: list = []
    real = reconciler_mod.realize_pane
    monkeypatch.setattr(
        reconciler_mod, "realize_pane",
        lambda s, p, b: (calls.append(p.id), real(s, p, b))[1],
    )

    view.set_clip(clip.plane_id, offset=0.4)
    view.contour = Contour("displacement_z")      # structural
    drain()

    assert calls == [view.id], "a slot edit must still cost a full realize"
    assert len(backend.clip_planes) == 1
    np.testing.assert_allclose(
        backend.clip_planes[0].origin, (0.4, 0.0, 0.0), atol=1e-9,
        err_msg="the full realize lost the clip edit that rode with it",
    )


def test_a7_a_clip_and_a_step_in_one_tick_both_land(rig, monkeypatch):
    """Both fast paths in one flush: re-cut AND re-step, no realize."""
    from apeGmsh.results.session import Instant

    session, view, backend, rec, drain = rig
    view.contour = Contour("displacement_z")
    view.deform = Deform("displacement", 3.0)
    clip = view.add_clip((1.0, 0.0, 0.0), offset=0.0)
    session.time = Instant(STAGE, 0)
    drain()

    calls: list = []
    real = reconciler_mod.realize_pane
    monkeypatch.setattr(
        reconciler_mod, "realize_pane",
        lambda s, p, b: (calls.append(p.id), real(s, p, b))[1],
    )

    view.set_clip(clip.plane_id, offset=0.35)
    session.time = Instant(STAGE, 2)
    drain()

    assert calls == [], "a clip + step tick re-realized the pane"
    np.testing.assert_allclose(
        backend.clip_planes[0].origin, (0.35, 0.0, 0.0), atol=1e-9)

    # ...and the picture is the one a realize would have painted.
    fast = _clip_snapshot(rec, backend)
    rec.schedule_forced()
    drain()
    slow = _clip_snapshot(rec, backend)
    assert fast[0] == slow[0]
    _assert_same(fast[1], slow[1], 0)


def test_a7_a_forced_flush_still_takes_the_full_path(rig, monkeypatch):
    """A7 criterion 7. The palette is not session state, so
    ``schedule_forced`` is the only signal a repaint is owed — the clip
    fast path must not swallow it any more than A4's does."""
    session, view, backend, rec, drain = rig
    view.contour = Contour("displacement_z")
    view.add_clip((1.0, 0.0, 0.0), offset=0.2)
    drain()

    calls: list = []
    real = reconciler_mod.realize_pane
    monkeypatch.setattr(
        reconciler_mod, "realize_pane",
        lambda s, p, b: (calls.append(p.id), real(s, p, b))[1],
    )
    rec.schedule_forced()
    drain()
    assert calls == [view.id], "a forced flush must re-realize"


@pytest.mark.allow_pump_failures
def test_a7_a_raising_reclip_reports_and_falls_back_in_the_same_flush(
    rig, monkeypatch, pump_failures,
):
    """A7 criterion 7's teeth: a half-cut pane is never left on screen.

    ``reclip_pane`` blows up; the flush must report it and complete the
    edit by the FULL path in the same flush, so the picture the user
    ends up looking at still matches the record.
    """
    session, view, backend, rec, drain = rig
    view.contour = Contour("displacement_z")
    clip = view.add_clip((1.0, 0.0, 0.0), offset=0.0)
    drain()

    def boom(view, backend):
        raise RuntimeError("re-cut exploded")

    monkeypatch.setattr(reconciler_mod, "reclip_pane", boom)
    view.set_clip(clip.plane_id, offset=0.45)
    drain()

    names = [name for name, _exc in pump_failures.failures]
    assert "pump.session_reclip" in names, (
        f"a failed re-cut must be reported, got {names}"
    )
    np.testing.assert_allclose(
        backend.clip_planes[0].origin, (0.45, 0.0, 0.0), atol=1e-9,
        err_msg="the fallback realize did not apply the new cut",
    )


def test_a7_two_panes_are_independent(rig, monkeypatch):
    """A7 criterion 9 — one pane's drag must not re-cut another's."""
    session, view, backend, rec, drain = rig
    other = session.add_view(name="second")
    other.add_clip((0.0, 1.0, 0.0), offset=0.9)

    backend_b = CountingBackend()
    queue_b: list = []
    rec_b = SessionReconciler(
        session, backend_b, pane_id=other.id, defer_fn=queue_b.append,
    )

    def drain_b() -> None:
        while queue_b:
            queue_b.pop(0)()

    clip = view.add_clip((1.0, 0.0, 0.0), offset=0.0)
    drain()
    drain_b()
    try:
        before = tuple(backend_b.clip_planes)
        for i in range(5):
            view.set_clip(clip.plane_id, offset=0.1 * (i + 1))
            drain()
            drain_b()
        assert tuple(backend_b.clip_planes) == before, (
            "dragging pane 1's plane changed pane 2's cut"
        )
        assert other.clips[0].offset == pytest.approx(0.9)
    finally:
        rec_b.dispose()


def test_a7_a_refusing_can_restep_falls_back_to_a_full_realize(
    rig, monkeypatch,
):
    """The A4 guard, gated at last.

    ``can_restep`` refuses when realize could not record what a re-step
    would need (no row map, or a slot that re-fits its LUT to the
    DISPLAYED values per step). Nothing asserted that the refusal was
    HONOURED — every existing fixture happens to be re-steppable, so
    deleting the guard left the whole suite green while a refusing pane
    got re-stepped anyway. Found by mutation-testing A7's restructure of
    this branch; the gap is A4's, the fix belongs with whoever touched
    the code.

    Note this is not merely "a realize happens": the ``pump_failures``
    fixture is autouse, so a bypassed guard that limps through
    ``restep_pane`` and reports would fail here too.
    """
    from apeGmsh.results.session import Instant

    session, view, backend, rec, drain = rig
    view.contour = Contour("displacement_z")
    view.deform = Deform("displacement", 3.0)
    session.time = Instant(STAGE, 0)
    drain()

    calls: list = []
    real = reconciler_mod.realize_pane
    monkeypatch.setattr(
        reconciler_mod, "realize_pane",
        lambda s, p, b: (calls.append(p.id), real(s, p, b))[1],
    )
    monkeypatch.setattr(reconciler_mod, "can_restep", lambda realized, view: False)

    session.time = Instant(STAGE, 2)
    drain()

    assert calls == [view.id], (
        "can_restep refused, so the flush must take the FULL path — "
        "re-stepping a pane realize could not record for is the "
        "half-moved picture A4.4 refuses"
    )


def test_a7_a_refusing_can_restep_still_re_cuts(rig, monkeypatch):
    """...and the clip half is independent of that refusal.

    A pane that cannot be re-stepped can still be re-cut: the two fast
    paths gate on different things, and a clip edit moves no points.
    The full realize applies the clips itself, so the cut is right
    either way — this pins that the combination does not lose it.
    """
    from apeGmsh.results.session import Instant

    session, view, backend, rec, drain = rig
    view.contour = Contour("displacement_z")
    view.deform = Deform("displacement", 3.0)
    clip = view.add_clip((1.0, 0.0, 0.0), offset=0.0)
    session.time = Instant(STAGE, 0)
    drain()

    monkeypatch.setattr(reconciler_mod, "can_restep", lambda realized, view: False)

    view.set_clip(clip.plane_id, offset=0.6)
    session.time = Instant(STAGE, 1)
    drain()

    np.testing.assert_allclose(
        backend.clip_planes[0].origin, (0.6, 0.0, 0.0), atol=1e-9,
        err_msg="the cut was lost when the cursor half refused",
    )


# ---------------------------------------------------------------------
# A7 — reference_bounds, the gizmo quad's box
# ---------------------------------------------------------------------

def test_a7_reference_bounds_are_the_undeformed_box(rig):
    """The gizmo quad is sized from this, so it must NOT follow the pose.

    Deformed bounds would resize the grab handle on every scrub frame:
    the user would watch the thing they are holding change size while
    the plane it represents stayed exactly where it was. It is also
    what makes the gizmo free on A4's fast path — a re-step cannot
    change a reference box, so nothing recomputes.

    The fixture's displacement is ``node_id + t*1000`` at scale 3, so a
    posed box is enormous and unmistakable next to the unit cube.
    """
    from apeGmsh.results.session import Instant

    session, view, backend, rec, drain = rig
    view.contour = Contour("displacement_z")
    session.time = Instant(STAGE, 0)
    drain()
    undeformed = rec.realized.reference_bounds
    assert undeformed is not None

    view.deform = Deform("displacement", 3.0)
    session.time = Instant(STAGE, 2)
    drain()

    posed_pts = np.asarray(rec.realized.restep.scene.grid.points)
    assert posed_pts[:, 2].max() > 100.0, (
        "precondition — the pose should have moved the mesh a long way"
    )
    assert rec.realized.reference_bounds == pytest.approx(undeformed), (
        "the gizmo box followed the deformation"
    )


def test_a7_reference_bounds_survive_a_scrub_unchanged(rig):
    """The A4 fast path must not disturb them either."""
    from apeGmsh.results.session import Instant

    session, view, backend, rec, drain = rig
    view.contour = Contour("displacement_z")
    view.deform = Deform("displacement", 3.0)
    session.time = Instant(STAGE, 0)
    drain()
    first = rec.realized.reference_bounds

    for step in (1, 2, 0):
        session.time = Instant(STAGE, step)
        drain()
        assert rec.realized.reference_bounds == pytest.approx(first)


def test_a7_reference_bounds_are_scoped_to_what_the_pane_draws():
    """A quad spanning geometry the pane does not show is a handle in
    the wrong place. Unit-level, because the shape under test is the
    row map — not the mesh.
    """
    from apeGmsh.viewers.session._realize import _reference_bounds

    class FakeScene:
        reference_points = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 1.0, 1.0],
            [9.0, 9.0, 9.0],        # only in the UNSCOPED answer
        ])

    scene = FakeScene()
    assert _reference_bounds(scene, None) == pytest.approx(
        (0.0, 0.0, 0.0, 9.0, 9.0, 9.0))
    assert _reference_bounds(scene, np.array([0, 1])) == pytest.approx(
        (0.0, 0.0, 0.0, 1.0, 1.0, 1.0))


def test_a7_reference_bounds_refuse_rather_than_guess():
    """A scene that cannot answer yields ``None`` — the pane then draws
    no gizmo instead of putting a handle somewhere arbitrary."""
    from apeGmsh.viewers.session._realize import _reference_bounds

    class Empty:
        reference_points = np.zeros((0, 3))

    class Broken:
        @property
        def reference_points(self):
            raise RuntimeError("no such thing")

    assert _reference_bounds(Empty(), None) is None
    assert _reference_bounds(Broken(), None) is None


# ---------------------------------------------------------------------
# The six-active-plane GL cap, at the backend seam
# ---------------------------------------------------------------------
#
# §3 puts this cap "where planes meet a backend, not in the IR", because
# it is a property of OpenGL rather than of what a view may describe:
# `gl_ClipDistance` guarantees six, and past that the driver is FREE to
# ignore the extras. A seventh active plane therefore does not fail — it
# draws a model that LOOKS cut and is not, which is the worst shape of
# wrong because nothing anywhere says so.


def test_a7_the_cut_never_pushes_more_than_six_planes(rig):
    """The record may hold seven actives (§3 keeps the cap out of the
    IR); the BACKEND must never be handed them."""
    session, view, backend, rec, drain = rig
    for i in range(MAX_ACTIVE_PLANES + 3):
        view.add_clip((1.0, 0.0, 0.0), offset=0.1 * i)
    drain()

    assert sum(1 for c in view.clips if c.active) == MAX_ACTIVE_PLANES + 3
    assert len(backend.clip_planes) == MAX_ACTIVE_PLANES, (
        f"{len(backend.clip_planes)} planes reached the driver; past "
        f"{MAX_ACTIVE_PLANES} it is free to ignore them and the cut lies"
    )


def test_a7_creation_order_decides_which_six_cut(rig):
    """Matching ``ClipPlaneSetController.add``: the first six to be
    activated keep cutting and a seventh is the one that waits. An
    arbitrary six would make the picture depend on dict ordering.
    """
    session, view, backend, rec, drain = rig
    for i in range(MAX_ACTIVE_PLANES + 2):
        view.add_clip((1.0, 0.0, 0.0), offset=float(i))
    drain()

    origins = [tuple(round(c, 6) for c in s.origin)
               for s in backend.clip_planes]
    assert origins == [(float(i), 0.0, 0.0) for i in range(MAX_ACTIVE_PLANES)]


def test_a7_inactive_planes_do_not_consume_a_slot(rig):
    """The cap counts what CUTS, not what exists. Ten planes with four
    cutting must all four cut."""
    session, view, backend, rec, drain = rig
    for i in range(10):
        view.add_clip((1.0, 0.0, 0.0), offset=float(i), active=(i % 3 == 0))
    drain()

    assert sum(1 for c in view.clips if c.active) == 4
    assert len(backend.clip_planes) == 4


def test_a7_the_cap_holds_on_the_reclip_fast_path_too(rig):
    """Both writers, one rule. ``reclip_pane`` delegates to
    ``_apply_clips``, so a cap enforced only on the realize path would
    be a cap the drag route walks straight past.
    """
    session, view, backend, rec, drain = rig
    view.contour = Contour("displacement_z")
    clips = [view.add_clip((1.0, 0.0, 0.0), offset=float(i))
             for i in range(MAX_ACTIVE_PLANES + 2)]
    drain()

    calls: list = []
    real = reconciler_mod.realize_pane
    import pytest as _pytest
    monkeypatch = _pytest.MonkeyPatch()
    try:
        monkeypatch.setattr(
            reconciler_mod, "realize_pane",
            lambda s, p, b: (calls.append(p.id), real(s, p, b))[1],
        )
        view.set_clip(clips[0].plane_id, offset=99.0)     # a drag frame
        drain()
    finally:
        monkeypatch.undo()

    assert calls == [], "precondition — this should have been the fast path"
    assert len(backend.clip_planes) == MAX_ACTIVE_PLANES


def test_a7_apply_clips_reports_how_many_it_dropped():
    """The count is a RETURN, not a log: this runs on every realize and
    every drag frame, so a warning here would fire thirty times a
    second and teach the reader to ignore it. The UI is what says it.
    """
    from apeGmsh.results.session._views import MeshView
    from apeGmsh.viewers.session._realize import _apply_clips

    class Sink:
        def __init__(self):
            self.specs = ()

        def set_clip_planes(self, specs):
            self.specs = tuple(specs)

    view = MeshView(pane_id="mesh-1")
    sink = Sink()
    assert _apply_clips(sink, view) == 0

    for i in range(MAX_ACTIVE_PLANES):
        view.add_clip((1.0, 0.0, 0.0), offset=float(i))
    assert _apply_clips(sink, view) == 0
    assert len(sink.specs) == MAX_ACTIVE_PLANES

    view.add_clip((1.0, 0.0, 0.0), offset=99.0)
    assert _apply_clips(sink, view) == 1
    assert len(sink.specs) == MAX_ACTIVE_PLANES

    view.add_clip((1.0, 0.0, 0.0), offset=98.0, active=False)
    assert _apply_clips(sink, view) == 1, (
        "an INACTIVE extra plane is not a dropped one — it was never "
        "asking to cut"
    )
