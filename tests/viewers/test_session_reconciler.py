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
