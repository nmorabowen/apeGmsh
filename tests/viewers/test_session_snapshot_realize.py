"""ADR 0098 S5a — the snapshot's real oracle: restored == built.

The round-trip oracle is at the **scene-IR layer**, not in pixels
(§11 S1's discipline, and S5c reuses this): realize the session a human
built, realize the session restored from its snapshot, and compare the
emitted layers structurally. A snapshot is only worth what it draws.

Two things make this test discriminating rather than decorative:

* **the selection is non-empty.** ``_realize`` emits a
  ``<pane>:selection`` layer from ``session.selection`` — so an empty
  selection round-trips identically whether or not the field is
  serialised at all, and a test written that way would stay green
  against a writer that drops it (plan decision 14). The layer's
  presence is asserted by name too, so the comparison cannot pass by
  both sides emitting nothing.
* **the comparison is structural, not a key list.** Layer ids matching
  proves nothing about the picture; the points, cells, scalars, colours
  and per-layer style of every layer are compared by value.

Default lane: no Qt, no GL — a ``RecordingBackend`` and array
comparisons, so this runs in CI on Mesa-free machines. There is no
widget in this slice, hence no ``[qt]`` companion; the S4 window suites
stay green unchanged.
"""
from __future__ import annotations

import dataclasses
from pathlib import Path

import numpy as np
import pytest

from apeGmsh.results import Results
from apeGmsh.results.session import (
    Contour,
    Deform,
    Gauss,
    Instant,
    MeshStyle,
    PlotSeries,
    PlotSource,
    Scope,
    restore_snapshot,
    save_snapshot,
    load_snapshot,
    snapshot,
)
from apeGmsh.results.writers import NativeWriter
from apeGmsh.viewers.session import realize_pane

from tests.conftest import _open_model_from_h5

STAGE = "grav"
N_STEPS = 4


# =====================================================================
# Fixture — one mode stage then one static stage (same recipe as S1-A)
# =====================================================================

def _all_element_ids(fem) -> np.ndarray:
    chunks = [np.asarray(gr.ids, dtype=np.int64) for gr in fem.elements]
    return (
        np.concatenate(chunks) if chunks else np.zeros(0, dtype=np.int64)
    )


@pytest.fixture
def snap_results(g, tmp_path: Path):
    g.model.geometry.add_box(0, 0, 0, 1, 1, 1, label="cube")
    g.physical.add_volume("cube", name="Body")
    g.mesh.sizing.set_global_size(2.0)
    g.mesh.generation.generate(dim=3)
    fem = g.mesh.queries.get_fem_data(dim=3)

    node_ids = np.asarray(fem.nodes.ids, dtype=np.int64)
    elem_ids = _all_element_ids(fem)

    disp = np.zeros((N_STEPS, node_ids.size), dtype=np.float64)
    for t in range(N_STEPS):
        disp[t] = node_ids + t * 1000.0
    sxx = np.zeros((N_STEPS, elem_ids.size, 1), dtype=np.float64)
    for t in range(N_STEPS):
        sxx[t, :, 0] = elem_ids * 10.0 + t
    shape = (2.0 * node_ids).astype(np.float64)[None, :]

    path = tmp_path / "session_s5a.h5"
    with NativeWriter(path) as w:
        w.open(fem=fem)
        sid = w.begin_stage(
            name="mode_1", kind="mode",
            time=np.zeros(1, dtype=np.float64),
            eigenvalue=4.0, frequency_hz=0.318, period_s=3.14,
            mode_index=1,
        )
        w.write_nodes(
            sid, "partition_0", node_ids=node_ids,
            components={"displacement_z": shape},
        )
        w.end_stage()
        sid = w.begin_stage(
            name=STAGE, kind="static", stage_id=STAGE,
            time=np.arange(N_STEPS, dtype=np.float64),
        )
        w.write_nodes(
            sid, "partition_0", node_ids=node_ids,
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


def _some_node_ids(session, count: int = 5) -> tuple[int, ...]:
    """A few real node ids, found the way the outline finds them.

    Through S4-2's ``select_all`` — the §8 fourth writer — so the ids
    are ones this model actually has, and then trimmed to a SUBSET: a
    selection of every node would let a writer that serialised "all"
    instead of the ids pass the oracle below.
    """
    from apeGmsh.viewers.session import select_all

    select_all(session, "nodes", axis="physical_groups", names=("Body",))
    ids = session.selection.nodes[:count]
    assert len(ids) == count, "fixture mesh is too small to subset"
    return ids


def _built_session(results):
    """What a human arranges in the window: a posed, contoured, scoped
    pane with Gauss values, glyphs on, and a live selection."""
    session = results.session()
    view = session.panes[0]
    view.deform = Deform("displacement", scale=2.0)
    view.contour = Contour("displacement_z", "unaveraged")
    view.gauss = Gauss("stress_xx")
    view.style = MeshStyle(
        mesh=True, outlines=True, nodes=True, gauss=True,
    )
    # NOT view.overlay: S1 left the undeformed overlay unrealizable, so
    # a realize oracle cannot carry it. It IS snapshot state, and the
    # pure round-trip test in tests/results/session/ covers it.
    view.scope = Scope("physical_groups", ("Body",))
    view.add_clip((1.0, 0.0, 0.0), offset=0.5, name="Cut A")
    session.time = Instant(STAGE, 2)

    node_ids = _some_node_ids(session)
    session.selection.set_nodes(node_ids)

    plot = session.add_plot("history", series=(
        PlotSeries(
            source=PlotSource.node(node_ids[0]),
            quantity="displacement_z",
        ),
    ))
    plot.cursor = Instant(STAGE, 1)
    return session


# =====================================================================
# Structural comparison of two realizes
# =====================================================================

def _canon(obj):
    """A value-comparable image of anything the scene IR holds.

    Arrays become (shape, dtype, bytes); dataclasses and plain objects
    become their fields, recursively. Generic on purpose: a comparison
    that enumerated the fields it knows about would go blind to the
    next field someone adds to a layer.
    """
    if isinstance(obj, np.ndarray):
        return (
            "ndarray", tuple(obj.shape), obj.dtype.str,
            np.ascontiguousarray(obj).tobytes(),
        )
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, (str, bytes, int, float, bool, type(None))):
        return obj
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return (type(obj).__name__, {
            f.name: _canon(getattr(obj, f.name))
            for f in dataclasses.fields(obj)
        })
    if isinstance(obj, dict):
        return {k: _canon(v) for k, v in sorted(obj.items(), key=repr)}
    if isinstance(obj, (list, tuple)):
        return [_canon(x) for x in obj]
    if hasattr(obj, "__dict__"):
        return (type(obj).__name__, {
            k: _canon(v) for k, v in sorted(vars(obj).items())
        })
    return repr(obj)


def _canon_layer(layer, stable_key: str):
    """One layer's value image, with the instance token normalised.

    ``MeshLayer.layer_id`` is derived from the emitting object's
    address (``contour_1cbb9d93140``), so it differs between ANY two
    realizes, including two of the same session. The stable identity is
    ``RealizedLayer.key`` — what the S2 reconciler diffs on, and the key
    of the dict below — so substituting it here drops nothing: every
    other field stays compared by value.
    """
    image = _canon(layer)
    assert image[1]["layer_id"], "a layer must carry an id"
    image[1]["layer_id"] = stable_key
    return image


def _scene(session, pane_id):
    """One pane's realize output, as comparable values."""
    from tests.viewers.conftest import RecordingBackend

    backend = RecordingBackend()
    realized = realize_pane(session, session.pane(pane_id), backend)
    return {
        "layers": {
            entry.key: _canon_layer(
                backend.layers[entry.layer_id], entry.key,
            )
            for entry in realized.layers
        },
        "bars": tuple(realized.scalar_bars),
        "clips": _canon(backend.clip_planes),
    }


def _plot_scene(session, pane_id):
    realized = realize_pane(session, session.pane(pane_id), None)
    return _canon(realized)


# =====================================================================
# The oracle
# =====================================================================

def test_the_digest_is_stable_across_two_realizes(snap_results):
    """Self-check on the oracle itself.

    Every comparison below is worthless if realizing one session twice
    already differs — the equality tests would be coin flips and the
    inequality tests would pass for the wrong reason.
    """
    built = _built_session(snap_results)

    assert _scene(built, "mesh-1") == _scene(built, "mesh-1")
    assert _plot_scene(built, "plot-2") == _plot_scene(built, "plot-2")


def test_restored_realize_equals_built_realize(snap_results):
    built = _built_session(snap_results)
    payload = snapshot(built)

    restored = restore_snapshot(payload, results=snap_results)

    assert restored.notices == ()
    assert _scene(restored.session, "mesh-1") == _scene(built, "mesh-1")


def test_the_selection_layer_is_in_that_comparison(snap_results):
    """Guards the test above against passing vacuously.

    If no ``<pane>:selection`` layer is emitted, the equality above
    holds with the selection field missing from the schema entirely —
    which is exactly the mutation decision 14 exists to prevent.
    """
    built = _built_session(snap_results)

    keys = set(_scene(built, "mesh-1")["layers"])

    assert "mesh-1:selection" in keys
    # ... and the restored side emits it with the same marked points.
    restored = restore_snapshot(snapshot(built), results=snap_results)
    assert (
        _scene(restored.session, "mesh-1")["layers"]["mesh-1:selection"]
        == _scene(built, "mesh-1")["layers"]["mesh-1:selection"]
    )


def test_dropping_the_selection_from_the_file_changes_the_picture(
    snap_results,
):
    """The mutation, run as a test: a snapshot without the selection
    block realizes a DIFFERENT scene. This is what makes the oracle a
    measurement rather than a tautology."""
    built = _built_session(snap_results)
    payload = snapshot(built)
    payload["selection"] = {"kind": None}

    restored = restore_snapshot(payload, results=snap_results)

    assert _scene(restored.session, "mesh-1") != _scene(built, "mesh-1")


def test_restored_plot_pane_resolves_the_same_arrays(snap_results):
    built = _built_session(snap_results)

    restored = restore_snapshot(snapshot(built), results=snap_results)

    assert _plot_scene(restored.session, "plot-2") == _plot_scene(
        built, "plot-2",
    )


def test_a_mode_posed_pane_round_trips_through_realize(snap_results):
    """A mode pose has no instant (§4/§7) — the pane realizes off its
    mode stage on both sides of the file."""
    session = snap_results.session()
    view = session.panes[0]
    view.deform = Deform("displacement", scale=1.0, mode=1)
    view.contour = Contour("displacement_z")
    session.time = Instant(STAGE, 3)

    restored = restore_snapshot(snapshot(session), results=snap_results)

    assert restored.session.effective_instant("mesh-1") is None
    assert _scene(restored.session, "mesh-1") == _scene(session, "mesh-1")


def test_an_unlinked_session_realizes_each_pane_at_its_own_instant(
    snap_results,
):
    """Decision 15, proven where it matters — in the picture.

    Two panes at two different steps of a field that varies with the
    step, so relinking them on restore would show up as equal scenes.
    """
    session = snap_results.session()
    first = session.panes[0]
    first.contour = Contour("displacement_z")
    second = session.add_view()
    second.contour = Contour("displacement_z")
    session.time_linked = False
    session.time = Instant(STAGE, 0)
    first.time = Instant(STAGE, 1)
    second.time = Instant(STAGE, 3)

    restored = restore_snapshot(snapshot(session), results=snap_results)

    scene_first = _scene(restored.session, "mesh-1")
    scene_second = _scene(restored.session, "mesh-2")
    # Each pane matches what it drew before the file.
    assert scene_first == _scene(session, "mesh-1")
    assert scene_second == _scene(session, "mesh-2")
    # And the two panes are NOT the same picture — the assertion that
    # fails if restore relinked them to one instant.
    assert scene_first["layers"] != scene_second["layers"]


# =====================================================================
# The model moved (trap 4b) — end to end, on a real Results
# =====================================================================

def test_a_snapshot_whose_stage_is_gone_degrades_and_still_draws(
    snap_results,
):
    """A renamed stage costs the human that instant — not the session.

    Validated at restore through the same ``results.stage()`` lookup
    realize uses, so the fault is reported against the FILE instead of
    surfacing as a traceback inside a repaint.
    """
    session = snap_results.session()
    session.panes[0].contour = Contour("displacement_z")
    session.time = Instant("stage_from_the_old_run", 2)
    payload = snapshot(session)

    restored = restore_snapshot(payload, results=snap_results)

    assert len(restored.notices) == 1
    assert "stage_from_the_old_run" in restored.notices[0]
    assert restored.session.time is None
    # The picture survived, and it draws: the documented fallback is
    # the last stage's last step.
    assert restored.session.effective_instant("mesh-1") is None
    scene = _scene(restored.session, "mesh-1")
    assert "mesh-1:contour" in scene["layers"]
    assert restored.session.panes[0].contour == Contour("displacement_z")


def test_a_step_past_a_shortened_run_degrades_and_still_draws(
    snap_results,
):
    session = snap_results.session()
    session.panes[0].contour = Contour("displacement_z")
    payload = snapshot(session)
    payload["time"] = {"stage": STAGE, "step": N_STEPS + 5}

    restored = restore_snapshot(payload, results=snap_results)

    assert "past stage" in restored.notices[0]
    assert "mesh-1:contour" in _scene(restored.session, "mesh-1")["layers"]


# =====================================================================
# Through a real file on disk
# =====================================================================

def test_the_file_on_disk_is_the_same_oracle(snap_results, tmp_path):
    """save → load → realize, not just snapshot → restore: the atomic
    write and the JSON encoding are part of the round trip."""
    built = _built_session(snap_results)
    path = tmp_path / "out.h5.session.json"

    save_snapshot(built, path)
    restored = load_snapshot(path, results=snap_results)

    assert restored is not None
    assert _scene(restored.session, "mesh-1") == _scene(built, "mesh-1")


def test_the_default_path_lands_beside_the_results_file(snap_results):
    """And it is NOT the path today's window still owns."""
    built = _built_session(snap_results)

    written = built.save_snapshot()

    assert written.name.endswith(".h5.session.json")
    assert written.parent == Path(snap_results._path).parent
    assert not written.with_name(
        written.name.replace(".session.json", ".viewer-session.json"),
    ).exists()
    restored = load_snapshot(written, results=snap_results)
    assert _scene(restored.session, "mesh-1") == _scene(built, "mesh-1")
