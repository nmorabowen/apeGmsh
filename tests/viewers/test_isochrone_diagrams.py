"""The three isochrone diagram kinds, end to end against a real Results.

Asserts on the *emitted layer* through the shared recording backend
(no GL), the way every migrated diagram is tested post-ADR-0042:

* ``IsochroneMapDiagram`` — a time-valued point scalar, never-arrived
  nodes excluded, a static field across steps, a time-titled bar.
* ``IsochroneProfileDiagram`` — an ordered path polyline in 3-D and a
  ``(C, P)`` curve family through ``read_profile``.
* ``IsochroneStrobeDiagram`` — N replicated wireframe frames in one
  layer, each carrying its own frame time.

Plus the deform-follow contract for all three (the +5Y shift / reset
case the structural guard in ``test_deform_follow_contract.py``
demands).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from apeGmsh.results import Results
from apeGmsh.results.writers import NativeWriter
from apeGmsh.viewers.diagrams import (
    DiagramSpec,
    IsochroneMapDiagram,
    IsochroneMapStyle,
    IsochroneProfileDiagram,
    IsochroneProfileStyle,
    IsochroneStrobeDiagram,
    IsochroneStrobeStyle,
    SlabSelector,
)
from apeGmsh.viewers.diagrams._base import NoDataError
from apeGmsh.viewers.scene.fem_scene import build_fem_scene
from apeGmsh.viewers.scene_ir import MeshLayer

from tests.conftest import _open_model_from_h5


N_STEPS = 5
#: Time vector of the fixture stage: 0, 1, 2, 3, 4.
TIMES = np.arange(N_STEPS, dtype=np.float64)


# =====================================================================
# Fixture — a bar whose "wave" arrives later at higher z
# =====================================================================

@pytest.fixture
def wave_results(g, tmp_path: Path):
    """A meshed box with a nodal field that switches on later with z.

    Node history: ``value(t, node) = 10`` once ``t >= trigger(node)``,
    else ``0``, where ``trigger`` grows with the node's z. That makes
    the arrival time a monotone function of height — a synthetic
    wavefront whose answer is known by construction — and leaves the
    top layer of nodes never triggering, so the never-arrived path is
    exercised too.

    Also writes ``displacement_x/_y/_z`` (a z-proportional sway growing
    with time) so the strobe has a real vector field to warp with.
    """
    g.model.geometry.add_box(0, 0, 0, 1, 1, 4, label="bar")
    g.physical.add_volume("bar", name="Body")
    g.mesh.sizing.set_global_size(2.0)
    g.mesh.generation.generate(dim=3)
    fem = g.mesh.queries.get_fem_data(dim=3)

    node_ids = np.asarray(fem.nodes.ids, dtype=np.int64)
    coords = np.asarray(fem.nodes.coords, dtype=np.float64)
    z = coords[:, 2]

    # trigger step = 1 for the lowest nodes, rising with z; nodes above
    # z_switch never trigger within the 5 steps.
    z_span = float(z.max() - z.min()) or 1.0
    trigger = 1.0 + 4.0 * (z - z.min()) / z_span      # 1 .. 5
    values = np.zeros((N_STEPS, node_ids.size), dtype=np.float64)
    for step in range(N_STEPS):
        values[step] = np.where(step >= trigger, 10.0, 0.0)

    # Sway: u_x grows linearly with both time and height.
    ux = np.outer(TIMES, z) * 0.01
    zeros = np.zeros_like(ux)

    path = tmp_path / "wave.h5"
    with NativeWriter(path) as w:
        w.open(fem=fem)
        sid = w.begin_stage(name="dyn", kind="transient", time=TIMES)
        w.write_nodes(
            sid, "partition_0",
            node_ids=node_ids,
            components={
                "wave": values,
                "displacement_x": ux,
                "displacement_y": zeros,
                "displacement_z": zeros,
            },
        )
        w.end_stage()
    results = Results.from_native(path, model=_open_model_from_h5(path))
    return results, node_ids, coords


@pytest.fixture
def scene(wave_results):
    results, _, _ = wave_results
    return build_fem_scene(results.fem)


# =====================================================================
# Helpers
# =====================================================================

def _map_diagram(results, **style_kwargs):
    return IsochroneMapDiagram(
        DiagramSpec(
            kind="isochrone_map",
            selector=SlabSelector(component="wave"),
            style=IsochroneMapStyle(**style_kwargs),
        ),
        results,
    )


def _profile_diagram(results, **style_kwargs):
    return IsochroneProfileDiagram(
        DiagramSpec(
            kind="isochrone_profile",
            selector=SlabSelector(component="wave", pg=("Body",)),
            style=IsochroneProfileStyle(**style_kwargs),
        ),
        results,
    )


def _strobe_diagram(results, **style_kwargs):
    return IsochroneStrobeDiagram(
        DiagramSpec(
            kind="isochrone_strobe",
            selector=SlabSelector(component="displacement_x"),
            style=IsochroneStrobeStyle(**style_kwargs),
        ),
        results,
    )


#: ``PointSet`` pins coordinates to float32 (the IR's zero-copy-to-VTK
#: contract), so a read-back of computed positions carries float32
#: rounding. Geometry assertions use this rather than a float64 epsilon.
COORD_ATOL = 1e-5


def _layer_points(diagram) -> np.ndarray:
    return np.asarray(diagram._layer.points.coords, dtype=np.float64).copy()


def _shifted(scene, dy: float = 5.0) -> np.ndarray:
    target = np.asarray(scene.grid.points, dtype=np.float64).copy()
    target[:, 1] += dy
    return target


def _assert_follows_and_resets(diagram, scene) -> None:
    """The shared deform-follow contract: +5Y moves, ``None`` resets."""
    before = _layer_points(diagram)
    diagram.sync_substrate_points(_shifted(scene), scene)
    after = _layer_points(diagram)
    np.testing.assert_allclose(
        after - before, np.tile([0.0, 5.0, 0.0], (before.shape[0], 1)),
        atol=1e-5,
    )
    diagram.sync_substrate_points(None, scene)
    np.testing.assert_allclose(_layer_points(diagram), before, atol=1e-5)


# =====================================================================
# IsochroneMapDiagram
# =====================================================================

def test_map_rejects_wrong_style(wave_results) -> None:
    results, _, _ = wave_results
    from apeGmsh.viewers.diagrams._styles import DiagramStyle
    bad = DiagramSpec(
        kind="isochrone_map",
        selector=SlabSelector(component="wave"),
        style=DiagramStyle(),
    )
    with pytest.raises(TypeError, match="IsochroneMapStyle"):
        IsochroneMapDiagram(bad, results)


def test_map_rejects_unknown_mode(wave_results) -> None:
    results, _, _ = wave_results
    with pytest.raises(ValueError, match="mode must be one of"):
        _map_diagram(results, mode="eventually")


def test_map_attach_requires_scene(wave_results, backend) -> None:
    results, _, _ = wave_results
    with pytest.raises(RuntimeError, match="FEMSceneData"):
        _map_diagram(results).attach(backend, results.fem)


def test_map_paints_times_not_values(wave_results, scene, backend) -> None:
    """The emitted scalar must be in the time range, not the field's."""
    results, _, _ = wave_results
    d = _map_diagram(results, threshold=5.0)
    d.attach(backend, results.fem, scene)

    layer = d._layer
    assert isinstance(layer, MeshLayer)
    field = layer.field_named(layer.color.array_name)
    assert field is not None
    assert field.location == "point"
    values = np.asarray(field.values)
    # Painted values are times drawn from TIMES, never the 10.0 field
    # magnitude.
    assert values.min() >= float(TIMES.min())
    assert values.max() <= float(TIMES.max())
    assert 10.0 not in set(values.tolist())


def test_map_arrival_rises_with_height(wave_results, scene, backend) -> None:
    """The synthetic wave switches on later higher up — so must the map."""
    results, _, _ = wave_results
    d = _map_diagram(results, threshold=5.0, interpolate=False)
    d.attach(backend, results.fem, scene)

    pts = _layer_points(d)
    times = np.asarray(d._layer.field_named("arrival_time").values)
    # Correlate painted arrival with z; the construction makes it
    # monotone, so any positive correlation confirms the orientation.
    z = pts[:, 2]
    assert np.corrcoef(z, times)[0, 1] > 0.5


def test_map_excludes_never_arrived_nodes(
    wave_results, scene, backend,
) -> None:
    """Nodes the front never reaches are dropped, not painted."""
    results, node_ids, _ = wave_results
    d = _map_diagram(results, threshold=5.0)
    d.attach(backend, results.fem, scene)
    assert d._n_arrived < d._n_selected, (
        "fixture should leave some nodes un-triggered"
    )
    assert np.isfinite(
        np.asarray(d._layer.field_named("arrival_time").values)
    ).all()


def test_map_time_to_peak_paints_every_node(
    wave_results, scene, backend,
) -> None:
    results, _, _ = wave_results
    d = _map_diagram(results, mode="time_to_peak")
    d.attach(backend, results.fem, scene)
    assert d._n_arrived == d._n_selected


def test_map_raises_when_nothing_ever_arrives(
    wave_results, scene, backend,
) -> None:
    results, _, _ = wave_results
    d = _map_diagram(results, threshold=1.0e9)
    with pytest.raises(NoDataError, match="ever reached"):
        d.attach(backend, results.fem, scene)


def test_map_reports_the_derived_threshold(
    wave_results, scene, backend,
) -> None:
    """An auto threshold is useless if the user can't see its value."""
    results, _, _ = wave_results
    d = _map_diagram(results, threshold=None, threshold_fraction=0.25)
    d.attach(backend, results.fem, scene)
    assert d.threshold_used == pytest.approx(2.5)    # 0.25 x peak(10)
    assert "2.5" in d.describe_criterion()
    assert "arrived" in d.describe_criterion()


def test_map_is_static_across_steps(wave_results, scene, backend) -> None:
    """Scrubbing must not change an arrival map — that's the point."""
    results, _, _ = wave_results
    d = _map_diagram(results, threshold=5.0)
    d.attach(backend, results.fem, scene)
    before = np.asarray(
        d._layer.field_named("arrival_time").values,
    ).copy()
    for step in range(N_STEPS):
        d.update_to_step(step)
    np.testing.assert_allclose(
        np.asarray(d._layer.field_named("arrival_time").values), before,
    )


def test_map_legend_key_is_a_time_quantity(
    wave_results, scene, backend,
) -> None:
    """The legend key must not be the bare component (ADR 0081).

    The key both labels the legend and decides which diagrams SHARE one
    colour scale. Keying an arrival map on its tracked component would
    label the legend ``wave`` while showing seconds AND collapse it onto
    one LUT range together with any contour of ``wave``.
    """
    results, _, _ = wave_results
    d = _map_diagram(results, threshold=5.0)
    d.attach(backend, results.fem, scene)
    geometry, quantity = d._legend_key
    assert geometry == ""
    assert quantity.startswith("t_arrival")
    assert "wave" in quantity
    assert quantity != "wave"

    d2 = _map_diagram(results, mode="time_to_peak")
    d2.attach(backend, results.fem, scene)
    assert d2._legend_key[1].startswith("t_peak")


def test_map_legend_does_not_collapse_onto_a_contour_of_the_same_component(
    wave_results, scene, backend,
) -> None:
    """Seconds and the response value must not share one scale.

    ADR 0081 deliberately collapses same-quantity legends, so this is
    the case that has to NOT collapse: the map's arrival times and a
    contour of the very same component are different quantities in
    different units.
    """
    from apeGmsh.viewers.diagrams import ContourDiagram, ContourStyle

    results, _, _ = wave_results
    d = _map_diagram(results, threshold=5.0)
    d.attach(backend, results.fem, scene)
    contour = ContourDiagram(
        DiagramSpec(
            kind="contour",
            selector=SlabSelector(component="wave"),
            style=ContourStyle(),
        ),
        results,
    )
    contour.attach(backend, results.fem, scene)
    assert d._legend_key != contour._legend_key


def test_two_isochrone_maps_of_one_component_do_share_a_legend(
    wave_results, scene, backend,
) -> None:
    """The flip side: same quantity, same scale — as ADR 0081 intends."""
    results, _, _ = wave_results
    a = _map_diagram(results, threshold=5.0)
    b = _map_diagram(results, threshold=5.0)
    a.attach(backend, results.fem, scene)
    b.attach(backend, results.fem, scene)
    assert a._legend_key == b._legend_key


def test_map_clim_spans_the_arrival_range(
    wave_results, scene, backend,
) -> None:
    results, _, _ = wave_results
    d = _map_diagram(results, threshold=5.0)
    d.attach(backend, results.fem, scene)
    lo, hi = d.current_clim()
    values = np.asarray(d._layer.field_named("arrival_time").values)
    assert lo <= values.min() + 1e-9
    assert hi >= values.max() - 1e-9


def test_map_follows_deformation(wave_results, scene, backend) -> None:
    results, _, _ = wave_results
    d = _map_diagram(results, threshold=5.0)
    d.attach(backend, results.fem, scene)
    _assert_follows_and_resets(d, scene)


def test_map_detach_clears_state(wave_results, scene, backend) -> None:
    results, _, _ = wave_results
    d = _map_diagram(results, threshold=5.0)
    d.attach(backend, results.fem, scene)
    layer_id = d._handle.layer_id
    d.detach()
    assert d._layer is None and d._handle is None
    assert not d.is_attached
    assert layer_id in backend.removed
    assert d._legend_key is None


# =====================================================================
# IsochroneProfileDiagram
# =====================================================================

def test_profile_rejects_wrong_style(wave_results) -> None:
    results, _, _ = wave_results
    from apeGmsh.viewers.diagrams._styles import DiagramStyle
    bad = DiagramSpec(
        kind="isochrone_profile",
        selector=SlabSelector(component="wave"),
        style=DiagramStyle(),
    )
    with pytest.raises(TypeError, match="IsochroneProfileStyle"):
        IsochroneProfileDiagram(bad, results)


def test_profile_rejects_bad_axis(wave_results) -> None:
    results, _, _ = wave_results
    with pytest.raises(ValueError, match="path_axis"):
        _profile_diagram(results, path_axis="w")


def test_profile_refuses_all_nodes(wave_results, scene, backend) -> None:
    """'All nodes' has no ordering — say so instead of drawing nonsense."""
    results, _, _ = wave_results
    d = IsochroneProfileDiagram(
        DiagramSpec(
            kind="isochrone_profile",
            selector=SlabSelector(component="wave"),
            style=IsochroneProfileStyle(),
        ),
        results,
    )
    with pytest.raises(NoDataError, match="explicit path"):
        d.attach(backend, results.fem, scene)


def test_profile_emits_an_ordered_polyline(
    wave_results, scene, backend,
) -> None:
    results, _, _ = wave_results
    d = _profile_diagram(results, path_axis="z")
    d.attach(backend, results.fem, scene)

    layer = d._layer
    assert isinstance(layer, MeshLayer)
    assert set(layer.cells.blocks) == {"line"}
    pts = _layer_points(d)
    # Ordered ascending along z, and one cell per consecutive pair.
    assert np.all(np.diff(pts[:, 2]) >= -1e-9)
    assert layer.cells.n_cells == pts.shape[0] - 1


def test_profile_auto_axis_picks_the_long_direction(
    wave_results, scene, backend,
) -> None:
    """The fixture box is 1x1x4, so 'auto' must resolve to z."""
    results, _, _ = wave_results
    d = _profile_diagram(results, path_axis="auto")
    d.attach(backend, results.fem, scene)
    assert d.path_axis_name == "z"


def test_profile_reads_a_curve_family(wave_results, scene, backend) -> None:
    results, _, _ = wave_results
    d = _profile_diagram(results, path_axis="z", n_curves=3)
    d.attach(backend, results.fem, scene)

    family = d.read_profile()
    assert family is not None
    position, times, values = family
    assert values.shape == (times.size, position.size)
    assert times.size == 3
    # Endpoints of the stage are always drawn.
    assert times[0] == pytest.approx(float(TIMES[0]))
    assert times[-1] == pytest.approx(float(TIMES[-1]))
    # Position is the z coordinate, ascending.
    assert np.all(np.diff(position) >= -1e-9)


def test_profile_family_shows_the_front_advancing(
    wave_results, scene, backend,
) -> None:
    """Later curves must have at least as many switched-on nodes."""
    results, _, _ = wave_results
    d = _profile_diagram(results, path_axis="z", n_curves=N_STEPS)
    d.attach(backend, results.fem, scene)
    _position, _times, values = d.read_profile()
    switched = (values > 5.0).sum(axis=1)
    assert np.all(np.diff(switched) >= 0)
    assert switched[-1] > switched[0]


def test_profile_read_at_step_matches_the_family(
    wave_results, scene, backend,
) -> None:
    """The highlight curve and the family's last curve are the same data."""
    results, _, _ = wave_results
    d = _profile_diagram(results, path_axis="z", n_curves=N_STEPS)
    d.attach(backend, results.fem, scene)
    _position, _times, values = d.read_profile()
    live = d.read_profile_at_step(N_STEPS - 1)
    assert live is not None
    np.testing.assert_allclose(live[1], values[-1])


def test_profile_value_axis_auto_is_upright_for_z(
    wave_results, scene, backend,
) -> None:
    """A depth profile draws value-horizontal (depth vertical)."""
    results, _, _ = wave_results
    d = _profile_diagram(results, path_axis="z")
    d.attach(backend, results.fem, scene)
    assert d.value_on_horizontal() is True

    d2 = _profile_diagram(results, path_axis="x")
    d2.attach(backend, results.fem, scene)
    assert d2.value_on_horizontal() is False


def test_profile_value_axis_override_wins(
    wave_results, scene, backend,
) -> None:
    results, _, _ = wave_results
    d = _profile_diagram(results, path_axis="z", value_axis="vertical")
    d.attach(backend, results.fem, scene)
    assert d.value_on_horizontal() is False


def test_profile_raises_on_missing_component(
    wave_results, scene, backend,
) -> None:
    """Fail at attach rather than hand the panel a blank chart."""
    results, _, _ = wave_results
    d = IsochroneProfileDiagram(
        DiagramSpec(
            kind="isochrone_profile",
            selector=SlabSelector(component="not_recorded", pg=("Body",)),
            style=IsochroneProfileStyle(),
        ),
        results,
    )
    with pytest.raises(NoDataError):
        d.attach(backend, results.fem, scene)


def test_profile_positions_are_material_not_deformed(
    wave_results, scene, backend,
) -> None:
    """The abscissa must not stretch when the substrate warps."""
    results, _, _ = wave_results
    d = _profile_diagram(results, path_axis="z")
    d.attach(backend, results.fem, scene)
    before = d.read_profile()[0].copy()
    d.sync_substrate_points(_shifted(scene, 5.0), scene)
    np.testing.assert_allclose(d.read_profile()[0], before)


def test_profile_follows_deformation(wave_results, scene, backend) -> None:
    results, _, _ = wave_results
    d = _profile_diagram(results, path_axis="z")
    d.attach(backend, results.fem, scene)
    _assert_follows_and_resets(d, scene)


def test_profile_detach_clears_state(wave_results, scene, backend) -> None:
    results, _, _ = wave_results
    d = _profile_diagram(results, path_axis="z")
    d.attach(backend, results.fem, scene)
    layer_id = d._handle.layer_id
    d.detach()
    assert d._layer is None and d.path_node_ids is None
    assert layer_id in backend.removed


# =====================================================================
# IsochroneStrobeDiagram
# =====================================================================

def test_strobe_rejects_wrong_style(wave_results) -> None:
    results, _, _ = wave_results
    from apeGmsh.viewers.diagrams._styles import DiagramStyle
    bad = DiagramSpec(
        kind="isochrone_strobe",
        selector=SlabSelector(component="displacement_x"),
        style=DiagramStyle(),
    )
    with pytest.raises(TypeError, match="IsochroneStrobeStyle"):
        IsochroneStrobeDiagram(bad, results)


def test_strobe_replicates_the_submesh_per_frame(
    wave_results, scene, backend,
) -> None:
    results, _, _ = wave_results
    d = _strobe_diagram(results, n_frames=3, scale=1.0)
    d.attach(backend, results.fem, scene)

    layer = d._layer
    assert isinstance(layer, MeshLayer)
    assert layer.wireframe is True
    n_base = int(scene.grid.n_points)
    assert layer.points.n_points == 3 * n_base
    assert layer.cells.n_cells == 3 * int(scene.grid.n_cells)


def test_strobe_points_carry_their_frame_time(
    wave_results, scene, backend,
) -> None:
    """One layer, one time scale — each frame's points hold its own time."""
    results, _, _ = wave_results
    d = _strobe_diagram(results, n_frames=3, scale=1.0)
    d.attach(backend, results.fem, scene)

    field = d._layer.field_named("frame_time")
    assert field is not None and field.location == "point"
    values = np.asarray(field.values)
    n_base = int(scene.grid.n_points)
    frames = values.reshape(3, n_base)
    # Constant within a frame, strictly increasing across frames.
    for row in frames:
        assert np.allclose(row, row[0])
    assert np.all(np.diff(frames[:, 0]) > 0)
    np.testing.assert_allclose(frames[:, 0], d.frame_times)


def test_strobe_frames_span_the_stage(wave_results, scene, backend) -> None:
    results, _, _ = wave_results
    d = _strobe_diagram(results, n_frames=3, scale=1.0)
    d.attach(backend, results.fem, scene)
    assert d.frame_times[0] == pytest.approx(float(TIMES[0]))
    assert d.frame_times[-1] == pytest.approx(float(TIMES[-1]))


def test_strobe_warps_each_frame_by_its_own_field(
    wave_results, scene, backend,
) -> None:
    """Frame k must sit at ``reference + scale x u(step_k)``."""
    results, _, _ = wave_results
    d = _strobe_diagram(results, n_frames=N_STEPS, scale=1.0)
    d.attach(backend, results.fem, scene)

    n_base = int(scene.grid.n_points)
    pts = _layer_points(d).reshape(N_STEPS, n_base, 3)
    reference = np.asarray(scene.grid.points, dtype=np.float64)
    z = reference[:, 2]
    for k, t in enumerate(d.frame_times):
        expected_ux = float(t) * z * 0.01
        np.testing.assert_allclose(
            pts[k, :, 0] - reference[:, 0], expected_ux, atol=COORD_ATOL,
        )
        # Only x sways in the fixture.
        np.testing.assert_allclose(
            pts[k, :, 1:], reference[:, 1:], atol=COORD_ATOL,
        )


def test_strobe_first_frame_is_undeformed(
    wave_results, scene, backend,
) -> None:
    """t=0 has zero displacement, so frame 0 is the reference shape."""
    results, _, _ = wave_results
    d = _strobe_diagram(results, n_frames=3, scale=1.0)
    d.attach(backend, results.fem, scene)
    n_base = int(scene.grid.n_points)
    first = _layer_points(d)[:n_base]
    np.testing.assert_allclose(
        first, np.asarray(scene.grid.points, dtype=np.float64),
        atol=COORD_ATOL,
    )


def test_strobe_auto_scale_fits_the_model_diagonal(
    wave_results, scene, backend,
) -> None:
    results, _, _ = wave_results
    d = _strobe_diagram(results, n_frames=3, scale=None,
                        auto_scale_fraction=0.1)
    d.attach(backend, results.fem, scene)
    n_base = int(scene.grid.n_points)
    pts = _layer_points(d).reshape(3, n_base, 3)
    reference = np.asarray(scene.grid.points, dtype=np.float64)
    largest = float(np.abs(pts - reference[None, :, :]).max())
    assert largest == pytest.approx(0.1 * scene.model_diagonal, rel=1e-6)


def test_strobe_set_scale_rewarps_in_place(
    wave_results, scene, backend,
) -> None:
    results, _, _ = wave_results
    d = _strobe_diagram(results, n_frames=3, scale=1.0)
    d.attach(backend, results.fem, scene)
    reference = np.asarray(scene.grid.points, dtype=np.float64)
    n_base = int(scene.grid.n_points)
    before = _layer_points(d).reshape(3, n_base, 3) - reference[None, :, :]

    d.set_scale(3.0)
    assert d.scale_used == pytest.approx(3.0)
    after = _layer_points(d).reshape(3, n_base, 3) - reference[None, :, :]
    np.testing.assert_allclose(after, before * 3.0, atol=COORD_ATOL)


def test_strobe_is_static_across_steps(
    wave_results, scene, backend,
) -> None:
    results, _, _ = wave_results
    d = _strobe_diagram(results, n_frames=3, scale=1.0)
    d.attach(backend, results.fem, scene)
    before = _layer_points(d)
    for step in range(N_STEPS):
        d.update_to_step(step)
    np.testing.assert_allclose(_layer_points(d), before)


def test_strobe_respects_the_point_budget(
    wave_results, scene, backend,
) -> None:
    """The replication cost must fail loud, naming the knobs to turn."""
    results, _, _ = wave_results
    d = _strobe_diagram(results, n_frames=4, scale=1.0, max_points=3)
    with pytest.raises(NoDataError, match="max_points"):
        d.attach(backend, results.fem, scene)


def test_strobe_raises_on_unrecorded_field(
    wave_results, scene, backend,
) -> None:
    results, _, _ = wave_results
    d = _strobe_diagram(results, field="rotation", scale=1.0)
    with pytest.raises(NoDataError, match="rotation_x"):
        d.attach(backend, results.fem, scene)


def test_strobe_legend_key_reports_time(
    wave_results, scene, backend,
) -> None:
    results, _, _ = wave_results
    d = _strobe_diagram(results, n_frames=3, scale=1.0)
    d.attach(backend, results.fem, scene)
    quantity = d._legend_key[1]
    assert quantity.startswith("t (")
    assert "strobe" in quantity


def test_strobe_follows_deformation(wave_results, scene, backend) -> None:
    results, _, _ = wave_results
    d = _strobe_diagram(results, n_frames=3, scale=1.0)
    d.attach(backend, results.fem, scene)
    _assert_follows_and_resets(d, scene)


def test_strobe_detach_clears_state(wave_results, scene, backend) -> None:
    results, _, _ = wave_results
    d = _strobe_diagram(results, n_frames=3, scale=1.0)
    d.attach(backend, results.fem, scene)
    layer_id = d._handle.layer_id
    d.detach()
    assert d._layer is None and d.frame_times is None
    assert layer_id in backend.removed
    assert d._legend_key is None


# =====================================================================
# Session round-trip (the S0 regression the registry guard exists for)
# =====================================================================

@pytest.mark.parametrize("kind,style", [
    ("isochrone_map", IsochroneMapStyle(mode="time_to_peak", threshold=1.5)),
    ("isochrone_profile", IsochroneProfileStyle(path_axis="z", n_curves=4)),
    ("isochrone_strobe", IsochroneStrobeStyle(n_frames=9, field="velocity")),
])
def test_isochrone_specs_round_trip_through_the_session_codec(
    kind, style,
) -> None:
    from apeGmsh.viewers.diagrams._session import (
        deserialize_spec, serialize_spec,
    )
    spec = DiagramSpec(
        kind=kind,
        selector=SlabSelector(component="wave", pg=("Body",)),
        style=style,
    )
    back = deserialize_spec(serialize_spec(spec))
    assert back.kind == kind
    assert back.style == style
    assert back.selector.pg == ("Body",)


# =====================================================================
# Settings-tab panels
#
# The three kinds must each dispatch to a real panel rather than the
# "No settings UI for kind" fallback, and their attach-time knobs must
# route through ``_rebuild_with_style`` (a live setter cannot recompute
# an arrival field or re-pick a frame set). Same offscreen-Qt + stub-
# director scaffolding as ``test_section_cut_panel.py``.
# =====================================================================

def _build_settings_tab(results):
    """A DiagramSettingsTab over a stub director bound to ``results``."""
    import os
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from qtpy import QtWidgets

    _ = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    from apeGmsh.viewers.diagrams._geometries import GeometryManager

    class _RegistryStub:
        def diagrams(self):
            return []

    class _CompMgrStub:
        active = None

    class _Director:
        def __init__(self):
            self.geometries = GeometryManager()
            self.stage_id = None
            self.results = results

        def stages(self):
            return []

        def subscribe_stage(self, _cb):
            return lambda: None

        def subscribe_diagrams(self, _cb):
            return lambda: None

        @property
        def registry(self):
            return _RegistryStub()

        @property
        def compositions(self):
            return _CompMgrStub()

    from apeGmsh.viewers.ui._diagram_settings_tab import DiagramSettingsTab
    return DiagramSettingsTab(_Director())


def _dispatch_into_fresh_card(tab, diagram):
    tab._pending_appliers = []
    tab._dispatch_kind_panel(diagram)


def _fallback_shown(tab) -> bool:
    from qtpy import QtWidgets
    return any(
        "No settings UI for kind" in label.text()
        for label in tab._widget.findChildren(QtWidgets.QLabel)
    )


def _find_checkbox(tab, text: str):
    from qtpy import QtWidgets
    for cb in tab._widget.findChildren(QtWidgets.QCheckBox):
        if cb.text() == text:
            return cb
    return None


def _find_combo_with(tab, item_data):
    """First combo box that carries ``item_data`` as an item's data."""
    from qtpy import QtWidgets
    for combo in tab._widget.findChildren(QtWidgets.QComboBox):
        if combo.findData(item_data) >= 0:
            return combo
    return None


@pytest.mark.parametrize("factory,style_kwargs", [
    (_map_diagram, {"threshold": 5.0}),
    (_profile_diagram, {"path_axis": "z"}),
    (_strobe_diagram, {"n_frames": 3, "scale": 1.0}),
])
def test_every_isochrone_kind_has_a_settings_panel(
    wave_results, scene, backend, factory, style_kwargs,
) -> None:
    results, _, _ = wave_results
    d = factory(results, **style_kwargs)
    d.attach(backend, results.fem, scene)
    tab = _build_settings_tab(results)
    _dispatch_into_fresh_card(tab, d)
    assert not _fallback_shown(tab)


def test_map_panel_shows_the_applied_criterion(
    wave_results, scene, backend,
) -> None:
    """The derived threshold must be visible somewhere in the card."""
    from qtpy import QtWidgets
    results, _, _ = wave_results
    d = _map_diagram(results, threshold=None, threshold_fraction=0.25)
    d.attach(backend, results.fem, scene)
    tab = _build_settings_tab(results)
    _dispatch_into_fresh_card(tab, d)
    texts = " | ".join(
        label.text() for label in tab._widget.findChildren(QtWidgets.QLabel)
    )
    assert "2.5" in texts


def test_map_panel_mode_change_rebuilds(
    wave_results, scene, backend,
) -> None:
    """Switching criterion cannot be a live setter — it must rebuild."""
    from unittest.mock import MagicMock
    results, _, _ = wave_results
    d = _map_diagram(results, threshold=5.0)
    d.attach(backend, results.fem, scene)
    tab = _build_settings_tab(results)
    tab._rebuild_with_style = MagicMock()
    _dispatch_into_fresh_card(tab, d)

    combo = _find_combo_with(tab, "time_to_peak")
    assert combo is not None
    combo.setCurrentIndex(combo.findData("time_to_peak"))
    tab._rebuild_with_style.assert_called_with(d, mode="time_to_peak")


def test_map_panel_recompute_button_passes_the_level(
    wave_results, scene, backend,
) -> None:
    from unittest.mock import MagicMock
    from qtpy import QtWidgets
    results, _, _ = wave_results
    d = _map_diagram(results, threshold=5.0)
    d.attach(backend, results.fem, scene)
    tab = _build_settings_tab(results)
    tab._rebuild_with_style = MagicMock()
    _dispatch_into_fresh_card(tab, d)

    button = next(
        b for b in tab._widget.findChildren(QtWidgets.QPushButton)
        if b.text() == "Recompute arrival"
    )
    button.click()
    tab._rebuild_with_style.assert_called_once()
    kwargs = tab._rebuild_with_style.call_args.kwargs
    # Explicit threshold on the style => the auto box is off, so the
    # spin-box level is what gets committed.
    assert kwargs["threshold"] == pytest.approx(5.0)


def test_profile_panel_axis_change_rebuilds(
    wave_results, scene, backend,
) -> None:
    from unittest.mock import MagicMock
    results, _, _ = wave_results
    d = _profile_diagram(results, path_axis="auto")
    d.attach(backend, results.fem, scene)
    tab = _build_settings_tab(results)
    tab._rebuild_with_style = MagicMock()
    _dispatch_into_fresh_card(tab, d)

    combo = _find_combo_with(tab, "auto")
    assert combo is not None
    combo.setCurrentIndex(combo.findData("x"))
    tab._rebuild_with_style.assert_called_with(d, path_axis="x")


def test_strobe_panel_scale_is_a_live_setter(
    wave_results, scene, backend,
) -> None:
    """Scale re-warps cached frames, so it stages instead of rebuilding."""
    from qtpy import QtWidgets
    results, _, _ = wave_results
    d = _strobe_diagram(results, n_frames=3, scale=1.0)
    d.attach(backend, results.fem, scene)
    tab = _build_settings_tab(results)
    _dispatch_into_fresh_card(tab, d)

    spin = next(
        s for s in tab._widget.findChildren(QtWidgets.QDoubleSpinBox)
        if abs(s.value() - 1.0) < 1e-9
    )
    spin.setValue(4.0)
    for applier in tab._pending_appliers:
        applier()
    assert d.scale_used == pytest.approx(4.0)


# =====================================================================
# The curve-family chart (IsochroneProfilePanel)
# =====================================================================

class _PanelDirector:
    """Minimal director surface the profile panel subscribes to."""

    def __init__(self, step: int = 0) -> None:
        self.step_index = step

    def subscribe_step(self, _cb):
        return lambda: None

    def subscribe_stage(self, _cb):
        return lambda: None


def _profile_panel(diagram, step: int = 0):
    import os
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    pytest.importorskip("matplotlib")
    from qtpy import QtWidgets

    _ = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    from apeGmsh.viewers.ui._isochrone_panel import IsochroneProfilePanel
    return IsochroneProfilePanel(diagram, _PanelDirector(step))


def test_panel_draws_one_line_per_instant_plus_the_highlight(
    wave_results, scene, backend,
) -> None:
    results, _, _ = wave_results
    d = _profile_diagram(results, path_axis="z", n_curves=3)
    d.attach(backend, results.fem, scene)
    panel = _profile_panel(d)
    try:
        # 3 family curves + 1 current-step highlight.
        assert len(panel._ax.lines) == 4
        assert panel._highlight is not None
        assert panel._cbar is not None      # time legend
    finally:
        panel.close()


def test_panel_omits_the_highlight_when_disabled(
    wave_results, scene, backend,
) -> None:
    results, _, _ = wave_results
    d = _profile_diagram(
        results, path_axis="z", n_curves=3, mark_current_step=False,
    )
    d.attach(backend, results.fem, scene)
    panel = _profile_panel(d)
    try:
        assert len(panel._ax.lines) == 3
        assert panel._highlight is None
    finally:
        panel.close()


def test_panel_axis_labels_follow_the_layout_choice(
    wave_results, scene, backend,
) -> None:
    """A z path draws value-horizontal, so x is the component."""
    results, _, _ = wave_results
    d = _profile_diagram(results, path_axis="z")
    d.attach(backend, results.fem, scene)
    panel = _profile_panel(d)
    try:
        assert panel._ax.get_xlabel() == "wave"
        assert panel._ax.get_ylabel() == "z"
    finally:
        panel.close()

    d2 = _profile_diagram(results, path_axis="z", value_axis="vertical")
    d2.attach(backend, results.fem, scene)
    panel2 = _profile_panel(d2)
    try:
        assert panel2._ax.get_xlabel() == "z"
        assert panel2._ax.get_ylabel() == "wave"
    finally:
        panel2.close()


def test_panel_redraw_does_not_stack_colorbars(
    wave_results, scene, backend,
) -> None:
    """A stage change re-reads and redraws — bars must not accumulate."""
    results, _, _ = wave_results
    d = _profile_diagram(results, path_axis="z", n_curves=3)
    d.attach(backend, results.fem, scene)
    panel = _profile_panel(d)
    try:
        for _ in range(3):
            panel.refresh()
        assert len(panel._fig.axes) == 2      # the plot + one colourbar
        assert len(panel._ax.lines) == 4
    finally:
        panel.close()


def test_panel_step_change_replaces_the_highlight(
    wave_results, scene, backend,
) -> None:
    results, _, _ = wave_results
    d = _profile_diagram(results, path_axis="z", n_curves=3)
    d.attach(backend, results.fem, scene)
    panel = _profile_panel(d)
    try:
        first = panel._highlight
        panel._director.step_index = N_STEPS - 1
        panel._refresh_highlight()
        assert panel._highlight is not first
        # Still exactly one highlight on top of the 3-curve family.
        assert len(panel._ax.lines) == 4
    finally:
        panel.close()


def test_diagram_makes_a_side_panel(wave_results, scene, backend) -> None:
    """``make_side_panel`` is how the plot pane gets the chart."""
    results, _, _ = wave_results
    d = _profile_diagram(results, path_axis="z")
    d.attach(backend, results.fem, scene)
    panel = d.make_side_panel(_PanelDirector())
    try:
        assert panel is not None
        assert panel.widget is not None
        assert hasattr(panel, "attach_dispatcher")
    finally:
        if panel is not None:
            panel.close()


def test_unattached_diagram_makes_no_side_panel(wave_results) -> None:
    results, _, _ = wave_results
    d = _profile_diagram(results, path_axis="z")
    assert d.make_side_panel(_PanelDirector()) is None


# =====================================================================
# Real-backend integration (offscreen PyVistaQtBackend)
# =====================================================================

def test_map_legend_reaches_the_controller(
    wave_results, scene, pv_backend,
) -> None:
    results, _, _ = wave_results
    d = _map_diagram(results, threshold=5.0)
    d.attach(pv_backend, results.fem, scene)
    from apeGmsh.viewers.core._legend import controller_for
    keys = [entry.key for entry in controller_for(pv_backend).entries()]
    assert ("", "t_arrival (wave)") in keys


def test_profile_polyline_reaches_the_plotter_as_lines(
    wave_results, scene, pv_backend,
) -> None:
    """The 'line' cell token must survive translation to real VTK cells."""
    results, _, _ = wave_results
    d = _profile_diagram(results, path_axis="z")
    d.attach(pv_backend, results.fem, scene)
    dataset = d._handle.dataset
    assert dataset.n_cells == d._layer.cells.n_cells
    # VTK_LINE == 3
    assert set(np.unique(dataset.celltypes).tolist()) == {3}


def test_strobe_wireframe_reaches_the_plotter(
    wave_results, scene, pv_backend,
) -> None:
    results, _, _ = wave_results
    d = _strobe_diagram(results, n_frames=3, scale=1.0)
    d.attach(pv_backend, results.fem, scene)
    dataset = d._handle.dataset
    assert dataset.n_points == 3 * int(scene.grid.n_points)
    assert "frame_time" in dataset.point_data
    # line_width made it onto the actor's property.
    assert d._handle.actor.prop.line_width == pytest.approx(
        d.spec.style.line_width,
    )


def test_profile_path_order_is_geometric_not_node_numbering(
    wave_results, scene, backend,
) -> None:
    """Ties on the path axis break by the other two coordinates.

    A physical group is the easiest selector to pick and is usually
    several nodes wide, so ties are the norm. Ordering them by mesh id
    would make the drawn path depend on node numbering; lexicographic
    order makes it depend only on geometry.
    """
    results, _, _ = wave_results
    d = _profile_diagram(results, path_axis="z")
    d.attach(backend, results.fem, scene)

    # Assert on the exact float64 coordinates the sort ran against —
    # the emitted PointSet is float32, whose rounding would blur the
    # tie groups this test is about.
    exact = np.asarray(scene.grid.points, dtype=np.float64)[
        d._substrate_rows
    ]
    # Ordered (z, x, y) lexicographically: each row is >= its
    # predecessor under that key.
    keys = exact[:, [2, 0, 1]]
    for prev, nxt in zip(keys[:-1], keys[1:]):
        assert tuple(prev) <= tuple(nxt), (
            f"path order is not lexicographic in (z, x, y): "
            f"{prev} came before {nxt}"
        )
    # And the fixture really does have ties to break, or the assertion
    # above would pass vacuously.
    assert np.unique(exact[:, 2]).size < exact.shape[0]


# =====================================================================
# Regressions found by adversarial review
# =====================================================================

def test_strobe_scale_survives_a_round_trip_through_zero(
    wave_results, scene, backend,
) -> None:
    """0 must not be a one-way door.

    ``set_scale`` used to derive the new warp from the previous SCALED
    array by the ratio ``new/old``, so a pass through 0 multiplied the
    field by 0 and every later scale multiplied 0 again — the strobe
    stayed flat for the rest of the session, with ``scale_used``
    cheerfully reporting the requested value. The settings panel's
    spin-box range starts at 0, so this was one drag away.
    """
    results, _, _ = wave_results
    d = _strobe_diagram(results, n_frames=3, scale=1.0)
    d.attach(backend, results.fem, scene)
    reference = np.asarray(scene.grid.points, dtype=np.float64)
    n_base = int(scene.grid.n_points)

    def warp() -> np.ndarray:
        pts = _layer_points(d).reshape(3, n_base, 3)
        return pts - reference[None, :, :]

    at_one = warp().copy()
    assert np.abs(at_one).max() > 0.0        # the fixture really warps

    d.set_scale(0.0)
    assert np.abs(warp()).max() == pytest.approx(0.0, abs=COORD_ATOL)

    d.set_scale(4.0)
    assert d.scale_used == pytest.approx(4.0)
    np.testing.assert_allclose(warp(), at_one * 4.0, atol=COORD_ATOL)


def test_strobe_scale_is_absolute_not_cumulative(
    wave_results, scene, backend,
) -> None:
    """Repeating the same scale must not compound it."""
    results, _, _ = wave_results
    d = _strobe_diagram(results, n_frames=3, scale=1.0)
    d.attach(backend, results.fem, scene)
    d.set_scale(3.0)
    once = _layer_points(d).copy()
    d.set_scale(3.0)
    np.testing.assert_allclose(_layer_points(d), once, atol=COORD_ATOL)


def test_map_refuses_an_over_budget_history_read(
    wave_results, scene, backend,
) -> None:
    """The (T x N) whole-history read is sized up front, not attempted.

    An arrival map must read every step of every selected node, so on a
    large model the read alone can be tens of GiB. Without a budget the
    viewer dies inside h5py with no diagnosis; the strobe already had
    ``max_points``, and this is the map's equivalent.
    """
    results, _, _ = wave_results
    d = _map_diagram(results, threshold=5.0, max_history_samples=4)
    with pytest.raises(NoDataError, match="max_history_samples"):
        d.attach(backend, results.fem, scene)


def test_map_budget_message_names_the_sizes(
    wave_results, scene, backend,
) -> None:
    """The refusal has to be actionable: steps, nodes, and the fix."""
    results, _, _ = wave_results
    d = _map_diagram(results, threshold=5.0, max_history_samples=4)
    with pytest.raises(NoDataError) as exc:
        d.attach(backend, results.fem, scene)
    message = str(exc.value)
    assert f"{N_STEPS} steps" in message
    assert "selector" in message


def test_map_budget_off_by_zero_or_negative(
    wave_results, scene, backend,
) -> None:
    """A non-positive budget means 'no limit', not 'refuse everything'."""
    results, _, _ = wave_results
    d = _map_diagram(results, threshold=5.0, max_history_samples=0)
    d.attach(backend, results.fem, scene)     # must not raise
    assert d._layer is not None


def test_map_generous_default_budget_allows_normal_models(
    wave_results, scene, backend,
) -> None:
    """The default must not get in a real user's way."""
    results, _, _ = wave_results
    d = _map_diagram(results, threshold=5.0)
    assert d.spec.style.max_history_samples >= 10_000_000
    d.attach(backend, results.fem, scene)
    assert d._layer is not None
