"""X3-c (#1733) — ADR 0098 Amendment 7 realized: slot style and the ghost.

**A7.3.** A colour-mapped slot's explicit ``clim`` / ``cmap`` reach the
picture: the drawn contour maps with exactly that range and that colour
map (``'jet'`` renders as jet), and the scale bar says the same. Unset,
nothing changes; that half is held by the X2 numeric oracle, the image
goldens and the X2-f staged tests, which this slice leaves untouched.

**A7.5.** ``MeshView.overlay`` draws the undeformed ghost: the view's
own cell set at scale 0 in the Outlines style, under the same scope,
stage-activation set and clip planes as the posed mesh, whether or not
the Outlines button is on, with no colour map and no legend. Toggling
``overlay`` is a full realize; a time change inside a stage keeps the
ghost (it sits in reference geometry), and a stage change re-realizes
it with that stage's active cells.

Seams: the ``capture_drawn`` hook of ``test_x2_numeric_oracle`` (the
numbers in the scene at the moment a still is shot), extended here
with the mapper's lookup table and clipping planes; and a headless
``RecordingBackend`` for the layer-level checks. Every test here fails
on the main before this slice: the slot style was stored but never
drawn, and ``overlay=True`` refused to realize.
"""
from __future__ import annotations

import contextlib
from typing import Any, Iterator

import numpy as np
import pytest

pytest.importorskip("pyvista")

from tests.viewers.test_x2_numeric_oracle import (  # noqa: E402
    _require_shots,
    capture_drawn,
    plate,  # noqa: F401  (fixture)
    plate_h5,  # noqa: F401  (fixture)
)
from tests.viewers.test_x2f_animation_stage_activation import (  # noqa: E402
    _EXPECT,
    _pg_eids,
    staged,  # noqa: F401  (fixture)
)

_WINDOW = (240, 180)
_CLIM = (-0.25, 0.75)


# ---------------------------------------------------------------------
# Seam: the LUT and clip planes of the drawn scene, per shot
# ---------------------------------------------------------------------


def _scene_props(plotter: Any) -> dict[str, dict[str, Any]]:
    """Per visible 3-D actor (by name): the LUT table of a colour-mapped
    one, and the number of clipping planes on its mapper."""
    out: dict[str, dict[str, Any]] = {}
    for renderer in plotter.renderers:
        for name, actor in renderer.actors.items():
            if not actor.GetVisibility() or actor.IsA("vtkActor2D"):
                continue
            mapper = getattr(actor, "mapper", None)
            if mapper is None:
                continue
            planes = mapper.GetClippingPlanes()
            ds = getattr(mapper, "dataset", None)
            rec: dict[str, Any] = {
                "clip_planes": 0 if planes is None else planes.GetNumberOfItems(),
                "lut": None,
                "points": None if ds is None else np.array(ds.points),
            }
            if mapper.GetScalarVisibility():
                lut = mapper.GetLookupTable()
                rec["lut"] = np.array([
                    lut.GetTableValue(i)[:3]
                    for i in range(lut.GetNumberOfTableValues())
                ])
            out[str(name)] = rec
    return out


@contextlib.contextmanager
def _capture() -> Iterator[tuple[list[dict[str, Any]], list[dict[str, Any]]]]:
    """``capture_drawn`` plus :func:`_scene_props` of the same shots."""
    from pyvista.plotting.plotter import BasePlotter

    props: list[dict[str, Any]] = []
    with capture_drawn() as records:
        shot = BasePlotter.screenshot  # capture_drawn's wrapper

        def _screenshot(self: Any, *a: Any, **k: Any) -> Any:
            props.append(_scene_props(self))
            return shot(self, *a, **k)

        BasePlotter.screenshot = _screenshot  # type: ignore[method-assign]
        try:
            yield records, props
        finally:
            BasePlotter.screenshot = shot  # type: ignore[method-assign]


def _cmap_table(name: str, n: int) -> np.ndarray:
    import matplotlib

    return np.asarray(matplotlib.colormaps[name](np.linspace(0.0, 1.0, n)))[:, :3]


def _mapped_lut(props: dict[str, dict[str, Any]]) -> np.ndarray:
    luts = [p["lut"] for p in props.values() if p["lut"] is not None]
    assert len(luts) == 1, f"expected one colour-mapped actor, got {len(luts)}"
    return luts[0]


# ---------------------------------------------------------------------
# A7.3 — clim / cmap
# ---------------------------------------------------------------------


def _contour_session(results: Any, **style: Any) -> tuple[Any, Any]:
    from apeGmsh.results.session import Contour

    session = results.session()
    view = session.panes[0]
    view.contour = Contour("displacement_y", **style)
    return session, view


def test_explicit_clim_and_jet_are_what_the_still_draws(plate, tmp_path):
    session, view = _contour_session(plate, clim=_CLIM, cmap="jet")
    with _capture() as (records, props):
        written = session.render(tmp_path / "jet.png", view, window_size=_WINDOW)
    _require_shots(records, 1, written)
    contour = records[0]["contour"]
    assert contour is not None
    np.testing.assert_allclose(contour["clim"], _CLIM, rtol=0, atol=1e-12)
    lut = _mapped_lut(props[0])
    # VTK stores the table as 8-bit colour: one quantum of tolerance.
    np.testing.assert_allclose(
        lut, _cmap_table("jet", len(lut)), rtol=0, atol=1.5 / 255,
    )


def test_unset_style_keeps_the_default_map_and_auto_range(plate, tmp_path):
    """The control: the same slot unset draws neither jet nor ``_CLIM``,
    so the test above cannot pass on a default that happens to match."""
    session, view = _contour_session(plate)
    with _capture() as (records, props):
        written = session.render(tmp_path / "dflt.png", view, window_size=_WINDOW)
    _require_shots(records, 1, written)
    assert not np.allclose(records[0]["contour"]["clim"], _CLIM)
    lut = _mapped_lut(props[0])
    assert not np.allclose(lut, _cmap_table("jet", len(lut)), atol=1.5 / 255)


def test_style_reaches_the_layer_and_the_scale_bar(plate):
    from apeGmsh.viewers.session._realize import _layer_lutspec, realize_pane

    from tests.viewers.conftest import RecordingBackend

    session, view = _contour_session(plate, clim=_CLIM, cmap="jet")
    realized = realize_pane(session, view, RecordingBackend())
    (diagram,) = realized.diagrams
    lut = _layer_lutspec(diagram)
    assert (float(lut.vmin), float(lut.vmax)) == _CLIM
    assert lut.name == "jet"
    assert len(realized.scalar_bars) == 1


def test_a_none_slot_takes_its_same_field_neighbours_style(staged):
    """A7.3 rule 2: one scale per field; the slot left at ``None`` draws
    with the range and map its same-field neighbour sets."""
    from apeGmsh.results.session import Contour, Sand
    from apeGmsh.viewers.session._realize import _layer_lutspec, realize_pane

    from tests.viewers.conftest import RecordingBackend

    results, _n = staged
    session = results.session()
    view = session.panes[0]
    view.sand = Sand("displacement_z")
    view.contour = Contour("displacement_z", clim=_CLIM, cmap="jet")
    realized = realize_pane(session, view, RecordingBackend())
    assert len(realized.diagrams) == 2
    for diagram in realized.diagrams:
        lut = _layer_lutspec(diagram)
        assert (float(lut.vmin), float(lut.vmax)) == _CLIM, diagram.spec.kind
        assert lut.name == "jet", diagram.spec.kind


# ---------------------------------------------------------------------
# A7.5 — the undeformed ghost
# ---------------------------------------------------------------------


def _ghost(realized: Any) -> Any:
    ghosts = [e for e in realized.layers if e.key.endswith(":ghost")]
    assert len(ghosts) <= 1
    return ghosts[0].layer if ghosts else None


def _feature_edges(grid: Any) -> Any:
    from apeGmsh.viewers.session._realize import _outline_feature_angle

    return grid.extract_surface().extract_feature_edges(
        feature_angle=_outline_feature_angle(),
        boundary_edges=True,
        feature_edges=True,
        manifold_edges=False,
        non_manifold_edges=False,
    )


def _reference_edges(results: Any, eids: "set[int] | None") -> Any:
    """Feature edges of the cell set ``eids`` in REFERENCE geometry,
    extracted independently of realize."""
    from apeGmsh.viewers.scene.fem_scene import build_fem_scene

    scene = build_fem_scene(results.fem)
    grid = scene.grid.copy()
    grid.points = np.asarray(scene.reference_points, dtype=np.float64)
    if eids is not None:
        cells = sorted(scene.element_id_to_cell[e] for e in eids)
        grid = grid.extract_cells(np.asarray(cells, dtype=np.int64))
    return _feature_edges(grid)


def _sorted_rows(pts: Any) -> np.ndarray:
    a = np.round(np.asarray(pts, dtype=np.float64), 9)
    return a[np.lexsort(a.T[::-1])]


def _unique_rows(pts: Any) -> np.ndarray:
    return np.unique(np.round(np.asarray(pts, dtype=np.float64), 9), axis=0)


@pytest.mark.parametrize("stage", sorted(_EXPECT))
def test_ghost_is_the_stage_active_cell_set(staged, stage):
    """The ghost draws exactly the realized stage's active cells (inactive
    elements absent), edge for edge, with the Outlines button OFF."""
    from apeGmsh.results.session import Instant, MeshStyle
    from apeGmsh.viewers.session._realize import realize_pane

    from tests.viewers.conftest import RecordingBackend

    results, _n = staged
    pgs = _pg_eids(results)
    _z, rock, fill = _EXPECT[stage]
    active = (pgs["Rock"] if rock else set()) | (pgs["Fill"] if fill else set())
    want = _reference_edges(results, active)

    session = results.session()
    session.time = Instant({s.name: s.id for s in results.stages}[stage], 0)
    view = session.panes[0]
    view.style = MeshStyle(outlines=False)
    view.overlay = True
    realized = realize_pane(session, view, RecordingBackend())
    ghost = _ghost(realized)
    assert ghost is not None
    assert not any(e.key.endswith(":outlines") for e in realized.layers)
    assert ghost.cells.blocks["line"].shape[0] == want.n_lines
    np.testing.assert_allclose(
        _unique_rows(ghost.points.coords), _unique_rows(want.points),
        atol=1e-9,
    )
    assert realized.scalar_bars == ()


def test_ghost_follows_scope_and_activation(staged):
    """Scope AND active: Rock-scoped at 'construction' ghosts Rock only."""
    from apeGmsh.results.session import Instant, Scope
    from apeGmsh.viewers.session._realize import realize_pane

    from tests.viewers.conftest import RecordingBackend

    results, _n = staged
    want = _reference_edges(results, _pg_eids(results)["Rock"])
    session = results.session()
    session.time = Instant(
        {s.name: s.id for s in results.stages}["construction"], 0,
    )
    view = session.panes[0]
    view.scope = Scope("physical_groups", ("Rock",))
    view.overlay = True
    ghost = _ghost(realize_pane(session, view, RecordingBackend()))
    assert ghost is not None
    assert ghost.cells.blocks["line"].shape[0] == want.n_lines
    assert float(np.max(ghost.points.coords[:, 2])) == pytest.approx(1.0)


def test_overlay_off_draws_no_ghost(plate):
    from apeGmsh.viewers.session._realize import realize_pane

    from tests.viewers.conftest import RecordingBackend

    session = plate.session()
    realized = realize_pane(session, session.panes[0], RecordingBackend())
    assert _ghost(realized) is None


def _deformed_session(results: Any) -> tuple[Any, Any]:
    from apeGmsh.results.session import Deform, Instant

    session = results.session()
    stage = results.stages[-1].id
    session.time = Instant(stage, 4)
    view = session.panes[0]
    view.deform = Deform("displacement", scale=1.0)
    view.overlay = True
    return session, view


def test_ghost_is_at_scale_zero_under_a_deformed_pose(plate):
    """Deform on: the outlines are posed, the ghost is the reference
    mesh, and with every slot empty there is still no legend."""
    from apeGmsh.viewers.session._realize import realize_pane

    from tests.viewers.conftest import RecordingBackend

    session, view = _deformed_session(plate)
    realized = realize_pane(session, view, RecordingBackend())
    ghost = _ghost(realized)
    want = _reference_edges(plate, None)
    assert ghost.cells.blocks["line"].shape[0] == want.n_lines
    np.testing.assert_allclose(
        _unique_rows(ghost.points.coords), _unique_rows(want.points),
        atol=1e-9,
    )
    outlines = next(e for e in realized.layers if e.key.endswith(":outlines"))
    assert not np.allclose(
        _sorted_rows(outlines.layer.points.coords),
        _sorted_rows(ghost.points.coords),
    ), "the fixture must deform, or this test proves nothing"
    assert realized.scalar_bars == ()
    assert ghost.color.mode == "solid"


def test_ghost_re_steps_without_staleness(plate):
    """A time change inside a stage re-steps the pane; the ghost after
    the re-step equals the ghost a full realize draws at that instant."""
    from apeGmsh.results.session import Instant
    from apeGmsh.viewers.session._realize import (
        can_restep, realize_pane, restep_pane,
    )

    from tests.viewers.conftest import RecordingBackend

    session, view = _deformed_session(plate)
    backend = RecordingBackend()
    stage = plate.stages[-1].id
    session.time = Instant(stage, 2)
    realized = realize_pane(session, view, backend)
    session.time = Instant(stage, 4)
    assert can_restep(realized, view)
    stepped = restep_pane(session, view, realized, backend)
    fresh = realize_pane(session, view, RecordingBackend())
    np.testing.assert_array_equal(
        _ghost(stepped).points.coords, _ghost(fresh).points.coords,
    )
    np.testing.assert_array_equal(
        _ghost(stepped).cells.blocks["line"], _ghost(fresh).cells.blocks["line"],
    )


def test_window_pane_ghost_follows_stage_and_overlay_toggle(staged):
    """The window's realized pane (headless reconciler): the ghost tracks
    each stage's active set across time changes, and toggling
    ``overlay`` adds or removes it (a structure change)."""
    from apeGmsh.results.session import Instant
    from apeGmsh.viewers.session import SessionReconciler

    from tests.viewers.conftest import RecordingBackend

    results, _n = staged
    pgs = _pg_eids(results)
    ids = {s.name: s.id for s in results.stages}
    session = results.session()
    queue: list = []
    rec = SessionReconciler(session, RecordingBackend(), defer_fn=queue.append)

    def drain() -> None:
        while queue:
            queue.pop(0)()

    def ghost_lines() -> "int | None":
        ghost = _ghost(rec.realized)
        return None if ghost is None else ghost.cells.blocks["line"].shape[0]

    def want(stage: str) -> int:
        _z, rock, fill = _EXPECT[stage]
        eids = (pgs["Rock"] if rock else set()) | (pgs["Fill"] if fill else set())
        return _reference_edges(results, eids).n_lines

    try:
        session.time = Instant(ids["g"], 0)
        drain()
        assert ghost_lines() is None
        session.panes[0].overlay = True
        drain()
        assert ghost_lines() == want("g")
        for stage in ("construction", "excavate", "g"):
            session.time = Instant(ids[stage], 0)
            drain()
            assert ghost_lines() == want(stage), stage
            session.time = Instant(ids[stage], 1)   # re-step in-stage
            drain()
            assert ghost_lines() == want(stage), stage
        session.panes[0].overlay = False
        drain()
        assert ghost_lines() is None
    finally:
        rec.dispose()


def test_ghost_is_cut_by_the_views_clip_planes(plate, tmp_path):
    """The drawn ghost actor carries the view's section plane, like the
    posed mesh does."""
    session, view = _deformed_session(plate)
    view.add_clip(normal=(1.0, 0.0, 0.0), offset=1.0)
    with _capture() as (records, props):
        written = session.render(tmp_path / "clip.png", view, window_size=_WINDOW)
    _require_shots(records, 1, written)
    # The ghost actor is the one drawing the REFERENCE feature edges.
    want = _unique_rows(_reference_edges(plate, None).points)
    actors = list(props[0].values())
    ghost = [
        p for p in actors
        if p["points"] is not None
        and _unique_rows(p["points"]).shape == want.shape
        and np.allclose(_unique_rows(p["points"]), want, atol=1e-6)
    ]
    assert len(ghost) == 1, len(actors)
    assert ghost[0]["clip_planes"] == 1
    assert len(actors) >= 3  # substrate, outlines, ghost
    assert all(p["clip_planes"] == 1 for p in actors)


def test_overlay_leaves_every_posed_layer_unchanged(plate):
    """The ghost reads the reference points from its own copy; turning
    it on must not un-pose the substrate or the outlines (a shallow
    copy shares the scene's vtkPoints, and pyvista's ``points`` setter
    writes into it, so a careless write would)."""
    from apeGmsh.viewers.session._realize import realize_pane

    from tests.viewers.conftest import RecordingBackend

    session, view = _deformed_session(plate)
    on = realize_pane(session, view, RecordingBackend())
    view.overlay = False
    off = realize_pane(session, view, RecordingBackend())
    on_layers = {e.key: e.layer for e in on.layers if e.layer is not None}
    off_layers = {e.key: e.layer for e in off.layers if e.layer is not None}
    assert set(on_layers) - set(off_layers) == {f"{view.id}:ghost"}
    for key, layer in off_layers.items():
        np.testing.assert_array_equal(
            on_layers[key].points.coords, layer.points.coords, err_msg=key,
        )
