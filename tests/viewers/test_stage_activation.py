"""Stage-activation masks (ADR 0055 viewer-consume V1).

Pure-math coverage for ``viewers/data/_stage_activation.py``: the
active-set semantics (``active(K) = global ∪ owned(1..K) −
removed(1..K)``, removals hidden IN their removing stage), the
ops-tag → fem-eid → cell join with fail-soft unmapped handling, the
name-keyed capture↔program pairing, and the controller's LAYER_STAGE
ownership (set on match, clear on unmatch/disable).
"""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from apeGmsh.viewers.data._stage_activation import (
    LAYER_STAGE,
    StageActivationController,
    build_stage_activation,
    pair_capture_to_program,
)


def _stage(name, owned=(), removed=()):
    return SimpleNamespace(
        name=name,
        owned_element_ids=tuple(owned),
        remove_elements=tuple(removed),
    )


# Grid fixture: 4 cells, fem eids 10/20/30/40 → cells 0/1/2/3.
# Ops tags 1..4 map 1:1 onto those eids.
TAG_TO_EID = {1: 10, 2: 20, 3: 30, 4: 40}
EID_TO_CELL = {10: 0, 20: 1, 30: 2, 40: 3}
N_CELLS = 4


# ---------------------------------------------------------------------------
# build_stage_activation — mask math
# ---------------------------------------------------------------------------


def test_ownership_is_cumulative() -> None:
    """Stage-owned cells are hidden until their owning stage; global
    cells (owned by no stage) are never hidden."""
    # tags 3 and 4 are stage-owned; tags 1, 2 are global.
    stages = [_stage("construction", owned=[3]), _stage("loading", owned=[4])]
    act = build_stage_activation(stages, TAG_TO_EID, EID_TO_CELL, N_CELLS)
    assert act is not None
    # Stage 1: cell of tag 4 (stage 2's) hidden; tag 3's visible.
    np.testing.assert_array_equal(
        act.hidden_by_name["construction"],
        np.array([False, False, False, True]),
    )
    # Stage 2: everything activated.
    np.testing.assert_array_equal(
        act.hidden_by_name["loading"],
        np.zeros(N_CELLS, dtype=bool),
    )
    np.testing.assert_array_equal(act.final_hidden, np.zeros(N_CELLS, bool))
    assert act.unmapped_tags == 0


def test_removal_hides_in_removing_stage_and_after() -> None:
    """remove_elements emit at stage start → hidden IN that stage,
    and stay hidden in later stages and the final configuration."""
    stages = [
        _stage("g"),
        _stage("excavate", removed=[2]),   # removes a GLOBAL element
        _stage("final"),
    ]
    act = build_stage_activation(stages, TAG_TO_EID, EID_TO_CELL, N_CELLS)
    assert act is not None
    np.testing.assert_array_equal(
        act.hidden_by_name["g"], np.zeros(N_CELLS, bool),
    )
    expected = np.array([False, True, False, False])
    np.testing.assert_array_equal(act.hidden_by_name["excavate"], expected)
    np.testing.assert_array_equal(act.hidden_by_name["final"], expected)
    np.testing.assert_array_equal(act.final_hidden, expected)


def test_unmapped_tags_are_skipped_and_counted() -> None:
    """Unknown tag, sentinel fem_eid, and eid-not-in-grid all degrade
    open (no hide) and increment the diagnostic counter."""
    tag_map = {1: 10, 2: -1, 3: 999}   # 2 → sentinel, 3 → eid off-grid
    stages = [_stage("a", owned=[1, 2, 3, 77])]   # 77 → unknown tag
    act = build_stage_activation(stages, tag_map, EID_TO_CELL, N_CELLS)
    assert act is not None
    # Only tag 1 mapped; it is owned by the (single) stage so visible
    # there, and nothing else got hidden by the unmappable tags.
    np.testing.assert_array_equal(
        act.hidden_by_name["a"], np.zeros(N_CELLS, bool),
    )
    assert act.unmapped_tags == 3


def test_empty_stages_returns_none() -> None:
    assert build_stage_activation([], TAG_TO_EID, EID_TO_CELL, N_CELLS) is None


def test_duplicate_stage_names_keep_last() -> None:
    stages = [_stage("x", owned=[3]), _stage("x", owned=[4])]
    act = build_stage_activation(stages, TAG_TO_EID, EID_TO_CELL, N_CELLS)
    assert act is not None
    # Last "x" = after stage 2: everything activated.
    np.testing.assert_array_equal(
        act.hidden_by_name["x"], np.zeros(N_CELLS, bool),
    )


def test_mask_for_unmatched_name_is_none() -> None:
    act = build_stage_activation(
        [_stage("a", owned=[3])], TAG_TO_EID, EID_TO_CELL, N_CELLS,
    )
    assert act is not None
    assert act.mask_for("nope") is None
    assert act.mask_for(None) is None


# ---------------------------------------------------------------------------
# pair_capture_to_program — name pairing with positional fallback
# ---------------------------------------------------------------------------


def test_pairing_by_name_when_any_name_matches() -> None:
    capture = [("stage_0", "construction"), ("stage_1", "loading")]
    out = pair_capture_to_program(capture, ["construction", "loading"])
    assert out == {"stage_0": "construction", "stage_1": "loading"}


def test_pairing_positional_for_mpco_style_names() -> None:
    """MPCO/Ladruno capture stages are named MODEL_STAGE[<stamp>] —
    no name ever matches a program stage, so equal counts pair by
    position."""
    capture = [
        ("stage_0", "MODEL_STAGE[1]"),
        ("stage_1", "MODEL_STAGE[2]"),
        ("stage_2", "MODEL_STAGE[3]"),
    ]
    out = pair_capture_to_program(
        capture, ["gravity", "excavate", "loading"],
    )
    assert out == {
        "stage_0": "gravity",
        "stage_1": "excavate",
        "stage_2": "loading",
    }


def test_pairing_keeps_names_when_counts_differ() -> None:
    """Count mismatch (e.g. merged MODEL_STAGE groups from an old run)
    → no positional guess; the name mapping stays and unmatched
    stages render unfiltered (fail-soft)."""
    capture = [("stage_0", "MODEL_STAGE[1]"), ("stage_1", "MODEL_STAGE[2]")]
    out = pair_capture_to_program(
        capture, ["gravity", "excavate", "loading"],
    )
    assert out == {
        "stage_0": "MODEL_STAGE[1]", "stage_1": "MODEL_STAGE[2]",
    }


def test_pairing_partial_name_match_disables_positional() -> None:
    """One real name match → trust names; never positionally remap
    the rest."""
    capture = [("stage_0", "gravity"), ("stage_1", "MODEL_STAGE[2]")]
    out = pair_capture_to_program(capture, ["gravity", "excavate"])
    assert out == {"stage_0": "gravity", "stage_1": "MODEL_STAGE[2]"}


# ---------------------------------------------------------------------------
# StageActivationController — LAYER_STAGE ownership
# ---------------------------------------------------------------------------


class _FakeEV:
    """Records set_layer / clear_layer calls (ElementVisibility shape)."""

    def __init__(self) -> None:
        self.layers: dict[str, np.ndarray] = {}
        self.calls: list[tuple[str, str]] = []

    def set_layer(self, name, mask) -> None:
        self.layers[name] = np.asarray(mask, bool)
        self.calls.append(("set", name))

    def clear_layer(self, name) -> None:
        self.layers.pop(name, None)
        self.calls.append(("clear", name))


def _controller(ev, *, names=None, combined="All stages"):
    act = build_stage_activation(
        [_stage("construction", owned=[3]), _stage("loading", owned=[4])],
        TAG_TO_EID, EID_TO_CELL, N_CELLS,
    )
    assert act is not None
    names = names if names is not None else {
        "stage_000": "construction", "stage_001": "loading",
    }
    return StageActivationController(
        ev, act, stage_name_for_id=names.get, combined_stage_id=combined,
    )


def test_controller_sets_layer_on_matched_stage() -> None:
    ev = _FakeEV()
    ctrl = _controller(ev)
    ctrl.on_stage_changed("stage_000")
    np.testing.assert_array_equal(
        ev.layers[LAYER_STAGE], np.array([False, False, False, True]),
    )


def test_controller_clears_layer_on_unmatched_stage() -> None:
    ev = _FakeEV()
    ctrl = _controller(ev, names={"stage_000": "no-such-program-stage"})
    ctrl.on_stage_changed("stage_000")
    assert LAYER_STAGE not in ev.layers
    assert ("clear", LAYER_STAGE) in ev.calls


def test_controller_combined_stage_uses_final_configuration() -> None:
    ev = _FakeEV()
    ctrl = _controller(ev)
    ctrl.on_stage_changed("All stages")
    np.testing.assert_array_equal(
        ev.layers[LAYER_STAGE], np.zeros(N_CELLS, bool),
    )


def test_controller_disable_clears_and_reenable_reapplies() -> None:
    ev = _FakeEV()
    ctrl = _controller(ev)
    ctrl.on_stage_changed("stage_000")
    assert LAYER_STAGE in ev.layers
    ctrl.set_enabled(False)
    assert LAYER_STAGE not in ev.layers
    ctrl.set_enabled(True)
    np.testing.assert_array_equal(
        ev.layers[LAYER_STAGE], np.array([False, False, False, True]),
    )


def test_controller_none_stage_id_clears() -> None:
    ev = _FakeEV()
    ctrl = _controller(ev)
    ctrl.on_stage_changed(None)
    assert LAYER_STAGE not in ev.layers


# ---------------------------------------------------------------------------
# Integration — LAYER_STAGE composes with the dim filter on a real
# ElementVisibility over a real grid (ADR 0045 layered ghost model).
# ---------------------------------------------------------------------------


def test_layer_stage_composes_with_dim_filter() -> None:
    pv = pytest.importorskip("pyvista")
    from apeGmsh.viewers.core.element_visibility import (
        LAYER_DIM,
        ElementVisibility,
    )

    # Two line cells: points 0-1 and 1-2.
    cells = np.array([2, 0, 1, 2, 1, 2])
    celltypes = np.array([3, 3], dtype=np.uint8)   # VTK_LINE
    points = np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
    )
    grid = pv.UnstructuredGrid(cells, celltypes, points)
    ev = ElementVisibility(grid)

    ev.set_layer(LAYER_DIM, np.array([True, False]))
    ev.set_layer(LAYER_STAGE, np.array([False, True]))
    np.testing.assert_array_equal(
        ev.hidden_mask(), np.array([True, True]),
    )
    ev.clear_layer(LAYER_STAGE)
    np.testing.assert_array_equal(
        ev.hidden_mask(), np.array([True, False]),
    )


# ---------------------------------------------------------------------------
# End-to-end — staged Composed capture file through the real viewer:
# the LAYER_STAGE mask hides stage-owned cells on the initial stage
# and clears when the owning stage is selected.
# ---------------------------------------------------------------------------


@pytest.fixture
def staged_results(g, tmp_path):
    """Composed capture run file: Rock global, Fill activates in stage
    'construction'; capture stages named after the program stages."""
    from apeGmsh.opensees.apesees import apeSees
    from apeGmsh.results import Results
    from apeGmsh.results.capture.spec import DomainCaptureSpec

    from tests.conftest import _open_model_from_h5
    from tests.test_results_domain_capture import _FakeOps

    g.model.geometry.add_box(0, 0, 0, 1, 1, 1, label="rock")
    g.model.geometry.add_box(0, 0, 1, 1, 1, 1, label="fill")
    g.physical.add_volume("rock", name="Rock")
    g.physical.add_volume("fill", name="Fill")
    g.mesh.sizing.set_global_size(2.0)
    g.mesh.generation.generate(dim=3)
    fem = g.mesh.queries.get_fem_data(dim=3)

    ops = apeSees(fem)
    ops.model(ndm=3, ndf=3)
    mat = ops.nDMaterial.ElasticIsotropic(E=1e6, nu=0.3)
    ops.element.FourNodeTetrahedron(pg="Rock", material=mat)
    ops.element.FourNodeTetrahedron(pg="Fill", material=mat)

    def chain():
        return {
            "test":        ops.test.NormDispIncr(tol=1e-4, max_iter=50),
            "algorithm":   ops.algorithm.Newton(),
            "integrator":  ops.integrator.LoadControl(dlam=0.1),
            "constraints": ops.constraints.Plain(),
            "numberer":    ops.numberer.RCM(),
            "system":      ops.system.UmfPack(),
            "analysis":    ops.analysis.Static(),
        }

    with ops.stage(name="g") as s:
        s.analysis(**chain())
        s.run(n_increments=1)
    with ops.stage(name="construction") as s:
        s.activate(pgs=["Fill"])
        s.analysis(**chain())
        s.run(n_increments=1)
    with ops.stage(name="excavate") as s:
        s.remove_element(pg="Rock")
        s.analysis(**chain())
        s.run(n_increments=1)

    spec = DomainCaptureSpec(opensees=ops)
    spec.nodes(
        ids=[int(fem.nodes.ids[0])],
        components=["displacement_x"], name="probe",
    )
    out = tmp_path / "staged_run.h5"
    with ops.domain_capture(spec, path=str(out), ops=_FakeOps()) as cap:
        for i, name in enumerate(["g", "construction", "excavate"]):
            cap.begin_stage(name, kind="static")
            cap.step(t=float(i + 1))
            cap.end_stage()
    return Results.from_native(out, model=_open_model_from_h5(out)), fem


def test_build_from_model_on_real_staged_capture_file(staged_results):
    """The wiring's exact data path, minus Qt: a real staged Composed
    capture file → ``build_from_model`` → controller over a real scene
    and ElementVisibility. The initial 'g' stage hides exactly Fill's
    cells; 'construction' (Fill's owning stage) clears; 'excavate'
    (removes Rock) hides exactly Rock's cells."""
    from apeGmsh.viewers.core.element_visibility import ElementVisibility
    from apeGmsh.viewers.data._stage_activation import build_from_model
    from apeGmsh.viewers.scene.fem_scene import build_fem_scene

    results, fem = staged_results
    scene = build_fem_scene(results.fem)
    ev = ElementVisibility(scene.grid)

    act = build_from_model(
        results.model, scene.element_id_to_cell, int(scene.grid.n_cells),
    )
    assert act is not None
    assert act.unmapped_tags == 0

    names = {s.id: s.name for s in results.stages}
    ctrl = StageActivationController(
        ev, act, stage_name_for_id=names.get, combined_stage_id="All stages",
    )

    def _hidden_eids():
        return {
            int(scene.cell_to_element_id[i])
            for i in np.nonzero(ev.hidden_mask())[0]
        }

    def _pg_eids(pg):
        return {
            int(e) for grp in fem.elements.select(pg=pg).groups()
            for e, _conn in grp
        }

    ctrl.on_stage_changed(results.stages[0].id)   # "g"
    assert _hidden_eids() == _pg_eids("Fill")

    ctrl.on_stage_changed(results.stages[1].id)   # "construction"
    assert int(ev.hidden_mask().sum()) == 0

    # Real-file removal coverage: the writer's remove_elements surface
    # round-trips and hides Rock IN its removing stage.
    ctrl.on_stage_changed(results.stages[2].id)   # "excavate"
    assert _hidden_eids() == _pg_eids("Rock")

    # Combined-stage entry uses the final configuration (= removals).
    ctrl.on_stage_changed("All stages")
    assert _hidden_eids() == _pg_eids("Rock")

    # Vanilla guard: a model without program stages opts out entirely.
    class _Vanilla:
        def stages(self):
            return ()

    assert build_from_model(
        _Vanilla(), scene.element_id_to_cell, int(scene.grid.n_cells),
    ) is None


@pytest.mark.qt
def test_stage_activation_wiring_end_to_end(staged_results):
    pytest.importorskip("pytestqt", reason="needs pytest-qt")
    pytest.importorskip("pyvistaqt")
    _ = pytest.importorskip("qtpy.QtWidgets").QApplication.instance() \
        or pytest.importorskip("qtpy.QtWidgets").QApplication([])
    from qtpy import QtCore

    from apeGmsh.viewers.results_viewer import ResultsViewer

    results, fem = staged_results
    fill_eids = {
        int(e) for grp in fem.elements.select(pg="Fill").groups()
        for e, _conn in grp
    }
    viewer = ResultsViewer(results, title="stage-activation-smoke")
    seen: dict = {}

    def _check_then_close():
        try:
            scene = viewer._scene
            ev = scene.element_visibility
            seen["controller"] = viewer._stage_activation

            def _hidden_eids():
                return {
                    int(scene.cell_to_element_id[i])
                    for i in np.nonzero(ev.hidden_mask())[0]
                }

            # Multi-stage files open stage-LESS (director.stage_id is
            # None — no stage context): unfiltered.
            seen["initial_stage_id"] = viewer.director.stage_id
            seen["initial_hidden_count"] = int(ev.hidden_mask().sum())
            # Selecting "g" (before Fill activates): Fill hidden.
            viewer.director.set_stage("g")
            seen["g_hidden_eids"] = _hidden_eids()
            # The owning stage: everything visible.
            viewer.director.set_stage("construction")
            seen["after_hidden_count"] = int(ev.hidden_mask().sum())
            # Toggle off: layer cleared regardless of stage.
            viewer.director.set_stage("g")
            viewer._stage_activation.set_enabled(False)
            seen["disabled_hidden_count"] = int(ev.hidden_mask().sum())
        finally:
            try:
                viewer._win.window.close()
            except Exception:
                pass

    QtCore.QTimer.singleShot(400, _check_then_close)
    viewer.show()

    assert seen["controller"] is not None
    assert seen["initial_stage_id"] is None
    assert seen["initial_hidden_count"] == 0
    assert seen["g_hidden_eids"] == fill_eids
    assert seen["after_hidden_count"] == 0
    assert seen["disabled_hidden_count"] == 0
