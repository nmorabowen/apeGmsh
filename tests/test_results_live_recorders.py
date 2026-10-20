"""``spec.emit_recorders`` against a fake ops domain.

These tests never touch openseespy; a stand-in ``FakeOps`` records
every ``recorder(...)`` and ``remove(...)`` call so we can assert the
exact sequence the LiveRecorders context manager fires across
``begin_stage`` / ``end_stage`` cycles.

Real-openseespy integration tests live in
``test_results_live_recorders_integration.py`` (gated on the venv
having openseespy).
"""
from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pytest

from apeGmsh.results.spec._resolved import (
    ResolvedRecorderRecord,
    ResolvedRecorderSpec,
)

from tests.conftest import _open_model_from_h5
from tests.fixtures.split_backend import split_backend
from tests.fixtures.recorder_models import identity_model


# =====================================================================
# Fake ops module — records calls, returns sequential tags
# =====================================================================

class FakeOps:
    """Minimal ops stand-in: records ``recorder`` and ``remove`` calls."""

    def __init__(self) -> None:
        self.recorder_calls: list[tuple] = []
        self.remove_calls: list[tuple] = []
        self._next_tag = 1

    def recorder(self, *args) -> int:
        self.recorder_calls.append(tuple(args))
        tag = self._next_tag
        self._next_tag += 1
        return tag

    def remove(self, *args) -> None:
        self.remove_calls.append(tuple(args))


def _make_spec(*records: ResolvedRecorderRecord) -> ResolvedRecorderSpec:
    return ResolvedRecorderSpec(
        fem_snapshot_id="testhash",
        records=tuple(records),
    )


def _m(spec: ResolvedRecorderSpec, *, ndm: int = 3, ndf: int = 6):
    """A model for ``spec``: element ids are their own tags, and every
    recorded node carries one ``(ndm, ndf)`` layout (3-D frame default)."""
    nodes = {int(n) for r in spec.records if r.node_ids is not None
             for n in r.node_ids}
    elems = {int(e) for r in spec.records if r.element_ids is not None
             for e in r.element_ids}
    return identity_model(elems, spec.fem_snapshot_id, ndm=ndm,
                          nodes_ndf={n: ndf for n in nodes})


# =====================================================================
# Single nodal record — happy path through one stage
# =====================================================================

def test_emit_recorders_single_stage_nodal(tmp_path: Path) -> None:
    spec = _make_spec(ResolvedRecorderRecord(
        category="nodes", name="top",
        components=("displacement_x", "displacement_y", "displacement_z"),
        dt=None, n_steps=None,
        node_ids=np.array([1, 2, 3]),
    ))
    fake = FakeOps()

    with spec.emit_recorders(str(tmp_path), ops=fake, model=_m(spec)) as live:
        # __enter__ alone does NOT issue recorders anymore
        assert fake.recorder_calls == []

        live.begin_stage("gravity", kind="static")
        assert len(fake.recorder_calls) == 1

        args = fake.recorder_calls[0]
        assert args[0] == "Node"
        assert "-file" in args
        file_idx = args.index("-file")
        # Stage prefix appears in filename
        assert args[file_idx + 1].endswith("gravity__top_disp.out")
        assert args[-1] == "disp"

        # Still inside the stage — no removes yet
        assert fake.remove_calls == []

        live.end_stage()
        # end_stage flushes via ops.remove
        assert fake.remove_calls == [("recorder", 1)]

    # __exit__ has nothing to do (stage already closed)
    assert fake.remove_calls == [("recorder", 1)]
    assert len(live.stages) == 1
    assert live.stages[0].name == "gravity"
    assert live.stages[0].kind == "static"
    assert live.stages[0].tags == (1,)


def test_emit_recorders_creates_output_dir(tmp_path: Path) -> None:
    target = tmp_path / "deep" / "nested" / "out"
    spec = _make_spec(ResolvedRecorderRecord(
        category="nodes", name="r",
        components=("displacement_x",),
        dt=None, n_steps=None,
        node_ids=np.array([1]),
    ))
    fake = FakeOps()
    assert not target.exists()
    with spec.emit_recorders(str(target), ops=fake, model=_m(spec)) as live:
        live.begin_stage("s")
        live.end_stage()
    assert target.is_dir()


# =====================================================================
# Multiple stages
# =====================================================================

def test_emit_recorders_multiple_stages(tmp_path: Path) -> None:
    """Two stages produce two distinct sets of recorders, prefixed
    differently."""
    spec = _make_spec(ResolvedRecorderRecord(
        category="nodes", name="r",
        components=("displacement_x",),
        dt=None, n_steps=None,
        node_ids=np.array([1, 2]),
    ))
    fake = FakeOps()

    with spec.emit_recorders(str(tmp_path), ops=fake, model=_m(spec)) as live:
        live.begin_stage("gravity", kind="static")
        assert len(fake.recorder_calls) == 1
        gravity_args = fake.recorder_calls[0]
        live.end_stage()

        live.begin_stage("dynamic", kind="transient")
        assert len(fake.recorder_calls) == 2
        dynamic_args = fake.recorder_calls[1]
        live.end_stage()

    # Filenames carry the stage prefix
    g_file = gravity_args[gravity_args.index("-file") + 1]
    d_file = dynamic_args[dynamic_args.index("-file") + 1]
    assert g_file.endswith("gravity__r_disp.out")
    assert d_file.endswith("dynamic__r_disp.out")
    assert g_file != d_file

    # Both stages were tracked
    assert [s.name for s in live.stages] == ["gravity", "dynamic"]
    assert [s.kind for s in live.stages] == ["static", "transient"]

    # Both stages flushed
    assert fake.remove_calls == [("recorder", 1), ("recorder", 2)]


def test_emit_recorders_one_record_two_ops_types(tmp_path: Path) -> None:
    """A node record with displacement + velocity components splits
    into two ``ops.recorder`` calls (one per ops_type) within one stage."""
    spec = _make_spec(ResolvedRecorderRecord(
        category="nodes", name="mixed",
        components=("displacement_x", "velocity_x"),
        dt=None, n_steps=None,
        node_ids=np.array([1]),
    ))
    fake = FakeOps()
    with spec.emit_recorders(str(tmp_path), ops=fake, model=_m(spec)) as live:
        live.begin_stage("s")
        assert len(fake.recorder_calls) == 2
        tokens = [args[-1] for args in fake.recorder_calls]
        assert set(tokens) == {"disp", "vel"}
        live.end_stage()
    assert len(fake.remove_calls) == 2


# =====================================================================
# begin_stage / end_stage state machine
# =====================================================================

def test_begin_stage_twice_without_end_raises(tmp_path: Path) -> None:
    spec = _make_spec(ResolvedRecorderRecord(
        category="nodes", name="r",
        components=("displacement_x",),
        dt=None, n_steps=None,
        node_ids=np.array([1]),
    ))
    fake = FakeOps()
    with spec.emit_recorders(str(tmp_path), ops=fake, model=_m(spec)) as live:
        live.begin_stage("a")
        with pytest.raises(RuntimeError, match="still open"):
            live.begin_stage("b")
        live.end_stage()


def test_end_stage_without_begin_raises(tmp_path: Path) -> None:
    spec = _make_spec(ResolvedRecorderRecord(
        category="nodes", name="r",
        components=("displacement_x",),
        dt=None, n_steps=None,
        node_ids=np.array([1]),
    ))
    fake = FakeOps()
    with spec.emit_recorders(str(tmp_path), ops=fake, model=_m(spec)) as live:
        with pytest.raises(RuntimeError, match="without a matching"):
            live.end_stage()
        # Still need at least one stage so we don't trip the
        # "no stages" warning at __exit__.
        live.begin_stage("recover")
        live.end_stage()


def test_begin_stage_outside_with_block_raises(tmp_path: Path) -> None:
    spec = _make_spec(ResolvedRecorderRecord(
        category="nodes", name="r",
        components=("displacement_x",),
        dt=None, n_steps=None,
        node_ids=np.array([1]),
    ))
    fake = FakeOps()
    live = spec.emit_recorders(str(tmp_path), ops=fake, model=_m(spec))
    with pytest.raises(RuntimeError, match="inside the ``with`` block"):
        live.begin_stage("s")


def test_empty_stage_name_raises(tmp_path: Path) -> None:
    spec = _make_spec(ResolvedRecorderRecord(
        category="nodes", name="r",
        components=("displacement_x",),
        dt=None, n_steps=None,
        node_ids=np.array([1]),
    ))
    fake = FakeOps()
    with spec.emit_recorders(str(tmp_path), ops=fake, model=_m(spec)) as live:
        with pytest.raises(ValueError, match="non-empty"):
            live.begin_stage("")
        live.begin_stage("s")
        live.end_stage()


def test_stage_name_with_double_underscore_raises(tmp_path: Path) -> None:
    """Double underscore is the stage/record filename separator."""
    spec = _make_spec(ResolvedRecorderRecord(
        category="nodes", name="r",
        components=("displacement_x",),
        dt=None, n_steps=None,
        node_ids=np.array([1]),
    ))
    fake = FakeOps()
    with spec.emit_recorders(str(tmp_path), ops=fake, model=_m(spec)) as live:
        with pytest.raises(ValueError, match="separator"):
            live.begin_stage("bad__name")
        live.begin_stage("ok")
        live.end_stage()


def test_exit_with_open_stage_auto_closes(tmp_path: Path) -> None:
    """Forgetting end_stage() before leaving the with-block auto-flushes."""
    spec = _make_spec(ResolvedRecorderRecord(
        category="nodes", name="r",
        components=("displacement_x",),
        dt=None, n_steps=None,
        node_ids=np.array([1]),
    ))
    fake = FakeOps()
    with spec.emit_recorders(str(tmp_path), ops=fake, model=_m(spec)) as live:
        live.begin_stage("s")
        # ... user forgets end_stage() ...
    # __exit__ flushed it
    assert fake.remove_calls == [("recorder", 1)]
    assert len(live.stages) == 1


def test_exit_without_any_stage_warns(tmp_path: Path) -> None:
    spec = _make_spec(ResolvedRecorderRecord(
        category="nodes", name="r",
        components=("displacement_x",),
        dt=None, n_steps=None,
        node_ids=np.array([1]),
    ))
    fake = FakeOps()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with spec.emit_recorders(str(tmp_path), ops=fake, model=_m(spec)):
            pass    # no begin_stage call
    assert any(
        "without any begin_stage" in str(w.message) for w in caught
    )


# =====================================================================
# Modal records raise on __enter__
# =====================================================================

def test_modal_record_raises_at_enter(tmp_path: Path) -> None:
    spec = _make_spec(
        ResolvedRecorderRecord(
            category="nodes", name="ok",
            components=("displacement_x",),
            dt=None, n_steps=None,
            node_ids=np.array([1]),
        ),
        ResolvedRecorderRecord(
            category="modal", name="modes",
            components=(),
            dt=None, n_steps=None,
            n_modes=10,
        ),
    )
    fake = FakeOps()
    with pytest.raises(RuntimeError, match="modal") as exc:
        with spec.emit_recorders(str(tmp_path), ops=fake, model=_m(spec)):
            pass
    assert "emit_mpco" in str(exc.value)
    assert "domain_capture" not in str(exc.value)

    # Nothing should have been emitted before the raise.
    assert fake.recorder_calls == []


# =====================================================================
# Phase 1b — element-level categories within a stage
# =====================================================================

def test_emit_recorders_gauss_stress(tmp_path: Path) -> None:
    spec = _make_spec(ResolvedRecorderRecord(
        category="gauss", name="body",
        components=("stress_xx", "stress_yy"),
        dt=None, n_steps=None,
        element_ids=np.array([100, 101, 102]),
    ))
    fake = FakeOps()

    with spec.emit_recorders(str(tmp_path), ops=fake, model=identity_model([100, 101, 102], spec.fem_snapshot_id)) as live:
        live.begin_stage("dyn")

        assert len(fake.recorder_calls) == 1
        args = fake.recorder_calls[0]
        assert args[0] == "Element"
        assert "-ele" in args
        ele_idx = args.index("-ele")
        assert args[ele_idx + 1] == 100
        assert args[ele_idx + 2] == 101
        assert args[ele_idx + 3] == 102
        assert "-dof" not in args
        assert args[-1] == "stresses"

        # Filename has stage prefix
        file_idx = args.index("-file")
        assert args[file_idx + 1].endswith("dyn__body_gauss.out")

        live.end_stage()

    assert fake.remove_calls == [("recorder", 1)]


def test_emit_recorders_gauss_strain(tmp_path: Path) -> None:
    spec = _make_spec(ResolvedRecorderRecord(
        category="gauss", name="body",
        components=("strain_xx",),
        dt=None, n_steps=None,
        element_ids=np.array([1]),
    ))
    fake = FakeOps()
    with spec.emit_recorders(str(tmp_path), ops=fake, model=identity_model([1], spec.fem_snapshot_id)) as live:
        live.begin_stage("s")
        args = fake.recorder_calls[0]
        assert args[-1] == "strains"
        live.end_stage()


def test_emit_recorders_elements_global_force(tmp_path: Path) -> None:
    spec = _make_spec(ResolvedRecorderRecord(
        category="elements", name="frame",
        components=("nodal_resisting_force_x", "nodal_resisting_moment_z"),
        dt=None, n_steps=None,
        element_ids=np.array([1, 2]),
    ))
    fake = FakeOps()
    with spec.emit_recorders(str(tmp_path), ops=fake, model=identity_model([1, 2], spec.fem_snapshot_id)) as live:
        live.begin_stage("s")
        args = fake.recorder_calls[0]
        assert args[0] == "Element"
        assert args[-1] == "globalForce"
        live.end_stage()


def test_emit_recorders_line_stations_emits_pair(tmp_path: Path) -> None:
    """A line_stations record produces *two* recorders (force +
    integrationPoints) — both should be issued and removed inside one
    stage."""
    spec = _make_spec(ResolvedRecorderRecord(
        category="line_stations", name="beam",
        components=("axial_force",),
        dt=None, n_steps=None,
        element_ids=np.array([1, 2, 3]),
    ))
    fake = FakeOps()

    with spec.emit_recorders(str(tmp_path), ops=fake, model=identity_model([1, 2, 3], spec.fem_snapshot_id)) as live:
        live.begin_stage("s")
        assert len(fake.recorder_calls) == 2

        force_args = fake.recorder_calls[0]
        assert force_args[0] == "Element"
        assert "section" in force_args and "force" in force_args

        gpx_args = fake.recorder_calls[1]
        assert gpx_args[0] == "Element"
        assert "integrationPoints" in gpx_args

        live.end_stage()

    assert fake.remove_calls == [("recorder", 1), ("recorder", 2)]


def test_emit_recorders_mixed_categories(tmp_path: Path) -> None:
    """Spec with nodes + gauss + line_stations + fibers — count the
    recorders carefully."""
    spec = _make_spec(
        ResolvedRecorderRecord(
            category="nodes", name="n",
            components=("displacement_x",),
            dt=None, n_steps=None,
            node_ids=np.array([1, 2]),
        ),
        ResolvedRecorderRecord(
            category="gauss", name="g",
            components=("stress_xx",),
            dt=None, n_steps=None,
            element_ids=np.array([10]),
        ),
        ResolvedRecorderRecord(
            category="line_stations", name="ls",
            components=("bending_moment_z",),
            dt=None, n_steps=None,
            element_ids=np.array([20]),
        ),
        ResolvedRecorderRecord(
            category="fibers", name="fib",
            components=("fiber_stress",),
            dt=None, n_steps=None,
            element_ids=np.array([30]),
            element_class_name="ForceBeamColumn3d",
        ),
    )
    fake = FakeOps()
    with spec.emit_recorders(str(tmp_path), ops=fake, model=_m(spec)) as live:
        live.begin_stage("s")
        # 1 (nodes) + 1 (gauss) + 2 (line_stations pair) + 2 (fibres pair)
        assert len(fake.recorder_calls) == 6
        assert live.tags == (1, 2, 3, 4, 5, 6)
        assert fake.recorder_calls[4][-2:] == ("section", "fiberData2")
        assert fake.recorder_calls[5][-1] == "integrationPoints"
        live.end_stage()
    assert len(fake.remove_calls) == 6


# =====================================================================
# Robustness
# =====================================================================

def test_remove_failure_warns_does_not_raise(tmp_path: Path) -> None:
    """If ops.remove raises (e.g. recorder already removed by user code),
    we warn rather than propagate — the analysis result is more
    important than perfect cleanup."""

    class BrokenRemove(FakeOps):
        def remove(self, *args):
            raise RuntimeError("simulated remove failure")

    spec = _make_spec(ResolvedRecorderRecord(
        category="nodes", name="r",
        components=("displacement_x",),
        dt=None, n_steps=None,
        node_ids=np.array([1]),
    ))
    fake = BrokenRemove()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with spec.emit_recorders(str(tmp_path), ops=fake, model=_m(spec)) as live:
            live.begin_stage("s")
            live.end_stage()
    assert any("failed to remove recorder" in str(w.message) for w in caught)


def test_xml_format(tmp_path: Path) -> None:
    spec = _make_spec(ResolvedRecorderRecord(
        category="nodes", name="r",
        components=("displacement_x",),
        dt=None, n_steps=None,
        node_ids=np.array([1]),
    ))
    fake = FakeOps()
    with spec.emit_recorders(str(tmp_path), file_format="xml", ops=fake, model=_m(spec)) as live:
        live.begin_stage("s")
        args = fake.recorder_calls[0]
        assert "-xml" in args
        assert "-file" not in args
        xml_idx = args.index("-xml")
        assert args[xml_idx + 1].endswith("s__r_disp.xml")
        live.end_stage()


def test_empty_node_ids_emits_nothing(tmp_path: Path) -> None:
    spec = _make_spec(ResolvedRecorderRecord(
        category="nodes", name="r",
        components=("displacement_x",),
        dt=None, n_steps=None,
        node_ids=np.array([], dtype=np.int64),
    ))
    fake = FakeOps()
    with spec.emit_recorders(str(tmp_path), ops=fake, model=_m(spec)) as live:
        live.begin_stage("s")
        assert fake.recorder_calls == []
        live.end_stage()


def test_single_use(tmp_path: Path) -> None:
    spec = _make_spec(ResolvedRecorderRecord(
        category="nodes", name="r",
        components=("displacement_x",),
        dt=None, n_steps=None,
        node_ids=np.array([1]),
    ))
    fake = FakeOps()
    live = spec.emit_recorders(str(tmp_path), ops=fake, model=_m(spec))
    with live:
        live.begin_stage("s")
        live.end_stage()
    with pytest.raises(RuntimeError, match="single-use"):
        with live:
            pass


def test_refuses_layers_record_by_name(tmp_path: Path) -> None:
    """A layers record raises at ``__enter__``, before any recorder, by
    name, pointing at the routes that record layers."""
    spec = _make_spec(
        ResolvedRecorderRecord(
            category="nodes", name="ok",
            components=("displacement_x",),
            dt=None, n_steps=None,
            node_ids=np.array([1]),
        ),
        ResolvedRecorderRecord(
            category="layers", name="non_recorder",
            components=("fiber_stress",),
            dt=None, n_steps=None,
            element_ids=np.array([100]),
        ),
    )
    fake = FakeOps()
    with pytest.raises(ValueError) as info:
        with spec.emit_recorders(str(tmp_path), ops=fake, model=_m(spec)):
            pass
    m = str(info.value)
    assert ("'non_recorder'" in m and "category='layers'" in m
            and "emit_mpco" in m and "emit_default" in m
            and "domain_capture" not in m), m
    assert fake.recorder_calls == []


# =====================================================================
# Integration with Results.from_recorders — file path matching
# =====================================================================

def test_emit_filename_matches_from_recorders_lookup(tmp_path: Path) -> None:
    """The filename LiveRecorders writes to MUST match what
    Results.from_recorders(stage_id=..., model=_open_model_from_h5(stage_id=...)) looks for. Tested via the
    shared helpers (no actual ops or transcoder run)."""
    from apeGmsh.results.writers._cache import list_source_files

    spec = _make_spec(ResolvedRecorderRecord(
        category="nodes", name="r",
        components=("displacement_x",),
        dt=None, n_steps=None,
        node_ids=np.array([1]),
    ))
    fake = FakeOps()
    with spec.emit_recorders(str(tmp_path), ops=fake, model=_m(spec)) as live:
        live.begin_stage("gravity")
        emit_path = fake.recorder_calls[0][
            fake.recorder_calls[0].index("-file") + 1
        ]
        live.end_stage()

    # What the read side would look for
    expected = list_source_files(spec, tmp_path, stage_id="gravity")
    assert len(expected) == 1
    # Path equivalence (forward-slash from emit, OS-native from list_source_files)
    assert Path(emit_path) == expected[0]


# =====================================================================
# List-typed IDs (e.g. ``fem.nodes.select(...).ids``) are coerced
# =====================================================================

def test_resolved_record_coerces_list_ids_to_ndarray(tmp_path: Path) -> None:
    rec = ResolvedRecorderRecord(
        category="nodes", name="top",
        components=("displacement_x",),
        dt=None, n_steps=None,
        node_ids=[1, 2, 3],
    )
    assert isinstance(rec.node_ids, np.ndarray)
    assert rec.node_ids.tolist() == [1, 2, 3]

    erec = ResolvedRecorderRecord(
        category="elements", name="beams",
        components=("nodal_resisting_force_x",),
        dt=None, n_steps=None,
        element_ids=(10, 11),
    )
    assert isinstance(erec.element_ids, np.ndarray)

    fake = FakeOps()
    spec = _make_spec(rec)
    with spec.emit_recorders(str(tmp_path), ops=fake,
                             model=_m(spec)) as live:
        live.begin_stage("gravity", kind="static")
        live.end_stage()
    assert len(fake.recorder_calls) == 1


# =====================================================================
# Without ops=: the module the bridge drives, never openseespy by name
# =====================================================================

def _nodal_spec() -> ResolvedRecorderSpec:
    return _make_spec(ResolvedRecorderRecord(
        category="nodes", name="r",
        components=("displacement_x",),
        dt=None, n_steps=None,
        node_ids=np.array([1, 2]),
    ))


def test_defaults_to_the_module_the_bridge_drives(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    bound, stray = FakeOps(), FakeOps()
    split_backend(monkeypatch, bound, stray)
    spec = _nodal_spec()
    with spec.emit_recorders(str(tmp_path), model=_m(spec)) as live:
        live.begin_stage("gravity")
        live.end_stage()
    assert len(bound.recorder_calls) == 1
    assert bound.remove_calls == [("recorder", 1)]
    assert stray.recorder_calls == []


def test_an_explicit_ops_beats_the_resolver(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    bound, stray, mine = FakeOps(), FakeOps(), FakeOps()
    split_backend(monkeypatch, bound, stray)
    spec = _nodal_spec()
    with spec.emit_recorders(str(tmp_path), ops=mine,
                             model=_m(spec)) as live:
        live.begin_stage("gravity")
        live.end_stage()
    assert len(mine.recorder_calls) == 1
    assert bound.recorder_calls == [] and stray.recorder_calls == []


# =====================================================================
# n_steps= without dt= warns at begin_stage for every category (#1947)
# =====================================================================

_CADENCE_COMPONENTS = {
    "nodes": ("displacement_x",),
    "elements": ("nodal_resisting_force_x",),
    "gauss": ("stress_xx",),
    "line_stations": ("axial_force",),
}


def _cadence_spec(
    category: str, *, dt: float | None, n_steps: int | None,
) -> ResolvedRecorderSpec:
    ids = np.array([1, 2])
    return _make_spec(ResolvedRecorderRecord(
        category=category, name=f"cad_{category}",
        components=_CADENCE_COMPONENTS[category],
        dt=dt, n_steps=n_steps,
        node_ids=ids if category == "nodes" else None,
        element_ids=None if category == "nodes" else ids,
    ))


def _cadence_begin_stage(
    out_dir: Path, spec: ResolvedRecorderSpec,
) -> tuple[list, list[tuple]]:
    """Run one begin_stage; return (warnings it raised, recorder calls)."""
    fake = FakeOps()
    model = _m(spec)
    live = spec.emit_recorders(str(out_dir), ops=fake, model=model)
    # __enter__ pre-translates quietly (#1980), so under
    # ``-W error::UserWarning`` a cadence warning here fails the test.
    live.__enter__()
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            live.begin_stage("s")
        live.end_stage()
    finally:
        live.__exit__(None, None, None)
    return caught, fake.recorder_calls


def _strip_dir(calls: list[tuple], out_dir: Path) -> list[tuple]:
    root = str(out_dir)
    return [
        tuple(a.replace(root, "<out>") if isinstance(a, str) else a
              for a in c)
        for c in calls
    ]


@pytest.mark.parametrize("category", list(_CADENCE_COMPONENTS))
def test_begin_stage_n_steps_without_dt_warns_once(
    tmp_path: Path, category: str,
) -> None:
    a, b = tmp_path / "a", tmp_path / "b"
    caught, calls = _cadence_begin_stage(
        a, _cadence_spec(category, dt=None, n_steps=5),
    )
    user = [w for w in caught if issubclass(w.category, UserWarning)]
    assert len(user) == 1, [str(w.message) for w in caught]
    msg = str(user[0].message)
    assert f"'cad_{category}'" in msg and "n_steps=5" in msg
    # Same recorder calls as the record without n_steps= (the output
    # directory aside).
    _, plain = _cadence_begin_stage(
        b, _cadence_spec(category, dt=None, n_steps=None),
    )
    assert calls and _strip_dir(calls, a) == _strip_dir(plain, b)
    assert all("-dT" not in c for c in calls)


@pytest.mark.parametrize("category", list(_CADENCE_COMPONENTS))
def test_begin_stage_n_steps_with_dt_emits_dT_silently(
    tmp_path: Path, category: str,
) -> None:
    caught, calls = _cadence_begin_stage(
        tmp_path, _cadence_spec(category, dt=0.01, n_steps=5),
    )
    assert [w for w in caught if issubclass(w.category, UserWarning)] == []
    assert calls
    for c in calls:
        assert c[c.index("-dT") + 1] == 0.01


# =====================================================================
# Node DOFs follow the node layout; model= is required (#2018)
# =====================================================================

def _dof_list(call: tuple) -> list[int]:
    i = call.index("-dof") + 1
    return list(call[i:call.index("-precision")])


@pytest.mark.parametrize(("ndm", "ndf", "want"), [
    (2, 3, [1, 2, 3]),          # planar frame: rotation_z is DOF 3
    (3, 6, [1, 2, 6]),          # 3-D frame: unchanged
])
def test_live_route_records_rotation_z_at_the_node_dof(
    tmp_path: Path, ndm: int, ndf: int, want: list[int],
) -> None:
    spec = _make_spec(ResolvedRecorderRecord(
        category="nodes", name="f",
        components=("displacement_x", "displacement_y", "rotation_z"),
        dt=None, n_steps=None, node_ids=np.array([1, 2]),
    ))
    fake = FakeOps()
    with spec.emit_recorders(str(tmp_path), ops=fake,
                             model=_m(spec, ndm=ndm, ndf=ndf)) as live:
        live.begin_stage("s")
        live.end_stage()
    [call] = fake.recorder_calls
    assert _dof_list(call) == want


@pytest.mark.parametrize("components", [
    ("displacement_x",), ("displacement_x", "rotation_z"),
])
def test_live_route_refuses_node_records_without_model(
    tmp_path: Path, components: tuple[str, ...],
) -> None:
    """Refused on ``__enter__``, before any recorder reaches the domain."""
    spec = _make_spec(ResolvedRecorderRecord(
        category="nodes", name="f", components=components,
        dt=None, n_steps=None, node_ids=np.array([1, 2]),
    ))
    fake = FakeOps()
    with pytest.raises(ValueError, match=r"Pass model="):
        with spec.emit_recorders(str(tmp_path), ops=fake):
            pass
    assert fake.recorder_calls == []
