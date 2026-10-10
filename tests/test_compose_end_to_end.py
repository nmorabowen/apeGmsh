"""End-to-end compose tests — Phase 3B.2c / ADR 0038, on the ADR 0117
``Assembly`` (v2) form.

Locks the merge engine as ``Assembly.bridge()`` drives it: FILTER-warning
emission at ``bridge()``, the module pattern-field (not namespaced), and
the ``compose_inspect`` / ``compose_list`` readers on an assembly archive
(``Assembly.h5`` + ``apeGmsh.from_h5``).

Every source is a small hand-built FEMData written with
``apeSees(fem).model(...)`` + ``ops.h5`` (``bridge()`` refuses a source
with no ``/opensees`` model) — no live Gmsh session is required.
"""
from __future__ import annotations

import warnings
from pathlib import Path

import h5py
import numpy as np
import pytest

from apeGmsh._core import apeGmsh
from apeGmsh._kernel.records._loads import NodalLoadRecord
from apeGmsh.assembly import Assembly
from apeGmsh.mesh._compose import ComposeFilterWarning
from apeGmsh.mesh._element_types import ElementGroup, make_type_info
from apeGmsh.mesh._group_set import LabelSet, PhysicalGroupSet
from apeGmsh.mesh.FEMData import (
    ElementComposite,
    FEMData,
    MeshInfo,
    NodeComposite,
)
from apeGmsh.opensees import apeSees


# ---------------------------------------------------------------------------
# Fixture builders
# ---------------------------------------------------------------------------


def _make_module_fem(
    *,
    node_ids: "np.ndarray | None" = None,
    elem_ids: "np.ndarray | None" = None,
    node_pgs: "dict | None" = None,
    nodal_loads: "list | None" = None,
    masses: "list | None" = None,
) -> FEMData:
    """Small FEMData with a single Line2 element type."""
    if node_ids is None:
        node_ids = np.array([1, 2, 3], dtype=np.int64)
    if elem_ids is None:
        elem_ids = np.array([10, 11], dtype=np.int64)

    n = node_ids.size
    node_coords = np.array(
        [[float(i), 0.0, 0.0] for i in range(n)],
        dtype=np.float64,
    )
    line_info = make_type_info(
        code=1, gmsh_name="Line 2", dim=1, order=1, npe=2,
        count=elem_ids.size,
    )
    conn_rows = []
    for i in range(elem_ids.size):
        a = int(node_ids[i % n])
        b = int(node_ids[(i + 1) % n])
        conn_rows.append([a, b])
    conn = np.array(conn_rows, dtype=np.int64)
    line_group = ElementGroup(
        element_type=line_info, ids=elem_ids, connectivity=conn,
    )

    nodes = NodeComposite(
        node_ids=node_ids,
        node_coords=node_coords,
        physical=PhysicalGroupSet(node_pgs or {}),
        labels=LabelSet({}),
        loads=nodal_loads,
        masses=masses,
    )
    elements = ElementComposite(
        groups={1: line_group},
        physical=PhysicalGroupSet({}),
        labels=LabelSet({}),
    )
    info = MeshInfo(
        n_nodes=n, n_elems=elem_ids.size, bandwidth=1,
        types=[line_info],
    )
    return FEMData(nodes=nodes, elements=elements, info=info)


def _save_module(fem: FEMData, path: Path) -> Path:
    """Write ``fem`` as an instanceable source (``/opensees`` model 3/3)."""
    ops = apeSees(fem)
    ops.model(ndm=3, ndf=3)
    ops.h5(str(path))
    return path


@pytest.fixture
def module_a_h5(tmp_path: Path) -> Path:
    """Saved module A — node ids 1..3, elem ids 10..11."""
    fem = _make_module_fem(
        node_ids=np.array([1, 2, 3], dtype=np.int64),
        elem_ids=np.array([10, 11], dtype=np.int64),
    )
    return _save_module(fem, tmp_path / "module_a.h5")


@pytest.fixture
def module_b_h5(tmp_path: Path) -> Path:
    """Saved module B with a slightly different tag range."""
    fem = _make_module_fem(
        node_ids=np.array([1, 2, 3, 4], dtype=np.int64),
        elem_ids=np.array([20, 21, 22], dtype=np.int64),
    )
    return _save_module(fem, tmp_path / "module_b.h5")


def _filter_warnings(caught) -> list[str]:
    return [
        str(w.message) for w in caught
        if issubclass(w.category, ComposeFilterWarning)
    ]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_compose_filter_warning_for_stages(tmp_path: Path) -> None:
    """Source H5 carrying ``/opensees/stages/...`` emits one
    :class:`ComposeFilterWarning` per kind (stages = 1) at ``bridge()``.

    Recorders / analysis-settings stay silent.
    """
    from tests.opensees.h5.test_h5_stages_writer import _chain

    # A real one-stage source (the reader refuses a hand-made stage stub).
    ops = apeSees(_make_module_fem())
    ops.model(ndm=3, ndf=3)
    with ops.stage(name="only") as s:
        s.analysis(**_chain(ops))
        s.run(n_increments=1)
    src_path = tmp_path / "module_with_stages.h5"
    ops.h5(str(src_path))
    # Hand-inject a /opensees/recorders/ sub-group → must stay silent.
    with h5py.File(str(src_path), "a") as f:
        assert "stages" in f["opensees"]
        assert "time_series" not in f["opensees"]
        f["opensees"].create_group("recorders").create_group("rec_0")

    asm = Assembly("stages").instance("A", src_path)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        asm.bridge(ndm=3, ndf=3)

    relevant = _filter_warnings(caught)
    # Exactly one stages warning, no recorder warning.
    assert len(relevant) == 1, relevant
    assert "stages" in relevant[0].lower()


def test_compose_pattern_field_not_namespaced(tmp_path: Path) -> None:
    """Regression for 3B.2a's pattern namespacing — patterns are
    FILTER-verdict, the bridge owns the pattern name."""
    # Module with a nodal load on a known pattern name.
    fem = _make_module_fem(
        nodal_loads=[
            NodalLoadRecord(
                node_id=2,
                force_xyz=(1.0, 0.0, 0.0),
                pattern="dead",
                name=None,
            ),
        ],
    )
    src = _save_module(fem, tmp_path / "module_with_pattern.h5")

    ops = Assembly("pattern").instance("A", src).bridge(ndm=3, ndf=3)

    loads = list(ops.fem.nodes.loads)
    assert len(loads) == 1
    # Pattern field must remain "dead" — NOT "A.dead".
    assert loads[0].pattern == "dead"


def test_compose_inspect_after_compose(
    module_a_h5: Path, tmp_path: Path,
) -> None:
    """``compose_inspect`` works on a session opened on an assembly
    archive (metadata-only read of the source)."""
    asm = Assembly("inspect").instance("A", module_a_h5)
    asm.bridge(ndm=3, ndf=3)
    archive = tmp_path / "archive.h5"
    asm.h5(str(archive))

    g = apeGmsh.from_h5(archive)
    info = g.compose_inspect(module_a_h5)
    assert "neutral_schema_version" in info
    # The source is uncomposed (it's the module file).
    assert info["composed_from"] == ()


def test_compose_list_returns_modules(
    module_a_h5: Path, module_b_h5: Path, tmp_path: Path,
) -> None:
    """``g.compose_list()`` returns the composed modules in label order."""
    asm = Assembly("list")
    asm.instance("A", module_a_h5)
    asm.instance("B", module_b_h5, translate=(100.0, 0.0, 0.0))
    asm.bridge(ndm=3, ndf=3)
    archive = tmp_path / "archive.h5"
    asm.h5(str(archive))

    modules = apeGmsh.from_h5(archive).compose_list()
    assert len(modules) == 2
    assert [m.label for m in modules] == ["A", "B"]


def test_from_h5_session_compose_workflow(
    module_a_h5: Path, tmp_path: Path,
) -> None:
    """The full instance → bridge → archive → reload workflow runs cleanly."""
    asm = Assembly("workflow").instance("A", module_a_h5)
    asm.bridge(ndm=3, ndf=3)
    out = tmp_path / "final.h5"
    asm.h5(str(out))
    # File exists + reloads correctly.
    assert out.exists()
    reloaded = FEMData.from_h5(str(out))
    assert "A" in reloaded.composed_from


# ---------------------------------------------------------------------------
# ADR 0055 Phase 3 — staged-archive FILTER+warn with a REAL staged source
# (the hand-injected fixture above predates real staged archives; since
# Phase 2/5 `ops.h5` writes them, the e2e contract is locked here).
# ---------------------------------------------------------------------------


def _write_real_staged_archive(tmp_path: Path) -> Path:
    """A real `ops.h5`-written staged archive (two stages, a stage fix,
    a stage pattern, a Linear time-series) on the two-quad mesh.

    It declares no element spec: a ``FourNodeQuad`` source is refused by
    ``bridge()`` (it is not rehydrated, ADR 0117 D4); the quad is
    declared on the bridge by dotted PG instead.
    """
    from tests.opensees.h5.test_h5_stages_reader import build_two_quad_fem
    from tests.opensees.h5.test_h5_stages_writer import _chain

    ops = apeSees(build_two_quad_fem(), default_orientation=None)
    ops.model(ndm=2, ndf=2)
    ops.fix(pg="Base", dofs=(1, 1))
    with ops.stage(name="construction") as s:
        s.fix(pg="FillTop", dofs=(1, 0))
        s.analysis(**_chain(ops))
        s.run(n_increments=5)
    with ops.stage(name="loading") as s:
        ts = ops.timeSeries.Linear()
        with s.pattern(series=ts) as p:
            p.load(pg="Fill", forces=(10.0, 0.0))
        s.analysis(**_chain(ops))
        s.run(n_increments=3, dt=0.01)

    src = tmp_path / "staged_module.h5"
    ops.h5(str(src))
    return src


def test_compose_filters_real_staged_archive(tmp_path: Path) -> None:
    """Instancing a REAL staged archive (ADR 0055 Phase 2/5 writer
    output) warns once per droppable kind at ``bridge()`` and the
    assembly archive carries ZERO ``/opensees/stages`` bytes — the staged
    program is never inherited (ADR 0038 §"Merge semantics" FILTER
    verdict; ADR 0055 Phasing #3)."""
    src = _write_real_staged_archive(tmp_path)
    # Pre-condition: the source genuinely carries a staged program.
    with h5py.File(str(src), "r") as f:
        assert "stages" in f["opensees"]
        assert "time_series" in f["opensees"]

    asm = Assembly("staged")
    asm.instance("staged_mod", src, translate=(50.0, 0.0, 0.0))
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        ops = asm.bridge(ndm=2, ndf=2)
    # The quad travels on the bridge, by dotted PG (row 47).
    mat = ops.nDMaterial.ElasticIsotropic(E=1e6, nu=0.3, rho=0.0)
    ops.element.FourNodeQuad(
        pg="staged_mod.Rock", thickness=1.0, material=mat)
    out = tmp_path / "composed.h5"
    asm.h5(str(out))

    msgs = _filter_warnings(caught)
    # Exactly one warning per droppable kind present on this source:
    # stages + time-series (the stage pattern itself is captured inside
    # the stage bucket, NOT under /opensees/patterns).
    assert len([m for m in msgs if "stages" in m]) == 1, msgs
    assert len([m for m in msgs if "time-series" in m]) == 1, msgs
    assert len(msgs) == 2, msgs

    # The archive inherits NOTHING from the staged program.
    with h5py.File(str(out), "r") as f:
        if "opensees" in f:
            assert "stages" not in f["opensees"]
            assert "time_series" not in f["opensees"]


def test_compose_inspect_filtered_audit(
    module_a_h5: Path, tmp_path: Path,
) -> None:
    """``compose_inspect`` surfaces the filtered-audit (ADR 0038
    §195-196 / ADR 0055 Phase 3): droppable warn-kind counts for a
    staged source, empty dict for a vanilla module."""
    src = _write_real_staged_archive(tmp_path)
    with h5py.File(str(src), "r") as f:
        expected_stages = len(f["opensees"]["stages"].keys())
        expected_ts = len(f["opensees"]["time_series"].keys())

    g = apeGmsh.from_h5(module_a_h5)
    info = g.compose_inspect(src)
    assert info["filtered"] == {
        "stages": expected_stages,
        "time_series": expected_ts,
    }
    # Vanilla module: nothing droppable.
    assert g.compose_inspect(module_a_h5)["filtered"] == {}
    g.end()
