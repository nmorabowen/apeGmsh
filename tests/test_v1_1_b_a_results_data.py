"""Compose v1.1-B slice A — results.viewer data-layer module mappings.

Proves that the per-entity compose-module-label mapping shipped in PR
#372 (Phase 3F.2a — mesh.viewer side) is **transparently available to
``results.viewer``** through the shared :class:`ViewerData` data layer.

Survey finding for slice B.a
----------------------------
``Results.viewer()`` constructs a :class:`ResultsViewer` whose
:meth:`_build_viewer_data` calls either:

* :meth:`ViewerData.from_h5` (when ``resolve_orientation_source``
  returns a path carrying ``/opensees/transforms`` +
  ``/opensees/element_meta``), or
* :meth:`ViewerData.from_fem` (otherwise — the typical composed-
  results landing).

Both factories were extended in PR #372 with
``module_by_eid`` / ``module_by_nid`` mappings on the resulting
:class:`ViewerElements` / :class:`ViewerNodes`.  Since
``mesh.viewer`` and ``results.viewer`` share this **single** data
layer, the module-label surface is already live on the results-side
viewer too — no separate plumbing required.

This file is the explicit lock that the data-flow stays unified.  The
follow-up slices (3F.2-style B.b controller + B.c UI) wire the existing
data into the results-side ColorMode dropdown.

Why these tests matter
----------------------
1. **Regression guard**: if anyone refactors ``results_viewer.py``
   or :class:`ResultsViewer._build_viewer_data` to fork off a
   separate data-layer construction path, these tests fail —
   forcing the refactor to keep the mapping wired.
2. **Cross-viewer parity**: the same composed FEMData yields the
   same module labels through both ``mesh.viewer``'s
   ``ViewerData.from_fem`` path and ``results.viewer``'s
   ``ResultsViewer._build_viewer_data`` path.
3. **Layering invariant**: no imports from ``apeGmsh.mesh`` are
   added under ``viewers/``; covered by the existing
   ``test_viewers_pure_h5_consumer.py`` AST walk (the assertion here
   is a docstring marker — the actual enforcement is centralised).
"""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Optional

import numpy as np
import pytest

from apeGmsh.mesh._element_types import ElementGroup, make_type_info
from apeGmsh.mesh._group_set import LabelSet, PhysicalGroupSet
from apeGmsh.mesh.FEMData import (
    ElementComposite,
    FEMData,
    MeshInfo,
    NodeComposite,
)
from apeGmsh.viewers.data import ViewerData
from apeGmsh.viewers.results_viewer import ResultsViewer


# ---------------------------------------------------------------------------
# Fixture builders — mirrors tests/test_phase_3f_2a_data.py shape
# ---------------------------------------------------------------------------


def _make_module_fem(
    *,
    node_ids: "np.ndarray | None" = None,
    elem_ids: "np.ndarray | None" = None,
) -> FEMData:
    """Tiny single-line-element FEMData with no compose state."""
    if node_ids is None:
        node_ids = np.array([1, 2, 3], dtype=np.int64)
    if elem_ids is None:
        elem_ids = np.array([10, 11], dtype=np.int64)

    n = node_ids.size
    if n > 0:
        node_coords = np.array(
            [[float(i), 0.0, 0.0] for i in range(n)],
            dtype=np.float64,
        )
    else:
        node_coords = np.zeros((0, 3), dtype=np.float64)
    line_info = make_type_info(
        code=1, gmsh_name="Line 2", dim=1, order=1, npe=2,
        count=elem_ids.size,
    )
    if elem_ids.size > 0:
        conn_rows = []
        for i in range(elem_ids.size):
            a = int(node_ids[i % n])
            b = int(node_ids[(i + 1) % n])
            conn_rows.append([a, b])
        conn = np.array(conn_rows, dtype=np.int64)
    else:
        conn = np.zeros((0, 2), dtype=np.int64)
    line_group = ElementGroup(
        element_type=line_info, ids=elem_ids, connectivity=conn,
    )
    nodes = NodeComposite(
        node_ids=node_ids,
        node_coords=node_coords,
        physical=PhysicalGroupSet({}),
        labels=LabelSet({}),
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


@pytest.fixture
def uncomposed_fem() -> FEMData:
    """Bare FEMData — no compose state at all."""
    return _make_module_fem()


def _write_source(fem: FEMData, path: Path) -> Path:
    """Write ``fem`` as an assembly source (it needs ``model(ndm, ndf)``)."""
    from apeGmsh.opensees import apeSees

    ops = apeSees(fem)
    ops.model(ndm=3, ndf=3)
    ops.h5(str(path))
    return path


@pytest.fixture
def composed_fem(tmp_path: Path) -> FEMData:
    """Host + 2 modules, merged by a v2 assembly -> return the merged
    FEMData that ``Results.fem`` would hand to the results.viewer.

    Pattern mirrors the ``composed_h5`` fixture in
    tests/test_phase_3f_2a_data.py but exposes the FEMData directly
    (no save/reload) since the results.viewer path consumes
    ``results.fem`` after the bind step. v2 has no live host (row-15
    ruling), so the host is an instance named ``host``.
    """
    from apeGmsh.assembly import Assembly

    host_path = _write_source(_make_module_fem(
        node_ids=np.array([1, 2, 3], dtype=np.int64),
        elem_ids=np.array([10, 11], dtype=np.int64),
    ), tmp_path / "host.h5")
    module_a_path = _write_source(_make_module_fem(
        node_ids=np.array([1, 2, 3], dtype=np.int64),
        elem_ids=np.array([10, 11], dtype=np.int64),
    ), tmp_path / "module_a.h5")
    module_b_path = _write_source(_make_module_fem(
        node_ids=np.array([1, 2, 3, 4], dtype=np.int64),
        elem_ids=np.array([20, 21, 22], dtype=np.int64),
    ), tmp_path / "module_b.h5")

    asm = Assembly("composed")
    asm.instance("host", host_path)
    asm.instance("A", module_a_path, translate=(10.0, 0.0, 0.0))
    asm.instance("B", module_b_path, translate=(100.0, 0.0, 0.0))
    # ``bridge(...).fem`` is the merged FEMData broker (ADR 0117). This
    # is what ``Results.fem`` would yield after a bind against a
    # results file from the same composed model.
    fem = asm.bridge(ndm=3, ndf=3).fem
    assert fem is not None
    return fem


# ---------------------------------------------------------------------------
# Results stub — minimal surface ResultsViewer.__init__ touches
# ---------------------------------------------------------------------------


class _ResultsStub:
    """Minimal Results stand-in for :class:`ResultsViewer.__init__`.

    Same shape as ``tests/viewers/test_viewer_orientation_from_model_h5.py``
    — keeps ResultsViewer construction off Qt and off OpenGL so the
    headless harness (no GPU on this machine per
    ``feedback_viewer_no_gpu``) can still exercise
    :meth:`ResultsViewer._build_viewer_data`.

    Reads:
      * ``results.fem`` — must be non-None.
      * ``results._path`` — the file-mediated read source.
      * ``results.model`` — touched only as a presence gate.
    """

    def __init__(
        self,
        *,
        fem: object,
        path: Optional[Path] = None,
        model: Optional[object] = None,
    ) -> None:
        self.fem = fem
        self._path = path
        self.model = model


# =====================================================================
# Core contract — results.viewer shares mesh.viewer's data layer
# =====================================================================


class TestResultsViewerSharesDataLayer:
    """``ResultsViewer._build_viewer_data`` produces the same
    :class:`ViewerData` shape as a direct ``ViewerData.from_fem`` —
    so the module-label surface PR #372 added is already on the
    results-side viewer."""

    def test_results_viewer_build_uses_viewer_data(
        self, composed_fem: FEMData,
    ) -> None:
        """``_build_viewer_data`` returns the exact ViewerData class
        — the construction path is shared with mesh.viewer."""
        stub = _ResultsStub(fem=composed_fem, path=None)
        viewer = ResultsViewer(stub)

        view = viewer._build_viewer_data()
        assert isinstance(view, ViewerData)

    def test_module_labels_reach_results_viewer(
        self, composed_fem: FEMData,
    ) -> None:
        """The composed FEMData's module labels are populated on the
        ViewerElements / ViewerNodes that results.viewer would
        consume — verifying PR #372's data layer is transparently
        live on results-side."""
        stub = _ResultsStub(fem=composed_fem, path=None)
        viewer = ResultsViewer(stub)

        view = viewer._build_viewer_data()
        # source_kind="fem" — no /opensees/ orientation zone, so the
        # resolver fell back to from_fem. This is the typical
        # composed-results landing.
        assert view.source_kind == "fem"

        # Both surfaces report has_modules True — at least one
        # module-owned entity is present in the composed FEMData.
        assert view.elements.has_modules is True
        assert view.nodes.has_modules is True

    def test_results_viewer_matches_direct_from_fem(
        self, composed_fem: FEMData,
    ) -> None:
        """Cross-reference: the module labels seen by results.viewer
        through ``_build_viewer_data`` are IDENTICAL to what a direct
        ``ViewerData.from_fem(fem)`` (the mesh.viewer side) produces
        on the same FEMData.  Locks the shared-data-layer claim."""
        stub = _ResultsStub(fem=composed_fem, path=None)
        viewer = ResultsViewer(stub)

        view_results = viewer._build_viewer_data()
        view_mesh = ViewerData.from_fem(composed_fem)

        # Same has_modules verdict.
        assert (
            view_results.elements.has_modules
            == view_mesh.elements.has_modules
        )
        assert (
            view_results.nodes.has_modules
            == view_mesh.nodes.has_modules
        )

        # Same per-element labels.
        for group in view_mesh.elements:
            for eid in group.ids:
                assert (
                    view_results.elements.module_for(int(eid))
                    == view_mesh.elements.module_for(int(eid))
                )

        # Same per-node labels.
        for nid in view_mesh.nodes.ids:
            assert (
                view_results.nodes.module_for(int(nid))
                == view_mesh.nodes.module_for(int(nid))
            )


# =====================================================================
# Degradation — uncomposed FEMData → has_modules False
# =====================================================================


class TestResultsViewerUncomposedDegrade:
    """Uncomposed FEMData → ``has_modules == False`` on the results
    side, matching the mesh.viewer contract."""

    def test_uncomposed_fem_has_no_modules(
        self, uncomposed_fem: FEMData,
    ) -> None:
        stub = _ResultsStub(fem=uncomposed_fem, path=None)
        viewer = ResultsViewer(stub)

        view = viewer._build_viewer_data()
        assert view.elements.has_modules is False
        assert view.nodes.has_modules is False
        # ``module_for`` returns None for every id.
        assert view.elements.module_for(10) is None
        assert view.nodes.module_for(1) is None


# =====================================================================
# Specific label content — composed labels match expectation
# =====================================================================


class TestResultsViewerModuleLabelContent:
    """The actual label strings the results-side viewer surfaces are
    the joined compose labels (PR #369 / 3E.1)."""

    def test_label_set_reaches_results_viewer(
        self, composed_fem: FEMData,
    ) -> None:
        """The two compose labels ``A`` and ``B`` both appear on the
        results-side ViewerData."""
        stub = _ResultsStub(fem=composed_fem, path=None)
        viewer = ResultsViewer(stub)
        view = viewer._build_viewer_data()

        elem_labels = {
            view.elements.module_for(int(eid))
            for group in view.elements
            for eid in group.ids
        }
        node_labels = {
            view.nodes.module_for(int(nid))
            for nid in view.nodes.ids
        }
        # Both module labels should be present on at least one
        # element (the host rows might map to None).
        assert "A" in elem_labels
        assert "B" in elem_labels
        # Same on the node side.
        assert "A" in node_labels
        assert "B" in node_labels


# =====================================================================
# Layering invariant marker
# =====================================================================


def test_no_apegmsh_mesh_imports_in_results_viewer() -> None:
    """Documentation marker — the actual enforcement lives in
    ``test_viewers_pure_h5_consumer.py``.  Listed here so the slice's
    architectural intent is explicit in this file's docstring +
    test list."""
    import apeGmsh.viewers.results_viewer as rv
    # The module imports from .data / .scene / .ui — never from
    # apeGmsh.mesh directly.  This assertion is a smoke check, not
    # the binding contract.
    src = Path(rv.__file__).read_text(encoding="utf-8", errors="ignore")
    assert "from apeGmsh.mesh" not in src
    assert "import apeGmsh.mesh" not in src
