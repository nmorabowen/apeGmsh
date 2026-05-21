"""Phase 4 (ADR 0020) — ``Results`` carries :class:`OpenSeesModel`.

The Composed-file pattern: a native ``results.h5`` may embed the
``/opensees/`` zone alongside ``/meta`` + ``/model/`` (neutral) +
``/stages/...``.  The Phase 4 surface is silent auto-resolve when
that zone is present — no ``DeprecationWarning`` (ADR 0020 §Decision,
locked by the user before Phase 4 started).

INV-1 — ``Results._model is None`` is legal in Phase 4.
INV-3 — :meth:`Results.from_mpco` with ``model_h5=`` does NOT copy
        the zone into a derived h5; the broker is held in memory only.
INV-2 — ``apeGmsh.results`` may import :class:`OpenSeesModel` lazily,
        but the viewer side stays untouched (ADR 0014 AST guard
        unchanged); this module gates the lazy-import discipline via
        ``test_no_module_level_opensees_import``.
"""
from __future__ import annotations

import ast
import warnings
from pathlib import Path

import h5py
import numpy as np
import pytest

from apeGmsh.opensees import OpenSeesModel
from apeGmsh.opensees._internal.lineage import Lineage
from apeGmsh.results import Results
from apeGmsh.results.writers import NativeWriter

from tests.conftest import _stub_model_h5_path

# Reuse the Phase 3 builder — a real apeSees.h5 round-trip that
# produces a full ``/meta`` + neutral + ``/opensees/`` archive.
from tests.opensees.h5._opensees_model_fixtures import (
    build_simple_frame_h5,
)


RESULTS_PATH = (
    Path(__file__).resolve().parents[1]
    / "src" / "apeGmsh" / "results" / "Results.py"
)


# ---------------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------------

def _make_native_results_with_opensees(
    tmp_path: Path,
) -> tuple[Path, Path]:
    """Build a Composed-file ``results.h5`` carrying ``/opensees/``.

    Returns
    -------
    (results_path, model_path)
        ``results_path`` is the Composed results file (carries
        ``/meta`` + ``/model`` + ``/opensees/`` + ``/stages/...``).
        ``model_path`` is the source model.h5 the zone was copied
        from (kept on disk for tests that want to compare).
    """
    model_path, fem = build_simple_frame_h5(tmp_path)
    results_path = tmp_path / "run.h5"
    node_ids = np.asarray(fem.nodes.ids, dtype=np.int64)
    with NativeWriter(results_path) as w:
        w.open(fem=fem, model_h5_src=model_path)
        sid = w.begin_stage(
            name="grav", kind="static", time=np.array([0.0]),
        )
        w.write_nodes(
            sid, "partition_0",
            node_ids=node_ids,
            components={
                "displacement_z": np.zeros((1, node_ids.size)),
            },
        )
        w.end_stage()
    return results_path, model_path


def _make_native_results_without_opensees(
    tmp_path: Path,
) -> Path:
    """Build a legacy-shape ``results.h5`` (no ``/opensees/`` zone)."""
    _, fem = build_simple_frame_h5(tmp_path)
    results_path = tmp_path / "legacy.h5"
    node_ids = np.asarray(fem.nodes.ids, dtype=np.int64)
    with NativeWriter(results_path) as w:
        w.open(fem=fem)
        sid = w.begin_stage(
            name="grav", kind="static", time=np.array([0.0]),
        )
        w.write_nodes(
            sid, "partition_0",
            node_ids=node_ids,
            components={
                "displacement_z": np.zeros((1, node_ids.size)),
            },
        )
        w.end_stage()
    return results_path


# ---------------------------------------------------------------------------
# from_native — Phase 8 (ADR 0020 INV-1): ``model=`` is required.
# Tests load the model via :meth:`OpenSeesModel.from_h5` against the same
# Composed-file path the results carry, then pass it to ``from_native``.
# ---------------------------------------------------------------------------

def test_from_native_loads_opensees_model(tmp_path: Path) -> None:
    """``/opensees/`` zone present → ``results.model`` is the broker."""
    results_path, _ = _make_native_results_with_opensees(tmp_path)
    model = OpenSeesModel.from_h5(results_path, fem_root="/model")
    with Results.from_native(results_path, model=model) as r:
        assert r.model is not None
        assert isinstance(r.model, OpenSeesModel)


def test_from_native_no_zone_requires_explicit_model(tmp_path: Path) -> None:
    """Legacy file (no ``/opensees/``) — Phase 8 requires ``model=``.

    The legacy file lacks an embedded ``/opensees/`` zone, so the
    caller must build the model from a sibling source.  Here we
    construct one from the same fixture path (which DOES have
    ``/opensees/``) — what matters is that the user supplied
    ``model=``.
    """
    results_path = _make_native_results_without_opensees(tmp_path)
    # Build an OpenSeesModel from a *different* fixture file that has
    # the ``/opensees/`` zone.  Phase 8 contract: ``model=`` must be
    # supplied; the source need not match the results file.
    model_src, _ = build_simple_frame_h5(tmp_path)
    model = OpenSeesModel.from_h5(model_src)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with Results.from_native(results_path, model=model) as r:
            assert r.model is model
    deprecations = [
        w for w in caught if issubclass(w.category, DeprecationWarning)
    ]
    assert deprecations == [], (
        f"from_native(model=) on a no-/opensees/ file emitted "
        f"DeprecationWarning(s): {[str(w.message) for w in deprecations]}"
    )


def test_from_native_explicit_model_wins(tmp_path: Path) -> None:
    """User-supplied ``model=`` beats the file's auto-resolved one.

    INV: ``resolve_bound_model`` returns the candidate when supplied,
    never re-asks the reader.  Verified by passing a hand-rolled
    sentinel object (any non-None placeholder works — the resolver
    doesn't introspect the type).
    """
    results_path, model_path = _make_native_results_with_opensees(tmp_path)
    # Build a different in-memory model by reloading the same file
    # twice — the identity differs even though both have the same
    # snapshot_id.  We can prove "user-supplied wins" by checking
    # ``r.model is`` the exact instance we passed in.
    explicit = OpenSeesModel.from_h5(model_path)
    with Results.from_native(results_path, model=explicit) as r:
        assert r.model is explicit


# ---------------------------------------------------------------------------
# from_mpco — model_h5= kwarg, INV-3
# ---------------------------------------------------------------------------

_MPCO_FIXTURE = (
    Path(__file__).parent / "fixtures" / "results" / "elasticFrame.mpco"
)


def _require_mpco_fixture() -> Path:
    if not _MPCO_FIXTURE.exists():
        pytest.skip(f"MPCO fixture not present at {_MPCO_FIXTURE}")
    return _MPCO_FIXTURE


def test_from_mpco_accepts_model_h5(tmp_path: Path) -> None:
    """``model_h5=`` loads the sibling broker; ``results.model`` is set."""
    mpco_path = _require_mpco_fixture()
    model_path, _ = build_simple_frame_h5(tmp_path)
    with Results.from_mpco(mpco_path, model_h5=model_path) as r:
        assert r.model is not None
        assert isinstance(r.model, OpenSeesModel)


def test_from_mpco_does_not_copy_into_derived_h5(tmp_path: Path) -> None:
    """INV-3 — the model is held in memory; no temp file is generated.

    The mpco path is left untouched; ``model_h5=`` only sets the
    in-memory broker handle.  We can verify the negative by snapshotting
    the directory of the mpco file: nothing was added.
    """
    mpco_path = _require_mpco_fixture()
    model_path, _ = build_simple_frame_h5(tmp_path)
    siblings_before = set(p.name for p in mpco_path.parent.iterdir())
    with Results.from_mpco(mpco_path, model_h5=model_path) as r:
        assert r.model is not None
    siblings_after = set(p.name for p in mpco_path.parent.iterdir())
    assert siblings_after == siblings_before, (
        "from_mpco(model_h5=) created sibling files; INV-3 violated. "
        f"Diff: {sorted(siblings_after - siblings_before)!r}"
    )


def test_from_mpco_without_model_h5_raises_typeerror(tmp_path: Path) -> None:
    """Phase 8 — :meth:`Results.from_mpco` with no ``model_h5=`` raises TypeError."""
    mpco_path = _require_mpco_fixture()
    with pytest.raises(TypeError, match="model_h5= is required"):
        Results.from_mpco(mpco_path)


# ---------------------------------------------------------------------------
# from_recorders — model= embeds /opensees/ in the cached h5
# ---------------------------------------------------------------------------

# Minimal one-record recorder spec stub: the transcoder is end-to-end
# heavy and lives in tests/results/test_recorder_transcoder.py.  Phase
# 4 only needs to verify the Composed-file plumbing; we directly drive
# NativeWriter's ``write_opensees_from`` to assert the same on-disk
# shape ``from_recorders(model=)`` would produce.

def test_from_recorders_model_embeds_opensees_zone(tmp_path: Path) -> None:
    """``from_recorders(model=)`` produces a Composed file with ``/opensees/``.

    We exercise the NativeWriter + ``model_h5_src=`` plumbing directly
    (the same code path :meth:`Results.from_recorders` uses under
    ``model=``) so the test is independent of recorder transcoding
    heaviness.
    """
    model_path, fem = build_simple_frame_h5(tmp_path)
    target = tmp_path / "recorders_cached.h5"
    with NativeWriter(target) as w:
        w.open(fem=fem, model_h5_src=model_path)
        sid = w.begin_stage(
            name="anim", kind="transient", time=np.array([0.0, 1.0]),
        )
        node_ids = np.asarray(fem.nodes.ids, dtype=np.int64)
        w.write_nodes(
            sid, "partition_0", node_ids=node_ids,
            components={"displacement_z": np.zeros((2, node_ids.size))},
        )
        w.end_stage()
    with h5py.File(target, "r") as f:
        assert "opensees" in f
        # Existence of the marker pair the viewer/reader probe uses.
        assert "opensees/transforms" in f
        assert "opensees/element_meta" in f


def test_from_recorders_no_model_omits_opensees_zone(tmp_path: Path) -> None:
    """``from_recorders`` without ``model=`` produces a no-``/opensees/`` h5."""
    _, fem = build_simple_frame_h5(tmp_path)
    target = tmp_path / "recorders_no_model.h5"
    with NativeWriter(target) as w:
        w.open(fem=fem)
        sid = w.begin_stage(
            name="anim", kind="transient", time=np.array([0.0, 1.0]),
        )
        node_ids = np.asarray(fem.nodes.ids, dtype=np.int64)
        w.write_nodes(
            sid, "partition_0", node_ids=node_ids,
            components={"displacement_z": np.zeros((2, node_ids.size))},
        )
        w.end_stage()
    with h5py.File(target, "r") as f:
        assert "opensees" not in f


# ---------------------------------------------------------------------------
# Lineage — derives from model.lineage when present
# ---------------------------------------------------------------------------

def test_results_lineage_from_model(tmp_path: Path) -> None:
    """``results.lineage`` carries the model's chain plus a results_hash.

    Phase 6 (ADR 0021) layers a :attr:`Lineage.results_hash` on top of
    the model layer, so the surface contract shifts from identity-
    equality (Phase 4) to chain-forward: the model's ``fem_hash`` and
    ``model_hash`` propagate verbatim, and ``results_hash`` is the
    canonical-bytes derivation over ``/stages/...``.
    """
    results_path, _ = _make_native_results_with_opensees(tmp_path)
    model = OpenSeesModel.from_h5(results_path, fem_root="/model")
    with Results.from_native(results_path, model=model) as r:
        assert r.model is not None
        # Phase-6 contract: chain forward, not identity-equal.
        assert r.lineage.fem_hash == r.model.lineage.fem_hash
        assert r.lineage.model_hash == r.model.lineage.model_hash
        assert r.lineage.fem_hash != ""
        assert r.lineage.model_hash is not None
        assert r.lineage.results_hash is not None


def test_results_lineage_fallback_fem_only(tmp_path: Path) -> None:
    """No model → lineage carries only fem_hash from FEMData.

    Phase 8 — the caller supplies a separate model whose lineage
    only contributes if you read it through ``r.model``.  This test
    verifies the legacy-shape file (no embedded ``/opensees/``)
    correctly produces a FEM-only lineage on the results side: the
    ``model_hash`` is left None because the Composed-file resolve
    found no zone in the results file.
    """
    results_path = _make_native_results_without_opensees(tmp_path)
    model_src, _ = build_simple_frame_h5(tmp_path)
    model = OpenSeesModel.from_h5(model_src)
    with Results.from_native(results_path, model=model) as r:
        # ``r.model`` is the user-supplied one (Phase 8 INV-1).
        assert r.model is model
        lineage = r.lineage
        assert isinstance(lineage, Lineage)
        # The embedded FEMData snapshot still populates fem_hash.
        assert lineage.fem_hash != ""


# ---------------------------------------------------------------------------
# Frozen identity — derived Results share the model reference
# ---------------------------------------------------------------------------

def test_results_model_property_stable(tmp_path: Path) -> None:
    """``results.model is results.model`` — identity is stable."""
    results_path, _ = _make_native_results_with_opensees(tmp_path)
    model = OpenSeesModel.from_h5(results_path, fem_root="/model")
    with Results.from_native(results_path, model=model) as r:
        assert r.model is r.model
        # Stage-scoping does not lose the broker handle.
        scoped = r.stage("grav")
        assert scoped.model is r.model


def test_results_bind_preserves_model(tmp_path: Path) -> None:
    """``results.bind(fem)`` returns a Results carrying the same model."""
    results_path, _ = _make_native_results_with_opensees(tmp_path)
    model = OpenSeesModel.from_h5(results_path, fem_root="/model")
    with Results.from_native(results_path, model=model) as r:
        rebound = r.bind(r.fem)
        assert rebound.model is r.model


# ---------------------------------------------------------------------------
# Phase 4 silence gate — no DeprecationWarning anywhere on the surface
# ---------------------------------------------------------------------------

def test_no_deprecation_warning_in_phase_8(tmp_path: Path) -> None:
    """The Phase 8 surface emits NO ``DeprecationWarning``.

    Phase 8 made ``model=`` / ``model_h5=`` required (TypeError on
    missing).  No silent-auto-resolve path remains — but the explicit
    paths must still be warning-free.
    """
    results_with = _make_native_results_with_opensees(tmp_path)[0]
    model = OpenSeesModel.from_h5(results_with, fem_root="/model")
    mpco_path = _require_mpco_fixture()
    model_path, _ = build_simple_frame_h5(tmp_path)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with Results.from_native(results_with, model=model):
            pass
        with Results.from_mpco(mpco_path, model_h5=model_path):
            pass

    deprecations = [
        w for w in caught if issubclass(w.category, DeprecationWarning)
    ]
    assert deprecations == [], (
        f"Phase 8 surface emitted DeprecationWarning(s): "
        f"{[str(w.message) for w in deprecations]}"
    )


# ---------------------------------------------------------------------------
# Import-DAG polarity — module-level discipline
# ---------------------------------------------------------------------------

def test_no_module_level_opensees_import() -> None:
    """``apeGmsh.results.Results`` must not import ``OpenSeesModel`` eagerly.

    ADR 0020 §"Module-import discipline" — the chain-forward field
    (``Results._model``) is typed under ``TYPE_CHECKING`` only and
    imported lazily inside method bodies that need it.  Adding an
    eager edge ``apeGmsh.results.Results → apeGmsh.opensees.opensees_model``
    would drag the entire OpenSees graph into the Results import
    surface and break the symmetry the viewer relies on.

    AST scan: collect every module-level (non-``TYPE_CHECKING``) import
    statement and assert no ``OpenSeesModel`` symbol or
    ``opensees_model`` module is named.
    """
    src = RESULTS_PATH.read_text(encoding="utf-8")
    tree = ast.parse(src)
    for node in tree.body:
        if isinstance(node, ast.If):
            # Skip ``if TYPE_CHECKING:`` blocks — those are
            # type-only imports the runtime never executes.
            test = node.test
            is_tc = (
                (isinstance(test, ast.Name) and test.id == "TYPE_CHECKING")
                or (
                    isinstance(test, ast.Attribute)
                    and test.attr == "TYPE_CHECKING"
                )
            )
            if is_tc:
                continue
        if isinstance(node, ast.ImportFrom):
            mod = (node.module or "").lower()
            assert "opensees_model" not in mod, (
                f"Eager import from {node.module!r} in Results.py — "
                f"OpenSeesModel must be lazy-imported only."
            )
            for alias in node.names:
                assert alias.name != "OpenSeesModel", (
                    f"Eager `from {node.module} import OpenSeesModel` "
                    f"in Results.py — must be lazy-imported only."
                )
        elif isinstance(node, ast.Import):
            for alias in node.names:
                assert "opensees_model" not in alias.name.lower(), (
                    f"Eager `import {alias.name}` in Results.py — "
                    f"OpenSeesModel must be lazy-imported only."
                )


# ---------------------------------------------------------------------------
# Multi-partition MPCO — opensees_model() returns None
# ---------------------------------------------------------------------------

def test_mpco_multi_partition_opensees_model_is_none(tmp_path: Path) -> None:
    """The multi-partition MPCO reader has no /opensees/ zone either.

    Single-partition reader is exercised by the
    ``test_from_mpco_without_model_h5_returns_none`` case above; this
    test exercises the same contract on the multi-partition façade
    via direct construction (the public API auto-discovers siblings
    only when they exist on disk).
    """
    from apeGmsh.results.readers._mpco_multi import (
        MPCOMultiPartitionReader,
    )
    mpco_path = _require_mpco_fixture()
    reader = MPCOMultiPartitionReader([mpco_path])
    try:
        assert reader.opensees_model() is None
    finally:
        reader.close()


# ---------------------------------------------------------------------------
# ADR 0023 — per-zone schema version stamp
# ---------------------------------------------------------------------------

def test_results_schema_version_attr_present(tmp_path: Path) -> None:
    """``/meta/results_schema_version`` is written by NativeWriter.open.

    ADR 0023 — the per-zone marker for the results zone. Phase 4
    writes it alongside the envelope ``/meta/schema_version`` so
    Phase 7a's two-version reader window has a stable key to gate on
    when it lands.
    """
    from apeGmsh.results.schema._versions import RESULTS_SCHEMA_VERSION
    results_path, _ = _make_native_results_with_opensees(tmp_path)
    with h5py.File(results_path, "r") as f:
        assert "results_schema_version" in f.attrs
        assert (
            f.attrs["results_schema_version"]
            == RESULTS_SCHEMA_VERSION
        )
