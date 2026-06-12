"""Tests for the Compose facade scaffold — Phase 3B.1 / ADR 0038.

This PR (3B.1) lands:

* the input-validation gates on ``g.compose(...)``,
* the ``g.compose_inspect(...)`` H5-metadata helper,
* the ``g.compose_list()`` session-side accessor,
* the :class:`ComposedModule` handle (identity surface only — the
  introspection methods are stubbed pending Phase 3B.2's merge engine),
* the typed exception hierarchy.

The merge engine itself raises :class:`NotImplementedError` until
Phase 3B.2 wires it.  These tests deliberately exercise the
"validation-fires-before-engine-stub" guarantee so future regressions
of the eager-gate contract are caught here, not on a partial merge.
"""
from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pytest

from apeGmsh._core import apeGmsh
from apeGmsh._kernel.records._compose import ComposeRecord
from apeGmsh._kernel.record_sets import ComposeSet
from apeGmsh.mesh._compose import (
    Compose,
    ComposeAnchorError,
    ComposeCapacityError,
    ComposeDepthExceededError,
    ComposeError,
    ComposeFilterWarning,
    ComposeLabelError,
    ComposeNamespaceCollisionError,
    ComposedModule,
)
from apeGmsh.mesh._element_types import ElementGroup, make_type_info
from apeGmsh.mesh._group_set import LabelSet, PhysicalGroupSet
from apeGmsh.mesh.FEMData import (
    ElementComposite,
    FEMData,
    MeshInfo,
    NodeComposite,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_simple_fem(
    composed_from: "ComposeSet | tuple[ComposeRecord, ...]" = (),
) -> FEMData:
    """Tiny FEMData — mirrors the schema-test builder for compose facades."""
    node_ids = np.array([1, 2, 3], dtype=np.int64)
    node_coords = np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
        dtype=np.float64,
    )
    line_info = make_type_info(
        code=1, gmsh_name="Line 2", dim=1, order=1, npe=2, count=2,
    )
    conn = np.array([[1, 2], [2, 3]], dtype=np.int64)
    elem_ids = np.array([10, 20], dtype=np.int64)
    line_group = ElementGroup(
        element_type=line_info, ids=elem_ids, connectivity=conn,
    )

    nodes = NodeComposite(
        node_ids=node_ids, node_coords=node_coords,
        physical=PhysicalGroupSet({}), labels=LabelSet({}),
    )
    elements = ElementComposite(
        groups={1: line_group},
        physical=PhysicalGroupSet({}), labels=LabelSet({}),
    )
    info = MeshInfo(
        n_nodes=3, n_elems=2, bandwidth=1, types=[line_info],
    )
    return FEMData(
        nodes=nodes, elements=elements, info=info,
        composed_from=composed_from,
    )


def _make_record(label: str, **overrides) -> ComposeRecord:
    """ComposeRecord builder for round-trip tests."""
    defaults = dict(
        label=label,
        source_path=f"{label}.h5",
        source_fem_hash=f"hash-{label}",
        source_neutral_schema_version="2.9.0",
        translate=(1.0, 2.0, 3.0),
        rotate=(0.0, 0.0, 1.0, 0.0),
        partition_rank=1,
        composed_at="2026-05-26T12:00:00Z",
        properties={"author": "test"},
    )
    defaults.update(overrides)
    return ComposeRecord(**defaults)


@pytest.fixture
def session() -> apeGmsh:
    """A bare :class:`apeGmsh` session — not begun, no gmsh state.

    Sufficient for facade unit tests because :class:`Compose` only
    touches the session through ``mesh.queries.get_fem_data()`` (which
    :meth:`Compose._current_fem` swallows defensively) and never reads
    the underlying gmsh kernel from 3B.1's surface.
    """
    return apeGmsh(model_name="compose_facade_test")


@pytest.fixture
def saved_uncomposed_h5(tmp_path: Path) -> Path:
    """A fresh ``model.h5`` with no composition — for ``compose_inspect``."""
    fem = _make_simple_fem()
    out = tmp_path / "uncomposed.h5"
    fem.to_h5(str(out))
    return out


@pytest.fixture
def saved_composed_h5(tmp_path: Path) -> Path:
    """A ``model.h5`` carrying two ``ComposeRecord`` provenance entries."""
    rec_a = _make_record("alpha", partition_rank=1)
    rec_b = _make_record(
        "beta", partition_rank=2, translate=(10.0, 0.0, 0.0),
    )
    fem = _make_simple_fem(composed_from=ComposeSet((rec_a, rec_b)))
    out = tmp_path / "composed.h5"
    fem.to_h5(str(out))
    return out


# ---------------------------------------------------------------------------
# Class-level surface
# ---------------------------------------------------------------------------


def test_reservation_granularity_class_attr() -> None:
    """``RESERVATION_GRANULARITY`` is the documented 1M default."""
    assert Compose.RESERVATION_GRANULARITY == 1_000_000


# ---------------------------------------------------------------------------
# Label validation — ADR 0038 §"g.compose() signature" line 94
# ---------------------------------------------------------------------------


def test_compose_label_empty_raises(
    session: apeGmsh, saved_uncomposed_h5: Path,
) -> None:
    """Empty ``label`` trips :class:`ComposeLabelError`."""
    with pytest.raises(ComposeLabelError):
        session.compose(saved_uncomposed_h5, label="")


def test_compose_label_dotted_raises(
    session: apeGmsh, saved_uncomposed_h5: Path,
) -> None:
    """``label='foo.bar'`` — '.' is the namespace separator."""
    with pytest.raises(ComposeLabelError):
        session.compose(saved_uncomposed_h5, label="foo.bar")


def test_compose_label_slash_raises(
    session: apeGmsh, saved_uncomposed_h5: Path,
) -> None:
    """``label='foo/bar'`` — '/' is the depth-boundary separator."""
    with pytest.raises(ComposeLabelError):
        session.compose(saved_uncomposed_h5, label="foo/bar")


def test_compose_label_whitespace_raises(
    session: apeGmsh, saved_uncomposed_h5: Path,
) -> None:
    """``label='foo bar'`` — whitespace is disallowed."""
    with pytest.raises(ComposeLabelError):
        session.compose(saved_uncomposed_h5, label="foo bar")


# ---------------------------------------------------------------------------
# Anchor / translate validation — ADR 0038 line 104
# ---------------------------------------------------------------------------


def test_compose_anchor_with_nonzero_translate_raises(
    session: apeGmsh, saved_uncomposed_h5: Path,
) -> None:
    """``anchor=`` and a non-zero ``translate=`` are mutually exclusive."""
    with pytest.raises(ComposeAnchorError):
        session.compose(
            saved_uncomposed_h5,
            label="m",
            anchor="some_pg",
            translate=(1.0, 0.0, 0.0),
        )


def test_compose_anchor_with_zero_translate_passes_validation(
    saved_uncomposed_h5: Path, tmp_path: Path,
) -> None:
    """``anchor=`` with default ``translate=(0, 0, 0)`` passes validation.

    Locks the validation-order contract: input gates fire before the
    engine.  Phase 3B.2c wires the merge engine, so a missing-PG
    anchor surfaces :class:`ComposeAnchorError` from the resolver
    rather than ``NotImplementedError``.  Either way, the
    anchor+translate validator does NOT trip.
    """
    # Build a chain-phase session whose FEM has no PG named
    # ``some_pg`` — anchor resolution must therefore raise
    # ComposeAnchorError, NOT the older NotImplementedError.
    g = apeGmsh.from_h5(saved_uncomposed_h5)
    with pytest.raises(ComposeAnchorError):
        g.compose(saved_uncomposed_h5, label="m", anchor="some_pg")


# ---------------------------------------------------------------------------
# Partition-rank validation — ADR 0038 §"Layer 2" line 420
# ---------------------------------------------------------------------------


def test_compose_partition_rank_negative_raises(
    session: apeGmsh, saved_uncomposed_h5: Path,
) -> None:
    """``partition_rank=-1`` violates the ``K >= 0`` rule."""
    with pytest.raises(ValueError):
        session.compose(
            saved_uncomposed_h5, label="m", partition_rank=-1,
        )


# ---------------------------------------------------------------------------
# Merge-engine stub — ADR 0038 / Phase 3B.2 marker
# ---------------------------------------------------------------------------


def test_compose_merge_engine_wired_returns_handle(
    saved_uncomposed_h5: Path,
) -> None:
    """Valid inputs now return a :class:`ComposedModule` handle.

    Phase 3B.2c wires the merge engine: ``g.compose(...)`` no longer
    stubs out, it returns the live handle for the composed module.
    This test locks the new contract — a regression to a stub would
    fail the assertion that the returned record's label matches.
    """
    g = apeGmsh.from_h5(saved_uncomposed_h5)
    handle = g.compose(saved_uncomposed_h5, label="m")
    assert isinstance(handle, ComposedModule)
    assert handle.label == "m"


# ---------------------------------------------------------------------------
# compose_inspect — ADR 0038 §"Companion helpers (v1)" line 121
# ---------------------------------------------------------------------------


def test_compose_inspect_returns_metadata_for_uncomposed_source(
    session: apeGmsh, saved_uncomposed_h5: Path,
) -> None:
    """``compose_inspect`` reads schema + inventory metadata only."""
    from tests.fixtures.schema import NEUTRAL_CURRENT

    info = session.compose_inspect(saved_uncomposed_h5)
    # Assert against the test-fixture single source of truth so the
    # next minor bump stays a one-file edit (this literal pin went
    # stale at both the 2.12.0 and 2.13.0 bumps and red-flagged main
    # each time; same fix as test_schema_version_is_current).
    assert info["neutral_schema_version"] == NEUTRAL_CURRENT
    assert info["tag_span_max"] > 0
    # Uncomposed source has no provenance.
    assert info["composed_from"] == ()
    # Inventory accessors are tuples in sorted order.
    assert isinstance(info["pg_inventory"], tuple)
    assert isinstance(info["label_inventory"], tuple)
    # Record-count dict carries the major kinds.
    counts = info["record_counts"]
    assert counts["nodes"] == 3
    assert counts["elements"] == 2
    # Properties is a forward-compatible placeholder.
    assert info["properties"] == {}


def test_compose_inspect_returns_composed_from_for_composed_source(
    session: apeGmsh, saved_composed_h5: Path,
) -> None:
    """``compose_inspect`` returns the source's own ``composed_from``."""
    info = session.compose_inspect(saved_composed_h5)
    composed = info["composed_from"]
    assert len(composed) == 2
    labels = sorted(r.label for r in composed)
    assert labels == ["alpha", "beta"]
    # Round-trip detail spot check.
    alpha = next(r for r in composed if r.label == "alpha")
    assert alpha.partition_rank == 1
    assert alpha.translate == (1.0, 2.0, 3.0)


def test_compose_inspect_does_not_mutate_session(
    session: apeGmsh, saved_uncomposed_h5: Path,
) -> None:
    """``compose_inspect`` is read-only — no broker mutation."""
    # No FEM exists on a non-begun session — defensive _current_fem
    # returns None; this also exercises the no-FEM path.
    before = session.compose_list()
    session.compose_inspect(saved_uncomposed_h5)
    after = session.compose_list()
    assert before == after == ()


# ---------------------------------------------------------------------------
# compose_list — round-trip from saved provenance
# ---------------------------------------------------------------------------


def test_compose_list_empty_session(session: apeGmsh) -> None:
    """Fresh session with no FEM — ``compose_list`` returns ``()``."""
    assert session.compose_list() == ()


def test_compose_list_empty_uncomposed_fem(session: apeGmsh) -> None:
    """Compose facade against a session whose current FEM has no
    composition returns the empty tuple — the ``None`` and
    ``empty-ComposeSet`` paths converge.
    """
    # Inject a fresh uncomposed FEMData via the lazy facade's
    # ``_current_fem`` hook so we don't need a begun gmsh session.
    facade = Compose(session)
    fem = _make_simple_fem()
    facade._current_fem = lambda: fem  # type: ignore[method-assign]
    assert facade.compose_list() == ()


def test_compose_list_populated_from_h5_round_trip(
    session: apeGmsh, saved_composed_h5: Path,
) -> None:
    """Loaded composed FEMData → ``compose_list`` returns wrapped handles
    in label-sorted order.

    The schema's :class:`ComposeSet` iterates ascending-label-order
    (per :class:`apeGmsh._kernel.record_sets.ComposeSet`), so the
    handles surface in compose-order-independent canonical order — the
    same shape ADR 0038 §"Lineage chain extension" relies on.
    """
    from apeGmsh.mesh.FEMData import FEMData as _FEM
    fem = _FEM.from_h5(str(saved_composed_h5))

    facade = Compose(session)
    facade._current_fem = lambda: fem  # type: ignore[method-assign]

    modules = facade.compose_list()
    assert len(modules) == 2
    assert [m.label for m in modules] == ["alpha", "beta"]

    alpha = modules[0]
    assert isinstance(alpha, ComposedModule)
    assert alpha.source_path == "alpha.h5"
    assert alpha.translate == (1.0, 2.0, 3.0)
    assert alpha.rotate == (0.0, 0.0, 1.0, 0.0)
    assert alpha.partition_rank == 1

    beta = modules[1]
    assert beta.source_path == "beta.h5"
    assert beta.translate == (10.0, 0.0, 0.0)
    assert beta.partition_rank == 2


# ---------------------------------------------------------------------------
# ComposedModule introspection stubs — Phase 3B.2 surface
# ---------------------------------------------------------------------------


def test_composed_module_introspection_stubbed() -> None:
    """``pgs`` / ``labels`` / ``record_counts`` remain stubbed in 3B.2c.

    These methods need the ``module_label`` parallel dataset and a
    bound ``_fem`` to walk it.  3B.2c ships the dataset population
    machinery but the ``ComposedModule`` introspection-API surface
    is folded into the wider 3D / 3E work; the stubs stay until then.
    """
    rec = _make_record("m")
    handle = ComposedModule(record=rec)

    for method, name in (
        (handle.pgs, "pgs"),
        (handle.labels, "labels"),
        (handle.record_counts, "record_counts"),
    ):
        with pytest.raises(NotImplementedError) as excinfo:
            method()
        msg = str(excinfo.value)
        assert "3B.2" in msg, (
            f"ComposedModule.{name}() should name Phase 3B.2 in its "
            f"NotImplementedError message; got: {msg!r}"
        )


# ---------------------------------------------------------------------------
# Exception hierarchy — fail-loud catch-all surface
# ---------------------------------------------------------------------------


def test_exception_hierarchy() -> None:
    """All typed compose errors subclass ``ComposeError`` AND
    ``ValueError`` so callers can catch either base."""
    typed = (
        ComposeLabelError,
        ComposeAnchorError,
        ComposeCapacityError,
        ComposeDepthExceededError,
        ComposeNamespaceCollisionError,
    )
    for cls in typed:
        assert issubclass(cls, ComposeError)
        assert issubclass(cls, ValueError)

    # The filter warning is a separate UserWarning surface.
    assert issubclass(ComposeFilterWarning, UserWarning)

    # ``except ComposeError`` catches every typed error.
    for cls in typed:
        try:
            raise cls("test")
        except ComposeError:
            pass
