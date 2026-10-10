"""Tests for the Compose facade readers — Phase 3B.1 / ADR 0038, on the
ADR 0117 ``Assembly`` (v2) form.

Locks:

* the ``g.compose_inspect(...)`` H5-metadata helper,
* the ``g.compose_list()`` session-side accessor, read from a real
  ``Assembly.h5`` archive,
* the :class:`ComposedModule` handle (identity surface; the
  introspection methods stay stubbed),
* the typed exception hierarchy.

The v1 writer (``g.compose``) and its input gates are gone; the
``Assembly.instance`` gates are locked in ``tests/assembly/``.
"""
from __future__ import annotations

from pathlib import Path

import math
import warnings

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
from apeGmsh.assembly import Assembly
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
    """A fresh instanceable ``model.h5`` (``/opensees`` model 3/3) with no
    composition — for ``compose_inspect`` and as the assembly's source."""
    ops = apeSees(_make_simple_fem())
    ops.model(ndm=3, ndf=3)
    out = tmp_path / "uncomposed.h5"
    ops.h5(str(out))
    return out


#: ``beta``'s placement: a quarter turn about +z, then a translate.
BETA_ROTATE = ((0.0, 0.0, 1.0), math.pi / 2)


@pytest.fixture
def saved_composed_h5(saved_uncomposed_h5: Path, tmp_path: Path) -> Path:
    """An ``Assembly.h5`` archive with two ranked instances of one source:
    ``alpha`` (rank 0, translate (1, 2, 3)) and ``beta`` (rank 1,
    rotated, translate (10, 0, 0))."""
    asm = Assembly("pair")
    asm.instance("alpha", saved_uncomposed_h5,
                 translate=(1.0, 2.0, 3.0), partition_rank=0)
    asm.instance("beta", saved_uncomposed_h5, translate=(10.0, 0.0, 0.0),
                 rotate=BETA_ROTATE, partition_rank=1)
    out = tmp_path / "composed.h5"
    with warnings.catch_warnings():
        # Two ranks with no declared numberer/system auto-emit (ADR 0027).
        warnings.simplefilter("ignore")
        asm.bridge(ndm=3, ndf=3)
        asm.h5(str(out))
    return out


# ---------------------------------------------------------------------------
# Class-level surface
# ---------------------------------------------------------------------------


def test_reservation_granularity_class_attr() -> None:
    """``RESERVATION_GRANULARITY`` is the documented 1M default."""
    assert Compose.RESERVATION_GRANULARITY == 1_000_000


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
    assert alpha.partition_rank == 0
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
    session: apeGmsh, saved_composed_h5: Path, saved_uncomposed_h5: Path,
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

    source = str(saved_uncomposed_h5)
    alpha = modules[0]
    assert isinstance(alpha, ComposedModule)
    assert alpha.source_path == source
    assert alpha.translate == (1.0, 2.0, 3.0)
    assert alpha.rotate is None  # identity: no rotation recorded
    assert alpha.partition_rank == 0

    beta = modules[1]
    assert beta.source_path == source
    assert beta.translate == (10.0, 0.0, 0.0)
    (ax, ay, az), theta = BETA_ROTATE
    assert beta.rotate == pytest.approx((ax, ay, az, theta))
    assert beta.partition_rank == 1


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
